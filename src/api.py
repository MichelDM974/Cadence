"""API Cadence (FastAPI) : décomposition du rythme, indice de régularité,
ruptures et prévision — à partir d'un restaurant du dataset de test (Kaggle)
ou d'un CSV importé, avec enrichissement optionnel par localisation (jours
fériés, vacances scolaires, météo). Sert aussi le frontend statique
(src/static/).
"""

import io
import json
import logging
import math
import os
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from prophet import Prophet
from starlette.middleware.sessions import SessionMiddleware
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

import exogenous
import lightgbm_forecast
import storage
from decompose_prototype import (
    DATA_DIR,
    detect_anomalies,
    event_components,
    load_series,
    regularity_index,
    seasonal_naive_forecast,
)

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

app = FastAPI(title="Cadence API")
storage.init_db()
app.add_middleware(SessionMiddleware, secret_key=storage.session_secret())

STATIC_DIR = Path(__file__).parent / "static"
MIN_BACKTEST_TRAIN_DAYS = 90


def current_user(request: Request) -> dict:
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(401, "Non connecté")
    user = storage.get_user(user_id)
    if user is None:
        request.session.clear()
        raise HTTPException(401, "Non connecté")
    return user


@app.get("/api/me")
def api_me(request: Request):
    user_id = request.session.get("user_id")
    if user_id is None:
        return {"logged_in": False, "needs_setup": not storage.has_any_user()}
    user = storage.get_user(user_id)
    if user is None:
        request.session.clear()
        return {"logged_in": False, "needs_setup": not storage.has_any_user()}
    return {"logged_in": True, "username": user["username"]}


@app.post("/api/setup")
def api_setup(request: Request, username: str = Form(...), password: str = Form(...)):
    if storage.has_any_user():
        raise HTTPException(403, "Un compte existe déjà — utilise la page de connexion")
    username = username.strip()
    if not username or not password:
        raise HTTPException(400, "Identifiant et mot de passe requis")
    user_id = storage.create_user(username, password)
    storage.claim_orphan_points(user_id)
    request.session["user_id"] = user_id
    return {"username": username}


# Anti-bruteforce : PBKDF2 à 200k itérations ne fait que plafonner le débit,
# il n'empêche pas d'essayer en continu. Compté par IP et non par identifiant,
# sinon n'importe qui pourrait bloquer le compte d'un tiers en enchaînant les
# échecs sur son nom.
LOGIN_MAX_ECHECS = 10
LOGIN_FENETRE_SECONDES = 300
_login_echecs: dict[str, list[float]] = {}


def _login_bloque(cle: str) -> bool:
    maintenant = time.time()
    recents = [t for t in _login_echecs.get(cle, []) if maintenant - t < LOGIN_FENETRE_SECONDES]
    if recents:
        _login_echecs[cle] = recents
    else:
        _login_echecs.pop(cle, None)
    return len(recents) >= LOGIN_MAX_ECHECS


@app.post("/api/login")
def api_login(request: Request, username: str = Form(...), password: str = Form(...)):
    cle = request.client.host if request.client else "inconnu"
    if _login_bloque(cle):
        raise HTTPException(429, "Trop de tentatives, réessaie dans quelques minutes")

    user = storage.verify_user(username.strip(), password)
    if user is None:
        _login_echecs.setdefault(cle, []).append(time.time())
        raise HTTPException(401, "Identifiant ou mot de passe incorrect")

    _login_echecs.pop(cle, None)
    request.session["user_id"] = user["id"]
    return {"username": user["username"]}


@app.post("/api/logout")
def api_logout(request: Request):
    request.session.clear()
    return {"logged_out": True}

# Jobs de prévision "tous magasins", suivis en mémoire pour permettre une
# barre de progression côté client (le traitement peut prendre plusieurs
# minutes sur un gros fichier, une requête HTTP unique ne peut pas reporter
# d'avancement en cours de route).
bulk_jobs: dict[str, dict] = {}

# Un job n'était retiré qu'au téléchargement réussi de son CSV : un job en
# erreur, ou dont l'utilisateur a simplement fermé l'onglet, restait en
# mémoire pour la vie du process — avec le CSV complet (plusieurs Mo sur un
# gros fichier) et la liste des entités. D'où cette purge par âge, faite au
# démarrage d'un nouveau job plutôt que par une tâche de fond.
BULK_JOB_TTL_SECONDS = 3600


def purge_stale_bulk_jobs(now=None):
    now = now if now is not None else time.time()
    for job_id, job in list(bulk_jobs.items()):
        if now - job["created_at"] > BULK_JOB_TTL_SECONDS:
            del bulk_jobs[job_id]

# Un modèle par entité est indépendant des autres : on les ajuste en parallèle
# sur plusieurs process (l'ajustement SARIMAX/Prophet est CPU-bound, un seul
# thread Python ne peut utiliser qu'un cœur). Créé à la demande plutôt qu'à
# l'import pour ne pas forker de process au chargement du module (tests...).
_bulk_executor: ProcessPoolExecutor | None = None


_bulk_executor_lock = threading.Lock()


def get_bulk_executor() -> ProcessPoolExecutor:
    global _bulk_executor
    # Verrou : deux démarrages simultanés pouvaient voir la globale à None
    # tous les deux, le second écrasant le pool du premier — qui restait
    # alors orphelin, avec ses os.cpu_count() process jamais arrêtés.
    if _bulk_executor is None:
        with _bulk_executor_lock:
            if _bulk_executor is None:
                _bulk_executor = ProcessPoolExecutor(max_workers=os.cpu_count() or 4)
    return _bulk_executor


def analyze(
    df: pd.DataFrame,
    flags: dict | None = None,
    weather: pd.DataFrame | None = None,
    unit_label: str = "unités",
) -> dict:
    stl = STL(df["visitors"], period=7, robust=True)
    result = stl.fit()

    event, effects = event_components(result.resid, flags or {})
    adjusted_resid = result.resid - event
    idx = regularity_index(adjusted_resid, df["visitors"])
    anomalies = detect_anomalies(adjusted_resid)

    weather_info = None
    if weather is not None:
        corr_temp = adjusted_resid.corr(weather["temp_max"])
        corr_rain = adjusted_resid.corr(weather["precipitation"])
        weather_info = {
            "temp_max": weather["temp_max"].round(1).tolist(),
            "precipitation": weather["precipitation"].round(1).tolist(),
            "corr_temp": None if pd.isna(corr_temp) else round(float(corr_temp), 2),
            "corr_precipitation": None if pd.isna(corr_rain) else round(float(corr_rain), 2),
        }

    return {
        "dates": [d.date().isoformat() for d in df.index],
        "observed": df["visitors"].round(1).tolist(),
        "trend": result.trend.round(1).tolist(),
        "seasonal": result.seasonal.round(1).tolist(),
        "event": event.round(1).tolist(),
        "resid": adjusted_resid.round(1).tolist(),
        "regularity_index": round(float(idx), 3),
        "effects": {name: round(float(value), 1) for name, value in effects.items()},
        "weather": weather_info,
        "unit_label": unit_label,
        "anomalies": [
            {"date": d.date().isoformat(), "resid": round(float(v), 1)}
            for d, v in anomalies.sort_values(key=abs, ascending=False).items()
        ],
    }


def forecast_sarimax(
    df: pd.DataFrame,
    horizon: int,
    exog_hist: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
) -> dict:
    model = SARIMAX(
        df["visitors"],
        exog=exog_hist,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    fc = fitted.get_forecast(steps=horizon, exog=exog_future)
    mean = fc.predicted_mean.clip(lower=0)
    ci = fc.conf_int(alpha=0.2).clip(lower=0)

    return {
        "dates": [d.date().isoformat() for d in mean.index],
        "prevision": mean.round(1).tolist(),
        "ic_bas": ci.iloc[:, 0].round(1).tolist(),
        "ic_haut": ci.iloc[:, 1].round(1).tolist(),
    }


def forecast_prophet(
    df: pd.DataFrame,
    horizon: int,
    exog_hist: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
    interval_width: float = 0.8,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    # multiplicative plutôt qu'additive (défaut brut de la bibliothèque
    # Prophet) : plus adapté à des données de type retail/activité, où
    # l'amplitude saisonnière (pic du week-end, etc.) grandit avec le niveau
    # général de l'activité plutôt que de rester fixe en valeur absolue.
    seasonality_mode: str = "multiplicative",
) -> dict:
    columns = list(exog_hist.columns) if exog_hist is not None else []

    train_df = pd.DataFrame({"ds": df.index, "y": df["visitors"].to_numpy()})
    if columns:
        train_df = train_df.join(exog_hist[columns].reset_index(drop=True))

    model = Prophet(
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
    )
    for column in columns:
        model.add_regressor(column)
    model.fit(train_df)

    future_dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})
    if columns:
        future_df = future_df.join(exog_future[columns].reset_index(drop=True))

    result = model.predict(future_df)

    return {
        "dates": [d.date().isoformat() for d in future_dates],
        "prevision": result["yhat"].clip(lower=0).round(1).tolist(),
        "ic_bas": result["yhat_lower"].clip(lower=0).round(1).tolist(),
        "ic_haut": result["yhat_upper"].clip(lower=0).round(1).tolist(),
    }


FORECAST_MODELS = {"sarimax": forecast_sarimax, "prophet": forecast_prophet}

# Modèles utilisables pour la génération en masse uniquement (section 4) :
# lightgbm entraîne un seul modèle sur toutes les entités à la fois (cf.
# lightgbm_forecast.py) — n'a de sens que là, jamais pour une entité seule,
# donc volontairement absent de FORECAST_MODELS/forecast() ci-dessus.
BULK_MODELS = {*FORECAST_MODELS, "lightgbm"}


def forecast(
    df: pd.DataFrame,
    horizon: int,
    exog_hist: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
    model: str = "sarimax",
    model_params: dict | None = None,
) -> dict:
    if model not in FORECAST_MODELS:
        raise HTTPException(400, f"Modèle inconnu : {model}")
    return FORECAST_MODELS[model](df, horizon, exog_hist=exog_hist, exog_future=exog_future, **(model_params or {}))


def backtest(
    df: pd.DataFrame,
    horizon: int,
    exog_hist: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
    unit_label: str = "unités",
    models: tuple[str, ...] = ("sarimax", "prophet"),
    model_params: dict[str, dict] | None = None,
) -> dict:
    """Compare, sur les derniers `horizon` jours déjà connus, la prévision de
    chaque modèle à une baseline naïve ("même jour la semaine dernière") —
    sert à montrer concrètement, pour cette activité précise, quel modèle
    apporte le plus de valeur. `model_params` : réglages par modèle (clé =
    nom du modèle), venus du panneau "Mon compte" côté app.
    """
    if len(df) < MIN_BACKTEST_TRAIN_DAYS + horizon:
        raise HTTPException(
            400,
            f"Historique trop court pour un backtest fiable "
            f"(minimum {MIN_BACKTEST_TRAIN_DAYS + horizon} jours, {len(df)} disponibles)",
        )

    train, test = df.iloc[:-horizon], df.iloc[-horizon:]
    actual = test["visitors"].to_numpy()
    model_params = model_params or {}

    predictions = {"naive": seasonal_naive_forecast(train["visitors"], horizon)}
    for model_name in models:
        result = forecast(
            train, horizon, exog_hist=exog_hist, exog_future=exog_future, model=model_name,
            model_params=model_params.get(model_name),
        )
        predictions[model_name] = np.array(result["prevision"])

    mae = {name: _mae(pred, actual) for name, pred in predictions.items()}
    mae_naive = mae["naive"]
    improvement_pct = {
        name: (None if mae_naive == 0 else round((mae_naive - value) / mae_naive * 100, 1))
        for name, value in mae.items()
        if name != "naive"
    }

    return {
        "dates": [d.date().isoformat() for d in test.index],
        "actual": [round(float(v), 1) for v in actual],
        "predictions": {name: [round(float(v), 1) for v in pred] for name, pred in predictions.items()},
        "mae": {name: round(value, 2) for name, value in mae.items()},
        "improvement_pct": improvement_pct,
        "unit_label": unit_label,
    }


def _mae(prevision: list[float], actual: np.ndarray) -> float:
    return float(np.abs(np.asarray(prevision) - actual).mean())


# d et D quasi toujours égaux à 1 en pratique (une seule différenciation de
# tendance/saisonnalité suffit presque toujours pour une activité journalière)
# : on les fixe pour concentrer la recherche sur p/q/P/Q, là où se jouent les
# vraies interactions. Descente par coordonnées plutôt qu'un grid search : à
# chaque étape on ne fait varier qu'un seul paramètre (les autres restent au
# meilleur trouvé jusque-là), sur 2 passes pour rattraper une partie des
# interactions entre paramètres — sans retester tout le produit cartésien.
_SARIMAX_D = 1
_SARIMAX_BIG_D = 1
_SARIMAX_PARAM_RANGE = (0, 1, 2)


def _sarimax_coordinate_search(
    train: pd.DataFrame,
    horizon: int,
    exog_hist: pd.DataFrame | None,
    exog_future: pd.DataFrame | None,
    actual: np.ndarray,
    s: int,
    passes: int = 2,
    initial_mae: float | None = None,
) -> tuple[dict, float]:
    best = {"p": 1, "q": 1, "P": 1, "Q": 1}

    def score(params: dict) -> float | None:
        order = (params["p"], _SARIMAX_D, params["q"])
        seasonal_order = (params["P"], _SARIMAX_BIG_D, params["Q"], s)
        try:
            prevision = forecast_sarimax(
                train, horizon, exog_hist=exog_hist, exog_future=exog_future, order=order, seasonal_order=seasonal_order
            )["prevision"]
        except Exception:
            return None
        return _mae(prevision, actual)

    # p=q=P=Q=1 est le point de départ ci-dessus : si l'appelant l'a déjà évalué
    # (ex. mae_before de la config actuelle, quand elle vaut déjà ce point de départ),
    # inutile de le refitter une seconde fois.
    if initial_mae is not None:
        best_mae = initial_mae
    else:
        best_mae = score(best)
        if best_mae is None:
            best_mae = float("inf")

    for _ in range(passes):
        improved = False
        for axis in ("p", "q", "P", "Q"):
            for value in _SARIMAX_PARAM_RANGE:
                if value == best[axis]:
                    continue
                candidate = {**best, axis: value}
                mae = score(candidate)
                if mae is not None and mae < best_mae:
                    best, best_mae = candidate, mae
                    improved = True
        if not improved:
            break  # aucune amélioration sur la passe précédente : converge, pas la peine de continuer

    order = [best["p"], _SARIMAX_D, best["q"]]
    seasonal_order = [best["P"], _SARIMAX_BIG_D, best["Q"], s]
    return {"order": order, "seasonal_order": seasonal_order}, best_mae


# Grille curatée (pas un produit cartésien complet) : l'optimisation tourne en
# synchrone au clic d'un bouton, elle doit rester de l'ordre de la minute.
_PROPHET_CANDIDATES = [
    (0.05, 10, "additive"),
    (0.01, 10, "additive"),
    (0.1, 10, "additive"),
    (0.3, 10, "additive"),
    (0.05, 1, "additive"),
    (0.05, 25, "additive"),
    (0.05, 10, "multiplicative"),
    (0.1, 25, "multiplicative"),
    (0.01, 1, "additive"),
    (0.3, 25, "multiplicative"),
]


def autotune(
    df: pd.DataFrame,
    horizon: int,
    exog_hist: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
    unit_label: str = "unités",
    models: tuple[str, ...] = ("sarimax", "prophet"),
    model_params: dict[str, dict] | None = None,
) -> dict:
    """Cherche, pour chaque modèle demandé, les hyperparamètres qui minimisent
    l'erreur sur les derniers `horizon` jours déjà connus (même découpage
    train/test que `backtest`) — évite le réglage manuel des paramètres avancés.
    """
    if len(df) < MIN_BACKTEST_TRAIN_DAYS + horizon:
        raise HTTPException(
            400,
            f"Historique trop court pour optimiser (minimum {MIN_BACKTEST_TRAIN_DAYS + horizon} jours, "
            f"{len(df)} disponibles)",
        )

    train, test = df.iloc[:-horizon], df.iloc[-horizon:]
    actual = test["visitors"].to_numpy()
    model_params = model_params or {}

    result = {"unit_label": unit_label}

    if "sarimax" in models:
        current = model_params.get("sarimax") or {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 7]}
        current_order = list(current.get("order") or [1, 1, 1])
        current_seasonal = list(current.get("seasonal_order") or [1, 1, 1, 7])
        s = int(current_seasonal[3]) if len(current_seasonal) > 3 else 7
        try:
            mae_before = _mae(
                forecast(train, horizon, exog_hist=exog_hist, exog_future=exog_future, model="sarimax", model_params=current)["prevision"],
                actual,
            )
        except Exception:
            # config actuelle injouable (ex. paramètres invalides ressaisis à la main) :
            # on ne peut pas la comparer, la recherche ci-dessous devient la seule option
            mae_before = None
        is_default_config = current_order == [1, 1, 1] and current_seasonal == [1, 1, 1, 7]
        found, found_mae = _sarimax_coordinate_search(
            train, horizon, exog_hist, exog_future, actual, s,
            initial_mae=mae_before if is_default_config and mae_before is not None else None,
        )
        # la recherche part toujours de p=q=P=Q=1 : si la config déjà en place
        # (potentiellement différente) fait mieux que tout ce qu'elle a trouvé,
        # on la garde plutôt que de régresser silencieusement.
        if mae_before is not None and mae_before <= found_mae:
            best_order, best_seasonal, best_mae = current_order, current_seasonal, mae_before
        else:
            best_order, best_seasonal, best_mae = found["order"], found["seasonal_order"], found_mae
        result["sarimax"] = {
            "order": best_order,
            "seasonal_order": best_seasonal,
            "mae_before": None if mae_before is None else round(mae_before, 2),
            # best_mae vaut l'infini quand aucun ajustement n'a abouti (série
            # constante, que du zéro...) : sérialisé tel quel, ça donne un
            # "Infinity" que JSON.parse refuse côté client. None dit la même
            # chose — score inconnu — et se traite déjà comme mae_before.
            "mae_after": round(best_mae, 2) if math.isfinite(best_mae) else None,
        }

    if "prophet" in models:
        current = model_params.get("prophet") or {}
        try:
            mae_before = _mae(
                forecast(train, horizon, exog_hist=exog_hist, exog_future=exog_future, model="prophet", model_params=current)["prevision"],
                actual,
            )
        except Exception:
            mae_before = None
        # évite de refitter le premier candidat de la grille s'il est identique à la
        # config actuelle déjà évaluée ci-dessus (cas courant : aucune config sauvegardée).
        current_cps = current.get("changepoint_prior_scale", 0.05)
        current_sps = current.get("seasonality_prior_scale", 10.0)
        current_mode = current.get("seasonality_mode", "multiplicative")
        best = (
            {"changepoint_prior_scale": current_cps, "seasonality_prior_scale": current_sps, "seasonality_mode": current_mode, "mae": mae_before}
            if mae_before is not None
            else None
        )
        for cps, sps, mode in _PROPHET_CANDIDATES:
            if best is not None and (cps, sps, mode) == (current_cps, current_sps, current_mode):
                continue
            try:
                prevision = forecast_prophet(
                    train, horizon, exog_hist=exog_hist, exog_future=exog_future,
                    changepoint_prior_scale=cps, seasonality_prior_scale=sps, seasonality_mode=mode,
                )["prevision"]
            except Exception:
                continue
            mae = _mae(prevision, actual)
            if best is None or mae < best["mae"]:
                best = {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps, "seasonality_mode": mode, "mae": mae}
        # même garde-fou que pour SARIMAX : ne jamais remplacer une config déjà
        # meilleure que tous les candidats testés (si elle est injouable, mae_before
        # vaut None et on garde le meilleur candidat trouvé, faute de mieux).
        if best is None or (mae_before is not None and best["mae"] > mae_before):
            best = {
                "changepoint_prior_scale": current.get("changepoint_prior_scale", 0.05),
                "seasonality_prior_scale": current.get("seasonality_prior_scale", 10.0),
                "seasonality_mode": current.get("seasonality_mode", "multiplicative"),
                "mae": mae_before,
            }
        result["prophet"] = {
            "changepoint_prior_scale": best["changepoint_prior_scale"],
            "seasonality_prior_scale": best["seasonality_prior_scale"],
            "seasonality_mode": best["seasonality_mode"],
            "mae_before": None if mae_before is None else round(mae_before, 2),
            "mae_after": None if best["mae"] is None else round(best["mae"], 2),
        }

    return result


MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 Mo : très au-dessus d'un export de caisse réel


async def read_upload_limited(file: UploadFile) -> bytes:
    """Lit un fichier uploadé en refusant tout ce qui dépasse
    MAX_UPLOAD_BYTES. Sans cette borne, le contenu entier est chargé en
    mémoire avant le moindre contrôle : un seul envoi de plusieurs Go
    suffisait à faire tomber le process."""
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Fichier trop volumineux (limite {MAX_UPLOAD_BYTES // (1024 * 1024)} Mo)")
    return content


def _read_csv_flexible(content: bytes) -> pd.DataFrame:
    """Essaie d'abord la virgule (moteur C, rapide même sur un gros fichier) ;
    ne bascule vers la détection automatique du séparateur (moteur Python,
    beaucoup plus lente) que si la virgule n'a manifestement pas fonctionné
    (une seule colonne, ex. CSV en point-virgule ou tabulation).
    """
    try:
        df = pd.read_csv(io.BytesIO(content))
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    try:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    except Exception:
        # Seconde tentative laissée nue jusqu'ici : un .xlsx renommé .csv ou
        # un fichier à nombre de colonnes variable remontait une ParserError
        # brute (500 + trace), là où toutes les autres erreurs d'import
        # renvoient un message lisible.
        raise HTTPException(400, "Fichier illisible : vérifie qu'il s'agit bien d'un CSV")


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse une colonne de dates. Essaie d'abord le format ISO strict
    (AAAA-MM-JJ, utilisé par tous les jeux de référence et les exports
    machine) ; si au moins une valeur ne correspond pas, retombe sur un
    parsing tolérant en jour-en-premier (JJ/MM/AAAA) — convention des
    restaurateurs visés (France). Sans ce fallback explicite, pandas devine
    MM/JJ par défaut et lit silencieusement un jour <= 12 comme le mauvais
    mois (03/04/2024 -> 4 mars au lieu du 3 avril), sans la moindre erreur.
    """
    iso = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    if iso.notna().all():
        return iso
    return pd.to_datetime(raw, errors="coerce", dayfirst=True)


def _daily_series(
    rows: pd.DataFrame, date_col: str, value_col: str, extra_cols: list[str] | None = None
) -> pd.DataFrame:
    """Agrège les lignes par jour (somme). `extra_cols` (ex. onpromotion)
    sont agrégées de la même façon et ajoutées comme colonnes supplémentaires
    — pour servir de variables exogènes tirées du fichier importé lui-même.
    """
    extra_cols = extra_cols or []
    s = rows[[date_col, value_col, *extra_cols]].dropna(subset=[date_col, value_col]).copy()
    s = s.rename(columns={date_col: "visit_date", value_col: "visitors"})
    s["visit_date"] = _parse_dates(s["visit_date"])
    if s["visit_date"].isna().all():
        raise HTTPException(400, f"Colonne '{date_col}' : aucune date reconnue (formats acceptés : AAAA-MM-JJ, JJ/MM/AAAA)")
    s = s.dropna(subset=["visit_date"])
    daily = s.groupby("visit_date")[["visitors", *extra_cols]].sum()

    if len(daily) < 21:
        raise HTTPException(400, "Historique trop court (minimum 3 semaines)")

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range).fillna(0)
    daily.index.name = "visit_date"
    return daily


def sorted_store_ids(series: pd.Series) -> list[str]:
    """Trie les identifiants magasin numériquement s'ils le sont tous (1, 2,
    ..., 1115 plutôt que "1", "10", "100"...), sinon alphabétiquement.
    """
    values = sorted(series.dropna().astype(str).unique())
    try:
        return sorted(values, key=float)
    except ValueError:
        return values


def parse_csv(
    content: bytes,
    date_col: str,
    value_col: str,
    store_col: str | None = None,
    store_value: str | None = None,
    category_col: str | None = None,
    category_value: str | None = None,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    raw = _read_csv_flexible(content)

    if date_col not in raw.columns or value_col not in raw.columns:
        raise HTTPException(400, "Colonnes introuvables dans le CSV")

    for col, value in ((store_col, store_value), (category_col, category_value)):
        if col:
            if col not in raw.columns:
                raise HTTPException(400, "Colonne identifiant introuvable dans le CSV")
            raw = raw[raw[col].astype(str) == str(value)]

    for col in extra_cols or []:
        if col not in raw.columns:
            raise HTTPException(400, "Colonne exogène introuvable dans le CSV")

    return _daily_series(raw, date_col, value_col, extra_cols)


def parse_form_list(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def parse_model_params(raw: str | None) -> dict:
    """Décode les réglages de modèle envoyés par le panneau "Mon compte"
    (JSON, ex. `{"sarimax": {"order": [1,1,1], ...}}`). Absent/vide -> {}
    (comportement par défaut inchangé)."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "model_params : JSON invalide")
    if not isinstance(parsed, dict):
        raise HTTPException(400, "model_params : doit être un objet JSON")
    return parsed


def parse_external_exog(
    content: bytes,
    date_col: str,
    value_cols: list[str],
    store_col: str | None = None,
    category_col: str | None = None,
) -> pd.DataFrame:
    """Parse un fichier CSV externe. Sans `store_col`/`category_col` (ex.
    prix du pétrole) : une série quotidienne unique, indépendante des
    entités — alignement par date seule. Avec l'un ou l'autre (ex.
    onpromotion, propre à chaque magasin/catégorie) : garde une ligne par
    (entité, date) — voir `external_exog_for_entity` / `_group_external_exog`
    pour en isoler une entité, ou toutes regroupées d'un coup.
    """
    group_cols = [c for c in (store_col, category_col) if c]
    raw = _read_csv_flexible(content)
    for col in (date_col, *group_cols, *value_cols):
        if col not in raw.columns:
            raise HTTPException(400, "Colonne introuvable dans le fichier exogène")

    s = raw[[date_col, *group_cols, *value_cols]].copy()
    s[date_col] = _parse_dates(s[date_col])
    s = s.dropna(subset=[date_col])
    if s.empty:
        raise HTTPException(400, "Fichier exogène vide ou dates illisibles")

    if not group_cols:
        s = s.groupby(date_col)[value_cols].mean()
        full_range = pd.date_range(s.index.min(), s.index.max(), freq="D")
        return s.reindex(full_range).ffill().bfill().rename_axis("date")

    for col in group_cols:
        s[col] = s[col].astype(str)
    return s.reset_index(drop=True)


def external_exog_for_entity(
    external: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    store_col: str | None,
    store_value: str | None,
    category_col: str | None,
    category_value: str | None,
) -> pd.DataFrame:
    """Isole, dans un fichier exogène externe multi-entités (une ligne par
    entité x date, cf. parse_external_exog), la série d'une seule entité —
    jointure par magasin et/ou catégorie en plus de la date.
    """
    subset = external
    for col, value in ((store_col, store_value), (category_col, category_value)):
        if col:
            subset = subset[subset[col] == str(value)]
    if subset.empty:
        raise HTTPException(400, "Aucune valeur exogène trouvée pour cette entité dans le fichier externe")

    s = subset.groupby(date_col)[value_cols].mean()
    full_range = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full_range).ffill().bfill().rename_axis("date")


def _group_external_exog(
    external: pd.DataFrame, date_col: str, value_cols: list[str], group_cols: list[str]
) -> dict[tuple[str, ...], pd.DataFrame]:
    """Regroupe un fichier exogène externe multi-entités par combinaison une
    seule fois (même principe que group_indices dans _run_bulk_forecast)
    plutôt que de le refiltrer à chaque entité — sur un gros fichier externe
    (potentiellement des millions de lignes), ce serait le vrai goulot
    d'étranglement, avant même l'ajustement des modèles.
    """
    if len(group_cols) == 1:
        groups = {(key,): idx for key, idx in external.groupby(group_cols[0]).groups.items()}
    else:
        groups = dict(external.groupby(group_cols).groups.items())

    result = {}
    for combo, idx in groups.items():
        s = external.loc[idx].groupby(date_col)[value_cols].mean()
        full_range = pd.date_range(s.index.min(), s.index.max(), freq="D")
        result[combo] = s.reindex(full_range).ffill().bfill().rename_axis("date")
    return result


def align_external_exog(external: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Aligne la série exogène externe sur les dates demandées, en reportant
    la dernière valeur connue au-delà de la couverture du fichier (ex.
    prévision sur un horizon plus loin que les cours de pétrole connus).
    """
    full_index = external.index.union(dates).sort_values()
    return external.reindex(full_index).ffill().bfill().loc[dates]


async def load_external_exog(
    exog_file: UploadFile | None,
    exog_date_col: str | None,
    exog_value_cols: str | None,
    exog_store_col: str | None = None,
    exog_category_col: str | None = None,
) -> pd.DataFrame | None:
    if exog_file is None:
        return None
    value_cols = parse_form_list(exog_value_cols)
    if not exog_date_col or not value_cols:
        raise HTTPException(400, "Colonnes du fichier exogène manquantes")
    return parse_external_exog(await read_upload_limited(exog_file), exog_date_col, value_cols, exog_store_col, exog_category_col)


def group_combinations(raw: pd.DataFrame, group_cols: list[str]) -> list[tuple[str, ...]]:
    """Combinaisons distinctes des colonnes de regroupement (ex. magasin ×
    catégorie), triées dimension par dimension (numériquement si possible).
    """
    if not group_cols:
        return []
    sub = raw[group_cols].astype(str).drop_duplicates()
    for col in reversed(group_cols):  # tri stable en partant de la dimension la moins prioritaire
        order = {value: i for i, value in enumerate(sorted_store_ids(sub[col]))}
        sub = sub.sort_values(by=col, key=lambda s: s.map(order), kind="stable")
    return list(sub.itertuples(index=False, name=None))


def geocode_or_400(location: str) -> dict:
    try:
        return exogenous.geocode(location)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Service de géocodage indisponible : {exc}") from exc


def exog_columns(use_ferie: bool, use_vacances: bool, use_meteo: bool) -> list[str]:
    columns = (["ferie"] if use_ferie else []) + (["vacances"] if use_vacances else [])
    columns += ["temp_max", "precipitation"] if use_meteo else []
    return columns


def flags_and_weather(
    df: pd.DataFrame, location: str | None, use_ferie: bool, use_vacances: bool, use_meteo: bool
):
    if not location:
        return None, None
    loc = geocode_or_400(location)
    exog = exogenous.location_exog(loc, df.index)
    flags = {}
    if use_ferie:
        flags["ferie"] = exog["ferie"]
    if use_vacances:
        flags["vacances"] = exog["vacances"]
    weather = exog[["temp_max", "precipitation"]] if use_meteo else None
    return flags, weather


def forecast_exog(
    df: pd.DataFrame,
    horizon: int,
    location: str | None,
    use_ferie: bool,
    use_vacances: bool,
    use_meteo: bool,
    extra_cols: list[str] | None = None,
    external_exog: pd.DataFrame | None = None,
):
    extra_cols = extra_cols or []
    columns = exog_columns(use_ferie, use_vacances, use_meteo)
    future_dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    hist_parts, future_parts = [], []
    if location and columns:
        loc = geocode_or_400(location)
        hist_parts.append(exogenous.location_exog(loc, df.index)[columns])
        future_parts.append(exogenous.location_exog(loc, future_dates)[columns])
    if extra_cols:
        # Pas de valeurs futures connues pour ces colonnes (ex. onpromotion
        # au-delà de l'historique importé) : on reporte la dernière valeur
        # observée sur tout l'horizon plutôt que d'inventer une prévision.
        hist_parts.append(df[extra_cols])
        last_values = df[extra_cols].iloc[-1]
        future_parts.append(pd.DataFrame([last_values] * horizon, index=future_dates))
    if external_exog is not None:
        hist_parts.append(align_external_exog(external_exog, df.index))
        future_parts.append(align_external_exog(external_exog, future_dates))

    if not hist_parts:
        return None, None
    return pd.concat(hist_parts, axis=1), pd.concat(future_parts, axis=1)


def backtest_exog(
    df: pd.DataFrame,
    horizon: int,
    location: str | None,
    use_ferie: bool,
    use_vacances: bool,
    use_meteo: bool,
    extra_cols: list[str] | None = None,
    external_exog: pd.DataFrame | None = None,
):
    extra_cols = extra_cols or []
    columns = exog_columns(use_ferie, use_vacances, use_meteo)

    parts = []
    if location and columns:
        loc = geocode_or_400(location)
        parts.append(exogenous.location_exog(loc, df.index)[columns])
    if extra_cols:
        parts.append(df[extra_cols])
    if external_exog is not None:
        parts.append(align_external_exog(external_exog, df.index))

    if not parts:
        return None, None
    exog = pd.concat(parts, axis=1)
    return exog.iloc[:-horizon], exog.iloc[-horizon:]


@app.get("/api/stores", dependencies=[Depends(current_user)])
def api_stores(n: int = 20):
    visits = pd.read_csv(f"{DATA_DIR}/air_visit_data.csv")
    counts = visits.groupby("air_store_id").size().sort_values(ascending=False)
    return [{"id": store_id, "n_days": int(count)} for store_id, count in counts.head(n).items()]


@app.get("/api/stores/{store_id}/analysis", dependencies=[Depends(current_user)])
def api_store_analysis(store_id: str):
    df = load_series(store_id)
    return analyze(df, flags={"ferie": df["holiday_flg"]}, unit_label="couverts")


@app.get("/api/stores/{store_id}/forecast", dependencies=[Depends(current_user)])
def api_store_forecast(
    store_id: str,
    horizon: int = Query(28, ge=7, le=90),
    model: str = Query("sarimax"),
    model_params: str | None = Query(None),
):
    return forecast(load_series(store_id), horizon, model=model, model_params=parse_model_params(model_params).get(model))


@app.get("/api/stores/{store_id}/backtest", dependencies=[Depends(current_user)])
def api_store_backtest(store_id: str, horizon: int = Query(28, ge=7, le=90), model_params: str | None = Query(None)):
    df = load_series(store_id)
    exog = df[["holiday_flg"]]
    return backtest(
        df, horizon, exog_hist=exog.iloc[:-horizon], exog_future=exog.iloc[-horizon:], unit_label="couverts",
        model_params=parse_model_params(model_params),
    )


@app.get("/api/stores/{store_id}/autotune", dependencies=[Depends(current_user)])
def api_store_autotune(store_id: str, horizon: int = Query(28, ge=7, le=90), model_params: str | None = Query(None)):
    df = load_series(store_id)
    exog = df[["holiday_flg"]]
    return autotune(
        df, horizon, exog_hist=exog.iloc[:-horizon], exog_future=exog.iloc[-horizon:], unit_label="couverts",
        model_params=parse_model_params(model_params),
    )


@app.post("/api/upload/store-ids", dependencies=[Depends(current_user)])
async def api_upload_store_ids(file: UploadFile = File(...), store_col: str = Form(...)):
    raw = _read_csv_flexible(await read_upload_limited(file))
    if store_col not in raw.columns:
        raise HTTPException(400, "Colonne identifiant introuvable dans le CSV")
    return {"store_ids": sorted_store_ids(raw[store_col])}


@app.post("/api/upload/columns/detect", dependencies=[Depends(current_user)])
async def api_upload_columns_detect(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
):
    """Repère, parmi les colonnes du CSV, celles qui ressemblent à des
    dimensions catégorielles (magasin, catégorie de produit...) plutôt qu'à
    un identifiant quasi unique ou une mesure continue — pour suggérer à
    l'utilisateur que le fichier contient plusieurs entités/catégories.
    """
    raw = _read_csv_flexible(await read_upload_limited(file))
    n_rows = len(raw)
    candidates = []
    if n_rows:
        max_unique = max(500, n_rows // 20)
        for col in raw.columns:
            if col in (date_col, value_col):
                continue
            n_unique = int(raw[col].nunique(dropna=True))
            if 1 < n_unique <= max_unique and n_unique < n_rows:
                candidates.append({"column": col, "n_unique": n_unique})
    candidates.sort(key=lambda c: c["n_unique"])
    return {"candidates": candidates}


@app.post("/api/upload/exog/validate", dependencies=[Depends(current_user)])
async def api_upload_exog_validate(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    exog_file: UploadFile = File(...),
    exog_date_col: str = Form(...),
    exog_value_cols: str = Form(...),
    exog_store_col: str | None = Form(None),
    exog_category_col: str | None = Form(None),
):
    """Vérifie qu'un fichier exogène externe est exploitable (colonnes
    présentes, dates lisibles — les mêmes erreurs que parse_external_exog,
    remontées ici avant l'ajout plutôt qu'au moment de la prévision) et
    indique jusqu'à combien de jours après la fin de l'historique du fichier
    principal il fournit de vraies valeurs futures — au-delà, la dernière
    valeur connue est reportée (cf. align_external_exog).
    """
    main_raw = _read_csv_flexible(await read_upload_limited(file))
    if date_col not in main_raw.columns:
        raise HTTPException(400, "Colonne date introuvable dans le fichier principal")
    main_dates = _parse_dates(main_raw[date_col]).dropna()
    if main_dates.empty:
        raise HTTPException(400, "Dates illisibles dans le fichier principal")
    main_first_date, main_last_date = main_dates.min(), main_dates.max()

    value_cols = parse_form_list(exog_value_cols)
    external = parse_external_exog(
        await read_upload_limited(exog_file), exog_date_col, value_cols, exog_store_col, exog_category_col
    )

    group_cols = [c for c in (exog_store_col, exog_category_col) if c]
    if group_cols:
        dates = pd.to_datetime(external[exog_date_col])
        n_entities = len(external[group_cols].drop_duplicates())
    else:
        dates = external.index.to_series()
        n_entities = None
    date_min, date_max = dates.min(), dates.max()

    # Au-delà de date_max, align_external_exog reporte la dernière valeur
    # connue (pas une vraie prévision) ; avant date_min, c'est le même
    # mécanisme en sens inverse — utile à savoir si le fichier exogène ne
    # couvre pas tout l'historique d'entraînement.
    days_covered = max(0, (date_max - main_last_date).days)
    history_gap_days = max(0, (date_min - main_first_date).days)

    return {
        "ok": True,
        "date_min": date_min.strftime("%Y-%m-%d"),
        "date_max": date_max.strftime("%Y-%m-%d"),
        "n_entities": n_entities,
        "main_first_date": main_first_date.strftime("%Y-%m-%d"),
        "main_last_date": main_last_date.strftime("%Y-%m-%d"),
        "days_covered": days_covered,
        "history_gap_days": history_gap_days,
    }


@app.post("/api/upload/analysis", dependencies=[Depends(current_user)])
async def api_upload_analysis(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    store_col: str | None = Form(None),
    store_value: str | None = Form(None),
    category_col: str | None = Form(None),
    category_value: str | None = Form(None),
    location: str | None = Form(None),
    use_ferie: bool = Form(True),
    use_vacances: bool = Form(True),
    use_meteo: bool = Form(True),
):
    df = parse_csv(await read_upload_limited(file), date_col, value_col, store_col, store_value, category_col, category_value)
    flags, weather = flags_and_weather(df, location, use_ferie, use_vacances, use_meteo)
    return analyze(df, flags=flags, weather=weather)


@app.post("/api/upload/forecast", dependencies=[Depends(current_user)])
async def api_upload_forecast(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    horizon: int = Form(28),
    model: str = Form("sarimax"),
    model_params: str | None = Form(None),
    store_col: str | None = Form(None),
    store_value: str | None = Form(None),
    category_col: str | None = Form(None),
    category_value: str | None = Form(None),
    extra_exog_cols: str | None = Form(None),
    exog_file: UploadFile | None = File(None),
    exog_date_col: str | None = Form(None),
    exog_value_cols: str | None = Form(None),
    exog_store_col: str | None = Form(None),
    exog_category_col: str | None = Form(None),
    location: str | None = Form(None),
    use_ferie: bool = Form(True),
    use_vacances: bool = Form(True),
    use_meteo: bool = Form(True),
):
    extra_cols = parse_form_list(extra_exog_cols)
    df = parse_csv(
        await read_upload_limited(file), date_col, value_col, store_col, store_value, category_col, category_value, extra_cols
    )
    external_exog = await load_external_exog(exog_file, exog_date_col, exog_value_cols, exog_store_col, exog_category_col)
    if external_exog is not None and (exog_store_col or exog_category_col):
        external_exog = external_exog_for_entity(
            external_exog, exog_date_col, parse_form_list(exog_value_cols),
            exog_store_col, store_value, exog_category_col, category_value,
        )
    exog_hist, exog_future = forecast_exog(
        df, horizon, location, use_ferie, use_vacances, use_meteo, extra_cols, external_exog
    )
    return forecast(
        df, horizon, exog_hist=exog_hist, exog_future=exog_future, model=model,
        model_params=parse_model_params(model_params).get(model),
    )


@app.post("/api/upload/backtest", dependencies=[Depends(current_user)])
async def api_upload_backtest(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    horizon: int = Form(28),
    model_params: str | None = Form(None),
    store_col: str | None = Form(None),
    store_value: str | None = Form(None),
    category_col: str | None = Form(None),
    category_value: str | None = Form(None),
    extra_exog_cols: str | None = Form(None),
    exog_file: UploadFile | None = File(None),
    exog_date_col: str | None = Form(None),
    exog_value_cols: str | None = Form(None),
    exog_store_col: str | None = Form(None),
    exog_category_col: str | None = Form(None),
    location: str | None = Form(None),
    use_ferie: bool = Form(True),
    use_vacances: bool = Form(True),
    use_meteo: bool = Form(True),
):
    extra_cols = parse_form_list(extra_exog_cols)
    df = parse_csv(
        await read_upload_limited(file), date_col, value_col, store_col, store_value, category_col, category_value, extra_cols
    )
    external_exog = await load_external_exog(exog_file, exog_date_col, exog_value_cols, exog_store_col, exog_category_col)
    if external_exog is not None and (exog_store_col or exog_category_col):
        external_exog = external_exog_for_entity(
            external_exog, exog_date_col, parse_form_list(exog_value_cols),
            exog_store_col, store_value, exog_category_col, category_value,
        )
    exog_hist, exog_future = backtest_exog(
        df, horizon, location, use_ferie, use_vacances, use_meteo, extra_cols, external_exog
    )
    return backtest(
        df, horizon, exog_hist=exog_hist, exog_future=exog_future, model_params=parse_model_params(model_params)
    )


@app.post("/api/upload/autotune", dependencies=[Depends(current_user)])
async def api_upload_autotune(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    horizon: int = Form(28),
    model_params: str | None = Form(None),
    store_col: str | None = Form(None),
    store_value: str | None = Form(None),
    category_col: str | None = Form(None),
    category_value: str | None = Form(None),
    extra_exog_cols: str | None = Form(None),
    exog_file: UploadFile | None = File(None),
    exog_date_col: str | None = Form(None),
    exog_value_cols: str | None = Form(None),
    exog_store_col: str | None = Form(None),
    exog_category_col: str | None = Form(None),
    location: str | None = Form(None),
    use_ferie: bool = Form(True),
    use_vacances: bool = Form(True),
    use_meteo: bool = Form(True),
):
    extra_cols = parse_form_list(extra_exog_cols)
    df = parse_csv(
        await read_upload_limited(file), date_col, value_col, store_col, store_value, category_col, category_value, extra_cols
    )
    external_exog = await load_external_exog(exog_file, exog_date_col, exog_value_cols, exog_store_col, exog_category_col)
    if external_exog is not None and (exog_store_col or exog_category_col):
        external_exog = external_exog_for_entity(
            external_exog, exog_date_col, parse_form_list(exog_value_cols),
            exog_store_col, store_value, exog_category_col, category_value,
        )
    exog_hist, exog_future = backtest_exog(
        df, horizon, location, use_ferie, use_vacances, use_meteo, extra_cols, external_exog
    )
    return autotune(
        df, horizon, exog_hist=exog_hist, exog_future=exog_future, model_params=parse_model_params(model_params)
    )


def _run_bulk_forecast(
    job_id: str,
    raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_cols: list[str],
    horizon: int,
    model: str,
    include_ci: bool,
    location: str | None,
    use_ferie: bool,
    use_vacances: bool,
    use_meteo: bool,
    extra_cols: list[str],
    external_exog: pd.DataFrame | None,
    store_col: str | None,
    category_col: str | None,
    exog_date_col: str | None,
    exog_value_cols: list[str],
    exog_store_col: str | None,
    exog_category_col: str | None,
    model_params: dict[str, dict],
) -> None:
    job = bulk_jobs[job_id]
    try:
        if model not in BULK_MODELS:
            raise HTTPException(400, f"Modèle inconnu : {model}")

        loc = geocode_or_400(location) if location else None
        columns = exog_columns(use_ferie, use_vacances, use_meteo)
        exog_cache: dict[tuple, pd.DataFrame] = {}

        def location_exog_cached(dates: pd.DatetimeIndex) -> pd.DataFrame:
            key = (dates.min(), dates.max(), len(dates))
            if key not in exog_cache:
                exog_cache[key] = exogenous.location_exog(loc, dates)
            return exog_cache[key]

        # Si le fichier exogène externe a une valeur par entité (magasin
        # et/ou catégorie, ex. onpromotion) plutôt qu'une seule série
        # globale (ex. prix du pétrole) : le regrouper une seule fois, même
        # principe que group_indices ci-dessous pour le fichier principal.
        exog_group_cols = [c for c in (exog_store_col, exog_category_col) if c]
        external_by_entity = (
            _group_external_exog(external_exog, exog_date_col, exog_value_cols, exog_group_cols)
            if external_exog is not None and exog_group_cols
            else None
        )

        # Dimension fixée (pas parcourue, une seule valeur) : constante après
        # le filtrage de resolve_bulk_groups, lisible directement sur `raw`.
        fixed_store_value = raw[store_col].iloc[0] if store_col and store_col not in group_cols and len(raw) else None
        fixed_category_value = (
            raw[category_col].iloc[0] if category_col and category_col not in group_cols and len(raw) else None
        )

        combos = job["combos"]
        rows = []
        skipped = 0

        # Regroupe les lignes par combinaison en un seul passage (un
        # groupby) plutôt que de refiltrer tout le fichier à chaque entité :
        # sur un gros CSV (des millions de lignes), refaire un astype(str) +
        # comparaison par entité était le vrai goulot d'étranglement, bien
        # avant l'ajustement des modèles.
        str_cols = {col: raw[col].astype(str) for col in group_cols}
        if len(group_cols) == 1:
            group_indices = {(key,): idx for key, idx in raw.groupby(str_cols[group_cols[0]]).groups.items()}
        else:
            group_indices = dict(raw.groupby([str_cols[c] for c in group_cols]).groups.items())

        if model == "lightgbm":
            # Modèle global : un seul entraînement sur toutes les entités à
            # la fois plutôt qu'un fit indépendant par entité — pas de pool
            # de process (LightGBM parallélise nativement), pas d'intervalle
            # de confiance en v1 (voir lightgbm_forecast.py).
            include_ci = False

            def on_progress(h: int, total_horizon: int) -> None:
                job["processed"] = round(h / total_horizon * len(combos))

            lgb_settings = dict(model_params.get("lightgbm") or {})
            lgb_num_boost_round = lgb_settings.pop("num_boost_round", None)

            rows, skipped_combos = lightgbm_forecast.run_bulk_lightgbm_forecast(
                raw, date_col, value_col, group_cols, horizon, combos, group_indices, extra_cols,
                location_exog_cached if loc else None, columns if loc else [],
                external_by_entity, external_exog, fixed_store_value, fixed_category_value,
                store_col, category_col, exog_store_col, exog_category_col,
                _daily_series, align_external_exog, on_progress,
                lgb_settings or None, lgb_num_boost_round,
            )
            skipped = len(skipped_combos)
            job["processed"] = len(combos)
        else:
            # Chaque entité est indépendante : on ne fait dans cette boucle
            # que du découpage de données (bon marché) et on envoie
            # l'ajustement du modèle — la partie coûteuse — aux process du
            # pool, en parallèle.
            executor = get_bulk_executor()
            futures = {}
            for combo in combos:
                idx = group_indices.get(combo)
                if idx is None:
                    skipped += 1
                    job["processed"] += 1
                    continue
                subset = raw.loc[idx]

                try:
                    df = _daily_series(subset, date_col, value_col, extra_cols)
                except HTTPException:
                    skipped += 1
                    job["processed"] += 1
                    continue

                future_dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
                hist_parts, future_parts = [], []
                if loc and columns:
                    hist_parts.append(location_exog_cached(df.index)[columns])
                    future_parts.append(location_exog_cached(future_dates)[columns])
                if extra_cols:
                    hist_parts.append(df[extra_cols])
                    last_values = df[extra_cols].iloc[-1]
                    future_parts.append(pd.DataFrame([last_values] * horizon, index=future_dates))

                entity_exog = None
                if external_by_entity is not None:
                    store_val = combo[group_cols.index(store_col)] if store_col in group_cols else fixed_store_value
                    category_val = (
                        combo[group_cols.index(category_col)] if category_col in group_cols else fixed_category_value
                    )
                    key = tuple(v for c, v in ((exog_store_col, store_val), (exog_category_col, category_val)) if c)
                    entity_exog = external_by_entity.get(key)
                elif external_exog is not None:
                    entity_exog = external_exog

                if entity_exog is not None:
                    hist_parts.append(align_external_exog(entity_exog, df.index))
                    future_parts.append(align_external_exog(entity_exog, future_dates))

                exog_hist = pd.concat(hist_parts, axis=1) if hist_parts else None
                exog_future = pd.concat(future_parts, axis=1) if future_parts else None

                future = executor.submit(
                    forecast, df, horizon, exog_hist=exog_hist, exog_future=exog_future, model=model,
                    model_params=model_params.get(model),
                )
                futures[future] = combo

            for future in as_completed(futures):
                combo = futures[future]
                try:
                    result = future.result()
                except Exception:
                    skipped += 1
                    job["processed"] += 1
                    continue

                for date, prevision, ic_bas, ic_haut in zip(
                    result["dates"], result["prevision"], result["ic_bas"], result["ic_haut"]
                ):
                    row = dict(zip(group_cols, combo))
                    row["date"] = date
                    row["prevision"] = prevision
                    if include_ci:
                        row["ic_bas"] = ic_bas
                        row["ic_haut"] = ic_haut
                    rows.append(row)

                job["processed"] += 1

        if not rows:
            job["status"] = "error"
            job["error"] = "Aucune entité n'a pu être prévue (historique trop court partout ?)"
            return

        # Les ajustements se terminent dans un ordre qui dépend de quel
        # worker finit en premier (pas de la soumission) : on retrie sur
        # l'ordre déjà correct de `combos` (numérique par dimension) plutôt
        # que de laisser un CSV mélangé selon la parallélisation.
        combo_rank = {combo: i for i, combo in enumerate(combos)}
        rows.sort(key=lambda r: (combo_rank[tuple(r[col] for col in group_cols)], r["date"]))

        job["csv"] = pd.DataFrame(rows).to_csv(index=False)
        job["skipped"] = skipped
        job["status"] = "done"
    except HTTPException as exc:
        job["status"] = "error"
        job["error"] = exc.detail
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)


def resolve_bulk_groups(
    raw: pd.DataFrame,
    store_col: str | None,
    store_value: str | None,
    category_col: str | None,
    category_value: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Applique les valeurs fixées (magasin et/ou catégorie donnés avec une
    valeur précise) et renvoie les colonnes restant "libres" — celles à
    parcourir pour une prévision groupée.
    """
    fixed = {}
    for col, value in ((store_col, store_value), (category_col, category_value)):
        if col:
            if col not in raw.columns:
                raise HTTPException(400, "Colonne identifiant introuvable dans le CSV")
            if value:
                fixed[col] = value

    for col, value in fixed.items():
        raw = raw[raw[col].astype(str) == str(value)]

    group_cols = [col for col in (store_col, category_col) if col and col not in fixed]
    return raw, group_cols


@app.post("/api/upload/forecast/bulk/count", dependencies=[Depends(current_user)])
async def api_upload_forecast_bulk_count(
    file: UploadFile = File(...),
    store_col: str | None = Form(None),
    store_value: str | None = Form(None),
    category_col: str | None = Form(None),
    category_value: str | None = Form(None),
):
    """Renvoie le nombre d'entités (magasins, catégories, ou combinaisons des
    deux) qu'une prévision groupée parcourrait — utilisé côté client pour
    afficher une estimation du nombre de lignes avant de lancer le job.
    """
    raw = _read_csv_flexible(await read_upload_limited(file))
    raw, group_cols = resolve_bulk_groups(raw, store_col, store_value, category_col, category_value)
    return {"n_entities": len(group_combinations(raw, group_cols)) if group_cols else 0}


@app.post("/api/upload/forecast/bulk/start", dependencies=[Depends(current_user)])
async def api_upload_forecast_bulk_start(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    store_col: str | None = Form(None),
    store_value: str | None = Form(None),
    category_col: str | None = Form(None),
    category_value: str | None = Form(None),
    horizon: int = Form(28),
    model: str = Form("sarimax"),
    model_params: str | None = Form(None),
    include_ci: bool = Form(True),
    extra_exog_cols: str | None = Form(None),
    exog_file: UploadFile | None = File(None),
    exog_date_col: str | None = Form(None),
    exog_value_cols: str | None = Form(None),
    exog_store_col: str | None = Form(None),
    exog_category_col: str | None = Form(None),
    location: str | None = Form(None),
    use_ferie: bool = Form(True),
    use_vacances: bool = Form(True),
    use_meteo: bool = Form(True),
):
    """Démarre en tâche de fond une prévision pour chaque entité détectée
    dans le fichier : une série par valeur distincte de store_col, par
    valeur distincte de category_col, ou — si les deux sont fournis sans
    valeur fixée — par combinaison magasin × catégorie. Renvoie tout de
    suite un job_id à interroger via /progress puis /download — permet
    d'afficher une progression côté client sur un traitement qui peut
    prendre plusieurs minutes.
    """
    raw = _read_csv_flexible(await read_upload_limited(file))
    for col in (date_col, value_col):
        if col not in raw.columns:
            raise HTTPException(400, "Colonnes introuvables dans le CSV")

    extra_cols = parse_form_list(extra_exog_cols)
    for col in extra_cols:
        if col not in raw.columns:
            raise HTTPException(400, "Colonne exogène introuvable dans le CSV")
    external_exog = await load_external_exog(exog_file, exog_date_col, exog_value_cols, exog_store_col, exog_category_col)

    raw, group_cols = resolve_bulk_groups(raw, store_col, store_value, category_col, category_value)
    if not group_cols:
        raise HTTPException(400, "Aucune dimension à parcourir pour la prévision groupée")

    if model not in BULK_MODELS:
        raise HTTPException(400, f"Modèle inconnu : {model}")

    # Pool créé ici, dans le thread principal, plutôt qu'à la demande depuis
    # le thread de travail : forker alors qu'uvicorn, cmdstan et libgomp ont
    # déjà lancé leurs threads peut laisser un worker figé sur un verrou
    # copié, et le job resterait "running" indéfiniment. Reste paresseux (pas
    # de fork à l'import, cf. get_bulk_executor) et inutile pour lightgbm,
    # qui entraîne un modèle global sans passer par le pool.
    if model != "lightgbm":
        get_bulk_executor()

    combos = group_combinations(raw, group_cols)
    job_id = uuid.uuid4().hex
    purge_stale_bulk_jobs()
    bulk_jobs[job_id] = {
        "status": "running",
        "processed": 0,
        "total": len(combos),
        "combos": combos,
        "skipped": 0,
        "csv": None,
        "error": None,
        "created_at": time.time(),
    }

    thread = threading.Thread(
        target=_run_bulk_forecast,
        args=(
            job_id, raw, date_col, value_col, group_cols, horizon, model, include_ci,
            location, use_ferie, use_vacances, use_meteo, extra_cols, external_exog,
            store_col, category_col, exog_date_col, parse_form_list(exog_value_cols),
            exog_store_col, exog_category_col, parse_model_params(model_params),
        ),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "total": len(combos)}


@app.get("/api/upload/forecast/bulk/progress/{job_id}", dependencies=[Depends(current_user)])
def api_upload_forecast_bulk_progress(job_id: str):
    job = bulk_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job introuvable")
    return {
        "status": job["status"],
        "processed": job["processed"],
        "total": job["total"],
        "skipped": job["skipped"],
        "error": job["error"],
    }


@app.get("/api/upload/forecast/bulk/download/{job_id}", dependencies=[Depends(current_user)])
def api_upload_forecast_bulk_download(job_id: str):
    job = bulk_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job introuvable")
    if job["status"] == "running":
        raise HTTPException(409, "Prévision encore en cours")
    if job["status"] == "error":
        # libéré ici aussi : sans ça un job en erreur n'était jamais retiré
        del bulk_jobs[job_id]
        raise HTTPException(400, job["error"])

    csv_text = job["csv"]
    processed = job["total"] - job["skipped"]
    skipped = job["skipped"]
    del bulk_jobs[job_id]

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="previsions_toutes_entites.csv"',
            "X-Stores-Processed": str(processed),
            "X-Stores-Skipped": str(skipped),
        },
    )


@app.post("/api/points")
async def api_save_point(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    name: str = Form(...),
    store_col: str | None = Form(None),
    store_value: str | None = Form(None),
    category_col: str | None = Form(None),
    category_value: str | None = Form(None),
    location: str | None = Form(None),
    use_ferie: bool = Form(True),
    use_vacances: bool = Form(True),
    use_meteo: bool = Form(True),
    user: dict = Depends(current_user),
):
    df = parse_csv(await read_upload_limited(file), date_col, value_col, store_col, store_value, category_col, category_value)
    point_id = storage.save_point(user["id"], name, df, location, use_ferie, use_vacances, use_meteo)
    return {"id": point_id, "name": name}


@app.get("/api/points")
def api_list_points(user: dict = Depends(current_user)):
    return storage.list_points(user["id"])


@app.delete("/api/points/{point_id}")
def api_delete_point(point_id: int, user: dict = Depends(current_user)):
    if not storage.delete_point(user["id"], point_id):
        raise HTTPException(404, "Point de vente introuvable")
    return {"deleted": True}


def load_point_or_404(user: dict, point_id: int) -> dict:
    point = storage.load_point(user["id"], point_id)
    if point is None:
        raise HTTPException(404, "Point de vente introuvable")
    return point


@app.get("/api/points/{point_id}/analysis")
def api_point_analysis(point_id: int, user: dict = Depends(current_user)):
    point = load_point_or_404(user, point_id)
    flags, weather = flags_and_weather(
        point["df"], point["location"], point["use_ferie"], point["use_vacances"], point["use_meteo"]
    )
    return analyze(point["df"], flags=flags, weather=weather)


@app.get("/api/points/{point_id}/forecast")
def api_point_forecast(
    point_id: int,
    horizon: int = Query(28, ge=7, le=90),
    model: str = Query("sarimax"),
    model_params: str | None = Query(None),
    user: dict = Depends(current_user),
):
    point = load_point_or_404(user, point_id)
    exog_hist, exog_future = forecast_exog(
        point["df"], horizon, point["location"], point["use_ferie"], point["use_vacances"], point["use_meteo"]
    )
    return forecast(
        point["df"], horizon, exog_hist=exog_hist, exog_future=exog_future, model=model,
        model_params=parse_model_params(model_params).get(model),
    )


@app.get("/api/points/{point_id}/backtest")
def api_point_backtest(
    point_id: int,
    horizon: int = Query(28, ge=7, le=90),
    model_params: str | None = Query(None),
    user: dict = Depends(current_user),
):
    point = load_point_or_404(user, point_id)
    exog_hist, exog_future = backtest_exog(
        point["df"], horizon, point["location"], point["use_ferie"], point["use_vacances"], point["use_meteo"]
    )
    return backtest(
        point["df"], horizon, exog_hist=exog_hist, exog_future=exog_future,
        model_params=parse_model_params(model_params),
    )


@app.get("/api/points/{point_id}/autotune")
def api_point_autotune(
    point_id: int,
    horizon: int = Query(28, ge=7, le=90),
    model_params: str | None = Query(None),
    user: dict = Depends(current_user),
):
    point = load_point_or_404(user, point_id)
    exog_hist, exog_future = backtest_exog(
        point["df"], horizon, point["location"], point["use_ferie"], point["use_vacances"], point["use_meteo"]
    )
    return autotune(
        point["df"], horizon, exog_hist=exog_hist, exog_future=exog_future,
        model_params=parse_model_params(model_params),
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
