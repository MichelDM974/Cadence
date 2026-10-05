"""Prévision en masse par modèle global LightGBM : un seul modèle entraîné
sur toutes les entités à la fois (magasin x catégorie, ...), plutôt qu'un fit
indépendant par entité comme SARIMAX/Prophet (cf. _run_bulk_forecast dans
api.py, qui reste le chemin par défaut). Réservé à la génération en masse —
volontairement absent de FORECAST_MODELS pour ne jamais apparaître dans les
endpoints d'analyse d'une seule entité (décomposition, régularité, backtest).

Ce module ne dépend pas de api.py (pour éviter un import circulaire) : tout
ce dont il a besoin (la fonction de mise en série quotidienne, l'alignement
d'un exogène externe, le cache d'exogène de localisation) lui est passé en
paramètre par l'appelant.
"""

import os
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)
TAIL_WINDOW_DAYS = 35  # >= max(lag, rolling) + marge : borne la fenêtre glissante de la boucle récursive

LGB_PARAMS = {
    # tweedie plutôt que L2 sur log1p(visitors) : la cible (fréquentation/
    # ventes quotidiennes) est typiquement zero-inflated et très asymétrique
    # (beaucoup d'entités à faible volume, quelques pics) — L2 sur cible
    # log-transformée introduit un biais de sous-prévision documenté sur ce
    # type de données (cf. benchmark Kaggle *Store Sales*, journal-de-bord :
    # 0.55166 -> 0.42475 RMSLE avec ce même changement). Approche utilisée
    # par la gagnante du Kaggle M5 Forecasting - Accuracy (LightGBM+tweedie).
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,  # garde-fou : évite un surapprentissage sur les entités à historique court
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "num_threads": os.cpu_count() or 4,
}
NUM_BOOST_ROUND = 300  # fixe, pas de early stopping/tuning en v1


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    lag_days: tuple[int, ...] = LAG_DAYS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
    include_yoy: bool = False,
) -> pd.DataFrame:
    """Ajoute retards et moyennes glissantes de log1p(visitors), calculés en
    quelques appels vectorisés sur toutes les entités à la fois (jamais une
    boucle Python par entité — le vrai goulot d'étranglement sur un gros
    fichier, cf. le même principe déjà appliqué ailleurs dans ce projet pour
    le regroupement des lignes par entité).

    `lag_days`/`rolling_windows`/`include_yoy` sont des overrides opt-in (cf.
    benchmarks *Store Sales* et *Recruit*, journal-de-bord) : les défauts
    reproduisent le comportement historique (lean) de l'app, inchangé tant
    que l'appelant ne passe rien. `include_yoy` suppose `28` dans
    `rolling_windows` (sert de base à `yoy_trend`) et au moins ~1 an
    d'historique pour être utile — sans quoi la feature reste NaN.
    """
    df = df.sort_values([*group_cols, "date"]).reset_index(drop=True)
    df["y"] = np.log1p(df["visitors"].clip(lower=0))

    grouped = df.groupby(group_cols, observed=True)["y"]
    for k in lag_days:
        df[f"lag_{k}"] = grouped.shift(k)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = grouped.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    if include_yoy:
        df["roll_mean_28_yoy"] = grouped.transform(lambda s: s.shift(1 + 364).rolling(28, min_periods=1).mean())
        df["yoy_trend"] = df["roll_mean_28"] - df["roll_mean_28_yoy"]

    return df


ENTITY_HISTORY_CV_THRESHOLD = 0.1
REDUCED_REGULARIZATION_PARAMS = {"min_data_in_leaf": 10, "lambda_l2": 0.05}


def _entity_history_length_cv(history: pd.DataFrame, group_cols: list[str]) -> float | None:
    """Coefficient de variation des longueurs d'historique entre entités —
    signal partagé derrière les deux décisions automatiques ci-dessous.
    `None` quand il ne peut pas être calculé de façon significative (moins
    de 2 entités) : les deux décisions traitent ça comme "pas de signal",
    jamais comme une hétérogénéité nulle par défaut.
    """
    lengths = history.groupby(group_cols, observed=True).size()
    if len(lengths) < 2 or lengths.mean() == 0:
        return None
    return float(lengths.std(ddof=0) / lengths.mean())


def _should_auto_include_yoy(history: pd.DataFrame, group_cols: list[str]) -> bool:
    """Décide s'il faut activer `yoy_trend` faute de préférence explicite de
    l'appelant, à partir de la dispersion des longueurs d'historique entre
    entités (coefficient de variation) : nul par construction sur un panel
    homogène (toutes les entités ont exactement le même historique, ex.
    *Store Sales* : cv=0.0) — l'axe n'y a jamais aidé (cf. benchmark
    04/09/2026, journal-de-bord) et reste désactivé, comportement lean
    inchangé. Nettement positif dès que les entités ont des historiques de
    longueurs très inégales (ex. *Recruit*, restaurants à dates d'ouverture
    différentes : cv=0.31) — l'axe y aide (même benchmark), donc activé
    automatiquement. Seuil choisi avec large marge entre ces deux cas connus,
    pas calé pile sur l'un des deux.
    """
    cv = _entity_history_length_cv(history, group_cols)
    return cv is not None and cv > ENTITY_HISTORY_CV_THRESHOLD


def _should_auto_reduce_regularization(history: pd.DataFrame, group_cols: list[str]) -> bool:
    """Pendant symétrique de `_should_auto_include_yoy`, sur le même signal :
    régularisation réduite (`REDUCED_REGULARIZATION_PARAMS`) activée quand
    l'historique est homogène entre entités (cv bas, ex. *Store Sales*) —
    confirmé en isolation le 07/09/2026 (RMSLE backtest local 0.54118 ->
    0.53694, cf. BENCHMARKS.md), alors que le seul test connu sur historique
    hétérogène combinait cet axe à d'autres (résultat non isolé, donc jamais
    activé faute de preuve) : gardé désactivé dès que cv > seuil ou
    indéterminé, jamais le contraire par défaut.
    """
    cv = _entity_history_length_cv(history, group_cols)
    return cv is not None and cv <= ENTITY_HISTORY_CV_THRESHOLD


def _build_code_maps(group_cols: list[str], combos: list[tuple[str, ...]]) -> dict[str, dict[str, int]]:
    """Encode chaque valeur d'entité en entier, une fois pour toutes, à
    partir de `combos` (déjà trié) — un code entier plutôt que le dtype
    'category' de pandas, qui ne survit pas de façon fiable aux `concat`
    répétés de la boucle de prévision récursive.
    """
    maps: dict[str, dict[str, int]] = {col: {} for col in group_cols}
    for combo in combos:
        for col, val in zip(group_cols, combo):
            if val not in maps[col]:
                maps[col][val] = len(maps[col])
    return maps


def _apply_code_maps(df: pd.DataFrame, group_cols: list[str], code_maps: dict[str, dict[str, int]]) -> None:
    for col in group_cols:
        df[f"{col}__code"] = df[col].map(code_maps[col]).astype("int32")


def _build_entity_frames(
    raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_cols: list[str],
    horizon: int,
    combos: list[tuple[str, ...]],
    group_indices: dict[tuple[str, ...], pd.Index],
    extra_cols: list[str],
    location_exog_cached: Callable[[pd.DatetimeIndex], pd.DataFrame] | None,
    location_columns: list[str],
    external_by_entity: dict[tuple[str, ...], pd.DataFrame] | None,
    external_exog: pd.DataFrame | None,
    fixed_store_value: str | None,
    fixed_category_value: str | None,
    store_col: str | None,
    category_col: str | None,
    exog_store_col: str | None,
    exog_category_col: str | None,
    daily_series_fn: Callable[..., pd.DataFrame],
    align_external_exog_fn: Callable[[pd.DataFrame, pd.DatetimeIndex], pd.DataFrame],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, set[tuple[str, ...]]]:
    """Assemble, pour chaque entité, sa série quotidienne historique (avec
    cible) et le tableau des exogènes connues sur l'horizon à prévoir (sans
    cible) — même logique d'assemblage que la boucle SARIMAX/Prophet de
    _run_bulk_forecast, réutilisée ici pour nourrir le modèle global. Les
    deux tableaux sont ensuite empilés (une seule fois) plutôt que
    reconstruits par entité à chaque usage.
    """
    history_frames, future_frames = [], []
    skipped: set[tuple[str, ...]] = set()

    for combo in combos:
        idx = group_indices.get(combo)
        if idx is None:
            skipped.add(combo)
            continue
        subset = raw.loc[idx]

        try:
            daily = daily_series_fn(subset, date_col, value_col, extra_cols)
        except Exception:
            skipped.add(combo)
            continue

        future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        future = pd.DataFrame(index=future_dates)

        if location_exog_cached is not None and location_columns:
            daily = daily.join(location_exog_cached(daily.index)[location_columns])
            future = future.join(location_exog_cached(future_dates)[location_columns])

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
            daily = daily.join(align_external_exog_fn(entity_exog, daily.index))
            future = future.join(align_external_exog_fn(entity_exog, future_dates))

        if extra_cols:
            last_values = daily[extra_cols].iloc[-1]
            for col in extra_cols:
                future[col] = last_values[col]

        daily.index.name = "date"
        future.index.name = "date"
        daily = daily.reset_index()
        future = future.reset_index()
        for col, val in zip(group_cols, combo):
            daily[col] = val
            future[col] = val

        history_frames.append(daily)
        future_frames.append(future)

    if not history_frames:
        return None, None, skipped

    history = pd.concat(history_frames, ignore_index=True, sort=False)
    future = pd.concat(future_frames, ignore_index=True, sort=False)
    return history, future, skipped


def train_bulk_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    lgb_params: dict | None = None,
    num_boost_round: int | None = None,
) -> lgb.Booster:
    dataset = lgb.Dataset(
        train_df[feature_cols],
        # cible en échelle brute (pas "y"/log1p) : tweedie applique son
        # propre log-link en interne, cf. LGB_PARAMS. "y" reste calculé par
        # add_lag_features, mais uniquement comme base des lags/moyennes
        # glissantes en features, pas comme cible d'entraînement.
        label=train_df["visitors"],
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
        free_raw_data=False,
    )
    return lgb.train(
        {**LGB_PARAMS, **(lgb_params or {})}, dataset, num_boost_round=num_boost_round or NUM_BOOST_ROUND
    )


def predict_bulk(
    booster: lgb.Booster | list[lgb.Booster],
    history: pd.DataFrame,
    future_exog: pd.DataFrame,
    group_cols: list[str],
    feature_cols: list[str],
    horizon: int,
    on_progress: Callable[[int, int], None] | None = None,
    lag_days: tuple[int, ...] = LAG_DAYS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
    include_yoy: bool = False,
    tail_window_days: int = TAIL_WINDOW_DAYS,
    round_decimals: int = 1,
) -> pd.DataFrame:
    """Prévision récursive : à chaque jour de l'horizon, ne retraite qu'une
    fenêtre glissante bornée (tail_window_days x nb d'entités, pas tout
    l'historique), recalcule les features pour toutes les entités d'un coup,
    un seul `predict()` pour ce jour, puis réinjecte la prédiction comme
    valeur connue pour l'étape suivante.
    """
    boosters = booster if isinstance(booster, list) else [booster]
    tail = history.sort_values([*group_cols, "date"]).reset_index(drop=True)
    rank_from_end = tail.groupby(group_cols, observed=True).cumcount(ascending=False)
    tail = tail[rank_from_end < tail_window_days].reset_index(drop=True)

    future_dates = sorted(future_exog["date"].unique())
    results = []

    for h, target_date in enumerate(future_dates, start=1):
        day_exog = future_exog[future_exog["date"] == target_date].reset_index(drop=True)
        day_exog["visitors"] = np.nan  # pas encore connu ; sert juste à passer par add_lag_features sans erreur

        combined = pd.concat([tail, day_exog], ignore_index=True, sort=False)
        combined = add_calendar_features(combined)
        # add_lag_features trie par [*group_cols, "date"] en interne : l'ordre
        # des lignes en sortie ne correspond plus à celui de `day_exog`. Les
        # labels (group_cols/date) doivent donc être relus depuis `combined`
        # elle-même, pas depuis `day_exog` — sinon les prévisions se
        # retrouvent attribuées à la mauvaise entité (bug réel, détecté via
        # un score Kaggle très dégradé : des magasins à volume quasi nul
        # héritaient des prévisions de magasins à gros volume).
        combined = add_lag_features(combined, group_cols, lag_days, rolling_windows, include_yoy)

        today_rows = combined.loc[combined["date"] == target_date].reset_index(drop=True)
        # tweedie prédit déjà en échelle brute (log-link inversé en interne) : pas d'expm1 ici
        # plusieurs graines (bagging) : moyenne à chaque pas, avant réinjection
        pred_visitors = np.clip(np.mean([b.predict(today_rows[feature_cols]) for b in boosters], axis=0), 0, None)

        day_result = today_rows[[*group_cols, "date"]].copy()
        day_result["prevision"] = np.round(pred_visitors, round_decimals)
        results.append(day_result)

        predicted_rows = today_rows[tail.columns].copy()
        predicted_rows["visitors"] = pred_visitors
        tail = pd.concat([tail, predicted_rows], ignore_index=True, sort=False)
        rank_from_end = tail.groupby(group_cols, observed=True).cumcount(ascending=False)
        tail = tail[rank_from_end < tail_window_days].reset_index(drop=True)

        if on_progress is not None:
            on_progress(h, horizon)

    return pd.concat(results, ignore_index=True)


def run_bulk_lightgbm_forecast(
    raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_cols: list[str],
    horizon: int,
    combos: list[tuple[str, ...]],
    group_indices: dict[tuple[str, ...], pd.Index],
    extra_cols: list[str],
    location_exog_cached: Callable[[pd.DatetimeIndex], pd.DataFrame] | None,
    location_columns: list[str],
    external_by_entity: dict[tuple[str, ...], pd.DataFrame] | None,
    external_exog: pd.DataFrame | None,
    fixed_store_value: str | None,
    fixed_category_value: str | None,
    store_col: str | None,
    category_col: str | None,
    exog_store_col: str | None,
    exog_category_col: str | None,
    daily_series_fn: Callable[..., pd.DataFrame],
    align_external_exog_fn: Callable[[pd.DataFrame, pd.DatetimeIndex], pd.DataFrame],
    on_progress: Callable[[int, int], None] | None = None,
    lgb_params: dict | None = None,
    num_boost_round: int | None = None,
    feature_config: dict | None = None,
) -> tuple[list[dict], set[tuple[str, ...]]]:
    """Point d'entrée appelé depuis _run_bulk_forecast (api.py) quand
    model == "lightgbm". Retourne (rows, skipped_combos) dans la forme
    attendue par l'assemblage CSV existant (tri + to_csv), réutilisé tel
    quel côté appelant.

    `feature_config` (opt-in, absent par défaut = comportement lean actuel) :
    `{"lag_days": (...), "rolling_windows": (...), "include_yoy": bool}` —
    cf. benchmarks *Store Sales*/*Recruit* (journal-de-bord, BENCHMARKS.md)
    pour l'origine de ces axes. `lag_days`/`rolling_windows` restent lean par
    défaut (pas de preuve isolée d'amélioration universelle). `include_yoy`
    et la régularisation (`lgb_params`, si non fourni explicitement par
    l'appelant) sont eux décidés automatiquement à partir du même signal —
    dispersion des longueurs d'historique entre entités, cf.
    `_should_auto_include_yoy`/`_should_auto_reduce_regularization` — les
    deux seuls axes dont l'utilité se détecte de façon fiable à partir des
    données elles-mêmes plutôt que de dépendre d'un choix utilisateur.
    """
    history, future_exog, skipped = _build_entity_frames(
        raw, date_col, value_col, group_cols, horizon, combos, group_indices, extra_cols,
        location_exog_cached, location_columns,
        external_by_entity, external_exog, fixed_store_value, fixed_category_value,
        store_col, category_col, exog_store_col, exog_category_col,
        daily_series_fn, align_external_exog_fn,
    )
    if history is None:
        return [], skipped

    code_maps = _build_code_maps(group_cols, combos)
    _apply_code_maps(history, group_cols, code_maps)
    _apply_code_maps(future_exog, group_cols, code_maps)

    feature_config = feature_config or {}
    lag_days = tuple(feature_config.get("lag_days", LAG_DAYS))
    rolling_windows = tuple(feature_config.get("rolling_windows", ROLLING_WINDOWS))
    if "include_yoy" in feature_config:
        include_yoy = bool(feature_config["include_yoy"])
    else:
        include_yoy = _should_auto_include_yoy(history, group_cols)
    if lgb_params is None and _should_auto_reduce_regularization(history, group_cols):
        lgb_params = REDUCED_REGULARIZATION_PARAMS
    tail_window_days = max(TAIL_WINDOW_DAYS, max(*lag_days, *rolling_windows) + 7, 364 + 28 + 7 if include_yoy else 0)

    train_df = add_lag_features(add_calendar_features(history.copy()), group_cols, lag_days, rolling_windows, include_yoy)
    code_cols = [f"{c}__code" for c in group_cols]
    non_feature = {"date", "visitors", "y", *group_cols}
    feature_cols = [c for c in train_df.columns if c not in non_feature]

    # `seeds` (opt-in) : un modèle par graine, prévisions moyennées.
    seeds = feature_config.get("seeds") or [None]
    boosters = [
        train_bulk_model(
            train_df, feature_cols, code_cols,
            {**(lgb_params or {}), "seed": seed} if seed is not None else lgb_params, num_boost_round,
        )
        for seed in seeds
    ]
    round_decimals = int(feature_config.get("round_decimals", 1))
    predictions = predict_bulk(
        boosters, history, future_exog, group_cols, feature_cols, horizon, on_progress,
        lag_days, rolling_windows, include_yoy, tail_window_days, round_decimals,
    )

    rows = []
    for rec in predictions.to_dict("records"):
        row = {col: rec[col] for col in group_cols}
        row["date"] = pd.Timestamp(rec["date"]).date().isoformat()
        row["prevision"] = round(float(rec["prevision"]), round_decimals)
        rows.append(row)

    return rows, skipped
