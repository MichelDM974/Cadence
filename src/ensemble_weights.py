"""Poids d'ensemble LightGBM / SARIMAX propres à chaque série,
à la FFORMA (Montero-Manso et al., 2020) : un méta-modèle LightGBM prédit
la part de LightGBM d'une série à partir de descripteurs de son historique
(niveau, bruit, zéros, saisonnalité hebdomadaire, tendance, écart entre les
deux membres). Il apprend sur des fenêtres de backtest qui précèdent la
période prévue : la part optimale de chaque série sur chaque fenêtre, pondérée
par ce que le choix du poids change pour elle. Généralise d'une série à
l'autre au lieu de mémoriser chacune (16 jours par série : trop bruité).

Score officiel (10/10/2026) : bat le 50/50 sur Store Sales (0.40371 contre
0.41075) et Recruit (0.55349 contre 0.55433), cf. BENCHMARKS.md.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

import kaggle_competition as kc

MEMBERS = ("lightgbm", "sarimax")
GRID = np.round(np.linspace(0, 1, 21), 2)  # part de LightGBM
# Fenêtres d'apprentissage exigées : avec une seule (historique court),
# le poids appris s'est montré instable (Recruit an-1 : tout LightGBM,
# 0.761 contre 0.742 en 50/50). En deçà, l'appelant garde le 50/50.
LEARNING_WINDOWS = 3

META_PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
               "feature_fraction": 0.8, "verbose": -1, "seed": 0}
META_NUM_BOOST_ROUND = 200


def align(targets, frames: dict, history, spec, zero_dead_days: int | None) -> pd.DataFrame:
    """Une ligne par ligne de `targets` (ordre conservé) : prévision de
    chaque membre (`frames` : membre -> predictions_frame), série éteinte
    ou non (`_alive`) et clé de série (`_key`)."""
    on = [*spec.entity_cols, "date"]
    df = targets.copy()
    for col in spec.entity_cols:
        df[col] = df[col].astype(str)
    for member in MEMBERS:
        df = df.merge(frames[member].rename(columns={"prevision": member}), on=on, how="left")
        # membre sans aucune prévision : colonne vide de type objet
        df[member] = pd.to_numeric(df[member], errors="coerce").astype(float)
    # un membre absent : l'autre seul (comme blend_predictions)
    df["lightgbm"] = df["lightgbm"].fillna(df["sarimax"])
    df["sarimax"] = df["sarimax"].fillna(df["lightgbm"])
    # aucun des deux : repli de la page (moyenne récente), quel que soit le poids
    if df["lightgbm"].isna().any():
        empty = kc.predictions_frame([], spec)
        fallback, _ = kc.attach_predictions(df[on], empty, history, spec, clip_zero=False)
        for member in MEMBERS:
            df[member] = df[member].fillna(pd.Series(fallback.to_numpy(), index=df.index))
    if zero_dead_days:
        dead = kc.zero_dead_series(df[on].assign(prevision=1.0), history, spec, zero_dead_days)
        df = df.merge(dead.rename(columns={"prevision": "_alive"}), on=on, how="left")
        df["_alive"] = df["_alive"].fillna(1.0)
    else:
        df["_alive"] = 1.0
    df["_key"] = df[spec.entity_cols].astype(str).agg("|".join, axis=1)
    return df


def descriptors(history, spec, df) -> pd.DataFrame:
    """Descripteurs par série, calculés sur le seul historique connu."""
    daily = history.groupby([*spec.entity_cols, spec.date_col], observed=True)[spec.target_col].sum().reset_index()
    daily["_key"] = daily[spec.entity_cols].astype(str).agg("|".join, axis=1)
    end = daily[spec.date_col].max()
    feats = {}
    for key, g in daily.groupby("_key", sort=False):
        y = g.set_index(spec.date_col)[spec.target_col].sort_index()
        nz = y[y > 0]
        start = nz.index.min() if len(nz) else end
        y = y[y.index >= start]
        recent = y[y.index > end - pd.Timedelta(days=90)]
        log = np.log1p(recent.clip(lower=0))
        dow = log.groupby(log.index.dayofweek).transform("mean")
        var = log.var()
        last28, prev = y[y.index > end - pd.Timedelta(days=28)], y[(y.index <= end - pd.Timedelta(days=28))
                                                                   & (y.index > end - pd.Timedelta(days=112))]
        feats[key] = {
            "length": (end - start).days,
            "level": float(np.log1p(recent.mean())) if len(recent) else 0.0,
            "cv": float(recent.std() / recent.mean()) if len(recent) and recent.mean() > 0 else 0.0,
            "zeros": float((recent == 0).mean()) if len(recent) else 1.0,
            "weekly": float(1 - ((log - dow).var() / var)) if var > 0 else 0.0,
            "trend": float(np.log1p(last28.mean()) - np.log1p(prev.mean())) if len(prev) else 0.0,
        }
    out = pd.DataFrame.from_dict(feats, orient="index")
    pred = df.groupby("_key")[["lightgbm", "sarimax"]].mean()
    out["gap"] = np.log1p(pred["lightgbm"].clip(lower=0)) - np.log1p(pred["sarimax"].clip(lower=0))
    return out.fillna(0.0)


def blend(df, weight) -> np.ndarray:
    """`weight` : scalaire ou part de LightGBM par ligne."""
    values = weight * df["lightgbm"].to_numpy() + (1 - weight) * df["sarimax"].to_numpy()
    return np.clip(values * df["_alive"].to_numpy(), 0, None)


def series_sse(df) -> pd.DataFrame:
    """Somme des carrés d'erreur log par série et par poids de la grille
    (`df` doit porter la vérité dans `actual`)."""
    log_actual = np.log1p(df["actual"].clip(lower=0).to_numpy())
    cols = {w: (np.log1p(blend(df, w)) - log_actual) ** 2 for w in GRID}
    return pd.DataFrame(cols).groupby(df["_key"].to_numpy()).sum()


def global_weight(windows) -> float:
    """Meilleur poids unique sur les fenêtres (df, descripteurs) ; séries
    absentes d'une fenêtre (pas encore ouvertes) : sommées sur les autres."""
    sse = pd.concat([series_sse(d) for d, _ in windows]).groupby(level=0).sum()
    return float(sse.sum().idxmin())


def fit(windows) -> lgb.Booster:
    """Méta-modèle : une ligne par (série, fenêtre) ; cible = poids optimal,
    pondérée par ce que le choix du poids change pour la série."""
    xs, ys, ws = [], [], []
    for d, f in windows:
        # séries ouvertes pendant la fenêtre notée : aucun historique à décrire
        s = series_sse(d)
        s = s.loc[s.index.intersection(f.index)]
        xs.append(f.loc[s.index])
        ys.append(s.idxmin(axis=1).astype(float))
        ws.append(s.max(axis=1) - s.min(axis=1))
    x, y, sw = pd.concat(xs), pd.concat(ys), pd.concat(ws)
    return lgb.train(META_PARAMS, lgb.Dataset(x, y, weight=sw.clip(lower=1e-6)), num_boost_round=META_NUM_BOOST_ROUND)


def series_weights(meta: lgb.Booster, feats: pd.DataFrame) -> pd.Series:
    """Part de LightGBM prédite pour chaque série (index = `_key`)."""
    return pd.Series(np.clip(meta.predict(feats), 0, 1), index=feats.index)


def row_weights(windows, df, feats) -> np.ndarray:
    """Part de LightGBM de chaque ligne de `df`, apprise sur `windows` ;
    série sans descripteur : poids global des mêmes fenêtres."""
    w = series_weights(fit(windows), feats)
    return w.reindex(df["_key"]).fillna(global_weight(windows)).to_numpy()
