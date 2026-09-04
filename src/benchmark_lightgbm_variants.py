"""Compare, sur deux jeux de données Kaggle réels (*Store Sales* : historique
long/homogène ; *Recruit* : historique court/hétérogène par restaurant), le
LightGBM lean actuel de l'app (`lightgbm_forecast.py`) contre les 3 pistes
candidates identifiées dans le journal-de-bord (lags fins, lag annuel/
yoy_trend, régularisation réduite) — en appelant directement
`run_bulk_lightgbm_forecast`, le vrai point d'entrée de production, pas une
réimplémentation séparée.

But : décider quels axes sont sûrs par défaut (améliorent ou n'affectent pas
les deux datasets) vs doivent rester opt-in (aident l'un, pénalisent l'autre,
notamment les entités à historique court côté Recruit).

Usage : python benchmark_lightgbm_variants.py [store_sales|recruit|both]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api import _daily_series, align_external_exog, group_combinations  # noqa: E402
from lightgbm_forecast import run_bulk_lightgbm_forecast  # noqa: E402

STORE_SALES_DIR = "/home/delhomme/Téléchargements/store-sales-time-series-forecasting"
RECRUIT_DIR = str(Path(__file__).resolve().parent.parent / "data" / "raw")

VARIANTS = {
    "lean (défaut actuel)": {},
    "lags fins": {"feature_config": {"lag_days": (1, 2, 3, 4, 5, 6, 7, 14, 21, 28), "rolling_windows": (7, 14, 28)}},
    "lag annuel + yoy_trend": {"feature_config": {"rolling_windows": (7, 28), "include_yoy": True}},
    "régularisation réduite": {"lgb_params": {"min_data_in_leaf": 10, "lambda_l2": 0.05}},
    "lags fins + yoy (combiné)": {
        "feature_config": {"lag_days": (1, 2, 3, 4, 5, 6, 7, 14, 21, 28), "rolling_windows": (7, 14, 28), "include_yoy": True}
    },
}


def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.clip(actual, 0, None)
    predicted = np.clip(predicted, 0, None)
    return float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)))


def _run_variant(raw: pd.DataFrame, group_cols: list[str], horizon: int, variant_kwargs: dict) -> pd.DataFrame:
    combos = group_combinations(raw, group_cols)
    str_cols = {c: raw[c].astype(str) for c in group_cols}
    group_indices = dict(raw.groupby([str_cols[c] for c in group_cols]).groups.items())

    rows, skipped = run_bulk_lightgbm_forecast(
        raw, "date", "sales", group_cols, horizon, combos, group_indices, [],
        None, [],
        None, None, None, None,
        group_cols[0], group_cols[1] if len(group_cols) > 1 else None, None, None,
        _daily_series, align_external_exog, None,
        **variant_kwargs,
    )
    if skipped:
        print(f"    ({len(skipped)} entités ignorées, historique < 3 semaines)")
    return pd.DataFrame(rows)


def _score(raw: pd.DataFrame, group_cols: list[str], horizon: int, truth: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label} ---")
    for name, kwargs in VARIANTS.items():
        start = time.time()
        preds = _run_variant(raw, group_cols, horizon, kwargs)
        preds["date"] = preds["date"].astype(str)
        merged = truth.merge(preds, on=[*group_cols, "date"], how="left")
        merged["prevision"] = merged["prevision"].fillna(0)
        score = rmsle(merged["sales"].to_numpy(), merged["prevision"].to_numpy())
        print(f"  {name:28s} RMSLE={score:.5f}  ({time.time() - start:.0f}s)")


def bench_store_sales() -> None:
    train = pd.read_csv(f"{STORE_SALES_DIR}/train.csv", parse_dates=["date"])
    train = train.rename(columns={"sales": "sales"})[["date", "store_nbr", "family", "sales"]]
    train["store_nbr"] = train["store_nbr"].astype(str)

    horizon = 16
    cutoff = train["date"].max() - pd.Timedelta(days=horizon - 1)
    fit_train = train[train["date"] < cutoff].copy()
    truth = train[train["date"] >= cutoff][["store_nbr", "family", "date", "sales"]].copy()
    truth["date"] = truth["date"].dt.strftime("%Y-%m-%d")
    fit_train["date"] = fit_train["date"].dt.strftime("%Y-%m-%d")

    print(f"Store Sales : {fit_train[['store_nbr', 'family']].drop_duplicates().shape[0]} entités, "
          f"backtest {horizon}j jusqu'au {truth['date'].max()}")
    _score(fit_train, ["store_nbr", "family"], horizon, truth, "Store Sales (historique long/homogène, 4 ans)")


def bench_recruit() -> None:
    visits = pd.read_csv(f"{RECRUIT_DIR}/air_visit_data.csv", parse_dates=["visit_date"])
    visits = visits.rename(columns={"visit_date": "date", "air_store_id": "store_nbr", "visitors": "sales"})
    visits["family"] = "visits"  # group_cols générique à 2 dimensions, comme l'app ; catégorie unique ici

    horizon = 39  # même longueur que l'horizon officiel Kaggle (2017-04-23 -> 2017-05-31)
    cutoff = visits["date"].max() - pd.Timedelta(days=horizon - 1)
    fit_train = visits[visits["date"] < cutoff].copy()
    truth = visits[visits["date"] >= cutoff][["store_nbr", "family", "date", "sales"]].copy()
    truth["date"] = truth["date"].dt.strftime("%Y-%m-%d")
    fit_train["date"] = fit_train["date"].dt.strftime("%Y-%m-%d")

    n_entities = fit_train["store_nbr"].nunique()
    hist_len = fit_train.groupby("store_nbr")["date"].count()
    print(f"Recruit : {n_entities} restaurants, backtest {horizon}j jusqu'au {truth['date'].max()} "
          f"(historique par restaurant : médiane {hist_len.median():.0f}j, min {hist_len.min()}j, max {hist_len.max()}j)")
    _score(fit_train, ["store_nbr", "family"], horizon, truth, "Recruit (historique court/hétérogène par restaurant)")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    if target in ("store_sales", "both"):
        bench_store_sales()
    if target in ("recruit", "both"):
        bench_recruit()
