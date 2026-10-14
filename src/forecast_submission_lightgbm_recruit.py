"""Soumission Kaggle *Recruit Restaurant Visitor Forecasting* avec le
LightGBM de l'app (`lightgbm_forecast.py`), config "lag annuel + yoy_trend"
— la seule des 3 pistes candidates (cf. benchmark_lightgbm_variants.py,
centre-de-controle) qui améliore ce dataset (historique court/hétérogène par
restaurant, médiane 247j) : RMSLE local 0.72323 vs 0.73057 en lean.

Contrairement à forecast_submission_lightgbm_store_sales.py (réimplémentation
bespoke), ce script appelle directement `run_bulk_lightgbm_forecast`, le
point d'entrée réel de l'app — la config `feature_config` ci-dessous n'est
donc PAS le défaut de l'app (qui reste lean, cf. centre-de-controle : aucune
piste ne gagne sur les deux benchmarks à la fois), seulement celle utilisée
pour cette soumission ponctuelle.

Deux modes :
- `--validate` (défaut) : backtest local sur les 39 derniers jours connus
  d'air_visit_data.csv (même horizon que le test Kaggle réel).
- `--submit` : entraîne sur tout l'historique et écrit le CSV à uploader
  (horizon officiel 2017-04-23 -> 2017-05-31).
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from api import _daily_series, align_external_exog, group_combinations
from lightgbm_forecast import run_bulk_lightgbm_forecast

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
HORIZON = 39  # 2017-04-23 -> 2017-05-31, horizon officiel Kaggle

FEATURE_CONFIG = {"rolling_windows": (7, 28), "include_yoy": True}


def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.clip(actual, 0, None)
    predicted = np.clip(predicted, 0, None)
    return float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)))


def _combos_and_indices(raw: pd.DataFrame, group_cols: list[str]):
    # groupby sur une liste à un seul élément renvoie des clés scalaires, pas
    # des tuples à 1 élément comme group_combinations() — même distinction
    # que _run_bulk_forecast (api.py) pour un seul group_col.
    combos = group_combinations(raw, group_cols)
    str_cols = {c: raw[c].astype(str) for c in group_cols}
    if len(group_cols) == 1:
        group_indices = {(key,): idx for key, idx in raw.groupby(str_cols[group_cols[0]]).groups.items()}
    else:
        group_indices = dict(raw.groupby([str_cols[c] for c in group_cols]).groups.items())
    return combos, group_indices


def _run(raw: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, set]:
    group_cols = ["air_store_id"]
    combos, group_indices = _combos_and_indices(raw, group_cols)
    rows, skipped = run_bulk_lightgbm_forecast(
        raw, "visit_date", "visitors", group_cols, horizon, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "air_store_id", None, None, None,
        _daily_series, align_external_exog, None,
        feature_config=FEATURE_CONFIG,
    )
    return pd.DataFrame(rows), skipped


def run_validate(visits: pd.DataFrame) -> None:
    cutoff = visits["visit_date"].max() - pd.Timedelta(days=HORIZON - 1)
    fit_train = visits[visits["visit_date"] < cutoff].copy()
    truth = visits[visits["visit_date"] >= cutoff][["air_store_id", "visit_date", "visitors"]].copy()

    print(f"backtest : entraînement jusqu'au {fit_train['visit_date'].max().date()}, "
          f"validation {truth['visit_date'].min().date()} -> {truth['visit_date'].max().date()} ({HORIZON} jours)")

    fit_train["visit_date"] = fit_train["visit_date"].dt.strftime("%Y-%m-%d")
    start = time.time()
    preds, skipped = _run(fit_train, HORIZON)
    print(f"terminé en {time.time() - start:.0f}s ({len(skipped)} entités ignorées, historique < 3 semaines)")

    preds = preds.rename(columns={"date": "visit_date"})
    preds["visit_date"] = pd.to_datetime(preds["visit_date"])
    merged = truth.merge(preds, on=["air_store_id", "visit_date"], how="left")
    merged["prevision"] = merged["prevision"].fillna(0)
    score = rmsle(merged["visitors"].to_numpy(), merged["prevision"].to_numpy())
    print(f"RMSLE local ({HORIZON} derniers jours connus) : {score:.5f}  (lean actuel de l'app : 0.73057)")


def run_submit(visits: pd.DataFrame) -> None:
    raw = visits.copy()
    raw["visit_date"] = raw["visit_date"].dt.strftime("%Y-%m-%d")

    start = time.time()
    preds, skipped = _run(raw, HORIZON)
    print(f"terminé en {time.time() - start:.0f}s ({len(skipped)} entités ignorées, historique < 3 semaines)")

    sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    sub["air_store_id"] = sub["id"].str.slice(0, 20)
    sub["visit_date"] = sub["id"].str.slice(21)

    result = sub[["id", "air_store_id", "visit_date"]].merge(
        preds.rename(columns={"date": "visit_date"}), on=["air_store_id", "visit_date"], how="left"
    )
    result["prevision"] = result["prevision"].fillna(0)
    result = result[["id"]].assign(visitors=result["prevision"].round(2))

    out_path = f"{DATA_DIR}/recruit_submission_lightgbm.csv"
    result.to_csv(out_path, index=False)
    print(f"{out_path} ({len(result)} lignes) — à uploader sur Kaggle pour le score officiel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="entraîne sur tout l'historique et écrit le CSV Kaggle")
    args = parser.parse_args()

    visits = pd.read_csv(f"{DATA_DIR}/air_visit_data.csv", parse_dates=["visit_date"])
    if args.submit:
        run_submit(visits)
    else:
        run_validate(visits)


if __name__ == "__main__":
    main()
