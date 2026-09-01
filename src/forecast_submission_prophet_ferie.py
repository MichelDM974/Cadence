"""Variante Prophet de forecast_submission_sarimax_ferie.py : même soumission Kaggle
('Recruit Restaurant Visitor Forecasting'), mais le modèle par restaurant
est Prophet (avec le jour férié en régresseur) au lieu de SARIMAX — pour
comparer les deux sur le leaderboard réel plutôt que sur un backtest local.
"""

import logging
import os

import pandas as pd
from prophet import Prophet

from decompose_prototype import OUTPUT_DIR, load_series
from forecast_submission_sarimax_ferie import MIN_TRAIN_DAYS, future_holiday_flags, run_parallel, submission_targets, timeout

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

FIT_TIMEOUT_S = 30  # Prophet est un peu plus lent à ajuster que SARIMAX


def fit_forecast(train: pd.DataFrame, exog_future: pd.DataFrame) -> pd.DataFrame:
    train_df = pd.DataFrame(
        {"ds": train.index, "y": train["visitors"].to_numpy(), "ferie": train["holiday_flg"].to_numpy()}
    )
    model = Prophet(interval_width=0.8)
    model.add_regressor("ferie")
    model.fit(train_df)

    future_df = pd.DataFrame({"ds": exog_future.index, "ferie": exog_future["holiday_flg"].to_numpy()})
    return model.predict(future_df)


def forecast_one_store(store_id: str, horizon_end: pd.Timestamp) -> tuple[str, pd.Series, bool]:
    train = load_series(store_id)

    # Certains restaurants ont un historique qui s'arrête avant le
    # 22/04/2017 (jusqu'à 22 jours d'écart) : on prévoit jusqu'à la fin de
    # l'horizon officiel pour chacun, puis on ne garde que les jours de
    # l'horizon — sinon les jours fériés en exogène seraient décalés.
    full_horizon = pd.date_range(train.index.max() + pd.Timedelta(days=1), horizon_end, freq="D")

    if len(train) < MIN_TRAIN_DAYS:
        return store_id, pd.Series(train["visitors"].mean(), index=full_horizon), True

    try:
        exog_full = future_holiday_flags(full_horizon)
        with timeout(FIT_TIMEOUT_S):
            forecast = fit_forecast(train, exog_full)
        pred = pd.Series(forecast["yhat"].to_numpy(), index=full_horizon)
        return store_id, pred, False
    except Exception as exc:
        print(f"  {store_id} : fallback moyenne ({exc})", flush=True)
        return store_id, pd.Series(train["visitors"].mean(), index=full_horizon), True


def main() -> None:
    sub = submission_targets()
    store_ids = sorted(sub["air_store_id"].unique())
    horizon_dates = pd.DatetimeIndex(sorted(sub["visit_date"].unique()))

    print(
        f"{len(store_ids)} restaurants, horizon {horizon_dates.min().date()} -> "
        f"{horizon_dates.max().date()} ({len(horizon_dates)} jours) — modèle Prophet"
    )

    rows, n_fallback = run_parallel(store_ids, horizon_dates, forecast_one_store)

    result = pd.DataFrame(rows)
    # aligne exactement sur les lignes attendues par sample_submission.csv
    result = sub[["id"]].merge(result, on="id", how="left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/forecast_submission_prophet_ferie.csv"
    result.to_csv(out_path, index=False)

    print(f"Prévisions de secours (historique trop court/échec fit) : {n_fallback}/{len(store_ids)}")
    print(f"{out_path} ({len(result)} lignes)")


if __name__ == "__main__":
    main()
