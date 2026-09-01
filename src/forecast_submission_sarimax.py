"""Variante SARIMAX sans aucune variable exogène de forecast_submission_sarimax_ferie.py :
même soumission Kaggle ('Recruit Restaurant Visitor Forecasting'), mais sans
jour férié ni météo — juste tendance + saisonnalité hebdomadaire. Sert de
point de comparaison pour juger l'apport réel des exogènes (jour férié seul,
puis jour férié + météo).
"""

import os

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from decompose_prototype import OUTPUT_DIR, load_series
from forecast_submission_sarimax_ferie import MIN_TRAIN_DAYS, run_parallel, submission_targets, timeout

FIT_TIMEOUT_S = 20


def fit_forecast(train: pd.DataFrame, horizon: int):
    model = SARIMAX(
        train["visitors"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted.get_forecast(steps=horizon)


def forecast_one_store(store_id: str, horizon_end: pd.Timestamp) -> tuple[str, pd.Series, bool]:
    train = load_series(store_id)
    full_horizon = pd.date_range(train.index.max() + pd.Timedelta(days=1), horizon_end, freq="D")

    if len(train) < MIN_TRAIN_DAYS:
        return store_id, pd.Series(train["visitors"].mean(), index=full_horizon), True

    try:
        with timeout(FIT_TIMEOUT_S):
            forecast = fit_forecast(train, len(full_horizon))
        pred = pd.Series(forecast.predicted_mean.to_numpy(), index=full_horizon)
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
        f"{horizon_dates.max().date()} ({len(horizon_dates)} jours) — SARIMAX sans exogène"
    )

    rows, n_fallback = run_parallel(store_ids, horizon_dates, forecast_one_store, progress_every=50)

    result = pd.DataFrame(rows)
    result = sub[["id"]].merge(result, on="id", how="left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/forecast_submission_sarimax.csv"
    result.to_csv(out_path, index=False)

    print(f"Prévisions de secours (historique trop court/échec fit) : {n_fallback}/{len(store_ids)}")
    print(f"{out_path} ({len(result)} lignes)")


if __name__ == "__main__":
    main()
