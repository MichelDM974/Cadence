"""Variante SARIMAX + météo de forecast_submission_sarimax_ferie.py : mêmes prévisions
Kaggle ('Recruit Restaurant Visitor Forecasting'), mais avec la météo réelle
du Japon (Open-Meteo, à partir des coordonnées GPS par restaurant dans
air_store_info.csv) ajoutée en exogène, en plus du jour férié déjà présent
dans date_info.csv.

Toutes les dates du dataset sont passées (2016-2017) : la météo utilisée est
donc de la vraie météo observée, pas une approximation par normales
saisonnières. La météo est mise en cache par (latitude, longitude) — 829
restaurants ne représentent que 108 lieux uniques.
"""

import os
import time

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import exogenous
from decompose_prototype import DATA_DIR, OUTPUT_DIR, load_series
from forecast_submission_sarimax_ferie import FitTimeout, MIN_TRAIN_DAYS, future_holiday_flags, submission_targets, timeout

FIT_TIMEOUT_S = 25


def store_locations() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}/air_store_info.csv").set_index("air_store_id")[["latitude", "longitude"]]


def build_weather_cache(locations: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    unique_coords = locations[["latitude", "longitude"]].drop_duplicates()
    cache = {}
    print(f"Récupération météo pour {len(unique_coords)} lieux uniques...")
    t0 = time.time()
    for i, (lat, lon) in enumerate(unique_coords.itertuples(index=False), 1):
        cache[(lat, lon)] = exogenous.weather_series(lat, lon, dates)
        if i % 20 == 0:
            print(f"  {i}/{len(unique_coords)} lieux ({time.time() - t0:.0f}s écoulées)", flush=True)
    print(f"Météo récupérée en {time.time() - t0:.0f}s")
    return cache


def fit_forecast(train: pd.DataFrame, exog_future: pd.DataFrame):
    model = SARIMAX(
        train["visitors"],
        exog=train[["holiday_flg", "temp_max", "precipitation"]],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted.get_forecast(steps=len(exog_future), exog=exog_future[["holiday_flg", "temp_max", "precipitation"]])


def main() -> None:
    sub = submission_targets()
    store_ids = sorted(sub["air_store_id"].unique())
    horizon_dates = pd.DatetimeIndex(sorted(sub["visit_date"].unique()))
    locations = store_locations()

    full_range = pd.date_range("2016-01-01", horizon_dates.max(), freq="D")
    weather_cache = build_weather_cache(locations.loc[store_ids], full_range)

    print(
        f"{len(store_ids)} restaurants, horizon {horizon_dates.min().date()} -> "
        f"{horizon_dates.max().date()} ({len(horizon_dates)} jours) — SARIMAX + jour férié + météo"
    )

    rows = []
    start = time.time()
    n_fallback = 0

    for i, store_id in enumerate(store_ids, 1):
        train = load_series(store_id)
        lat, lon = locations.loc[store_id, "latitude"], locations.loc[store_id, "longitude"]
        weather = weather_cache[(lat, lon)]

        full_horizon = pd.date_range(
            train.index.max() + pd.Timedelta(days=1), horizon_dates.max(), freq="D"
        )

        if len(train) < MIN_TRAIN_DAYS:
            pred = pd.Series(train["visitors"].mean(), index=full_horizon)
            n_fallback += 1
        else:
            try:
                holiday_full = future_holiday_flags(full_horizon)
                exog_future = holiday_full.join(weather.reindex(full_horizon))
                train_exog = train.join(weather.reindex(train.index))
                with timeout(FIT_TIMEOUT_S):
                    forecast = fit_forecast(train_exog, exog_future)
                pred = pd.Series(forecast.predicted_mean.to_numpy(), index=full_horizon)
            except Exception as exc:
                print(f"  {store_id} : fallback moyenne ({exc})", flush=True)
                pred = pd.Series(train["visitors"].mean(), index=full_horizon)
                n_fallback += 1

        pred = pred.reindex(horizon_dates).clip(lower=0)

        for date, p in zip(horizon_dates, pred.to_numpy()):
            rows.append({"id": f"{store_id}_{date.date()}", "visitors": round(float(p), 1)})

        if i % 10 == 0:
            print(f"{i}/{len(store_ids)} traités ({time.time() - start:.0f}s écoulées)", flush=True)

    result = pd.DataFrame(rows)
    result = sub[["id"]].merge(result, on="id", how="left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/forecast_submission_sarimax_ferie_meteo.csv"
    result.to_csv(out_path, index=False)

    print(f"Terminé en {time.time() - start:.0f}s")
    print(f"Prévisions de secours (historique trop court/échec fit) : {n_fallback}/{len(store_ids)}")
    print(f"{out_path} ({len(result)} lignes)")


if __name__ == "__main__":
    main()
