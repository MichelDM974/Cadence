"""Prévision du rythme pour tous les restaurants du dataset, avec évaluation
de l'erreur par point de vente (MAE / MAPE) sur un horizon de test commun.

Génère :
- data/output/forecast_all.csv     : une ligne par (restaurant, jour de test)
- data/output/forecast_summary.csv : une ligne par restaurant (MAE, MAPE, statut)
"""

import os
import time

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from decompose_prototype import DATA_DIR, OUTPUT_DIR, load_series

HORIZON = 28
MIN_TRAIN_DAYS = 90  # séries plus courtes que ça jugées trop courtes pour un fit fiable


def all_store_ids() -> list[str]:
    visits = pd.read_csv(f"{DATA_DIR}/air_visit_data.csv")
    return sorted(visits["air_store_id"].unique())


def fit_forecast(train: pd.DataFrame, test: pd.DataFrame):
    model = SARIMAX(
        train["visitors"],
        exog=train[["holiday_flg"]],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted.get_forecast(steps=len(test), exog=test[["holiday_flg"]])


def main() -> None:
    store_ids = all_store_ids()
    print(f"{len(store_ids)} restaurants à traiter")

    daily_rows = []
    summary_rows = []
    start = time.time()

    for i, store_id in enumerate(store_ids, 1):
        df = load_series(store_id)
        if len(df) < MIN_TRAIN_DAYS + HORIZON:
            summary_rows.append(
                {"air_store_id": store_id, "status": "trop_court", "n_jours": len(df), "mae": None, "mape": None}
            )
            continue

        train, test = df.iloc[:-HORIZON], df.iloc[-HORIZON:]
        try:
            forecast = fit_forecast(train, test)
        except Exception as exc:
            summary_rows.append(
                {"air_store_id": store_id, "status": f"erreur: {exc}", "n_jours": len(df), "mae": None, "mape": None}
            )
            continue

        mean = forecast.predicted_mean
        ci = forecast.conf_int(alpha=0.2)
        mae = (test["visitors"] - mean).abs().mean()
        mape = ((test["visitors"] - mean).abs() / test["visitors"].replace(0, float("nan"))).mean() * 100

        for date, reel, prevision, bas, haut in zip(
            test.index, test["visitors"], mean, ci.iloc[:, 0], ci.iloc[:, 1]
        ):
            daily_rows.append(
                {
                    "air_store_id": store_id,
                    "visit_date": date.date(),
                    "reel": reel,
                    "prevision": prevision,
                    "ic_bas": bas,
                    "ic_haut": haut,
                }
            )
        summary_rows.append(
            {"air_store_id": store_id, "status": "ok", "n_jours": len(df), "mae": mae, "mape": mape}
        )

        if i % 50 == 0:
            elapsed = time.time() - start
            print(f"{i}/{len(store_ids)} traités ({elapsed:.0f}s écoulées)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(daily_rows).to_csv(f"{OUTPUT_DIR}/forecast_all.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{OUTPUT_DIR}/forecast_summary.csv", index=False)

    ok = summary[summary["status"] == "ok"]
    print(f"Terminé en {time.time() - start:.0f}s")
    print(f"OK : {len(ok)} / {len(store_ids)} (échecs/trop courts : {len(store_ids) - len(ok)})")
    if len(ok):
        print(f"MAE médian : {ok['mae'].median():.1f}  MAPE médian : {ok['mape'].median():.1f}%")
    print(f"data/output/forecast_all.csv ({len(daily_rows)} lignes)")
    print(f"data/output/forecast_summary.csv ({len(summary)} lignes)")


if __name__ == "__main__":
    main()
