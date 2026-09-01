"""Prototype de prévision du rythme (couverts) avec intervalle de confiance.

SARIMAX : tendance + saisonnalité hebdomadaire + jours fériés en variable
exogène (contrairement à la météo, le calendrier est connu à l'avance,
ce qui en fait un régresseur exploitable directement en prévision).
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from decompose_prototype import OUTPUT_DIR, load_series, pick_most_complete_store

HORIZON = 28  # jours mis de côté pour évaluer la prévision


def train_test_split(df, horizon: int = HORIZON):
    return df.iloc[:-horizon], df.iloc[-horizon:]


def fit_forecast(train, test):
    model = SARIMAX(
        train["visitors"],
        exog=train[["holiday_flg"]],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=len(test), exog=test[["holiday_flg"]])
    return fitted, forecast


def main() -> None:
    store_id = pick_most_complete_store()
    df = load_series(store_id)
    train, test = train_test_split(df)

    print(f"Point de vente : {store_id}")
    print(f"Train : {train.index.min().date()} -> {train.index.max().date()} ({len(train)} jours)")
    print(f"Test  : {test.index.min().date()} -> {test.index.max().date()} ({len(test)} jours)")

    _, forecast = fit_forecast(train, test)
    mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.2)  # intervalle de confiance à 80 %

    mae = (test["visitors"] - mean).abs().mean()
    mape = ((test["visitors"] - mean).abs() / test["visitors"].replace(0, float("nan"))).mean() * 100
    print(f"MAE  : {mae:.1f} couverts")
    print(f"MAPE : {mape:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index[-60:], train["visitors"].iloc[-60:], label="Historique")
    ax.plot(test.index, test["visitors"], label="Réel", color="black")
    ax.plot(test.index, mean, label="Prévision", color="tab:orange")
    ax.fill_between(
        test.index, ci.iloc[:, 0], ci.iloc[:, 1], color="tab:orange", alpha=0.2, label="Intervalle de confiance 80%"
    )
    ax.set_title(f"Prévision du rythme — {store_id}")
    ax.legend()
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/forecast_{store_id}.png"
    fig.savefig(out_path)
    print(f"Graphique sauvegardé : {out_path}")

    csv = pd.DataFrame(
        {
            "visit_date": test.index,
            "reel": test["visitors"].values,
            "prevision": mean.values,
            "ic_bas": ci.iloc[:, 0].values,
            "ic_haut": ci.iloc[:, 1].values,
        }
    )
    csv_path = f"{OUTPUT_DIR}/forecast_{store_id}.csv"
    csv.to_csv(csv_path, index=False)
    print(f"CSV sauvegardé : {csv_path}")


if __name__ == "__main__":
    main()
