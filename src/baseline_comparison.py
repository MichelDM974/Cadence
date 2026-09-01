"""Compare SARIMAX à une baseline naïve saisonnière ("même jour la semaine
dernière", répété sur tout l'horizon) sur l'ensemble des restaurants du
dataset de test, pour vérifier que le modèle apporte une vraie valeur
ajoutée avant de l'afficher dans l'application.
"""

import signal
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from decompose_prototype import DATA_DIR, OUTPUT_DIR, load_series, seasonal_naive_forecast

HORIZON = 28
SEASON = 7
MIN_TRAIN_DAYS = 90
FIT_TIMEOUT_S = 20


class FitTimeout(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    def _handler(signum, frame):
        raise FitTimeout(f"fit > {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def all_store_ids() -> list[str]:
    visits = pd.read_csv(f"{DATA_DIR}/air_visit_data.csv")
    return sorted(visits["air_store_id"].unique())


def sarimax_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
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
    return forecast.predicted_mean.clip(lower=0).to_numpy()


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(pred - actual).mean())


def mape(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = actual != 0
    return float((np.abs(pred[mask] - actual[mask]) / actual[mask]).mean() * 100)


def main() -> None:
    store_ids = all_store_ids()
    print(f"{len(store_ids)} restaurants à traiter")

    rows = []
    start = time.time()

    for i, store_id in enumerate(store_ids, 1):
        df = load_series(store_id)
        if len(df) < MIN_TRAIN_DAYS + HORIZON:
            continue

        train, test = df.iloc[:-HORIZON], df.iloc[-HORIZON:]
        actual = test["visitors"].to_numpy()

        naive_pred = seasonal_naive_forecast(train["visitors"], HORIZON)

        try:
            with timeout(FIT_TIMEOUT_S):
                sarimax_pred = sarimax_forecast(train, test)
        except Exception:
            continue

        rows.append(
            {
                "air_store_id": store_id,
                "mae_naive": mae(naive_pred, actual),
                "mae_sarimax": mae(sarimax_pred, actual),
                "mape_naive": mape(naive_pred, actual),
                "mape_sarimax": mape(sarimax_pred, actual),
            }
        )

        if i % 50 == 0:
            print(f"{i}/{len(store_ids)} traités ({time.time() - start:.0f}s écoulées)", flush=True)

    result = pd.DataFrame(rows)
    result["sarimax_gagne_mae"] = result["mae_sarimax"] < result["mae_naive"]
    result["amelioration_mae_pct"] = (
        (result["mae_naive"] - result["mae_sarimax"]) / result["mae_naive"] * 100
    )

    result.to_csv(f"{OUTPUT_DIR}/baseline_comparison.csv", index=False)

    print(f"\nTerminé en {time.time() - start:.0f}s sur {len(result)} restaurants")
    print(f"MAE médian  — naïve: {result['mae_naive'].median():.1f}  | SARIMAX: {result['mae_sarimax'].median():.1f}")
    print(f"MAPE médian — naïve: {result['mape_naive'].median():.1f}% | SARIMAX: {result['mape_sarimax'].median():.1f}%")
    print(f"SARIMAX bat la baseline (MAE) sur {result['sarimax_gagne_mae'].mean() * 100:.0f}% des restaurants")
    print(f"Amélioration MAE médiane : {result['amelioration_mae_pct'].median():+.1f}%")


if __name__ == "__main__":
    main()
