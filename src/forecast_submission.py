"""Génère une prévision au format 'soumission Kaggle' : un couvert prévu par
(restaurant, jour), sur l'horizon officiel de test de la compétition
(2017-04-23 -> 2017-05-31), pour les 821 restaurants concernés.

Contrairement au prototype d'évaluation (forecast_prototype.py /
forecast_batch.py), il n'y a pas de valeur réelle à comparer ici : Kaggle
garde les réponses secrètes. Le modèle est donc entraîné sur tout
l'historique disponible par restaurant, sans découpage train/test.
"""

import os
import signal
import time
from contextlib import contextmanager

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from decompose_prototype import DATA_DIR, OUTPUT_DIR, load_series

MIN_TRAIN_DAYS = 90  # en dessous, prévision de secours = moyenne historique
FIT_TIMEOUT_S = 20  # au-delà, on abandonne ce restaurant (série difficile à optimiser)


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


def submission_targets() -> pd.DataFrame:
    sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    sub["air_store_id"] = sub["id"].str.slice(0, 20)
    sub["visit_date"] = pd.to_datetime(sub["id"].str.slice(21))
    return sub


def future_holiday_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    date_info = pd.read_csv(f"{DATA_DIR}/date_info.csv", parse_dates=["calendar_date"])
    return date_info.set_index("calendar_date").loc[dates, ["holiday_flg"]]


def fit_forecast(train: pd.DataFrame, exog_future: pd.DataFrame):
    model = SARIMAX(
        train["visitors"],
        exog=train[["holiday_flg"]],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted.get_forecast(steps=len(exog_future), exog=exog_future)


def main() -> None:
    sub = submission_targets()
    store_ids = sorted(sub["air_store_id"].unique())
    horizon_dates = pd.DatetimeIndex(sorted(sub["visit_date"].unique()))

    print(
        f"{len(store_ids)} restaurants, horizon {horizon_dates.min().date()} -> "
        f"{horizon_dates.max().date()} ({len(horizon_dates)} jours)"
    )

    rows = []
    start = time.time()
    n_fallback = 0

    for i, store_id in enumerate(store_ids, 1):
        train = load_series(store_id)

        # Certains restaurants ont un historique qui s'arrête avant le
        # 22/04/2017 (jusqu'à 22 jours d'écart) : on prévoit jusqu'à la fin
        # de l'horizon officiel pour chacun, puis on ne garde que les jours
        # de l'horizon — sinon les jours fériés en exogène seraient décalés.
        full_horizon = pd.date_range(
            train.index.max() + pd.Timedelta(days=1), horizon_dates.max(), freq="D"
        )

        if len(train) < MIN_TRAIN_DAYS:
            pred = pd.Series(train["visitors"].mean(), index=full_horizon)
            lo = hi = pred
            n_fallback += 1
        else:
            try:
                exog_full = future_holiday_flags(full_horizon)
                with timeout(FIT_TIMEOUT_S):
                    forecast = fit_forecast(train, exog_full)
                pred = pd.Series(forecast.predicted_mean.to_numpy(), index=full_horizon)
                ci = forecast.conf_int(alpha=0.2)
                lo = pd.Series(ci.iloc[:, 0].to_numpy(), index=full_horizon)
                hi = pd.Series(ci.iloc[:, 1].to_numpy(), index=full_horizon)
            except Exception as exc:
                print(f"  {store_id} : fallback moyenne ({exc})", flush=True)
                pred = pd.Series(train["visitors"].mean(), index=full_horizon)
                lo = hi = pred
                n_fallback += 1

        pred = pred.reindex(horizon_dates).clip(lower=0)
        lo = lo.reindex(horizon_dates).clip(lower=0)
        hi = hi.reindex(horizon_dates).clip(lower=0)

        for date, p, l, h in zip(horizon_dates, pred.to_numpy(), lo.to_numpy(), hi.to_numpy()):
            rows.append(
                {
                    "id": f"{store_id}_{date.date()}",
                    "visitors": round(float(p), 1),
                    "ic_bas": round(float(l), 1),
                    "ic_haut": round(float(h), 1),
                }
            )

        if i % 10 == 0:
            print(f"{i}/{len(store_ids)} traités ({time.time() - start:.0f}s écoulées)", flush=True)

    result = pd.DataFrame(rows)
    # aligne exactement sur les lignes attendues par sample_submission.csv
    result = sub[["id"]].merge(result, on="id", how="left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/forecast_submission.csv"
    result.to_csv(out_path, index=False)

    print(f"Terminé en {time.time() - start:.0f}s")
    print(f"Prévisions de secours (historique trop court/échec fit) : {n_fallback}/{len(store_ids)}")
    print(f"{out_path} ({len(result)} lignes)")


if __name__ == "__main__":
    main()
