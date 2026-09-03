"""Génère une prévision au format 'soumission Kaggle' : un couvert prévu par
(restaurant, jour), sur l'horizon officiel de test de la compétition
(2017-04-23 -> 2017-05-31), pour les 821 restaurants concernés.

Contrairement à une évaluation classique (voir baseline_comparison.py), il
n'y a pas de valeur réelle à comparer ici : Kaggle garde les réponses
secrètes. Le modèle est donc entraîné sur tout l'historique disponible par
restaurant, sans découpage train/test.

Chaque restaurant est indépendant des autres : le fit est distribué sur un
pool de processus (un par cœur) plutôt que traité séquentiellement.
"""

import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Callable

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from decompose_prototype import DATA_DIR, OUTPUT_DIR, load_date_info, load_series

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
    return load_date_info().loc[dates, ["holiday_flg"]]


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


def forecast_one_store(store_id: str, horizon_end: pd.Timestamp) -> tuple[str, pd.Series, bool]:
    """Prévision d'un seul restaurant — tourne dans un worker du pool."""
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
        pred = pd.Series(forecast.predicted_mean.to_numpy(), index=full_horizon)
        return store_id, pred, False
    except Exception as exc:
        print(f"  {store_id} : fallback moyenne ({exc})", flush=True)
        return store_id, pd.Series(train["visitors"].mean(), index=full_horizon), True


def run_parallel(
    store_ids: list[str],
    horizon_dates: pd.DatetimeIndex,
    forecast_one: Callable[..., tuple[str, pd.Series, bool]],
    extra_args: Callable[[str], tuple] = lambda store_id: (),
    progress_every: int = 10,
) -> tuple[list[dict], int]:
    """Distribue `forecast_one` sur un pool de processus (un restaurant =
    une tâche indépendante) et assemble les lignes de résultat. `forecast_one`
    doit être une fonction de module (picklable), pas une closure. Toute
    donnée dont un worker a besoin doit être passée explicitement via
    `extra_args` (construite ici, dans le process principal) : le pool
    démarre par défaut en mode "forkserver" sur cette version de Python, les
    workers réimportent le module à froid et n'héritent donc pas des
    variables globales mutées après l'import (contrairement à un "fork").
    """
    rows: list[dict] = []
    n_fallback = 0
    n_done = 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {
            executor.submit(forecast_one, store_id, horizon_dates.max(), *extra_args(store_id)): store_id
            for store_id in store_ids
        }

        for future in as_completed(futures):
            store_id, pred, was_fallback = future.result()
            n_fallback += was_fallback

            pred = pred.reindex(horizon_dates).clip(lower=0)
            for date, p in zip(horizon_dates, pred.to_numpy()):
                rows.append({"id": f"{store_id}_{date.date()}", "visitors": round(float(p), 1)})

            n_done += 1
            if n_done % progress_every == 0:
                print(f"{n_done}/{len(store_ids)} traités ({time.time() - start:.0f}s écoulées)", flush=True)

    print(f"Terminé en {time.time() - start:.0f}s")
    return rows, n_fallback


def main() -> None:
    sub = submission_targets()
    store_ids = sorted(sub["air_store_id"].unique())
    horizon_dates = pd.DatetimeIndex(sorted(sub["visit_date"].unique()))

    print(
        f"{len(store_ids)} restaurants, horizon {horizon_dates.min().date()} -> "
        f"{horizon_dates.max().date()} ({len(horizon_dates)} jours)"
    )

    rows, n_fallback = run_parallel(store_ids, horizon_dates, forecast_one_store)

    result = pd.DataFrame(rows)
    # aligne exactement sur les lignes attendues par sample_submission.csv
    result = sub[["id"]].merge(result, on="id", how="left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/forecast_submission_sarimax_ferie.csv"
    result.to_csv(out_path, index=False)

    print(f"Prévisions de secours (historique trop court/échec fit) : {n_fallback}/{len(store_ids)}")
    print(f"{out_path} ({len(result)} lignes)")


if __name__ == "__main__":
    main()
