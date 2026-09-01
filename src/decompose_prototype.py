"""Premier prototype : décomposition et indice de régularité du rythme
d'un restaurant, à partir du jeu de données Kaggle Recruit Restaurant
Visitor Forecasting (data/raw/air_visit_data.csv, data/raw/date_info.csv).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(ROOT / "data" / "raw")
OUTPUT_DIR = str(ROOT / "data" / "output")


@lru_cache(maxsize=1)
def load_date_info() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}/date_info.csv", parse_dates=["calendar_date"]).set_index("calendar_date")


@lru_cache(maxsize=1)
def _visits_by_store() -> dict[str, pd.Series]:
    """Groupe air_visit_data.csv par restaurant une seule fois par process
    (au lieu de relire + refiltrer tout le fichier à chaque appel de
    load_series) — y compris dans chaque worker d'un pool de processus, où
    ce cache se construit à la première tâche traitée par ce worker et sert
    pour toutes les suivantes."""
    visits = pd.read_csv(f"{DATA_DIR}/air_visit_data.csv", parse_dates=["visit_date"])
    return {
        store_id: group.set_index("visit_date")["visitors"]
        for store_id, group in visits.groupby("air_store_id")
    }


def load_series(store_id: str) -> pd.DataFrame:
    store = _visits_by_store()[store_id]
    full_range = pd.date_range(store.index.min(), store.index.max(), freq="D")
    store = store.reindex(full_range)
    store.index.name = "visit_date"

    # Jours absents du fichier = restaurant fermé -> 0 couvert.
    store = store.fillna(0)

    df = store.to_frame("visitors")
    df = df.join(load_date_info()[["holiday_flg"]])
    return df


def pick_most_complete_store() -> str:
    visits = pd.read_csv(f"{DATA_DIR}/air_visit_data.csv")
    counts = visits.groupby("air_store_id").size()
    return counts.idxmax()


def regularity_index(resid: pd.Series, observed: pd.Series) -> float:
    """Score de prévisibilité : 1 - (dispersion résiduelle / dispersion observée).
    Proche de 1 = rythme très régulier, proche de 0 = rythme erratique.
    """
    return max(0.0, 1 - resid.std() / observed.std())


def detect_anomalies(resid: pd.Series, n_std: float = 2.5) -> pd.Series:
    threshold = n_std * resid.std()
    return resid[resid.abs() > threshold]


def seasonal_naive_forecast(train: pd.Series, horizon: int, season: int = 7):
    """Baseline "même jour la semaine dernière" : répète le dernier cycle
    hebdomadaire connu sur tout l'horizon de prévision, sans utiliser de
    modèle — sert de référence pour juger si SARIMAX apporte une vraie
    valeur ajoutée.
    """
    last_season = train.iloc[-season:].to_numpy()
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]


def event_components(resid: pd.Series, flags: dict) -> tuple[pd.Series, dict]:
    """Isole, pour chaque calendrier binaire fourni (jour férié, vacances
    scolaires...), l'effet moyen qu'il produit dans le résidu STL, pour en
    faire une composante événementielle à part entière (cf. tendance/
    saisonnier/événementiel/résidu du README) plutôt que de la laisser
    polluer le résidu. Retourne la somme des effets et le détail par calendrier.
    """
    combined = pd.Series(0.0, index=resid.index)
    effects = {}
    for name, flag in flags.items():
        if flag.sum() == 0:
            effects[name] = 0.0
            continue
        effect = resid[flag == 1].mean()
        combined = combined + flag * effect
        effects[name] = effect
    return combined, effects


def main() -> None:
    store_id = pick_most_complete_store()
    df = load_series(store_id)
    print(f"Point de vente sélectionné : {store_id}")
    print(f"Période : {df.index.min().date()} -> {df.index.max().date()} ({len(df)} jours)")

    stl = STL(df["visitors"], period=7, robust=True)
    result = stl.fit()

    event, effects = event_components(result.resid, {"ferie": df["holiday_flg"]})
    adjusted_resid = result.resid - event
    print(f"Effet moyen d'un jour férié sur le rythme : {effects['ferie']:+.1f} couverts")

    idx = regularity_index(adjusted_resid, df["visitors"])
    print(f"Indice de régularité du rythme : {idx:.2f}")

    anomalies = detect_anomalies(adjusted_resid)
    print(f"Ruptures de rythme détectées (hors effet jours fériés) : {len(anomalies)}")
    if len(anomalies):
        top = anomalies.sort_values(key=abs, ascending=False).head(10).to_frame("resid")
        top["holiday"] = df.loc[top.index, "holiday_flg"]
        print(top)

    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(df.index, df["visitors"])
    axes[0].set_ylabel("Observé")
    axes[1].plot(result.trend)
    axes[1].set_ylabel("Tendance")
    axes[2].plot(result.seasonal)
    axes[2].set_ylabel("Saisonnier (hebdo)")
    axes[3].plot(event)
    axes[3].set_ylabel("Événementiel\n(férié)")
    axes[4].plot(adjusted_resid)
    axes[4].set_ylabel("Résidu")
    fig.suptitle(f"Décomposition du rythme — {store_id}")
    fig.tight_layout()

    out_path = f"{OUTPUT_DIR}/decomposition_{store_id}.png"
    fig.savefig(out_path)
    print(f"Graphique sauvegardé : {out_path}")


if __name__ == "__main__":
    main()
