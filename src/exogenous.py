"""Enrichissement du calendrier à partir d'une localisation (ville ou code
postal) : jours fériés, vacances scolaires, météo — variables exogènes
optionnelles pour la décomposition et la prévision du rythme.

Sources :
- Géocodage : API Adresse (api-adresse.data.gouv.fr)
- Vacances scolaires : data.education.gouv.fr (dataset fr-en-calendrier-scolaire)
- Jours fériés : bibliothèque `holidays` (France, avec Alsace-Moselle)
- Météo : Open-Meteo (archive historique + normales saisonnières pour le futur)
"""

from functools import lru_cache

import holidays
import pandas as pd
import requests

ALSACE_MOSELLE_DEPCODES = {"57", "67", "68"}

# Centroïdes des 30 académies (source : data.education.gouv.fr,
# dataset fr-en-contour-academies-2020), pour rattacher une position à
# l'académie la plus proche (approximation par plus proche centroïde).
ACADEMIE_CENTROIDS = {
    "Paris": (48.8566, 2.3428),
    "Créteil": (48.6440, 2.8994),
    "Versailles": (48.7816, 2.0509),
    "Lille": (50.4727, 2.7147),
    "Amiens": (49.6428, 2.8077),
    "Reims": (48.7364, 4.5414),
    "Strasbourg": (48.3303, 7.4352),
    "Nancy-Metz": (48.7624, 6.1418),
    "Dijon": (47.2478, 4.1494),
    "Besançon": (47.2079, 6.0881),
    "Orléans-Tours": (47.4848, 1.6844),
    "Rennes": (48.1776, -2.8428),
    "Nantes": (47.4774, -0.8173),
    "Poitiers": (46.1627, -0.0817),
    "Limoges": (45.7741, 1.7142),
    "Bordeaux": (44.3550, -0.2281),
    "Toulouse": (43.7686, 1.4874),
    "Montpellier": (43.5936, 3.2257),
    "Aix-Marseille": (44.1028, 5.7931),
    "Nice": (43.6492, 6.6090),
    "Corse": (42.1510, 9.1064),
    "Lyon": (45.9175, 4.7728),
    "Grenoble": (45.2202, 5.5967),
    "Clermont-Ferrand": (45.6562, 3.1771),
    "Normandie": (49.1043, -0.3062),
    "Guadeloupe": (16.1976, -61.5399),
    "Martinique": (14.6534, -61.0189),
    "Guyane": (3.9243, -53.2385),
}


def geocode(query: str) -> dict:
    """Géocode une ville ou un code postal via l'API Adresse (data.gouv.fr)."""
    resp = requests.get(
        "https://api-adresse.data.gouv.fr/search/",
        params={"q": query, "limit": 1},
        timeout=10,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise ValueError(f"Localisation introuvable : {query}")

    props = features[0]["properties"]
    lon, lat = features[0]["geometry"]["coordinates"]
    return {
        "label": props.get("label"),
        "lat": lat,
        "lon": lon,
        "depcode": props.get("depcode", ""),
    }


def nearest_academie(lat: float, lon: float) -> str:
    def dist2(coord):
        return (coord[0] - lat) ** 2 + (coord[1] - lon) ** 2

    return min(ACADEMIE_CENTROIDS, key=lambda name: dist2(ACADEMIE_CENTROIDS[name]))


@lru_cache(maxsize=64)
def _vacation_periods(academie: str) -> tuple:
    resp = requests.get(
        "https://data.education.gouv.fr/api/records/1.0/search/",
        params={"dataset": "fr-en-calendrier-scolaire", "refine.location": academie, "rows": 100},
        timeout=10,
    )
    resp.raise_for_status()
    periods = []
    for record in resp.json().get("records", []):
        fields = record["fields"]
        if fields.get("population") not in ("-", "Élèves"):
            continue  # exclut les dates spécifiques aux enseignants (été)
        start = pd.Timestamp(fields["start_date"]).tz_localize(None).normalize()
        end = pd.Timestamp(fields["end_date"]).tz_localize(None).normalize()
        periods.append((start, end))
    return tuple(periods)


def vacation_flags(academie: str, dates: pd.DatetimeIndex) -> pd.Series:
    """1 si le jour tombe dans une période de vacances scolaires de l'académie, 0 sinon."""
    periods = _vacation_periods(academie)
    flags = pd.Series(0, index=dates)
    for start, end in periods:
        flags.loc[(dates >= start) & (dates <= end)] = 1
    return flags


def holiday_flags(depcode: str, dates: pd.DatetimeIndex) -> pd.Series:
    """1 si le jour est férié en France (avec jours locaux Alsace-Moselle si pertinent)."""
    subdiv = "Alsace-Moselle" if depcode in ALSACE_MOSELLE_DEPCODES else None
    years = sorted(set(dates.year)) or [pd.Timestamp.today().year]
    fr_holidays = holidays.France(years=years, subdiv=subdiv)
    return pd.Series([1 if d.date() in fr_holidays else 0 for d in dates], index=dates)


def _open_meteo_archive(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "daily": "temperature_2m_max,precipitation_sum",
            "timezone": "Europe/Paris",
        },
        timeout=15,
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]
    return pd.DataFrame(
        {"temp_max": daily["temperature_2m_max"], "precipitation": daily["precipitation_sum"]},
        index=pd.to_datetime(daily["time"]),
    )


def _climatology(lat: float, lon: float, dates: pd.DatetimeIndex, reference_years: int = 3) -> pd.DataFrame:
    """Normale saisonnière (moyenne par jour de l'année sur N années passées),
    utilisée pour les dates futures où aucune observation réelle n'existe.
    """
    today = pd.Timestamp.today().normalize()
    ref_start = today - pd.DateOffset(years=reference_years)
    hist = _open_meteo_archive(lat, lon, ref_start, today - pd.Timedelta(days=1))
    avg_by_doy = hist.groupby(hist.index.dayofyear)[["temp_max", "precipitation"]].mean()

    doy = dates.dayofyear
    return pd.DataFrame(
        {
            "temp_max": avg_by_doy["temp_max"].reindex(doy).to_numpy(),
            "precipitation": avg_by_doy["precipitation"].reindex(doy).to_numpy(),
        },
        index=dates,
    )


def weather_series(lat: float, lon: float, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Température max / précipitations par jour : observations réelles pour le
    passé, normales saisonnières pour le futur (aucune prévision météo fiable
    au-delà de ~2 semaines).
    """
    today = pd.Timestamp.today().normalize()
    past = dates[dates < today]
    future = dates[dates >= today]

    frames = []
    if len(past):
        frames.append(_open_meteo_archive(lat, lon, past.min(), past.max()))
    if len(future):
        frames.append(_climatology(lat, lon, future))

    if not frames:
        return pd.DataFrame(columns=["temp_max", "precipitation"], index=dates)

    weather = pd.concat(frames)
    weather = weather[~weather.index.duplicated()].reindex(dates)
    return weather.interpolate().bfill().ffill()


def location_exog(loc: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Construit les variables exogènes (jours fériés, vacances scolaires,
    météo) pour une localisation géocodée et un ensemble de dates.
    """
    academie = nearest_academie(loc["lat"], loc["lon"])
    weather = weather_series(loc["lat"], loc["lon"], dates)
    return pd.DataFrame(
        {
            "ferie": holiday_flags(loc["depcode"], dates),
            "vacances": vacation_flags(academie, dates),
            "temp_max": weather["temp_max"].to_numpy(),
            "precipitation": weather["precipitation"].to_numpy(),
        },
        index=dates,
    )
