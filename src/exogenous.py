"""Enrichissement du calendrier à partir d'une localisation (ville ou code
postal, n'importe où dans le monde) : jours fériés, vacances scolaires,
météo — variables exogènes optionnelles pour la décomposition et la
prévision du rythme.

Sources :
- Géocodage : API de géocodage Open-Meteo (mondiale)
- Jours fériés : bibliothèque `holidays` (tous pays supportés, avec Alsace-Moselle pour la France)
- Vacances scolaires : data.education.gouv.fr (France uniquement — aucune source équivalente
  intégrée pour les autres pays, la variable vaut 0 ailleurs)
- Météo : Open-Meteo (archive historique + normales saisonnières pour le futur, mondiale)
"""

import time
from functools import lru_cache

import holidays
import pandas as pd
import requests

ALSACE_MOSELLE_DEPCODES = {"57", "67", "68"}

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _get_json(url: str, params: dict, timeout: int = 15, tentatives: int = 3) -> requests.Response:
    """GET avec quelques essais espacés sur incident passager (réseau coupé,
    429 de quota, 5xx). Ces API sont interrogées en boucle sur beaucoup de
    lieux — le banc d'essai météo en enchaîne 108 — et un seul incident au
    milieu faisait tout échouer en perdant le cache déjà constitué.
    """
    for essai in range(tentatives):
        dernier = essai == tentatives - 1
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUS and not dernier:
                time.sleep(2**essai)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if dernier:
                raise
            time.sleep(2**essai)

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
    # Manquaient jusqu'ici alors que le compte annoncé était de 30 : un point
    # de vente à Saint-Denis se voyait rattacher à l'académie la plus proche
    # parmi les autres, soit la Corse, et recevait donc un calendrier
    # d'hémisphère nord sans rapport avec le sien.
    "Réunion": (-21.1151, 55.5364),
    "Mayotte": (-12.8275, 45.1662),
}


def geocode(query: str) -> dict:
    """Géocode une ville ou un code postal, n'importe où dans le monde, via
    l'API de géocodage Open-Meteo.
    """
    resp = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": query, "count": 1, "language": "fr", "format": "json"},
        timeout=10,
    )
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"Localisation introuvable : {query}")

    place = results[0]
    label = ", ".join(part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part)
    country_code = place.get("country_code", "")
    postcodes = place.get("postcodes") or []
    # Code département français (2 premiers chiffres du code postal) : sert
    # uniquement à détecter l'Alsace-Moselle pour les jours fériés locaux.
    depcode = postcodes[0][:2] if country_code == "FR" and postcodes else ""

    return {
        "label": label,
        "lat": place["latitude"],
        "lon": place["longitude"],
        "country_code": country_code,
        "depcode": depcode,
    }


def nearest_academie(lat: float, lon: float) -> str:
    def dist2(coord):
        return (coord[0] - lat) ** 2 + (coord[1] - lon) ** 2

    return min(ACADEMIE_CENTROIDS, key=lambda name: dist2(ACADEMIE_CENTROIDS[name]))


@lru_cache(maxsize=64)
def _vacation_periods(academie: str) -> tuple:
    # rows large et tri décroissant : à ~6 périodes ajoutées par an, la borne
    # de 100 aurait fini par tronquer silencieusement (68 enregistrements
    # aujourd'hui pour Paris), et le tri garantit que ce sont les plus
    # anciennes qui sauteraient le jour où ça arriverait, pas des périodes
    # récentes prises au hasard.
    resp = _get_json(
        "https://data.education.gouv.fr/api/records/1.0/search/",
        {
            "dataset": "fr-en-calendrier-scolaire",
            "refine.location": academie,
            "rows": 1000,
            "sort": "-end_date",
        },
        timeout=10,
    )
    periods = []
    for record in resp.json().get("records", []):
        fields = record["fields"]
        if fields.get("population") not in ("-", "Élèves"):
            continue  # exclut les dates spécifiques aux enseignants (été)
        # L'API encode des minuits parisiens en UTC ("2019-04-19T22:00:00+00:00"
        # = 20/04 00:00 à Paris) : convertir avant de retirer le fuseau, sinon
        # .normalize() retient la veille. Et l'intervalle est semi-ouvert —
        # end_date est le minuit du jour de *reprise* (06/05 pour des vacances
        # finissant le 05/05) — d'où le -1 jour, `vacation_flags` comparant des
        # bornes inclusives. Les deux corrections vont ensemble : n'en faire
        # qu'une déplacerait le décalage d'un bout de la période à l'autre.
        start = pd.Timestamp(fields["start_date"]).tz_convert("Europe/Paris").tz_localize(None).normalize()
        reprise = pd.Timestamp(fields["end_date"]).tz_convert("Europe/Paris").tz_localize(None).normalize()
        end = reprise - pd.Timedelta(days=1)
        periods.append((start, end))
    return tuple(periods)


def vacation_flags(academie: str, dates: pd.DatetimeIndex) -> pd.Series:
    """1 si le jour tombe dans une période de vacances scolaires de l'académie, 0 sinon."""
    periods = _vacation_periods(academie)
    flags = pd.Series(0, index=dates)
    for start, end in periods:
        flags.loc[(dates >= start) & (dates <= end)] = 1
    return flags


def holiday_flags(country_code: str, depcode: str, dates: pd.DatetimeIndex) -> pd.Series:
    """1 si le jour est férié dans le pays donné (jours locaux Alsace-Moselle
    inclus pour la France si pertinent). Renvoie une série à 0 si le pays
    n'est pas couvert par la bibliothèque `holidays`.
    """
    subdiv = "Alsace-Moselle" if country_code == "FR" and depcode in ALSACE_MOSELLE_DEPCODES else None
    years = sorted(set(dates.year)) or [pd.Timestamp.today().year]
    try:
        country_holidays = holidays.country_holidays(country_code, subdiv=subdiv, years=years)
    except NotImplementedError:
        return pd.Series(0, index=dates)
    return pd.Series([1 if d.date() in country_holidays else 0 for d in dates], index=dates)


def _open_meteo_archive(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    resp = _get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "daily": "temperature_2m_max,precipitation_sum",
            # "auto" = fuseau déduit des coordonnées. En forçant Europe/Paris,
            # les agrégats quotidiens (max de température, cumul de pluie)
            # étaient découpés sur des journées parisiennes : pour un point de
            # vente à Tokyo ou Montréal, le "max du 5 mars" couvrait une
            # fenêtre décalée de 7 à 14 h par rapport à sa propre journée,
            # donc une exogène désalignée d'une demi-journée avec la série.
            "timezone": "auto",
        },
        timeout=15,
    )
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
    météo) pour une localisation géocodée et un ensemble de dates. Les
    vacances scolaires ne sont disponibles que pour la France (seule source
    de données intégrée) — elles valent 0 ailleurs, en filet de sécurité si
    la variable est tout de même sollicitée pour un pays étranger.
    """
    weather = weather_series(loc["lat"], loc["lon"], dates)
    if loc["country_code"] == "FR":
        academie = nearest_academie(loc["lat"], loc["lon"])
        vacances = vacation_flags(academie, dates)
    else:
        vacances = pd.Series(0, index=dates)

    return pd.DataFrame(
        {
            "ferie": holiday_flags(loc["country_code"], loc["depcode"], dates),
            "vacances": vacances,
            "temp_max": weather["temp_max"].to_numpy(),
            "precipitation": weather["precipitation"].to_numpy(),
        },
        index=dates,
    )
