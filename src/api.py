"""API Cadence (FastAPI) : décomposition du rythme, indice de régularité,
ruptures et prévision — à partir d'un restaurant du dataset de test (Kaggle)
ou d'un CSV importé, avec enrichissement optionnel par localisation (jours
fériés, vacances scolaires, météo). Sert aussi le frontend statique
(src/static/).
"""

import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

import exogenous
from decompose_prototype import (
    DATA_DIR,
    detect_anomalies,
    event_components,
    load_series,
    regularity_index,
)

app = FastAPI(title="Cadence API")

STATIC_DIR = Path(__file__).parent / "static"


def analyze(df: pd.DataFrame, flags: dict | None = None, weather: pd.DataFrame | None = None) -> dict:
    stl = STL(df["visitors"], period=7, robust=True)
    result = stl.fit()

    event, effects = event_components(result.resid, flags or {})
    adjusted_resid = result.resid - event
    idx = regularity_index(adjusted_resid, df["visitors"])
    anomalies = detect_anomalies(adjusted_resid)

    weather_info = None
    if weather is not None:
        corr_temp = adjusted_resid.corr(weather["temp_max"])
        corr_rain = adjusted_resid.corr(weather["precipitation"])
        weather_info = {
            "temp_max": weather["temp_max"].round(1).tolist(),
            "precipitation": weather["precipitation"].round(1).tolist(),
            "corr_temp": None if pd.isna(corr_temp) else round(float(corr_temp), 2),
            "corr_precipitation": None if pd.isna(corr_rain) else round(float(corr_rain), 2),
        }

    return {
        "dates": [d.date().isoformat() for d in df.index],
        "observed": df["visitors"].round(1).tolist(),
        "trend": result.trend.round(1).tolist(),
        "seasonal": result.seasonal.round(1).tolist(),
        "event": event.round(1).tolist(),
        "resid": adjusted_resid.round(1).tolist(),
        "regularity_index": round(float(idx), 3),
        "effects": {name: round(float(value), 1) for name, value in effects.items()},
        "weather": weather_info,
        "anomalies": [
            {"date": d.date().isoformat(), "resid": round(float(v), 1)}
            for d, v in anomalies.sort_values(key=abs, ascending=False).items()
        ],
    }


def forecast(
    df: pd.DataFrame,
    horizon: int,
    exog_hist: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
) -> dict:
    model = SARIMAX(
        df["visitors"],
        exog=exog_hist,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    fc = fitted.get_forecast(steps=horizon, exog=exog_future)
    mean = fc.predicted_mean.clip(lower=0)
    ci = fc.conf_int(alpha=0.2).clip(lower=0)

    return {
        "dates": [d.date().isoformat() for d in mean.index],
        "prevision": mean.round(1).tolist(),
        "ic_bas": ci.iloc[:, 0].round(1).tolist(),
        "ic_haut": ci.iloc[:, 1].round(1).tolist(),
    }


def parse_csv(content: bytes, date_col: str, value_col: str) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(content))
    if date_col not in raw.columns or value_col not in raw.columns:
        raise HTTPException(400, "Colonnes introuvables dans le CSV")

    s = raw[[date_col, value_col]].dropna().copy()
    s.columns = ["visit_date", "visitors"]
    s["visit_date"] = pd.to_datetime(s["visit_date"], errors="coerce")
    s = s.dropna(subset=["visit_date"]).groupby("visit_date")["visitors"].sum()

    if len(s) < 21:
        raise HTTPException(400, "Historique trop court (minimum 3 semaines)")

    full_range = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_range).fillna(0)
    s.index.name = "visit_date"
    return s.to_frame("visitors")


def geocode_or_400(location: str) -> dict:
    try:
        return exogenous.geocode(location)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Service de géocodage indisponible : {exc}") from exc


@app.get("/api/stores")
def api_stores(n: int = 20):
    visits = pd.read_csv(f"{DATA_DIR}/air_visit_data.csv")
    counts = visits.groupby("air_store_id").size().sort_values(ascending=False)
    return [{"id": store_id, "n_days": int(count)} for store_id, count in counts.head(n).items()]


@app.get("/api/stores/{store_id}/analysis")
def api_store_analysis(store_id: str):
    df = load_series(store_id)
    return analyze(df, flags={"ferie": df["holiday_flg"]})


@app.get("/api/stores/{store_id}/forecast")
def api_store_forecast(store_id: str, horizon: int = Query(28, ge=7, le=90)):
    return forecast(load_series(store_id), horizon)


@app.post("/api/upload/analysis")
async def api_upload_analysis(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    location: str | None = Form(None),
    use_ferie: bool = Form(True),
    use_vacances: bool = Form(True),
    use_meteo: bool = Form(True),
):
    df = parse_csv(await file.read(), date_col, value_col)

    flags, weather = None, None
    if location:
        loc = geocode_or_400(location)
        exog = exogenous.location_exog(loc, df.index)
        flags = {}
        if use_ferie:
            flags["ferie"] = exog["ferie"]
        if use_vacances:
            flags["vacances"] = exog["vacances"]
        if use_meteo:
            weather = exog[["temp_max", "precipitation"]]

    return analyze(df, flags=flags, weather=weather)


@app.post("/api/upload/forecast")
async def api_upload_forecast(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    horizon: int = Form(28),
    location: str | None = Form(None),
    use_ferie: bool = Form(True),
    use_vacances: bool = Form(True),
    use_meteo: bool = Form(True),
):
    df = parse_csv(await file.read(), date_col, value_col)

    exog_hist, exog_future = None, None
    columns = (["ferie"] if use_ferie else []) + (["vacances"] if use_vacances else [])
    columns += ["temp_max", "precipitation"] if use_meteo else []

    if location and columns:
        loc = geocode_or_400(location)
        exog_hist = exogenous.location_exog(loc, df.index)[columns]
        future_dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        exog_future = exogenous.location_exog(loc, future_dates)[columns]

    return forecast(df, horizon, exog_hist=exog_hist, exog_future=exog_future)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
