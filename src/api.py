"""API Cadence (FastAPI) : décomposition du rythme, indice de régularité,
ruptures et prévision — à partir d'un restaurant du dataset de test (Kaggle)
ou d'un CSV importé, avec enrichissement optionnel par localisation (jours
fériés, vacances scolaires, météo). Sert aussi le frontend statique
(src/static/).
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

import exogenous
import storage
from decompose_prototype import (
    DATA_DIR,
    detect_anomalies,
    event_components,
    load_series,
    regularity_index,
    seasonal_naive_forecast,
)

app = FastAPI(title="Cadence API")
storage.init_db()

STATIC_DIR = Path(__file__).parent / "static"
MIN_BACKTEST_TRAIN_DAYS = 90


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


def backtest(
    df: pd.DataFrame,
    horizon: int,
    exog_hist: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
) -> dict:
    """Compare, sur les derniers `horizon` jours déjà connus, la prévision
    SARIMAX à une baseline naïve ("même jour la semaine dernière") — sert à
    montrer concrètement l'apport du modèle pour ce restaurant précis.
    """
    if len(df) < MIN_BACKTEST_TRAIN_DAYS + horizon:
        raise HTTPException(
            400,
            f"Historique trop court pour un backtest fiable "
            f"(minimum {MIN_BACKTEST_TRAIN_DAYS + horizon} jours, {len(df)} disponibles)",
        )

    train, test = df.iloc[:-horizon], df.iloc[-horizon:]
    naive_pred = seasonal_naive_forecast(train["visitors"], horizon)
    sarimax_result = forecast(train, horizon, exog_hist=exog_hist, exog_future=exog_future)
    sarimax_pred = np.array(sarimax_result["prevision"])
    actual = test["visitors"].to_numpy()

    mae_naive = float(np.abs(naive_pred - actual).mean())
    mae_sarimax = float(np.abs(sarimax_pred - actual).mean())
    improvement_pct = None if mae_naive == 0 else round((mae_naive - mae_sarimax) / mae_naive * 100, 1)

    return {
        "dates": [d.date().isoformat() for d in test.index],
        "actual": [round(float(v), 1) for v in actual],
        "naive": [round(float(v), 1) for v in naive_pred],
        "sarimax": [round(float(v), 1) for v in sarimax_pred],
        "mae_naive": round(mae_naive, 1),
        "mae_sarimax": round(mae_sarimax, 1),
        "improvement_pct": improvement_pct,
    }


def parse_csv(content: bytes, date_col: str, value_col: str) -> pd.DataFrame:
    try:
        raw = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    except Exception:
        raw = pd.read_csv(io.BytesIO(content))  # repli : virgule standard

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


def exog_columns(use_ferie: bool, use_vacances: bool, use_meteo: bool) -> list[str]:
    columns = (["ferie"] if use_ferie else []) + (["vacances"] if use_vacances else [])
    columns += ["temp_max", "precipitation"] if use_meteo else []
    return columns


def flags_and_weather(
    df: pd.DataFrame, location: str | None, use_ferie: bool, use_vacances: bool, use_meteo: bool
):
    if not location:
        return None, None
    loc = geocode_or_400(location)
    exog = exogenous.location_exog(loc, df.index)
    flags = {}
    if use_ferie:
        flags["ferie"] = exog["ferie"]
    if use_vacances:
        flags["vacances"] = exog["vacances"]
    weather = exog[["temp_max", "precipitation"]] if use_meteo else None
    return flags, weather


def forecast_exog(
    df: pd.DataFrame, horizon: int, location: str | None, use_ferie: bool, use_vacances: bool, use_meteo: bool
):
    columns = exog_columns(use_ferie, use_vacances, use_meteo)
    if not (location and columns):
        return None, None
    loc = geocode_or_400(location)
    exog_hist = exogenous.location_exog(loc, df.index)[columns]
    future_dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    exog_future = exogenous.location_exog(loc, future_dates)[columns]
    return exog_hist, exog_future


def backtest_exog(
    df: pd.DataFrame, horizon: int, location: str | None, use_ferie: bool, use_vacances: bool, use_meteo: bool
):
    columns = exog_columns(use_ferie, use_vacances, use_meteo)
    if not (location and columns):
        return None, None
    loc = geocode_or_400(location)
    exog = exogenous.location_exog(loc, df.index)[columns]
    return exog.iloc[:-horizon], exog.iloc[-horizon:]


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


@app.get("/api/stores/{store_id}/backtest")
def api_store_backtest(store_id: str, horizon: int = Query(28, ge=7, le=90)):
    df = load_series(store_id)
    exog = df[["holiday_flg"]]
    return backtest(df, horizon, exog_hist=exog.iloc[:-horizon], exog_future=exog.iloc[-horizon:])


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
    flags, weather = flags_and_weather(df, location, use_ferie, use_vacances, use_meteo)
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
    exog_hist, exog_future = forecast_exog(df, horizon, location, use_ferie, use_vacances, use_meteo)
    return forecast(df, horizon, exog_hist=exog_hist, exog_future=exog_future)


@app.post("/api/upload/backtest")
async def api_upload_backtest(
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
    exog_hist, exog_future = backtest_exog(df, horizon, location, use_ferie, use_vacances, use_meteo)
    return backtest(df, horizon, exog_hist=exog_hist, exog_future=exog_future)


@app.post("/api/points")
async def api_save_point(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    name: str = Form(...),
    location: str | None = Form(None),
    use_ferie: bool = Form(True),
    use_vacances: bool = Form(True),
    use_meteo: bool = Form(True),
):
    df = parse_csv(await file.read(), date_col, value_col)
    point_id = storage.save_point(name, df, location, use_ferie, use_vacances, use_meteo)
    return {"id": point_id, "name": name}


@app.get("/api/points")
def api_list_points():
    return storage.list_points()


@app.delete("/api/points/{point_id}")
def api_delete_point(point_id: int):
    if not storage.delete_point(point_id):
        raise HTTPException(404, "Point de vente introuvable")
    return {"deleted": True}


def load_point_or_404(point_id: int) -> dict:
    point = storage.load_point(point_id)
    if point is None:
        raise HTTPException(404, "Point de vente introuvable")
    return point


@app.get("/api/points/{point_id}/analysis")
def api_point_analysis(point_id: int):
    point = load_point_or_404(point_id)
    flags, weather = flags_and_weather(
        point["df"], point["location"], point["use_ferie"], point["use_vacances"], point["use_meteo"]
    )
    return analyze(point["df"], flags=flags, weather=weather)


@app.get("/api/points/{point_id}/forecast")
def api_point_forecast(point_id: int, horizon: int = Query(28, ge=7, le=90)):
    point = load_point_or_404(point_id)
    exog_hist, exog_future = forecast_exog(
        point["df"], horizon, point["location"], point["use_ferie"], point["use_vacances"], point["use_meteo"]
    )
    return forecast(point["df"], horizon, exog_hist=exog_hist, exog_future=exog_future)


@app.get("/api/points/{point_id}/backtest")
def api_point_backtest(point_id: int, horizon: int = Query(28, ge=7, le=90)):
    point = load_point_or_404(point_id)
    exog_hist, exog_future = backtest_exog(
        point["df"], horizon, point["location"], point["use_ferie"], point["use_vacances"], point["use_meteo"]
    )
    return backtest(point["df"], horizon, exog_hist=exog_hist, exog_future=exog_future)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
