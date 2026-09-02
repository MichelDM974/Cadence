import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import lightgbm_forecast as lf
from api import FORECAST_MODELS, align_external_exog, app, _daily_series

client = TestClient(app)


def _multi_entity_frame(n_days: int, entities: list[tuple[str, str]]) -> pd.DataFrame:
    """Deux entités par défaut, saisonnalité hebdo déterministe + tendance
    légère + un décalage de niveau entre entités (pas de bruit aléatoire,
    même convention que _synthetic_series dans test_api.py)."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for offset, (store, family) in enumerate(entities):
        for i, d in enumerate(dates):
            value = 20 + offset * 10 + 5 * (i % 7) + 0.05 * i
            rows.append({"date": d.strftime("%Y-%m-%d"), "store_nbr": store, "family": family, "sales": value})
    return pd.DataFrame(rows)


def _combos_and_indices(raw: pd.DataFrame, group_cols: list[str]):
    combos = sorted(set(zip(*(raw[c].astype(str) for c in group_cols))))
    str_cols = {c: raw[c].astype(str) for c in group_cols}
    group_indices = dict(raw.groupby([str_cols[c] for c in group_cols]).groups.items())
    return combos, group_indices


def test_add_calendar_features_matches_known_dates():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-07", "2024-01-08"])})  # dimanche, lundi
    df = lf.add_calendar_features(df)
    assert df["dow"].tolist() == [6, 0]
    assert df["is_weekend"].tolist() == [1, 0]


def test_add_lag_features_computes_expected_values():
    df = pd.DataFrame(
        {
            "store_nbr": ["1"] * 10,
            "date": pd.date_range("2024-01-01", periods=10),
            "visitors": [float(i) for i in range(10)],  # 0, 1, 2, ..., 9
        }
    )
    result = lf.add_lag_features(df, ["store_nbr"])

    # lag_k du dernier jour = valeur (log1p) d'il y a k jours
    last = result.iloc[-1]
    assert last["lag_1"] == np.log1p(8)
    assert last["lag_7"] == np.log1p(2)
    # premières lignes : lag indisponible (pas assez d'historique dans le groupe)
    assert pd.isna(result.iloc[0]["lag_1"])


def test_run_bulk_lightgbm_forecast_returns_expected_shape():
    raw = _multi_entity_frame(60, [("1", "A"), ("2", "B")])
    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)

    rows, skipped = lf.run_bulk_lightgbm_forecast(
        raw, "date", "sales", group_cols, 10, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, None,
    )

    assert skipped == set()
    assert len(rows) == len(combos) * 10
    assert {tuple(sorted(r.keys())) for r in rows} == {("date", "family", "prevision", "store_nbr")}
    assert all(r["prevision"] >= 0 for r in rows)


def test_run_bulk_lightgbm_forecast_skips_short_history_entities():
    raw = _multi_entity_frame(30, [("1", "A")])
    # ajoute une entité avec un historique trop court (< 21 jours, seuil de _daily_series)
    short = pd.DataFrame(
        {"date": ["2024-01-01", "2024-01-02"], "store_nbr": ["2", "2"], "family": ["B", "B"], "sales": [10.0, 12.0]}
    )
    raw = pd.concat([raw, short], ignore_index=True)

    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)

    rows, skipped = lf.run_bulk_lightgbm_forecast(
        raw, "date", "sales", group_cols, 5, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, None,
    )

    assert skipped == {("2", "B")}
    assert len(rows) == 5  # une seule entité restante x horizon


def test_run_bulk_lightgbm_forecast_handles_horizon_beyond_lag_window():
    raw = _multi_entity_frame(90, [("1", "A")])
    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)

    progress = []
    rows, skipped = lf.run_bulk_lightgbm_forecast(
        raw, "date", "sales", group_cols, 40, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, lambda h, hz: progress.append((h, hz)),
    )

    assert skipped == set()
    assert len(rows) == 40
    assert progress[-1] == (40, 40)
    assert all(r["prevision"] >= 0 for r in rows)


def test_lightgbm_excluded_from_single_entity_forecast_models():
    assert "lightgbm" not in FORECAST_MODELS


def test_bulk_lightgbm_endpoint_produces_csv_without_ci():
    raw = _multi_entity_frame(30, [("1", "A"), ("2", "B")])
    csv_bytes = raw.to_csv(index=False).encode()

    start = client.post(
        "/api/upload/forecast/bulk/start",
        files={"file": ("sales.csv", csv_bytes, "text/csv")},
        data={
            "date_col": "date",
            "value_col": "sales",
            "store_col": "store_nbr",
            "category_col": "family",
            "horizon": "5",
            "model": "lightgbm",
            "include_ci": "true",  # doit être ignoré pour lightgbm
        },
    )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    for _ in range(200):
        progress = client.get(f"/api/upload/forecast/bulk/progress/{job_id}").json()
        if progress["status"] != "running":
            break
    assert progress["status"] == "done"

    download = client.get(f"/api/upload/forecast/bulk/download/{job_id}")
    assert download.status_code == 200
    df = pd.read_csv(io.StringIO(download.text))
    assert len(df) == 2 * 5
    assert "ic_bas" not in df.columns
    assert "ic_haut" not in df.columns
    assert set(df.columns) == {"store_nbr", "family", "date", "prevision"}
