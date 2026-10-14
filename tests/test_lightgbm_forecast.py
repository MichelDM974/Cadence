import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import lightgbm_forecast as lf
from api import FORECAST_MODELS, align_external_exog, app, group_combinations, _daily_series

client = TestClient(app)


@pytest.fixture(autouse=True)
def _session():
    """L'endpoint de génération en masse exige une session (cf.
    conftest.connecter)."""
    from conftest import connecter

    connecter(client)


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
    # group_combinations (api.py) trie numériquement les identifiants
    # ("2" < "9" < "10"), un ordre différent d'un tri lexicographique de
    # chaînes — reproduire exactement l'ordre de production ici est
    # important : c'est ce décalage d'ordre, entre la construction de
    # future_exog et le tri interne de add_lag_features, qui a révélé le bug
    # de ré-étiquetage corrigé dans predict_bulk.
    combos = group_combinations(raw, group_cols)
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


def test_add_lag_features_include_yoy_computes_trend_from_last_year():
    # 400 jours : assez pour que shift(365).rolling(28) ait des valeurs non-NaN
    # sur la fin de la série (cf. add_lag_features(include_yoy=True)).
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    df = pd.DataFrame({"store_nbr": ["1"] * 400, "date": dates, "visitors": [10.0 + (i % 7) for i in range(400)]})
    result = lf.add_lag_features(df, ["store_nbr"], rolling_windows=(7, 28), include_yoy=True)

    assert "roll_mean_28_yoy" in result.columns
    assert "yoy_trend" in result.columns
    last = result.iloc[-1]
    assert not pd.isna(last["roll_mean_28_yoy"])
    assert last["yoy_trend"] == pytest.approx(last["roll_mean_28"] - last["roll_mean_28_yoy"])
    # historique insuffisant (< 1 an) : la feature reste NaN, pas une erreur
    assert pd.isna(result.iloc[10]["roll_mean_28_yoy"])


def test_run_bulk_lightgbm_forecast_feature_config_is_opt_in():
    """feature_config par défaut (absent) doit reproduire exactement le
    comportement lean actuel pour lags fins/rolling windows — ces deux axes
    restent opt-in, jamais des changements de défaut (cf. centre-de-controle :
    aucun n'améliore Store Sales et Recruit à la fois). `include_yoy` est
    l'exception, désormais auto-détecté (cf. tests dédiés ci-dessous) — ce
    test-ci le fixe explicitement à True pour ne pas dépendre de cette
    détection et rester focalisé sur les 2 autres axes."""
    raw = _multi_entity_frame(60, [("1", "A"), ("2", "B")])
    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)

    rows, skipped = lf.run_bulk_lightgbm_forecast(
        raw, "date", "sales", group_cols, 10, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, None,
        None, None,
        {"rolling_windows": (7, 28), "include_yoy": True},
    )

    assert skipped == set()
    assert len(rows) == len(combos) * 10


def test_should_auto_include_yoy_false_for_homogeneous_entity_lengths():
    raw = _multi_entity_frame(200, [("1", "A"), ("2", "B"), ("3", "C")])
    assert lf._should_auto_include_yoy(raw, ["store_nbr", "family"]) is False


def test_should_auto_include_yoy_true_for_heterogeneous_entity_lengths():
    long_entity = _multi_entity_frame(400, [("1", "A")])
    short_entity = _multi_entity_frame(60, [("2", "B")])
    raw = pd.concat([long_entity, short_entity], ignore_index=True)
    assert lf._should_auto_include_yoy(raw, ["store_nbr", "family"]) is True


def test_should_auto_include_yoy_false_for_single_entity():
    raw = _multi_entity_frame(200, [("1", "A")])
    assert lf._should_auto_include_yoy(raw, ["store_nbr", "family"]) is False


def test_should_auto_reduce_regularization_true_for_homogeneous_entity_lengths():
    raw = _multi_entity_frame(200, [("1", "A"), ("2", "B"), ("3", "C")])
    assert lf._should_auto_reduce_regularization(raw, ["store_nbr", "family"]) is True


def test_should_auto_reduce_regularization_false_for_heterogeneous_entity_lengths():
    long_entity = _multi_entity_frame(400, [("1", "A")])
    short_entity = _multi_entity_frame(60, [("2", "B")])
    raw = pd.concat([long_entity, short_entity], ignore_index=True)
    assert lf._should_auto_reduce_regularization(raw, ["store_nbr", "family"]) is False


def test_should_auto_reduce_regularization_false_for_single_entity():
    raw = _multi_entity_frame(200, [("1", "A")])
    assert lf._should_auto_reduce_regularization(raw, ["store_nbr", "family"]) is False


def _entity_frame_ending(n_days: int, store: str, family: str, end: str = "2024-06-30") -> pd.DataFrame:
    dates = pd.date_range(end=end, periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "store_nbr": store,
            "family": family,
            "sales": [20 + 5 * (i % 7) + 0.05 * i for i in range(n_days)],
        }
    )


def test_run_bulk_lightgbm_forecast_auto_enables_yoy_for_heterogeneous_history():
    raw = pd.concat(
        [_entity_frame_ending(400, "1", "A"), _entity_frame_ending(60, "2", "B")], ignore_index=True
    )
    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)
    call_args = (
        raw, "date", "sales", group_cols, 10, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, None,
    )

    auto_rows, _ = lf.run_bulk_lightgbm_forecast(*call_args, None, None, None)
    explicit_true_rows, _ = lf.run_bulk_lightgbm_forecast(*call_args, None, None, {"include_yoy": True})

    assert auto_rows == explicit_true_rows


def test_run_bulk_lightgbm_forecast_auto_reduces_regularization_for_homogeneous_history():
    raw = _multi_entity_frame(200, [("1", "A"), ("2", "B"), ("3", "C")])
    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)
    call_args = (
        raw, "date", "sales", group_cols, 10, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, None,
    )

    auto_rows, _ = lf.run_bulk_lightgbm_forecast(*call_args, None, None, None)
    explicit_reduced_rows, _ = lf.run_bulk_lightgbm_forecast(
        *call_args, lf.HOMOGENEOUS_PANEL_PARAMS, lf.HOMOGENEOUS_PANEL_NUM_BOOST_ROUND, None
    )

    assert auto_rows == explicit_reduced_rows


def test_run_bulk_lightgbm_forecast_keeps_regularization_default_for_heterogeneous_history():
    raw = pd.concat(
        [_entity_frame_ending(400, "1", "A"), _entity_frame_ending(60, "2", "B")], ignore_index=True
    )
    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)
    call_args = (
        raw, "date", "sales", group_cols, 10, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, None,
    )

    auto_rows, _ = lf.run_bulk_lightgbm_forecast(*call_args, None, None, None)
    explicit_default_rows, _ = lf.run_bulk_lightgbm_forecast(*call_args, {}, None, None)

    assert auto_rows == explicit_default_rows


def test_run_bulk_lightgbm_forecast_keeps_yoy_off_for_homogeneous_history():
    raw = _multi_entity_frame(200, [("1", "A"), ("2", "B")])
    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)
    call_args = (
        raw, "date", "sales", group_cols, 10, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, None,
    )

    auto_rows, _ = lf.run_bulk_lightgbm_forecast(*call_args, None, None, None)
    explicit_false_rows, _ = lf.run_bulk_lightgbm_forecast(*call_args, None, None, {"include_yoy": False})

    assert auto_rows == explicit_false_rows


def test_run_bulk_lightgbm_forecast_accepts_custom_lgb_params():
    raw = _multi_entity_frame(60, [("1", "A"), ("2", "B")])
    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)

    rows, skipped = lf.run_bulk_lightgbm_forecast(
        raw, "date", "sales", group_cols, 10, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, None,
        {"num_leaves": 7, "learning_rate": 0.2}, 15,
    )

    assert skipped == set()
    assert len(rows) == len(combos) * 10
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


def test_run_bulk_lightgbm_forecast_does_not_mix_up_entities():
    """Régression : add_lag_features trie ses lignes en interne par
    [*group_cols, "date"], un ordre différent de celui de future_exog/combos
    — si predict_bulk réutilise l'ordre non trié pour étiqueter les
    prévisions, elles se retrouvent attribuées à la mauvaise entité. Utilise
    des identifiants magasin dont le tri lexicographique diffère du tri
    numérique ("10" < "2" en tri de chaînes) et des niveaux de vente très
    séparés, pour que tout mélange soit immédiatement visible.
    """
    # Assez de jours pour que chaque entité ait largement plus de lignes que
    # min_data_in_leaf (sinon LightGBM ne peut pas former de feuille dédiée
    # par entité et mélange les niveaux — un faux positif à ne pas confondre
    # avec le vrai bug de ré-étiquetage visé par ce test).
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    levels = {"9": 5.0, "10": 500.0, "2": 50.0}
    rows = [
        {"date": d.strftime("%Y-%m-%d"), "store_nbr": store, "family": "A", "sales": level}
        for store, level in levels.items()
        for d in dates
    ]
    raw = pd.DataFrame(rows)
    group_cols = ["store_nbr", "family"]
    combos, group_indices = _combos_and_indices(raw, group_cols)

    forecast_rows, skipped = lf.run_bulk_lightgbm_forecast(
        raw, "date", "sales", group_cols, 5, combos, group_indices, [],
        None, [],
        None, None, None, None,
        "store_nbr", "family", None, None,
        _daily_series, align_external_exog, None,
    )

    assert skipped == set()
    by_store = pd.DataFrame(forecast_rows).groupby("store_nbr")["prevision"].mean()
    for store, level in levels.items():
        assert abs(by_store[store] - level) < level * 0.5 + 5, (
            f"magasin {store} (niveau réel {level}) a reçu une prévision moyenne de {by_store[store]} "
            f"— probable mélange d'entités"
        )


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
        time.sleep(0.05)  # laisse le job d'arrière-plan avancer plutôt que de saturer en poll serré
    assert progress["status"] == "done"

    download = client.get(f"/api/upload/forecast/bulk/download/{job_id}")
    assert download.status_code == 200
    df = pd.read_csv(io.StringIO(download.text))
    assert len(df) == 2 * 5
    assert "ic_bas" not in df.columns
    assert "ic_haut" not in df.columns
    assert set(df.columns) == {"store_nbr", "family", "date", "prevision"}
