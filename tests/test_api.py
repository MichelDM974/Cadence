import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api import (
    align_external_exog,
    analyze,
    app,
    autotune,
    backtest,
    backtest_exog,
    external_exog_for_entity,
    forecast,
    forecast_exog,
    group_combinations,
    parse_csv,
    parse_external_exog,
    parse_model_params,
)


def make_csv_bytes(rows):
    lines = ["date,value"] + [f"{d},{v}" for d, v in rows]
    return "\n".join(lines).encode()


def test_parse_csv_builds_daily_series():
    rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 25)]  # 24 jours
    df = parse_csv(make_csv_bytes(rows), "date", "value")
    assert len(df) == 24
    assert list(df.columns) == ["visitors"]


def test_parse_csv_rejects_unknown_columns():
    content = make_csv_bytes([("2024-01-01", 10)])
    with pytest.raises(HTTPException):
        parse_csv(content, "nope", "value")


def test_parse_csv_detects_semicolon_delimiter():
    rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 25)]  # 24 jours
    lines = ["date;value"] + [f"{d};{v}" for d, v in rows]
    content = "\n".join(lines).encode()
    df = parse_csv(content, "date", "value")
    assert len(df) == 24
    assert list(df.columns) == ["visitors"]


def test_parse_csv_rejects_short_history():
    rows = [(f"2024-01-{d:02d}", 10) for d in range(1, 10)]  # 9 jours < 21
    with pytest.raises(HTTPException):
        parse_csv(make_csv_bytes(rows), "date", "value")


def make_multi_store_csv_bytes(store_rows):
    lines = ["store,date,value"] + [f"{s},{d},{v}" for s, d, v in store_rows]
    return "\n".join(lines).encode()


def test_parse_csv_filters_by_store():
    rows = []
    for d in range(1, 25):
        rows.append(("A", f"2024-01-{d:02d}", 10 + d))
        rows.append(("B", f"2024-01-{d:02d}", 100 + d))
    content = make_multi_store_csv_bytes(rows)

    df_a = parse_csv(content, "date", "value", store_col="store", store_value="A")
    df_b = parse_csv(content, "date", "value", store_col="store", store_value="B")

    assert len(df_a) == 24
    assert len(df_b) == 24
    assert df_a["visitors"].iloc[0] == 11
    assert df_b["visitors"].iloc[0] == 101


def test_parse_csv_rejects_unknown_store_column():
    rows = [("A", f"2024-01-{d:02d}", 10 + d) for d in range(1, 25)]
    content = make_multi_store_csv_bytes(rows)
    with pytest.raises(HTTPException):
        parse_csv(content, "date", "value", store_col="nope", store_value="A")


def test_parse_csv_rejects_short_history_for_one_store():
    rows = [("A", f"2024-01-{d:02d}", 10 + d) for d in range(1, 25)]
    rows += [("B", f"2024-01-{d:02d}", 100 + d) for d in range(1, 5)]  # B : 4 jours seulement
    content = make_multi_store_csv_bytes(rows)
    with pytest.raises(HTTPException):
        parse_csv(content, "date", "value", store_col="store", store_value="B")


def make_store_category_csv_bytes(rows):
    lines = ["store,family,date,value"] + [f"{s},{f},{d},{v}" for s, f, d, v in rows]
    return "\n".join(lines).encode()


def test_parse_csv_filters_by_store_and_category():
    rows = []
    for d in range(1, 25):
        rows.append(("1", "BEAUTY", f"2024-01-{d:02d}", 10 + d))
        rows.append(("1", "DAIRY", f"2024-01-{d:02d}", 50 + d))
        rows.append(("2", "BEAUTY", f"2024-01-{d:02d}", 100 + d))
    content = make_store_category_csv_bytes(rows)

    df = parse_csv(
        content, "date", "value", store_col="store", store_value="1", category_col="family", category_value="DAIRY"
    )

    assert len(df) == 24
    assert df["visitors"].iloc[0] == 51


def test_group_combinations_sorted_numerically_per_dimension():
    rows = []
    for store in ("10", "2"):
        for family in ("DAIRY", "BEAUTY"):
            for d in range(1, 22):
                rows.append((store, family, f"2024-01-{d:02d}", 1))
    raw = pd.read_csv(pd.io.common.BytesIO(make_store_category_csv_bytes(rows)))

    combos = group_combinations(raw, ["store", "family"])

    assert combos == [
        ("2", "BEAUTY"),
        ("2", "DAIRY"),
        ("10", "BEAUTY"),
        ("10", "DAIRY"),
    ]


def test_group_combinations_single_dimension():
    rows = [("A", "X", "2024-01-01", 1), ("B", "X", "2024-01-01", 1)]
    raw = pd.read_csv(pd.io.common.BytesIO(make_store_category_csv_bytes(rows)))

    assert group_combinations(raw, ["store"]) == [("A",), ("B",)]
    assert group_combinations(raw, []) == []


def make_csv_with_extra_bytes(rows):
    lines = ["date,value,promo"] + [f"{d},{v},{p}" for d, v, p in rows]
    return "\n".join(lines).encode()


def test_parse_csv_includes_extra_exog_column():
    rows = [(f"2024-01-{d:02d}", 10 + d, d % 2) for d in range(1, 25)]  # 24 jours
    df = parse_csv(make_csv_with_extra_bytes(rows), "date", "value", extra_cols=["promo"])

    assert list(df.columns) == ["visitors", "promo"]
    assert len(df) == 24
    assert df["promo"].iloc[0] == 1  # jour 1 : 1 % 2 == 1


def test_parse_csv_rejects_unknown_extra_column():
    rows = [(f"2024-01-{d:02d}", 10 + d, 0) for d in range(1, 25)]
    content = make_csv_with_extra_bytes(rows)
    with pytest.raises(HTTPException):
        parse_csv(content, "date", "value", extra_cols=["nope"])


def make_external_csv_bytes(rows):
    lines = ["date,price"] + [f"{d},{p}" for d, p in rows]
    return "\n".join(lines).encode()


def test_parse_external_exog_fills_gaps():
    # séries avec un trou (comme un fichier de cours sans week-ends)
    rows = [("2024-01-01", 100), ("2024-01-02", 102), ("2024-01-05", 110)]
    external = parse_external_exog(make_external_csv_bytes(rows), "date", ["price"])

    assert len(external) == 5  # 2024-01-01 -> 2024-01-05, comblé
    assert external.loc["2024-01-03", "price"] == 102  # report de la dernière valeur connue
    assert external.loc["2024-01-05", "price"] == 110


def test_align_external_exog_extends_beyond_known_range():
    rows = [("2024-01-01", 100), ("2024-01-02", 102)]
    external = parse_external_exog(make_external_csv_bytes(rows), "date", ["price"])

    future_dates = pd.date_range("2024-01-10", periods=3, freq="D")
    aligned = align_external_exog(external, future_dates)

    assert len(aligned) == 3
    assert (aligned["price"] == 102).all()  # dernière valeur connue reportée sur le futur


def make_external_csv_with_entities_bytes(rows):
    lines = ["date,store,family,promo"] + [f"{d},{s},{f},{p}" for d, s, f, p in rows]
    return "\n".join(lines).encode()


def test_parse_external_exog_keeps_one_row_per_entity_and_date():
    rows = [
        ("2024-01-01", "1", "A", 5),
        ("2024-01-02", "1", "A", 7),
        ("2024-01-01", "2", "B", 0),
        ("2024-01-02", "2", "B", 1),
    ]
    external = parse_external_exog(
        make_external_csv_with_entities_bytes(rows), "date", ["promo"], store_col="store", category_col="family"
    )

    # contrairement au cas sans store_col/category_col, pas d'agrégation par
    # date : une ligne par entité x date, prête à être filtrée par entité.
    assert len(external) == 4
    assert set(external["store"]) == {"1", "2"}


def test_external_exog_for_entity_isolates_one_entity_with_real_future_values():
    rows = [
        ("2024-01-01", "1", "A", 5),
        ("2024-01-02", "1", "A", 7),
        ("2024-01-01", "2", "B", 0),
        ("2024-01-02", "2", "B", 1),
    ]
    external = parse_external_exog(
        make_external_csv_with_entities_bytes(rows), "date", ["promo"], store_col="store", category_col="family"
    )

    entity = external_exog_for_entity(external, "date", ["promo"], "store", "1", "family", "A")

    assert len(entity) == 2
    assert entity.loc["2024-01-01", "promo"] == 5
    assert entity.loc["2024-01-02", "promo"] == 7  # varie vraiment, pas de report de la dernière valeur


def test_external_exog_for_entity_rejects_unknown_entity():
    rows = [("2024-01-01", "1", "A", 5)]
    external = parse_external_exog(
        make_external_csv_with_entities_bytes(rows), "date", ["promo"], store_col="store", category_col="family"
    )

    with pytest.raises(HTTPException):
        external_exog_for_entity(external, "date", ["promo"], "store", "999", "family", "A")


client = TestClient(app)


def test_exog_validate_reports_full_future_coverage():
    main_rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 11)]  # 2024-01-01 -> 2024-01-10
    exog_rows = [
        (f"2024-01-{d:02d}", "1", "A", d) for d in range(1, 21)  # couvre jusqu'au 2024-01-20
    ]

    resp = client.post(
        "/api/upload/exog/validate",
        files={
            "file": ("main.csv", make_csv_bytes(main_rows), "text/csv"),
            "exog_file": ("exog.csv", make_external_csv_with_entities_bytes(exog_rows), "text/csv"),
        },
        data={
            "date_col": "date",
            "exog_date_col": "date",
            "exog_value_cols": "promo",
            "exog_store_col": "store",
            "exog_category_col": "family",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["n_entities"] == 1
    assert body["days_covered"] == 10  # 2024-01-20 - 2024-01-10
    assert body["history_gap_days"] == 0  # couvre depuis avant le début de l'historique


def test_exog_validate_reports_missing_history_coverage():
    main_rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 11)]  # 2024-01-01 -> 2024-01-10
    exog_rows = [(f"2024-01-{d:02d}", "1", "A", d) for d in range(8, 15)]  # ne commence que le 2024-01-08

    resp = client.post(
        "/api/upload/exog/validate",
        files={
            "file": ("main.csv", make_csv_bytes(main_rows), "text/csv"),
            "exog_file": ("exog.csv", make_external_csv_with_entities_bytes(exog_rows), "text/csv"),
        },
        data={
            "date_col": "date",
            "exog_date_col": "date",
            "exog_value_cols": "promo",
            "exog_store_col": "store",
            "exog_category_col": "family",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["history_gap_days"] == 7  # 2024-01-08 - 2024-01-01


def test_exog_validate_rejects_missing_column():
    main_rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 11)]
    exog_rows = [(f"2024-01-{d:02d}", "1", "A", d) for d in range(1, 11)]

    resp = client.post(
        "/api/upload/exog/validate",
        files={
            "file": ("main.csv", make_csv_bytes(main_rows), "text/csv"),
            "exog_file": ("exog.csv", make_external_csv_with_entities_bytes(exog_rows), "text/csv"),
        },
        data={
            "date_col": "date",
            "exog_date_col": "date",
            "exog_value_cols": "promo",
            "exog_store_col": "nope",
        },
    )

    assert resp.status_code == 400


def _synthetic_series(n_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    # saisonnalité hebdo + légère tendance + bruit déterministe, pour que la
    # baseline naïve ne soit pas parfaite (cas dégénéré peu réaliste)
    values = [20 + 5 * (i % 7) + 0.05 * i + (i % 3) for i in range(n_days)]
    return pd.DataFrame({"visitors": values}, index=dates)


def test_analyze_without_flags_returns_expected_shape():
    df = _synthetic_series()
    result = analyze(df, unit_label="appels")
    assert len(result["dates"]) == len(df)
    assert len(result["trend"]) == len(df)
    assert "regularity_index" in result
    assert result["effects"] == {}
    assert result["weather"] is None
    assert result["unit_label"] == "appels"


def test_analyze_with_flags_reports_effects():
    df = _synthetic_series()
    flag = pd.Series(0, index=df.index)
    flag.iloc[::10] = 1  # un jour sur dix marqué
    result = analyze(df, flags={"ferie": flag})
    assert "ferie" in result["effects"]


def test_forecast_returns_requested_horizon():
    df = _synthetic_series()
    result = forecast(df, horizon=10)
    assert len(result["dates"]) == 10
    assert len(result["prevision"]) == 10
    assert len(result["ic_bas"]) == 10
    assert len(result["ic_haut"]) == 10


def test_forecast_prophet_returns_requested_horizon():
    df = _synthetic_series()
    result = forecast(df, horizon=10, model="prophet")
    assert len(result["dates"]) == 10
    assert len(result["prevision"]) == 10
    assert len(result["ic_bas"]) == 10
    assert len(result["ic_haut"]) == 10


def test_forecast_rejects_unknown_model():
    df = _synthetic_series()
    with pytest.raises(HTTPException):
        forecast(df, horizon=10, model="nope")


def test_forecast_sarimax_respects_custom_order():
    df = _synthetic_series()
    default_result = forecast(df, horizon=10)
    custom_result = forecast(df, horizon=10, model_params={"order": [2, 1, 2], "seasonal_order": [0, 1, 1, 7]})
    assert default_result["prevision"] != custom_result["prevision"]


def test_forecast_prophet_respects_custom_seasonality_mode():
    df = _synthetic_series()
    result = forecast(df, horizon=10, model="prophet", model_params={"seasonality_mode": "multiplicative"})
    assert len(result["prevision"]) == 10


def test_backtest_respects_model_params_per_model():
    df = _synthetic_series(n_days=150)
    result = backtest(
        df, horizon=20,
        model_params={"sarimax": {"order": [2, 1, 1]}, "prophet": {"changepoint_prior_scale": 0.5}},
    )
    assert set(result["predictions"]) == {"naive", "sarimax", "prophet"}


def test_parse_model_params_empty_returns_empty_dict():
    assert parse_model_params(None) == {}
    assert parse_model_params("") == {}


def test_parse_model_params_decodes_json():
    assert parse_model_params('{"sarimax": {"order": [2, 1, 1]}}') == {"sarimax": {"order": [2, 1, 1]}}


def test_parse_model_params_rejects_invalid_json():
    with pytest.raises(HTTPException):
        parse_model_params("not json")


def test_parse_model_params_rejects_non_object():
    with pytest.raises(HTTPException):
        parse_model_params("[1, 2, 3]")


def test_backtest_rejects_short_history():
    df = _synthetic_series(n_days=60)
    with pytest.raises(HTTPException):
        backtest(df, horizon=28)


def test_backtest_returns_comparison():
    df = _synthetic_series(n_days=150)
    result = backtest(df, horizon=20, unit_label="appels")
    assert len(result["dates"]) == 20
    assert len(result["actual"]) == 20
    assert set(result["predictions"]) == {"naive", "sarimax", "prophet"}
    assert len(result["predictions"]["sarimax"]) == 20
    assert len(result["predictions"]["prophet"]) == 20
    assert set(result["mae"]) == {"naive", "sarimax", "prophet"}
    assert set(result["improvement_pct"]) == {"sarimax", "prophet"}
    assert result["unit_label"] == "appels"


def test_autotune_rejects_short_history():
    df = _synthetic_series(n_days=60)
    with pytest.raises(HTTPException):
        autotune(df, horizon=28)


def test_autotune_returns_better_or_equal_sarimax_params():
    df = _synthetic_series(n_days=150)
    result = autotune(df, horizon=20, models=("sarimax",))
    assert "sarimax" in result
    assert "prophet" not in result
    assert len(result["sarimax"]["order"]) == 3
    assert len(result["sarimax"]["seasonal_order"]) == 4
    assert result["sarimax"]["mae_after"] <= result["sarimax"]["mae_before"]


def test_autotune_sarimax_keeps_d_and_bigd_fixed_at_one():
    # d/D quasi toujours optimaux à 1 en pratique : la recherche ne les fait
    # pas varier, elle se concentre sur p/q/P/Q (cf. discussion produit).
    df = _synthetic_series(n_days=150)
    result = autotune(df, horizon=20, models=("sarimax",))
    assert result["sarimax"]["order"][1] == 1  # d
    assert result["sarimax"]["seasonal_order"][1] == 1  # D


def test_autotune_sarimax_never_regresses_vs_current_config(monkeypatch):
    # Si la config déjà en place bat tout ce que la descente par coordonnées
    # trouve (ici forcée à un résultat volontairement mauvais), autotune doit
    # la conserver telle quelle plutôt que de renvoyer un résultat pire.
    import api

    monkeypatch.setattr(
        api, "_sarimax_coordinate_search", lambda *a, **k: ({"order": [2, 1, 2], "seasonal_order": [2, 1, 2, 7]}, 9999.0)
    )
    df = _synthetic_series(n_days=150)
    current = {"sarimax": {"order": [3, 1, 0], "seasonal_order": [0, 1, 3, 7]}}
    result = autotune(df, horizon=20, models=("sarimax",), model_params=current)
    assert result["sarimax"]["order"] == [3, 1, 0]
    assert result["sarimax"]["seasonal_order"] == [0, 1, 3, 7]
    assert result["sarimax"]["mae_after"] == result["sarimax"]["mae_before"]


def test_autotune_returns_better_or_equal_prophet_params():
    df = _synthetic_series(n_days=150)
    result = autotune(df, horizon=20, models=("prophet",))
    assert "prophet" in result
    assert "sarimax" not in result
    assert result["prophet"]["seasonality_mode"] in {"additive", "multiplicative"}
    assert result["prophet"]["mae_after"] <= result["prophet"]["mae_before"]


def test_autotune_keeps_configured_seasonal_period():
    df = _synthetic_series(n_days=150)
    result = autotune(df, horizon=20, models=("sarimax",), model_params={"sarimax": {"seasonal_order": [1, 1, 1, 14]}})
    assert result["sarimax"]["seasonal_order"][3] == 14


def _synthetic_series_with_extra(n_days: int = 60) -> pd.DataFrame:
    df = _synthetic_series(n_days)
    df["promo"] = [i % 3 for i in range(n_days)]
    return df


def test_forecast_exog_carries_forward_extra_col_into_future():
    df = _synthetic_series_with_extra(60)
    exog_hist, exog_future = forecast_exog(
        df, horizon=10, location=None, use_ferie=False, use_vacances=False, use_meteo=False, extra_cols=["promo"]
    )

    assert list(exog_hist.columns) == ["promo"]
    assert len(exog_hist) == 60
    assert len(exog_future) == 10
    # pas de valeurs futures connues : la dernière valeur observée est reportée sur tout l'horizon
    assert (exog_future["promo"] == df["promo"].iloc[-1]).all()


def test_forecast_exog_merges_external_series():
    df = _synthetic_series_with_extra(60)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    external = parse_external_exog(
        make_external_csv_bytes(list(zip(dates, range(len(dates))))), "date", ["price"]
    )

    exog_hist, exog_future = forecast_exog(
        df, horizon=10, location=None, use_ferie=False, use_vacances=False, use_meteo=False, external_exog=external
    )

    assert list(exog_hist.columns) == ["price"]
    assert len(exog_hist) == 60
    assert len(exog_future) == 10


def test_backtest_exog_uses_known_future_values_for_extra_col():
    df = _synthetic_series_with_extra(150)
    exog_hist, exog_future = backtest_exog(
        df, horizon=20, location=None, use_ferie=False, use_vacances=False, use_meteo=False, extra_cols=["promo"]
    )

    assert len(exog_hist) == 130
    assert len(exog_future) == 20
    # contrairement au vrai forecast, le backtest connaît déjà ces valeurs (données passées)
    assert (exog_future["promo"].to_numpy() == df["promo"].iloc[-20:].to_numpy()).all()
