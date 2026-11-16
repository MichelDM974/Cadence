import io
import time

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
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


def test_parse_csv_reads_slash_dates_day_first():
    # format JJ/MM/AAAA (restaurateurs FR) : "03/04/2024" doit être lu comme
    # le 3 avril, pas le 4 mars (piège du MM/JJ par défaut de pandas).
    rows = [(f"{d:02d}/04/2024", 10 + d) for d in range(1, 22)]  # 21 jours
    df = parse_csv(make_csv_bytes(rows), "date", "value")
    assert pd.Timestamp("2024-04-03") in df.index
    assert pd.Timestamp("2024-03-04") not in df.index
    assert len(df) == 21


def test_parse_csv_rejects_unreadable_dates():
    rows = [(f"pas-une-date-{d}", 10) for d in range(1, 22)]
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


@pytest.fixture(autouse=True)
def _session():
    """Les endpoints de calcul exigent une session (cf. conftest.connecter).
    Le client étant partagé par le module, on ouvre la session avant chaque
    test qui en a besoin."""
    from conftest import connecter

    connecter(client)


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


# --- Authentification des endpoints de calcul --------------------------

def test_endpoints_de_calcul_refusent_un_client_sans_session():
    """Un client anonyme (cookies propres, sans passer par _session) ne doit
    pouvoir déclencher aucun calcul : ces endpoints étaient ouverts alors
    que seul l'affichage HTML était protégé côté navigateur."""
    anonyme = TestClient(app)
    rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 11)]

    assert anonyme.get("/api/stores").status_code == 401
    assert anonyme.post(
        "/api/upload/store-ids",
        files={"file": ("main.csv", make_csv_bytes(rows), "text/csv")},
        data={"store_col": "store"},
    ).status_code == 401
    assert anonyme.post(
        "/api/upload/forecast/bulk/count",
        files={"file": ("main.csv", make_csv_bytes(rows), "text/csv")},
        data={"date_col": "date", "value_col": "visitors"},
    ).status_code == 401


def test_login_et_setup_restent_accessibles_sans_session():
    """Le pendant : les routes d'amorçage doivent rester publiques, sinon
    plus personne ne peut se connecter."""
    anonyme = TestClient(app)
    assert anonyme.get("/api/me").status_code == 200


def test_upload_trop_volumineux_rejete(monkeypatch):
    """La borne évite de charger un fichier de plusieurs Go en mémoire avant
    le moindre contrôle. Limite abaissée ici pour ne pas fabriquer 100 Mo."""
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 50)
    rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 11)]
    resp = client.post(
        "/api/upload/store-ids",
        files={"file": ("main.csv", make_csv_bytes(rows), "text/csv")},
        data={"store_col": "store"},
    )
    assert resp.status_code == 413


# --- Robustesse : purge des jobs, CSV illisible, score infini ----------

def test_purge_des_jobs_expires():
    """Un job n'était retiré qu'au téléchargement réussi : en erreur, ou si
    l'utilisateur ferme l'onglet, il restait en mémoire avec son CSV
    complet pour la vie du process."""
    api.bulk_jobs.clear()
    api.bulk_jobs["vieux"] = {"created_at": 0.0, "csv": "x" * 1000}
    api.bulk_jobs["recent"] = {"created_at": time.time(), "csv": None}

    api.purge_stale_bulk_jobs()

    assert "vieux" not in api.bulk_jobs
    assert "recent" in api.bulk_jobs
    api.bulk_jobs.clear()


def test_csv_illisible_renvoie_400_et_non_500():
    resp = client.post(
        "/api/upload/store-ids",
        files={"file": ("main.csv", b"\x50\x4b\x03\x04 pas du tout un csv\x00\xff", "text/csv")},
        data={"store_col": "store"},
    )
    assert resp.status_code == 400


def test_modele_inconnu_rejete_immediatement():
    """Le modèle était validé dans le thread : l'appel renvoyait 200 + un
    job_id, et l'erreur n'apparaissait qu'au premier /progress."""
    rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 25)]
    resp = client.post(
        "/api/upload/forecast/bulk/start",
        files={"file": ("main.csv", make_multi_store_csv_bytes(
            [("A", d, v) for d, v in rows] + [("B", d, v) for d, v in rows]
        ), "text/csv")},
        data={"date_col": "date", "value_col": "value", "store_col": "store", "model": "nawak"},
    )
    assert resp.status_code == 400
    assert "nawak" in resp.text


# --- Exogènes : robustesse réseau -------------------------------------

def test_get_json_retente_sur_429(monkeypatch):
    """Un seul 429 au milieu d'une boucle sur beaucoup de lieux faisait tout
    échouer et perdait le cache déjà constitué."""
    import exogenous

    class FausseReponse:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    appels = []

    def faux_get(url, params, timeout):
        appels.append(url)
        return FausseReponse(429 if len(appels) == 1 else 200)

    monkeypatch.setattr(exogenous.requests, "get", faux_get)
    monkeypatch.setattr(exogenous.time, "sleep", lambda _: None)

    resp = exogenous._get_json("http://exemple", {})
    assert resp.status_code == 200
    assert len(appels) == 2  # le premier essai a été retenté


def test_get_json_abandonne_apres_les_tentatives(monkeypatch):
    import exogenous

    monkeypatch.setattr(exogenous.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        exogenous.requests, "get",
        lambda url, params, timeout: (_ for _ in ()).throw(exogenous.requests.RequestException("réseau coupé")),
    )
    with pytest.raises(exogenous.requests.RequestException):
        exogenous._get_json("http://exemple", {})


# --- Connexion : anti-bruteforce et secret de session ------------------

def test_login_bloque_apres_trop_d_echecs():
    api._login_echecs.clear()
    anonyme = TestClient(app)
    api.storage.create_user("cible", "le-bon-mot-de-passe")

    codes = [
        anonyme.post("/api/login", data={"username": "cible", "password": "faux"}).status_code
        for _ in range(api.LOGIN_MAX_ECHECS + 2)
    ]
    assert codes[0] == 401
    assert codes[-1] == 429  # plus même la peine de vérifier le mot de passe

    # et le blocage tient même avec le bon mot de passe, tant que la fenêtre court
    assert anonyme.post(
        "/api/login", data={"username": "cible", "password": "le-bon-mot-de-passe"}
    ).status_code == 429
    api._login_echecs.clear()


def test_login_reussi_remet_le_compteur_a_zero():
    api._login_echecs.clear()
    anonyme = TestClient(app)
    api.storage.create_user("bob", "secret")

    anonyme.post("/api/login", data={"username": "bob", "password": "faux"})
    assert api._login_echecs
    assert anonyme.post("/api/login", data={"username": "bob", "password": "secret"}).status_code == 200
    assert not api._login_echecs


def test_secret_de_session_non_lisible_par_les_autres():
    import storage

    storage.session_secret()
    mode = (storage.DB_PATH.parent / ".session_secret").stat().st_mode & 0o777
    assert mode == 0o600, oct(mode)


# --- Endpoints d'upload : couverture HTTP ------------------------------
# Ces trois routes n'avaient aucun test HTTP alors qu'elles partagent 34
# lignes identiques (préparation du CSV et des exogènes) : de quoi laisser
# passer une régression sur l'une sans que rien ne le signale.

def _csv_deux_saisons():
    """~20 semaines : le backtest exige au moins 97 jours pour un horizon
    de 7 (train + fenêtre de test), l'autotune davantage encore."""
    base = pd.date_range("2024-01-01", periods=140, freq="D")
    return make_csv_bytes([(d.date().isoformat(), 100 + 20 * (d.dayofweek >= 5)) for d in base])


def test_upload_forecast_renvoie_une_prevision():
    resp = client.post(
        "/api/upload/forecast",
        files={"file": ("main.csv", _csv_deux_saisons(), "text/csv")},
        data={"date_col": "date", "value_col": "value", "horizon": "7", "model": "sarimax"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["prevision"]) == 7
    assert len(body["dates"]) == 7


def test_upload_backtest_compare_a_la_baseline():
    resp = client.post(
        "/api/upload/backtest",
        files={"file": ("main.csv", _csv_deux_saisons(), "text/csv")},
        data={"date_col": "date", "value_col": "value", "horizon": "7"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "naive" in body["mae"]
    assert len(body["actual"]) == 7


def test_upload_autotune_renvoie_une_config():
    resp = client.post(
        "/api/upload/autotune",
        files={"file": ("main.csv", _csv_deux_saisons(), "text/csv")},
        data={"date_col": "date", "value_col": "value", "horizon": "7", "models": "sarimax"},
    )
    assert resp.status_code == 200, resp.text
    assert "sarimax" in resp.json()


def test_serie_json_remplace_les_non_finis_par_null():
    """Un ajustement non convergé peut produire des NaN (clip les laisse
    passer) : sérialisés tels quels, ils faisaient tomber l'endpoint en 500,
    "Out of range float values are not JSON compliant". Reproduit en CI sur
    une série déterministe, jamais en local — d'où ce test qui ne dépend
    d'aucune version de statsmodels."""
    import math

    assert api._serie_json([1.0, float("nan"), 3.14159]) == [1.0, None, 3.1]
    assert api._serie_json([float("inf"), float("-inf")]) == [None, None]
    assert all(v is None or math.isfinite(v) for v in api._serie_json([float("nan"), 2.0]))


def test_forecast_serialisable_meme_si_le_modele_renvoie_des_nan(monkeypatch):
    """Bout en bout : même avec un modèle qui ne renvoie que des NaN, la
    réponse doit rester du JSON valide (valeurs à null) plutôt qu'un 500."""
    import numpy as np

    def faux_sarimax(df, horizon, **kwargs):
        dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        nans = [np.nan] * horizon
        return {
            "dates": [d.date().isoformat() for d in dates],
            "prevision": api._serie_json(nans),
            "ic_bas": api._serie_json(nans),
            "ic_haut": api._serie_json(nans),
        }

    monkeypatch.setitem(api.FORECAST_MODELS, "sarimax", faux_sarimax)
    resp = client.post(
        "/api/upload/forecast",
        files={"file": ("main.csv", _csv_deux_saisons(), "text/csv")},
        data={"date_col": "date", "value_col": "value", "horizon": "7", "model": "sarimax"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["prevision"] == [None] * 7


# --- Génération en masse : chemin SARIMAX/Prophet (pool de process) ----
# Ce chemin n'avait aucun test : seul le branchement lightgbm était couvert,
# alors que c'est ici qu'on forke le pool et qu'on soumet une tâche par
# entité. C'est aussi le code le plus coûteux de l'app.

def _frame_entites(n_jours, entites):
    """Saisonnalité hebdo + tendance + une variation qui évite une série
    parfaitement lisse : celles-ci font diverger l'ajustement SARIMAX selon
    la version de statsmodels (cf. le bug NaN trouvé par la CI)."""
    dates = pd.date_range("2024-01-01", periods=n_jours, freq="D")
    lignes = []
    for rang, (magasin, famille) in enumerate(entites):
        for i, d in enumerate(dates):
            valeur = 50 + rang * 40 + 8 * (i % 7) + 0.1 * i + (3 if i % 5 == 0 else 0)
            lignes.append(
                {"date": d.strftime("%Y-%m-%d"), "store_nbr": magasin, "family": famille, "sales": valeur}
            )
    return pd.DataFrame(lignes)


def _lancer_bulk(csv_bytes, **champs):
    """Démarre un job et attend sa fin. Poll espacé : un poll serré a déjà
    rendu un test bulk instable en le faisant concurrencer le thread de
    travail."""
    donnees = {"date_col": "date", "value_col": "sales", "store_col": "store_nbr",
               "category_col": "family", "horizon": "5", **champs}
    start = client.post(
        "/api/upload/forecast/bulk/start",
        files={"file": ("ventes.csv", csv_bytes, "text/csv")},
        data=donnees,
    )
    assert start.status_code == 200, start.text
    job_id = start.json()["job_id"]

    for _ in range(600):
        progress = client.get(f"/api/upload/forecast/bulk/progress/{job_id}").json()
        if progress["status"] != "running":
            break
        time.sleep(0.1)
    return job_id, progress


def test_bulk_sarimax_traverse_le_pool_et_produit_un_csv():
    raw = _frame_entites(60, [("1", "A"), ("2", "B")])
    job_id, progress = _lancer_bulk(raw.to_csv(index=False).encode(), model="sarimax", include_ci="true")
    assert progress["status"] == "done", progress

    download = client.get(f"/api/upload/forecast/bulk/download/{job_id}")
    assert download.status_code == 200
    df = pd.read_csv(io.StringIO(download.text))
    assert len(df) == 2 * 5
    # contrairement à lightgbm, SARIMAX fournit un intervalle de confiance
    assert {"ic_bas", "ic_haut"} <= set(df.columns)
    assert set(df["store_nbr"].astype(str)) == {"1", "2"}


def test_bulk_sarimax_ne_melange_pas_les_entites():
    """Chaque entité part dans un process distinct : une erreur d'appariement
    entre le future rendu et son combo passerait inaperçue sans ça."""
    raw = _frame_entites(60, [("1", "A"), ("2", "B")])
    job_id, progress = _lancer_bulk(raw.to_csv(index=False).encode(), model="sarimax")
    assert progress["status"] == "done", progress

    df = pd.read_csv(io.StringIO(client.get(f"/api/upload/forecast/bulk/download/{job_id}").text))
    par_entite = df.dropna(subset=["prevision"]).groupby("store_nbr")["prevision"].mean()
    if len(par_entite) == 2:  # les deux ont convergé
        # l'entité "2" est construite 40 unités au-dessus de l'entité "1"
        assert par_entite.loc[2] > par_entite.loc[1]


def test_bulk_sarimax_ignore_les_entites_trop_courtes():
    raw = pd.concat([
        _frame_entites(60, [("1", "A")]),
        _frame_entites(10, [("2", "B")]),  # < 21 jours : doit être ignorée
    ])
    job_id, progress = _lancer_bulk(raw.to_csv(index=False).encode(), model="sarimax")
    assert progress["status"] == "done", progress
    assert progress["skipped"] == 1
    assert progress["total"] == 2

    df = pd.read_csv(io.StringIO(client.get(f"/api/upload/forecast/bulk/download/{job_id}").text))
    assert set(df["store_nbr"].astype(str)) == {"1"}


def test_bulk_sarimax_draine_au_dela_du_plafond(monkeypatch):
    """Avec plus d'entités que MAX_BULK_EN_VOL, le drainage en cours de
    soumission entre en jeu — chemin qu'aucun test à deux entités ne
    touche. Plafond abaissé pour ne pas fabriquer 64 entités."""
    monkeypatch.setattr(api, "MAX_BULK_EN_VOL", 2)
    entites = [(str(i), "A") for i in range(1, 6)]  # 5 entités > plafond de 2
    raw = _frame_entites(45, entites)

    job_id, progress = _lancer_bulk(raw.to_csv(index=False).encode(), model="sarimax")
    assert progress["status"] == "done", progress
    assert progress["total"] == 5
    assert progress["processed"] == 5  # aucune entité perdue par le drainage

    df = pd.read_csv(io.StringIO(client.get(f"/api/upload/forecast/bulk/download/{job_id}").text))
    assert set(df["store_nbr"].astype(str)) == {"1", "2", "3", "4", "5"}
    assert len(df) == 5 * 5  # 5 entités x horizon 5, aucun doublon


# --- Cookie de session : Secure conditionnel --------------------------

def test_cookie_https_only_lit_la_variable(monkeypatch):
    monkeypatch.delenv("CADENCE_HTTPS", raising=False)
    assert api.cookie_https_only() is False
    for valeur in ("1", "true", "YES", " true "):
        monkeypatch.setenv("CADENCE_HTTPS", valeur)
        assert api.cookie_https_only() is True, valeur
    for valeur in ("0", "", "non"):
        monkeypatch.setenv("CADENCE_HTTPS", valeur)
        assert api.cookie_https_only() is False, valeur


def test_cookie_de_session_non_secure_par_defaut():
    """En local l'app est servie en http:// : un cookie Secure ne serait
    jamais renvoyé par le navigateur et la connexion serait cassée. C'est
    la raison pour laquelle https_only n'est pas activé d'office."""
    # le compte est déjà créé par la fixture _session : on se connecte,
    # ce qui pose le même cookie de session
    anonyme = TestClient(app)
    resp = anonyme.post("/api/login", data={"username": "testeur", "password": "motdepasse"})
    assert resp.status_code == 200, resp.text
    # Starlette écrit les attributs en minuscules ("httponly; samesite=lax") :
    # comparer sans la casse, sinon l'absence de "Secure" serait vérifiée par
    # accident et un "secure" bien présent passerait inaperçu.
    entete = resp.headers.get("set-cookie", "").lower()
    assert "session=" in entete
    assert "secure" not in entete
    assert "httponly" in entete  # posé d'office par Starlette
    assert "samesite=lax" in entete  # défaut conservé : couvre l'essentiel du CSRF
