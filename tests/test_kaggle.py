import gzip
import io
import time
import zipfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import kaggle_competition as kc
from api import app
from conftest import connecter


def make_competition(n_days=120, test_days=14, gap=0):
    """Jeu synthétique au format *Store Sales* : 3 magasins × 2 familles,
    rythme hebdomadaire, une exogène connue à l'avance (promo)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for store in (1, 2, 3):
        for family in ("A", "B"):
            for d in dates:
                promo = int(rng.random() < 0.2)
                sales = 10 * store + (5 if d.dayofweek >= 5 else 0) + 8 * promo + rng.random()
                rows.append({"date": d.strftime("%Y-%m-%d"), "store_nbr": store, "family": family,
                             "onpromotion": promo, "sales": round(sales, 2)})
    train = pd.DataFrame(rows)
    test_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1 + gap), periods=test_days, freq="D")
    test = pd.DataFrame([
        {"id": i, "date": d.strftime("%Y-%m-%d"), "store_nbr": s, "family": f, "onpromotion": 0}
        for i, (d, s, f) in enumerate((d, s, f) for d in test_dates for s in (1, 2, 3) for f in ("A", "B"))
    ])
    return train, test


def spec_store_sales(**kw):
    return kc.CompetitionSpec(date_col="date", target_col="sales", entity_cols=["store_nbr", "family"],
                              id_col="id", **kw)


# --- Correspondance de colonnes -----------------------------------------

def test_suggestion_format_store_sales():
    train, test = make_competition()
    sug = kc.suggest_mapping(train, test)
    assert sug["date_col"] == "date"
    assert sug["target_col"] == "sales"
    assert sug["id_col"] == "id"
    assert sug["entity_cols"] == ["store_nbr", "family"]
    # entier à peu de valeurs, mais pas une clé : exogène, pas une entité
    assert sug["exog_cols"] == ["onpromotion"]
    assert sug["composite_sep"] is None


def test_suggestion_identifiant_composite_recruit():
    train = pd.DataFrame({
        "air_store_id": ["air_a1"] * 30 + ["air_b2"] * 30,
        "visit_date": [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-01-01", periods=30)] * 2,
        "visitors": range(60),
    })
    test = pd.DataFrame({"id": ["air_a1_2024-01-31", "air_b2_2024-01-31"], "visitors": [0, 0]})
    sug = kc.suggest_mapping(train, test)
    assert sug["composite_sep"] == "_"
    assert sug["entity_cols"] == ["air_store_id"]
    assert sug["date_col"] == "visit_date"
    assert sug["target_col"] == "visitors"


def test_cles_identifiant_composite_coupe_au_dernier_separateur():
    """L'identifiant d'entité contient lui-même le séparateur (air_xxx)."""
    spec = kc.CompetitionSpec(date_col="visit_date", target_col="visitors", entity_cols=["air_store_id"],
                              id_col="id", composite_sep="_")
    keys = kc.submission_keys(pd.DataFrame({"id": ["air_00a9_2017-04-23"]}), spec)
    assert keys.loc[0, "air_store_id"] == "air_00a9"
    assert keys.loc[0, "date"] == pd.Timestamp("2017-04-23")


def test_composite_refuse_deux_entites():
    with pytest.raises(ValueError):
        kc.CompetitionSpec(date_col="d", target_col="y", entity_cols=["a", "b"], id_col="id", composite_sep="_")


# --- Horizon, backtest, repli --------------------------------------------

def test_horizon_compte_l_ecart_entre_train_et_test():
    train, test = make_competition(test_days=10, gap=3)
    spec = spec_store_sales()
    tr = kc.prepare_train(train, spec)
    horizon, scored = kc.horizon_days(tr, spec, kc.submission_keys(test, spec))
    assert (horizon, scored) == (13, 10)


def test_test_qui_chevauche_le_train_rejete():
    train, test = make_competition()
    test["date"] = train["date"].iloc[-1]
    spec = spec_store_sales()
    with pytest.raises(ValueError, match="chevauchent"):
        kc.horizon_days(kc.prepare_train(train, spec), spec, kc.submission_keys(test, spec))


def test_backtest_rejoue_l_ecart_sans_le_noter():
    train, _ = make_competition(n_days=100)
    spec = spec_store_sales()
    tr = kc.prepare_train(train, spec)
    fit, truth = kc.backtest_split(tr, spec, horizon=13, scored_days=10)
    end = tr["date"].max()
    assert fit["date"].max() == end - pd.Timedelta(days=13)
    assert truth["date"].min() == end - pd.Timedelta(days=9)
    assert len(truth) == 10 * 6


def test_repli_sur_moyenne_recente_de_l_entite():
    spec = kc.CompetitionSpec(date_col="date", target_col="y", entity_cols=["e"], id_col="id")
    history = pd.DataFrame({"e": ["a"] * 20, "date": pd.date_range("2024-01-01", periods=20), "y": [5.0] * 20})
    targets = pd.DataFrame({"e": ["a", "inconnue"], "date": pd.to_datetime(["2024-01-21"] * 2)})
    values, n_fallback = kc.attach_predictions(targets, kc.predictions_frame([], spec), history, spec, True)
    assert n_fallback == 2
    assert list(values) == [5.0, 0.0]


def test_entite_prolongee_jusqu_a_la_fin_du_train():
    spec = kc.CompetitionSpec(date_col="date", target_col="y", entity_cols=["e"], id_col="id")
    train = pd.DataFrame({"e": ["a", "a", "b"], "date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-05"]),
                          "y": [1, 1, 1]})
    padded = kc.pad_to(train, spec, pd.Timestamp("2024-01-10"))
    assert padded.groupby("e")["date"].max().eq(pd.Timestamp("2024-01-10")).all()


def test_metriques():
    m = kc.compute_metrics(np.array([0, 10]), np.array([0, 10]))
    assert m == {"rmsle": 0.0, "rmse": 0.0, "mae": 0.0, "smape": 0.0}
    m = kc.compute_metrics(np.array([10.0]), np.array([20.0]))
    assert m["mae"] == 10 and m["rmse"] == 10
    assert m["smape"] == pytest.approx(100 * 2 * 10 / 30)


def test_exogenes_futures_lues_dans_le_test():
    train, test = make_competition(test_days=3)
    test["onpromotion"] = 7
    spec = spec_store_sales(exog_cols=["onpromotion"])
    tr = kc.prepare_train(train, spec)
    keys = kc.submission_keys(test, spec)
    long = kc.exog_long_frame(tr, test, keys, spec)
    assert (long.loc[long["date"] > tr["date"].max(), "onpromotion"] == 7).all()


# --- Archives -------------------------------------------------------------

def test_decompression_zip_et_gz():
    csv = b"a,b\n1,2\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("train.csv", csv)
    assert kc.decompress_upload(buf.getvalue(), "train.csv.zip") == csv
    assert kc.decompress_upload(gzip.compress(csv), "train.csv.gz") == csv
    assert kc.decompress_upload(csv, "train.csv") == csv


def test_zip_a_plusieurs_csv_choisit_celui_de_l_archive():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("train.csv", b"a\n1\n")
        zf.writestr("test.csv", b"a\n2\n")
    assert kc.decompress_upload(buf.getvalue(), "test.csv.zip") == b"a\n2\n"
    with pytest.raises(ValueError):
        kc.decompress_upload(buf.getvalue(), "competition.zip")


# --- API --------------------------------------------------------------------

def _files(train, test):
    return {
        "train_file": ("train.csv.gz", gzip.compress(train.to_csv(index=False).encode()), "application/gzip"),
        "test_file": ("test.csv", test.to_csv(index=False).encode(), "text/csv"),
    }


def _attendre(client, job_id):
    # 2 min : large, pour rester vert sur une machine chargée (benchmark en parallèle)
    for _ in range(1200):
        progress = client.get(f"/api/kaggle/progress/{job_id}").json()
        if progress["status"] != "running":
            return progress
        time.sleep(0.1)
    raise AssertionError("job Kaggle jamais terminé")


MAPPING = {"date_col": "date", "target_col": "sales", "entity_cols": "store_nbr,family", "id_col": "id",
           "exog_cols": "onpromotion", "model": "lightgbm", "competition": "store-sales"}


def test_parcours_kaggle_backtest_puis_soumission():
    client = TestClient(app)
    connecter(client)
    train, test = make_competition()

    resp = client.post("/api/kaggle/inspect", files=_files(train, test))
    assert resp.status_code == 200, resp.text
    assert resp.json()["suggestion"]["entity_cols"] == ["store_nbr", "family"]

    resp = client.post("/api/kaggle/start", files=_files(train, test), data={**MAPPING, "mode": "backtest"})
    assert resp.status_code == 200, resp.text
    progress = _attendre(client, resp.json()["job_id"])
    assert progress["status"] == "done", progress["error"]
    scores = progress["result"]["scores"]
    assert progress["result"]["scored_days"] == 14
    # le rythme synthétique est simple : le modèle doit battre le naïf
    assert scores["model"]["rmsle"] < scores["naive"]["rmsle"]

    resp = client.post("/api/kaggle/start", files=_files(train, test), data={**MAPPING, "mode": "submission"})
    job_id = resp.json()["job_id"]
    progress = _attendre(client, job_id)
    assert progress["status"] == "done", progress["error"]
    sub = pd.read_csv(io.BytesIO(client.get(f"/api/kaggle/download/{job_id}").content))
    assert list(sub.columns) == ["id", "sales"]
    assert sorted(sub["id"]) == sorted(test["id"])
    assert sub["sales"].notna().all() and (sub["sales"] >= 0).all()

    runs = client.get("/api/kaggle/runs", params={"competition": "store-sales"}).json()
    assert [r["mode"] for r in runs] == ["submission", "backtest"]
    assert client.get("/api/kaggle/competitions").json() == ["store-sales"]

    run_id = runs[0]["id"]
    resp = client.patch(f"/api/kaggle/runs/{run_id}", data={"kaggle_public": "0,4321", "note": "1er essai"})
    assert resp.status_code == 200
    assert resp.json()["kaggle_public"] == pytest.approx(0.4321)
    assert client.delete(f"/api/kaggle/runs/{run_id}").status_code == 200
    assert len(client.get("/api/kaggle/runs").json()) == 1


def test_job_et_essais_isoles_par_compte():
    import storage

    client = TestClient(app)
    connecter(client)
    train, test = make_competition(n_days=60, test_days=7)
    job_id = client.post("/api/kaggle/start", files=_files(train, test),
                         data={**MAPPING, "mode": "backtest"}).json()["job_id"]
    _attendre(client, job_id)
    run_id = client.get("/api/kaggle/runs").json()[0]["id"]

    storage.create_user("autre", "motdepasse")
    autre = TestClient(app)
    autre.post("/api/login", data={"username": "autre", "password": "motdepasse"})
    assert autre.get(f"/api/kaggle/progress/{job_id}").status_code == 404
    assert autre.get("/api/kaggle/runs").json() == []
    assert autre.patch(f"/api/kaggle/runs/{run_id}", data={"note": "x"}).status_code == 404
    assert autre.delete(f"/api/kaggle/runs/{run_id}").status_code == 404


def test_mode_et_entites_invalides_rejetes():
    client = TestClient(app)
    connecter(client)
    train, test = make_competition(n_days=30, test_days=3)
    resp = client.post("/api/kaggle/start", files=_files(train, test), data={**MAPPING, "mode": "nawak"})
    assert resp.status_code == 400
    resp = client.post("/api/kaggle/start", files=_files(train, test),
                       data={**MAPPING, "mode": "backtest", "entity_cols": "store_nbr,family,date"})
    assert resp.status_code == 400


def test_kaggle_exige_une_session():
    train, test = make_competition(n_days=30, test_days=3)
    assert TestClient(app).post("/api/kaggle/inspect", files=_files(train, test)).status_code == 401


# --- Pistes de précision -----------------------------------------------------

def test_moyenne_ponderee_ignore_le_membre_qui_n_a_pas_prevu():
    spec = kc.CompetitionSpec(date_col="date", target_col="y", entity_cols=["e"], id_col="id")
    d = pd.Timestamp("2024-01-01")
    a = pd.DataFrame({"e": ["x", "y"], "date": [d, d], "prevision": [10.0, 4.0]})
    b = pd.DataFrame({"e": ["x"], "date": [d], "prevision": [20.0]})
    out = kc.blend_predictions([(a, 0.75), (b, 0.25)], spec).set_index("e")["prevision"]
    assert out["x"] == pytest.approx(12.5)
    assert out["y"] == pytest.approx(4.0)  # seul `a` l'a prévue


def test_series_eteintes_prevues_a_zero():
    spec = kc.CompetitionSpec(date_col="date", target_col="y", entity_cols=["e"], id_col="id")
    dates = pd.date_range("2024-01-01", periods=30)
    history = pd.DataFrame({
        "e": ["morte"] * 30 + ["vivante"] * 30,
        "date": list(dates) * 2,
        "y": [5.0] * 10 + [0.0] * 20 + [0.0] * 29 + [3.0],
    })
    future = pd.Timestamp("2024-01-31")
    preds = pd.DataFrame({"e": ["morte", "vivante"], "date": [future] * 2, "prevision": [1.2, 2.5]})
    out = kc.zero_dead_series(preds, history, spec, 14).set_index("e")["prevision"]
    assert out["morte"] == 0.0
    assert out["vivante"] == 2.5  # une vente récente suffit à la garder en vie


def test_series_eteintes_actives_par_defaut_et_desactivables():
    client = TestClient(app)
    connecter(client)
    train, test = make_competition(n_days=60, test_days=7)
    for flag in (None, "false"):
        # nom propre : un job d'un test précédent peut finir dans cette base
        data = {**MAPPING, "mode": "backtest", "competition": "defaut-eteintes"}
        if flag:
            data["zero_dead_series"] = flag
        _attendre(client, client.post("/api/kaggle/start", files=_files(train, test), data=data).json()["job_id"])
    sans, avec = client.get("/api/kaggle/runs", params={"competition": "defaut-eteintes"}).json()
    assert avec["config"]["options"] == {"zero_dead_days": 14}
    assert sans["config"]["options"] == {}
