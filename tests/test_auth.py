import json
import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import storage
from api import app


def make_csv_bytes(rows):
    lines = ["date,value"] + [f"{d},{v}" for d, v in rows]
    return "\n".join(lines).encode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    storage.init_db()
    return TestClient(app)


def test_me_reports_needs_setup_when_no_account_exists(client):
    assert client.get("/api/me").json() == {"logged_in": False, "needs_setup": True}


def test_points_requires_login(client):
    assert client.get("/api/points").status_code == 401


def test_setup_creates_account_and_logs_in(client):
    resp = client.post("/api/setup", data={"username": "admin", "password": "hunter2"})
    assert resp.status_code == 200
    assert client.get("/api/me").json() == {"logged_in": True, "username": "admin"}


def test_setup_refused_once_an_account_exists(client):
    client.post("/api/setup", data={"username": "admin", "password": "hunter2"})
    resp = client.post("/api/setup", data={"username": "bob", "password": "hunter2"})
    assert resp.status_code == 403


def test_setup_claims_points_created_before_accounts_existed(client):
    # simule un point de vente créé avant l'introduction des comptes (owner_id NULL)
    dates = pd.date_range("2024-01-01", periods=21)
    series_json = json.dumps({"dates": [d.date().isoformat() for d in dates], "visitors": list(range(21))})
    conn = sqlite3.connect(storage.DB_PATH)
    conn.execute("INSERT INTO points_de_vente (name, series_json) VALUES (?, ?)", ("Ancien", series_json))
    conn.commit()
    conn.close()

    client.post("/api/setup", data={"username": "admin", "password": "hunter2"})

    resp = client.get("/api/points")
    assert resp.status_code == 200
    assert [p["name"] for p in resp.json()] == ["Ancien"]


def test_login_logout_flow(client):
    client.post("/api/setup", data={"username": "admin", "password": "hunter2"})
    client.post("/api/logout")
    assert client.get("/api/me").json()["logged_in"] is False

    resp = client.post("/api/login", data={"username": "admin", "password": "hunter2"})
    assert resp.status_code == 200
    assert client.get("/api/me").json() == {"logged_in": True, "username": "admin"}


def test_login_rejects_wrong_password(client):
    client.post("/api/setup", data={"username": "admin", "password": "hunter2"})
    client.post("/api/logout")
    resp = client.post("/api/login", data={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401
    assert client.get("/api/me").json()["logged_in"] is False


def test_points_isolated_between_two_accounts(client):
    client.post("/api/setup", data={"username": "alice", "password": "hunter2"})
    rows = [(f"2024-01-{d:02d}", 10 + d) for d in range(1, 22)]
    resp = client.post(
        "/api/points",
        files={"file": ("data.csv", make_csv_bytes(rows), "text/csv")},
        data={"date_col": "date", "value_col": "value", "name": "Chez Alice"},
    )
    assert resp.status_code == 200
    point_id = resp.json()["id"]

    # bob est créé "hors bande" (script create_user.py en usage réel)
    storage.create_user("bob", "hunter2")
    client.post("/api/logout")
    client.post("/api/login", data={"username": "bob", "password": "hunter2"})

    assert client.get("/api/points").json() == []
    assert client.get(f"/api/points/{point_id}/analysis").status_code == 404
    assert client.delete(f"/api/points/{point_id}").status_code == 404
