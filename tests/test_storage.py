import pandas as pd
import pytest

import storage


def _sample_df(n_days=10):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({"visitors": list(range(n_days))}, index=dates)
    df.index.name = "visit_date"
    return df


def _init(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    storage.init_db()


def test_save_list_load_delete(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    owner_id = storage.create_user("alice", "hunter2")

    df = _sample_df()
    point_id = storage.save_point(owner_id, "Mon Bistrot", df, "Lyon", True, False, True)

    points = storage.list_points(owner_id)
    assert len(points) == 1
    assert points[0]["name"] == "Mon Bistrot"
    assert points[0]["location"] == "Lyon"
    assert points[0]["n_days"] == 10

    loaded = storage.load_point(owner_id, point_id)
    assert loaded["name"] == "Mon Bistrot"
    assert loaded["location"] == "Lyon"
    assert loaded["use_ferie"] is True
    assert loaded["use_vacances"] is False
    assert loaded["use_meteo"] is True
    assert len(loaded["df"]) == 10
    assert list(loaded["df"]["visitors"]) == list(range(10))

    assert storage.delete_point(owner_id, point_id) is True
    assert storage.list_points(owner_id) == []


def test_migration_drops_vertical_column_from_existing_db(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(storage, "DB_PATH", db_path)

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE points_de_vente (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT,
            vertical TEXT NOT NULL DEFAULT 'flux_physique',
            use_ferie INTEGER NOT NULL DEFAULT 1,
            use_vacances INTEGER NOT NULL DEFAULT 1,
            use_meteo INTEGER NOT NULL DEFAULT 1,
            series_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    conn.close()

    storage.init_db()  # doit retirer la colonne héritée de l'expérimentation "verticaux"
    columns = {row[1] for row in sqlite3.connect(db_path).execute("PRAGMA table_info(points_de_vente)")}
    assert "vertical" not in columns

    owner_id = storage.create_user("bob", "hunter2")
    point_id = storage.save_point(owner_id, "Legacy", _sample_df(), None, True, True, True)
    assert storage.load_point(owner_id, point_id)["name"] == "Legacy"


def test_migration_adds_owner_id_and_orphans_stay_unclaimed(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy2.db"
    monkeypatch.setattr(storage, "DB_PATH", db_path)

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE points_de_vente (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT,
            use_ferie INTEGER NOT NULL DEFAULT 1,
            use_vacances INTEGER NOT NULL DEFAULT 1,
            use_meteo INTEGER NOT NULL DEFAULT 1,
            series_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "INSERT INTO points_de_vente (name, series_json) VALUES ('Ancien', '{\"dates\": [], \"visitors\": []}')"
    )
    conn.commit()
    conn.close()

    storage.init_db()
    columns = {row[1] for row in sqlite3.connect(db_path).execute("PRAGMA table_info(points_de_vente)")}
    assert "owner_id" in columns

    owner_id = storage.create_user("admin", "hunter2")
    assert storage.list_points(owner_id) == []  # pas encore rattaché

    claimed = storage.claim_orphan_points(owner_id)
    assert claimed == 1
    assert len(storage.list_points(owner_id)) == 1


def test_load_missing_point_returns_none(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    owner_id = storage.create_user("alice", "hunter2")
    assert storage.load_point(owner_id, 9999) is None


def test_delete_missing_point_returns_false(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    owner_id = storage.create_user("alice", "hunter2")
    assert storage.delete_point(owner_id, 9999) is False


def test_save_without_location(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    owner_id = storage.create_user("alice", "hunter2")

    point_id = storage.save_point(owner_id, "Sans localisation", _sample_df(), None, False, False, False)
    loaded = storage.load_point(owner_id, point_id)
    assert loaded["location"] is None
    assert loaded["use_ferie"] is False


def test_point_isolated_between_owners(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    alice = storage.create_user("alice", "hunter2")
    bob = storage.create_user("bob", "hunter2")

    point_id = storage.save_point(alice, "Bistrot d'Alice", _sample_df(), None, True, True, True)

    assert storage.load_point(bob, point_id) is None
    assert storage.list_points(bob) == []
    assert storage.delete_point(bob, point_id) is False
    assert storage.load_point(alice, point_id) is not None


def test_create_user_rejects_duplicate_username(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    storage.create_user("alice", "hunter2")
    with pytest.raises(ValueError):
        storage.create_user("alice", "autre-mot-de-passe")


def test_verify_user_checks_password(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    storage.create_user("alice", "hunter2")

    assert storage.verify_user("alice", "hunter2") is not None
    assert storage.verify_user("alice", "mauvais-mot-de-passe") is None
    assert storage.verify_user("inconnue", "hunter2") is None


def test_has_any_user(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    assert storage.has_any_user() is False
    storage.create_user("alice", "hunter2")
    assert storage.has_any_user() is True
