import pandas as pd

import storage


def _sample_df(n_days=10):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({"visitors": list(range(n_days))}, index=dates)
    df.index.name = "visit_date"
    return df


def test_save_list_load_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    storage.init_db()

    df = _sample_df()
    point_id = storage.save_point("Mon Bistrot", df, "Lyon", True, False, True)

    points = storage.list_points()
    assert len(points) == 1
    assert points[0]["name"] == "Mon Bistrot"
    assert points[0]["location"] == "Lyon"
    assert points[0]["n_days"] == 10

    loaded = storage.load_point(point_id)
    assert loaded["name"] == "Mon Bistrot"
    assert loaded["location"] == "Lyon"
    assert loaded["use_ferie"] is True
    assert loaded["use_vacances"] is False
    assert loaded["use_meteo"] is True
    assert len(loaded["df"]) == 10
    assert list(loaded["df"]["visitors"]) == list(range(10))

    assert storage.delete_point(point_id) is True
    assert storage.list_points() == []


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

    point_id = storage.save_point("Legacy", _sample_df(), None, True, True, True)
    assert storage.load_point(point_id)["name"] == "Legacy"


def test_load_missing_point_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    storage.init_db()
    assert storage.load_point(9999) is None


def test_delete_missing_point_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    storage.init_db()
    assert storage.delete_point(9999) is False


def test_save_without_location(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    storage.init_db()

    point_id = storage.save_point("Sans localisation", _sample_df(), None, False, False, False)
    loaded = storage.load_point(point_id)
    assert loaded["location"] is None
    assert loaded["use_ferie"] is False
