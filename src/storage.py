"""Persistance locale (SQLite) des points de vente : permet de sauvegarder
un CSV importé une fois, puis de revenir consulter son analyse/prévision
sans le ré-importer à chaque visite. Pas de compte utilisateur pour
l'instant — une seule base locale, partagée par tous les points de vente.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cadence.db"


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS points_de_vente (
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


def save_point(
    name: str,
    df: pd.DataFrame,
    location: str | None,
    use_ferie: bool,
    use_vacances: bool,
    use_meteo: bool,
) -> int:
    series_json = json.dumps(
        {
            "dates": [d.date().isoformat() for d in df.index],
            "visitors": df["visitors"].tolist(),
        }
    )
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO points_de_vente (name, location, use_ferie, use_vacances, use_meteo, series_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, location, int(use_ferie), int(use_vacances), int(use_meteo), series_json),
        )
        return cursor.lastrowid


def list_points() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, location, created_at, series_json FROM points_de_vente ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "location": row["location"],
            "created_at": row["created_at"],
            "n_days": len(json.loads(row["series_json"])["dates"]),
        }
        for row in rows
    ]


def load_point(point_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM points_de_vente WHERE id = ?", (point_id,)).fetchone()
    if row is None:
        return None

    series = json.loads(row["series_json"])
    df = pd.DataFrame({"visitors": series["visitors"]}, index=pd.to_datetime(series["dates"]))
    df.index.name = "visit_date"

    return {
        "id": row["id"],
        "name": row["name"],
        "location": row["location"],
        "use_ferie": bool(row["use_ferie"]),
        "use_vacances": bool(row["use_vacances"]),
        "use_meteo": bool(row["use_meteo"]),
        "df": df,
    }


def delete_point(point_id: int) -> bool:
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM points_de_vente WHERE id = ?", (point_id,))
        return cursor.rowcount > 0
