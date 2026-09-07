"""Persistance locale (SQLite) des points de vente : permet de sauvegarder
un CSV importé une fois, puis de revenir consulter son analyse/prévision
sans le ré-importer à chaque visite. Comptes utilisateurs légers (identifiant
+ mot de passe, pas d'email) : chaque point de vente appartient à un compte,
qui ne voit que les siens.
"""

import hashlib
import hmac
import json
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cadence.db"
PBKDF2_ITERATIONS = 200_000


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
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS points_de_vente (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id INTEGER REFERENCES users(id),
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
        # Nettoyage : les bases créées pendant l'expérimentation "verticaux"
        # (abandonnée, cf. discussion produit) ont cette colonne en trop.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(points_de_vente)")}
        if "vertical" in columns:
            conn.execute("ALTER TABLE points_de_vente DROP COLUMN vertical")
        # Comptes introduits après coup (07/09/2026) : les bases existantes
        # n'ont pas encore la colonne owner_id, ajoutée nullable — les points
        # de vente déjà présents sont rattachés au premier compte créé,
        # cf. claim_orphan_points (appelé par le flux /setup).
        if "owner_id" not in columns:
            conn.execute("ALTER TABLE points_de_vente ADD COLUMN owner_id INTEGER REFERENCES users(id)")


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def has_any_user() -> bool:
    with _connect() as conn:
        return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def create_user(username: str, password: str) -> int:
    salt_hex, hash_hex = _hash_password(password)
    with _connect() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, password_salt) VALUES (?, ?, ?)",
                (username, hash_hex, salt_hex),
            )
        except sqlite3.IntegrityError:
            raise ValueError("Nom d'utilisateur déjà pris")
        return cursor.lastrowid


def verify_user(username: str, password: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return None
    _, hash_hex = _hash_password(password, bytes.fromhex(row["password_salt"]))
    if not hmac.compare_digest(hash_hex, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"]}


def get_user(user_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    return {"id": row["id"], "username": row["username"]} if row else None


def claim_orphan_points(owner_id: int) -> int:
    """Rattache au compte `owner_id` les points de vente créés avant
    l'introduction des comptes (owner_id NULL) — appelé une fois, à la
    création du premier compte via /setup.
    """
    with _connect() as conn:
        cursor = conn.execute("UPDATE points_de_vente SET owner_id = ? WHERE owner_id IS NULL", (owner_id,))
        return cursor.rowcount


def save_point(
    owner_id: int,
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
            INSERT INTO points_de_vente (owner_id, name, location, use_ferie, use_vacances, use_meteo, series_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (owner_id, name, location, int(use_ferie), int(use_vacances), int(use_meteo), series_json),
        )
        return cursor.lastrowid


def list_points(owner_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, location, created_at, series_json FROM points_de_vente"
            " WHERE owner_id = ? ORDER BY created_at DESC",
            (owner_id,),
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


def load_point(owner_id: int, point_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM points_de_vente WHERE id = ? AND owner_id = ?", (point_id, owner_id)
        ).fetchone()
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


def delete_point(owner_id: int, point_id: int) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM points_de_vente WHERE id = ? AND owner_id = ?", (point_id, owner_id)
        )
        return cursor.rowcount > 0


def session_secret() -> str:
    """Clé de signature des cookies de session — générée une fois puis
    conservée à côté de la base, pour ne pas déconnecter tout le monde à
    chaque redémarrage du serveur.
    """
    path = DB_PATH.parent / ".session_secret"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_hex(32))
    return path.read_text().strip()
