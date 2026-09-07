import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(autouse=True)
def base_isolee(tmp_path, monkeypatch):
    """Redirige la base SQLite vers un fichier temporaire, propre à chaque
    test. Sans ça, tout test créant un compte ou un point de vente écrirait
    dans le data/cadence.db réel de l'utilisateur."""
    import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "cadence.db")
    # Le schéma est créé à l'import du module (donc sur la vraie base) :
    # le fichier temporaire, lui, part vide et doit être initialisé ici.
    storage.init_db()


def connecter(client, username="testeur", password="motdepasse"):
    """Ouvre une session sur ce client : les endpoints de calcul exigent
    une authentification. La base étant vide à chaque test (base_isolee),
    /api/setup crée le premier compte et connecte dans la foulée."""
    resp = client.post("/api/setup", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp
