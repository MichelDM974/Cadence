"""CLI pour créer un compte Cadence. Pas de signup public (MVP B2B, choix
du 07/09/2026) : c'est l'opérateur qui crée les comptes
de ses restaurateurs avec ce script. Le tout premier compte se crée plutôt
via l'écran /setup de l'app (il récupère aussi les points de vente créés
avant l'introduction des comptes).
"""

import argparse
import getpass

import storage


def main() -> None:
    parser = argparse.ArgumentParser(description="Crée un compte Cadence")
    parser.add_argument("username")
    args = parser.parse_args()

    password = getpass.getpass("Mot de passe : ")
    if password != getpass.getpass("Confirmer : "):
        raise SystemExit("Les mots de passe ne correspondent pas")

    storage.init_db()
    user_id = storage.create_user(args.username, password)
    print(f"Compte '{args.username}' créé (id={user_id})")


if __name__ == "__main__":
    main()
