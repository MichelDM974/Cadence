#!/usr/bin/env python3
"""Driver de Cadence : lance l'app sur une base jetable, la pilote, la
photographie.

Trois sous-commandes, à lancer depuis la racine du projet avec .venv/bin/python :

    serve   démarre l'app sur une base isolée (bloquant, Ctrl-C pour arrêter)
    smoke   vérifie l'API de bout en bout (création de compte incluse)
    ui      pilote l'interface dans chromium et écrit des captures

`serve` n'écrit jamais dans data/cadence.db : sans ça, le moindre essai
crée des comptes et des points de vente dans les vraies données.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

RACINE = pathlib.Path(__file__).resolve().parents[3]
SRC = RACINE / "src"
PORT_DEFAUT = 8011
IDENTIFIANT, MOTDEPASSE = "driver", "driver-motdepasse"


# --- serve -----------------------------------------------------------

def commande_serve(args):
    """Lance uvicorn après avoir détourné storage.DB_PATH vers un dossier
    jetable. Le détournement doit précéder l'import de api, qui appelle
    storage.session_secret() au chargement du module."""
    sys.path.insert(0, str(SRC))
    import storage

    data = pathlib.Path(args.data).resolve()
    data.mkdir(parents=True, exist_ok=True)
    storage.DB_PATH = data / "cadence.db"
    storage.init_db()

    import uvicorn
    from api import app

    print(f"Cadence sur http://127.0.0.1:{args.port} (base : {storage.DB_PATH})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


# --- petit client HTTP (pas de dépendance hors stdlib) ----------------

class Client:
    """Conserve le cookie de session entre les appels : depuis l'ajout des
    comptes, tous les endpoints de calcul renvoient 401 sans session."""

    def __init__(self, base):
        self.base = base
        self.cookies = {}

    def _requete(self, methode, chemin, donnees=None, fichiers=None):
        url = self.base + chemin
        corps, entetes = None, {}
        if fichiers:
            corps, entetes["Content-Type"] = _multipart(donnees or {}, fichiers)
        elif donnees is not None:
            corps = urllib.parse.urlencode(donnees).encode()
            entetes["Content-Type"] = "application/x-www-form-urlencoded"
        if self.cookies:
            entetes["Cookie"] = "; ".join(f"{k}={v}" for k, v in self.cookies.items())

        req = urllib.request.Request(url, data=corps, headers=entetes, method=methode)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                self._memorise_cookies(resp)
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            self._memorise_cookies(e)
            return e.code, e.read()

    def _memorise_cookies(self, resp):
        for entete in resp.headers.get_all("Set-Cookie") or []:
            paire = entete.split(";", 1)[0]
            if "=" in paire:
                cle, valeur = paire.split("=", 1)
                self.cookies[cle] = valeur

    def get(self, chemin):
        return self._requete("GET", chemin)

    def post(self, chemin, donnees=None, fichiers=None):
        return self._requete("POST", chemin, donnees, fichiers)


def _multipart(champs, fichiers):
    limite = "----cadencedriver"
    morceaux = []
    for cle, valeur in champs.items():
        morceaux.append(
            f"--{limite}\r\nContent-Disposition: form-data; name=\"{cle}\"\r\n\r\n{valeur}\r\n".encode()
        )
    for cle, (nom, contenu) in fichiers.items():
        morceaux.append(
            f"--{limite}\r\nContent-Disposition: form-data; name=\"{cle}\"; filename=\"{nom}\"\r\n"
            f"Content-Type: text/csv\r\n\r\n".encode() + contenu + b"\r\n"
        )
    morceaux.append(f"--{limite}--\r\n".encode())
    return b"".join(morceaux), f"multipart/form-data; boundary={limite}"


def attendre(base, secondes=60):
    for _ in range(secondes * 2):
        try:
            urllib.request.urlopen(base + "/api/me", timeout=2).read()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def csv_demo(jours=140):
    """Série avec une saisonnalité hebdomadaire et un peu de bruit
    déterministe : une série parfaitement lisse fait diverger l'ajustement
    SARIMAX (cf. Pièges dans SKILL.md)."""
    lignes = ["date,value"]
    depart = time.strptime("2024-01-01", "%Y-%m-%d")
    base = time.mktime(depart)
    for i in range(jours):
        jour = time.localtime(base + i * 86400)
        weekend = jour.tm_wday >= 5
        lignes.append(f"{time.strftime('%Y-%m-%d', jour)},{100 + 20 * weekend + (i % 7) * 1.5:.1f}")
    return "\n".join(lignes).encode()


def kaggle_demo(jours=120, horizon=14):
    """Mini-compétition au format *Store Sales* (train + test, 2 magasins ×
    2 familles, une exogène connue à l'avance) : assez petite pour qu'un
    backtest LightGBM tienne en quelques secondes."""
    train, test = ["date,store_nbr,family,onpromotion,sales"], ["id,date,store_nbr,family,onpromotion"]
    base = time.mktime(time.strptime("2024-01-01", "%Y-%m-%d"))
    n = 0
    for i in range(jours + horizon):
        jour = time.localtime(base + i * 86400)
        date = time.strftime("%Y-%m-%d", jour)
        for magasin in (1, 2):
            for famille in ("A", "B"):
                promo = int((i * magasin + len(famille)) % 5 == 0)
                if i < jours:
                    ventes = 20 * magasin + 10 * (jour.tm_wday >= 5) + 6 * promo + (i % 3)
                    train.append(f"{date},{magasin},{famille},{promo},{ventes}")
                else:
                    test.append(f"{n},{date},{magasin},{famille},{promo}")
                    n += 1
    return "\n".join(train).encode(), "\n".join(test).encode()


# --- smoke -----------------------------------------------------------

def commande_smoke(args):
    base = f"http://127.0.0.1:{args.port}"
    if not attendre(base):
        print(f"ECHEC : rien ne répond sur {base} — lancer `serve` d'abord", file=sys.stderr)
        return 1

    c = Client(base)
    ok = True

    def verifier(libelle, attendu, obtenu, extra=""):
        nonlocal ok
        bon = obtenu == attendu
        ok = ok and bon
        print(f"  [{'ok ' if bon else 'ECHEC'}] {libelle} -> {obtenu} (attendu {attendu}) {extra}")

    code, _ = c.get("/api/stores")
    verifier("endpoint de calcul sans session", 401, code)

    code, corps = c.post("/api/setup", {"username": IDENTIFIANT, "password": MOTDEPASSE})
    if code == 403:  # un compte existe déjà sur cette base
        code, corps = c.post("/api/login", {"username": IDENTIFIANT, "password": MOTDEPASSE})
    verifier("création/connexion du compte", 200, code, json.loads(corps or b"{}").get("username", ""))

    code, corps = c.get("/api/me")
    verifier("session active", 200, code, json.loads(corps).get("username", ""))

    code, corps = c.post(
        "/api/upload/analysis",
        {"date_col": "date", "value_col": "value"},
        {"file": ("demo.csv", csv_demo())},
    )
    verifier("décomposition d'un CSV", 200, code)

    code, corps = c.post(
        "/api/upload/forecast",
        {"date_col": "date", "value_col": "value", "horizon": "7", "model": "sarimax"},
        {"file": ("demo.csv", csv_demo())},
    )
    detail = ""
    if code == 200:
        prevision = json.loads(corps)["prevision"]
        detail = f"{len(prevision)} jours, 1er = {prevision[0]}"
    verifier("prévision SARIMAX", 200, code, detail)

    train, test = kaggle_demo()
    code, corps = c.post(
        "/api/kaggle/start",
        {"competition": "driver-demo", "mode": "backtest", "model": "lightgbm", "date_col": "date",
         "target_col": "sales", "entity_cols": "store_nbr,family", "id_col": "id", "exog_cols": "onpromotion"},
        {"train_file": ("train.csv", train), "test_file": ("test.csv", test)},
    )
    detail, statut = "", "error"
    if code == 200:
        job_id = json.loads(corps)["job_id"]
        for _ in range(120):
            _, corps = c.get(f"/api/kaggle/progress/{job_id}")
            suivi = json.loads(corps)
            statut = suivi["status"]
            if statut != "running":
                break
            time.sleep(0.5)
        detail = (f"RMSLE {suivi['result']['scores']['model']['rmsle']:.4f}" if statut == "done"
                  else suivi.get("error") or "")
    verifier("backtest Kaggle (LightGBM)", "done", statut, detail)

    print("\nsmoke :", "OK" if ok else "ECHEC")
    return 0 if ok else 1


# --- ui --------------------------------------------------------------

CHROMIUMS = sorted(pathlib.Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux*/chrome"))


def commande_ui(args):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ECHEC : playwright absent. `pip install playwright` (cf. SKILL.md).", file=sys.stderr)
        return 1
    if not CHROMIUMS:
        print("ECHEC : aucun chromium en cache. `playwright install chromium`.", file=sys.stderr)
        return 1

    base = f"http://127.0.0.1:{args.port}"
    if not attendre(base):
        print(f"ECHEC : rien ne répond sur {base} — lancer `serve` d'abord", file=sys.stderr)
        return 1

    sortie = pathlib.Path(args.out).resolve()
    sortie.mkdir(parents=True, exist_ok=True)
    erreurs = []

    with sync_playwright() as p:
        # executable_path explicite : le chromium en cache ne correspond
        # pas forcément à la version que réclame le playwright installé.
        nav = p.chromium.launch(executable_path=str(CHROMIUMS[-1]))
        page = nav.new_page(viewport={"width": 1400, "height": 1000})
        page.on("pageerror", lambda e: erreurs.append(str(e)))

        page.goto(f"{base}/login.html")
        besoin_setup = json.loads(
            page.evaluate("fetch('/api/me').then(r => r.text())")
        ).get("needs_setup")
        if besoin_setup:
            page.goto(f"{base}/setup.html")
        page.fill("#login-username" if not besoin_setup else "#setup-username", IDENTIFIANT)
        page.fill("#login-password" if not besoin_setup else "#setup-password", MOTDEPASSE)
        page.click("button[type=submit]")
        page.wait_for_url(f"{base}/index.html", timeout=20000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(sortie / "01-accueil.png"), full_page=True)
        print(f"  connecté en tant que {page.inner_text('#account-username')!r}")

        # Parcours réel : analyser le jeu Kaggle intégré, attendre les
        # graphes. #results est masqué tant que l'analyse n'a pas abouti,
        # donc son apparition est la preuve que le calcul est allé au bout.
        page.click("#analyze-btn")
        page.wait_for_selector("#results:not([hidden])", timeout=120000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(sortie / "02-analyse.png"), full_page=True)
        graphes = page.eval_on_selector_all("#results canvas", "els => els.length")
        print(f"  analyse effectuée : {graphes} graphes rendus")

        page.goto(f"{base}/theorie.html")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(sortie / "03-theorie.png"), full_page=True)
        rendues = page.eval_on_selector_all(
            ".model-formula-eq[data-latex]", "els => els.filter(e => e.querySelector('.katex')).length"
        )
        print(f"  page théorie : {rendues} formules rendues par KaTeX")

        page.goto(f"{base}/models.html")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(sortie / "04-modeles.png"), full_page=True)

        # Page Kaggle : fichiers -> colonnes proposées -> backtest réel.
        train, test = kaggle_demo()
        page.goto(f"{base}/kaggle.html")
        page.fill("#competition-input", "driver-demo")
        page.dispatch_event("#competition-input", "change")
        page.set_input_files("#train-file", {"name": "train.csv", "mimeType": "text/csv", "buffer": train})
        page.set_input_files("#test-file", {"name": "test.csv", "mimeType": "text/csv", "buffer": test})
        page.wait_for_selector("#mapping-panel:not([hidden])", timeout=60000)
        page.click("#backtest-btn")
        page.wait_for_selector("#run-result:not([hidden])", timeout=120000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(sortie / "05-kaggle.png"), full_page=True)
        print(f"  page Kaggle : {page.inner_text('#run-summary')[:90]}...")
        page.set_viewport_size({"width": 390, "height": 900})
        page.wait_for_timeout(300)
        page.screenshot(path=str(sortie / "06-kaggle-mobile.png"), full_page=True)
        largeur = page.evaluate("document.documentElement.scrollWidth")
        print(f"  largeur de page sur téléphone : {largeur}px (attendu 390)")

        nav.close()

    print(f"  captures dans {sortie}")
    print("  erreurs JS :", erreurs if erreurs else "aucune")
    return 1 if erreurs else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = p.add_subparsers(dest="commande", required=True)

    s = sous.add_parser("serve", help="démarre l'app sur une base jetable")
    s.add_argument("--port", type=int, default=PORT_DEFAUT)
    s.add_argument("--data", default="/tmp/cadence-driver")
    s.set_defaults(fn=commande_serve)

    s = sous.add_parser("smoke", help="vérifie l'API de bout en bout")
    s.add_argument("--port", type=int, default=PORT_DEFAUT)
    s.set_defaults(fn=commande_smoke)

    s = sous.add_parser("ui", help="pilote l'interface et écrit des captures")
    s.add_argument("--port", type=int, default=PORT_DEFAUT)
    s.add_argument("--out", default="/tmp/cadence-captures")
    s.set_defaults(fn=commande_ui)

    args = p.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
