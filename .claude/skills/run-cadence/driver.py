#!/usr/bin/env python3
"""Driver de Cadence : lance l'app sur une base jetable, la pilote, la
photographie.

Trois sous-commandes, à lancer depuis la racine du projet avec .venv/bin/python :

    serve   démarre l'app sur une base isolée (bloquant, Ctrl-C pour arrêter)
    smoke   vérifie l'API de bout en bout (création de compte incluse)
    ui      pilote l'interface dans chromium et écrit des captures (stdlib
            seule, sans playwright : le python système suffit)

`serve` n'écrit jamais dans data/cadence.db : sans ça, le moindre essai
crée des comptes et des points de vente dans les vraies données.
"""
import argparse
import base64
import fcntl
import json
import os
import pathlib
import select
import subprocess
import sys
import tempfile
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

# Le chromium que `playwright install chromium` met en cache ; à défaut, son
# chrome-headless-shell, qui suffit aussi. Playwright lui-même n'est pas
# requis (#13) : le driver parle DevTools en stdlib.
CHROMIUMS = (sorted(pathlib.Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux*/chrome"))
             or sorted(pathlib.Path.home().glob(
                 ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux*/chrome-headless-shell")))
# Ceux des réglages de playwright qui changent ce que la page montre (barres
# de défilement, survol, couleurs) ; le trousseau GNOME, verrouillé au
# démarrage, figerait chromium sans `basic`.
OPTIONS_CHROMIUM = [
    "--headless", "--no-sandbox", "--hide-scrollbars", "--mute-audio", "--force-color-profile=srgb",
    "--blink-settings=primaryHoverType=2,availableHoverTypes=2,primaryPointerType=4,availablePointerTypes=4",
    "--no-first-run", "--no-default-browser-check", "--password-store=basic",
    "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
]
# Avant un clic, comme playwright : défile jusqu'à l'élément s'il n'est pas
# tout entier à l'écran, puis rend sa boîte.
BOITE = """el => {
    let r = el.getBoundingClientRect();
    if (r.top < 0 || r.left < 0 || r.bottom > innerHeight || r.right > innerWidth) {
        el.scrollIntoView({block: 'center', inline: 'center', behavior: 'instant'});
        r = el.getBoundingClientRect();
    }
    return {x: r.x, y: r.y, width: r.width, height: r.height};
}"""
# La visibilité selon playwright (état par défaut de wait_for_selector).
VISIBLE = """el => { const r = el.getBoundingClientRect();
    return el.checkVisibility() && getComputedStyle(el).visibility === 'visible' && r.width > 0 && r.height > 0; }"""


class Chromium:
    """Le chromium en cache, piloté sans playwright (repris d'un autre de
    mes drivers, réduit à ce dont `ui` se sert) : le protocole DevTools passe par
    un tube (`--remote-debugging-pipe` : du JSON terminé par un octet nul,
    lu sur son fd 3, écrit sur son fd 4)."""

    def __init__(self, binaire):
        self.profil = tempfile.TemporaryDirectory(prefix="cadence-chromium-")
        self.journal = pathlib.Path(self.profil.name) / "chromium.log"
        lecture, self.vers = os.pipe()
        self.depuis, ecriture = os.pipe()
        # Recopiés au-dessus de 4 : les dup2 de l'enfant vers 3 et 4 ne
        # peuvent plus s'écraser l'un l'autre.
        enfant = [fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 5) for fd in (lecture, ecriture)]
        os.close(lecture)
        os.close(ecriture)
        with open(self.journal, "wb") as journal:
            self.processus = subprocess.Popen(
                [str(binaire), *OPTIONS_CHROMIUM, f"--user-data-dir={self.profil.name}",
                 "--remote-debugging-pipe", "about:blank"],
                stdout=journal, stderr=journal, close_fds=False,
                preexec_fn=lambda: (os.dup2(enfant[0], 3), os.dup2(enfant[1], 4)))
        for fd in enfant:
            os.close(fd)
        self.tampon, self.numero, self.reponses, self.pages = b"", 0, {}, {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.profil is None:
            return
        try:
            self._ecrire({"id": 0, "method": "Browser.close"})
            self.processus.wait(10)
        except (OSError, subprocess.TimeoutExpired):
            self.processus.kill()
            self.processus.wait()
        os.close(self.vers)
        os.close(self.depuis)
        self.profil.cleanup()
        self.profil = None

    def _ecrire(self, message):
        donnees = json.dumps(message).encode() + b"\0"
        while donnees:
            donnees = donnees[os.write(self.vers, donnees):]

    def _lire(self, delai):
        """Le prochain message de chromium, ou None passé `delai` secondes."""
        fin = time.monotonic() + delai
        while b"\0" not in self.tampon:
            if not select.select([self.depuis], [], [], max(0, fin - time.monotonic()))[0]:
                return None
            morceau = os.read(self.depuis, 1 << 20)
            if not morceau:
                raise RuntimeError("chromium s'est arrêté : " + self.journal.read_text(errors="replace")[-500:])
            self.tampon += morceau
        message, _, self.tampon = self.tampon.partition(b"\0")
        return json.loads(message)

    def attendre(self, condition, secondes):
        """Traite ce qu'envoie chromium jusqu'à ce que `condition()` soit
        vraie ; faux si elle ne l'est pas au bout de `secondes`."""
        fin = time.monotonic() + secondes
        while not condition():
            reste = fin - time.monotonic()
            if reste <= 0:
                return False
            message = self._lire(reste)
            if message is None:
                continue
            if "id" in message:
                self.reponses[message["id"]] = message
            elif message.get("sessionId") in self.pages:
                self.pages[message["sessionId"]].evenement(message["method"], message.get("params", {}))
        return True

    def commande(self, methode, session=None, **params):
        self.numero += 1
        numero = self.numero
        self._ecrire({"id": numero, "method": methode, "params": params, **({"sessionId": session} if session else {})})
        if not self.attendre(lambda: numero in self.reponses, 30):
            raise RuntimeError(f"{methode} : pas de réponse de chromium en 30 s")
        reponse = self.reponses.pop(numero)
        if "error" in reponse:
            raise RuntimeError(f"{methode} : {reponse['error'].get('message')}")
        return reponse.get("result", {})

    def new_page(self, viewport):
        """Un onglet dans un contexte à lui (cookies compris), comme le
        `new_page` d'un navigateur playwright."""
        contexte = self.commande("Target.createBrowserContext")["browserContextId"]
        cible = self.commande("Target.createTarget", url="about:blank", browserContextId=contexte)["targetId"]
        session = self.commande("Target.attachToTarget", targetId=cible, flatten=True)["sessionId"]
        page = self.pages[session] = Page(self, session)
        for domaine in ("Page", "Runtime"):
            page.cdp(f"{domaine}.enable")
        page.cdp("Emulation.setFocusEmulationEnabled", enabled=True)
        page.set_viewport_size(viewport)
        return page


class Page:
    def __init__(self, nav, session):
        self.nav, self.session = nav, session
        self.chargements = 0  # événements « load » du document principal
        self.rappels = {"pageerror": []}

    def cdp(self, methode, **params):
        return self.nav.commande(methode, self.session, **params)

    def on(self, evenement, rappel):
        """« pageerror » reçoit le message de l'exception."""
        self.rappels[evenement].append(rappel)

    def evenement(self, methode, params):
        if methode == "Page.loadEventFired":
            self.chargements += 1
        elif methode == "Runtime.exceptionThrown":
            details = params["exceptionDetails"]
            for rappel in self.rappels["pageerror"]:
                rappel(details.get("exception", {}).get("description") or details.get("text"))

    def js(self, expression):
        resultat = self.cdp("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in resultat:
            details = resultat["exceptionDetails"]
            raise RuntimeError("script : " + (details.get("exception", {}).get("description") or details.get("text")))
        return resultat["result"].get("value")

    def _sonder(self, expression, ms, quoi):
        """Attend que `expression` soit vraie. Une page qui change sous le
        script (« context destroyed ») compte pour un non, pas une panne."""
        fin = time.monotonic() + ms / 1000
        while True:
            try:
                if self.js(expression):
                    return
            except RuntimeError:
                pass
            if time.monotonic() > fin:
                raise RuntimeError(f"{quoi} : pas obtenu en {ms / 1000:.0f} s")
            self.wait_for_timeout(100)

    def goto(self, url):
        avant = self.chargements
        resultat = self.cdp("Page.navigate", url=url)
        if resultat.get("errorText"):
            raise RuntimeError(f"{url} : {resultat['errorText']}")
        if not self.nav.attendre(lambda: self.chargements > avant, 30):
            raise RuntimeError(f"{url} : aucune page chargée en 30 s")

    def wait_for_timeout(self, ms):
        self.nav.attendre(lambda: False, ms / 1000)

    def wait_for_url(self, url, timeout=30000):
        self._sonder(f"location.href === {json.dumps(url)} && document.readyState === 'complete'",
                     timeout, f"adresse {url}")

    def wait_for_selector(self, css, timeout=30000):
        """Comme playwright : attend qu'un élément du sélecteur soit visible."""
        self._sonder(f"(() => {{ const el = document.querySelector({json.dumps(css)}); "
                     f"return !!el && ({VISIBLE})(el); }})()", timeout, css)

    def evaluate(self, expression):
        return self.js(expression)

    def eval_on_selector_all(self, css, fonction):
        return self.js(f"({fonction})(Array.from(document.querySelectorAll({json.dumps(css)})))")

    def set_viewport_size(self, viewport):
        self.viewport = dict(viewport)
        self._fenetre(viewport["width"], viewport["height"])

    def _fenetre(self, largeur, hauteur):
        self.cdp("Emulation.setDeviceMetricsOverride", width=largeur, height=hauteur,
                 deviceScaleFactor=1, mobile=False)

    def locator(self, css):
        return Locator(self, css)

    def fill(self, css, texte):
        self.locator(css).fill(texte)

    def click(self, css):
        self.locator(css).click()

    def inner_text(self, css):
        return self.locator(css).inner_text()

    def dispatch_event(self, css, type_):
        self.locator(css).dispatch_event(type_)

    def set_input_files(self, css, fichier):
        self.locator(css).set_input_files(fichier)

    def screenshot(self, path, full_page=False):
        """Pleine page : la fenêtre est d'abord étirée à la hauteur de la
        page, puis rendue à sa taille. `captureBeyondViewport` fait croire à
        un redimensionnement : les graphes Chart.js (responsive) s'y
        redessinaient au tiers de leur largeur, au milieu de la capture. La
        page peut grandir une fois étirée (graphes redessinés) : on étire
        de nouveau jusqu'à ce que sa hauteur ne bouge plus, sinon le bas
        n'est pas peint."""
        if not full_page:
            donnees = self.cdp("Page.captureScreenshot", format="png")["data"]
        else:
            self.js("window.scrollTo(0, 0)")
            hauteur = 0
            for _ in range(5):
                nouvelle = max(self.js("document.documentElement.scrollHeight"), self.viewport["height"])
                if nouvelle == hauteur:
                    break
                hauteur = nouvelle
                self._fenetre(self.viewport["width"], hauteur)
                self.wait_for_timeout(300)
            donnees = self.cdp("Page.captureScreenshot", format="png", clip={
                "x": 0, "y": 0, "width": self.viewport["width"], "height": hauteur, "scale": 1})["data"]
            self._fenetre(self.viewport["width"], self.viewport["height"])
            self.wait_for_timeout(150)
        pathlib.Path(path).write_bytes(base64.b64decode(donnees))


class Locator:
    """Un sélecteur CSS ; chaque action vise son premier élément, attendu
    jusqu'à 10 s comme le ferait playwright."""

    def __init__(self, page, css):
        self.page, self.css = page, css

    def _trouver(self):
        return f"document.querySelector({json.dumps(self.css)})"

    def _sur(self, fonction):
        self.page._sonder(f"!!{self._trouver()}", 10000, self.css)
        return self.page.js(f"({fonction})({self._trouver()})")

    def click(self):
        boite = self._sur(BOITE)
        x, y = boite["x"] + boite["width"] / 2, boite["y"] + boite["height"] / 2
        for geste in ("mouseMoved", "mousePressed", "mouseReleased"):
            self.page.cdp("Input.dispatchMouseEvent", type=geste, x=x, y=y, clickCount=1,
                          button="none" if geste == "mouseMoved" else "left")

    def fill(self, texte):
        """Comme une frappe : remplace le contenu et déclenche `input`. Vidé
        par script plutôt que sélectionné : une valeur posée entre-temps par
        la page (kaggle.js remplit la compétition à son chargement) se
        retrouvait devant le texte tapé."""
        self._sur("el => { el.focus(); el.value = ''; }")
        self.page.cdp("Input.insertText", text=texte)

    def inner_text(self):
        return self._sur("el => el.innerText")

    def dispatch_event(self, type_):
        self._sur(f"el => el.dispatchEvent(new Event({json.dumps(type_)}, {{bubbles: true}}))")

    def set_input_files(self, fichier):
        """`fichier` à la playwright ({name, mimeType, buffer}) : écrit dans
        le profil jetable de chromium, puis donné au champ, qui reçoit
        `input` et `change` comme après un vrai choix."""
        self._sur("el => true")
        dossier = pathlib.Path(tempfile.mkdtemp(dir=self.page.nav.profil.name, prefix="fichiers-"))
        chemin = dossier / fichier["name"]
        chemin.write_bytes(fichier["buffer"])
        objet = self.page.cdp("Runtime.evaluate", expression=self._trouver())["result"]["objectId"]
        self.page.cdp("DOM.setFileInputFiles", files=[str(chemin)], objectId=objet)


def commande_ui(args):
    if not CHROMIUMS:
        print("ECHEC : aucun chromium dans ~/.cache/ms-playwright (`playwright install chromium`, "
              "d'où qu'il soit lancé).", file=sys.stderr)
        return 1

    base = f"http://127.0.0.1:{args.port}"
    if not attendre(base):
        print(f"ECHEC : rien ne répond sur {base} — lancer `serve` d'abord", file=sys.stderr)
        return 1

    sortie = pathlib.Path(args.out).resolve()
    sortie.mkdir(parents=True, exist_ok=True)
    erreurs = []

    with Chromium(CHROMIUMS[-1]) as nav:
        page = nav.new_page(viewport={"width": 1400, "height": 1000})
        page.on("pageerror", lambda e: erreurs.append(str(e)))

        # Lu hors de la page : sur une base neuve, auth.js quitte login.html
        # pour setup.html pendant qu'un evaluate y tournerait.
        besoin_setup = json.loads(
            urllib.request.urlopen(base + "/api/me", timeout=5).read()
        ).get("needs_setup")
        page.goto(f"{base}/{'setup' if besoin_setup else 'login'}.html")
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
        page.wait_for_timeout(500)  # la page pose d'abord la dernière compétition
        page.fill("#competition-input", "driver-demo")
        page.dispatch_event("#competition-input", "change")
        page.set_input_files("#train-file", {"name": "train.csv", "mimeType": "text/csv", "buffer": train})
        page.set_input_files("#test-file", {"name": "test.csv", "mimeType": "text/csv", "buffer": test})
        page.wait_for_selector("#mapping-panel:not([hidden])", timeout=60000)
        page.click("#backtest-btn")
        page.wait_for_selector("#run-result:not([hidden])", timeout=300000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(sortie / "05-kaggle.png"), full_page=True)
        print(f"  page Kaggle : {page.inner_text('#run-summary')[:90]}...")
        page.set_viewport_size({"width": 390, "height": 900})
        page.wait_for_timeout(300)
        page.screenshot(path=str(sortie / "06-kaggle-mobile.png"), full_page=True)
        largeur = page.evaluate("document.documentElement.scrollWidth")
        print(f"  largeur de page sur téléphone : {largeur}px (attendu 390)")

    print(f"  captures dans {sortie}")
    print("  erreurs JS :", erreurs if erreurs else "aucune")
    return 1 if erreurs or largeur != 390 else 0


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
