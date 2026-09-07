---
name: run-cadence
description: Lancer, piloter, tester et photographier l'application Cadence (prévision de séries temporelles, FastAPI + HTML/JS). À utiliser pour démarrer le serveur, vérifier qu'un changement fonctionne dans l'app réelle, prendre une capture d'écran, ou lancer la suite de tests.
---

# Lancer et piloter Cadence

Web app FastAPI (`src/api.py`) + HTML/JS sans framework (`src/static/`),
persistance SQLite. Elle se pilote par `.claude/skills/run-cadence/driver.py`,
qui la démarre **sur une base jetable**, vérifie l'API, et conduit
l'interface dans chromium.

Tous les chemins ci-dessous sont relatifs à `Cadence/`.

## Prérequis

Le venv du projet suffit pour tout sauf les captures d'écran :

```bash
.venv/bin/python -c "import fastapi, uvicorn, statsmodels, prophet, lightgbm; print('ok')"
```

S'il manque (machine neuve) :

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

## Lancer (chemin agent)

Deux terminaux. D'abord le serveur, qui reste au premier plan :

```bash
.venv/bin/python .claude/skills/run-cadence/driver.py serve --port 8011 --data /tmp/cadence-driver
```

Puis, dans un second terminal, la vérification de l'API :

```bash
.venv/bin/python .claude/skills/run-cadence/driver.py smoke --port 8011
```

Sortie obtenue :

```
  [ok ] endpoint de calcul sans session -> 401 (attendu 401)
  [ok ] création/connexion du compte -> 200 (attendu 200) driver
  [ok ] session active -> 200 (attendu 200) driver
  [ok ] décomposition d'un CSV -> 200 (attendu 200)
  [ok ] prévision SARIMAX -> 200 (attendu 200) 7 jours, 1er = 100.0

smoke : OK
```

`smoke` n'utilise que la stdlib : aucune dépendance à installer.

## Captures d'écran et parcours dans l'interface

`ui` se connecte, **clique sur "Analyser"**, attend les graphes, puis
photographie l'accueil, l'analyse, la page théorie et la page modèles.

Playwright n'est pas une dépendance du projet — l'installer à part pour ne
pas polluer le venv :

```bash
python3 -m venv /tmp/pwenv && /tmp/pwenv/bin/pip install playwright
/tmp/pwenv/bin/python .claude/skills/run-cadence/driver.py ui --port 8011 --out /tmp/cadence-captures
```

Sortie obtenue :

```
  connecté en tant que 'driver'
  analyse effectuée : 8 graphes rendus
  page théorie : 7 formules rendues par KaTeX
  captures dans /tmp/cadence-captures
  erreurs JS : aucune
```

Les captures atterrissent dans `--out` : `01-accueil.png`,
`02-analyse.png`, `03-theorie.png`, `04-modeles.png`. **Les regarder** —
une image vide ou une page d'erreur veut dire que le lancement a échoué
malgré un code de sortie à 0.

## Lancer (chemin humain)

```bash
cd src && ../.venv/bin/uvicorn api:app
```

Sert sur http://127.0.0.1:8000 **et écrit dans `data/cadence.db`**, la
vraie base. À réserver à un usage manuel : pour tout essai, préférer
`driver.py serve`, qui isole les données.

## Tests

```bash
.venv/bin/python -m pytest -q      # 108 tests, ~30 s
```

## Pièges

- **Tout endpoint de calcul renvoie 401 sans session.** `/api/upload/*` et
  `/api/stores*` exigent un cookie de session depuis l'ajout des comptes ;
  seuls `/api/me`, `/api/setup`, `/api/login` et `/api/logout` sont
  publics. Un `curl` nu sur `/api/stores` donne 401 — ce n'est pas une
  panne. Le `Client` du driver conserve le cookie entre les appels.
- **Le premier compte se crée par `/api/setup`, pas par `/api/login`.** Il
  n'y a pas d'inscription publique : `/api/setup` ne marche que tant
  qu'aucun compte n'existe (403 ensuite), les suivants passent par
  `src/create_user.py`. Sur une base neuve, `/api/me` renvoie
  `needs_setup: true` et l'interface redirige vers `setup.html`.
- **`storage.DB_PATH` doit être détourné avant d'importer `api`**, qui
  appelle `storage.session_secret()` au chargement du module — sinon le
  secret est écrit à côté de la vraie base. C'est ce que fait
  `commande_serve`, dans cet ordre précis.
- **Une série trop lisse fait diverger l'ajustement SARIMAX.** Une série
  parfaitement déterministe (constante, ou week-end sans bruit) peut
  produire des NaN selon la version de statsmodels : le résultat sort
  aujourd'hui à `null` plutôt qu'en 500, mais mieux vaut un jeu d'essai
  avec du bruit — `csv_demo()` en fournit un.
- **Les navigateurs de `~/.cache/ms-playwright` ne correspondent pas
  forcément à la version réclamée par le playwright installé.** D'où
  l'`executable_path` explicite du driver, qui prend le chromium en cache
  le plus récent au lieu de laisser playwright chercher le sien.
- **Le venv local est en Python 3.14, la CI et Docker en 3.12.** Des
  comportements diffèrent réellement entre les deux (un ajustement qui
  converge d'un côté et pas de l'autre) : un test vert en local ne
  garantit pas la CI.

## Dépannage

| Symptôme | Cause et correctif |
|---|---|
| `ModuleNotFoundError: itsdangerous` | `SessionMiddleware` en dépend et fastapi ne le tire pas. Il est dans `requirements.txt` ; réinstaller avec `.venv/bin/pip install -r requirements-dev.txt`. |
| `smoke` : `ECHEC : rien ne répond` | `serve` n'est pas lancé, ou pas sur le même `--port`. |
| `ui` : `Executable doesn't exist at .../chromium_headless_shell-XXXX` | Playwright cherche sa version à lui. Le driver contourne via `executable_path` ; si le cache est vide, `playwright install chromium`. |
| `/api/setup` renvoie 403 | Un compte existe déjà sur cette base. Se connecter via `/api/login`, ou repartir d'un `--data` neuf. |
| Port déjà pris | Un `serve` traîne. `pkill -f "driver[.]py serve"` (les crochets évitent que le motif tue le shell appelant). |
