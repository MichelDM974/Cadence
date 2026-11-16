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

Le venv du projet suffit pour `serve` et les tests ; `smoke` et `ui` n'ont
besoin que de la stdlib (le python système suffit) :

```bash
.venv/bin/python -c "import fastapi, uvicorn, statsmodels, prophet, lightgbm; print('ok')"
```

S'il manque (machine neuve) :

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt -c constraints.txt
```

## Lancer (chemin agent)

Serveur (bloquant : le lancer en arrière-plan), puis vérification de
l'API :

```bash
.venv/bin/python .claude/skills/run-cadence/driver.py serve --port 8011 --data /tmp/cadence-driver
.venv/bin/python .claude/skills/run-cadence/driver.py smoke --port 8011
```

Sortie attendue : une ligne `[ok ]` par vérification (session, compte,
décomposition, prévision, backtest Kaggle), puis `smoke : OK`. `smoke` n'utilise que la
stdlib : aucune dépendance à installer.

## Captures d'écran et parcours dans l'interface

`ui` se connecte, **clique sur "Analyser"**, attend les graphes, puis
photographie l'accueil, l'analyse, la page théorie et la page modèles,
puis lance un vrai backtest sur la page Kaggle (mini-compétition générée) et
la photographie en largeur bureau et téléphone.

Sans playwright (#13) : le driver pilote le chromium en cache dans
`~/.cache/ms-playwright` (à défaut son `chrome-headless-shell`) par le
protocole DevTools sur un tube, en stdlib — classes `Chromium`, `Page` et
`Locator` reprises d'un autre de mes drivers, réduites à ce dont `ui` se sert.

```bash
python3 .claude/skills/run-cadence/driver.py ui --port 8011 --out /tmp/cadence-captures
```

Compter une à deux minutes : le backtest Kaggle tourne le modèle par
défaut de la page (ensemble LightGBM + SARIMAX, poids appris sur trois
fenêtres).

Sortie attendue : connexion, nombre de graphes et de formules KaTeX rendus,
puis `erreurs JS : aucune` ; code de sortie ≠ 0 sur une erreur JS ou une
largeur sur téléphone différente de 390. Les captures atterrissent dans `--out` : `01-accueil.png`,
`02-analyse.png`, `03-theorie.png`, `04-modeles.png`, `05-kaggle.png`,
`06-kaggle-mobile.png` (largeur de page sur téléphone affichée : doit valoir 390). **Les regarder** —
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
.venv/bin/python -m pytest -q      # ~30 s
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
- **Capture pleine page : fenêtre étirée, pas `captureBeyondViewport`.**
  Ce dernier fait croire à un redimensionnement : les graphes Chart.js
  (responsive) se redessinaient au tiers de leur largeur en pleine
  capture. `Page.screenshot` étire la fenêtre à la hauteur de la page,
  capture, puis la rétablit.

## Dépannage

| Symptôme | Cause et correctif |
|---|---|
| `ModuleNotFoundError: itsdangerous` | `SessionMiddleware` en dépend et fastapi ne le tire pas. Il est dans `requirements.txt` ; réinstaller avec `.venv/bin/pip install -r requirements-dev.txt -c constraints.txt`. |
| `smoke` : `ECHEC : rien ne répond` | `serve` n'est pas lancé, ou pas sur le même `--port`. |
| `ui` : `aucun chromium dans ~/.cache/ms-playwright` | Cache vide : `playwright install chromium`, depuis n'importe quel environnement où playwright est installé (le driver, lui, n'en a pas besoin). |
| `/api/setup` renvoie 403 | Un compte existe déjà sur cette base. Se connecter via `/api/login`, ou repartir d'un `--data` neuf. |
| Port déjà pris | Un `serve` traîne : `fuser -k -n tcp 8011`. |
