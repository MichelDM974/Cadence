# Benchmarks

Historique des scores mesurés sur deux jeux de données Kaggle réels, utilisés
comme bancs d'essai pour comparer les modèles et les axes de précision entre
eux (cf. `src/forecast_submission_*.py`, `src/benchmark_lightgbm_variants.py`).
Décisions actées à partir de ces chiffres → `journal-de-bord/cadence.md`.

- **Recruit Restaurant Visitor Forecasting** — 800+ restaurants, un modèle
  par restaurant, historique **court et hétérogène** (20 à 478 jours selon
  le restaurant, dates d'ouverture différentes).
- **Store Sales - Time Series Forecasting** — 54 magasins × 33 catégories
  (1782 entités), historique **long et homogène** (1684 jours, identique
  pour toutes les entités).

Deux types de scores, à ne pas confondre :

- **Kaggle (officiel)** : vraie soumission notée par Kaggle sur son jeu de
  test caché — la mesure qui compte, mais coûteuse (une par config).
- **Backtest interne (derniers jours)** : RMSLE calculé localement en
  retenant les N derniers jours connus de chaque dataset comme test (39j
  Recruit, 16j Store Sales) et en comparant à la vraie valeur — rejouable à
  volonté, sert à trier les configs avant de soumettre à Kaggle. Les deux ne
  concordent pas forcément exactement (cf. note sous le tableau Store Sales).

## Scores Kaggle officiels

### Recruit

| Configuration | Private RMSLE | Public RMSLE |
|---|---|---|
| SARIMAX + jour férié | **0.57956** | **0.51544** |
| SARIMAX seul (aucun exogène) | 0.58205 | 0.51553 |
| SARIMAX + jour férié + météo | 0.59146 | 0.51875 |
| Prophet + jour férié | 0.60576 | 0.55071 |

Tête du leaderboard public, pour référence : 0.501.

### Store Sales

| Configuration | RMSLE |
|---|---|
| LightGBM, objectif tweedie (config app depuis le 03/09/2026) | **0.42475** |
| SARIMAX seul (aucun exogène) | 0.42726 |
| SARIMAX + `onpromotion` (jointure par entité) | 0.44460 |
| Prophet + `onpromotion` (jointure par entité) | 0.51275 |
| LightGBM + `onpromotion` (jointure par entité, génération en masse) | 0.55166 |

Tête du leaderboard, pour référence : 0.37294.

## Backtests internes récents (RMSLE, derniers jours connus)

### Recruit — 5 configurations LightGBM bulk (07/09/2026, `benchmark_lightgbm_variants.py`)

Rejoué en appelant directement `run_bulk_lightgbm_forecast`, le point d'entrée
réel de production — pas une réimplémentation séparée.

| Configuration | RMSLE |
|---|---|
| régularisation réduite | 0.72230 |
| **lean (défaut actuel — yoy_trend auto-détecté)** | **0.72323** |
| lag annuel + yoy_trend (forcé) | 0.72323 |
| lags fins | 0.72426 |
| lags fins + yoy (combiné) | 0.72426 |
| lean, ancien défaut (yoy_trend forcé désactivé) | 0.73057 |

`yoy_trend` auto-détecté (07/09/2026, cf. journal-de-bord) reproduit
exactement le score de la config explicite équivalente — confirme que la
détection automatique (coefficient de variation des longueurs d'historique
par entité) s'active correctement sur ce dataset hétérogène (cv=0.31),
contre l'ancien défaut désactivé. `régularisation réduite` reste légèrement
meilleure ici mais dégrade Store Sales (cf. ci-dessous) — pas retenue comme
défaut, cf. règle "améliore tous les cas connus ou reste optionnelle"
(`journal-de-bord/cadence.md`, Décisions actées).

### Store Sales — combiné (lags fins + yoy + régularisation réduite), 03/09/2026

| Configuration | RMSLE |
|---|---|
| Tweedie + lags fins/annuels + yoy_trend + régularisation réduite | 0.498 (agrégat) / 0.4232 (hors une famille) |

Le score agrégat (0.498) est plombé par une seule famille de produits
(rentrée scolaire, croissance hors de portée d'un arbre de décision) ; hors
cette famille, 0.4232 — moins bon que le tweedie seul officiel (0.42475 ci-
dessus), d'où la décision de ne pas combiner ces axes par défaut. Backtest
plus ancien (méthodologie légèrement différente de celle utilisée ci-dessus
pour Recruit) ; pas encore rejoué avec `benchmark_lightgbm_variants.py`
variante par variante sur Store Sales depuis l'introduction de `yoy_trend`
auto-détecté (dataset volumineux, 1782 entités — plusieurs dizaines de
minutes par variante) : le résultat est cependant garanti inchangé sur ce
dataset précis, puisque `yoy_trend` ne s'active que sur historique
hétérogène (cv > 0.1) et que Store Sales est parfaitement homogène (cv=0.0
— toutes les entités ont exactement le même historique).

## Reproduire

```
cd src
python benchmark_lightgbm_variants.py both   # LightGBM, les deux datasets
python forecast_submission_sarimax_ferie.py  # SARIMAX, soumission Kaggle Recruit
python forecast_submission_lightgbm_recruit.py
python forecast_submission_lightgbm_store_sales.py
```

Jeux de données attendus : `data/raw/` (Recruit, déjà dans le dépôt) et
`~/Téléchargements/store-sales-time-series-forecasting/` (Store Sales, à
télécharger depuis Kaggle — chemin en dur dans `benchmark_lightgbm_variants.py`).
