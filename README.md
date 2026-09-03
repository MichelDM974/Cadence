# Cadence

Plateforme web de prévision de rythme d'activité, basée sur l'analyse de séries temporelles.

## Le concept

Cadence traduit une problématique d'**analyse de séries temporelles** en langage métier : ce que l'utilisateur perçoit intuitivement (rythme hebdomadaire, saisonnalité, pics événementiels) est en réalité un signal temporel structuré, que Cadence décompose, modélise et projette.

## Approche technique

Chaque flux de données client (couverts, CA, etc.) est traité comme une **série temporelle** structurée :

- **Décomposition** : séparation tendance / saisonnalité (hebdo, mensuelle, annuelle) / composante événementielle / résidu — logique proche d'un modèle additif ou multiplicatif type STL
- **Variables exogènes** : intégration de séries temporelles externes (météo, calendrier jours fériés/vacances) comme régresseurs enrichissant la prévision
- **Prévision** : projection du rythme sur un horizon donné, avec intervalle de confiance
- **Détection d'anomalies** : identification des écarts significatifs entre rythme observé et rythme attendu (rupture de série)
- **Indice de régularité** : mesure de la prévisibilité du signal (ex. dispersion résiduelle après décomposition), traduite en score lisible pour l'utilisateur non technique

Généraliste dès le départ : le moteur (décomposition + prévision) et l'import CSV s'adressent à n'importe quel secteur ayant une série temporelle quotidienne (date + valeur numérique). La restauration (prévision de couverts) a servi de premier cas de validation.

## Fonctionnalités

- Décomposition du rythme, indice de régularité, alertes de rupture, prévision avec intervalle de confiance, export CSV
- Trois modèles de prévision : SARIMAX et Prophet (comparaison sur une entité, avec backtest contre une baseline naïve), LightGBM (génération en masse multi-entités)
- Variables exogènes mondiales, activables individuellement par l'utilisateur : jours fériés (tous pays), météo (historique réelle + normales saisonnières), vacances scolaires (France, par académie)
- Import CSV multi-entités (jusqu'à 2 dimensions de regroupement, ex. magasin × catégorie), avec génération de prévisions en masse pour toutes les combinaisons
- Variables exogènes personnalisées : colonnes numériques déjà présentes dans le fichier importé, ou un second fichier externe joint automatiquement (série globale ou par entité, y compris valeurs futures réelles)
- Persistance locale (SQLite) des points de vente/séries analysées, réutilisables sans réimport
- Hyperparamètres des 3 modèles éditables depuis l'interface
- Mode sombre

## Résultats

Prototype validé sur données réelles (Kaggle *Recruit Restaurant Visitor Forecasting*, 800+ restaurants, un modèle par restaurant) :

| Configuration | Private RMSLE | Public RMSLE |
|---|---|---|
| SARIMAX + jour férié | **0.57956** | **0.51544** |
| SARIMAX seul (aucun exogène) | 0.58205 | 0.51553 |
| SARIMAX + jour férié + météo | 0.59146 | 0.51875 |
| Prophet + jour férié | 0.60576 | 0.55071 |

Et sur *Store Sales - Time Series Forecasting* (54 magasins × 33 catégories, 1782 entités) :

| Configuration | RMSLE |
|---|---|
| LightGBM, configuration affinée* | **0.42475** |
| SARIMAX seul (aucun exogène) | 0.42726 |
| SARIMAX + `onpromotion` (jointure par entité) | 0.44460 |
| Prophet + `onpromotion` (jointure par entité) | 0.51275 |
| LightGBM + `onpromotion` (jointure par entité, génération en masse) | 0.55166 |

\* meilleur score obtenu à ce jour (banc d'essai indépendant, cf. `src/forecast_submission_lightgbm_store_sales.py`). L'objectif tweedie qui explique l'essentiel du gain est repris dans l'app depuis le 2026-09-03 ; le reste (lags étendus, tendance annuelle, régularisation) ne l'est pas encore.

(0.501 et 0.37294 respectivement en tête des leaderboards Kaggle, pour référence.) SARIMAX bat par ailleurs une baseline naïve ("même jour la semaine dernière") sur 91% des restaurants testés (amélioration médiane +28,4% sur le MAE).

## Installation

```
cd src
pip install -r requirements.txt
uvicorn api:app
```

Tests (aussi exécutés en CI sur chaque push/PR vers `main`) :

```
pip install -r requirements-dev.txt
pytest tests/ -v
```
