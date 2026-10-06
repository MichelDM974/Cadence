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
| **Page Kaggle : LightGBM « gros » (255 feuilles, 0.03, 1200 it.) + séries éteintes à 0 (05/10/2026)** | **0.42150** |
| Script dédié `forecast_submission_lightgbm_store_sales.py`, tweedie (03/09/2026) | 0.42475 |
| Page Kaggle : LightGBM défaut, sans exogène (05/10/2026) | 0.46408 |
| Page Kaggle : LightGBM défaut + `onpromotion` futur de test.csv (05/10/2026) | 0.53171 |
| SARIMAX seul (aucun exogène) | 0.42726 |
| SARIMAX + `onpromotion` (jointure par entité) | 0.44460 |
| Prophet + `onpromotion` (jointure par entité) | 0.51275 |
| LightGBM + `onpromotion` (jointure par entité, génération en masse) | 0.55166 |

Tête du leaderboard, pour référence : 0.37294.

## Backtests internes récents (RMSLE, derniers jours connus)

### Store Sales — banc `benchmark_kaggle.py` (05/10/2026)

Même chemin que la page (`api.kaggle_backtest`). « Récent » = moyenne de 3
fenêtres de 16 j en fin de train ; « an-1 » = la période du test un an plus
tôt (16–31/08/2016). Seules les pistes qui gagnent **sur les deux** sont
retenues : la fenêtre récente seule a classé `onpromotion` à l'envers du
score officiel (cf. tableau Kaggle ci-dessus).

| Variante | Récent | an-1 |
|---|---|---|
| base (défaut, avec `onpromotion`) | 0.47606 | 0.54297 |
| sans exogène | 0.48811 | 0.54201 |
| séries éteintes à 0 (14 j) | 0.46080 | 0.52790 |
| séries éteintes à 0 (28 j) | 0.46164 | 0.52869 |
| sans exogène + éteintes 14 j | 0.47420 | 0.52687 |
| lags fins (1–7, 14, 21, 28 ; fenêtres 7–56) | 0.47588 | 0.54660 |
| arrondi à 4 décimales | 0.47606 | 0.54298 |
| 3 graines moyennées | 0.47818 | 0.54539 |
| **LightGBM « gros » (255 feuilles, 0.03, 1200 it.)** | **0.44420** | **0.50491** |

« an-1 » a suivi le score officiel sur la soumission retenue (0.543 -> 0.505
en local, 0.464 -> 0.4215 officiel) ; elle reste approximative sur
l'ampleur (exogène : 0.001 en local, 0.068 officiel).

### Ensemble LightGBM + SARIMAX et réglage tweedie (06/10/2026, `benchmark_kaggle.py --cache`)

Même banc (3 fenêtres récentes + an-1), séries éteintes à 0 partout,
**avec `onpromotion`** (spec du banc) : les écarts entre variantes
comptent, pas le niveau. Ensemble = moyenne pondérée des prévisions
(`options["ensemble"]`), poids = part de LightGBM.

Store Sales :

| Variante | Récent | an-1 |
|---|---|---|
| LightGBM « gros » (référence) | 0.42516 | 0.49236 |
| SARIMAX seul (sans séries éteintes) | 0.42394 | 0.49999 |
| ensemble 90 % LightGBM | 0.42063 | 0.48761 |
| ensemble 80 % | 0.41676 | 0.48363 |
| ensemble 70 % | 0.41358 | 0.48043 |
| **ensemble 50 %** | **0.40957** | **0.47669** |
| tweedie 1.3 (au lieu de 1.1) | 0.41769 | 0.48212 |
| tweedie 1.5 | 0.41293 | 0.48279 |
| tweedie 1.3 + ensemble 50 % | 0.40631 | **0.47306** |
| **tweedie 1.5 + ensemble 50 %** | **0.40430** | 0.47319 |
| tweedie 1.5 + ensemble 40 % | 0.40504 | 0.47427 |

Sur une fenêtre récente seule (tri), 40 % ≈ 50 % > 30 % pour l'ensemble ;
tweedie 1.2 < 1.3 ≈ 1.4 ≈ 1.5 > 1.6. Les autres axes de la grille autour du
profil « panel homogène » (feature_fraction 0.6, min_data_in_leaf 30,
lambda_l2 1.0, num_leaves 127, learning_rate 0.02 ou 0.05 à budget
équivalent) perdent tous sur la fenêtre récente : le profil était déjà
bien placé, la puissance tweedie est le seul levier trouvé.

Recruit (même banc, 39 j) — l'ensemble gagne aussi, alors que SARIMAX seul
y est nettement moins bon que LightGBM :

| Variante | Récent | an-1 |
|---|---|---|
| LightGBM (défaut, séries éteintes neutres) | 0.59721 | 0.74713 |
| SARIMAX seul | 0.62921 | 0.79539 |
| ensemble 50 % | 0.58907 | 0.72788 |
| **ensemble 70 %** | **0.58868** | **0.72554** |

La puissance tweedie ne touche que le profil « panel homogène » : sans
effet sur Recruit, auto-détecté hétérogène.

### Store Sales — exogène future lue dans test.csv (05/10/2026, page Kaggle)

Même backtest LightGBM (défaut app, 16 derniers jours), lancé depuis la
page Kaggle, qui passe par la même génération en masse :

| Configuration | RMSLE |
|---|---|
| sans exogène | 0.53694 |
| **+ `onpromotion`, valeurs futures réelles de test.csv** | **0.49947** |

Le 0.53694 reproduit exactement la ligne « régularisation réduite » plus
bas : même pipeline. L'ancien `onpromotion` (0.55166 au Kaggle officiel)
reportait la dernière valeur connue sur tout l'horizon ; c'est la vraie
valeur future qui fait le gain. Pas encore soumis à Kaggle.

### Recruit — repli et prolongement des entités (05/10/2026, page Kaggle)

LightGBM défaut, 39 derniers jours : **0.57506** (naïf saisonnier
0.83966), contre 0.72323 avec `benchmark_lightgbm_variants.py`. Écart dû
à la méthode, pas au modèle : le script laisse à 0 les entités sans
prévision, et celles dont l'historique s'arrête avant la coupure ; la page
leur applique leur moyenne récente et les prolonge jusqu'à la fin du
train. Les chiffres Recruit antérieurs ne se comparent donc qu'entre eux.

### Recruit — 5 configurations LightGBM bulk (07/09/2026, `benchmark_lightgbm_variants.py`)

Rejoué en appelant directement `run_bulk_lightgbm_forecast`, le point d'entrée
réel de production — pas une réimplémentation séparée.

| Configuration | RMSLE |
|---|---|
| régularisation réduite (forcée) | 0.72230 |
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
meilleure ici en isolation, mais n'est pas auto-activée sur ce dataset
hétérogène (cv=0.31 > seuil) : gardée opt-in faute de preuve isolée sur
historique hétérogène (le seul test connu la combinait à d'autres axes,
cf. ci-dessous) — cf. règle "améliore tous les cas connus ou reste
optionnelle" (`journal-de-bord/cadence.md`, Décisions actées).

### Store Sales — régularisation réduite isolée (07/09/2026, `benchmark_lightgbm_variants.py`)

| Configuration | RMSLE |
|---|---|
| lean (ancien défaut, avant auto-détection) | 0.54118 |
| **régularisation réduite (= défaut actuel, auto-détecté)** | **0.53694** |

Confirme en isolation ce que le combiné du 03/09 (ci-dessous) laissait
supposer : sur ce dataset homogène (cv=0.0), la régularisation réduite
seule améliore bien le score (~0.8%), pas seulement en mélange avec
d'autres axes. Auto-détectée depuis le 07/09/2026 (même signal que
`yoy_trend`, cf. `_should_auto_reduce_regularization`) — "lean" ci-dessus
égale donc désormais ce résultat. Les valeurs (0.53-0.54) sont loin du
score Kaggle officiel (0.42475, tableau plus haut) : le backtest local
sous-estime la vraie performance sur ce dataset (fenêtre de 16 jours moins
représentative que le test Kaggle réel, déjà documenté le 03/09) — la
comparaison **relative** entre les deux lignes reste fiable (même run, même
fenêtre), l'écart absolu avec Kaggle non.

### Store Sales — transactions.csv + cluster×famille (07/09/2026, `forecast_submission_lightgbm_store_sales.py`)

| Configuration | RMSLE |
|---|---|
| lags fins + yoy + régularisation réduite (03/09, ci-dessous) | 0.498 (agrégat) / 0.4232 (hors SCHOOL AND OFFICE SUPPLIES) |
| **+ fréquentation par magasin (`transactions.csv`) + moyenne cluster×famille** | **0.49359 (agrégat) / 0.42235 (hors la même famille)** |

Gain marginal (~0.2%) hors la famille pathologique, agrégat quasi inchangé —
`transactions.csv` (gelé sur le dernier jour connu pendant l'horizon, absent
au-delà contrairement à `oil.csv`) et l'agrégat cluster×famille apportent un
signal complémentaire mais ne s'attaquent pas au vrai verrou : la croissance
de SCHOOL AND OFFICE SUPPLIES (rentrée scolaire) reste hors de portée d'un
arbre de décision, quelles que soient les features en entrée (cf. ligne
ci-dessous). Pas resoumis à Kaggle pour un gain de cette taille.

### Store Sales — combiné (lags fins + yoy + régularisation réduite), 03/09/2026

| Configuration | RMSLE |
|---|---|
| Tweedie + lags fins/annuels + yoy_trend + régularisation réduite | 0.498 (agrégat) / 0.4232 (hors une famille) |

Le score agrégat (0.498) est plombé par une seule famille de produits
(rentrée scolaire, croissance hors de portée d'un arbre de décision) ; hors
cette famille, 0.4232 — moins bon que le tweedie seul officiel (0.42475 ci-
dessus), d'où la décision d'alors de ne pas combiner ces axes par défaut.
Backtest plus ancien (méthodologie légèrement différente du run isolé
ci-dessus) ; `lags fins`/`yoy_trend` combinés à la régularisation n'ont pas
été retestés isolément sur Store Sales depuis (dataset volumineux,
plusieurs dizaines de minutes par variante) — `yoy_trend` resterait de
toute façon désactivé ici par l'auto-détection (historique homogène).

## Reproduire

```
cd src
python benchmark_kaggle.py store-sales dead14 dead14+ens_lgb_sarimax --cache /tmp/cadence-cache
python benchmark_kaggle.py store-sales noexog+dead14+ens_lgb_sarimax --cache /tmp/cadence-cache --submit ~/soumissions
python benchmark_lightgbm_variants.py both   # LightGBM, les deux datasets
python forecast_submission_sarimax_ferie.py  # SARIMAX, soumission Kaggle Recruit
python forecast_submission_lightgbm_recruit.py
python forecast_submission_lightgbm_store_sales.py
```

Jeux de données attendus : `data/raw/` (Recruit, déjà dans le dépôt) et
`~/Téléchargements/store-sales-time-series-forecasting/` (Store Sales, à
télécharger depuis Kaggle — chemin en dur dans `benchmark_lightgbm_variants.py`).
