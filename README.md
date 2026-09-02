# Cadence

Plateforme web de prévision de rythme d'activité, basée sur l'analyse de séries temporelles.

## Le concept

Cadence traduit une problématique d'**analyse de séries temporelles** en langage métier : ce que l'utilisateur perçoit intuitivement (rythme hebdomadaire, saisonnalité, pics événementiels) est en réalité un signal temporel structuré, que Cadence décompose, modélise et projette.

## Décisions actées

- **Nom** : Cadence (définitif)
- **Cible** : B2B et B2C, MVP centré B2B
- **Scope** : Option B — verticaux spécialisés par secteur, chaque module encodant les composantes temporelles propres à son activité (saisonnalité métier, cycles hebdomadaires, effets calendaires spécifiques)
- **Premier vertical MVP** : Restauration (prévision de couverts)
- **Généralisation** : le moteur (décomposition + prévision) est agnostique du secteur, l'import CSV accepte n'importe quelle série quotidienne (date + valeur numérique). Une piste de "verticaux" avec presets d'exogènes par secteur a été explorée puis abandonnée : mathématiquement ça ne changeait que la présélection de 3 checkboxes déjà modifiables manuellement — sans valeur ajoutée réelle. L'utilisateur active/désactive directement jours fériés / vacances scolaires / météo selon ce qui est pertinent pour son activité

## Approche technique

Chaque flux de données client (couverts, CA, etc.) est traité comme une **série temporelle** structurée :

- **Décomposition** : séparation tendance / saisonnalité (hebdo, mensuelle, annuelle) / composante événementielle / résidu — logique proche d'un modèle additif ou multiplicatif type STL
- **Variables exogènes** : intégration de séries temporelles externes (météo, calendrier jours fériés/vacances) comme régresseurs enrichissant la prévision
- **Prévision** : projection du rythme sur un horizon donné, avec intervalle de confiance
- **Détection d'anomalies** : identification des écarts significatifs entre rythme observé et rythme attendu (rupture de série)
- **Indice de régularité** : mesure de la prévisibilité du signal (ex. dispersion résiduelle après décomposition), traduite en score lisible pour l'utilisateur non technique

## MVP Restauration

- **Entrée** : upload CSV du nombre de couverts/CA par jour ou service → constitution d'une série temporelle par point de vente
- **Connecteurs (séries exogènes)** : météo, jours fériés/vacances scolaires (événements locaux → V2)
- **Fonctionnalités clés** :
  1. Décomposition du rythme (hebdo / saisonnier / événementiel / tendance)
  2. Indice de régularité du rythme (score de prévisibilité basé sur l'analyse des résidus)
  3. Alertes de rupture de rythme (détection d'anomalie sur la série)

## État d'avancement

- **Prototype technique validé** sur données réelles (Kaggle *Recruit Restaurant Visitor Forecasting*) : pipeline décomposition (STL) + prévision fonctionnel sur 800+ restaurants, testé en soumissions Kaggle réelles (têtes de leaderboard à 0.501 RMSLE, pour référence) — comparaison de 4 configurations, un modèle par restaurant à chaque fois (`src/forecast_submission_*.py`) :

  | Configuration | Private RMSLE | Public RMSLE |
  |---|---|---|
  | SARIMAX + jour férié | **0.57956** | **0.51544** |
  | SARIMAX seul (aucun exogène) | 0.58205 | 0.51553 |
  | SARIMAX + jour férié + météo | 0.59146 | 0.51875 |
  | Prophet + jour férié | 0.60576 | 0.55071 |

  Enseignements : le jour férié aide à peine (écart quasi nul avec "aucun exogène") ; la météo est **activement nuisible** en SARIMAX (pire que sans rien) — trop de paramètres à estimer sur un historique court par restaurant ; Prophet reste nettement en retrait des trois variantes SARIMAX. Le plus gros levier de progression vers 0.501 est probablement l'architecture (un modèle global type LightGBM sur tous les restaurants plutôt qu'un modèle isolé par série) plutôt que le choix SARIMAX/Prophet — voir "À trancher"
- **Web app locale fonctionnelle** (FastAPI + HTML/JS, `src/api.py` + `src/static/`) : décomposition (tendance/saisonnier/événementiel/résidu), indice de régularité, alertes de rupture, prévision avec intervalle de confiance, export CSV, thème clair/sombre/système
- **Variables exogènes opérationnelles mondialement**, ajoutées une par une par l'utilisateur (aucune active par défaut), à partir d'une ville/code postal n'importe où dans le monde (géocodage Open-Meteo) : jours fériés (bibliothèque `holidays`, tous pays supportés, avec Alsace-Moselle pour la France), météo (historique réelle + normales saisonnières pour le futur, mondiale) ; vacances scolaires reste France uniquement (par académie, seule source de données intégrée — indisponible ailleurs, retirée de la liste des variables proposées hors France)
- **Modèle validé contre une baseline naïve** : SARIMAX bat "même jour la semaine dernière" sur 91% des restaurants testés (818, backtest 28 jours), amélioration médiane +28,4% sur le MAE — voir `src/baseline_comparison.py` ; comparaison aussi exposée en direct dans l'app, par restaurant
- **Second modèle de prévision : Prophet** — l'utilisateur choisit SARIMAX ou Prophet pour générer la prévision (section 4), et le backtest (section 5) compare désormais les deux modèles à la baseline naïve simultanément, pour voir concrètement lequel gagne sur telle série précise. Prophet reçoit les mêmes exogènes que SARIMAX (jours fériés/vacances/météo, en `add_regressor`) — comparaison à jeu de données égal
- **Persistance locale** (SQLite, `src/storage.py`) : un CSV importé peut être sauvegardé comme "point de vente" (nom + série + localisation) et ré-analysé plus tard sans le ré-importer — pas encore de compte utilisateur, base locale partagée
- **Import CSV multi-entités, à deux dimensions** : deux colonnes optionnelles ("magasin" et "catégorie", ex. famille de produits) permettent d'importer un fichier au format long avec jusqu'à deux niveaux de regroupement (une ligne par magasin × catégorie × jour, ex. dataset Kaggle *Store Sales - Time Series Forecasting*, 54 magasins × 33 catégories) — les colonnes candidates sont détectées automatiquement, l'utilisateur choisit ensuite la combinaison précise à analyser, comme pour le restaurant de démo. Un bouton dédié génère aussi un CSV de prévisions pour **toutes** les combinaisons du fichier en une fois (modèle au choix, job en arrière-plan avec suivi de progression, sans limite de taille — peut prendre plusieurs minutes sur un gros fichier)
- **Variables exogènes venues du fichier importé lui-même** : les colonnes numériques déjà présentes dans le CSV (ex. `onpromotion`) peuvent être ajoutées comme régresseurs sans ressaisie — mais leur valeur future reste inconnue de Cadence (dernière valeur observée reportée sur tout l'horizon, faute de mieux). Un second fichier externe peut aussi être uploadé et joint automatiquement, de deux façons : soit une série globale alignée par date seule (ex. prix du pétrole, une valeur pour tout le monde), soit — si le fichier a ses propres colonnes magasin/catégorie — une vraie valeur par entité et par date, y compris sur l'horizon à prévoir (ex. le calendrier de promotions réel des jours à venir, connu à l'avance dans une compétition Kaggle). Une vérification automatique du format s'affiche à l'ajout (colonnes trouvées, nombre d'entités détectées, nombre de jours de l'horizon réellement couverts par de vraies valeurs futures vs reportées). Testé sur *Store Sales*, SARIMAX brut sans aucune de ces exogènes en baseline : 0.42726 RMSLE (0.37294 en tête de leaderboard) ; `onpromotion` joint par date seule (moyenne nationale, pas la vraie valeur par magasin) a donné 0.44522, et la jointure corrigée par entité 0.44460 — quasiment aucune amélioration malgré la correction, ce qui déplace le diagnostic : le problème n'est pas la qualité de la donnée mais la difficulté de SARIMAX à intégrer un régresseur de plus sur des séries courtes (420 `ConvergenceWarning` sur 1782 fits) — même constat structurel que la météo sur *Recruit*
- **Troisième modèle réservé à la génération en masse : LightGBM** (`src/lightgbm_forecast.py`) — un seul modèle entraîné sur toutes les entités du fichier à la fois (retards 1/7/14/28j, moyennes glissantes 7/28j, calendrier, entités en catégorielles natives, cible en `log1p`), au lieu d'un fit indépendant par entité comme SARIMAX/Prophet. Prévision multi-jours récursive et vectorisée (une fenêtre glissante bornée par entité, recalculée pour toutes les entités à la fois à chaque jour d'horizon — jamais de boucle par entité). Disponible uniquement dans la génération de CSV en masse (section 4), pas dans l'analyse d'une entité (décomposition/régularité/anomalies restent SARIMAX/Prophet, qui n'ont de sens que sur une série isolée). 78s pour les 1782 entités de *Store Sales*, contre 20+ min pour l'équivalent SARIMAX
  - **Validé sur Kaggle, résultat mitigé.** Un premier essai a donné 1.128 RMSLE — très dégradé, signe d'un bug plutôt que d'une faiblesse de modèle (confirmé par un backtest local à 0.428, correct). Cause trouvée : la boucle de prévision récursive trie ses lignes en interne pour calculer les features, puis réutilisait par erreur l'ordre non trié pour étiqueter les résultats — des prévisions de magasins à gros volume atterrissaient sur des magasins à volume quasi nul, et inversement (corrigé, test de régression ajouté). Une fois corrigé : **0.55166 RMSLE — toujours moins bon que SARIMAX** (0.42726 baseline, 0.44460 avec `onpromotion`). Le bug expliquait l'essentiel de l'écart catastrophique, mais un vrai écart de qualité de modèle restait en dessous — voir "Prochaines étapes"
- CI GitHub Actions (tests automatiques à chaque push/PR sur `main`)
- Pour lancer en local : `uvicorn api:app` depuis `src/` (dépendances dans `requirements.txt`, tests dans `requirements-dev.txt`)

## À trancher

- Modèle économique (freemium, SaaS B2B pur, etc.)
- Mode d'entrée des données à terme (upload vs connexion source, avec granularité temporelle variable — jour vs service)
- Faisabilité d'export des données côté restaurateurs (formats disponibles depuis les POS/caisses usuels)
- Choix des méthodes de modélisation (statistiques classiques type SARIMA/Prophet vs approches ML) — SARIMAX et Prophet coexistent pour l'analyse d'une entité (l'utilisateur compare), LightGBM (global, plusieurs entités à la fois) pour la génération en masse ; testé sur Kaggle, ne bat pas encore SARIMAX (voir "Prochaines étapes") — reste le choix par défaut recommandé pour l'instant
- Besoin réel du B2C
- Disponibilité domaine/marque "Cadence" (INPI)
- Comptes utilisateurs / authentification : la persistance actuelle (SQLite) n'a pas de notion de compte, une seule base locale partagée par tous les points de vente

## Prochaines étapes

- **LightGBM ne bat pas encore SARIMAX** (0.55166 vs 0.42726) — SARIMAX reste le choix par défaut recommandé en pratique. Pistes identifiées mais pas encore essayées : un retard annuel (`lag_365`, aucun signal saisonnier long terme actuellement — seulement 1/7/14/28 jours, alors que le dataset couvre ~4,5 ans) ; réglage d'hyperparamètres/validation croisée (actuellement fixes) ; le biais de sur-prédiction observé sur les combinaisons magasin×catégorie à très faible volume (le modèle généralise depuis les magasins à fort volume de la même catégorie) pourrait aussi valoir une features spécifique (ex. volume historique moyen de l'entité). Restent aussi en dette technique v1 : intervalles de confiance par régression quantile (aucun actuellement)
- Comptes utilisateurs si l'app doit être partagée entre plusieurs restaurateurs (la persistance locale actuelle suppose un seul utilisateur)
- Continuer à corriger les bugs de robustesse (formats de date non standards, etc. — le séparateur CSV `;` est déjà corrigé)
- Identifier et contacter des restaurateurs cibles pour valider le besoin réel (aucun contact identifié à ce jour — volontairement mis de côté tant que le produit n'est pas jugé assez solide)
- Vérifier domaine/marque
