# Cadence

Plateforme web de prévision de rythme d'activité, basée sur l'analyse de séries temporelles.

## Le concept

Cadence traduit une problématique d'**analyse de séries temporelles** en langage métier : ce que l'utilisateur perçoit intuitivement (rythme hebdomadaire, saisonnalité, pics événementiels) est en réalité un signal temporel structuré, que Cadence décompose, modélise et projette.

## Décisions actées

- **Nom** : Cadence (définitif)
- **Cible** : B2B et B2C, MVP centré B2B
- **Scope** : Option B — verticaux spécialisés par secteur, chaque module encodant les composantes temporelles propres à son activité (saisonnalité métier, cycles hebdomadaires, effets calendaires spécifiques)
- **Premier vertical MVP** : Restauration (prévision de couverts)

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

- **Prototype technique validé** sur données réelles (Kaggle *Recruit Restaurant Visitor Forecasting*) : pipeline décomposition (STL) + prévision (SARIMAX) fonctionnel sur 800+ restaurants, testé en soumission Kaggle réelle (score 0.580 RMSLE, à comparer au 0.501 en tête de leaderboard — bon baseline, pas optimisé)
- **Web app locale fonctionnelle** (FastAPI + HTML/JS, `src/api.py` + `src/static/`) : décomposition (tendance/saisonnier/événementiel/résidu), indice de régularité, alertes de rupture, prévision avec intervalle de confiance, export CSV, thème clair/sombre/système
- **Variables exogènes opérationnelles** pour données françaises, activables individuellement, à partir d'une simple ville/code postal saisie par l'utilisateur : jours fériés (avec Alsace-Moselle), vacances scolaires (par académie), météo (historique réelle + normales saisonnières pour le futur)
- **Modèle validé contre une baseline naïve** : SARIMAX bat "même jour la semaine dernière" sur 91% des restaurants testés (818, backtest 28 jours), amélioration médiane +28,4% sur le MAE — voir `src/baseline_comparison.py` ; comparaison aussi exposée en direct dans l'app, par restaurant
- **Persistance locale** (SQLite, `src/storage.py`) : un CSV importé peut être sauvegardé comme "point de vente" (nom + série + localisation) et ré-analysé plus tard sans le ré-importer — pas encore de compte utilisateur, base locale partagée
- CI GitHub Actions (tests automatiques à chaque push/PR sur `main`)
- Pour lancer en local : `uvicorn api:app` depuis `src/` (dépendances dans `requirements.txt`, tests dans `requirements-dev.txt`)

## À trancher

- Modèle économique (freemium, SaaS B2B pur, etc.)
- Mode d'entrée des données à terme (upload vs connexion source, avec granularité temporelle variable — jour vs service)
- Faisabilité d'export des données côté restaurateurs (formats disponibles depuis les POS/caisses usuels)
- Choix des méthodes de modélisation (statistiques classiques type SARIMA/Prophet vs approches ML) — arbitrage à faire selon volume de données client ; SARIMAX retenu pour le prototype actuel, validé contre une baseline naïve (voir État d'avancement)
- Besoin réel du B2C
- Disponibilité domaine/marque "Cadence" (INPI)
- Comptes utilisateurs / authentification : la persistance actuelle (SQLite) n'a pas de notion de compte, une seule base locale partagée par tous les points de vente

## Prochaines étapes

- Comptes utilisateurs si l'app doit être partagée entre plusieurs restaurateurs (la persistance locale actuelle suppose un seul utilisateur)
- Continuer à corriger les bugs de robustesse (formats de date non standards, etc. — le séparateur CSV `;` est déjà corrigé)
- Identifier et contacter des restaurateurs cibles pour valider le besoin réel (aucun contact identifié à ce jour — volontairement mis de côté tant que le produit n'est pas jugé assez solide)
- Vérifier domaine/marque
