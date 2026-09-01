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
- Pour lancer en local : `uvicorn api:app` depuis `src/` (dépendances dans `requirements.txt`)

## À trancher

- Modèle économique (freemium, SaaS B2B pur, etc.)
- Mode d'entrée des données à terme (upload vs connexion source, avec granularité temporelle variable — jour vs service)
- Faisabilité d'export des données côté restaurateurs (formats disponibles depuis les POS/caisses usuels)
- Choix des méthodes de modélisation (statistiques classiques type SARIMA/Prophet vs approches ML) — arbitrage à faire selon volume de données client ; SARIMAX retenu pour le prototype actuel, pas encore comparé à une baseline naïve
- Besoin réel du B2C
- Disponibilité domaine/marque "Cadence" (INPI)
- Persistance / comptes utilisateurs : l'app actuelle est sans état (chaque analyse repart de zéro, pas d'historique ni de compte)

## Prochaines étapes

- Comparer le modèle SARIMAX à une baseline naïve (ex. "même jour la semaine dernière") pour valider qu'il apporte une vraie valeur ajoutée
- Corriger les bugs de robustesse connus (ex. CSV avec `;` comme séparateur, formats de date non standards)
- Réfléchir à la persistance (historique des analyses, notion de compte/point de vente) pour que l'app soit utilisable dans la durée, pas juste en aller-retour
- Identifier et contacter des restaurateurs cibles pour valider le besoin réel (aucun contact identifié à ce jour — volontairement mis de côté tant que le produit n'est pas jugé assez solide)
- Vérifier domaine/marque
