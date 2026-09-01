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

## À trancher

- Modèle économique (freemium, SaaS B2B pur, etc.)
- Mode d'entrée des données à terme (upload vs connexion source, avec granularité temporelle variable — jour vs service)
- Choix des méthodes de modélisation (statistiques classiques type SARIMA/Prophet vs approches ML) — arbitrage à faire selon volume de données client
- Besoin réel du B2C
- Disponibilité domaine/marque "Cadence" (INPI)

## Prochaines étapes

- Vérifier domaine/marque
- Préciser le besoin utilisateur restauration
- Esquisser dashboard / page d'accueil (visualisation de séries temporelles = enjeu UX central)
- Prototype sur le cas restauration, avec un premier pipeline de décomposition/prévision
