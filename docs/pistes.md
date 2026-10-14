# Profil de proposition : Cadence

Ce qu'est une bonne piste pour Cadence. Lu par le bouton « Proposer des
pistes » de jdb (`journal-de-bord/scripts/machine.py`) : une règle de
proposition se change ici, pas dans sa consigne.

Le bouton lit ce fichier par programme : garder les quatre titres de rubrique
tels quels et n'y écrire que des puces ; dans « Champs en plus », la forme
`` `nom` (Libellé) : ce qu'on y met ``. Ces deux paragraphes ne sont pas lus.

## Priorités

- La précision des prévisions d'abord (#8, ticket permanent), mesurée au
  score officiel Kaggle sur Recruit et Store Sales. Références au
  10/10/2026, poids FFORMA par série : Store Sales 0.40371, Recruit 0.55349.
- Le produit avant la validation terrain (choix de l'utilisateur) : des
  pistes techniques ; la prospection peut être citée une fois, sans insister.
- Généraliste : toute série quotidienne date + valeur, aucun secteur dédié.

## À lire

- `Cadence/BENCHMARKS.md`, en entier : chaque score mesuré et ce qui a été
  essayé. C'est la liste de ce qui est déjà jugé côté précision.

## Bonne piste

- Une piste de précision se juge sur les fenêtres récentes et sur la même
  période un an plus tôt ; elle doit gagner sur au moins un des deux jeux
  sans dégrader l'autre, ou rester optionnelle. Elle dit l'ordre de
  grandeur espéré face au score de référence.
- Un autre réglage d'un axe déjà mesuré dans `BENCHMARKS.md` n'est pas une
  piste, sauf à dire précisément ce qui change.
- Rien de ce que les décisions actées excluent : connexion directe aux
  caisses, API Kaggle, préréglages d'exogènes par secteur.

## Champs en plus

- `mesure` (Mesure) : sur quel jeu et quel score elle se juge, la référence
  à battre et le gain espéré ; « hors précision » pour une piste de produit,
  avec ce qu'elle apporte à l'utilisateur.
- `cout` (Coût) : temps de calcul en plus, données à télécharger et d'où.
