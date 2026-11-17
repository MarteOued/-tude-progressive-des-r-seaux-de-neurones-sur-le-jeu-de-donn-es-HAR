# Étude progressive des réseaux de neurones sur le jeu de données HAR

Projet de Deep Learning du M2 MALIA (2026–2027).

On cherche à reconnaître l'activité d'une personne à partir des capteurs de son smartphone (accéléromètre et gyroscope). Il y a 6 classes : WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
Le projet compare des réseaux de neurones multicouches (MLP) de complexité croissante à une régression logistique de référence, avec une méthodologie rigoureuse : validation par volontaire, plusieurs graines aléatoires, et jeu de test utilisé une seule fois.

## Données

Les données ne sont pas incluses dans ce dépôt. Pour les obtenir :

1. Télécharger le jeu de données sur Kaggle : <https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones>
2. Placer les fichiers ainsi :

```
HAR/
├── train.csv   (7352 lignes, 21 volontaires)
└── test.csv    (2947 lignes, 9 autres volontaires)
```

Chaque fichier contient 561 variables, plus les colonnes `subject` (le volontaire) et `Activity` (la classe à prédire).

## Installation

Python 3.12 :

```bash
pip install -r requirements.txt
# ou, avec uv :
uv pip install -r requirements.txt
```

## Structure

| Fichier | Contenu |
|---|---|
| `requirements.txt` | Bibliothèques Python et leurs versions |
| `1_une_couche_cachee.ipynb` | Notebook Version 1 : exploration des données (Partie 1 terminée), découpage par volontaire, régression logistique de référence. Réseau à une couche cachée en cours (Partie 2). |
| `2_reseau_profond.ipynb` | Notebook Version 2 : réseaux plus profonds (3.2 fait) ; régularisation, visualisation (t-SNE / UMAP) à venir |
