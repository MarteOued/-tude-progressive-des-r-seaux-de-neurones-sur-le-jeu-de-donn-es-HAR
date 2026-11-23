# Étude progressive des réseaux de neurones sur le jeu de données HAR

Projet de Deep Learning, M2 MALIA (2026–2027). Auteure : Martine Ouedraogo. Enseignant : Walid Bechkit.

## 1. En deux minutes

**Le problème.** À partir des mesures de l'accéléromètre et du gyroscope d'un smartphone, on veut reconnaître ce que fait la personne : marcher, monter ou descendre un escalier, être assise, debout ou allongée (6 classes).

**Ce qu'on fait.** On part d'un modèle simple (une régression logistique), puis on construit des réseaux de neurones de plus en plus complexes (un MLP, c'est-à-dire un réseau de couches « entièrement connectées ») : plus de neurones, plus de couches, puis des techniques pour limiter le surapprentissage (dropout, Batch Normalization, arrêt précoce...). À chaque étape, on se demande si la complexité apporte vraiment quelque chose.

**Ce qu'on trouve.**
- Plus de neurones ou de couches n'améliore pas la validation au-delà du bruit.
- La régularisation améliore la validation de 0,8 à 1,8 point, mais presque plus sur le test.
- Le modèle le mieux classé en validation (0,9723) fait 0,9406 sur le test, sous la régression logistique (0,9535).
- Les erreurs se concentrent sur SITTING / STANDING et sur les trois types de marche.

Le détail est dans le rapport : [rapport/rapport.pdf](rapport/rapport.pdf).

## 2. Les fichiers et dossiers

```
Projet/
├── README.md                    ce fichier
├── requirements.txt             bibliothèques Python et versions exactes
├── .gitignore                   fichiers non versionnés (données, environnement virtuel...)
│
├── 1_une_couche_cachee.ipynb    NOTEBOOK 1 : données, référence, réseau à UNE couche cachée
├── 2_reseau_profond.ipynb       NOTEBOOK 2 : réseaux profonds, régularisation, test, t-SNE/UMAP, bonus
│
├── HAR/                         DONNÉES (à télécharger, non versionnées, voir section 3)
│   ├── train.csv                7352 lignes, 21 volontaires
│   └── test.csv                 2947 lignes, 9 autres volontaires
│
└── rapport/
    ├── rapport.pdf              le rapport final (5 pages)
    ├── annexes.pdf              figures complémentaires, hors des 6 pages
    ├── rapport.html             source du rapport (le PDF en est l'impression)
    ├── annexes.html             source des annexes
    ├── figures/                 images du rapport (fig1 à fig4) et des annexes (annexe_*)
    ├── generer_figures.py       régénère les images du dossier figures/
    └── verifier_resultats.py    réentraîne les modèles clés et vérifie qu'on retrouve les chiffres du rapport
```

**Dans quel ordre lire ?** Notebook 1, puis notebook 2, puis le rapport. Le notebook 2 reprend exactement la préparation des données du notebook 1 (mêmes volontaires, même standardisation), donc leurs scores sont comparables.

## 3. Données

Les fichiers CSV ne sont pas dans le dépôt (trop gros). À télécharger sur Kaggle : <https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones>, puis à placer ainsi :

```
Projet/HAR/train.csv
Projet/HAR/test.csv
```

Chaque ligne est une **fenêtre** de 2,56 secondes de signal (deux fenêtres consécutives se chevauchent de moitié). Elle est décrite par 561 variables calculées à l'avance (moyenne, écart-type, énergie... de chaque signal), plus deux colonnes :
- `subject` : le volontaire (un numéro). On ne s'en sert **pas** comme entrée du modèle, seulement pour découper les données.
- `Activity` : la classe à prédire.

Le test contient des volontaires qui ne sont **jamais** dans l'entraînement : il mesure donc la capacité à reconnaître des personnes inconnues.

## 4. Installation et lancement

Il faut Python 3.12. Depuis le dossier `Projet/` :

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt    # ou : uv pip install -r requirements.txt
jupyter lab                        # puis ouvrir un notebook et "Run All"
```

Points importants :
- **Lancer Jupyter depuis `Projet/`** : les notebooks lisent `HAR/train.csv` avec un chemin relatif.
- Les notebooks sont **déjà exécutés** : on peut les lire sans rien relancer. Les relancer entièrement prend de l'ordre de plusieurs dizaines de minutes sur un ordinateur portable (CPU, pas de GPU nécessaire ; durée non mesurée précisément).
- Les résultats sont reproductibles : toutes les graines aléatoires sont fixées et les versions des bibliothèques sont dans `requirements.txt`. Une autre machine ou une autre version de PyTorch peut donner des différences très faibles.

## 5. Les conventions du projet (à connaître pour lire le code)

| Convention | Valeur | Pourquoi |
|---|---|---|
| Variables d'entrée | 540 (561 moins 21 colonnes qui sont des copies exactes d'une autre) | Les copies n'apportent aucune information |
| Standardisation | `StandardScaler` appris sur l'entraînement seul | Rien de la validation ni du test ne doit fuir dans l'entraînement |
| Validation | 5 volontaires entiers : 1, 3, 15, 25, 27 (1801 fenêtres) | Des fenêtres voisines d'un même volontaire se ressemblent : un tirage aléatoire de lignes surestimerait les scores |
| Entraînement | 16 autres volontaires (5551 fenêtres) | |
| Test | utilisé **une seule fois par modèle**, en fin de projet (partie 5.2) | Il ne doit servir à choisir aucun réglage |
| Graines | 5 graines : 0, 1, 2, 3, 4 | On rapporte « moyenne ± écart-type » sur ces 5 entraînements |
| Réglages par défaut des réseaux | Adam, pas 3e-4, mini-batches de 64, 30 époques, perte `CrossEntropyLoss` | Réglage de l'optimiseur fait en partie 2.2 |
| Métriques | accuracy et F1 macro | Le F1 macro compte chaque classe de la même façon |
| Règle « différence nette » | un écart entre deux configurations est jugé net s'il dépasse **2 × le plus grand écart-type** des deux | Avec 5 graines, un petit écart peut n'être que du hasard. Ce n'est pas un test statistique |

La numérotation des sections des notebooks suit celle de l'énoncé du projet (Partie 1, question 1.1, étape 1.1.1...). Chaque partie commence par un encadré **« Demandé »** qui recopie la consigne, et le code est précédé d'une explication en français.

## 6. Notebook 1 : `1_une_couche_cachee.ipynb`

| Section | Contenu |
|---|---|
| Préliminaires | Imports, graine |
| **1. Jeu de données** | |
| 1.1 | Chargement, aperçu |
| 1.2 | Exploration : nombre de lignes et de volontaires, classes, statistiques, histogrammes, boxplots, corrélations, qualité des données, choix du prétraitement |
| 1.3 | Construction de la validation par volontaire ; comparaison avec un découpage aléatoire |
| 1.4 | Régression logistique de référence (3 246 paramètres) |
| **2. Un réseau à une couche cachée** | |
| 2.1 | Construction (540 → 64 → 6), vérification de la perte initiale (attendue : ln 6 ≈ 1,79) et du surapprentissage d'un petit lot |
| 2.2 | Comparaison de 3 activations × 3 optimiseurs, avec réglage du pas d'apprentissage ; accuracy contre F1 |
| 2.3 | Répétition avec 5 graines ; règle des 2 écarts-types ; variabilité due au choix des volontaires |
| **3.1** | Taille de la couche cachée (16 à 512 neurones), nombre de paramètres, surapprentissage |
| **5. Visualisation** | |
| 5.1 | Courbes d'apprentissage (16, 128, 512 neurones) |
| 5.2 | **Évaluation sur le test** (seule cellule qui le touche), matrices de confusion, réponse à la question 1.3 (b) |

## 7. Notebook 2 : `2_reseau_profond.ipynb`

| Section | Contenu |
|---|---|
| Préparation | Même code que le notebook 1 (données, découpage, standardisation, fonctions d'entraînement) |
| **3.2** | Réseaux de 1 à 4 couches cachées avec 16 combinaisons d'activations ; temps et nombre de paramètres |
| **4. Régularisation** | 4.1 : dropout, *weight decay*, Batch Normalization, arrêt précoce (14 configurations × 5 graines) ; 4.2 : effet sur la généralisation et la stabilité, choix de 3 configurations |
| **5. Visualisation** | 5.1 : courbes d'apprentissage ; 5.2 : **évaluation sur le test** des 3 configurations retenues + références, matrices de confusion, classes difficiles |
| **6. Interprétabilité** | t-SNE et UMAP sur les variables d'entrée et sur la dernière couche cachée ; silhouette ; lien avec la matrice de confusion ; effet de la perplexité et des graines |
| **7. Bonus** | 7.1 : LayerNorm contre Batch Normalization ; 7.2 : importance de groupes de variables par permutation |

## 8. Le code : fonctions et classes importantes

Le même schéma revient dans les deux notebooks : **un modèle (classe) + une fonction d'entraînement `fit...` qui renvoie le modèle et son historique**.

| Nom | Notebook | Rôle |
|---|---|---|
| `MLP(n_entrees, n_caches, n_classes, activation)` | 1 | Réseau à une couche cachée : `Linear → activation → Linear`. Sort des scores bruts (le softmax est dans la perte) |
| `train_epoch(model, loader, optimizer, lossfn)` | 1 et 2 | Une époque : pour chaque mini-batch, remise à zéro des gradients, passage avant, perte, rétropropagation, mise à jour |
| `evaluate(model, loader, lossfn)` | 1 et 2 | Perte, accuracy et F1 macro sur un jeu de données, sans calcul de gradient |
| `fit(activation, optimiseur, lr, n_epochs, n_caches, seed)` | 1 | Entraîne un MLP à une couche ; renvoie `(modèle, historique)`, l'historique contenant la perte et l'accuracy par époque |
| `MLPProfond(n_entrees, largeur, n_classes, activations)` | 2 | Réseau dont la liste `activations` fixe à la fois la profondeur et l'activation de chaque couche |
| `fit_profond(activations, lr, ..., seed)` | 2 | Entraînement correspondant |
| `MLPReg(..., n_couches, dropout, bn)` | 2 | Réseau à `n_couches` couches `Linear → (BatchNorm) → ReLU → (Dropout)` |
| `fit_reg(dropout, wd, bn, early, lr, seed, ...)` | 2 | Entraînement avec les options de régularisation ; `wd` = *weight decay*, `early` = arrêt précoce (patience 8 sur le F1 de validation, poids de la meilleure époque restaurés) |
| `MLPNorm`, `fit_norme(norme, seed)` | 2 | Variante avec `norme` = `None`, `"batch"` ou `"layer"` (partie 7.1) |
| `RETENUES` | 2 | Dictionnaire des 3 configurations retenues en 4.2.3, réentraînées en 5.2 |
| `moy_std(serie)` | 1 et 2 | Met en forme « moyenne ± écart-type » (écart-type à n−1) |
| `melange(E, y)` | 2 | Pour chaque classe, la part des 10 plus proches voisins (dans le plan 2D) qui sont d'une autre classe |

Dans les deux notebooks, les résultats de chaque série d'entraînements sont rangés dans un tableau `pandas` (un `DataFrame`) avec une ligne par entraînement (configuration, graine, scores). Les tableaux affichés sont des regroupements de ces `DataFrame`.

## 9. Le rapport et ses scripts

- `rapport/rapport.pdf` : article court (5 pages), avec la déclaration d'usage de l'IA et la façon dont les résultats ont été vérifiés.
- `rapport/annexes.pdf` : 7 figures supplémentaires (profondeur, régularisation, tailles, importance, t-SNE...).
- Le PDF est l'impression de `rapport.html` (navigateur Chrome ou Edge en mode sans fenêtre, par exemple : `"…/Microsoft Edge" --headless --no-pdf-header-footer --print-to-pdf=rapport.pdf file://$PWD/rapport/rapport.html`).
- `python rapport/generer_figures.py` (à lancer depuis `Projet/`) régénère `rapport/figures/`. Il n'invente rien : il exécute les cellules du notebook 2 qui définissent les modèles, réentraîne, puis dessine. Les figures d'annexe sont extraites telles quelles des notebooks.
- `python rapport/verifier_resultats.py` réentraîne cinq modèles clés et affiche leurs scores, à comparer aux tableaux du rapport. Il évalue le test : il sert à vérifier, pas à choisir un modèle.

## 10. Résultats principaux

Accuracy, moyenne ± écart-type sur 5 graines (la régression logistique n'a pas de graine). Le test porte sur 9 volontaires jamais vus.

| Modèle | Paramètres | Validation | Test |
|---|---|---|---|
| Régression logistique | 3 246 | 0,9589 | **0,9535** |
| 1 couche, 128 neurones (ReLU) | 70 022 | 0,9592 ± 0,0030 | 0,9427 ± 0,0024 |
| 3 couches, sans régularisation | 103 046 | 0,9545 ± 0,0041 | 0,9379 ± 0,0023 |
| 3 couches + Batch Normalization | 103 814 | 0,9667 ± 0,0054 | 0,9482 ± 0,0054 |
| 3 couches + dropout 0,1 | 103 046 | 0,9670 ± 0,0055 | 0,9407 ± 0,0044 |
| 3 couches + dropout 0,3 + arrêt précoce | 103 046 | 0,9723 ± 0,0036 | 0,9406 ± 0,0008 |

Les autres tableaux sont dans les notebooks et le rapport.

## 11. Limites à garder en tête

- La validation ne contient que 5 volontaires : avec la régression logistique, 10 tirages différents de 5 volontaires donnent de 0,862 à 0,963 d'accuracy. Le choix des volontaires pèse plus que celui de la graine ou d'une petite amélioration du réseau.
- Le test ne contient que 9 volontaires : un écart de moins d'un point entre deux modèles n'est pas fiable.
- On n'a pas fait de recherche automatique d'hyperparamètres (elle s'ajusterait aux mêmes 5 volontaires), ni de réseau sur les signaux bruts (absents des fichiers fournis).
- Les explications par les capteurs (gravité, inclinaison du téléphone) et par l'affaiblissement des gradients sont des **hypothèses** cohérentes avec les mesures, pas des démonstrations.

## 12. Lexique

- **Fenêtre** : un morceau de 2,56 s de signal, une ligne du fichier.
- **Volontaire** : la personne qui porte le téléphone ; sert à découper les données.
- **Époque** : un passage complet sur les données d'entraînement.
- **Graine** : nombre qui fixe le hasard (initialisation du réseau, ordre des exemples). Changer la graine donne un entraînement légèrement différent.
- **Surapprentissage** : le modèle réussit très bien sur l'entraînement mais moins bien sur des données nouvelles.
- **Dropout, weight decay, Batch Normalization, arrêt précoce** : techniques pour limiter le surapprentissage.
- **F1 macro** : moyenne des F1 de chaque classe (un F1 combine précision et rappel).
- **t-SNE, UMAP** : méthodes qui dessinent des données de grande dimension en 2D en gardant les voisinages.
