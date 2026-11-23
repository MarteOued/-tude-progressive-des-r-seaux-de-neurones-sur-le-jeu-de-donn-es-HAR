"""Vérifie que les résultats du rapport se reproduisent hors des notebooks.

À lancer depuis le dossier Projet/ :   python rapport/verifier_resultats.py
Réentraîne (5 graines) la référence à 3 couches, dropout 0,1, Batch Normalization et dropout 0,3 + arrêt
précoce, puis la régression logistique, avec le code des cellules du notebook 2_reseau_profond.ipynb.
Affiche accuracy de validation et de test (moyenne, écart-type), F1 de test et nombre de paramètres.
ATTENTION : ce script évalue le test. Il sert à vérifier la reproductibilité, pas à choisir un modèle.
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn, nbformat
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# On réutilise EXACTEMENT le code des cellules du notebook V2 (préparation, fonctions, classes)
nb = nbformat.read('2_reseau_profond.ipynb', 4)
g = {}
for i in (2, 3, 4, 8, 14, 30):
    src = nb.cells[i].source
    src = src.split("lignes, hist_profond")[0] if i == 14 else src.split("print(MLPProfond")[0] if i == 8 else src
    exec(src, g)
exec("R, T, S = nn.ReLU, nn.Tanh, nn.Sigmoid", g)
fit_reg, fit_profond, SEEDS = g['fit_reg'], g['fit_profond'], g['SEEDS']
test, scaler, feature_cols, encoder = g['test'], g['scaler'], g['feature_cols'], g['encoder']
Xt = torch.tensor(scaler.transform(test[feature_cols].to_numpy()), dtype=torch.float32)
yt = encoder.transform(test['Activity'])
def pred(m):
    m.eval()
    with torch.no_grad(): return m(Xt).argmax(1).numpy()
params = lambda m: sum(p.numel() for p in m.parameters())
out = {}
cfgs = {"réf 3 couches": {}, "dropout 0,1": dict(dropout=0.1), "batch norm": dict(bn=True), "dropout 0,3 + ES": dict(dropout=0.3, early=True)}
for nom, kw in cfgs.items():
    rows = []
    for s in SEEDS:
        m, h = fit_reg(seed=s, **kw); i = h["ep_retenue"]
        p = pred(m)
        rows.append((h["val_acc"][i], accuracy_score(yt, p), f1_score(yt, p, average="macro"), h["train_acc"], params(m)))
    va = np.array([r[0] for r in rows]); ta = np.array([r[1] for r in rows]); tf = np.array([r[2] for r in rows])
    out[nom] = dict(val=(va.mean(), va.std(ddof=1)), test=(ta.mean(), ta.std(ddof=1)), f1t=(tf.mean(), tf.std(ddof=1)), params=rows[0][4])
    print(nom, {k: (tuple(round(x, 4) for x in v) if isinstance(v, tuple) else v) for k, v in out[nom].items()})
    if nom in ("réf 3 couches", "batch norm"):
        tr = np.mean([r[3] for r in rows], axis=0)
        print("   acc train moyenne : première époque >= 0.98 :", int(np.argmax(tr >= 0.98)) + 1 if (tr >= 0.98).any() else None)
# réseau 1 couche, 128 neurones (V1) et 512
for n in (16, 64, 128, 512):
    print("1 couche", n, "neurones : paramètres =", 547 * n + 6)
lr = LogisticRegression(max_iter=2000, random_state=42).fit(g['X_tr_s'], g['y_tr'])
print("LR params", lr.coef_.size + lr.intercept_.size, "test", accuracy_score(yt, lr.predict(Xt.numpy())))
# paramètres des architectures retenues
print("3x128 ReLU :", params(g['MLPReg'](540, 128, 6, 3)), "| BN :", params(g['MLPReg'](540, 128, 6, 3, bn=True)))
