"""Régénère les figures du rapport (rapport/figures/).

À lancer depuis le dossier Projet/ :   python rapport/generer_figures.py
Prérequis : HAR/train.csv, HAR/test.csv et les deux notebooks exécutés (leurs images servent aux annexes).
Les fonctions d'entraînement ne sont pas réécrites : le script exécute les cellules du notebook
2_reseau_profond.ipynb qui les définissent (préparation des données, MLPReg, fit_reg...), donc le code
est exactement celui du notebook. Durée : quelques minutes (entraînements + UMAP).
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, nbformat, base64, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
OUT = "rapport/figures/"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
nb = nbformat.read('2_reseau_profond.ipynb', 4)
g = {}
for i in (2, 3, 4, 8, 14, 30):
    src = nb.cells[i].source
    src = src.split("lignes, hist_profond")[0] if i == 14 else src.split("print(MLPProfond")[0] if i == 8 else src
    exec(src, g)
fit_reg, SEEDS = g['fit_reg'], g['SEEDS']
test, scaler, feature_cols, encoder = g['test'], g['scaler'], g['feature_cols'], g['encoder']
Xt = torch.tensor(scaler.transform(test[feature_cols].to_numpy()), dtype=torch.float32)
yt = encoder.transform(test['Activity'])
classes = [c.replace("WALKING_", "W_") for c in encoder.classes_]
BLUE, ORANGE, GREY = "#1f77b4", "#d95f02", "#777777"

# ---- Figure 2 du rapport : validation contre test (valeurs vérifiées, tableaux 5.2 des deux notebooks)
mods = [("rég. logistique", .9589, None, .9535, None),
        ("1 couche, 128", .9592, .0030, .9427, .0024), ("1 couche, 512", .9622, .0032, .9458, .0019),
        ("1 couche, Sigmoid 64", .9647, .0046, .9469, .0012),
        ("3 couches, sans régul.", .9545, .0041, .9379, .0023), ("3 c. + dropout 0,1", .9670, .0055, .9407, .0044),
        ("3 c. + batch norm", .9667, .0054, .9482, .0054), ("3 c. + dropout 0,3 + ES", .9723, .0036, .9406, .0008)]
fig, ax = plt.subplots(figsize=(6.7, 2.7))
y = np.arange(len(mods))[::-1]
for k, (n, v, vs, t, ts) in zip(y, mods):
    ax.plot([t, v], [k, k], color="#bbbbbb", lw=1, zorder=1)
    ax.errorbar(v, k, xerr=vs, fmt="o", color=BLUE, ms=4, capsize=2, zorder=2, label="validation (5 volontaires)" if k == y[0] else None)
    ax.errorbar(t, k, xerr=ts, fmt="s", color=ORANGE, ms=4, capsize=2, zorder=2, label="test (9 volontaires)" if k == y[0] else None)
ax.set_yticks(y); ax.set_yticklabels([m[0] for m in mods])
ax.set_xlabel("accuracy (moyenne ± écart-type sur 5 graines)")
ax.axvline(.9535, color=GREY, ls=":", lw=.8)
ax.legend(loc="upper left", frameon=False)
plt.tight_layout(); plt.savefig(OUT + "fig2_val_test.png", dpi=250); plt.close()

# ---- Figure 1 du rapport : courbes d'apprentissage (moyenne ± écart-type sur 5 graines)
cfg = {"sans régularisation": dict(), "dropout 0,1": dict(dropout=0.1), "batch norm": dict(bn=True)}
H = {n: [fit_reg(seed=s, **kw)[1] for s in SEEDS] for n, kw in cfg.items()}
fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.2))
for (n, hs), c in zip(H.items(), [GREY, ORANGE, BLUE]):
    for ax, key, titre in [(axes[0], "val_loss", "perte de validation"), (axes[1], "train_acc", "accuracy d'entraînement"), (axes[2], "val_acc", "accuracy de validation")]:
        m = np.mean([h[key] for h in hs], 0); s = np.std([h[key] for h in hs], 0, ddof=1)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, color=c, label=n, lw=1.2); ax.fill_between(x, m - s, m + s, color=c, alpha=.18)
        ax.set_title(titre); ax.set_xlabel("époque")
axes[0].set_yscale("log"); axes[1].set_ylim(.93, 1.002); axes[2].set_ylim(.90, .98)
axes[0].legend(frameon=False)
plt.tight_layout(); plt.savefig(OUT + "fig1_courbes.png", dpi=250); plt.close()

# ---- Figure 3 du rapport : matrices de confusion sur le test (réseau choisi sur la validation, 5 graines, contre régression logistique)
cm_net = 0; preds0 = None
for s in SEEDS:
    m, h = fit_reg(seed=s, dropout=0.3, early=True); m.eval()
    with torch.no_grad(): p = m(Xt).argmax(1).numpy()
    if s == 0: preds0, model0 = p, m
    cm_net = cm_net + confusion_matrix(yt, p, labels=range(6))
lr = LogisticRegression(max_iter=2000, random_state=42).fit(g['X_tr_s'], g['y_tr'])
cm_lr = confusion_matrix(yt, lr.predict(Xt.numpy()), labels=range(6))
fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9))
for a, cm, t in [(axes[0], cm_net, "MLP 3 couches, dropout 0,3 + ES (5 graines)"), (axes[1], cm_lr, "régression logistique")]:
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=classes, yticklabels=classes,
                ax=a, cbar=False, annot_kws={"size": 6})
    a.set_title(t, fontsize=7.5); a.set_xlabel("classe prédite"); a.tick_params(axis="x", rotation=45, labelsize=6); a.tick_params(axis="y", labelsize=6)
axes[1].set_ylabel(""); axes[0].set_ylabel("vraie classe")
plt.tight_layout(); plt.savefig(OUT + "fig3_confusion.png", dpi=250); plt.close()

# ---- Figure 4 du rapport : UMAP des variables d'entrée et de la dernière couche cachée (graine 0, mêmes réglages que le notebook)
import umap
with torch.no_grad(): Hh = model0.reseau[:-1](Xt).numpy()
err = preds0 != yt
print("erreurs graine 0 :", err.sum())
fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0))
cols = plt.cm.tab10(np.arange(6))
for a, X, t in [(axes[0], Xt.numpy(), "variables d'entrée (540)"), (axes[1], Hh, "dernière couche cachée (128)")]:
    E = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)
    for c in range(6):
        a.scatter(E[yt == c, 0], E[yt == c, 1], s=2.5, color=cols[c], label=classes[c], alpha=.7, rasterized=True)
    a.scatter(E[err, 0], E[err, 1], s=9, marker="x", color="k", linewidths=.5, label="erreur du réseau")
    a.set_title("UMAP sur " + t); a.set_xticks([]); a.set_yticks([])
axes[0].legend(markerscale=3, frameon=False, fontsize=5.5, loc="best")
plt.tight_layout(); plt.savefig(OUT + "fig4_umap.png", dpi=250); plt.close()

# ---- Annexes : images déjà produites dans les notebooks (extraites telles quelles)
def extraire(nbfile, idx, nom):
    n = nbformat.read(nbfile, 4)
    for o in n.cells[idx].outputs:
        if "data" in o and "image/png" in o["data"]:
            open(OUT + nom, "wb").write(base64.b64decode(o["data"]["image/png"])); print("annexe", nom); return
    raise SystemExit("pas d'image " + nom)
for idx, nom in [(18, "annexe_profondeur.png"), (41, "annexe_regularisation.png"), (79, "annexe_tsne_umap.png"), (93, "annexe_reglages.png"), (105, "annexe_importance.png")]:
    extraire('2_reseau_profond.ipynb', idx, nom)
for idx, nom in [(114, "annexe_graines_v1.png"), (134, "annexe_tailles_v1.png")]:
    extraire('1_une_couche_cachee.ipynb', idx, nom)
print("fini")
