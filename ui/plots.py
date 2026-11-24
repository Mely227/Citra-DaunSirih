"""Grafik matplotlib (tanpa dependensi Streamlit)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN = "#198754"


def confusion_figure(cm, labels):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    thr = cm.max() / 2.0 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#212529", fontsize=11)
    fig.tight_layout()
    return fig


def heatmap_figure(cv_df):
    """Heatmap akurasi CV kernel RBF (C vs gamma). None bila RBF tidak dipakai."""
    if "param_svc__kernel" not in cv_df.columns or "param_svc__gamma" not in cv_df.columns:
        return None
    d = cv_df[cv_df["param_svc__kernel"] == "rbf"].copy()
    if d.empty:
        return None
    d["gamma"] = d["param_svc__gamma"].astype(str)
    d["C"] = d["param_svc__C"].astype(float)
    pv = d.pivot_table(index="C", columns="gamma", values="mean_test_score", aggfunc="max")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(pv.values, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([("%g" % c) for c in pv.index])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, "%.3f" % v, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Akurasi CV")
    fig.tight_layout()
    return fig


def svm_boundary_figure(clf, X, y):
    """Daerah keputusan, margin (garis putus-putus), dan support vector."""
    pad = 0.7
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 260),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 260))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#d1e7dd", "#ffe8cc"], alpha=0.85)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["#0f5132", "#212529", "#0f5132"],
               linestyles=["--", "-", "--"], linewidths=[1.2, 2.2, 1.2])
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=GREEN, edgecolor="white", s=42, label="Kelas A")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#fd7e14", edgecolor="white", s=42, label="Kelas B")
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=170,
               facecolors="none", edgecolors="#212529", linewidths=1.4, label="Support vector")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def glcm_figure(P, title=""):
    """Heatmap matriks GLCM (akar kuadrat agar nilai kecil tetap terlihat)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(np.sqrt(P), cmap="YlGn", origin="upper")
    ax.set_xlabel("Intensitas piksel tetangga (j)")
    ax.set_ylabel("Intensitas piksel acuan (i)")
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="sqrt(probabilitas)")
    fig.tight_layout()
    return fig
