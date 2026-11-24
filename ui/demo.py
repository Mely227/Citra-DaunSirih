"""Data & model mini untuk demo interaktif SVM (tanpa Streamlit)."""
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.svm import SVC

DATASETS = {"Dua kelompok (linear)": "blobs", "Dua bulan sabit": "moons", "Lingkaran bersarang": "circles"}


def make_data(kind, n=140, noise=0.2, seed=7):
    if kind == "moons":
        return make_moons(n_samples=n, noise=noise, random_state=seed)
    if kind == "circles":
        return make_circles(n_samples=n, noise=noise * 0.5, factor=0.45, random_state=seed)
    return make_blobs(n_samples=n, centers=[(-1.4, -1.0), (1.4, 1.0)],
                      cluster_std=0.55 + noise * 1.6, random_state=seed)


def fit_demo(X, y, kernel, C, gamma, degree):
    clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree).fit(X, y)
    return clf, {"n_sv": int(clf.n_support_.sum()), "acc": float(clf.score(X, y)), "n": len(y)}


def explain(kind, kernel, C, gamma, degree=3):
    """Penjelasan dinamis (jenis, teks) agar pengguna paham efek parameter."""
    tips = []
    nonlinear = kind in ("moons", "circles")
    if kernel == "linear" and nonlinear:
        tips.append(("warning", "Kernel linear hanya bisa membuat garis lurus, sehingga pola melengkung tidak terpisah baik. Coba kernel RBF."))
    elif kernel == "linear":
        tips.append(("success", "Data ini terpisah secara linear, jadi kernel linear sudah cukup."))
    elif nonlinear:
        tips.append(("success", "Kernel non-linear memetakan data ke ruang berdimensi lebih tinggi sehingga batas bisa melengkung."))
    if C <= 0.1:
        tips.append(("info", "C kecil: margin lebar dan toleran terhadap kesalahan. Risiko underfitting."))
    elif C >= 100:
        tips.append(("warning", "C besar: model dipaksa benar pada hampir semua titik, margin menyempit. Risiko overfitting."))
    if kernel == "rbf":
        if gamma >= 10:
            tips.append(("warning", "gamma besar: tiap titik hanya berpengaruh di sekitarnya, batas sangat berliku. Risiko overfitting."))
        elif gamma <= 0.1:
            tips.append(("info", "gamma kecil: pengaruh tiap titik luas, batas lebih halus dan sederhana."))
    if kernel == "poly" and degree >= 4:
        tips.append(("warning", "Derajat polinomial tinggi membuat batas keputusan sangat kompleks."))
    return tips
