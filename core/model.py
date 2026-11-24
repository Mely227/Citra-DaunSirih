"""Training, tuning, evaluasi, dan prediksi SVM (tanpa dependensi Streamlit)."""
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import (GridSearchCV, ParameterGrid, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from core.features import extract_features


def build_param_grid(kernels, Cs, gammas, degrees=(2, 3)):
    grid = []
    if "linear" in kernels:
        grid.append({"svc__kernel": ["linear"], "svc__C": list(Cs)})
    if "rbf" in kernels:
        grid.append({"svc__kernel": ["rbf"], "svc__C": list(Cs), "svc__gamma": list(gammas)})
    if "poly" in kernels:
        grid.append({"svc__kernel": ["poly"], "svc__C": list(Cs),
                     "svc__gamma": list(gammas), "svc__degree": list(degrees)})
    return grid


def count_fits(grid, cv_folds):
    return len(ParameterGrid(grid)) * cv_folds


def evaluate(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                       zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = pd.DataFrame(classification_report(y_true, y_pred, labels=labels,
                                                output_dict=True, zero_division=0)).T
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "labels": labels, "cm": cm,
            "report": report}


def top_confusions(cm, labels, k=3):
    """Pasangan (aktual, prediksi, jumlah) yang paling sering tertukar."""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((labels[i], labels[j], int(cm[i, j])))
    return sorted(pairs, key=lambda t: -t[2])[:k]


def train_and_tune(X, y, grid, test_size=0.2, cv_folds=5, seed=42, balanced=True, n_jobs=-1):
    """GridSearchCV pada data latih, lalu evaluasi pada data uji."""
    y = np.asarray(y)
    counts = Counter(y)
    if len(counts) < 2:
        raise ValueError("Klasifikasi butuh minimal 2 kelas, sedangkan data hanya punya %d kelas." % len(counts))
    if min(counts.values()) < 3:
        raise ValueError("Ada kelas dengan kurang dari 3 citra. Tambah citra per kelas.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    cv = min(cv_folds, min(Counter(y_tr).values()))
    if cv < 2:
        raise ValueError("Data latih terlalu sedikit untuk cross-validation.")

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svc", SVC(class_weight="balanced" if balanced else None))])
    gs = GridSearchCV(pipe, grid,
                      cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                      scoring="accuracy", n_jobs=n_jobs, return_train_score=True)
    gs.fit(X_tr, y_tr)

    # model akhir ber-probabilitas (untuk halaman prediksi); jatuh balik bila tidak didukung
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            final = clone(gs.best_estimator_).set_params(svc__probability=True).fit(X_tr, y_tr)
        except Exception:
            final = clone(gs.best_estimator_).fit(X_tr, y_tr)

    y_pred = final.predict(X_te)
    return {
        "model": final,
        "best_params": {k.replace("svc__", ""): v for k, v in gs.best_params_.items()},
        "best_cv": float(gs.best_score_),
        "cv_folds": cv,
        "cv_df": pd.DataFrame(gs.cv_results_),
        "y_te": y_te, "y_pred": y_pred,
        "n_train": len(y_tr), "n_test": len(y_te),
        "metrics": evaluate(y_te, y_pred),
    }


def tuning_table(cv_df):
    """Tabel ringkas seluruh kombinasi parameter (untuk analisis di laporan)."""
    cols = ["param_svc__kernel", "param_svc__C", "param_svc__gamma", "param_svc__degree",
            "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    df = cv_df.reindex(columns=cols).sort_values("rank_test_score")
    df = df.rename(columns={
        "param_svc__kernel": "kernel", "param_svc__C": "C", "param_svc__gamma": "gamma",
        "param_svc__degree": "degree", "mean_train_score": "akurasi_latih",
        "mean_test_score": "akurasi_CV", "std_test_score": "std_CV",
        "rank_test_score": "peringkat"})
    for c in ("akurasi_latih", "akurasi_CV", "std_CV"):
        df[c] = df[c].astype(float).round(4)
    for c in ("kernel", "C", "gamma", "degree"):
        df[c] = df[c].map(lambda v: "-" if pd.isna(v) else str(v))
    return df.reset_index(drop=True)


# ----------------------------------------------------------------- simpan & prediksi
def save_model(path, model, cfg, classes):
    joblib.dump({"model": model, "cfg": cfg, "classes": list(classes)}, path)


def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


def predict_image(pack, img_bgr):
    """Return (label, {kelas: probabilitas}, vektor_fitur)."""
    model, cfg = pack["model"], pack["cfg"]
    feats = extract_features(img_bgr, cfg).reshape(1, -1)
    label = model.predict(feats)[0]
    try:
        proba = model.predict_proba(feats)[0]
    except Exception:
        d = np.atleast_2d(model.decision_function(feats))[0]
        e = np.exp(d - d.max())
        proba = e / e.sum()
    return str(label), {str(c): float(p) for c, p in zip(model.classes_, proba)}, feats
