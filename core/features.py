"""Pra-pengolahan citra & ekstraksi fitur GLCM (+ warna HSV opsional)."""
import random

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from core.config import GLCM_PROPS
from core.imageio import read_image_path


def extract_features(img_bgr, cfg):
    """Resize -> grayscale -> kuantisasi -> GLCM (+ HSV opsional) -> vektor fitur."""
    img = cv2.resize(img_bgr, (cfg["size"], cfg["size"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    levels = cfg["levels"]
    q = np.clip(gray // (256 // levels), 0, levels - 1).astype(np.uint8)
    angles = [np.deg2rad(a) for a in cfg["angles"]]
    glcm = graycomatrix(q, distances=cfg["distances"], angles=angles,
                        levels=levels, symmetric=True, normed=True)

    feats = []
    for prop in GLCM_PROPS:
        m = graycoprops(glcm, prop)  # bentuk (jarak, sudut)
        feats.extend(m.mean(axis=1) if cfg["avg_angle"] else m.flatten())

    if cfg["use_color"]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for c in range(3):
            feats.append(hsv[:, :, c].mean())
            feats.append(hsv[:, :, c].std())
    return np.nan_to_num(np.array(feats, dtype=np.float64))


def feature_names(cfg):
    names = []
    for prop in GLCM_PROPS:
        for d in cfg["distances"]:
            if cfg["avg_angle"]:
                names.append("%s_d%s" % (prop, d))
            else:
                for a in cfg["angles"]:
                    names.append("%s_d%s_a%s" % (prop, d, a))
    if cfg["use_color"]:
        for c in "HSV":
            names += [c + "_mean", c + "_std"]
    return names


def build_dataset(classes, cfg, max_per_class, seed=42, progress_cb=None):
    """Ekstraksi fitur seluruh dataset. progress_cb(i, n) dipanggil berkala.
    Return X, y, jumlah_gagal."""
    rng = random.Random(seed)
    items = []
    for label, files in classes.items():
        files = list(files)
        rng.shuffle(files)
        items.extend((f, label) for f in files[:max_per_class])

    X, y, bad = [], [], 0
    n = len(items)
    for i, (path, label) in enumerate(items, 1):
        img = read_image_path(path)
        if img is None:
            bad += 1
        else:
            X.append(extract_features(img, cfg))
            y.append(label)
        if progress_cb and (i % 10 == 0 or i == n):
            progress_cb(i, n)
    return np.array(X), np.array(y), bad
