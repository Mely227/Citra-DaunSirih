"""Alat bantu belajar GLCM (tanpa Streamlit)."""
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

from core.config import GLCM_PROPS
from core.imageio import read_image_path


def quantize(gray, levels):
    return np.clip(gray // (256 // levels), 0, levels - 1).astype(np.uint8)


def preprocess_steps(img_bgr, size, levels):
    """Return (resize_bgr, gray, q, q_untuk_tampil)."""
    small = cv2.resize(img_bgr, (size, size))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    q = quantize(gray, levels)
    shown = (q.astype(np.float64) * (255.0 / max(levels - 1, 1))).astype(np.uint8)
    return small, gray, q, shown


def glcm_single(q, levels, distance, angle_deg):
    """GLCM ternormalisasi (simetris) untuk satu jarak & sudut. Bentuk (L, L, 1, 1)."""
    return graycomatrix(q, [distance], [np.deg2rad(angle_deg)], levels=levels,
                        symmetric=True, normed=True)


def props_of(glcm):
    return {p: float(graycoprops(glcm, p)[0, 0]) for p in GLCM_PROPS}


def toy_example():
    """Contoh kecil 4x4 untuk dipelajari langkah demi langkah (jarak 1, sudut 0)."""
    M = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 2, 2, 2],
                  [2, 2, 3, 3]], dtype=np.uint8)
    counts = graycomatrix(M, [1], [0], levels=4, symmetric=False, normed=False)[:, :, 0, 0]
    sym = graycomatrix(M, [1], [0], levels=4, symmetric=True, normed=False)[:, :, 0, 0]
    return M, counts, sym, sym / float(sym.sum())


def class_profile(classes, size=128, levels=32, n_per_class=20, progress_cb=None):
    """Rata-rata properti GLCM (jarak 1, rata-rata 4 sudut) per kelas."""
    angles = [np.deg2rad(a) for a in (0, 45, 90, 135)]
    rows, total, done = {}, len(classes) * n_per_class, 0
    for label, files in classes.items():
        step = max(1, len(files) // n_per_class)
        vals = []
        for path in files[::step][:n_per_class]:
            img = read_image_path(path)
            done += 1
            if img is None:
                continue
            gray = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2GRAY)
            g = graycomatrix(quantize(gray, levels), [1], angles, levels=levels,
                             symmetric=True, normed=True)
            vals.append([float(graycoprops(g, p).mean()) for p in GLCM_PROPS])
            if progress_cb:
                progress_cb(min(done, total), total)
        if vals:
            rows[label] = np.mean(vals, axis=0)
    return pd.DataFrame(rows, index=GLCM_PROPS).T
