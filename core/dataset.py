"""Memuat & memindai dataset citra (tanpa dependensi Streamlit)."""
import os
import shutil
import tempfile
import zipfile

from core.config import IGNORED_FOLDER_NAMES, IMG_EXT


# ----------------------------------------------------------------- struktur folder
def _is_hidden(name):
    return name.startswith("__") or name.startswith(".")


def has_images(folder):
    try:
        return any(f.lower().endswith(IMG_EXT) for f in os.listdir(folder))
    except OSError:
        return False


def find_dataset_root(path):
    """Turun ke dalam folder tunggal (hasil ZIP bersarang) sampai ketemu isi dataset."""
    for _ in range(6):
        try:
            subs = [d for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d)) and not _is_hidden(d)]
        except OSError:
            break
        if len(subs) == 1 and not has_images(path):
            path = os.path.join(path, subs[0])
        else:
            break
    return path


def class_label_for(dirpath, root):
    """Nama kelas = nama folder terdekat yang bukan folder generik/pembagi data.
    Mengembalikan None bila tidak ada (mis. gambar langsung di folder utama)."""
    cur = os.path.normpath(dirpath)
    root_n = os.path.normpath(root)
    while cur != root_n:
        name = os.path.basename(cur)
        if name.lower() not in IGNORED_FOLDER_NAMES:
            return name
        parent = os.path.dirname(cur)
        if parent == cur:  # sudah sampai akar disk
            break
        cur = parent
    return None


def folder_summary(root, max_depth=4, max_rows=60):
    """Ringkasan susunan folder untuk membantu debug struktur dataset."""
    rows = []
    base_depth = os.path.normpath(root).count(os.sep)
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if not _is_hidden(d))
        depth = os.path.normpath(dp).count(os.sep) - base_depth
        if depth > max_depth:
            dn[:] = []
            continue
        n_img = sum(f.lower().endswith(IMG_EXT) for f in fn)
        name = os.path.basename(dp) or dp
        rows.append("%s%s/  (%d gambar, %d file lain)" % ("    " * depth, name, n_img, len(fn) - n_img))
        if len(rows) >= max_rows:
            rows.append("...")
            break
    return "\n".join(rows)


# ------------------------------------------------------- cadangan: label dari tabel
def _norm_name(value):
    return os.path.basename(str(value).replace("\\", "/")).strip().lower()


def labels_from_table(base, image_paths):
    """Cadangan bila gambar tidak terpisah per folder kelas: cari file .csv/.xlsx
    yang memuat nama file gambar + kolom label. Return (dict_kelas, nama_tabel) atau None."""
    try:
        import pandas as pd
    except ImportError:
        return None

    tables = []
    for dp, _dn, fn in os.walk(base):
        for f in sorted(fn):
            if f.lower().endswith((".csv", ".xlsx")):
                tables.append(os.path.join(dp, f))
        if len(tables) >= 10:
            break
    if not tables or not image_paths:
        return None

    # hanya nama file yang unik yang bisa dipetakan dengan aman
    counts, owner = {}, {}
    for p in image_paths:
        low = os.path.basename(p).lower()
        for key in (low, os.path.splitext(low)[0]):
            counts[key] = counts.get(key, 0) + 1
            owner[key] = p
    unique = {k: owner[k] for k, c in counts.items() if c == 1}

    keywords = ("class", "label", "kelas", "kategori", "category", "disease", "penyakit", "target")

    for table in tables:
        frames = []
        try:
            if table.lower().endswith(".csv"):
                frames = [pd.read_csv(table)]
            else:
                frames = list(pd.read_excel(table, sheet_name=None).values())
        except Exception:
            continue

        for df in frames:
            text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            best_col, best_frac = None, 0.0
            for c in text_cols:
                vals = df[c].dropna().astype(str).map(_norm_name)
                if len(vals) == 0:
                    continue
                frac = float(vals.isin(unique).mean())
                if frac > best_frac:
                    best_col, best_frac = c, frac
            if best_col is None or best_frac < 0.5:
                continue

            cand = []
            for c in text_cols:
                if c == best_col:
                    continue
                values = df[c].dropna().astype(str).str.strip()
                uniq = set(values.str.lower().unique())
                if 2 <= len(uniq) <= 30 and not uniq.issubset(IGNORED_FOLDER_NAMES):
                    cand.append((c, len(uniq)))
            if not cand:
                continue
            named = [x for x in cand if any(k in str(x[0]).lower() for k in keywords)]
            label_col = (named or sorted(cand, key=lambda x: -x[1]))[0][0]

            mapped, seen = {}, set()
            for fname, lab in zip(df[best_col], df[label_col]):
                path = unique.get(_norm_name(fname))
                if path is None or path in seen or str(lab).strip() == "" or str(lab) == "nan":
                    continue
                seen.add(path)
                mapped.setdefault(str(lab).strip(), []).append(path)
            if len(mapped) >= 2 and len(seen) >= 0.5 * len(image_paths):
                return mapped, os.path.basename(table)
    return None


# ------------------------------------------------------------------------ pemindaian
def scan_dataset(root, base=None):
    """Pindai dataset. Return dict:
       classes  : {nama_kelas: [path, ...]}
       unlabeled: gambar yang kelasnya tidak bisa ditentukan
       source   : "folder" atau "tabel"
       table    : nama file tabel (bila source == "tabel")"""
    classes, unlabeled = {}, []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if not _is_hidden(d))
        imgs = [os.path.join(dp, f) for f in sorted(fn) if f.lower().endswith(IMG_EXT)]
        if not imgs:
            continue
        label = class_label_for(dp, root)
        if label is None:
            unlabeled.extend(imgs)
        else:
            classes.setdefault(label, []).extend(imgs)

    result = {"classes": dict(sorted(classes.items())), "unlabeled": unlabeled,
              "source": "folder", "table": None}

    if len(classes) < 2:
        all_imgs = unlabeled + [p for v in classes.values() for p in v]
        found = labels_from_table(base or root, all_imgs)
        if found:
            mapped, table = found
            result.update(classes=dict(sorted(mapped.items())), unlabeled=[],
                          source="tabel", table=table)
    return result


def count_images(scan):
    return sum(len(v) for v in scan["classes"].values())


# ---------------------------------------------------------------------------- ZIP
def extract_zip(fileobj):
    """Ekstrak ZIP ke folder sementara. Return (base_dir, dataset_root)."""
    base = tempfile.mkdtemp(prefix="sirih_")
    try:
        with zipfile.ZipFile(fileobj) as zf:
            zf.extractall(base)
    except Exception:
        shutil.rmtree(base, ignore_errors=True)
        raise
    return base, find_dataset_root(base)


def cleanup_temp(path):
    """Hapus folder sementara hasil ekstrak (hanya yang dibuat aplikasi ini)."""
    if not path:
        return
    tmp = os.path.normpath(tempfile.gettempdir())
    p = os.path.normpath(path)
    if os.path.basename(p).startswith("sirih_") and os.path.dirname(p) == tmp:
        shutil.rmtree(p, ignore_errors=True)
