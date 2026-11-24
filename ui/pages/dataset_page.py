"""Halaman Dataset & Ekstraksi Fitur."""
import os
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from core import dataset, features
from core.config import DEFAULT_CFG
from core.imageio import bgr_to_rgb, read_image_path, thumbnail
from ui import components as C, nav, state


def _load_scan(base, root):
    with st.spinner("Memindai dataset..."):
        scan = dataset.scan_dataset(root, base)
    state.set_dataset(base, root, scan)


def _section_load():
    C.show(C.section_title(1, "Muat dataset", "Satu sub-folder per kelas. Folder tambahan seperti 'images' atau 'train' diabaikan."))
    src = st.radio("Sumber dataset", ["Folder di komputer", "Upload file ZIP"],
                   horizontal=True, label_visibility="collapsed")

    if src == "Folder di komputer":
        path = st.text_input("Path folder dataset", placeholder=r"D:\DaunSirih DataSet")
        if st.button("📂 Muat dataset", type="primary"):
            path = path.strip().strip('"')
            if path and os.path.isdir(path):
                _load_scan(path, dataset.find_dataset_root(path))
            else:
                C.show(C.alert("danger", "Folder tidak ditemukan. Salin path dari File Explorer, contoh: D:\\DaunSirih DataSet"))
    else:
        up = st.file_uploader("File ZIP dataset", type=["zip"])
        if up is not None and st.button("📦 Ekstrak & muat", type="primary"):
            try:
                with st.spinner("Mengekstrak ZIP (dataset besar bisa beberapa menit)..."):
                    base, root = dataset.extract_zip(up)
                _load_scan(base, root)
            except zipfile.BadZipFile:
                C.show(C.alert("danger", "File bukan ZIP yang valid atau rusak."))


def _show_samples(classes):
    tabs = st.tabs(list(classes.keys()))
    for tab, (label, files) in zip(tabs, classes.items()):
        with tab:
            idx = np.linspace(0, len(files) - 1, num=min(4, len(files)), dtype=int)
            for col, i in zip(st.columns(len(idx)), idx):
                img = read_image_path(files[int(i)])
                if img is not None:
                    col.image(bgr_to_rgb(thumbnail(img)), caption=os.path.basename(files[int(i)]))


def _section_summary():
    scan = st.session_state.get("scan")
    if not scan:
        C.show(C.alert("info", "Belum ada dataset yang dimuat. Pilih folder atau ZIP di atas."))
        return False

    classes = scan["classes"]
    if len(classes) < 2:
        C.show(C.alert("danger",
                       "<b>Hanya %d kelas terdeteksi.</b> Klasifikasi butuh minimal 2 kelas. "
                       "Susunan folder yang terbaca ada di bawah; pastikan setiap kelas punya folder sendiri." % len(classes),
                       raw=True))
        st.code(dataset.folder_summary(st.session_state["root"]))
        return False

    n_img = dataset.count_images(scan)
    note = " Label diambil dari tabel <b>%s</b>." % C.esc(scan["table"]) if scan["source"] == "tabel" else ""
    C.show(C.alert("success", "Dataset terbaca: <b>%d kelas</b>, <b>%d citra</b>.%s" % (len(classes), n_img, note), raw=True))
    if scan["unlabeled"]:
        C.show(C.alert("warning", "%d gambar tidak punya kelas yang jelas dan diabaikan." % len(scan["unlabeled"])))

    counts = {k: len(v) for k, v in classes.items()}
    C.show(C.class_grid(counts))
    left, right = st.columns([2, 3])
    with left:
        st.bar_chart(pd.Series(counts, name="Jumlah citra"))
    with right:
        st.caption("Contoh citra per kelas")
        _show_samples(classes)
    if max(counts.values()) > 2 * min(counts.values()):
        C.show(C.alert("warning", "Jumlah citra antar kelas tidak seimbang. Biarkan opsi class_weight='balanced' aktif saat training."))
    return True


def _section_features():
    C.show(C.section_title(2, "Parameter ekstraksi fitur", "Atur lalu jalankan. Hasilnya dipakai pada tahap training."))
    d = DEFAULT_CFG
    with st.form("form_fitur"):
        c1, c2, c3 = st.columns(3)
        size = c1.select_slider("Ukuran resize (px)", [64, 128, 192, 256], value=d["size"])
        levels = c2.select_slider("Level keabuan GLCM", [8, 16, 32, 64], value=d["levels"])
        max_pc = c3.number_input("Maks. citra per kelas", 10, 5000, 300, step=50,
                                 help="Kecilkan agar ekstraksi lebih cepat.")
        c4, c5 = st.columns(2)
        distances = c4.multiselect("Jarak piksel", [1, 2, 3, 4, 5], default=d["distances"])
        angles = c5.multiselect("Sudut (derajat)", [0, 45, 90, 135], default=d["angles"])
        c6, c7 = st.columns(2)
        avg_angle = c6.checkbox("Rata-ratakan semua sudut (fitur lebih ringkas)", value=d["avg_angle"])
        use_color = c7.checkbox("Tambahkan fitur warna HSV (di luar tekstur)", value=d["use_color"],
                                help="Judul proyek berbasis tekstur. Aktifkan hanya untuk percobaan pembanding.")
        go = st.form_submit_button("🚀 Ekstraksi fitur", type="primary")

    if not go:
        return
    if not distances or not angles:
        C.show(C.alert("danger", "Pilih minimal satu jarak dan satu sudut."))
        return

    cfg = {"size": size, "levels": levels, "distances": sorted(distances),
           "angles": sorted(angles), "avg_angle": avg_angle, "use_color": use_color}
    bar = st.progress(0.0, text="Mengekstraksi fitur...")

    def cb(i, n):
        bar.progress(i / float(n), text="Mengekstraksi fitur... %d/%d" % (i, n))

    X, y, bad = features.build_dataset(st.session_state["scan"]["classes"], cfg, int(max_pc), progress_cb=cb)
    bar.empty()
    if len(X) == 0:
        C.show(C.alert("danger", "Tidak ada gambar yang berhasil dibaca."))
        return
    state.set_features(cfg, X, y)
    msg = "Ekstraksi selesai: <b>%d citra</b>, <b>%d fitur</b> per citra." % (X.shape[0], X.shape[1])
    if bad:
        msg += " (%d file gagal dibaca)" % bad
    C.show(C.alert("success", msg, raw=True))


def _section_results():
    X = st.session_state.get("X")
    if X is None:
        return
    cfg, y = st.session_state["cfg"], st.session_state["y"]
    C.show(C.section_title(3, "Hasil ekstraksi", "Tabel fitur siap dipakai SVM."))
    df = pd.DataFrame(X, columns=features.feature_names(cfg))
    df.insert(0, "label", y)
    st.dataframe(df.head(50), hide_index=True)
    st.caption("Rata-rata fitur per kelas (bahan analisis untuk laporan)")
    st.dataframe(df.groupby("label").mean().round(4))
    st.download_button("⬇️ Unduh fitur (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "fitur_glcm_sirih.csv", "text/csv")
    st.button("Lanjut ke training ➜", type="primary", on_click=state.goto, args=(nav.TRAIN,))


def render():
    C.show(C.page_header("Dataset & Ekstraksi Fitur", "Muat dataset, pahami isinya, lalu ekstraksi fitur tekstur GLCM."))
    _section_load()
    ok = _section_summary()
    if ok:
        _section_features()
        _section_results()
