"""Halaman Beranda."""
import streamlit as st

from core.config import TARGET_ACCURACY
from core.dataset import count_images
from ui import components as C, nav, state


def render():
    C.show(C.hero(
        "Klasifikasi Penyakit Daun Sirih",
        "Support Vector Machine (SVM) berdasarkan ekstraksi ciri tekstur Gray Level Co-occurrence Matrix (GLCM).",
        ["Pengolahan Citra Digital", "SVM", "GLCM", "Klasifikasi Multi-kelas"]))

    scan = st.session_state.get("scan")
    classes = scan["classes"] if scan else {}
    X = st.session_state.get("X")
    res = st.session_state.get("result")

    cards = [
        ("Kelas", len(classes) if classes else "–", "dari dataset", "success", "🏷️"),
        ("Citra dataset", count_images(scan) if scan else "–", "total file gambar", "info", "🖼️"),
        ("Fitur / citra", X.shape[1] if X is not None else "–", "hasil ekstraksi", "warning", "🧮"),
        ("Akurasi uji", "%.1f%%" % (res["metrics"]["accuracy"] * 100) if res else "–",
         "target > %d%%" % int(TARGET_ACCURACY * 100), "dark", "🎯"),
    ]
    for col, c in zip(st.columns(4), cards):
        with col:
            C.show(C.stat_card(*c))

    C.show(C.section_title("", "Alur metode", "Dari citra daun sampai prediksi penyakit"))
    C.show(C.steps_grid([
        ("🖼️", "Citra RGB", "Daun sirih dari dataset Kaggle"),
        ("📐", "Pra-pengolahan", "Resize, grayscale, kuantisasi level keabuan"),
        ("🧩", "Ekstraksi GLCM", "Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM"),
        ("⚖️", "Normalisasi", "StandardScaler agar skala fitur seimbang"),
        ("🤖", "SVM + Tuning", "GridSearchCV: kernel, C, gamma"),
        ("📊", "Evaluasi", "Akurasi, presisi, recall, confusion matrix"),
    ]))

    left, right = st.columns([3, 2])
    with left:
        C.show(C.card("Cara memakai aplikasi", (
            "<p><b>1.</b> Buka <b>Dataset &amp; Fitur</b>, muat dataset (folder atau ZIP), lalu ekstraksi fitur.</p>"
            "<p><b>2.</b> Buka <b>Training &amp; Tuning</b>, pilih grid parameter, lalu latih model.</p>"
            "<p><b>3.</b> Lihat hasil di <b>Evaluasi</b> dan simpan tabel tuning untuk laporan.</p>"
            "<p><b>4.</b> Coba model di <b>Prediksi</b> dengan mengunggah foto daun.</p>")))
        st.button("Mulai: muat dataset ➜", type="primary", on_click=state.goto, args=(nav.DATA,))
    with right:
        C.show(C.card("Tentang metode", (
            "<p><b>GLCM</b> menghitung seberapa sering pasangan intensitas piksel muncul berdampingan, "
            "sehingga menggambarkan pola tekstur permukaan daun.</p>"
            "<p><b>SVM</b> mencari bidang pemisah dengan margin terbesar antar kelas; "
            "parameter <b>C</b>, <b>gamma</b>, dan <b>kernel</b> di-tuning dengan cross-validation.</p>")))
