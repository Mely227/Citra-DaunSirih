"""Halaman Prediksi citra."""
import cv2
import pandas as pd
import streamlit as st

from core import features, model
from core.imageio import bgr_to_rgb, read_image_bytes, thumbnail
from ui import components as C, nav, state


def render():
    C.show(C.page_header("Prediksi Citra Daun", "Unggah foto daun sirih, model akan menentukan kelasnya."))
    pack = state.get_model_pack()
    if not pack:
        C.show(C.alert("warning", "Belum ada model. Latih model dulu di halaman Training & Tuning."))
        st.button("⬅️ Ke Training & Tuning", on_click=state.goto, args=(nav.TRAIN,))
        return
    if pack.get("from_file"):
        C.show(C.alert("info", "Memakai model tersimpan dari file (latih ulang untuk memperbarui)."))

    up = st.file_uploader("Foto daun sirih", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is None:
        return
    img = read_image_bytes(up.read())
    if img is None:
        C.show(C.alert("danger", "Gambar tidak bisa dibaca."))
        return

    label, proba, feats = model.predict_image(pack, img)
    cfg = pack["cfg"]

    left, right = st.columns(2)
    with left:
        st.image(bgr_to_rgb(thumbnail(img, 420)), caption="Citra asli")
        gray = cv2.cvtColor(cv2.resize(img, (cfg["size"], cfg["size"])), cv2.COLOR_BGR2GRAY)
        st.image(gray, caption="Hasil pra-pengolahan (resize + grayscale)")
    with right:
        conf = max(proba.values()) * 100
        C.show('<div class="result-box"><div class="result-label">Hasil prediksi</div>'
               '<div class="result-name">%s</div><div>Keyakinan %s</div></div>'
               % (C.esc(label), C.badge("%.1f%%" % conf, "success" if conf >= 60 else "warning")))
        if conf < 50:
            C.show(C.alert("warning", "Keyakinan rendah. Pastikan foto jelas dan mirip data latih."))
        bars = "".join(C.prob_bar(k, v * 100, top=(k == label))
                       for k, v in sorted(proba.items(), key=lambda kv: -kv[1]))
        C.show(C.card("Probabilitas tiap kelas", bars))
        with st.expander("Nilai fitur yang diekstraksi"):
            st.dataframe(pd.DataFrame(feats, columns=features.feature_names(cfg)).T.rename(columns={0: "nilai"}))
