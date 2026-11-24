"""
Klasifikasi Penyakit Daun Sirih dengan SVM berbasis ekstraksi ciri tekstur (GLCM).
Jalankan dari dalam folder ini:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="SVM Daun Sirih", page_icon="🌿", layout="wide",
                   initial_sidebar_state="expanded")

from ui import components as C, nav, state, theme  # noqa: E402
from ui.pages import (dataset_page, evaluation_page, home, predict_page,  # noqa: E402
                      references_page, training_page)

ROUTES = {
    nav.HOME: home.render,
    nav.DATA: dataset_page.render,
    nav.TRAIN: training_page.render,
    nav.EVAL: evaluation_page.render,
    nav.PRED: predict_page.render,
    nav.REF: references_page.render,
}

state.init()
theme.load_css()

with st.sidebar:
    C.show(C.sidebar_brand())
    page = st.radio("Navigasi", nav.PAGE_ORDER, key="nav", label_visibility="collapsed")
    C.show(C.sidebar_status([
        ("Dataset dimuat", state.dataset_ready()),
        ("Fitur diekstraksi", st.session_state.get("X") is not None),
        ("Model dilatih", st.session_state.get("result") is not None),
    ]))
    st.caption("Proyek UAS Pengolahan Citra Digital")

ROUTES[page]()
