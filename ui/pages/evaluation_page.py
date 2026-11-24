"""Halaman Evaluasi."""
import streamlit as st

from core import model, plots
from core.config import TARGET_ACCURACY
from ui import components as C, nav, state


def render():
    C.show(C.page_header("Evaluasi Model", "Kinerja SVM pada data uji yang tidak dipakai saat training."))
    r = st.session_state.get("result")
    if not r:
        C.show(C.alert("warning", "Model belum dilatih. Lakukan training di halaman Training & Tuning."))
        st.button("⬅️ Ke Training & Tuning", on_click=state.goto, args=(nav.TRAIN,))
        return

    m = r["metrics"]
    tiles = [("Akurasi", m["accuracy"], "success", "🎯"), ("Presisi (macro)", m["precision"], "info", "🔬"),
             ("Recall (macro)", m["recall"], "warning", "🔁"), ("F1 (macro)", m["f1"], "dark", "⚖️")]
    for col, (label, val, var, ico) in zip(st.columns(4), tiles):
        with col:
            C.show(C.stat_card(label, "%.2f%%" % (val * 100), "%d citra uji" % r["n_test"], var, ico))

    if m["accuracy"] >= TARGET_ACCURACY:
        C.show(C.alert("success", "Target akurasi di atas %d%% <b>tercapai</b>." % int(TARGET_ACCURACY * 100), raw=True))
    else:
        C.show(C.alert("warning",
                       "Akurasi di bawah target %d%%. Coba perbanyak jarak GLCM, perbesar grid C dan gamma, "
                       "tambah citra per kelas, atau bandingkan dengan fitur warna HSV." % int(TARGET_ACCURACY * 100)))

    left, right = st.columns(2)
    with left:
        C.show(C.section_title("", "Confusion matrix", "Baris = kelas aktual, kolom = prediksi"))
        st.pyplot(plots.confusion_figure(m["cm"], m["labels"]))
    with right:
        C.show(C.section_title("", "Laporan per kelas", "Presisi, recall, F1, dan jumlah data uji"))
        st.dataframe(m["report"].round(3))
        conf = model.top_confusions(m["cm"], m["labels"])
        if conf:
            items = "".join("<p>• <b>%s</b> sering diprediksi sebagai <b>%s</b> (%d citra)</p>"
                            % (C.esc(a), C.esc(b), n) for a, b, n in conf)
            C.show(C.card("Kelas yang paling sering tertukar", items))
        else:
            C.show(C.alert("success", "Tidak ada kesalahan klasifikasi pada data uji."))

    params = ", ".join("%s=%s" % kv for kv in r["best_params"].items())
    st.caption("Latih: %d citra · Uji: %d citra · Parameter: %s" % (r["n_train"], r["n_test"], params))
