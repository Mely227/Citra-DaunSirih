"""Halaman Training & Tuning."""
import streamlit as st

from core import model, plots
from core.config import MODEL_PATH
from ui import components as C, nav, state


def _results(r):
    C.show(C.section_title(2, "Hasil tuning", "Parameter terbaik berdasarkan akurasi cross-validation."))
    left, right = st.columns([2, 3])
    with left:
        C.show(C.kv_card("Parameter terbaik", r["best_params"]))
        C.show(C.stat_card("Akurasi CV (data latih)", "%.2f%%" % (r["best_cv"] * 100),
                           "%d-fold" % r["cv_folds"], "success", "✅"))
    with right:
        fig = plots.heatmap_figure(r["cv_df"])
        if fig is not None:
            st.caption("Heatmap akurasi CV kernel RBF (C vs gamma)")
            st.pyplot(fig)
        else:
            C.show(C.alert("info", "Heatmap hanya tersedia bila kernel RBF ikut di-tuning."))

    st.caption("Seluruh kombinasi parameter (urut peringkat). Gunakan untuk analisis di laporan.")
    table = model.tuning_table(r["cv_df"])
    st.dataframe(table, hide_index=True)
    st.download_button("⬇️ Unduh hasil tuning (CSV)", table.to_csv(index=False).encode("utf-8"),
                       "hasil_tuning_svm.csv", "text/csv")
    st.button("Lihat evaluasi ➜", type="primary", on_click=state.goto, args=(nav.EVAL,))


def render():
    C.show(C.page_header("Training & Tuning Parameter SVM", "Cari kombinasi kernel, C, dan gamma terbaik dengan GridSearchCV."))
    X = st.session_state.get("X")
    if X is None:
        C.show(C.alert("warning", "Fitur belum diekstraksi. Selesaikan dulu langkah di halaman Dataset & Fitur."))
        st.button("⬅️ Ke Dataset & Fitur", on_click=state.goto, args=(nav.DATA,))
        return

    C.show(C.section_title(1, "Pengaturan", "%d citra · %d fitur · %d kelas" % (
        X.shape[0], X.shape[1], len(set(st.session_state["y"])))))
    with st.form("form_train"):
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Porsi data uji", 0.1, 0.4, 0.2, 0.05)
        cv_folds = c2.slider("Fold cross-validation", 3, 10, 5)
        seed = c3.number_input("Random seed", 0, 9999, 42)
        kernels = st.multiselect("Kernel", ["rbf", "linear", "poly"], default=["rbf", "linear"])
        g1, g2 = st.columns(2)
        Cs = g1.multiselect("Nilai C", [0.1, 1, 10, 100, 1000], default=[0.1, 1, 10, 100])
        gammas = g2.multiselect("Nilai gamma (rbf/poly)", ["scale", 0.001, 0.01, 0.1, 1],
                                default=["scale", 0.001, 0.01, 0.1])
        balanced = st.checkbox("class_weight='balanced' (disarankan bila kelas tidak seimbang)", value=True)
        go = st.form_submit_button("🔧 Latih & tuning", type="primary")

    if go:
        if not kernels or not Cs or (set(kernels) - {"linear"} and not gammas):
            C.show(C.alert("danger", "Lengkapi pilihan kernel, C, dan gamma (gamma tidak perlu untuk kernel linear)."))
        else:
            grid = model.build_param_grid(kernels, Cs, gammas)
            n_fits = model.count_fits(grid, int(cv_folds))
            try:
                with st.spinner("Menjalankan GridSearchCV (%d fit)..." % n_fits):
                    result = model.train_and_tune(X, st.session_state["y"], grid, test_size=float(test_size),
                                                  cv_folds=int(cv_folds), seed=int(seed), balanced=balanced)
            except ValueError as e:
                C.show(C.alert("danger", str(e)))
            else:
                state.set_result(result)
                try:
                    model.save_model(MODEL_PATH, result["model"], st.session_state["cfg"],
                                     result["metrics"]["labels"])
                except OSError:
                    pass
                C.show(C.alert("success", "Training selesai dan model disimpan."))

    r = st.session_state.get("result")
    if r:
        _results(r)
