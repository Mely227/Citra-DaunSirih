"""Pengelolaan session_state (penghubung antara tampilan dan logika)."""
import streamlit as st

from core import dataset
from core.config import MODEL_PATH
from core.model import load_model

DEFAULTS = {"base": None, "root": None, "scan": None, "cfg": None,
            "X": None, "y": None, "result": None}


def init():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def goto(label):
    """Dipakai sebagai on_click callback untuk berpindah halaman."""
    st.session_state["nav"] = label


def set_dataset(base, root, scan):
    old = st.session_state.get("base")
    if old and old != base:
        dataset.cleanup_temp(old)
    st.session_state.update(base=base, root=root, scan=scan, cfg=None, X=None, y=None, result=None)


def set_features(cfg, X, y):
    st.session_state.update(cfg=cfg, X=X, y=y, result=None)


def set_result(result):
    st.session_state["result"] = result


def dataset_ready():
    scan = st.session_state.get("scan")
    return bool(scan) and len(scan["classes"]) >= 2


def get_model_pack():
    """Model dari sesi ini, atau dari file tersimpan bila ada."""
    res = st.session_state.get("result")
    if res is not None and st.session_state.get("cfg"):
        return {"model": res["model"], "cfg": st.session_state["cfg"],
                "classes": res["metrics"]["labels"], "from_file": False}
    pack = load_model(MODEL_PATH)
    if pack:
        pack["from_file"] = True
    return pack
