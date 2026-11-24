"""Memuat CSS kustom + latar belakang animasi (daun jatuh, aurora, spora)."""
import os
import random

import streamlit as st

_CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")


def _background_html():
    """Lapisan latar tetap: blob aurora, daun sirih berguguran, dan spora melayang."""
    rnd = random.Random(7)  # tetap konsisten tiap rerun
    parts = ['<div class="bg-layer" aria-hidden="true">',
             '<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>']
    for i in range(16):
        size = rnd.randint(26, 64)
        parts.append(
            '<span class="leaf l%d" style="left:%d%%;width:%dpx;height:%dpx;'
            'animation-duration:%ds,%ds;animation-delay:-%ds,-%ds;opacity:%.2f"></span>'
            % (i % 3 + 1, rnd.randint(1, 97), size, size,
               rnd.randint(18, 34), rnd.randint(4, 8), rnd.randint(0, 30), rnd.randint(0, 8),
               rnd.uniform(.28, .55)))
    for i in range(14):
        size = rnd.randint(6, 16)
        parts.append(
            '<span class="spore" style="left:%d%%;width:%dpx;height:%dpx;'
            'animation-duration:%ds;animation-delay:-%ds"></span>'
            % (rnd.randint(1, 98), size, size, rnd.randint(14, 30), rnd.randint(0, 25)))
    parts.append("</div>")
    return "".join(parts)


def load_css():
    try:
        with open(_CSS_PATH, encoding="utf-8") as f:
            css = f.read()
    except OSError:
        return
    st.markdown("<style>%s</style>" % css, unsafe_allow_html=True)
    st.markdown(_background_html(), unsafe_allow_html=True)
