"""Komponen HTML bergaya Bootstrap. Murni tampilan, tanpa logika SVM."""
import html as _html
import re

import streamlit as st

_ICONS = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "danger": "⛔"}


def esc(x):
    return _html.escape(str(x))


def _compact(s):
    s = re.sub(r"\s*\n\s*", " ", s)
    return re.sub(r">\s+<", "><", s).strip()


def show(html):
    """Render HTML. Semua komponen diawali <div> agar tidak diproses sebagai markdown."""
    st.markdown(_compact(html), unsafe_allow_html=True)


def badge(text, variant="success"):
    return '<span class="badge-b badge-%s">%s</span>' % (variant, esc(text))


def hero(title, subtitle, badges=()):
    chips = "".join(badge(b, "light") for b in badges)
    deco = ('<span class="hd hd1">🍃</span><span class="hd hd2">🌿</span>'
            '<span class="hd hd3">🍃</span><span class="hd hd4">✨</span>')
    return ('<div class="hero"><div class="hero-deco">%s</div><div class="hero-title">%s</div>'
            '<div class="hero-sub">%s</div><div class="hero-badges">%s</div></div>'
            % (deco, esc(title), esc(subtitle), chips))


def page_header(title, subtitle=""):
    return ('<div class="page-head"><div class="page-title">%s</div><div class="page-sub">%s</div></div>'
            % (esc(title), esc(subtitle)))


def section_title(num, title, subtitle=""):
    circle = '<div class="sec-num">%s</div>' % esc(num) if num != "" else ""
    return ('<div class="sec">%s<div><div class="sec-title">%s</div><div class="sec-sub">%s</div></div></div>'
            % (circle, esc(title), esc(subtitle)))


def stat_card(label, value, hint="", variant="success", icon=""):
    if isinstance(value, int) and not isinstance(value, bool):
        val = '<div class="stat-value count" style="--n:%d" aria-label="%d"></div>' % (value, value)
    else:
        val = '<div class="stat-value">%s</div>' % esc(value)
    return ('<div class="stat-card stat-%s"><div class="stat-icon">%s</div><div>'
            '<div class="stat-label">%s</div>%s<div class="stat-hint">%s</div></div></div>'
            % (variant, icon, esc(label), val, esc(hint)))


def alert(kind, text, raw=False):
    body = text if raw else esc(text)
    return '<div class="b-alert %s"><div>%s</div><div>%s</div></div>' % (kind, _ICONS.get(kind, ""), body)


def card(title, body_html):
    return '<div class="b-card"><div class="b-card-title">%s</div>%s</div>' % (esc(title), body_html)


def kv_card(title, mapping):
    rows = "".join('<div class="kv"><span>%s</span><b>%s</b></div>' % (esc(k), esc(v))
                   for k, v in mapping.items())
    return card(title, rows)


def class_grid(counts):
    chips = "".join('<div class="class-chip"><div class="class-name">%s</div>'
                    '<div class="class-count">%s</div><div class="class-sub">citra</div></div>'
                    % (esc(k), esc(v)) for k, v in counts.items())
    return '<div class="class-grid">%s</div>' % chips


def steps_grid(steps):
    """steps: daftar (ikon, judul, deskripsi)."""
    items = "".join('<div class="step"><div class="step-num">%d</div><div class="step-ico">%s</div>'
                    '<div class="step-title">%s</div><div class="step-desc">%s</div></div>'
                    % (i, ico, esc(t), esc(d)) for i, (ico, t, d) in enumerate(steps, 1))
    return '<div class="step-grid">%s</div>' % items


def prob_bar(label, pct, top=False):
    return ('<div class="prob-row"><div class="prob-head"><span>%s</span><b>%.1f%%</b></div>'
            '<div class="progress-b%s"><div style="width:%.1f%%"></div></div></div>'
            % (esc(label), pct, " top" if top else "", max(0.0, min(100.0, pct))))


def sidebar_brand():
    return ('<div class="brand"><div class="brand-logo">🌿</div><div><div class="brand-name">SVM Daun Sirih</div>'
            '<div class="brand-sub">Pengolahan Citra Digital</div></div></div>')


def sidebar_status(items):
    rows = "".join('<div class="status-item"><span class="dot%s"></span>%s</div>'
                   % (" on" if done else "", esc(label)) for label, done in items)
    return '<div><div class="side-label">Progres</div>%s</div>' % rows
