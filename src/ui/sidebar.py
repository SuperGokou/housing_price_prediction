"""Sidebar: language, city, data source (CSV upload or sample), and settings."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.data.io import read_csv
from src.ui.i18n import APP_ICON, LANG_OPTIONS, t

# Strip anything that isn't a letter (any language), digit, space, or basic
# punctuation. Blocks HTML/Markdown-link injection since the city string is
# rendered into st.markdown() and Plotly titles downstream.
_CITY_DISALLOWED = re.compile(r"[^\w\s,.\-]", re.UNICODE)


def _sanitize_city(raw: str) -> str:
    """Make a user-typed city safe to interpolate into markdown/titles."""
    return _CITY_DISALLOWED.sub("", raw).strip()[:60]


@dataclass
class SidebarState:
    city: str
    df: pd.DataFrame | None
    confidence: int
    source: str
    error: str | None
    lang: str


def render_sidebar(sample_path: str) -> SidebarState:
    """Render sidebar controls and return the chosen configuration."""
    lang_choice = st.sidebar.radio("语言 / Language", options=list(LANG_OPTIONS.keys()),
                                   horizontal=True)
    lang = LANG_OPTIONS[lang_choice]

    st.sidebar.markdown(f"## {APP_ICON} {t(lang, 'sidebar_title')}")
    st.sidebar.caption(t(lang, "sidebar_subtitle"))
    st.sidebar.divider()

    city_raw = st.sidebar.text_input(t(lang, "city_label"), value="Seattle, WA",
                                     help=t(lang, "city_help"))
    city = _sanitize_city(city_raw)

    uploaded = st.sidebar.file_uploader(t(lang, "upload_label"), type=["csv"],
                                        help=t(lang, "upload_help"))
    st.sidebar.caption(t(lang, "csv_hint"))

    confidence = st.sidebar.selectbox(t(lang, "confidence_label"), options=[80, 90, 95],
                                      index=0, format_func=lambda c: f"{c}%")

    df: pd.DataFrame | None = None
    error: str | None = None
    if uploaded is not None:
        source = t(lang, "src_uploaded")
        try:
            df = read_csv(uploaded)
        except ValueError as exc:  # our own controlled, user-facing messages
            error = t(lang, "err_parse", detail=exc)
        except Exception:  # don't leak pandas/path internals to the UI
            logging.exception("Unexpected error parsing uploaded CSV")
            error = t(lang, "err_parse_generic")
    else:
        df = read_csv(sample_path)
        source = t(lang, "src_sample")

    st.sidebar.divider()
    st.sidebar.caption(t(lang, "sidebar_disclaimer"))
    return SidebarState(city=city, df=df, confidence=confidence, source=source,
                        error=error, lang=lang)
