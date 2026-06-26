"""Housing Price Multi-Model Prediction — Streamlit entry point.

Run with:  streamlit run app.py
Loads the bundled Seattle sample by default, so the first render is a full demo.
UI language (中文 / English) is chosen in the sidebar.
"""

from __future__ import annotations

import streamlit as st

from src.data.validation import build_series
from src.eda.insights import compute_insights
from src.features.external_factors import ExternalFactor, external_factor_analysis
from src.models import exponential_smoothing, growth_curve, mean_reversion
from src.models.ensemble import combine
from src.ui import ensemble_section, eda_section, models_section
from src.ui.i18n import APP_ICON, direction_badge, freq_label, t
from src.ui.sidebar import render_sidebar

SAMPLE_PATH = "sample_data/seattle.csv"

_DIRECTION_COLOR = {"up": "#059669", "down": "#DC2626", "uncertain": "#D97706"}


def _render_factors(factors: list[ExternalFactor], lang: str) -> None:
    st.subheader(t(lang, "sec3_title"))
    cols = st.columns(len(factors))
    for col, f in zip(cols, factors):
        label = direction_badge(lang, f.direction)
        color = _DIRECTION_COLOR.get(f.direction, "#D97706")
        with col:
            st.markdown(f"##### {f.name}")
            st.markdown(
                f"<span style='background:{color};color:white;padding:2px 8px;"
                f"border-radius:6px;font-size:0.8rem'>{label}</span>",
                unsafe_allow_html=True)
            st.write(f.detail)


def main() -> None:
    # set_page_config must be the first Streamlit call; tab title stays bilingual.
    st.set_page_config(page_title="Housing Forecaster · 房价多模型预测",
                       page_icon=APP_ICON, layout="wide")

    state = render_sidebar(SAMPLE_PATH)
    lang = state.lang

    st.title(f"{APP_ICON} {t(lang, 'app_title')}")
    st.markdown(t(lang, "subtitle"))

    if state.error:
        st.error(state.error)
        st.stop()
    if state.df is None:
        st.info(t(lang, "err_no_data"))
        st.stop()

    try:
        series = build_series(state.df, city=state.city)
    except ValueError as exc:
        st.error(t(lang, "err_validate", detail=exc))
        st.stop()

    st.caption(t(lang, "data_source", source=state.source,
                 freq=freq_label(lang, series.frequency), n=series.n, h=series.horizon))

    insights = compute_insights(series)
    conf = state.confidence
    results = [
        exponential_smoothing.forecast(series, conf),
        growth_curve.forecast(series, conf),
        mean_reversion.forecast(series, conf),
    ]
    ensemble = combine(series, results, conf)

    eda_section.render(series, insights, lang)
    st.divider()
    models_section.render(series, results, ensemble, lang)
    st.divider()
    _render_factors(external_factor_analysis(series, insights, lang), lang)
    st.divider()
    ensemble_section.render(series, ensemble, lang)

    st.divider()
    st.caption(t(lang, "disclaimer"))


if __name__ == "__main__":
    main()
