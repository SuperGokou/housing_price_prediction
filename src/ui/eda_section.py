"""Section 1: EDA — key metrics, YoY chart, and an auto-written insight."""

from __future__ import annotations

import streamlit as st

from src.core.types import EdaInsights, PriceSeries
from src.ui.format import fmt_pct
from src.ui.i18n import eda_insight, t
from src.viz import charts


def render(series: PriceSeries, insights: EdaInsights, lang: str = "zh") -> None:
    st.subheader(t(lang, "eda_title"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t(lang, "m_cagr"), fmt_pct(insights.cagr, signed=True))
    c2.metric(t(lang, "m_total_growth"), fmt_pct(insights.total_growth, signed=True))
    c3.metric(t(lang, "m_volatility"), fmt_pct(insights.annualized_volatility))
    c4.metric(t(lang, "m_drawdown"), fmt_pct(insights.max_drawdown))

    st.markdown(eda_insight(lang, series, insights))

    if insights.yoy_growth:
        st.plotly_chart(charts.yoy_growth(insights, lang), use_container_width=True)
