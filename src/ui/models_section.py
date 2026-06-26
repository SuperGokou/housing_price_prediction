"""Section 2: the three independent model forecasts."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.core.types import EnsembleResult, ForecastResult, PriceSeries
from src.ui.format import fmt_pct, fmt_range, fmt_usd
from src.ui.i18n import model_label, model_rationale, t
from src.viz import charts


def _model_card(col: Any, result: ForecastResult, last_value: float, lang: str) -> None:
    change = result.point / last_value - 1
    with col:
        st.markdown(f"##### {model_label(lang, result.model_name)}")
        st.metric(t(lang, "card_forecast"), fmt_usd(result.point),
                  delta=fmt_pct(change, signed=True))
        st.caption(t(lang, "card_interval", conf=result.confidence,
                     range=fmt_range(result.lower, result.upper)))
        st.write(model_rationale(lang, result))


def render(series: PriceSeries, results: list[ForecastResult],
           ensemble: EnsembleResult, lang: str = "zh") -> None:
    st.subheader(t(lang, "models_title"))
    st.plotly_chart(charts.history_and_forecast(series, results, ensemble, lang),
                    use_container_width=True)

    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        _model_card(col, result, series.last_value, lang)

    st.plotly_chart(charts.model_comparison(results, ensemble, lang),
                    use_container_width=True)
