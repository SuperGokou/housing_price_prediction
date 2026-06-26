"""Section 2: the three independent model forecasts."""

from __future__ import annotations

import streamlit as st

from src.core.types import EnsembleResult, ForecastResult, PriceSeries
from src.ui.format import fmt_pct, fmt_range, fmt_usd
from src.viz import charts


def _model_card(col, result: ForecastResult, last_value: float) -> None:
    change = result.point / last_value - 1
    with col:
        st.markdown(f"##### {result.model_name}")
        st.metric("+1 年预测", fmt_usd(result.point),
                  delta=fmt_pct(change, signed=True))
        st.caption(f"{result.confidence}% 区间：{fmt_range(result.lower, result.upper)}")
        st.write(result.rationale)


def render(series: PriceSeries, results: list[ForecastResult],
           ensemble: EnsembleResult) -> None:
    st.subheader("2 · 多模型独立预测")
    st.plotly_chart(charts.history_and_forecast(series, results, ensemble),
                    use_container_width=True)

    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        _model_card(col, result, series.last_value)

    st.plotly_chart(charts.model_comparison(results, ensemble),
                    use_container_width=True)
