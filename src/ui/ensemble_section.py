"""Section 4: the weighted ensemble forecast and its rationale."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.core.types import EnsembleResult, PriceSeries
from src.ui.format import fmt_pct, fmt_range, fmt_usd

_METHOD_LABELS = {
    "expert": "专家固定权重",
    "backtest-inverse-error": "回测逆误差自动权重",
}


def render(series: PriceSeries, ensemble: EnsembleResult) -> None:
    st.subheader("4 · 综合集成预测 (Ensemble)")
    change = ensemble.point / series.last_value - 1

    left, right = st.columns([1, 1])
    with left:
        st.metric(f"最终预测（下一年，{series.city or '该城市'}）",
                  fmt_usd(ensemble.point), delta=fmt_pct(change, signed=True))
        st.markdown(f"**{ensemble.confidence}% 预测区间：** {fmt_range(ensemble.lower, ensemble.upper)}")
        method = _METHOD_LABELS.get(ensemble.weighting_method, ensemble.weighting_method)
        st.caption(f"权重方法：{method}")
        st.write(ensemble.rationale)

    with right:
        weights_df = pd.DataFrame(
            [{"模型": name, "权重": f"{w:.0%}"} for name, w in ensemble.weights.items()]
        )
        st.markdown("**模型权重分配**")
        st.dataframe(weights_df, hide_index=True, use_container_width=True)
