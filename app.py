"""Housing Price Multi-Model Prediction — Streamlit entry point.

Run with:  streamlit run app.py
Loads the bundled Seattle sample by default, so the first render is a full demo.
"""

from __future__ import annotations

import streamlit as st

from src.data.validation import build_series
from src.eda.insights import compute_insights
from src.features.external_factors import ExternalFactor, external_factor_analysis
from src.models import exponential_smoothing, growth_curve, mean_reversion
from src.models.ensemble import combine
from src.ui import ensemble_section, eda_section, models_section
from src.ui.sidebar import render_sidebar

SAMPLE_PATH = "sample_data/seattle.csv"

_DIRECTION_BADGE = {
    "up": ("⬆️ 上修", "#059669"),
    "down": ("⬇️ 下修", "#DC2626"),
    "uncertain": ("↕️ 双向", "#D97706"),
}


def _page_header() -> None:
    st.set_page_config(page_title="房价多模型预测", page_icon="🏠", layout="wide")
    st.title("🏠 房价多模型预测 · Multi-Model Housing Forecaster")
    st.markdown(
        "上传某美国城市过去几年的房价历史，应用以**三种不同逻辑的透明统计模型**推演下一年走势，"
        "并给出加权**集成预测区间**。")


def _render_factors(factors: list[ExternalFactor]) -> None:
    st.subheader("3 · 潜在外部特征影响 (Feature Engineering)")
    cols = st.columns(len(factors))
    for col, f in zip(cols, factors):
        label, color = _DIRECTION_BADGE.get(f.direction, ("↕️ 双向", "#D97706"))
        with col:
            st.markdown(f"##### {f.name}")
            st.markdown(
                f"<span style='background:{color};color:white;padding:2px 8px;"
                f"border-radius:6px;font-size:0.8rem'>{label}</span>",
                unsafe_allow_html=True)
            st.write(f.detail)


def main() -> None:
    _page_header()
    state = render_sidebar(SAMPLE_PATH)

    if state.error:
        st.error(state.error)
        st.stop()
    if state.df is None:
        st.info("请在左侧上传 CSV，或使用内置示例数据。")
        st.stop()

    try:
        series = build_series(state.df, city=state.city)
    except ValueError as exc:
        st.error(f"数据校验失败：{exc}")
        st.stop()

    st.caption(
        f"数据源：{state.source} · 频率：{series.frequency} · "
        f"{series.n} 个观测点 · 预测步长：{series.horizon} 期（=1 年）")

    insights = compute_insights(series)
    conf = state.confidence
    results = [
        exponential_smoothing.forecast(series, conf),
        growth_curve.forecast(series, conf),
        mean_reversion.forecast(series, conf),
    ]
    ensemble = combine(series, results, conf)

    eda_section.render(series, insights)
    st.divider()
    models_section.render(series, results, ensemble)
    st.divider()
    _render_factors(external_factor_analysis(series, insights))
    st.divider()
    ensemble_section.render(series, ensemble)

    st.divider()
    st.caption(
        "免责声明：本工具使用轻量统计方法对小样本进行说明性推演，不构成任何投资、"
        "购房或财务建议。真实预测需纳入利率、就业、库存等外部变量与更高频数据。")


if __name__ == "__main__":
    main()
