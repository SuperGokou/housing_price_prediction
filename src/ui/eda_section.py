"""Section 1: EDA — key metrics, YoY chart, and an auto-written insight."""

from __future__ import annotations

import streamlit as st

from src.core.types import EdaInsights, PriceSeries
from src.ui.format import fmt_pct, fmt_usd
from src.viz import charts


def _insight_text(series: PriceSeries, ins: EdaInsights) -> str:
    city = series.city or "该城市"
    parts = [
        f"过去约 **{ins.span_years:.1f} 年**，{city}房价从 {fmt_usd(series.first_value)} "
        f"变化到 {fmt_usd(ins.latest_price)}，累计 **{fmt_pct(ins.total_growth, signed=True)}**，"
        f"复合年增长率 (CAGR) **{fmt_pct(ins.cagr, signed=True)}**。",
        f"期间峰值 {fmt_usd(ins.peak_price)}（{ins.peak_label}），"
        f"谷值 {fmt_usd(ins.trough_price)}（{ins.trough_label}），"
        f"最大回撤 **{fmt_pct(ins.max_drawdown)}**；年化波动率 {fmt_pct(ins.annualized_volatility)}，"
        f"对数线性趋势拟合 R² = {ins.trend_r2:.2f}。",
    ]
    if ins.inflections:
        parts.append(f"检测到趋势拐点：**{', '.join(ins.inflections)}**（同比增速变向）。")
    else:
        parts.append("未检测到明显的同比拐点，趋势相对单向。")
    return " ".join(parts)


def render(series: PriceSeries, insights: EdaInsights) -> None:
    st.subheader("1 · 数据趋势初步洞察 (EDA)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR 复合年增长", fmt_pct(insights.cagr, signed=True))
    c2.metric("累计涨幅", fmt_pct(insights.total_growth, signed=True))
    c3.metric("年化波动率", fmt_pct(insights.annualized_volatility))
    c4.metric("最大回撤", fmt_pct(insights.max_drawdown))

    st.markdown(_insight_text(series, insights))

    if insights.yoy_growth:
        st.plotly_chart(charts.yoy_growth(insights), use_container_width=True)
