"""Plotly chart builders for the housing-price app.

Pure functions: each takes domain objects and returns a ``plotly.graph_objects``
Figure. No Streamlit, no I/O — so they are easy to smoke-test.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.core.types import (
    MODEL_EXP_SMOOTHING,
    MODEL_GROWTH_CURVE,
    MODEL_MEAN_REVERSION,
    EdaInsights,
    EnsembleResult,
    ForecastResult,
    PriceSeries,
)

# Deliberate, distinct palette (one color = one model, reused everywhere).
COLOR_HISTORY = "#111827"
COLOR_ENSEMBLE = "#7C3AED"
COLOR_BAND = "rgba(124, 58, 237, 0.12)"
COLOR_SPREAD = "rgba(17, 24, 39, 0.07)"
MODEL_COLORS = {
    MODEL_EXP_SMOOTHING: "#2563EB",
    MODEL_GROWTH_CURVE: "#059669",
    MODEL_MEAN_REVERSION: "#D97706",
}

_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", color="#1A1B25"),
    margin=dict(l=10, r=10, t=92, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
)


def _title(text: str) -> dict:
    """Consistent left-aligned title that sits clear of the top legend."""
    return dict(text=text, x=0, xanchor="left", y=0.97, yanchor="top",
                font=dict(size=15))


def _ensemble_path(results: list[ForecastResult], weights: dict) -> list[float]:
    """Weighted-average path across member models (matches ensemble point)."""
    paths = np.array([r.path for r in results], dtype=float)
    w = np.array([weights.get(r.model_name, 0.0) for r in results], dtype=float)
    return list(np.average(paths, axis=0, weights=w))


def history_and_forecast(
    series: PriceSeries,
    results: list[ForecastResult],
    ensemble: EnsembleResult,
) -> go.Figure:
    """History line + each model's forecast path + ensemble cone and band."""
    fig = go.Figure()
    hist_labels = list(series.labels)
    hist_vals = [float(v) for v in series.values]
    last_label, last_val = hist_labels[-1], hist_vals[-1]

    # Model-disagreement cone (min/max across model paths).
    fut_labels = results[0].path_labels
    stacked = np.array([r.path for r in results], dtype=float)
    upper_cone = [last_val] + list(stacked.max(axis=0))
    lower_cone = [last_val] + list(stacked.min(axis=0))
    cone_x = [last_label] + list(fut_labels)
    fig.add_trace(go.Scatter(x=cone_x, y=upper_cone, mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=cone_x, y=lower_cone, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=COLOR_SPREAD, hoverinfo="skip",
                             name="模型分歧区间"))

    # History.
    fig.add_trace(go.Scatter(x=hist_labels, y=hist_vals, mode="lines+markers",
                             name="历史房价", line=dict(color=COLOR_HISTORY, width=2.5),
                             marker=dict(size=4)))

    # Each model path, anchored to the last actual.
    for r in results:
        color = MODEL_COLORS.get(r.model_name, "#6B7280")
        fig.add_trace(go.Scatter(
            x=[last_label] + list(r.path_labels), y=[last_val] + list(r.path),
            mode="lines", name=r.model_name,
            line=dict(color=color, width=1.8, dash="dot")))

    # Ensemble path + final interval band.
    ens_path = _ensemble_path(results, ensemble.weights)
    fig.add_trace(go.Scatter(
        x=[last_label] + list(fut_labels), y=[last_val] + ens_path, mode="lines",
        name="集成预测", line=dict(color=COLOR_ENSEMBLE, width=3.5)))
    fig.add_trace(go.Scatter(
        x=[fut_labels[-1], fut_labels[-1]], y=[ensemble.lower, ensemble.upper],
        mode="lines", line=dict(color=COLOR_ENSEMBLE, width=10),
        opacity=0.25, name=f"{ensemble.confidence}% 区间"))

    title = f"{series.city or '房价'} — 历史走势与下一年多模型预测"
    fig.update_layout(title=_title(title), yaxis_title="房价 (USD)", **_LAYOUT)
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def model_comparison(results: list[ForecastResult], ensemble: EnsembleResult) -> go.Figure:
    """Point estimate + interval for each model and the ensemble."""
    fig = go.Figure()
    rows = [(r.model_name, r.point, r.lower, r.upper,
             MODEL_COLORS.get(r.model_name, "#6B7280")) for r in results]
    rows.append(("集成预测", ensemble.point, ensemble.lower, ensemble.upper, COLOR_ENSEMBLE))
    for name, point, low, high, color in rows:
        fig.add_trace(go.Scatter(
            x=[point], y=[name], mode="markers",
            marker=dict(color=color, size=13),
            error_x=dict(type="data", symmetric=False,
                         array=[high - point], arrayminus=[point - low],
                         color=color, thickness=2, width=8),
            name=name, showlegend=False))
    fig.update_layout(title=_title("各模型点估计与区间对比"),
                      xaxis_title="+1 年预测房价 (USD)", height=320, **_LAYOUT)
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    return fig


def yoy_growth(insights: EdaInsights) -> go.Figure:
    """Year-over-year growth bars, colored by sign."""
    fig = go.Figure()
    colors = ["#DC2626" if g < 0 else "#059669" for g in insights.yoy_growth]
    fig.add_trace(go.Bar(x=insights.yoy_labels,
                         y=[g * 100 for g in insights.yoy_growth],
                         marker_color=colors, name="同比增长"))
    fig.update_layout(title=_title("同比增长率 (YoY)"), yaxis_title="增长率 (%)",
                      height=300, **_LAYOUT)
    fig.update_yaxes(ticksuffix="%")
    return fig
