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
from src.ui.i18n import model_label, t

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
    lang: str = "zh",
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
                             name=t(lang, "trace_spread")))

    # History.
    fig.add_trace(go.Scatter(x=hist_labels, y=hist_vals, mode="lines+markers",
                             name=t(lang, "trace_history"),
                             line=dict(color=COLOR_HISTORY, width=2.5),
                             marker=dict(size=4)))

    # Each model path, anchored to the last actual.
    for r in results:
        color = MODEL_COLORS.get(r.model_name, "#6B7280")
        fig.add_trace(go.Scatter(
            x=[last_label] + list(r.path_labels), y=[last_val] + list(r.path),
            mode="lines", name=model_label(lang, r.model_name),
            line=dict(color=color, width=1.8, dash="dot")))

    # Ensemble path + final interval band.
    ens_path = _ensemble_path(results, ensemble.weights)
    fig.add_trace(go.Scatter(
        x=[last_label] + list(fut_labels), y=[last_val] + ens_path, mode="lines",
        name=t(lang, "trace_ensemble"), line=dict(color=COLOR_ENSEMBLE, width=3.5)))
    fig.add_trace(go.Scatter(
        x=[fut_labels[-1], fut_labels[-1]], y=[ensemble.lower, ensemble.upper],
        mode="lines", line=dict(color=COLOR_ENSEMBLE, width=10),
        opacity=0.25, name=t(lang, "trace_interval", conf=ensemble.confidence)))

    title = t(lang, "chart_main_title", city=series.city or t(lang, "default_city"))
    fig.update_layout(title=_title(title), yaxis_title=t(lang, "chart_price_axis"), **_LAYOUT)
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def model_comparison(results: list[ForecastResult], ensemble: EnsembleResult,
                     lang: str = "zh") -> go.Figure:
    """Point estimate + interval for each model and the ensemble."""
    fig = go.Figure()
    rows = [(model_label(lang, r.model_name), r.point, r.lower, r.upper,
             MODEL_COLORS.get(r.model_name, "#6B7280")) for r in results]
    rows.append((t(lang, "label_ensemble"), ensemble.point, ensemble.lower,
                 ensemble.upper, COLOR_ENSEMBLE))
    for name, point, low, high, color in rows:
        fig.add_trace(go.Scatter(
            x=[point], y=[name], mode="markers",
            marker=dict(color=color, size=13),
            error_x=dict(type="data", symmetric=False,
                         array=[high - point], arrayminus=[point - low],
                         color=color, thickness=2, width=8),
            name=name, showlegend=False))
    fig.update_layout(title=_title(t(lang, "chart_cmp_title")),
                      xaxis_title=t(lang, "chart_cmp_axis"), height=320, **_LAYOUT)
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    return fig


def yoy_growth(insights: EdaInsights, lang: str = "zh") -> go.Figure:
    """Year-over-year growth bars, colored by sign."""
    fig = go.Figure()
    colors = ["#DC2626" if g < 0 else "#059669" for g in insights.yoy_growth]
    fig.add_trace(go.Bar(x=insights.yoy_labels,
                         y=[g * 100 for g in insights.yoy_growth],
                         marker_color=colors, name=t(lang, "chart_yoy_title")))
    fig.update_layout(title=_title(t(lang, "chart_yoy_title")),
                      yaxis_title=t(lang, "chart_yoy_axis"), height=300, **_LAYOUT)
    fig.update_yaxes(ticksuffix="%")
    return fig
