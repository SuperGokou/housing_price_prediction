"""End-to-end pipeline smoke tests + chart builder sanity checks."""

from __future__ import annotations

import plotly.graph_objects as go

from src.data.io import read_csv
from src.data.validation import build_series
from src.eda.insights import compute_insights
from src.models import exponential_smoothing, growth_curve, mean_reversion
from src.models.ensemble import combine
from src.viz import charts

SAMPLE = "sample_data/seattle.csv"


def _run_pipeline(path: str = SAMPLE):
    series = build_series(read_csv(path), city="Seattle, WA")
    insights = compute_insights(series)
    results = [
        exponential_smoothing.forecast(series, 80),
        growth_curve.forecast(series, 80),
        mean_reversion.forecast(series, 80),
    ]
    ensemble = combine(series, results, 80)
    return series, insights, results, ensemble


def test_full_pipeline_produces_sane_ensemble():
    series, insights, results, ensemble = _run_pipeline()

    assert series.n == 72
    assert insights.cagr > 0
    assert len(results) == 3
    # Ensemble point sits within the span of member point estimates.
    points = [r.point for r in results]
    assert min(points) <= ensemble.point <= max(points)
    assert ensemble.lower < ensemble.point < ensemble.upper
    assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-9


def test_pipeline_handles_annual_sample():
    series, _, results, ensemble = _run_pipeline("sample_data/seattle_annual.csv")
    assert series.frequency == "annual"
    assert series.horizon == 1
    assert all(len(r.path) == 1 for r in results)
    assert ensemble.weighting_method == "expert"  # n < 8 -> expert weights


def test_charts_build_without_error():
    series, insights, results, ensemble = _run_pipeline()

    assert isinstance(charts.history_and_forecast(series, results, ensemble), go.Figure)
    assert isinstance(charts.model_comparison(results, ensemble), go.Figure)
    assert isinstance(charts.yoy_growth(insights), go.Figure)


def test_history_forecast_chart_has_all_series():
    series, _, results, ensemble = _run_pipeline()
    fig = charts.history_and_forecast(series, results, ensemble)
    names = {t.name for t in fig.data}
    assert "历史房价" in names
    assert "集成预测" in names
