"""Tests for the three transparent forecasting models.

Fixtures are built directly from :mod:`src.core.types` so these tests do not
depend on the (separately owned) data-loading layer. Each model is exercised
on monotone-up, constant, tiny, and realistic series.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.core.periods import format_label, future_labels
from src.core.types import (
    MODEL_EXP_SMOOTHING,
    MODEL_GROWTH_CURVE,
    MODEL_MEAN_REVERSION,
    Z_SCORES,
    PriceSeries,
)
from src.models import exponential_smoothing, growth_curve, mean_reversion

MODELS = [exponential_smoothing, growth_curve, mean_reversion]
EXPECTED_NAMES = {
    exponential_smoothing: MODEL_EXP_SMOOTHING,
    growth_curve: MODEL_GROWTH_CURVE,
    mean_reversion: MODEL_MEAN_REVERSION,
}


# --- Fixture builders -------------------------------------------------------


def _make_series(values, frequency: str = "annual", city: str = "Test") -> PriceSeries:
    """Build a PriceSeries from a list of prices with synthetic dates."""
    arr = np.asarray(values, dtype=float)
    months = {"annual": 12, "quarterly": 3, "monthly": 1}[frequency]
    dates = []
    year, month = 2010, 1
    for _ in range(len(arr)):
        dates.append(datetime(year, month, 1))
        month += months
        while month > 12:
            month -= 12
            year += 1
    labels = [format_label(d, frequency) for d in dates]
    return PriceSeries(dates=dates, values=arr, frequency=frequency, labels=labels, city=city)


def _monotone_up(frequency: str = "annual", n: int = 6) -> PriceSeries:
    return _make_series([100.0 * (1.08**i) for i in range(n)], frequency)


def _constant(frequency: str = "annual", n: int = 6, value: float = 500.0) -> PriceSeries:
    return _make_series([value] * n, frequency)


# --- Shared structural assertions ------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_path_length_equals_horizon(model):
    series = _monotone_up("monthly", n=30)
    result = model.forecast(series)
    assert len(result.path) == series.horizon
    assert series.horizon == 12


@pytest.mark.parametrize("model", MODELS)
def test_path_labels_match_future_labels(model):
    series = _monotone_up("quarterly", n=12)
    result = model.forecast(series)
    assert result.path_labels == future_labels(series, series.horizon)
    assert len(result.path_labels) == len(result.path)


@pytest.mark.parametrize("model", MODELS)
def test_point_is_last_path_value(model):
    series = _monotone_up("annual", n=6)
    result = model.forecast(series)
    assert result.point == pytest.approx(result.path[-1])


@pytest.mark.parametrize("model", MODELS)
def test_correct_model_name(model):
    result = model.forecast(_monotone_up())
    assert result.model_name == EXPECTED_NAMES[model]


@pytest.mark.parametrize("model", MODELS)
def test_interval_ordering_and_positive(model):
    for series in (_monotone_up(), _constant(), _monotone_up("monthly", 36)):
        result = model.forecast(series)
        assert result.lower < result.point < result.upper
        assert result.lower > 0


@pytest.mark.parametrize("model", MODELS)
def test_monotone_up_forecasts_above_last(model):
    series = _monotone_up("annual", n=6)
    result = model.forecast(series)
    assert result.point > series.last_value


@pytest.mark.parametrize("model", MODELS)
def test_constant_series_forecasts_near_last(model):
    series = _constant("annual", n=6, value=500.0)
    result = model.forecast(series)
    assert result.point == pytest.approx(series.last_value, rel=0.02)


@pytest.mark.parametrize("model", MODELS)
def test_tiny_sample_does_not_crash(model):
    series = _make_series([100.0, 110.0, 125.0], "annual")
    result = model.forecast(series)
    assert len(result.path) == series.horizon
    assert result.lower < result.point < result.upper
    assert result.lower > 0


@pytest.mark.parametrize("model", MODELS)
def test_params_and_rationale_populated(model):
    result = model.forecast(_monotone_up())
    assert isinstance(result.params, dict) and result.params
    assert isinstance(result.rationale, str) and result.rationale.strip()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("confidence", [80, 90, 95])
def test_confidence_widens_interval(model, confidence):
    series = _monotone_up("monthly", n=36)
    result = model.forecast(series, confidence=confidence)
    assert result.confidence == confidence
    half_width = result.upper - result.lower
    assert half_width > 0
    # Higher confidence -> wider band.
    narrow = model.forecast(series, confidence=80)
    if confidence > 80:
        assert (result.upper - result.lower) >= (narrow.upper - narrow.lower)


# --- Model-specific behaviour ----------------------------------------------


def test_mean_reversion_is_most_conservative():
    """Mean reversion must stay closest to the last value among the three."""
    series = _monotone_up("annual", n=8)
    last = series.last_value
    mr = mean_reversion.forecast(series).point
    es = exponential_smoothing.forecast(series).point
    gc = growth_curve.forecast(series).point
    assert abs(mr - last) <= abs(es - last)
    assert abs(mr - last) <= abs(gc - last)


def test_growth_curve_uses_log_linear_params():
    result = growth_curve.forecast(_monotone_up("annual", n=8))
    assert "b" in result.params  # log-linear slope


def test_exp_smoothing_grid_search_params():
    result = exponential_smoothing.forecast(_monotone_up("annual", n=8))
    assert 0.1 <= result.params["alpha"] <= 0.9
    assert 0.1 <= result.params["beta"] <= 0.9


def test_exp_smoothing_falls_back_on_tiny_sample():
    result = exponential_smoothing.forecast(_make_series([100.0, 110.0, 121.0], "annual"))
    assert result.params["alpha"] == pytest.approx(0.5)
    assert result.params["beta"] == pytest.approx(0.3)


def test_mean_reversion_theta_clamped():
    result = mean_reversion.forecast(_monotone_up("annual", n=8))
    assert 0.15 <= result.params["theta"] <= 0.85


def test_z_score_used_for_interval():
    """Interval half-width should scale with the configured Z score."""
    series = _monotone_up("monthly", n=36)
    r80 = growth_curve.forecast(series, confidence=80)
    r95 = growth_curve.forecast(series, confidence=95)
    ratio = (r95.upper - r95.point) / (r80.upper - r80.point)
    assert ratio == pytest.approx(Z_SCORES[95] / Z_SCORES[80], rel=0.05)
