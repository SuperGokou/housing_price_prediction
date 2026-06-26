"""Tests for the ensemble combiner.

Covers weight normalisation, point bounding, interval ordering, and the
switch from expert weights (n<8) to backtest inverse-error weights (n>=8).
The realistic n>=8 case loads ``sample_data/seattle.csv`` (monthly).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.core.periods import format_label
from src.core.types import (
    MODEL_EXP_SMOOTHING,
    MODEL_GROWTH_CURVE,
    MODEL_MEAN_REVERSION,
    Z_SCORES,
    PriceSeries,
)
from src.models import ensemble, exponential_smoothing, growth_curve, mean_reversion

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "sample_data" / "seattle.csv"


# --- Fixture builders -------------------------------------------------------


def _make_series(values, frequency: str = "annual") -> PriceSeries:
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
    return PriceSeries(dates=dates, values=arr, frequency=frequency, labels=labels)


def _load_seattle() -> PriceSeries:
    dates, values = [], []
    with SAMPLE_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            dates.append(datetime.strptime(row["date"], "%Y-%m-%d"))
            values.append(float(row["price"]))
    labels = [format_label(d, "monthly") for d in dates]
    return PriceSeries(
        dates=dates,
        values=np.asarray(values, dtype=float),
        frequency="monthly",
        labels=labels,
        city="Seattle, WA",
    )


def _all_results(series: PriceSeries, confidence: int = 80):
    return [
        exponential_smoothing.forecast(series, confidence),
        growth_curve.forecast(series, confidence),
        mean_reversion.forecast(series, confidence),
    ]


# --- Core ensemble properties ----------------------------------------------


def test_weights_sum_to_one_expert():
    series = _make_series([100.0 * 1.08**i for i in range(5)], "annual")  # n=5 < 8
    result = ensemble.combine(series, _all_results(series))
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_weights_sum_to_one_backtest():
    series = _load_seattle()  # n=72 >= 8
    result = ensemble.combine(series, _all_results(series))
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_point_within_member_range():
    series = _make_series([100.0 * 1.08**i for i in range(5)], "annual")
    members = _all_results(series)
    result = ensemble.combine(series, members)
    points = [m.point for m in members]
    assert min(points) - 1e-6 <= result.point <= max(points) + 1e-6


def test_interval_ordering_and_positive():
    series = _make_series([100.0 * 1.08**i for i in range(5)], "annual")
    result = ensemble.combine(series, _all_results(series))
    assert result.lower < result.point < result.upper
    assert result.lower > 0


def test_expert_weighting_when_small_sample():
    series = _make_series([100.0 * 1.05**i for i in range(5)], "annual")  # n=5
    result = ensemble.combine(series, _all_results(series))
    assert result.weighting_method == "expert"
    assert result.weights[MODEL_EXP_SMOOTHING] == pytest.approx(0.40)
    assert result.weights[MODEL_GROWTH_CURVE] == pytest.approx(0.35)
    assert result.weights[MODEL_MEAN_REVERSION] == pytest.approx(0.25)


def test_backtest_weighting_when_large_sample():
    series = _load_seattle()  # n=72
    result = ensemble.combine(series, _all_results(series))
    assert result.weighting_method == "backtest-inverse-error"
    # All three models present with positive weights.
    assert set(result.weights) == {
        MODEL_EXP_SMOOTHING,
        MODEL_GROWTH_CURVE,
        MODEL_MEAN_REVERSION,
    }
    assert all(w > 0 for w in result.weights.values())


def test_backtest_threshold_boundary():
    """Exactly n==8 should trigger the backtest path."""
    series = _make_series([100.0 * 1.04**i for i in range(8)], "annual")
    result = ensemble.combine(series, _all_results(series))
    assert result.weighting_method == "backtest-inverse-error"


def test_n_seven_still_expert():
    series = _make_series([100.0 * 1.04**i for i in range(7)], "annual")
    result = ensemble.combine(series, _all_results(series))
    assert result.weighting_method == "expert"


def test_members_preserved_and_rationale():
    series = _load_seattle()
    members = _all_results(series)
    result = ensemble.combine(series, members)
    assert result.members == members
    assert isinstance(result.rationale, str) and result.rationale.strip()


def test_confidence_propagated_and_widens():
    series = _make_series([100.0 * 1.06**i for i in range(5)], "annual")
    r80 = ensemble.combine(series, _all_results(series, 80), confidence=80)
    r95 = ensemble.combine(series, _all_results(series, 95), confidence=95)
    assert r80.confidence == 80
    assert r95.confidence == 95
    assert (r95.upper - r95.lower) >= (r80.upper - r80.lower)


def test_seattle_point_is_reasonable():
    """Sanity: Seattle ensemble point should land near recent price levels."""
    series = _load_seattle()
    result = ensemble.combine(series, _all_results(series))
    assert 600_000 < result.point < 1_100_000
    assert result.lower < result.point < result.upper


def test_z_score_consistency():
    series = _load_seattle()
    members = _all_results(series, 80)
    result = ensemble.combine(series, members, confidence=80)
    # sigma reconstructed from bounds should be positive and finite.
    z = Z_SCORES[80]
    sigma = (result.upper - result.point) / z
    assert sigma > 0 and np.isfinite(sigma)
