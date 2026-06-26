"""Regression tests for the code-review edge-case fixes.

- growth_curve must not emit a degenerate near-zero interval on a constant
  series (HIGH-2: log_residual_std floor).
- _trend_r2 must stay finite/bounded and warning-free on non-positive values
  (HIGH-1: np.log guard).
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np

from src.core.types import PriceSeries
from src.eda.insights import _trend_r2
from src.models import growth_curve


def _constant_series(price: float = 500_000.0) -> PriceSeries:
    dates = [datetime(2020, m, 1) for m in range(1, 13)]
    return PriceSeries(
        dates=dates,
        values=np.full(12, price),
        frequency="monthly",
        labels=[f"2020-{m:02d}" for m in range(1, 13)],
        city="Test",
    )


def test_growth_curve_constant_series_has_meaningful_band():
    result = growth_curve.forecast(_constant_series(), confidence=80)

    assert result.lower < result.point < result.upper
    assert result.upper - result.lower > 100  # not a degenerate ~$0 band


def test_trend_r2_is_finite_and_warning_free_on_nonpositive():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning from log() fails here
        r2 = _trend_r2(np.array([100.0, -50.0, 200.0, 150.0]))

    assert np.isfinite(r2)
    assert 0.0 <= r2 <= 1.0
