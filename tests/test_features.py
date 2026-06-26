"""Tests for the external-factor narrative generator."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.core.types import EdaInsights, PriceSeries
from src.features.external_factors import external_factor_analysis


def _series(city: str = "Seattle, WA") -> PriceSeries:
    dates = [datetime(2019 + i, 12, 31) for i in range(5)]
    return PriceSeries(dates=dates, values=np.array([600, 650, 720, 780, 760], float),
                       frequency="annual", labels=[str(d.year) for d in dates], city=city)


def _insights(last_yoy: float) -> EdaInsights:
    return EdaInsights(cagr=0.06, total_growth=0.27, annualized_volatility=0.05,
                       mean_price=700.0, latest_price=760.0, peak_price=780.0,
                       peak_label="2022", trough_price=600.0, trough_label="2019",
                       max_drawdown=-0.026, yoy_growth=[0.08, 0.11, 0.08, last_yoy],
                       yoy_labels=["2020", "2021", "2022", "2023"], inflections=["2023"],
                       trend_r2=0.95, span_years=4.0)


def test_returns_three_factors_with_valid_directions():
    factors = external_factor_analysis(_series(), _insights(last_yoy=-0.03))

    assert len(factors) == 3
    assert all(f.direction in {"up", "down", "uncertain"} for f in factors)
    assert all(f.name and f.detail for f in factors)


def test_cooling_market_tilts_rates_downward():
    cooling = external_factor_analysis(_series(), _insights(last_yoy=-0.05))
    assert cooling[0].direction == "down"


def test_hot_market_marks_rates_uncertain():
    hot = external_factor_analysis(_series(), _insights(last_yoy=0.07))
    assert hot[0].direction == "uncertain"


def test_city_name_is_woven_into_employment_factor():
    factors = external_factor_analysis(_series(city="Austin, TX"), _insights(0.04))
    assert "Austin, TX" in factors[1].name
