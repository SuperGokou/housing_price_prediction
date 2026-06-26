"""Tests for the EDA layer (``src.eda.insights.compute_insights``).

Written test-first (TDD). Fixtures use known closed-form cases so the
descriptive statistics can be asserted exactly (within tolerance). One
realistic smoke test loads the bundled Seattle sample CSV.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.core.types import PriceSeries
from src.core.periods import format_label
from src.eda.insights import compute_insights


# --- Fixture helpers --------------------------------------------------------


def _annual_series(values: list[float], start_year: int = 2010) -> PriceSeries:
    """Build an annual PriceSeries from yearly values (Jan 1 each year)."""
    dates = [datetime(start_year + i, 1, 1) for i in range(len(values))]
    labels = [str(d.year) for d in dates]
    return PriceSeries(
        dates=dates,
        values=np.asarray(values, dtype=float),
        frequency="annual",
        labels=labels,
        city="Testville",
    )


# --- cagr / total_growth ----------------------------------------------------


def test_doubling_over_one_year_gives_unit_total_growth_and_cagr():
    # Arrange: two annual points exactly one year apart, price doubles.
    series = _annual_series([100.0, 200.0], start_year=2010)

    # Act
    insights = compute_insights(series)

    # Assert
    assert insights.total_growth == pytest.approx(1.0, abs=1e-9)
    assert insights.cagr == pytest.approx(1.0, abs=1e-3)


def test_total_growth_is_last_over_first_minus_one():
    # Arrange
    series = _annual_series([200.0, 250.0, 300.0])

    # Act
    insights = compute_insights(series)

    # Assert
    assert insights.total_growth == pytest.approx(0.5, abs=1e-9)


def test_cagr_matches_closed_form_over_multiple_years():
    # Arrange: 100 -> 400 over 2 years -> CAGR = 100%.
    series = _annual_series([100.0, 200.0, 400.0])

    # Act
    insights = compute_insights(series)

    # Assert
    assert insights.cagr == pytest.approx(1.0, abs=1e-3)


# --- trend_r2 ---------------------------------------------------------------


def test_perfectly_geometric_series_has_unit_trend_r2():
    # Arrange: pure geometric growth -> log is perfectly linear -> R^2 == 1.
    series = _annual_series([100.0 * (1.1**i) for i in range(8)])

    # Act
    insights = compute_insights(series)

    # Assert
    assert insights.trend_r2 == pytest.approx(1.0, abs=1e-9)


# --- max_drawdown / peak / trough -------------------------------------------


def test_monotone_increasing_series_has_zero_drawdown():
    # Arrange
    series = _annual_series([100.0, 110.0, 121.0, 133.1, 146.41])

    # Act
    insights = compute_insights(series)

    # Assert
    assert insights.max_drawdown == 0.0
    assert insights.peak_label == "2014"  # last (highest) point
    assert insights.peak_price == pytest.approx(146.41)


def test_peak_then_dip_series_has_negative_drawdown_and_inflection():
    # Arrange: rise to a peak at index 2, then decline.
    series = _annual_series([100.0, 150.0, 200.0, 160.0, 120.0])

    # Act
    insights = compute_insights(series)

    # Assert: peak-to-trough = 120/200 - 1 = -0.4
    assert insights.max_drawdown == pytest.approx(-0.4, abs=1e-9)
    assert insights.max_drawdown < 0
    assert insights.peak_price == pytest.approx(200.0)
    assert insights.peak_label == "2012"
    assert insights.trough_price == pytest.approx(100.0)
    assert insights.trough_label == "2010"
    # YoY (annual) flips from positive (rising) to negative (falling).
    assert len(insights.inflections) >= 1


def test_inflection_label_is_the_later_point_when_sign_flips():
    # Arrange: up, up, down. YoY signs: +, +, -, - -> one flip into 2013.
    series = _annual_series([100.0, 120.0, 140.0, 100.0, 90.0])

    # Act
    insights = compute_insights(series)

    # Assert: sign change happens at the 2013 datapoint.
    assert "2013" in insights.inflections


# --- constant series (degenerate guards) ------------------------------------


def test_constant_series_has_zero_growth_and_zero_volatility():
    # Arrange: flat prices.
    series = _annual_series([300.0, 300.0, 300.0, 300.0])

    # Act
    insights = compute_insights(series)

    # Assert
    assert insights.cagr == pytest.approx(0.0, abs=1e-12)
    assert insights.total_growth == pytest.approx(0.0, abs=1e-12)
    assert insights.annualized_volatility == pytest.approx(0.0, abs=1e-12)
    assert insights.max_drawdown == 0.0
    assert insights.trend_r2 == pytest.approx(0.0, abs=1e-9)


def test_three_point_series_does_not_crash():
    # Arrange: minimum-length series (n == 3).
    series = _annual_series([100.0, 105.0, 110.0])

    # Act
    insights = compute_insights(series)

    # Assert: volatility is well-defined for >= 2 returns.
    assert insights.annualized_volatility >= 0.0
    assert insights.span_years == pytest.approx(2.0, abs=0.02)


# --- volatility -------------------------------------------------------------


def test_volatility_uses_sample_std_of_simple_returns_annualized():
    # Arrange: annual returns of +10% then -10%.
    series = _annual_series([100.0, 110.0, 99.0])
    returns = np.array([0.1, -0.1])
    expected = float(np.std(returns, ddof=1))  # periods_per_year == 1 -> sqrt(1)

    # Act
    insights = compute_insights(series)

    # Assert
    assert insights.annualized_volatility == pytest.approx(expected, abs=1e-12)


def test_single_period_return_gives_zero_volatility():
    # Arrange: only one return (n == 2) -> ddof=1 std undefined -> 0.0.
    series = _annual_series([100.0, 130.0])

    # Act
    insights = compute_insights(series)

    # Assert
    assert insights.annualized_volatility == 0.0


# --- yoy for sub-annual data ------------------------------------------------


def test_yoy_growth_for_quarterly_data_compares_four_back():
    # Arrange: 8 quarters, each year doubling -> YoY == 1.0 for points 4..7.
    base = [100.0, 110.0, 120.0, 130.0]
    values = base + [v * 2 for v in base]
    dates = [datetime(2020, 1 + 3 * (i % 4), 1) for i in range(4)]
    dates += [datetime(2021, 1 + 3 * (i % 4), 1) for i in range(4)]
    labels = [f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in dates]
    series = PriceSeries(
        dates=dates,
        values=np.asarray(values, dtype=float),
        frequency="quarterly",
        labels=labels,
    )

    # Act
    insights = compute_insights(series)

    # Assert: four eligible YoY points, all equal to +100%.
    assert len(insights.yoy_growth) == 4
    assert insights.yoy_growth == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert insights.yoy_labels == ["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"]


# --- scalar passthroughs ----------------------------------------------------


def test_mean_and_latest_price_passthrough():
    # Arrange
    series = _annual_series([100.0, 200.0, 300.0])

    # Act
    insights = compute_insights(series)

    # Assert
    assert insights.mean_price == pytest.approx(200.0)
    assert insights.latest_price == pytest.approx(300.0)


def test_does_not_mutate_input_series():
    # Arrange
    series = _annual_series([100.0, 120.0, 90.0, 130.0])
    values_before = series.values.copy()

    # Act
    compute_insights(series)

    # Assert
    np.testing.assert_array_equal(series.values, values_before)


# --- realistic smoke test ---------------------------------------------------


def _load_seattle_series() -> PriceSeries:
    """Build a monthly PriceSeries straight from the bundled Seattle CSV.

    Constructed with pandas only (no dependency on the data-ingestion layer,
    which is owned by another module) so this suite stays self-contained.
    """
    df = pd.read_csv("sample_data/seattle.csv")
    dates = [d.to_pydatetime() for d in pd.to_datetime(df["date"])]
    values = np.asarray(df["price"], dtype=float)
    labels = [format_label(d, "monthly") for d in dates]
    return PriceSeries(
        dates=dates,
        values=values,
        frequency="monthly",
        labels=labels,
        city="Seattle, WA",
    )


def test_seattle_sample_smoke():
    # Arrange: load the bundled monthly Seattle CSV.
    series = _load_seattle_series()

    # Act
    insights = compute_insights(series)

    # Assert: sane, finite descriptive stats for a real monthly series.
    assert insights.annualized_volatility >= 0.0
    assert insights.max_drawdown <= 0.0
    assert 0.0 <= insights.trend_r2 <= 1.0
    assert np.isfinite(insights.cagr)
    assert insights.latest_price == pytest.approx(788000.0)
