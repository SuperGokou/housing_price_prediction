"""Descriptive insights (EDA) for a validated housing-price series.

The single public entry point is :func:`compute_insights`, which turns a
:class:`~src.core.types.PriceSeries` into an
:class:`~src.core.types.EdaInsights` value object. Everything here is pure:
no I/O, no mutation of the input, numpy/pandas only.

All growth/return values are fractions (``0.07`` == 7%).
"""

from __future__ import annotations

import numpy as np

from src.core.types import EdaInsights, PriceSeries

#: Below this magnitude a per-period return is treated as flat (sign == 0),
#: so floating-point noise in a constant series never registers as a flip.
_FLAT_RETURN_EPS = 1e-12


def compute_insights(series: PriceSeries) -> EdaInsights:
    """Compute descriptive statistics for ``series``.

    Args:
        series: A validated, chronologically sorted price series (n >= 1).

    Returns:
        An :class:`EdaInsights` populated per the design spec. The input
        ``series`` is never mutated.
    """
    values = np.asarray(series.values, dtype=float)
    first, last = float(values[0]), float(values[-1])

    cagr = _compute_cagr(first, last, series.span_years, series.n)
    volatility = _annualized_volatility(values, series.periods_per_year)
    yoy_growth, yoy_labels = _year_over_year(values, series)
    inflections = _inflections(yoy_growth, yoy_labels)
    peak_idx, trough_idx = int(np.argmax(values)), int(np.argmin(values))

    return EdaInsights(
        cagr=cagr,
        total_growth=last / first - 1.0,
        annualized_volatility=volatility,
        mean_price=float(np.mean(values)),
        latest_price=last,
        peak_price=float(values[peak_idx]),
        peak_label=series.labels[peak_idx],
        trough_price=float(values[trough_idx]),
        trough_label=series.labels[trough_idx],
        max_drawdown=_max_drawdown(values),
        yoy_growth=yoy_growth,
        yoy_labels=yoy_labels,
        inflections=inflections,
        trend_r2=_trend_r2(values),
        span_years=series.span_years,
    )


def _compute_cagr(first: float, last: float, span_years: float, n: int) -> float:
    """Compound annual growth rate of ``first`` -> ``last`` over ``span_years``.

    Falls back to a per-period CAGR over ``n - 1`` periods when the calendar
    span is non-positive (e.g. a single observation or coincident dates).
    """
    ratio = last / first
    if span_years > 0:
        return ratio ** (1.0 / span_years) - 1.0
    periods = max(n - 1, 1)
    return ratio ** (1.0 / periods) - 1.0


def _simple_returns(values: np.ndarray) -> np.ndarray:
    """Per-period simple returns ``v[i] / v[i-1] - 1`` (length n - 1)."""
    return values[1:] / values[:-1] - 1.0


def _annualized_volatility(values: np.ndarray, periods_per_year: int) -> float:
    """Sample std (ddof=1) of per-period returns, annualized by sqrt(periods).

    Returns ``0.0`` when there are fewer than two returns (std is undefined).
    """
    returns = _simple_returns(values)
    if returns.size < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def _year_over_year(
    values: np.ndarray, series: PriceSeries
) -> tuple[list[float], list[str]]:
    """Year-over-year growth and aligned labels.

    For each index ``i >= periods_per_year`` compute
    ``values[i] / values[i - periods_per_year] - 1``. For annual data
    (``periods_per_year == 1``) this reduces to period-over-period growth.
    """
    ppy = series.periods_per_year
    growth: list[float] = []
    labels: list[str] = []
    for i in range(ppy, len(values)):
        growth.append(float(values[i] / values[i - ppy] - 1.0))
        labels.append(series.labels[i])
    return growth, labels


def _sign(value: float) -> int:
    """Sign of ``value`` treating near-zero as flat (0)."""
    if value > _FLAT_RETURN_EPS:
        return 1
    if value < -_FLAT_RETURN_EPS:
        return -1
    return 0


def _inflections(yoy_growth: list[float], yoy_labels: list[str]) -> list[str]:
    """Labels (the later point) where consecutive YoY growth changes sign.

    A change between non-zero signs of opposite direction is an inflection;
    flat segments (sign 0) do not, by themselves, start an inflection.
    """
    result: list[str] = []
    for i in range(1, len(yoy_growth)):
        prev, curr = _sign(yoy_growth[i - 1]), _sign(yoy_growth[i])
        if prev != 0 and curr != 0 and prev != curr:
            result.append(yoy_labels[i])
    return result


def _max_drawdown(values: np.ndarray) -> float:
    """Most negative peak-to-subsequent-trough decline as a fraction (<= 0).

    Returns ``0.0`` when the series never declines below a running peak.
    """
    running_peak = values[0]
    worst = 0.0
    for value in values[1:]:
        if value > running_peak:
            running_peak = value
        else:
            decline = value / running_peak - 1.0
            worst = min(worst, decline)
    return float(worst)


def _trend_r2(values: np.ndarray) -> float:
    """R^2 of the OLS fit ``log(value) ~ t`` over ``t = 0..n-1`` (0..1).

    Returns ``0.0`` when there are fewer than two points or the log-series
    has no variance (a constant series), where R^2 is undefined.
    """
    n = values.size
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=float)
    log_values = np.log(values)
    total_ss = float(np.sum((log_values - log_values.mean()) ** 2))
    if total_ss == 0.0:
        return 0.0
    slope, intercept = np.polyfit(t, log_values, 1)
    residuals = log_values - (slope * t + intercept)
    residual_ss = float(np.sum(residuals**2))
    return float(max(0.0, min(1.0, 1.0 - residual_ss / total_ss)))
