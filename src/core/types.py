"""Core domain types shared across every layer of the app.

These dataclasses are the contracts that the data, EDA, model, and UI layers
all agree on. Everything here is pure data — no I/O, no Streamlit, no plotting.

Conventions:
- All prices are USD floats and strictly positive.
- A :class:`PriceSeries` is always chronologically sorted (ascending dates).
- Growth/return values are fractions (0.07 == 7%), never percentages.
- These objects are treated as immutable; never mutate them in place — build a
  new instance instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# --- Constants --------------------------------------------------------------

#: Z-scores for two-sided prediction intervals keyed by confidence percentage.
Z_SCORES: dict[int, float] = {80: 1.2816, 90: 1.6449, 95: 1.9600}

#: How many observation periods make up a calendar year for each frequency.
FREQUENCY_PERIODS_PER_YEAR: dict[str, int] = {
    "annual": 1,
    "quarterly": 4,
    "monthly": 12,
}

#: Canonical display names for the three forecasting models.
MODEL_EXP_SMOOTHING = "指数平滑 (Holt)"
MODEL_GROWTH_CURVE = "增长曲线 (对数线性)"
MODEL_MEAN_REVERSION = "均值回归 (保守)"


# --- Value objects ----------------------------------------------------------


@dataclass
class PriceSeries:
    """A validated, chronologically sorted housing-price time series.

    Attributes:
        dates: Parsed observation timestamps, strictly ascending.
        values: float64 array of prices (> 0), aligned with ``dates``.
        frequency: One of ``FREQUENCY_PERIODS_PER_YEAR`` keys.
        labels: Human-readable period labels (e.g. ``"2021"`` or ``"2021-Q3"``).
        city: Optional city name for display, e.g. ``"Seattle, WA"``.
    """

    dates: list[datetime]
    values: np.ndarray
    frequency: str
    labels: list[str]
    city: str = ""

    @property
    def periods_per_year(self) -> int:
        return FREQUENCY_PERIODS_PER_YEAR[self.frequency]

    @property
    def n(self) -> int:
        return int(len(self.values))

    @property
    def first_value(self) -> float:
        return float(self.values[0])

    @property
    def last_value(self) -> float:
        return float(self.values[-1])

    @property
    def horizon(self) -> int:
        """Number of forecast steps that make up one year ahead."""
        return self.periods_per_year

    @property
    def span_years(self) -> float:
        """Total calendar years between the first and last observation."""
        delta_days = (self.dates[-1] - self.dates[0]).days
        return delta_days / 365.25 if delta_days > 0 else 0.0


@dataclass
class EdaInsights:
    """Descriptive statistics produced by the EDA layer.

    All growth/return fields are fractions (0.07 == 7%).
    """

    cagr: float
    total_growth: float
    annualized_volatility: float
    mean_price: float
    latest_price: float
    peak_price: float
    peak_label: str
    trough_price: float
    trough_label: str
    max_drawdown: float  # most negative peak-to-trough decline, <= 0
    yoy_growth: list[float]
    yoy_labels: list[str]
    inflections: list[str]  # period labels where YoY growth changed sign
    trend_r2: float  # R^2 of the log-linear trend fit (0..1)
    span_years: float


@dataclass
class ForecastResult:
    """A single model's forecast for the one-year-ahead horizon.

    Attributes:
        model_name: Display name (one of the ``MODEL_*`` constants).
        point: Headline point estimate at the +1-year horizon.
        lower: Lower bound of the prediction interval at ``confidence``.
        upper: Upper bound of the prediction interval at ``confidence``.
        path: Forecasted values for every step of the horizon (len == horizon).
        path_labels: Period labels aligned with ``path``.
        rationale: Short human explanation of the model's reasoning.
        params: Fitted parameters (e.g. alpha/beta, slope, theta).
        confidence: Interval confidence percentage (one of ``Z_SCORES``).
    """

    model_name: str
    point: float
    lower: float
    upper: float
    path: list[float]
    path_labels: list[str]
    rationale: str
    params: dict = field(default_factory=dict)
    confidence: int = 80


@dataclass
class EnsembleResult:
    """The weighted combination of every model's forecast."""

    point: float
    lower: float
    upper: float
    weights: dict  # model_name -> weight (sums to 1.0)
    weighting_method: str  # "expert" or "backtest-inverse-error"
    rationale: str
    members: list[ForecastResult]
    confidence: int = 80
