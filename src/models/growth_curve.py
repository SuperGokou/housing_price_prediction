"""Log-linear growth-curve forecaster.

Fits ``log(price) ~ a + b * t`` by ordinary least squares and projects the
exponential trend forward. For sub-annual data with enough history it adds an
additive seasonal component (the mean log-residual for each period-of-year).
The prediction interval is multiplicative, derived from the log-residual std.

The OLS log-linear fit defined here is also reused by the mean-reversion model
as its "long-run fair value" anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.periods import future_labels
from src.core.types import MODEL_GROWTH_CURVE, Z_SCORES, ForecastResult, PriceSeries

_LOG_FLOOR = 1e-9


@dataclass(frozen=True)
class LogLinearFit:
    """Result of an OLS fit of ``log(price) ~ a + b * t``."""

    a: float  # intercept
    b: float  # slope (per-period log growth)
    log_residual_std: float
    seasonal: np.ndarray  # additive log-seasonal per period-of-year (len p or 1)


def _safe_log(values: np.ndarray) -> np.ndarray:
    """Natural log guarded against non-positive inputs (defensive)."""
    return np.log(np.maximum(np.asarray(values, dtype=float), _LOG_FLOOR))


def _seasonal_component(log_resid: np.ndarray, periods_per_year: int, n: int) -> np.ndarray:
    """Mean log-residual per period-of-year when enough sub-annual data exists."""
    if periods_per_year <= 1 or n < 2 * periods_per_year:
        return np.zeros(1)
    season = np.zeros(periods_per_year)
    for p in range(periods_per_year):
        members = log_resid[p::periods_per_year]
        season[p] = float(np.mean(members)) if members.size else 0.0
    return season - float(np.mean(season))  # keep seasonal mean-zero


def fit_log_linear(series: PriceSeries) -> LogLinearFit:
    """OLS fit of log(price) on a 0-based time index, plus seasonal residuals."""
    values = np.asarray(series.values, dtype=float)
    n = len(values)
    t = np.arange(n, dtype=float)
    log_y = _safe_log(values)
    if n >= 2 and np.ptp(t) > 0:
        b, a = np.polyfit(t, log_y, 1)
    else:
        a, b = float(log_y[0]), 0.0
    fitted = a + b * t
    log_resid = log_y - fitted
    seasonal = _seasonal_component(log_resid, series.periods_per_year, n)
    deseasonal = log_resid - _apply_seasonal(seasonal, np.arange(n), series.periods_per_year)
    std = float(np.std(deseasonal, ddof=1)) if n >= 3 else float(np.std(log_resid))
    return LogLinearFit(a=float(a), b=float(b), log_residual_std=std, seasonal=seasonal)


def _apply_seasonal(seasonal: np.ndarray, indices: np.ndarray, periods_per_year: int) -> np.ndarray:
    """Map each (0-based) time index to its additive seasonal log term."""
    if seasonal.size <= 1:
        return np.zeros(len(indices))
    return seasonal[indices % periods_per_year]


def trend_value(fit: LogLinearFit, step_index: int, periods_per_year: int) -> float:
    """Fair-value (price) of the trend+seasonal fit at a 0-based time index."""
    season = _apply_seasonal(fit.seasonal, np.array([step_index]), periods_per_year)[0]
    return float(np.exp(fit.a + fit.b * step_index + season))


def forecast(series: PriceSeries, confidence: int = 80) -> ForecastResult:
    """Forecast the next ``series.horizon`` periods with a log-linear trend."""
    fit = fit_log_linear(series)
    n = series.n
    horizon = series.horizon
    ppy = series.periods_per_year
    z = Z_SCORES[confidence]

    path = [trend_value(fit, n - 1 + step, ppy) for step in range(1, horizon + 1)]
    point = path[-1]
    # Multiplicative band from log-residual std, scaled by sqrt(step).
    log_half = z * fit.log_residual_std * float(np.sqrt(horizon))
    lower = max(point * float(np.exp(-log_half)), point * 1e-3)
    upper = point * float(np.exp(log_half))

    rationale = (
        f"对数线性增长曲线：OLS 拟合 log(price)=a+b*t，斜率 b={fit.b:.4f}"
        f"（每期约 {np.expm1(fit.b) * 100:.2f}% 增长），外推 {horizon} 步。"
    )
    if fit.seasonal.size > 1:
        rationale += " 已加入季节性附加项。"
    params = {
        "a": fit.a,
        "b": fit.b,
        "log_residual_std": fit.log_residual_std,
        "seasonal_terms": fit.seasonal.tolist(),
    }
    return ForecastResult(
        model_name=MODEL_GROWTH_CURVE,
        point=point,
        lower=lower,
        upper=upper,
        path=path,
        path_labels=future_labels(series, horizon),
        rationale=rationale,
        params=params,
        confidence=confidence,
    )
