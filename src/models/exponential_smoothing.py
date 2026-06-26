"""Holt's linear (double) exponential smoothing forecaster.

A transparent, dependency-light stand-in for an ETS/ARIMA-style model. The
``alpha`` (level) and ``beta`` (trend) smoothing parameters are chosen by grid
search over ``{0.1, ..., 0.9}`` to minimise the in-sample one-step-ahead SSE;
on samples too small to fit (n < 4) we fall back to sensible defaults.

Forecast at step ``h`` ahead is ``level + h * trend``; the prediction interval
widens with the square root of the step using the one-step residual standard
deviation.
"""

from __future__ import annotations

import numpy as np

from src.core.periods import future_labels
from src.core.types import MODEL_EXP_SMOOTHING, Z_SCORES, ForecastResult, PriceSeries

_GRID = [round(0.1 * k, 1) for k in range(1, 10)]  # 0.1 .. 0.9
_DEFAULT_ALPHA = 0.5
_DEFAULT_BETA = 0.3
_MIN_FIT_N = 4


def _holt_one_step(values: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Return in-sample one-step-ahead forecasts for the given smoothing pair.

    The forecast for observation ``t`` is made from the level/trend fitted on
    observations up to ``t-1``; entry 0 is left as NaN (no prior history).
    """
    n = len(values)
    preds = np.full(n, np.nan)
    level = float(values[0])
    trend = float(values[1] - values[0]) if n > 1 else 0.0
    for t in range(1, n):
        preds[t] = level + trend  # one-step-ahead forecast for time t
        prev_level = level
        level = alpha * values[t] + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return preds


def _final_state(values: np.ndarray, alpha: float, beta: float) -> tuple[float, float]:
    """Return the final (level, trend) after running Holt's recursion."""
    level = float(values[0])
    trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
    for t in range(1, len(values)):
        prev_level = level
        level = alpha * values[t] + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return level, trend


def _grid_search(values: np.ndarray) -> tuple[float, float]:
    """Pick (alpha, beta) minimising one-step SSE; fall back if sample tiny."""
    if len(values) < _MIN_FIT_N:
        return _DEFAULT_ALPHA, _DEFAULT_BETA
    best, best_sse = (_DEFAULT_ALPHA, _DEFAULT_BETA), np.inf
    for alpha in _GRID:
        for beta in _GRID:
            preds = _holt_one_step(values, alpha, beta)
            resid = values[1:] - preds[1:]
            sse = float(np.sum(resid**2))
            if sse < best_sse:
                best, best_sse = (alpha, beta), sse
    return best


def _residual_std(values: np.ndarray, alpha: float, beta: float) -> float:
    """One-step-ahead residual std, with a small positive floor."""
    preds = _holt_one_step(values, alpha, beta)
    resid = values[1:] - preds[1:]
    if resid.size >= 2:
        std = float(np.std(resid, ddof=1))
    else:
        std = 0.0
    floor = max(abs(float(values[-1])) * 1e-3, 1e-6)
    return max(std, floor)


def forecast(series: PriceSeries, confidence: int = 80) -> ForecastResult:
    """Forecast the next ``series.horizon`` periods with Holt's linear method."""
    values = np.asarray(series.values, dtype=float)
    horizon = series.horizon
    alpha, beta = _grid_search(values)
    level, trend = _final_state(values, alpha, beta)
    std = _residual_std(values, alpha, beta)
    z = Z_SCORES[confidence]

    path = [float(level + step * trend) for step in range(1, horizon + 1)]
    point = path[-1]
    half = z * std * float(np.sqrt(horizon))
    lower = max(point - half, point * 1e-3)

    rationale = (
        f"Holt 线性指数平滑：网格搜索得到 alpha={alpha}, beta={beta}，"
        f"按 level + h*trend 外推 {horizon} 步，趋势项 trend={trend:.1f}。"
    )
    params = {
        "alpha": alpha,
        "beta": beta,
        "level": float(level),
        "trend": float(trend),
        "residual_std": std,
    }
    return ForecastResult(
        model_name=MODEL_EXP_SMOOTHING,
        point=point,
        lower=lower,
        upper=point + half,
        path=path,
        path_labels=future_labels(series, horizon),
        rationale=rationale,
        params=params,
        confidence=confidence,
    )
