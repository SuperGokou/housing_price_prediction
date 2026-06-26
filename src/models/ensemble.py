"""Weighted ensemble of the three transparent forecasting models.

Two weighting regimes:

* **Expert weights** (default, and the only option when ``series.n < 8``):
  exp-smoothing 0.40, growth-curve 0.35, mean-reversion 0.25.
* **Backtest inverse-error weights** (``series.n >= 8``): a rolling
  one-step-ahead backtest over the final (up to) three origins refits every
  model on a truncated series and scores its next-step prediction against the
  actual. Each model's weight is ``1 / (MAE + eps)``, normalised.

The combined interval blends within-model variance (each member's own band)
with between-model dispersion (disagreement about the point).
"""

from __future__ import annotations

import numpy as np

from src.core.types import (
    MODEL_EXP_SMOOTHING,
    MODEL_GROWTH_CURVE,
    MODEL_MEAN_REVERSION,
    Z_SCORES,
    EnsembleResult,
    ForecastResult,
    PriceSeries,
)
from src.models import exponential_smoothing, growth_curve, mean_reversion

_EXPERT_WEIGHTS: dict[str, float] = {
    MODEL_EXP_SMOOTHING: 0.40,
    MODEL_GROWTH_CURVE: 0.35,
    MODEL_MEAN_REVERSION: 0.25,
}
_MODEL_FOR_NAME = {
    MODEL_EXP_SMOOTHING: exponential_smoothing,
    MODEL_GROWTH_CURVE: growth_curve,
    MODEL_MEAN_REVERSION: mean_reversion,
}
_BACKTEST_MIN_N = 8
_MAX_ORIGINS = 3
_EPS = 1e-6


def _truncate(series: PriceSeries, end: int) -> PriceSeries:
    """Return a new PriceSeries containing only the first ``end`` observations."""
    return PriceSeries(
        dates=list(series.dates[:end]),
        values=np.asarray(series.values[:end], dtype=float),
        frequency=series.frequency,
        labels=list(series.labels[:end]),
        city=series.city,
    )


def _backtest_mae(series: PriceSeries) -> dict[str, float]:
    """Mean absolute one-step-ahead error per model over the final origins.

    For each origin ``i`` we refit every model on ``series[:i]`` and compare its
    first forecast step (``path[0]``) to the actual value at ``i``.
    """
    n = series.n
    origins = range(max(_BACKTEST_MIN_N - 1, n - _MAX_ORIGINS), n)
    errors: dict[str, list[float]] = {name: [] for name in _MODEL_FOR_NAME}
    for i in origins:
        train = _truncate(series, i)
        actual = float(series.values[i])
        for name, module in _MODEL_FOR_NAME.items():
            predicted = module.forecast(train).path[0]
            errors[name].append(abs(predicted - actual))
    return {name: float(np.mean(errs)) if errs else 0.0 for name, errs in errors.items()}


def _inverse_error_weights(mae: dict[str, float]) -> dict[str, float]:
    """Normalised inverse-MAE weights (lower error -> higher weight)."""
    raw = {name: 1.0 / (err + _EPS) for name, err in mae.items()}
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def _choose_weights(series: PriceSeries) -> tuple[dict[str, float], str, str]:
    """Return (weights, method, rationale) based on the available sample size."""
    if series.n < _BACKTEST_MIN_N:
        rationale = (
            "样本量较小（n<8），采用专家先验权重："
            "指数平滑 0.40 / 增长曲线 0.35 / 均值回归 0.25。"
        )
        return dict(_EXPERT_WEIGHTS), "expert", rationale
    mae = _backtest_mae(series)
    weights = _inverse_error_weights(mae)
    detail = ", ".join(f"{name}: MAE={mae[name]:.0f}" for name in weights)
    rationale = (
        "样本量足够（n>=8），基于滚动一步回测的逆误差加权（权重∝1/MAE）。"
        f" 回测误差 — {detail}。"
    )
    return weights, "backtest-inverse-error", rationale


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    """Re-normalise so weights sum to exactly 1.0 (defensive against drift)."""
    total = sum(weights.values())
    if total <= 0:
        equal = 1.0 / len(weights)
        return {name: equal for name in weights}
    return {name: value / total for name, value in weights.items()}


def _combine_interval(
    members: list[ForecastResult], weights: dict[str, float], point: float, z: float
) -> tuple[float, float]:
    """Blend within-model and between-model variance into a final band."""
    within = 0.0
    between = 0.0
    for member in members:
        w = weights.get(member.model_name, 0.0)
        member_sigma = (member.upper - member.lower) / (2 * z)
        within += w * member_sigma**2
        between += w * (member.point - point) ** 2
    sigma = float(np.sqrt(within + between))
    half = z * sigma
    lower = max(point - half, point * 1e-3)
    return lower, point + half


def combine(
    series: PriceSeries,
    results: list[ForecastResult],
    confidence: int = 80,
) -> EnsembleResult:
    """Combine per-model forecasts into a single weighted ensemble forecast."""
    weights, method, rationale = _choose_weights(series)
    weights = _normalise(weights)
    z = Z_SCORES[confidence]

    point = float(sum(weights.get(m.model_name, 0.0) * m.point for m in results))
    lower, upper = _combine_interval(results, weights, point, z)

    return EnsembleResult(
        point=point,
        lower=lower,
        upper=upper,
        weights=weights,
        weighting_method=method,
        rationale=rationale,
        members=list(results),
        confidence=confidence,
    )
