"""Conservative mean-reversion forecaster.

Anchors on the log-linear "fair value" trend (reused from
:mod:`src.models.growth_curve`) and pulls the current price toward it at a
reversion speed ``theta``. ``theta`` is estimated from how quickly past
deviations from trend decayed — an AR(1) coefficient on the deviation series —
then clamped to ``[0.15, 0.85]`` (default 0.4 when it can't be estimated).

Because each step only moves a fraction of the way from the last value toward
fair value, this is by construction the most conservative of the three models
(its point stays closest to the last observed price).
"""

from __future__ import annotations

import numpy as np

from src.core.periods import future_labels
from src.core.types import MODEL_MEAN_REVERSION, Z_SCORES, ForecastResult, PriceSeries
from src.models.growth_curve import LogLinearFit, fit_log_linear, trend_value

_THETA_MIN = 0.15
_THETA_MAX = 0.85
_THETA_DEFAULT = 0.4


def _deviations(series: PriceSeries, fit: LogLinearFit) -> np.ndarray:
    """Price minus log-linear fair value at each historical index."""
    ppy = series.periods_per_year
    fair = np.array([trend_value(fit, i, ppy) for i in range(series.n)])
    return np.asarray(series.values, dtype=float) - fair


def _estimate_theta(deviations: np.ndarray) -> float:
    """AR(1)-style reversion speed from deviation_t on deviation_{t-1}.

    The OLS slope ``rho`` measures persistence; reversion speed is ``1 - rho``.
    Falls back to the default when there is not enough variation to fit.
    """
    if deviations.size < 3:
        return _THETA_DEFAULT
    prev, curr = deviations[:-1], deviations[1:]
    if float(np.ptp(prev)) <= 1e-9:
        return _THETA_DEFAULT
    rho, _ = np.polyfit(prev, curr, 1)
    theta = 1.0 - float(rho)
    if not np.isfinite(theta):
        return _THETA_DEFAULT
    return float(np.clip(theta, _THETA_MIN, _THETA_MAX))


def _residual_std(series: PriceSeries, fit: LogLinearFit, theta: float) -> float:
    """Std of one-step reversion residuals, with a small positive floor."""
    values = np.asarray(series.values, dtype=float)
    ppy = series.periods_per_year
    preds = []
    for i in range(1, series.n):
        fair_i = trend_value(fit, i, ppy)
        preds.append(values[i - 1] + theta * (fair_i - values[i - 1]))
    resid = values[1:] - np.array(preds) if preds else np.array([0.0])
    std = float(np.std(resid, ddof=1)) if resid.size >= 2 else 0.0
    floor = max(abs(float(values[-1])) * 1e-3, 1e-6)
    return max(std, floor)


def _iterate_path(series: PriceSeries, fit: LogLinearFit, theta: float) -> list[float]:
    """Walk from last value toward fair value, ``theta`` of the gap each step."""
    ppy = series.periods_per_year
    horizon = series.horizon
    prev = series.last_value
    path: list[float] = []
    for step in range(1, horizon + 1):
        fair_step = trend_value(fit, series.n - 1 + step, ppy)
        prev = prev + theta * (fair_step - prev)
        path.append(float(prev))
    return path


def forecast(series: PriceSeries, confidence: int = 80) -> ForecastResult:
    """Forecast the next ``series.horizon`` periods via conservative reversion."""
    fit = fit_log_linear(series)
    theta = _estimate_theta(_deviations(series, fit))
    std = _residual_std(series, fit, theta)
    z = Z_SCORES[confidence]
    horizon = series.horizon

    path = _iterate_path(series, fit, theta)
    point = path[-1]
    half = z * std * float(np.sqrt(horizon))
    lower = max(point - half, point * 1e-3)

    rationale = (
        f"均值回归（保守）：以对数线性趋势为长期公允价值，"
        f"按回归速度 theta={theta:.2f} 从最新价向公允价收敛，外推 {horizon} 步。"
    )
    params = {
        "theta": theta,
        "fair_value_end": trend_value(fit, series.n - 1 + horizon, series.periods_per_year),
        "residual_std": std,
        "last_value": series.last_value,
    }
    return ForecastResult(
        model_name=MODEL_MEAN_REVERSION,
        point=point,
        lower=lower,
        upper=point + half,
        path=path,
        path_labels=future_labels(series, horizon),
        rationale=rationale,
        params=params,
        confidence=confidence,
    )
