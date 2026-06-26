"""Infer the observation frequency of a price series from its date spacing."""

from __future__ import annotations

import statistics
from datetime import datetime

#: Canonical day-spacing for each supported frequency. The inferred frequency
#: is the one whose reference spacing is nearest the observed median spacing.
_REFERENCE_DAYS: dict[str, float] = {
    "annual": 365.0,
    "quarterly": 91.0,
    "monthly": 30.0,
}


def infer_frequency(dates: list[datetime]) -> str:
    """Infer ``"annual" | "quarterly" | "monthly"`` from median day-spacing.

    The median gap (in days) between consecutive, chronologically sorted dates
    is compared against reference spacings (~365 / ~91 / ~30) and the nearest
    match is returned.

    Args:
        dates: Observation timestamps (order does not matter).

    Returns:
        The inferred frequency key.

    Raises:
        ValueError: If fewer than two dates are supplied (no spacing exists).
    """
    if len(dates) < 2:
        raise ValueError(
            "Cannot infer frequency from fewer than two dates; "
            "at least two observations are required to measure spacing."
        )

    ordered = sorted(dates)
    gaps = [
        (ordered[i + 1] - ordered[i]).days for i in range(len(ordered) - 1)
    ]
    median_gap = statistics.median(gaps)

    return min(
        _REFERENCE_DAYS,
        key=lambda freq: abs(_REFERENCE_DAYS[freq] - median_gap),
    )
