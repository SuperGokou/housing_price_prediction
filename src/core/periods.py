"""Calendar/period helpers shared by the data and model layers.

Kept dependency-free (no python-dateutil) so the package stays lightweight.
"""

from __future__ import annotations

from datetime import datetime

from src.core.types import PriceSeries


def format_label(d: datetime, frequency: str) -> str:
    """Render a timestamp as a period label appropriate to the frequency."""
    if frequency == "annual":
        return str(d.year)
    if frequency == "quarterly":
        quarter = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{quarter}"
    return f"{d.year}-{d.month:02d}"


def add_months(d: datetime, months: int) -> datetime:
    """Return ``d`` advanced by ``months`` calendar months (day clamped to 28)."""
    total = (d.year * 12 + (d.month - 1)) + months
    year, month = divmod(total, 12)
    # Clamp day to 28 to stay valid across month lengths; day-of-month is not
    # meaningful for period-level housing data.
    day = min(d.day, 28)
    return datetime(year, month + 1, day)


def advance(d: datetime, frequency: str, steps: int) -> datetime:
    """Advance ``d`` by ``steps`` periods of the given frequency."""
    if frequency == "annual":
        return add_months(d, 12 * steps)
    if frequency == "quarterly":
        return add_months(d, 3 * steps)
    return add_months(d, steps)


def future_dates(series: PriceSeries, horizon: int) -> list[datetime]:
    """The next ``horizon`` observation timestamps after the series end."""
    last = series.dates[-1]
    return [advance(last, series.frequency, i) for i in range(1, horizon + 1)]


def future_labels(series: PriceSeries, horizon: int) -> list[str]:
    """The next ``horizon`` period labels after the series end."""
    return [format_label(d, series.frequency) for d in future_dates(series, horizon)]
