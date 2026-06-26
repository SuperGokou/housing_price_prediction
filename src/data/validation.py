"""Validate and normalize a raw CSV DataFrame into a :class:`PriceSeries`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.periods import format_label
from src.core.types import PriceSeries
from src.data.frequency import infer_frequency

#: Minimum number of valid rows required to build a usable series.
_MIN_ROWS = 3


def _clean_prices(raw: pd.Series) -> pd.Series:
    """Strip ``$``, commas, and whitespace, then coerce to numeric floats.

    Unparseable values become ``NaN`` so the caller can drop them.
    """
    cleaned = (
        raw.astype(str)
        .str.replace(r"[$,\s]", "", regex=True)
        .replace({"": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_series(df: pd.DataFrame, city: str = "") -> PriceSeries:
    """Validate a raw ``date``/``price`` DataFrame into a ``PriceSeries``.

    Normalization steps:
        - Parse dates; strip ``$``/commas/whitespace and coerce prices to float.
        - Drop rows with unparseable dates or prices.
        - Reject prices that are not strictly positive.
        - Sort ascending by date; drop duplicate dates (keep the last).
        - Infer frequency and build human-readable period labels.

    Args:
        df: A DataFrame with ``date`` and ``price`` columns (see ``read_csv``).
        city: Optional display name for the series.

    Returns:
        A validated, chronologically sorted :class:`PriceSeries`.

    Raises:
        ValueError: If, after cleaning, fewer than three valid rows remain or
            any surviving price is non-positive.
    """
    if "date" not in df.columns or "price" not in df.columns:
        raise ValueError("Input must have 'date' and 'price' columns.")

    work = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce", format="mixed"),
            "price": _clean_prices(df["price"]),
        }
    )

    # Drop rows that failed date or price parsing.
    work = work.dropna(subset=["date", "price"])
    if work.empty:
        raise ValueError(
            "No valid rows found: every row had an unparseable date or price."
        )

    # Strictly-positive prices only; flag rather than silently drop.
    if (work["price"] <= 0).any():
        raise ValueError(
            "All prices must be strictly positive (> 0); "
            "found zero or negative values."
        )

    # Sort ascending and drop exact-duplicate dates, keeping the last row.
    work = work.sort_values("date").drop_duplicates(subset="date", keep="last")

    if len(work) < _MIN_ROWS:
        raise ValueError(
            f"At least {_MIN_ROWS} valid observations are required; "
            f"only {len(work)} remained after cleaning."
        )

    dates = list(work["date"].dt.to_pydatetime())
    values = work["price"].to_numpy(dtype=np.float64)
    frequency = infer_frequency(dates)
    labels = [format_label(d, frequency) for d in dates]

    return PriceSeries(
        dates=dates,
        values=values,
        frequency=frequency,
        labels=labels,
        city=city,
    )
