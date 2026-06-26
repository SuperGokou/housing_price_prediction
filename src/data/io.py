"""Read a raw housing-price CSV from disk, bytes, or a file-like object.

The output is a 2-column DataFrame with columns named exactly ``date`` and
``price``. Values are returned raw and unparsed; cleaning happens later in the
validation layer.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

#: Substrings (case-insensitive) that mark a column as date-like / price-like.
_DATE_KEYWORDS = ("date", "period", "month", "year")
_PRICE_KEYWORDS = ("price", "value", "median", "cost", "amount")


def _detect_column(columns: list[str], keywords: tuple[str, ...]) -> str | None:
    """Return the first column whose lowercased name contains any keyword."""
    for col in columns:
        lowered = str(col).lower()
        if any(keyword in lowered for keyword in keywords):
            return col
    return None


def _to_buffer(file: Any) -> Any:
    """Normalize bytes input to a text buffer; pass other inputs through.

    Path strings and file-like objects (including Streamlit ``UploadedFile``)
    are handed straight to :func:`pandas.read_csv`.
    """
    if isinstance(file, bytes):
        return io.BytesIO(file)
    return file


def read_csv(file: Any) -> pd.DataFrame:
    """Read a CSV and return a raw 2-column ``date``/``price`` DataFrame.

    Args:
        file: A path string, raw ``bytes``, or a file-like object such as a
            Streamlit ``UploadedFile``.

    Returns:
        A DataFrame with exactly two columns named ``date`` and ``price``,
        preserving the original (unparsed) cell values as strings.

    Raises:
        ValueError: If the CSV has fewer than two columns.
    """
    raw = pd.read_csv(_to_buffer(file), dtype=str)
    columns = list(raw.columns)
    if len(columns) < 2:
        raise ValueError(
            "CSV must contain at least two columns (a date column and a "
            f"price column); found {len(columns)}."
        )

    date_col = _detect_column(columns, _DATE_KEYWORDS)
    price_col = _detect_column(columns, _PRICE_KEYWORDS)

    # Fall back to positional columns when names are ambiguous or collide.
    if date_col is None or price_col is None or date_col == price_col:
        date_col, price_col = columns[0], columns[1]

    result = raw[[date_col, price_col]].copy()
    result.columns = ["date", "price"]
    return result
