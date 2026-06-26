"""Small display-formatting helpers (pure, UI-agnostic)."""

from __future__ import annotations


def fmt_usd(value: float, decimals: int = 0) -> str:
    """Format a number as USD, e.g. 788000 -> ``$788,000``."""
    return f"${value:,.{decimals}f}"


def fmt_pct(fraction: float, decimals: int = 1, signed: bool = False) -> str:
    """Format a fraction as a percentage, e.g. 0.0495 -> ``+4.9%``."""
    sign = "+" if signed and fraction >= 0 else ""
    return f"{sign}{fraction * 100:.{decimals}f}%"


def fmt_range(lower: float, upper: float) -> str:
    """Format an interval as ``$x – $y``."""
    return f"{fmt_usd(lower)} – {fmt_usd(upper)}"
