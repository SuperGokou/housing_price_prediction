"""Tests for display-formatting helpers."""

from __future__ import annotations

from src.ui.format import fmt_pct, fmt_range, fmt_usd


def test_fmt_usd_adds_thousands_separators():
    assert fmt_usd(788000) == "$788,000"


def test_fmt_pct_signed_positive_has_plus():
    assert fmt_pct(0.123, signed=True) == "+12.3%"


def test_fmt_pct_negative_keeps_minus():
    assert fmt_pct(-0.094) == "-9.4%"


def test_fmt_range_joins_bounds():
    assert fmt_range(716913, 841344) == "$716,913 – $841,344"
