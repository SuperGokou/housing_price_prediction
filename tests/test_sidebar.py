"""Tests for sidebar input sanitization (security: H1)."""

from __future__ import annotations

from src.ui.sidebar import _sanitize_city


def test_plain_city_is_unchanged():
    assert _sanitize_city("Seattle, WA") == "Seattle, WA"


def test_unicode_city_is_preserved():
    assert _sanitize_city("北京") == "北京"


def test_html_and_markdown_injection_is_stripped():
    assert _sanitize_city("<script>alert(1)</script>") == "scriptalert1script"
    assert "[" not in _sanitize_city("[x](javascript:alert(1))")
    assert "(" not in _sanitize_city("[x](javascript:alert(1))")


def test_overlong_input_is_truncated():
    assert len(_sanitize_city("A" * 200)) == 60
