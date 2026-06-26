"""Tests for the bilingual i18n layer."""

from __future__ import annotations

from src.core.types import MODEL_EXP_SMOOTHING
from src.data.io import read_csv
from src.data.validation import build_series
from src.eda.insights import compute_insights
from src.features.external_factors import external_factor_analysis
from src.models import exponential_smoothing
from src.ui import i18n


def _series():
    return build_series(read_csv("sample_data/seattle.csv"), city="Seattle, WA")


def test_t_formats_and_localizes():
    assert i18n.t("en", "card_forecast") == "+1yr forecast"
    assert i18n.t("zh", "card_forecast") == "+1 年预测"
    assert i18n.t("en", "card_interval", conf=80, range="$1 – $2") == "80% interval: $1 – $2"


def test_t_unknown_lang_falls_back_to_english():
    assert i18n.t("fr", "card_forecast") == "+1yr forecast"


def test_model_label_switches_language():
    assert i18n.model_label("en", MODEL_EXP_SMOOTHING) == "Exp. Smoothing (Holt)"
    assert i18n.model_label("zh", MODEL_EXP_SMOOTHING) == MODEL_EXP_SMOOTHING


def test_freq_and_method_and_direction_labels():
    assert i18n.freq_label("en", "monthly") == "monthly"
    assert i18n.freq_label("zh", "monthly") == "月度"
    assert i18n.method_label("en", "expert") == "Expert fixed weights"
    assert "Upward" in i18n.direction_badge("en", "up")


def test_english_model_rationale_is_regenerated():
    result = exponential_smoothing.forecast(_series(), 80)
    en = i18n.model_rationale("en", result)
    zh = i18n.model_rationale("zh", result)
    assert "Holt" in en and "α=" in en
    assert en != zh
    assert zh == result.rationale  # zh reuses the model's own text


def test_eda_insight_is_bilingual_and_names_city():
    series = _series()
    ins = compute_insights(series)
    zh = i18n.eda_insight("zh", series, ins)
    en = i18n.eda_insight("en", series, ins)
    assert "Seattle, WA" in en and "CAGR" in en
    assert "复合年增长率" in zh
    assert zh != en


def test_external_factors_english():
    series = _series()
    ins = compute_insights(series)
    factors = external_factor_analysis(series, ins, lang="en")
    assert factors[0].name == "Mortgage Rates"
    assert "Seattle, WA" in factors[1].name
