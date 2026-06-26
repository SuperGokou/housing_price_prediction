"""Bilingual (中文 / English) UI strings and narrative generators.

The forecasting/EDA layers stay language-agnostic. For Chinese we reuse the
text those layers already produce (``result.rationale`` etc.); for English we
regenerate from the model ``params``. All user-facing UI strings live in
``TR`` and are looked up via :func:`t`.
"""

from __future__ import annotations

import numpy as np

from src.core.types import (
    MODEL_EXP_SMOOTHING,
    MODEL_GROWTH_CURVE,
    MODEL_MEAN_REVERSION,
    EdaInsights,
    EnsembleResult,
    ForecastResult,
    PriceSeries,
)
from src.ui.format import fmt_pct, fmt_usd

APP_ICON = "📈"

#: Sidebar language picker: display label -> internal code.
LANG_OPTIONS: dict[str, str] = {"中文": "zh", "English": "en"}

#: Localized display labels for each canonical model name.
MODEL_LABELS: dict[str, dict[str, str]] = {
    MODEL_EXP_SMOOTHING: {"zh": MODEL_EXP_SMOOTHING, "en": "Exp. Smoothing (Holt)"},
    MODEL_GROWTH_CURVE: {"zh": MODEL_GROWTH_CURVE, "en": "Growth Curve (Log-Linear)"},
    MODEL_MEAN_REVERSION: {"zh": MODEL_MEAN_REVERSION, "en": "Mean Reversion (Conservative)"},
}

_FREQ_LABELS = {
    "zh": {"annual": "年度", "quarterly": "季度", "monthly": "月度"},
    "en": {"annual": "annual", "quarterly": "quarterly", "monthly": "monthly"},
}

_METHOD_LABELS = {
    "zh": {"expert": "专家固定权重", "backtest-inverse-error": "回测逆误差自动权重"},
    "en": {"expert": "Expert fixed weights", "backtest-inverse-error": "Backtest inverse-error weights"},
}

DIRECTION_BADGE = {
    "up": {"zh": "⬆️ 上修", "en": "⬆️ Upward"},
    "down": {"zh": "⬇️ 下修", "en": "⬇️ Downward"},
    "uncertain": {"zh": "↕️ 双向", "en": "↕️ Two-sided"},
}

TR: dict[str, dict[str, str]] = {
    "zh": {
        "page_title": "房价多模型预测",
        "app_title": "房价多模型预测",
        "subtitle": "上传某美国城市过去几年的房价历史，应用以**三种不同逻辑的透明统计模型**推演下一年走势，并给出加权**集成预测区间**。",
        "sidebar_title": "房价多模型预测",
        "sidebar_subtitle": "US Housing Price Multi-Model Forecaster",
        "lang_label": "语言 / Language",
        "city_label": "城市名称",
        "city_help": "仅用于图表与文案展示",
        "upload_label": "上传房价 CSV",
        "upload_help": "需包含日期列与价格列（年/季/月均可）。留空则使用内置 Seattle 示例。",
        "csv_hint": "CSV 示例：`date,price` — 例如 `2023-12-31,760000`",
        "confidence_label": "预测区间置信度",
        "sidebar_disclaimer": "⚠️ 小样本统计推演，仅供研究演示，非投资建议。",
        "src_uploaded": "上传文件",
        "src_sample": "内置示例 (Seattle, 月度)",
        "err_parse": "无法解析上传的 CSV：{detail}",
        "err_parse_generic": "无法解析上传的 CSV：文件格式无效，请确认是标准 CSV 文件。",
        "err_validate": "数据校验失败：{detail}",
        "err_no_data": "请在左侧上传 CSV，或使用内置示例数据。",
        "data_source": "数据源：{source} · 频率：{freq} · {n} 个观测点 · 预测步长：{h} 期（=1 年）",
        "eda_title": "1 · 数据趋势初步洞察 (EDA)",
        "m_cagr": "CAGR 复合年增长",
        "m_total_growth": "累计涨幅",
        "m_volatility": "年化波动率",
        "m_drawdown": "最大回撤",
        "models_title": "2 · 多模型独立预测",
        "card_forecast": "+1 年预测",
        "card_interval": "{conf}% 区间：{range}",
        "sec3_title": "3 · 潜在外部特征影响 (Feature Engineering)",
        "ensemble_title": "4 · 综合集成预测 (Ensemble)",
        "final_forecast": "最终预测（下一年，{city}）",
        "ens_interval_label": "{conf}% 预测区间：",
        "weight_method_label": "权重方法：{method}",
        "weights_heading": "模型权重分配",
        "col_model": "模型",
        "col_weight": "权重",
        "disclaimer": "免责声明：本工具使用轻量统计方法对小样本进行说明性推演，不构成任何投资、购房或财务建议。真实预测需纳入利率、就业、库存等外部变量与更高频数据。",
        "chart_main_title": "{city} — 历史走势与下一年多模型预测",
        "chart_price_axis": "房价 (USD)",
        "trace_spread": "模型分歧区间",
        "trace_history": "历史房价",
        "trace_ensemble": "集成预测",
        "trace_interval": "{conf}% 区间",
        "chart_cmp_title": "各模型点估计与区间对比",
        "chart_cmp_axis": "+1 年预测房价 (USD)",
        "label_ensemble": "集成预测",
        "chart_yoy_title": "同比增长率 (YoY)",
        "chart_yoy_axis": "增长率 (%)",
        "default_city": "房价",
        "the_city": "该城市",
    },
    "en": {
        "page_title": "Housing Price Forecaster",
        "app_title": "Multi-Model Housing Forecaster",
        "subtitle": "Upload a US city's recent home-price history and project next year with **three transparent statistical models**, plus a weighted **ensemble forecast interval**.",
        "sidebar_title": "Housing Forecaster",
        "sidebar_subtitle": "US Housing Price Multi-Model Forecaster",
        "lang_label": "语言 / Language",
        "city_label": "City name",
        "city_help": "Used for chart and text display only",
        "upload_label": "Upload price CSV",
        "upload_help": "Needs a date column and a price column (annual/quarterly/monthly). Leave empty to use the built-in Seattle sample.",
        "csv_hint": "CSV example: `date,price` — e.g. `2023-12-31,760000`",
        "confidence_label": "Prediction interval confidence",
        "sidebar_disclaimer": "⚠️ Small-sample statistical projection — for research/demo only, not investment advice.",
        "src_uploaded": "Uploaded file",
        "src_sample": "Built-in sample (Seattle, monthly)",
        "err_parse": "Could not parse the uploaded CSV: {detail}",
        "err_parse_generic": "Could not parse the uploaded CSV: invalid format, please ensure it is a standard CSV file.",
        "err_validate": "Data validation failed: {detail}",
        "err_no_data": "Upload a CSV on the left, or use the built-in sample data.",
        "data_source": "Source: {source} · Frequency: {freq} · {n} observations · Horizon: {h} steps (= 1 year)",
        "eda_title": "1 · Exploratory Data Analysis (EDA)",
        "m_cagr": "CAGR (compound annual)",
        "m_total_growth": "Total growth",
        "m_volatility": "Annualized volatility",
        "m_drawdown": "Max drawdown",
        "models_title": "2 · Independent Model Forecasts",
        "card_forecast": "+1yr forecast",
        "card_interval": "{conf}% interval: {range}",
        "sec3_title": "3 · External Feature Impact (Feature Engineering)",
        "ensemble_title": "4 · Ensemble Forecast",
        "final_forecast": "Final forecast (next year, {city})",
        "ens_interval_label": "{conf}% prediction interval: ",
        "weight_method_label": "Weighting method: {method}",
        "weights_heading": "Model weight allocation",
        "col_model": "Model",
        "col_weight": "Weight",
        "disclaimer": "Disclaimer: this tool uses lightweight statistics for illustrative small-sample projection and does not constitute investment, real-estate, or financial advice. Real forecasts require external variables (rates, employment, inventory) and higher-frequency data.",
        "chart_main_title": "{city} — History & Next-Year Multi-Model Forecast",
        "chart_price_axis": "Home price (USD)",
        "trace_spread": "Model disagreement band",
        "trace_history": "Price history",
        "trace_ensemble": "Ensemble",
        "trace_interval": "{conf}% interval",
        "chart_cmp_title": "Per-model point estimates & intervals",
        "chart_cmp_axis": "+1yr forecast price (USD)",
        "label_ensemble": "Ensemble",
        "chart_yoy_title": "Year-over-Year Growth (YoY)",
        "chart_yoy_axis": "Growth (%)",
        "default_city": "Home price",
        "the_city": "this city",
    },
}


def t(lang: str, key: str, **kwargs: object) -> str:
    """Look up a localized string, formatting in any ``kwargs``."""
    template = TR.get(lang, TR["en"]).get(key, key)
    return template.format(**kwargs) if kwargs else template


def model_label(lang: str, model_name: str) -> str:
    """Localized display name for a canonical model name."""
    return MODEL_LABELS.get(model_name, {}).get(lang, model_name)


def freq_label(lang: str, frequency: str) -> str:
    return _FREQ_LABELS.get(lang, _FREQ_LABELS["en"]).get(frequency, frequency)


def method_label(lang: str, method: str) -> str:
    return _METHOD_LABELS.get(lang, _METHOD_LABELS["en"]).get(method, method)


def direction_badge(lang: str, direction: str) -> str:
    return DIRECTION_BADGE.get(direction, DIRECTION_BADGE["uncertain"]).get(lang, direction)


def model_rationale(lang: str, result: ForecastResult) -> str:
    """Chinese reuses the model's own rationale; English is regenerated."""
    if lang == "zh":
        return result.rationale
    p = result.params
    h = len(result.path)
    name = result.model_name
    if name == MODEL_EXP_SMOOTHING:
        return (f"Holt's linear exponential smoothing: grid-searched α={p['alpha']}, "
                f"β={p['beta']}; extrapolated {h} steps as level + h·trend "
                f"(trend={p['trend']:.1f}).")
    if name == MODEL_GROWTH_CURVE:
        text = (f"Log-linear growth: OLS log(price)=a+b·t, slope b={p['b']:.4f} "
                f"(~{np.expm1(p['b']) * 100:.2f}% per period), extrapolated {h} steps.")
        if len(p.get("seasonal_terms", [])) > 1:
            text += " Seasonal term included."
        return text
    if name == MODEL_MEAN_REVERSION:
        return (f"Mean reversion (conservative): reverts from the latest price toward the "
                f"log-linear fair value at speed θ={p['theta']:.2f}, over {h} steps.")
    return result.rationale


def ensemble_rationale(lang: str, ensemble: EnsembleResult) -> str:
    if lang == "zh":
        return ensemble.rationale
    if ensemble.weighting_method == "expert":
        return ("Small sample (n<8): expert prior weights — Exp. Smoothing 0.40 / "
                "Growth 0.35 / Mean Reversion 0.25.")
    return ("Sufficient sample (n≥8): inverse-error weights from a rolling "
            "one-step-ahead backtest (weight ∝ 1/MAE).")


def eda_insight(lang: str, series: PriceSeries, ins: EdaInsights) -> str:
    """The auto-written EDA paragraph, in the chosen language."""
    city = series.city or t(lang, "the_city")
    if lang == "zh":
        parts = [
            f"过去约 **{ins.span_years:.1f} 年**，{city}房价从 {fmt_usd(series.first_value)} "
            f"变化到 {fmt_usd(ins.latest_price)}，累计 **{fmt_pct(ins.total_growth, signed=True)}**，"
            f"复合年增长率 (CAGR) **{fmt_pct(ins.cagr, signed=True)}**。",
            f"期间峰值 {fmt_usd(ins.peak_price)}（{ins.peak_label}），"
            f"谷值 {fmt_usd(ins.trough_price)}（{ins.trough_label}），"
            f"最大回撤 **{fmt_pct(ins.max_drawdown)}**；年化波动率 {fmt_pct(ins.annualized_volatility)}，"
            f"对数线性趋势拟合 R² = {ins.trend_r2:.2f}。",
        ]
        parts.append(
            f"检测到趋势拐点：**{', '.join(ins.inflections)}**（同比增速变向）。"
            if ins.inflections else "未检测到明显的同比拐点，趋势相对单向。"
        )
        return " ".join(parts)
    parts = [
        f"Over ~**{ins.span_years:.1f} years**, {city} home prices moved from "
        f"{fmt_usd(series.first_value)} to {fmt_usd(ins.latest_price)}, a cumulative "
        f"**{fmt_pct(ins.total_growth, signed=True)}** (CAGR **{fmt_pct(ins.cagr, signed=True)}**).",
        f"Peak {fmt_usd(ins.peak_price)} ({ins.peak_label}), trough {fmt_usd(ins.trough_price)} "
        f"({ins.trough_label}), max drawdown **{fmt_pct(ins.max_drawdown)}**; annualized "
        f"volatility {fmt_pct(ins.annualized_volatility)}, log-linear trend R² = {ins.trend_r2:.2f}.",
    ]
    parts.append(
        f"Detected trend inflections: **{', '.join(ins.inflections)}** (YoY growth changed direction)."
        if ins.inflections else "No clear YoY inflection; the trend is fairly one-directional."
    )
    return " ".join(parts)
