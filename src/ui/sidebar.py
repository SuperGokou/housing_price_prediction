"""Sidebar: city, data source (CSV upload or bundled sample), and settings."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.data.io import read_csv

# Strip anything that isn't a letter (any language), digit, space, or basic
# punctuation. Blocks HTML/Markdown-link injection since the city string is
# rendered into st.markdown() and Plotly titles downstream.
_CITY_DISALLOWED = re.compile(r"[^\w\s,.\-]", re.UNICODE)


def _sanitize_city(raw: str) -> str:
    """Make a user-typed city safe to interpolate into markdown/titles."""
    return _CITY_DISALLOWED.sub("", raw).strip()[:60]


@dataclass
class SidebarState:
    city: str
    df: pd.DataFrame | None
    confidence: int
    source: str
    error: str | None


def render_sidebar(sample_path: str) -> SidebarState:
    """Render sidebar controls and return the chosen configuration."""
    st.sidebar.markdown("## 🏠 房价多模型预测")
    st.sidebar.caption("US Housing Price Multi-Model Forecaster")
    st.sidebar.divider()

    city_raw = st.sidebar.text_input("城市名称", value="Seattle, WA",
                                     help="仅用于图表与文案展示")
    city = _sanitize_city(city_raw)

    uploaded = st.sidebar.file_uploader(
        "上传房价 CSV", type=["csv"],
        help="需包含日期列与价格列（年/季/月均可）。留空则使用内置 Seattle 示例。")

    st.sidebar.caption("CSV 示例：`date,price` — 例如 `2023-12-31,760000`")

    confidence = st.sidebar.selectbox("预测区间置信度", options=[80, 90, 95], index=0,
                                      format_func=lambda c: f"{c}%")

    df: pd.DataFrame | None = None
    error: str | None = None
    if uploaded is not None:
        source = "上传文件"
        try:
            df = read_csv(uploaded)
        except ValueError as exc:  # our own controlled, user-facing messages
            error = f"无法解析上传的 CSV：{exc}"
        except Exception:  # don't leak pandas/path internals to the UI
            logging.exception("Unexpected error parsing uploaded CSV")
            error = "无法解析上传的 CSV：文件格式无效，请确认是标准 CSV 文件。"
    else:
        df = read_csv(sample_path)
        source = "内置示例 (Seattle, 月度)"

    st.sidebar.divider()
    st.sidebar.caption("⚠️ 小样本统计推演，仅供研究演示，非投资建议。")
    return SidebarState(city=city, df=df, confidence=confidence,
                        source=source, error=error)
