"""Data-aware external-factor narrative (Section 3 of the analysis).

Produces 2-3 macro variables that, if present, would materially shift the
forecast — and the direction each would push it. The narrative is templated
(no LLM) but tilts its emphasis based on the series' recent momentum.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.types import EdaInsights, PriceSeries


@dataclass(frozen=True)
class ExternalFactor:
    """One macro variable and its likely effect on the forecast."""

    name: str
    direction: str  # "down" | "up" | "uncertain" — the dominant near-term risk
    detail: str


def _recent_momentum(insights: EdaInsights) -> float:
    """Most recent YoY growth, or CAGR as a fallback signal."""
    if insights.yoy_growth:
        return insights.yoy_growth[-1]
    return insights.cagr


def external_factor_analysis(
    series: PriceSeries, insights: EdaInsights
) -> list[ExternalFactor]:
    """Return the macro variables most likely to revise the forecast."""
    momentum = _recent_momentum(insights)
    cooling = momentum < 0
    city = series.city or "该城市"

    rate_dir = "down" if cooling else "uncertain"
    rates = ExternalFactor(
        name="抵押贷款利率 (Mortgage Rates)",
        direction=rate_dir,
        detail=(
            "利率与房价通常负相关。若 30 年期固定利率继续上行，购房负担能力下降、"
            "需求受抑，预测应向下修正；若进入降息周期，则应上修。"
            + ("近期同比已转弱，利率压力可能是主因，下行风险更值得关注。" if cooling
               else "当前动能仍偏强，利率走向是最大的双向摆动来源。")
        ),
    )

    employment = ExternalFactor(
        name=f"核心产业就业率 ({city} Employment)",
        direction="up" if not cooling else "uncertain",
        detail=(
            f"{city}房价高度依赖本地核心产业（如科技、医疗、制造）的就业与薪资。"
            "净流入的高薪岗位会抬升需求并向上修正预测；大规模裁员或产业外迁则向下修正。"
        ),
    )

    inventory = ExternalFactor(
        name="库存 / 挂牌供给 (Inventory & Months of Supply)",
        direction="down" if cooling else "uncertain",
        detail=(
            "库存（months of supply）上升意味着卖方议价能力减弱，对价格形成下行压力，"
            "预测应向下修正；库存持续低于 3 个月的卖方市场则支撑价格、向上修正。"
        ),
    )

    return [rates, employment, inventory]
