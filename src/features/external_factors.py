"""Data-aware external-factor narrative (Section 3 of the analysis).

Produces 2-3 macro variables that, if present, would materially shift the
forecast — and the direction each would push it. The narrative is templated
(no LLM, bilingual) but tilts its emphasis based on the series' recent momentum.
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


def _factors_zh(city: str, cooling: bool, rate_dir: str, emp_dir: str,
                inv_dir: str) -> list[ExternalFactor]:
    rate_tail = ("近期同比已转弱，利率压力可能是主因，下行风险更值得关注。" if cooling
                 else "当前动能仍偏强，利率走向是最大的双向摆动来源。")
    return [
        ExternalFactor("抵押贷款利率 (Mortgage Rates)", rate_dir,
                       "利率与房价通常负相关。若 30 年期固定利率继续上行，购房负担能力下降、"
                       "需求受抑，预测应向下修正；若进入降息周期，则应上修。" + rate_tail),
        ExternalFactor(f"核心产业就业率 ({city} Employment)", emp_dir,
                       f"{city}房价高度依赖本地核心产业（如科技、医疗、制造）的就业与薪资。"
                       "净流入的高薪岗位会抬升需求并向上修正预测；大规模裁员或产业外迁则向下修正。"),
        ExternalFactor("库存 / 挂牌供给 (Inventory & Months of Supply)", inv_dir,
                       "库存（months of supply）上升意味着卖方议价能力减弱，对价格形成下行压力，"
                       "预测应向下修正；库存持续低于 3 个月的卖方市场则支撑价格、向上修正。"),
    ]


def _factors_en(city: str, cooling: bool, rate_dir: str, emp_dir: str,
                inv_dir: str) -> list[ExternalFactor]:
    rate_tail = (" Recent YoY has weakened, so rate pressure may be the main driver — "
                 "downside risk deserves attention." if cooling
                 else " Momentum is still firm; rate direction is the largest two-way swing factor.")
    return [
        ExternalFactor("Mortgage Rates", rate_dir,
                       "Rates are usually inversely related to prices. If the 30-year fixed rate "
                       "keeps rising, affordability falls, demand softens, and the forecast should "
                       "be revised down; an easing cycle revises it up." + rate_tail),
        ExternalFactor(f"{city} Core-Industry Employment", emp_dir,
                       f"{city} home prices depend heavily on local core industries (tech, "
                       "healthcare, manufacturing). Net inflows of high-wage jobs lift demand and "
                       "revise the forecast up; large layoffs or relocation revise it down."),
        ExternalFactor("Inventory & Months of Supply", inv_dir,
                       "Rising months-of-supply means weaker seller pricing power and downward "
                       "pressure, revising the forecast down; a seller's market with supply "
                       "persistently below 3 months supports prices and revises up."),
    ]


def external_factor_analysis(
    series: PriceSeries, insights: EdaInsights, lang: str = "zh"
) -> list[ExternalFactor]:
    """Return the macro variables most likely to revise the forecast."""
    cooling = _recent_momentum(insights) < 0
    city = series.city or ("该城市" if lang == "zh" else "this city")
    rate_dir = "down" if cooling else "uncertain"
    emp_dir = "up" if not cooling else "uncertain"
    inv_dir = "down" if cooling else "uncertain"
    builder = _factors_en if lang == "en" else _factors_zh
    return builder(city, cooling, rate_dir, emp_dir, inv_dir)
