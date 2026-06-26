"""Section 4: the weighted ensemble forecast and its rationale."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.core.types import EnsembleResult, PriceSeries
from src.ui.format import fmt_pct, fmt_range, fmt_usd
from src.ui.i18n import ensemble_rationale, method_label, model_label, t


def render(series: PriceSeries, ensemble: EnsembleResult, lang: str = "zh") -> None:
    st.subheader(t(lang, "ensemble_title"))
    change = ensemble.point / series.last_value - 1
    city = series.city or t(lang, "the_city")

    left, right = st.columns([1, 1])
    with left:
        st.metric(t(lang, "final_forecast", city=city), fmt_usd(ensemble.point),
                  delta=fmt_pct(change, signed=True))
        st.markdown(f"**{t(lang, 'ens_interval_label', conf=ensemble.confidence)}**"
                    f"{fmt_range(ensemble.lower, ensemble.upper)}")
        method = method_label(lang, ensemble.weighting_method)
        st.caption(t(lang, "weight_method_label", method=method))
        st.write(ensemble_rationale(lang, ensemble))

    with right:
        weights_df = pd.DataFrame([
            {t(lang, "col_model"): model_label(lang, name), t(lang, "col_weight"): f"{w:.0%}"}
            for name, w in ensemble.weights.items()
        ])
        st.markdown(f"**{t(lang, 'weights_heading')}**")
        st.dataframe(weights_df, hide_index=True, use_container_width=True)
