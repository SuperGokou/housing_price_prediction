# Housing Price Multi-Model Prediction — Design Spec

Date: 2026-06-26

## 1. Goal

An interactive Streamlit web app where a user uploads a US city's recent housing
price history (CSV) and receives a four-section analysis matching the original
prompt:

1. **EDA** — CAGR, volatility, drawdown, inflection points.
2. **Three independent model forecasts** — each with a point estimate, interval,
   forecast path, and a transparent rationale.
3. **External-factor narrative** — 2-3 variables (mortgage rates, employment,
   inventory) and the direction each would push the forecast.
4. **Ensemble forecast** — weighted combination with a stated final range.

## 2. Key constraint: small samples

The canonical example has only 5 annual points. Real ARIMA/Prophet are not
statistically valid there, so the three "models" are **transparent statistical
methods that embody each model family's logic** — fully reproducible, no heavy
ML dependencies (numpy/pandas only). The UI must prominently disclaim that
small-sample results are illustrative, not investment advice.

## 3. Architecture & file ownership

```
src/core/types.py        # CONTRACT (written): PriceSeries, EdaInsights,
                         #   ForecastResult, EnsembleResult, constants
src/core/periods.py      # CONTRACT (written): label/date helpers
src/data/                # [Agent: data] io.py, validation.py, frequency.py
src/eda/insights.py      # [Agent: eda] compute_insights()
src/models/              # [Agent: models] exponential_smoothing.py,
                         #   growth_curve.py, mean_reversion.py, ensemble.py
src/features/            # [Main] external_factors.py
src/viz/charts.py        # [Main] plotly figures
src/ui/                  # [Main] streamlit sections
app.py                   # [Main] entry point
tests/                   # each agent owns the tests for its module
```

The "next year" headline = the forecasted value at the **+1-year horizon**
(`series.horizon` steps ahead = `path[-1]`). Growth/return values are fractions.

## 4. Function contracts

### Data layer (`src/data`)
- `frequency.py: infer_frequency(dates: list[datetime]) -> str` returns
  `"annual" | "quarterly" | "monthly"` from median spacing (≈365 → annual,
  ≈91 → quarterly, ≈30 → monthly; nearest match).
- `io.py: read_csv(file) -> pandas.DataFrame` accepts a path or file-like /
  bytes (Streamlit `UploadedFile`). Detects a date-ish column and a price-ish
  column case-insensitively (date|period|month|year, price|value|median|cost).
- `validation.py: build_series(df, city="") -> PriceSeries` validates and
  normalizes: ≥3 rows, parseable dates, numeric & strictly-positive prices,
  sort ascending by date, drop exact-duplicate dates (keep last), strip commas
  and `$` from prices. Raise `ValueError` with a clear Chinese-or-English
  message on any violation (caller shows it via `st.error`). Build labels with
  `src.core.periods.format_label`.

### EDA (`src/eda/insights.py`)
- `compute_insights(series: PriceSeries) -> EdaInsights`.
  - `cagr = (last/first) ** (1/span_years) - 1` (span_years from dates;
    guard span_years <= 0).
  - `total_growth = last/first - 1`.
  - `annualized_volatility = std(per-period simple returns, ddof=1) *
    sqrt(periods_per_year)`.
  - `yoy_growth`: year-over-year (compare value to the one `periods_per_year`
    back) for each eligible point; `yoy_labels` aligned.
  - `inflections`: labels where consecutive YoY growth changes sign.
  - peak/trough across the series; `max_drawdown` = most negative
    peak-to-subsequent-trough decline (<= 0).
  - `trend_r2`: R² of OLS fit of `log(value) ~ t`.

### Models (`src/models`) — every model implements
`forecast(series: PriceSeries, confidence: int = 80) -> ForecastResult`
with `horizon = series.horizon`, `path` length == horizon, `path_labels` from
`src.core.periods.future_labels(series, horizon)`, and a populated `rationale`
+ `params`. Interval half-width uses `Z_SCORES[confidence]`.

1. **exponential_smoothing.py** — Holt's linear (double) exponential smoothing.
   Grid-search `alpha, beta in {0.1..0.9}` minimizing one-step-ahead SSE
   (fall back to alpha=0.5, beta=0.3 if n too small). Forecast level + h*trend.
   Interval from in-sample one-step residual std scaled by `sqrt(step)`.
   `model_name = MODEL_EXP_SMOOTHING`.
2. **growth_curve.py** — log-linear growth: OLS fit `log(price) ~ a + b*t`,
   forecast `exp(a + b*(n-1+step))`. If sub-annual (periods_per_year>1) and
   enough data (n >= 2*periods_per_year), add an additive seasonal component
   (mean residual per period-of-year). Apply mild damping so b does not explode
   on tiny samples is optional. Interval from log-residual std (multiplicative).
   `model_name = MODEL_GROWTH_CURVE`.
3. **mean_reversion.py** — conservative reversion toward long-run fair value.
   Fair value = log-linear trend value at each future step. `theta` (reversion
   speed in (0,1]) estimated from how fast past deviations from trend decayed
   (OLS of deviation_t on deviation_{t-1}); default 0.4; clamp to [0.15, 0.85].
   `next = prev + theta * (fair_step - prev)` iterated across the horizon
   starting from `last_value`. Interval from residual std. This model should be
   the most conservative (closest to last value / long-run mean).
   `model_name = MODEL_MEAN_REVERSION`.
4. **ensemble.py** —
   `combine(series, results: list[ForecastResult], confidence=80) -> EnsembleResult`.
   - Default **expert weights**: exp-smoothing 0.40, growth 0.35,
     mean-reversion 0.25 (keyed by model_name).
   - If `n >= 8`, run a one-step-ahead holdout backtest (refit each model on
     `series[:-1]`, error vs last actual) and switch to **inverse-error
     weights** (`w_i ∝ 1/(mae_i + eps)`, normalized); set `weighting_method`
     accordingly and explain in `rationale`.
   - Ensemble point = Σ w_i · point_i. Interval combines within-model and
     between-model variance:
     `sigma = sqrt(Σ w_i * ((upper_i-lower_i)/(2*z))**2 + Σ w_i*(point_i-point)**2)`,
     bounds = point ± z*sigma. weights dict sums to 1.0.

## 5. UI / app

Three-section layout: sidebar (city name, CSV upload + sample loader, frequency
override, confidence selector, run button), main area with EDA metrics + chart,
per-model cards, ensemble headline + combined band chart, external-factor
narrative, disclaimer. App loads `sample_data/seattle.csv` by default so the
first render is a full demo. Plotly for all charts. Errors surfaced via
`st.error`; never crash on bad input.

## 6. Testing (TDD, target 80%+)

Deterministic pure functions → exact assertions. Each agent writes tests for
its module under `tests/`. Cover: bad/!sorted/duplicate/non-positive CSV;
frequency inference for annual/quarterly/monthly spacing; known-CAGR EDA;
monotone-up series → forecast > last; flat series → forecast ≈ last; ensemble
weights sum to 1 and interval contains the point. Use `pytest`; imports are
`from src.core.types import ...` (pythonpath set in pyproject.toml).
