"""Tests for the data-ingestion layer: frequency, io, validation.

Written test-first (TDD). All DataFrames are built in-memory except the two
integration checks that read the bundled sample CSVs.
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.core.types import PriceSeries
from src.data.frequency import infer_frequency
from src.data.io import read_csv
from src.data.validation import build_series


# --- frequency.infer_frequency ----------------------------------------------


def test_infer_frequency_detects_annual_from_yearly_spacing():
    # Arrange
    dates = [datetime(2019, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)]

    # Act
    result = infer_frequency(dates)

    # Assert
    assert result == "annual"


def test_infer_frequency_detects_quarterly_from_three_month_spacing():
    # Arrange
    dates = [
        datetime(2021, 1, 1),
        datetime(2021, 4, 1),
        datetime(2021, 7, 1),
        datetime(2021, 10, 1),
    ]

    # Act
    result = infer_frequency(dates)

    # Assert
    assert result == "quarterly"


def test_infer_frequency_detects_monthly_from_one_month_spacing():
    # Arrange
    dates = [
        datetime(2021, 1, 1),
        datetime(2021, 2, 1),
        datetime(2021, 3, 1),
        datetime(2021, 4, 1),
    ]

    # Act
    result = infer_frequency(dates)

    # Assert
    assert result == "monthly"


def test_infer_frequency_handles_unsorted_dates_via_median_spacing():
    # Arrange — same yearly cadence but shuffled
    dates = [datetime(2021, 1, 1), datetime(2019, 1, 1), datetime(2020, 1, 1)]

    # Act
    result = infer_frequency(dates)

    # Assert
    assert result == "annual"


def test_infer_frequency_raises_on_single_element():
    # Arrange
    dates = [datetime(2021, 1, 1)]

    # Act / Assert
    with pytest.raises(ValueError):
        infer_frequency(dates)


def test_infer_frequency_raises_on_empty():
    # Act / Assert
    with pytest.raises(ValueError):
        infer_frequency([])


# --- io.read_csv ------------------------------------------------------------


def test_read_csv_detects_named_date_and_price_columns():
    # Arrange
    raw = "Period,Median Price\n2019,500000\n2020,520000\n"
    buf = io.StringIO(raw)

    # Act
    df = read_csv(buf)

    # Assert
    assert list(df.columns) == ["date", "price"]
    assert df["date"].tolist() == ["2019", "2020"]
    assert df["price"].tolist() == ["500000", "520000"]


def test_read_csv_detects_columns_regardless_of_position():
    # Arrange — price column comes before the date column
    raw = "home_value,observation_month\n500000,2019-01\n520000,2019-02\n"
    buf = io.StringIO(raw)

    # Act
    df = read_csv(buf)

    # Assert
    assert list(df.columns) == ["date", "price"]
    assert df["date"].tolist() == ["2019-01", "2019-02"]
    assert df["price"].tolist() == ["500000", "520000"]


def test_read_csv_falls_back_to_positional_columns_when_ambiguous():
    # Arrange — neither column name matches a known keyword
    raw = "col_a,col_b\n2019,500000\n2020,520000\n"
    buf = io.StringIO(raw)

    # Act
    df = read_csv(buf)

    # Assert
    assert list(df.columns) == ["date", "price"]
    assert df["date"].tolist() == ["2019", "2020"]
    assert df["price"].tolist() == ["500000", "520000"]


def test_read_csv_accepts_bytes_input():
    # Arrange
    raw = b"date,price\n2019,500000\n2020,520000\n"

    # Act
    df = read_csv(raw)

    # Assert
    assert list(df.columns) == ["date", "price"]
    assert df["date"].tolist() == ["2019", "2020"]


def test_read_csv_preserves_raw_unparsed_values():
    # Arrange — prices carry $ and (quoted) commas; survive read_csv untouched
    raw = 'date,price\n2019,"$500,000"\n2020,"$520,000"\n'
    buf = io.StringIO(raw)

    # Act
    df = read_csv(buf)

    # Assert
    assert df["price"].tolist() == ["$500,000", "$520,000"]


def test_read_csv_reads_from_path_string(tmp_path):
    # Arrange
    p = tmp_path / "prices.csv"
    p.write_text("date,price\n2019,500000\n2020,520000\n", encoding="utf-8")

    # Act
    df = read_csv(str(p))

    # Assert
    assert list(df.columns) == ["date", "price"]
    assert df["date"].tolist() == ["2019", "2020"]


# --- validation.build_series ------------------------------------------------


def _df(dates: list[str], prices: list) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "price": prices})


def test_build_series_happy_path_returns_price_series():
    # Arrange
    df = _df(["2019-01-01", "2020-01-01", "2021-01-01"], [600000, 650000, 720000])

    # Act
    series = build_series(df, city="Seattle, WA")

    # Assert
    assert isinstance(series, PriceSeries)
    assert series.city == "Seattle, WA"
    assert series.frequency == "annual"
    assert series.n == 3
    assert series.labels == ["2019", "2020", "2021"]
    assert series.values.dtype == np.float64
    np.testing.assert_array_equal(series.values, np.array([600000.0, 650000.0, 720000.0]))


def test_build_series_rejects_fewer_than_three_rows():
    # Arrange
    df = _df(["2019-01-01", "2020-01-01"], [600000, 650000])

    # Act / Assert
    with pytest.raises(ValueError):
        build_series(df)


def test_build_series_rejects_non_positive_prices():
    # Arrange — a zero and a negative price; only one valid row remains -> < 3 rows
    df = _df(["2019-01-01", "2020-01-01", "2021-01-01"], [600000, 0, -5])

    # Act / Assert
    with pytest.raises(ValueError):
        build_series(df)


def test_build_series_strips_dollar_signs_and_commas_from_prices():
    # Arrange
    df = _df(
        ["2019-01-01", "2020-01-01", "2021-01-01"],
        ["$600,000", "$650,000 ", " 720000 "],
    )

    # Act
    series = build_series(df)

    # Assert
    np.testing.assert_array_equal(
        series.values, np.array([600000.0, 650000.0, 720000.0])
    )


def test_build_series_sorts_unsorted_input_ascending_by_date():
    # Arrange — deliberately out of order
    df = _df(
        ["2021-01-01", "2019-01-01", "2020-01-01"],
        [720000, 600000, 650000],
    )

    # Act
    series = build_series(df)

    # Assert
    assert series.labels == ["2019", "2020", "2021"]
    np.testing.assert_array_equal(
        series.values, np.array([600000.0, 650000.0, 720000.0])
    )


def test_build_series_dedupes_duplicate_dates_keeping_last():
    # Arrange — 2020 appears twice; the later row (655000) should win
    df = _df(
        ["2019-01-01", "2020-01-01", "2020-01-01", "2021-01-01"],
        [600000, 650000, 655000, 720000],
    )

    # Act
    series = build_series(df)

    # Assert
    assert series.n == 3
    assert series.labels == ["2019", "2020", "2021"]
    np.testing.assert_array_equal(
        series.values, np.array([600000.0, 655000.0, 720000.0])
    )


def test_build_series_drops_rows_with_unparseable_dates():
    # Arrange — one junk date; three good rows remain
    df = _df(
        ["2019-01-01", "not-a-date", "2020-01-01", "2021-01-01"],
        [600000, 640000, 650000, 720000],
    )

    # Act
    series = build_series(df)

    # Assert
    assert series.n == 3
    assert series.labels == ["2019", "2020", "2021"]


def test_build_series_drops_rows_with_non_numeric_prices():
    # Arrange — one non-numeric price; three good rows remain
    df = _df(
        ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"],
        [600000, "garbage", 650000, 720000],
    )

    # Act
    series = build_series(df)

    # Assert
    assert series.n == 3
    np.testing.assert_array_equal(
        series.values, np.array([600000.0, 650000.0, 720000.0])
    )


def test_build_series_raises_value_error_on_all_garbage():
    # Arrange — nothing is salvageable
    df = _df(["x", "y", "z"], ["a", "b", "c"])

    # Act / Assert
    with pytest.raises(ValueError):
        build_series(df)


def test_build_series_defaults_city_to_empty_string():
    # Arrange
    df = _df(["2019-01-01", "2020-01-01", "2021-01-01"], [600000, 650000, 720000])

    # Act
    series = build_series(df)

    # Assert
    assert series.city == ""


# --- integration checks against bundled sample data -------------------------


def test_read_and_build_seattle_monthly_sample():
    # Arrange / Act
    df = read_csv("sample_data/seattle.csv")
    series = build_series(df, city="Seattle")

    # Assert
    assert series.frequency == "monthly"
    assert series.n == 72
    assert series.values.dtype == np.float64
    assert series.last_value > 0


def test_read_and_build_seattle_annual_sample():
    # Arrange / Act
    df = read_csv("sample_data/seattle_annual.csv")
    series = build_series(df)

    # Assert
    assert series.frequency == "annual"
    assert series.n == 5
    assert series.labels[0] == "2019"
