"""Tests for DataAgent — validation, edge cases, and context integration."""
import numpy as np
import pandas as pd
import pytest

from agents.data_agent import DataAgent


def _make_agent(df: pd.DataFrame, key: str = "data") -> DataAgent:
    return DataAgent("TEST", "2020-01-01", "2023-01-01", context_key=key, data_override=df)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_normal_data_passes(sample_ohlcv):
    ctx = _make_agent(sample_ohlcv).run({"data": None})
    assert ctx["data"] is not None
    assert len(ctx["data"]) == len(sample_ohlcv)


def test_context_key_respected(sample_ohlcv):
    ctx = _make_agent(sample_ohlcv, key="benchmark_data").run({"benchmark_data": None})
    assert ctx["benchmark_data"] is not None


def test_data_is_copy(sample_ohlcv):
    """DataAgent must store a copy so callers cannot mutate the agent's source."""
    ctx = _make_agent(sample_ohlcv).run({"data": None})
    ctx["data"].iloc[0, 0] = -999
    assert sample_ohlcv.iloc[0, 0] != -999


def test_preserves_datetime_index(sample_ohlcv):
    ctx = _make_agent(sample_ohlcv).run({"data": None})
    assert isinstance(ctx["data"].index, pd.DatetimeIndex)


def test_required_columns_present(sample_ohlcv):
    ctx = _make_agent(sample_ohlcv).run({"data": None})
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(ctx["data"].columns)


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------

def test_empty_dataframe_raises():
    df = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex([]),
    )
    with pytest.raises(AssertionError, match="Insufficient rows"):
        _make_agent(df).run({"data": None})


def test_nan_in_close_raises(nan_ohlcv):
    with pytest.raises(AssertionError, match="NaN"):
        _make_agent(nan_ohlcv).run({"data": None})


def test_negative_close_raises(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[5, df.columns.get_loc("Close")] = -1.0
    with pytest.raises(AssertionError, match="positive"):
        _make_agent(df).run({"data": None})


def test_unsorted_index_raises(sample_ohlcv):
    df = sample_ohlcv.iloc[::-1].copy()
    with pytest.raises(AssertionError, match="sorted"):
        _make_agent(df).run({"data": None})


def test_missing_column_raises(sample_ohlcv):
    df = sample_ohlcv.drop(columns=["Volume"])
    with pytest.raises(AssertionError, match="Missing"):
        _make_agent(df).run({"data": None})


def test_too_few_rows_raises():
    import numpy as np
    dates = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame(
        {
            "Open": np.ones(10),
            "High": np.ones(10) * 1.01,
            "Low": np.ones(10) * 0.99,
            "Close": np.ones(10),
            "Volume": np.ones(10) * 1e6,
        },
        index=dates,
    )
    with pytest.raises(AssertionError, match="Insufficient rows"):
        _make_agent(df).run({"data": None})


def test_non_datetime_index_raises(sample_ohlcv):
    df = sample_ohlcv.reset_index(drop=True)
    with pytest.raises(AssertionError, match="DatetimeIndex"):
        _make_agent(df).run({"data": None})
