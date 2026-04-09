"""Tests for LabelingAgent — label correctness, no-leakage, edge cases."""
import numpy as np
import pandas as pd
import pytest

from agents.labeling_agent import LabelingAgent


def _run(df: pd.DataFrame, **kwargs) -> pd.Series:
    ctx = LabelingAgent(**kwargs).run({"data": df})
    return ctx["labels"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_labels_are_binary(sample_ohlcv):
    labels = _run(sample_ohlcv)
    assert set(labels.unique()).issubset({0, 1})


def test_no_nan_in_labels(sample_ohlcv):
    labels = _run(sample_ohlcv)
    assert labels.notna().all()


def test_label_count(sample_ohlcv):
    """Labels = len(data) - forward_period."""
    labels = _run(sample_ohlcv, forward_period=1)
    assert len(labels) == len(sample_ohlcv) - 1


def test_label_count_multiday(sample_ohlcv):
    labels = _run(sample_ohlcv, forward_period=3)
    assert len(labels) == len(sample_ohlcv) - 3


def test_labels_named_label(sample_ohlcv):
    labels = _run(sample_ohlcv)
    assert labels.name == "label"


def test_index_is_datetime(sample_ohlcv):
    labels = _run(sample_ohlcv)
    assert isinstance(labels.index, pd.DatetimeIndex)


def test_both_classes_present(sample_ohlcv):
    """A random-walk series should produce both up and down labels."""
    labels = _run(sample_ohlcv)
    assert 0 in labels.values and 1 in labels.values


# ---------------------------------------------------------------------------
# No-leakage check
# ---------------------------------------------------------------------------

def test_label_uses_future_price_correctly(sample_ohlcv):
    """Label[t] must equal 1 iff close[t+1] > close[t]."""
    close = sample_ohlcv["Close"]
    labels = _run(sample_ohlcv, forward_period=1, threshold=0.0)
    for i in range(len(labels)):
        t = labels.index[i]
        t_next = sample_ohlcv.index[i + 1]
        expected = int(close[t_next] > close[t])
        assert labels.iloc[i] == expected, (
            f"Label mismatch at {t}: expected {expected}, got {labels.iloc[i]}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_constant_prices_all_zero():
    """Constant prices → zero forward return → all labels == 0 (threshold=0 is exclusive)."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    df = pd.DataFrame(
        {
            "Open": np.ones(100),
            "High": np.ones(100),
            "Low": np.ones(100),
            "Close": np.ones(100),
            "Volume": np.ones(100) * 1e6,
        },
        index=dates,
    )
    labels = _run(df, threshold=0.0)
    assert (labels == 0).all(), "Constant prices should produce all-zero labels"


def test_strictly_rising_prices_all_one():
    """Monotonically rising prices → all labels == 1."""
    dates = pd.bdate_range("2020-01-01", periods=50)
    close = np.arange(1, 51, dtype=float)
    df = pd.DataFrame(
        {"Open": close, "High": close + 0.1, "Low": close - 0.1, "Close": close, "Volume": np.ones(50) * 1e6},
        index=dates,
    )
    labels = _run(df, threshold=0.0)
    assert (labels == 1).all(), "Rising prices should produce all-one labels"


def test_high_threshold_all_zero(sample_ohlcv):
    """With threshold=1.0 (100%), no daily return should exceed it → all zeros."""
    labels = _run(sample_ohlcv, threshold=1.0)
    assert (labels == 0).all()


def test_invalid_forward_period():
    with pytest.raises(AssertionError):
        LabelingAgent(forward_period=0)
