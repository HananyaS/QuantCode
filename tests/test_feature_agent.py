"""Tests for FeatureAgent — feature computation, no-leakage, edge cases."""
import numpy as np
import pandas as pd
import pytest

from agents.feature_agent import FeatureAgent


def _run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    ctx = FeatureAgent(**kwargs).run({"data": df})
    return ctx["features"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_returns_dataframe(sample_ohlcv):
    features = _run(sample_ohlcv)
    assert isinstance(features, pd.DataFrame)


def test_no_nan_in_output(sample_ohlcv):
    features = _run(sample_ohlcv)
    assert not features.isna().any().any(), "Feature matrix contains NaN after dropna"


def test_expected_columns_present(sample_ohlcv):
    features = _run(sample_ohlcv)
    expected = {"SMA_5_ratio", "SMA_10_ratio", "SMA_20_ratio", "RSI", "BB_pos", "MOM_5", "RET_1", "VOL_ratio"}
    assert expected.issubset(set(features.columns))


def test_fewer_rows_than_input(sample_ohlcv):
    """Warm-up rows must be dropped."""
    features = _run(sample_ohlcv)
    assert len(features) < len(sample_ohlcv)


def test_index_is_subset_of_data_index(sample_ohlcv):
    features = _run(sample_ohlcv)
    assert features.index.isin(sample_ohlcv.index).all()


def test_index_sorted(sample_ohlcv):
    features = _run(sample_ohlcv)
    assert features.index.is_monotonic_increasing


def test_label_not_in_features(sample_ohlcv):
    features = _run(sample_ohlcv)
    assert "label" not in features.columns


def test_custom_sma_windows(sample_ohlcv):
    features = _run(sample_ohlcv, sma_windows=[3, 7])
    assert "SMA_3_ratio" in features.columns
    assert "SMA_7_ratio" in features.columns
    assert "SMA_5_ratio" not in features.columns


def test_sma_ratio_near_one(sample_ohlcv):
    """Price / SMA should hover around 1 for trending prices."""
    features = _run(sample_ohlcv)
    mean_ratio = features["SMA_20_ratio"].mean()
    assert 0.5 < mean_ratio < 2.0, f"SMA_20_ratio mean={mean_ratio} looks unreasonable"


def test_rsi_bounded(sample_ohlcv):
    """RSI must always be in [0, 100]."""
    features = _run(sample_ohlcv)
    assert (features["RSI"] >= 0).all() and (features["RSI"] <= 100).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_constant_close_raises(constant_ohlcv):
    """All features are NaN for constant prices → assertion fails after dropna."""
    with pytest.raises(AssertionError):
        _run(constant_ohlcv)


def test_small_dataset_works(small_ohlcv):
    """60-row dataset is enough for rolling windows up to 20."""
    features = _run(small_ohlcv)
    assert len(features) > 0


def test_no_future_data_in_rsi(sample_ohlcv):
    """RSI at row t must not depend on data after row t.

    We verify by zeroing out all rows after t and checking RSI is unchanged.
    """
    features_full = _run(sample_ohlcv)
    # Use 50th row as the check point
    t = 50
    truncated = sample_ohlcv.iloc[: t + 1].copy()
    features_trunc = _run(truncated)
    # The last value of the truncated run must match the t-th value of the full run
    rsi_full_at_t = features_full["RSI"].iloc[features_full.index.get_loc(truncated.index[-1])]
    rsi_trunc_last = features_trunc["RSI"].iloc[-1]
    assert abs(rsi_full_at_t - rsi_trunc_last) < 1e-9, (
        f"RSI is not backward-looking: full={rsi_full_at_t}, trunc={rsi_trunc_last}"
    )
