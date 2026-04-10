"""Tests for CrossSectionalFeatureAgent."""
import numpy as np
import pandas as pd
import pytest

from agents.cs_feature_agent import CrossSectionalFeatureAgent


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_output_is_multiindex(universe_data):
    agent = CrossSectionalFeatureAgent(cross_sectional=False)
    ctx = agent.run({"universe_data": universe_data})
    assert isinstance(ctx["cs_features"].index, pd.MultiIndex)
    assert ctx["cs_features"].index.names == ["date", "ticker"]


def test_expected_base_feature_columns(universe_data):
    agent = CrossSectionalFeatureAgent(
        returns_windows=[1, 5],
        vol_window=10,
        rsi_period=14,
        sma_windows=[10],
        cross_sectional=False,
    )
    ctx = agent.run({"universe_data": universe_data})
    cols = set(ctx["cs_features"].columns)
    assert "ret_1d" in cols
    assert "ret_5d" in cols
    assert "vol_10d" in cols
    assert "rsi" in cols
    assert "sma_10_ratio" in cols


def test_cross_sectional_columns_added(universe_data):
    agent = CrossSectionalFeatureAgent(
        returns_windows=[1],
        vol_window=10,
        sma_windows=[],
        cross_sectional=True,
    )
    ctx = agent.run({"universe_data": universe_data})
    cols = set(ctx["cs_features"].columns)
    assert "rank_ret_1d" in cols
    assert "zscore_ret_1d" in cols


def test_no_nan_after_warmup_drop(universe_data):
    agent = CrossSectionalFeatureAgent(cross_sectional=False)
    ctx = agent.run({"universe_data": universe_data})
    assert ctx["cs_features"].isna().sum().sum() == 0


def test_all_tickers_present(universe_data):
    agent = CrossSectionalFeatureAgent(cross_sectional=False)
    ctx = agent.run({"universe_data": universe_data})
    tickers_in_features = set(
        ctx["cs_features"].index.get_level_values("ticker").unique()
    )
    assert tickers_in_features == set(universe_data.keys())


# ---------------------------------------------------------------------------
# Cross-sectional normalization
# ---------------------------------------------------------------------------

def test_rank_is_between_0_and_1(universe_data):
    agent = CrossSectionalFeatureAgent(
        returns_windows=[1], sma_windows=[], cross_sectional=True
    )
    ctx = agent.run({"universe_data": universe_data})
    rank_col = ctx["cs_features"]["rank_ret_1d"]
    assert (rank_col >= 0).all() and (rank_col <= 1).all()


def test_zscore_mean_near_zero_per_date(universe_data):
    agent = CrossSectionalFeatureAgent(
        returns_windows=[1], sma_windows=[], cross_sectional=True
    )
    ctx = agent.run({"universe_data": universe_data})
    zscore_col = ctx["cs_features"]["zscore_ret_1d"]
    for date, group in zscore_col.groupby(level="date"):
        assert abs(group.mean()) < 1e-9 or len(group) < 2  # mean ~0


def test_cs_normalization_uses_no_future_data(universe_data):
    """Verify cs features computed up to date T do not depend on dates > T.
    Strategy: compute features on [0:N] and [0:N//2] and check the overlap is identical.
    """
    tickers = list(universe_data.keys())
    n = len(next(iter(universe_data.values())))
    half = n // 2

    half_universe = {t: df.iloc[:half] for t, df in universe_data.items()}

    agent = CrossSectionalFeatureAgent(
        returns_windows=[1], sma_windows=[], cross_sectional=True
    )
    ctx_full = agent.run({"universe_data": universe_data})
    ctx_half = agent.run({"universe_data": half_universe})

    common_dates = (
        ctx_full["cs_features"].index.get_level_values("date").unique()
        .intersection(ctx_half["cs_features"].index.get_level_values("date").unique())
    )
    assert len(common_dates) > 0

    for date in common_dates[:5]:  # check a sample
        full_row = ctx_full["cs_features"].xs(date, level="date").sort_index()
        half_row = ctx_half["cs_features"].xs(date, level="date").sort_index()
        pd.testing.assert_frame_equal(full_row, half_row, check_exact=False, atol=1e-10)


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def test_label_column_not_in_features(universe_data):
    agent = CrossSectionalFeatureAgent(cross_sectional=False)
    ctx = agent.run({"universe_data": universe_data})
    assert "label" not in ctx["cs_features"].columns


# ---------------------------------------------------------------------------
# Missing context
# ---------------------------------------------------------------------------

def test_raises_if_universe_data_missing():
    agent = CrossSectionalFeatureAgent()
    with pytest.raises(AssertionError, match="universe_data"):
        agent.run({})
