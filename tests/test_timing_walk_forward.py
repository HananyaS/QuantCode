"""Tests for utils/timing_walk_forward.py — multi-fold walk-forward Sharpe
evaluation for a single-instrument rule-based timing signal.

Unlike utils/walk_forward.py (which fits an ML model per fold), a timing
signal here is rule-based with no per-fold fit — folds exist to check
*consistency* across regimes, not train/test generalization of a fitted
model. The overfitting-analogue penalty is cross-fold variance, not a
train/test metric gap.
"""
import numpy as np
import pandas as pd
import pytest

from utils.timing_walk_forward import timing_walk_forward_validate


def test_returns_one_row_per_fold():
    n = 250
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.Series(np.random.RandomState(0).normal(0.0005, 0.01, n), index=dates)

    result = timing_walk_forward_validate(rets, n_splits=4, periods_per_year=252)
    assert len(result) == 4
    assert {"fold", "test_start", "test_end", "test_sharpe"}.issubset(result.columns)


def test_perfectly_consistent_returns_give_zero_cross_fold_std():
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.Series([0.001] * n, index=dates)  # identical every day -> identical Sharpe... actually zero vol

    result = timing_walk_forward_validate(rets, n_splits=3, periods_per_year=252)
    # constant returns => zero std => Sharpe defined as 0.0 per fold (no div by zero)
    assert (result["test_sharpe"] == 0.0).all()


def test_fold_windows_are_chronological_and_non_overlapping():
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.Series(np.random.RandomState(1).normal(0, 0.01, n), index=dates)

    result = timing_walk_forward_validate(rets, n_splits=5, periods_per_year=252)
    starts = pd.to_datetime(result["test_start"])
    ends = pd.to_datetime(result["test_end"])
    for i in range(len(result) - 1):
        assert ends.iloc[i] < starts.iloc[i + 1]
