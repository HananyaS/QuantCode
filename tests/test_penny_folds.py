"""Tests for the cross-validation contract.

A broken split is the most expensive bug in this project because it produces a
*better* number, so nothing looks wrong. These tests assert the contract
directly rather than inferring it from results.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.eval.folds import (
    Fold,
    assert_folds_are_clean,
    split_frame,
    walk_forward_folds,
)


def _dates(n=400, start="2022-01-03"):
    return pd.bdate_range(start, periods=n)


# ---- Basic structure ----


def test_folds_are_time_ordered_and_disjoint():
    folds = walk_forward_folds(_dates(), n_splits=5, embargo_days=5, min_train_days=120)
    assert len(folds) == 5
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert not (set(fold.train_dates) & set(fold.test_dates))


def test_test_windows_are_contiguous_and_non_overlapping():
    folds = walk_forward_folds(_dates(), n_splits=4, embargo_days=3, min_train_days=100)
    seen = set()
    for fold in folds:
        overlap = seen & set(fold.test_dates)
        assert not overlap, "test windows must not overlap across folds"
        seen |= set(fold.test_dates)


def test_expanding_training_window_grows():
    folds = walk_forward_folds(_dates(), n_splits=4, embargo_days=5, expanding=True, min_train_days=100)
    sizes = [len(f.train_dates) for f in folds]
    assert sizes == sorted(sizes), "expanding folds must have non-decreasing train size"


def test_rolling_window_does_not_grow_without_bound():
    folds = walk_forward_folds(
        _dates(), n_splits=4, embargo_days=5, expanding=False, min_train_days=100
    )
    sizes = [len(f.train_dates) for f in folds]
    assert max(sizes) <= 100


# ---- The embargo ----


def test_embargo_creates_a_real_gap_between_train_and_test():
    embargo = 7
    folds = walk_forward_folds(_dates(), n_splits=4, embargo_days=embargo, min_train_days=120)
    for fold in folds:
        assert len(fold.embargoed_dates) == embargo
        # Nothing embargoed may appear on either side.
        assert not (set(fold.embargoed_dates) & set(fold.train_dates))
        assert not (set(fold.embargoed_dates) & set(fold.test_dates))


def test_embargo_is_taken_from_train_not_test():
    """Shrinking test would change what is measured between folds."""
    no_embargo = walk_forward_folds(_dates(), n_splits=4, embargo_days=0, min_train_days=120)
    with_embargo = walk_forward_folds(_dates(), n_splits=4, embargo_days=10, min_train_days=120)
    for a, b in zip(no_embargo, with_embargo):
        assert len(a.test_dates) == len(b.test_dates), "test window size must not change"
        assert len(b.train_dates) < len(a.train_dates), "train must absorb the embargo"


def test_zero_embargo_leaves_train_adjacent_to_test():
    folds = walk_forward_folds(_dates(), n_splits=3, embargo_days=0, min_train_days=120)
    for fold in folds:
        assert len(fold.embargoed_dates) == 0


# ---- The guard itself ----


def test_clean_folds_pass_the_guard():
    folds = walk_forward_folds(_dates(), n_splits=5, embargo_days=5, min_train_days=120)
    assert_folds_are_clean(folds, embargo_days=5)


def test_guard_catches_train_test_overlap():
    dates = _dates(200)
    bad = [Fold(0, dates[:100], dates[90:150], pd.DatetimeIndex([]))]
    with pytest.raises(AssertionError, match="both train and test"):
        assert_folds_are_clean(bad, embargo_days=0)


def test_guard_catches_training_on_the_future():
    dates = _dates(200)
    bad = [Fold(0, dates[100:150], dates[0:50], pd.DatetimeIndex([]))]
    with pytest.raises(AssertionError, match="training on the future"):
        assert_folds_are_clean(bad, embargo_days=0)


def test_guard_catches_embargo_leaking_into_train():
    dates = _dates(200)
    bad = [Fold(0, dates[:100], dates[110:150], dates[95:105])]
    with pytest.raises(AssertionError, match="leaked into train"):
        assert_folds_are_clean(bad, embargo_days=5)


# ---- Frame splitting ----


def test_split_frame_partitions_by_whole_days():
    """Same-day trades must never be divided across train and test."""
    dates = _dates(300)
    df = pd.DataFrame(
        {
            "date": np.repeat(dates, 3),
            "symbol": ["A", "B", "C"] * len(dates),
            "net_return": 0.01,
        }
    )
    folds = walk_forward_folds(dates, n_splits=3, embargo_days=5, min_train_days=120)
    for fold in folds:
        train, test = split_frame(df, fold)
        shared = set(train["date"]) & set(test["date"])
        assert not shared, "a day appeared in both train and test"
        # Every day is present in full or not at all.
        for part in (train, test):
            counts = part.groupby("date").size()
            assert (counts == 3).all(), "a day was split across folds"


def test_insufficient_history_raises_rather_than_silently_shrinking():
    with pytest.raises(ValueError, match="at least"):
        walk_forward_folds(_dates(50), n_splits=5, min_train_days=120)
