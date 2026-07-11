"""Tests for utils/walk_forward.py — multi-fold walk-forward validation."""
import numpy as np
import pandas as pd
import pytest

from utils.walk_forward import generate_folds, walk_forward_validate


# ---------------------------------------------------------------------------
# generate_folds
# ---------------------------------------------------------------------------

def test_generate_folds_returns_requested_count():
    dates = pd.bdate_range("2020-01-01", periods=300)
    folds = generate_folds(dates, n_splits=4, purge_days=5)
    assert len(folds) == 4


def test_generate_folds_expanding_train_window():
    dates = pd.bdate_range("2020-01-01", periods=300)
    folds = generate_folds(dates, n_splits=4, purge_days=5)
    train_sizes = [len(f.train_dates) for f in folds]
    assert train_sizes == sorted(train_sizes), "train window must expand fold over fold"
    assert all(b > a for a, b in zip(train_sizes, train_sizes[1:])), (
        "each fold's train window must strictly grow"
    )


def test_generate_folds_purge_gap_enforced():
    dates = pd.bdate_range("2020-01-01", periods=300)
    purge = 5
    folds = generate_folds(dates, n_splits=4, purge_days=purge)
    for f in folds:
        train_end = f.train_dates[-1]
        test_start = f.test_dates[0]
        gap_days = len(pd.bdate_range(train_end, test_start)) - 1
        assert gap_days >= purge


def test_generate_folds_no_overlap():
    dates = pd.bdate_range("2020-01-01", periods=300)
    folds = generate_folds(dates, n_splits=4, purge_days=5)
    for f in folds:
        assert set(f.train_dates).isdisjoint(set(f.test_dates))


def test_generate_folds_raises_when_too_few_dates():
    dates = pd.bdate_range("2020-01-01", periods=20)
    with pytest.raises(AssertionError):
        generate_folds(dates, n_splits=10, purge_days=5)


# ---------------------------------------------------------------------------
# walk_forward_validate
# ---------------------------------------------------------------------------

def test_walk_forward_validate_returns_dataframe_with_expected_columns(cs_features_and_labels):
    features, labels = cs_features_and_labels
    result = walk_forward_validate(
        features, labels, n_splits=3, purge_days=5, n_estimators=10, random_state=42,
    )
    assert isinstance(result, pd.DataFrame)
    for col in ("train_start", "train_end", "test_start", "test_end", "train_ic", "test_ic"):
        assert col in result.columns


def test_walk_forward_validate_row_count_matches_folds(cs_features_and_labels):
    features, labels = cs_features_and_labels
    result = walk_forward_validate(
        features, labels, n_splits=3, purge_days=5, n_estimators=10, random_state=42,
    )
    assert len(result) == 3


def test_walk_forward_validate_no_leakage(cs_features_and_labels):
    """Every fold's test_start must be strictly after that fold's train_end."""
    features, labels = cs_features_and_labels
    result = walk_forward_validate(
        features, labels, n_splits=3, purge_days=5, n_estimators=10, random_state=42,
    )
    assert (pd.to_datetime(result["test_start"]) > pd.to_datetime(result["train_end"])).all()


def test_walk_forward_validate_ic_values_finite(cs_features_and_labels):
    features, labels = cs_features_and_labels
    result = walk_forward_validate(
        features, labels, n_splits=3, purge_days=5, n_estimators=10, random_state=42,
    )
    assert np.isfinite(result["train_ic"]).all()
    assert np.isfinite(result["test_ic"]).all()


def test_walk_forward_validate_deterministic(cs_features_and_labels):
    features, labels = cs_features_and_labels
    r1 = walk_forward_validate(features, labels, n_splits=3, purge_days=5, n_estimators=10, random_state=42)
    r2 = walk_forward_validate(features, labels, n_splits=3, purge_days=5, n_estimators=10, random_state=42)
    pd.testing.assert_frame_equal(r1, r2)
