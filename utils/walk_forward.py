"""walk_forward.py — multi-fold walk-forward validation for the ranking model.

Why walk-forward instead of a single train/test split
-------------------------------------------------------
`RankingModelAgent` fits and evaluates on exactly one purged temporal split.
A single split is weak evidence of a durable edge — it could easily be one
lucky (or unlucky) period. Walk-forward evaluation refits across multiple
expanding-window folds and reports the *distribution* of out-of-sample IC,
plus the train-vs-test IC gap per fold as a direct overfitting signal.

Fold construction
------------------
Unique sorted dates are split into `n_splits` expanding-window folds: fold i
trains on all dates before its test window and tests on the i-th of
`n_splits` equal-sized trailing chunks. A `purge_days` gap is dropped from
the end of each fold's training window so training labels (forward returns)
never overlap with the test window's start.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # pragma: no cover
    _HAS_LGB = False


@dataclass
class Fold:
    """One walk-forward fold's train/test date partition."""
    fold_idx: int
    train_dates: List[pd.Timestamp]
    test_dates: List[pd.Timestamp]


def generate_folds(
    dates: pd.DatetimeIndex,
    n_splits: int,
    purge_days: int,
) -> List[Fold]:
    """Partition sorted unique dates into `n_splits` expanding-window folds.

    Args:
        dates: All available dates (need not be pre-sorted or de-duplicated).
        n_splits: Number of walk-forward folds.
        purge_days: Business days dropped from the end of each fold's
            training window to prevent forward-return label leakage.

    Returns:
        List of `Fold`, ordered chronologically. Fold i's test window is the
        i-th of `n_splits` equal trailing chunks; its train window is every
        earlier date, purged.
    """
    unique_dates = pd.DatetimeIndex(dates).unique().sort_values()
    n = len(unique_dates)
    assert n_splits >= 1, "n_splits must be >= 1"
    # Reserve at least 1 date per test chunk plus room for a training window
    # ahead of the first chunk.
    assert n >= n_splits + 1, (
        f"Need at least n_splits+1={n_splits + 1} dates, got {n}"
    )

    test_chunk_size = n // (n_splits + 1)
    assert test_chunk_size > purge_days, (
        f"Not enough dates ({n}) for {n_splits} splits with purge_days={purge_days} "
        f"— each test chunk (~{test_chunk_size} dates) must exceed the purge window"
    )

    folds: List[Fold] = []
    for i in range(n_splits):
        test_start_idx = (i + 1) * test_chunk_size
        test_end_idx = n if i == n_splits - 1 else (i + 2) * test_chunk_size
        test_dates = list(unique_dates[test_start_idx:test_end_idx])
        if not test_dates:
            continue

        purge_end_idx = max(test_start_idx - purge_days, 1)
        train_dates = list(unique_dates[:purge_end_idx])
        assert train_dates, f"Fold {i}: empty training window after purge"

        folds.append(Fold(fold_idx=i, train_dates=train_dates, test_dates=test_dates))

    return folds


def _fit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
):
    if _HAS_LGB and model_type == "lightgbm":
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    model.fit(X_train.values, y_train.values)
    return model


def _mean_ic(scores: pd.Series, actual: pd.Series) -> float:
    """Average Spearman rank IC across dates present in both series."""
    ics = []
    for date in scores.index.get_level_values("date").unique():
        try:
            s = scores.xs(date, level="date")
            a = actual.xs(date, level="date")
        except KeyError:
            continue
        aligned = pd.concat([s, a], axis=1).dropna()
        if len(aligned) >= 10:
            corr, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            if not np.isnan(corr):
                ics.append(corr)
    return float(np.mean(ics)) if ics else 0.0


def walk_forward_validate(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
    purge_days: int = 5,
    model_type: str = "lightgbm",
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run multi-fold walk-forward validation and report per-fold IC.

    Args:
        features: DataFrame indexed by MultiIndex(date, ticker).
        labels: Series indexed by MultiIndex(date, ticker), aligned to features.
        n_splits: Number of expanding-window folds.
        purge_days: Business days purged from the end of each fold's train window.
        model_type, n_estimators, max_depth, learning_rate, random_state:
            Passed through to the underlying regressor for each fold.

    Returns:
        DataFrame with one row per fold: fold, train_start, train_end,
        test_start, test_end, train_ic, test_ic, ic_gap (train_ic - test_ic).
    """
    combined = features.join(labels, how="inner").dropna()
    assert len(combined) > 0, "No rows remain after aligning features and labels"

    X = combined.drop(columns=["label"])
    y = combined["label"]

    dates = X.index.get_level_values("date")
    folds = generate_folds(dates, n_splits=n_splits, purge_days=purge_days)

    rows = []
    for fold in folds:
        train_mask = dates.isin(fold.train_dates)
        test_mask = dates.isin(fold.test_dates)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model = _fit(X_train, y_train, model_type, n_estimators, max_depth, learning_rate, random_state)

        train_scores = pd.Series(model.predict(X_train.values), index=X_train.index, name="score")
        test_scores = pd.Series(model.predict(X_test.values), index=X_test.index, name="score")

        train_ic = _mean_ic(train_scores, y_train)
        test_ic = _mean_ic(test_scores, y_test)

        rows.append({
            "fold": fold.fold_idx,
            "train_start": str(fold.train_dates[0].date()),
            "train_end": str(fold.train_dates[-1].date()),
            "test_start": str(fold.test_dates[0].date()),
            "test_end": str(fold.test_dates[-1].date()),
            "train_ic": train_ic,
            "test_ic": test_ic,
            "ic_gap": train_ic - test_ic,
        })

    return pd.DataFrame(rows)
