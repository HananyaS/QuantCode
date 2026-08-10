"""folds.py — purged, embargoed, date-blocked cross-validation.

Why not plain K-fold
--------------------
Random K-fold over trades is invalid here for three separate reasons, any one
of which alone would be fatal:

1. **Temporal order.** A fold containing 2026 data used to predict 2022 is not
   a forecast. Splits must respect time.
2. **Same-day clustering.** Trades on one session share a market factor. If
   some of a day's trades land in train and the rest in test, the model can
   learn that day's regime from the training half and "predict" the test half.
   Folds are therefore split on **whole days**, never on individual trades.
3. **Feature-window bleed.** Features use trailing windows of up to 60 days. A
   test day immediately following the train boundary is described by features
   computed partly from training data. An **embargo** removes days adjacent to
   the boundary.

Labels here are same-session (entry 09:35, exit 15:55), so label horizons do
not overlap and the classic purge for overlapping labels is not needed. The
embargo addresses feature-window bleed and serial correlation, which do exist.
This distinction is deliberate: purging is often applied mechanically without
asking which of the two problems is actually present.

Reference: Lopez de Prado, *Advances in Financial Machine Learning* (2018),
ch. 7 (purged K-fold and embargoing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """One train/test split, expressed as sets of dates."""

    index: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    embargoed_dates: pd.DatetimeIndex

    @property
    def train_start(self):
        return self.train_dates.min() if len(self.train_dates) else None

    @property
    def train_end(self):
        return self.train_dates.max() if len(self.train_dates) else None

    @property
    def test_start(self):
        return self.test_dates.min() if len(self.test_dates) else None

    @property
    def test_end(self):
        return self.test_dates.max() if len(self.test_dates) else None

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start:%Y-%m-%d}..{self.train_end:%Y-%m-%d} "
            f"({len(self.train_dates)}d) | embargo {len(self.embargoed_dates)}d | "
            f"test {self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d} ({len(self.test_dates)}d)"
        )


def walk_forward_folds(
    dates: Sequence,
    n_splits: int = 5,
    embargo_days: int = 5,
    expanding: bool = True,
    min_train_days: int = 120,
) -> List[Fold]:
    """Sequential train/test splits over unique sorted dates.

    ``expanding=True`` grows the training window (each fold trains on all prior
    history); ``False`` uses a rolling window of fixed length. Expanding is the
    default because it matches how the strategy would actually be run.

    The embargo is carved out of the **end of train**, never out of test:
    shrinking the test set would silently change what is being measured
    between folds.
    """
    unique = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates)).unique()))
    n = len(unique)
    if n < min_train_days + n_splits:
        raise ValueError(
            f"need at least {min_train_days + n_splits} distinct dates for "
            f"{n_splits} folds; got {n}"
        )

    test_size = (n - min_train_days) // n_splits
    if test_size < 1:
        raise ValueError("test window would be empty; reduce n_splits or min_train_days")

    folds: List[Fold] = []
    for i in range(n_splits):
        test_start = min_train_days + i * test_size
        test_end = test_start + test_size if i < n_splits - 1 else n

        test_dates = unique[test_start:test_end]
        train_upper = test_start - embargo_days
        if train_upper <= 0:
            continue

        train_lower = 0 if expanding else max(0, train_upper - min_train_days)
        train_dates = unique[train_lower:train_upper]
        embargoed = unique[max(train_upper, 0) : test_start]

        if len(train_dates) == 0 or len(test_dates) == 0:
            continue

        folds.append(Fold(i, train_dates, test_dates, embargoed))

    return folds


def assert_folds_are_clean(folds: List[Fold], embargo_days: int) -> None:
    """Fail loudly if any fold violates the contract.

    Called in tests and before every reported result. A silently broken split
    is the most expensive kind of bug in this project: it produces a *better*
    number, so nothing looks wrong.
    """
    for fold in folds:
        train, test = set(fold.train_dates), set(fold.test_dates)

        overlap = train & test
        assert not overlap, f"fold {fold.index}: {len(overlap)} dates in both train and test"

        assert fold.train_end < fold.test_start, (
            f"fold {fold.index}: train ends {fold.train_end} but test starts "
            f"{fold.test_start} -- training on the future"
        )

        # Must be a real comparison. `gap_days >= min(embargo_days, gap_days)`
        # holds for every possible value, so the original form could never fail
        # -- a dead tripwire in the very function whose job is to fail loudly.
        gap_days = len(fold.embargoed_dates)
        assert gap_days >= embargo_days, (
            f"fold {fold.index}: embargo is {gap_days} days, expected {embargo_days}"
        )

        assert not (set(fold.embargoed_dates) & train), (
            f"fold {fold.index}: embargoed dates leaked into train"
        )
        assert not (set(fold.embargoed_dates) & test), (
            f"fold {fold.index}: embargoed dates leaked into test"
        )


def split_frame(df: pd.DataFrame, fold: Fold, date_col: str = "date"):
    """Materialise one fold as (train, test) frames."""
    dates = pd.to_datetime(df[date_col])
    train = df[dates.isin(fold.train_dates)]
    test = df[dates.isin(fold.test_dates)]
    return train, test
