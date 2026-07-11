"""timing_walk_forward.py — multi-fold walk-forward Sharpe for a
single-instrument, rule-based timing signal.

Why this differs from utils/walk_forward.py
---------------------------------------------
utils.walk_forward fits an ML model per fold and compares train vs test IC
to catch overfitting. A rule-based timing signal (agents/timing) has no
per-fold fit — its "hyperparameters" (SMA window, thresholds, combine mode)
are set once in config and mutated between autoresearch iterations, not
per-fold. Folds here exist to check *consistency across market regimes*
instead: reusing the same expanding-window date partition machinery
(utils.walk_forward.generate_folds), but reporting per-fold test-window
Sharpe only. The overfitting-analogue signal is cross-fold variance — a
rule that only "works" in one regime (e.g. only during the one big fold that
contains 2008) is a red flag the same way a large train/test IC gap is for
the ML case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.walk_forward import generate_folds


def _sharpe(returns: pd.Series, periods_per_year: int) -> float:
    sigma = returns.std()
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(returns.mean() / sigma * np.sqrt(periods_per_year))


def timing_walk_forward_validate(
    returns: pd.Series,
    n_splits: int = 5,
    periods_per_year: int = 252,
    purge_days: int = 0,
) -> pd.DataFrame:
    """Partition `returns` into `n_splits` expanding-window folds and report
    each fold's test-window Sharpe.

    Args:
        returns: Strategy (or benchmark) returns indexed by date.
        n_splits: Number of walk-forward folds.
        periods_per_year: Annualization factor (252 for daily bars).
        purge_days: Passed through to generate_folds; irrelevant here since
            only each fold's test window is used (no per-fold training), but
            kept for interface symmetry with utils.walk_forward.

    Returns:
        DataFrame with one row per fold: fold, test_start, test_end, test_sharpe.
    """
    folds = generate_folds(returns.index, n_splits=n_splits, purge_days=purge_days)

    rows = []
    for fold in folds:
        test_mask = returns.index.isin(fold.test_dates)
        test_ret = returns[test_mask]
        rows.append({
            "fold": fold.fold_idx,
            "test_start": str(fold.test_dates[0].date()),
            "test_end": str(fold.test_dates[-1].date()),
            "test_sharpe": _sharpe(test_ret, periods_per_year),
        })

    return pd.DataFrame(rows)
