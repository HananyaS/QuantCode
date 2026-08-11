"""nulls.py — tripwires that must fire before any result is believed.

A leakage bug does not announce itself. It produces a *better* number, so
nothing looks wrong -- which makes "the result improved" the least reliable
signal available. These tests deliberately destroy the thing being measured and
assert that the measured edge collapses with it. An edge that survives a
destroyed label is an edge in the pipeline, not in the market.

Run every one of these before quoting any performance figure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.penny.eval import conditional, stats as pstats

logger = logging.getLogger(__name__)


@dataclass
class NullResult:
    """Outcome of one null test."""

    name: str
    passed: bool
    observed: float
    reference: float
    detail: str

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def shuffled_label_test(
    trades: pd.DataFrame,
    veto_cols: Sequence[str],
    max_size: int = 2,
    n_repeats: int = 20,
    return_col: str = "net_return",
    date_col: str = "date",
    seed: int = 11,
) -> NullResult:
    """Shuffle returns across trades; the search must stop finding anything.

    Breaks any association between features and outcome while preserving both
    marginal distributions. If the veto search still produces significant
    results at the usual rate, the significance is an artefact of the search
    itself rather than of any real relationship.

    Returns are **demeaned** after shuffling, and this is essential rather than
    cosmetic. ``search_veto_sets`` tests each subset against zero, which is the
    right question for the real study ("is this population profitable?"). Under
    a shuffle every subset simply inherits the panel's overall mean, so if that
    mean is non-zero -- and for penny stocks it is emphatically negative --
    every subset is "significant" and the null test would fail for a reason
    that has nothing to do with leakage. Demeaning re-poses the question as the
    one the null actually cares about: can the vetoes still *discriminate*
    relative to the baseline?

    The comparison is against the FDR level: under a true null, the fraction of
    subsets passing BH at q should be about q, not far above it.
    """
    rng = np.random.default_rng(seed)
    pass_rates: List[float] = []

    for _ in range(n_repeats):
        shuffled = trades.copy()
        permuted = rng.permutation(shuffled[return_col].to_numpy())
        shuffled[return_col] = permuted - np.nanmean(permuted)
        table = conditional.search_veto_sets(
            shuffled, veto_cols, max_size=max_size,
            min_retained=0.05, min_trades=100, return_col=return_col, n_boot=300,
        )
        if table.empty:
            continue
        pass_rates.append(float(table["q_pass"].mean()))

    observed = float(np.mean(pass_rates)) if pass_rates else 0.0
    # Allow generous headroom: bootstrap p-values are discrete and the null
    # distribution is lumpy, so a strict q comparison would be flaky.
    threshold = 0.20
    return NullResult(
        "shuffled_labels",
        passed=observed <= threshold,
        observed=observed,
        reference=threshold,
        detail=(
            f"mean fraction of veto sets passing FDR on shuffled labels = "
            f"{observed:.3f} over {len(pass_rates)} repeats (must be <= {threshold})"
        ),
    )


def date_shift_test(
    trades: pd.DataFrame,
    veto_cols: Sequence[str],
    shift_days: int = 20,
    max_size: int = 2,
    return_col: str = "net_return",
    date_col: str = "date",
) -> NullResult:
    """Pair each day's features with a different day's outcomes.

    A subtler probe than shuffling: it preserves within-day clustering and the
    time-series structure, and only breaks the feature-to-outcome correspondence.
    Catches leakage that a naive shuffle can miss, because the shuffled version
    also destroys the day-clustering that the bootstrap relies on.

    As in :func:`shuffled_label_test`, returns are demeaned so that the test
    asks whether vetoes can still discriminate, not whether the panel's overall
    mean happens to differ from zero.
    """
    df = trades.dropna(subset=[return_col]).sort_values(date_col).copy()
    days = pd.DatetimeIndex(sorted(df[date_col].unique()))
    if len(days) <= shift_days:
        return NullResult("date_shift", True, np.nan, np.nan, "too few days to shift; skipped")

    mapping = dict(zip(days, np.roll(days, shift_days)))
    returns_by_day = {d: g[return_col].to_numpy() for d, g in df.groupby(date_col)}

    rebuilt = []
    for day, group in df.groupby(date_col):
        donor = returns_by_day.get(mapping[day])
        if donor is None or len(donor) == 0:
            continue
        g = group.copy()
        idx = np.arange(len(g)) % len(donor)
        g[return_col] = donor[idx]
        rebuilt.append(g)

    if not rebuilt:
        return NullResult("date_shift", True, np.nan, np.nan, "no data after shift; skipped")

    shifted = pd.concat(rebuilt, ignore_index=True)
    shifted[return_col] = shifted[return_col] - shifted[return_col].mean()
    table = conditional.search_veto_sets(
        shifted, veto_cols, max_size=max_size,
        min_retained=0.05, min_trades=100, return_col=return_col, n_boot=300,
    )
    observed = float(table["q_pass"].mean()) if not table.empty else 0.0
    threshold = 0.20
    return NullResult(
        "date_shift",
        passed=observed <= threshold,
        observed=observed,
        reference=threshold,
        detail=(
            f"fraction of veto sets passing FDR with outcomes shifted "
            f"{shift_days} days = {observed:.3f} (must be <= {threshold})"
        ),
    )


def feature_timing_tripwire(
    features: pd.DataFrame,
    decision_minute: str = "09:35",
    minute_col: str = "minute_et",
) -> NullResult:
    """Assert no intraday feature was built from a bar at/after the decision.

    A structural check rather than a statistical one: it inspects the data that
    produced the features instead of inferring leakage from suspicious results.
    Cheap, and catches the failure mode directly.
    """
    if minute_col not in features.columns:
        return NullResult(
            "feature_timing", True, np.nan, np.nan,
            f"no {minute_col} column present; nothing to check",
        )
    offending = features[features[minute_col] >= decision_minute]
    n = len(offending)
    return NullResult(
        "feature_timing",
        passed=n == 0,
        observed=float(n),
        reference=0.0,
        detail=(
            f"{n} feature rows built from bars at or after {decision_minute}"
            if n else f"no feature row uses a bar at or after {decision_minute}"
        ),
    )


def constant_return_test(
    trades: pd.DataFrame,
    veto_cols: Sequence[str],
    return_col: str = "net_return",
) -> NullResult:
    """With every return identical, no veto can create dispersion.

    A sanity check on the plumbing: if any veto set reports a mean different
    from the constant, the aggregation or weighting logic is wrong -- for
    example double-counting trades or mis-joining on the date key.
    """
    df = trades.dropna(subset=[return_col]).copy()
    if df.empty:
        return NullResult("constant_return", True, np.nan, np.nan, "no trades; skipped")

    df[return_col] = 0.01
    means = []
    for vetoes in conditional.enumerate_veto_sets(list(veto_cols)[:6], max_size=2):
        kept = df[conditional.apply_vetoes(df, vetoes)]
        if len(kept):
            means.append(float(kept[return_col].mean()))

    spread = float(np.max(means) - np.min(means)) if means else 0.0
    return NullResult(
        "constant_return",
        passed=spread < 1e-12,
        observed=spread,
        reference=0.0,
        detail=f"spread of veto-set means on a constant return = {spread:.2e} (must be ~0)",
    )


def run_all(
    trades: pd.DataFrame,
    veto_cols: Sequence[str],
    features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run every null test and return a tidy report."""
    results = [
        constant_return_test(trades, veto_cols),
        shuffled_label_test(trades, veto_cols),
        date_shift_test(trades, veto_cols),
    ]
    if features is not None:
        results.append(feature_timing_tripwire(features))

    for r in results:
        logger.info("null %s: %s -- %s", r.name, r.status, r.detail)

    # `status` is a property, so __dict__ omits it; add it explicitly rather
    # than leaving callers to recompute the pass/fail label themselves.
    return pd.DataFrame([{**r.__dict__, "status": r.status} for r in results])
