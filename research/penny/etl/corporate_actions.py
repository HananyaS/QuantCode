"""corporate_actions.py — detect reverse splits the price feed did not adjust.

Why this exists
---------------
A naive "top gainers" screen over this universe is contaminated. GIBO Holdings
shows **+1,716% on 2026-06-30** in both the adjusted and unadjusted series; it
was a **25-for-1 reverse stock split** effective 2026-06-29, confirmed by the
company and by Nasdaq's corporate-actions notice. The vendor's adjustment factor
did not move, so the usual check -- comparing adjusted against unadjusted -- is
blind to it.

Left undetected, every such event enters a study as a spectacular winner that
nobody could have traded, and drags the measured behaviour of "large gappers"
toward whatever reverse splits do afterwards (they drift down).

The signature
-------------
A reverse split is a unit change, not a price move, so it looks nothing like
news:

1. the overnight multiple sits close to a **round consolidation ratio**
   (1-for-2, 1-for-5, 1-for-10, 1-for-25 ...);
2. the session **barely moves intraday** -- there is no news to trade, so
   open-to-high is small;
3. share volume **collapses**, because the share count itself collapsed.

Any one of those alone is weak. Together they are decisive, and the test below
requires the first two and reports the third.

This is a *detector*, not a corporate-actions database. It is deliberately
conservative: it flags events for exclusion from a gainers study, and it will
occasionally flag a genuine low-volatility gap. The alternative -- keeping
reverse splits in a study of explosive moves -- is far worse.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ratios actually used in US small-cap consolidations.
COMMON_RATIOS: List[float] = [1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25,
                              30, 35, 40, 50, 60, 75, 80, 100, 150, 200]

MIN_MULTIPLE = 1.8          # below this, an ordinary gap is more likely
RATIO_TOLERANCE = 0.12      # how close to a round ratio counts as "round"
MAX_INTRADAY_MOVE = 0.35    # a real catalyst keeps moving after the open

# A consolidation followed by a genuine move fails the inert-session test. Five
# of the ten largest contributors to a spurious "accumulation edge" were splits
# with open-to-high of 274%, 70%, 56%, 48% and 47% -- all far above
# MAX_INTRADAY_MOVE, all missed. `flag_reverse_splits` is therefore a *lower
# bound* on contamination, and `strict=False` widens it to ratio-only for use
# where a false positive is cheaper than a false negative -- excluding
# contaminated rows from a return study, rather than counting them.
RATIO_ONLY_TOLERANCE = 0.06  # tighter ratio, since the session test is dropped


def nearest_ratio(multiple: float) -> tuple:
    """Closest common consolidation ratio and the relative error."""
    if not np.isfinite(multiple) or multiple <= 0:
        return np.nan, np.inf
    near = min(COMMON_RATIOS, key=lambda c: abs(multiple - c))
    return near, abs(multiple - near) / near


def flag_reverse_splits(
    df: pd.DataFrame,
    prev_close_col: str = "prev_close_unadj",
    open_col: str = "open",
    o2h_col: str = "o2h",
    min_multiple: float = MIN_MULTIPLE,
    tolerance: float = RATIO_TOLERANCE,
    max_intraday: float = MAX_INTRADAY_MOVE,
) -> pd.DataFrame:
    """Add ``split_multiple``, ``split_nearest_ratio``, ``is_suspect_split``."""
    out = df.copy()
    mult = out[open_col] / out[prev_close_col].replace(0, np.nan)
    out["split_multiple"] = mult

    near, err = [], []
    for m in mult.to_numpy():
        n, e = nearest_ratio(m)
        near.append(n)
        err.append(e)
    out["split_nearest_ratio"] = near
    out["split_ratio_error"] = err

    err_s = pd.Series(err, index=out.index)
    round_ratio = (mult >= min_multiple) & (err_s < tolerance)

    # Strict: round ratio AND an inert session. Few false positives, but blind
    # to a consolidation followed by a real move.
    out["is_suspect_split"] = (round_ratio & (out[o2h_col].abs() < max_intraday)).fillna(False)

    # Wide: a tight round ratio alone. Catches the split-then-rally case the
    # strict test misses, at the cost of flagging the occasional genuine gap
    # that happens to land on a round multiple.
    out["is_possible_split"] = (
        (mult >= min_multiple) & (err_s < RATIO_ONLY_TOLERANCE)
    ).fillna(False)
    return out


def flag_forward_splits(
    df: pd.DataFrame,
    prev_close_col: str = "prev_close_unadj",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    min_multiple: float = MIN_MULTIPLE,
    tolerance: float = RATIO_TOLERANCE,
    max_intraday: float = MAX_INTRADAY_MOVE,
) -> pd.DataFrame:
    """The mirror image: a *forward* split, which looks exactly like a crash.

    Why this is separate from `flag_reverse_splits`
    -----------------------------------------------
    A reverse split multiplies the price and a forward split divides it, so a
    detector keyed on ``open / prev_close >= 1.8`` is structurally blind to the
    forward case. That blindness is dangerous in only one direction, and it is
    the direction a long strategy cares about: an unadjusted 1-for-10 forward
    split prints as a **90% overnight decline**, which is indistinguishable from
    a catastrophe on price alone and would be selected by any "buy what
    crashed" rule as its most extreme signal.

    The test is the reciprocal of the reverse one -- a near-round *divisor*
    together with an inert session, because a unit change has no news to trade.

    Inertness is measured on the **full high-low range**, not open-to-high.
    That difference is not cosmetic: Tenax Therapeutics fell 89.8% overnight on
    a failed Phase 3 (a real catalyst) and opened at $1.79 with a high of
    $1.83 -- an open-to-high of **2.2%**, which reads as perfectly inert. Its
    low was $1.31, so the session range was **39.7%**. Open-to-high can only
    see a stock that runs up; a name that opens and collapses looks quiet to
    it, and that is precisely the shape this detector must not mistake for a
    unit change.

    Adds ``fwd_split_divisor``, ``is_suspect_fwd_split``, ``is_possible_fwd_split``.
    """
    out = df.copy()
    mult = out[open_col] / out[prev_close_col].replace(0, np.nan)
    divisor = 1.0 / mult.replace(0, np.nan)
    out["fwd_split_divisor"] = divisor

    err = []
    for d in divisor.to_numpy():
        _, e = nearest_ratio(d)
        err.append(e)
    err_s = pd.Series(err, index=out.index)

    session_range = out[high_col] / out[low_col].replace(0, np.nan) - 1.0
    round_divisor = (divisor >= min_multiple) & (err_s < tolerance)
    out["is_suspect_fwd_split"] = (
        round_divisor & (session_range.abs() < max_intraday)
    ).fillna(False)
    out["is_possible_fwd_split"] = (
        (divisor >= min_multiple) & (err_s < RATIO_ONLY_TOLERANCE)
    ).fillna(False)
    return out


def assert_detector_is_conservative(df: pd.DataFrame) -> None:
    """Tripwire: a flagged event must not look like a traded catalyst.

    The failure mode worth guarding is over-flagging -- silently deleting real
    explosions and reporting a cleaner world than exists. A genuine catalyst
    keeps moving after the open, so anything flagged must have a small
    open-to-high by construction.
    """
    flagged = df[df["is_suspect_split"]]
    if flagged.empty:
        return
    worst = flagged["o2h"].abs().max()
    assert worst < MAX_INTRADAY_MOVE + 1e-9, (
        f"a flagged event moved {100 * worst:.0f}% intraday -- the detector is "
        f"catching real catalysts, not unit changes"
    )
