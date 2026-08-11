"""barrier_free.py — path statistics measured over the WHOLE session.

Why this exists
---------------
Round 1 measured maximum favourable excursion only up to the point a barrier
fired. That is the right quantity for describing a *trade*, and the wrong one
for describing an *opportunity* -- and the two were conflated. With a +10%
target, MFE can never exceed ~10%, so the truncated statistic reported
``P(MFE >= 20%) = 0.004`` when the true figure is **0.199**.

The distinction matters because it changes the diagnosis. Truncated excursions
say "the price never goes anywhere"; barrier-free excursions say "the price
goes a long way, and the exit rule threw it away". Only the second is true.

Everything here is measured from the same 09:35 entry as
:mod:`research.penny.labels.triple_barrier`, over 09:36-15:55, with **no
barriers of any kind**. These are descriptive and model-target quantities, not
tradable returns: ``ret_c2c`` assumes a position held blindly to the close,
which no one would do, and it carries no execution cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from research.penny.labels.triple_barrier import (
    DECISION_MINUTE,
    VERTICAL_MINUTE,
)

logger = logging.getLogger(__name__)


@dataclass
class PathStats:
    """Barrier-free description of one symbol-day's intraday path."""

    symbol: str
    date: pd.Timestamp
    entry_price: float
    ret_c2c: float          # 09:35 -> 15:55, held blindly, gross
    mfe_full: float         # best unrealised gain at any point after entry
    mae_full: float         # worst unrealised loss at any point after entry
    minutes_to_mfe: int     # how long the best moment took to arrive
    minutes_to_mae: int
    n_scan_minutes: int
    mfe_before_mae: bool    # did the high come before the low?


def path_stats_one(
    symbol: str,
    day: pd.Timestamp,
    minute_path: pd.DataFrame,
    decision_minute: str = DECISION_MINUTE,
    vertical_minute: str = VERTICAL_MINUTE,
) -> Optional[PathStats]:
    """Barrier-free statistics for one symbol-day, or None if unenterable."""
    path = minute_path.sort_values("minute_et")

    entry_bar = path[path["minute_et"] == decision_minute]
    if entry_bar.empty:
        return None

    entry = float(entry_bar.iloc[0]["open"])
    if not np.isfinite(entry) or entry <= 0:
        return None

    scan = path[
        (path["minute_et"] > decision_minute) & (path["minute_et"] <= vertical_minute)
    ]
    if scan.empty:
        return None

    highs = scan["high"].to_numpy(dtype=float)
    lows = scan["low"].to_numpy(dtype=float)

    i_hi = int(np.nanargmax(highs))
    i_lo = int(np.nanargmin(lows))

    return PathStats(
        symbol=symbol,
        date=day,
        entry_price=entry,
        ret_c2c=float(scan.iloc[-1]["close"]) / entry - 1.0,
        mfe_full=float(highs[i_hi]) / entry - 1.0,
        mae_full=float(lows[i_lo]) / entry - 1.0,
        minutes_to_mfe=i_hi + 1,
        minutes_to_mae=i_lo + 1,
        n_scan_minutes=len(scan),
        mfe_before_mae=bool(i_hi < i_lo),
    )


def path_stats_day(minute_panel: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Barrier-free statistics for every symbol in one session."""
    out: List[PathStats] = []
    for symbol, path in minute_panel.groupby("symbol", sort=False):
        stats = path_stats_one(symbol, day, path)
        if stats is not None:
            out.append(stats)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame([s.__dict__ for s in out])


def meta_label(
    ret: pd.Series, cost: float, direction: str = "long"
) -> pd.Series:
    """Binary meta-label: would this trade have cleared costs?

    Meta-labelling (Lopez de Prado 2018, ch. 3) splits the problem in two: a
    primary rule decides *what* to consider (here, the candidate screen), and a
    secondary model decides *whether to act*. The secondary model's target is
    not the return but this binary -- which is what makes it a precision
    problem rather than a forecasting one.

    The threshold is the cost, not zero. A trade that returns +20 bps gross is
    a losing trade once 437 bps of friction is paid, and labelling it a success
    would train the model to chase exactly the trades that cannot survive
    execution.
    """
    if direction != "long":
        raise ValueError(f"only long is in scope for this study, got {direction!r}")
    return (ret > cost).astype(int)


def assert_dominates_truncated(
    full: pd.Series, truncated: pd.Series, tolerance: float = 1e-9
) -> None:
    """Barrier-free MFE must never be below the barrier-truncated MFE.

    A sanity check with real force: the truncated statistic is a prefix of the
    full one, so it can only be smaller. If this fails, the two are being
    computed over different windows or joined on the wrong key -- the exact
    class of bug that produced the round-1 artifact.
    """
    both = pd.concat([full, truncated], axis=1).dropna()
    if both.empty:
        return
    violations = (both.iloc[:, 0] < both.iloc[:, 1] - tolerance).sum()
    assert violations == 0, (
        f"{violations} rows where barrier-free MFE < truncated MFE -- the two "
        "are not measured over the same path"
    )
