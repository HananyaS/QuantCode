"""screens.py — make §72's bug impossible to write again.

What went wrong
---------------
§65 screened its entry price band on `close` -- the **split-adjusted** series.
Adjusted prices are computed backwards from corporate actions that happen
later, so a stock printing $0.60 today that reverse-splits 10-for-1 next month
carries an adjusted price of $6.00 **today**. The band therefore excluded every
name that would go on to consolidate, and companies consolidate precisely
because their price keeps collapsing. The filter was a survivorship screen built
from future information, it inflated a −4.00% strategy to +4.97%, and it
survived seven subsequent sections of audit because the bias sat *upstream* of
every test.

That was the fourth adjusted/unadjusted failure in this project (§57, §64, §67,
§72). Per-analysis vigilance has now failed four times, so this module replaces
vigilance with an exception.

The rule
--------
**Adjusted prices are for returns. Filters use the tape.**

A *ratio* within a split-free window is adjustment-invariant, so returns and
intraday ratios may use either series. A *level* -- a price band, a market-cap
band, a distance from a moving average or a 52-week high -- must come from the
unadjusted print, because the adjusted level encodes every future action.

    from research.penny.etl.screens import assert_unadjusted, price_band
    mask = price_band(u, 0.50, 2.00)          # raises unless the column is _unadj
    assert_unadjusted(["close_unadj", "adv20"])
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns carrying an ADJUSTED price level. Screening on any of these conditions
# on information that did not exist at decision time.
ADJUSTED_LEVEL_COLUMNS = frozenset({
    "open", "high", "low", "close", "prev_close", "vwap",
    "close_5b", "close_10b", "hi20", "lo20",
})

# Columns that are unadjusted levels, or are ratios/derived quantities where the
# adjustment factor cancels within a split-free window.
_SAFE_SUFFIXES = ("_unadj",)
_SAFE_EXACT = frozenset({
    "symbol", "date", "volume", "prev_volume", "dv", "adv20",
    "gap", "c2c", "o2h", "o2c", "l2h", "overnight",          # intraday ratios
    "any_split", "is_suspect_split", "is_possible_split",
    "is_suspect_fwd_split", "is_possible_fwd_split",
    "split_multiple", "split_nearest_ratio", "split_ratio_error",
    "fwd_split_divisor", "cost", "vol_shock", "dv_shock",
})


class AdjustedPriceInScreenError(AssertionError):
    """Raised when a selection filter references an adjusted price level."""


def is_safe_for_screen(col: str) -> bool:
    if col in _SAFE_EXACT:
        return True
    if any(col.endswith(s) for s in _SAFE_SUFFIXES):
        return True
    if col in ADJUSTED_LEVEL_COLUMNS:
        return False
    # Returns over a window are ratios -- adjustment-invariant.
    if col.startswith("ret_") or col.startswith("fwd_"):
        return True
    return True  # unknown derived column: allowed, but see assert_unadjusted


def assert_unadjusted(columns: Iterable[str]) -> None:
    """Raise if any column is an adjusted price *level*.

    Deliberately loud. A screen that trips this is not a style problem -- it is
    the exact defect that invalidated a strategy and seven sections of audit.
    """
    bad = sorted({c for c in columns if c in ADJUSTED_LEVEL_COLUMNS})
    if bad:
        raise AdjustedPriceInScreenError(
            f"selection filter references adjusted price level(s) {bad}. "
            f"Adjusted prices are computed from FUTURE corporate actions, so "
            f"screening on them conditions on information that did not exist "
            f"at decision time (see LOG §72). Use the *_unadj column instead."
        )


def price_band(df: pd.DataFrame, lo: float, hi: float,
               col: str = "close_unadj") -> np.ndarray:
    """Boolean mask for `lo <= price < hi`, unadjusted only."""
    assert_unadjusted([col])
    if col not in df.columns:
        raise KeyError(f"{col!r} not in frame; available: {sorted(df.columns)[:12]}...")
    return ((df[col] >= lo) & (df[col] < hi)).fillna(False).to_numpy()


def adjustment_tripwire(df: pd.DataFrame, mask: np.ndarray,
                        adj_col: str = "close", unadj_col: str = "close_unadj",
                        warn_share: float = 0.05) -> dict:
    """Report how far the two price series diverge on a candidate set.

    This is the one-line check that would have caught §72 the day it was
    written: on the retracted sample, **46.8%** of candidate rows had a ratio
    above 1.5 and one had a ratio of 1,000,000. A material share means a future
    corporate action is leaking into the selection.
    """
    sub = df.loc[mask] if mask.dtype == bool else df.iloc[mask]
    if len(sub) == 0:
        return {"n": 0, "median_ratio": float("nan"), "share_gt_1_5": 0.0}
    ratio = (sub[adj_col] / sub[unadj_col]).replace([np.inf, -np.inf], np.nan).dropna()
    share = float((ratio > 1.5).mean()) if len(ratio) else 0.0
    out = {"n": int(len(sub)), "median_ratio": float(ratio.median()) if len(ratio) else float("nan"),
           "share_gt_1_5": share, "max_ratio": float(ratio.max()) if len(ratio) else float("nan")}
    if share > warn_share:
        logger.warning(
            "adjustment tripwire: %.1f%% of %d selected rows have adj/unadj > 1.5 "
            "(max %.0fx). Two readings: (a) if any filter used an ADJUSTED level, "
            "that share was wrongly excluded and the result is §72 all over again; "
            "(b) if filters are clean, the population is simply dominated by names "
            "that later reverse-split -- i.e. terminal decliners. Check which.",
            100 * share, len(sub), out["max_ratio"])
    return out
