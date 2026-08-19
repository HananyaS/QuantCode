"""cost_model.py — round-trip cost as a function of the DOLLAR position size.

Why this replaces the flat schedule
------------------------------------
Every round so far charged a flat round trip by ADV band (150 bps at
$250k-$1M), measured at roughly $2,000 orders. Two things were wrong with using
it here.

**1. It was charged at half.** `COST_BPS` is a round trip, and the slot
simulations charged `0.5 * cost` -- one side. A long-only strategy that buys and
later sells pays both.

**2. Cost depends on position size, and these strategies do not trade $2,000.**
The measured NBBO gives a median inside-quote notional of **$1,900**, and the
round-4 model is

    round_trip = 2 * half_spread * (1 + max(order/inside - 1, 0))

so cost is **flat below the inside quote and linear above it**:

    $100 -> 76 bps      $1,000 -> 83 bps     $5,000 -> 199 bps
    $250 -> 76 bps      $2,000 -> 100 bps   $25,000 -> 873 bps

The two errors nearly cancel at $250 a position (75 charged, 76 true) and do
not at $3,333 (75 charged, 144 true). So the flat schedule flattered exactly the
**low**-position-count configurations, which are the ones being argued for.

The consequence is the opposite of the intuition: **more positions is cheaper**,
because a fixed account split more ways puts each order further inside the
quote. Position count is therefore an operational constraint, not a cost one.

`half_spread` and `inside` are fitted from the 9,794 measured symbol-days as
medians by ADV decile, then applied to every row by its own ADV.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd

from research.penny.etl import bars

logger = logging.getLogger(__name__)

_N_BINS = 10


@lru_cache(maxsize=1)
def _fit() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median half-spread (bps) and inside-quote notional ($) by ADV decile."""
    d = pd.read_parquet(bars.DATA_ROOT / "cost_table.parquet",
                        columns=["adv_20", "half_spread_bps", "ask_notional"])
    d = d.dropna()
    d = d[(d["adv_20"] > 0) & (d["ask_notional"] > 0) & (d["half_spread_bps"] > 0)]
    d["bin"] = pd.qcut(np.log10(d["adv_20"]), _N_BINS, labels=False, duplicates="drop")
    g = d.groupby("bin").agg(edge=("adv_20", "max"),
                             hs=("half_spread_bps", "median"),
                             inside=("ask_notional", "median")).sort_values("edge")
    return (g["edge"].to_numpy(), g["hs"].to_numpy(), g["inside"].to_numpy())


def round_trip_bps(adv: np.ndarray, position_usd: np.ndarray | float) -> np.ndarray:
    """Round-trip cost in basis points for an order of `position_usd`.

    Flat below the inside quote, linear above it. Both sides charged.
    """
    edges, hs, inside = _fit()
    idx = np.searchsorted(edges, np.asarray(adv, dtype=float), side="left")
    idx = np.clip(idx, 0, len(edges) - 1)
    half = hs[idx]
    ins = inside[idx]
    pos = np.asarray(position_usd, dtype=float)
    shortfall = np.maximum(pos / np.maximum(ins, 1.0) - 1.0, 0.0)
    return 2.0 * half * (1.0 + shortfall)


def round_trip_frac(adv: np.ndarray, position_usd: np.ndarray | float) -> np.ndarray:
    return round_trip_bps(adv, position_usd) / 1e4


def describe() -> pd.DataFrame:
    edges, hs, inside = _fit()
    return pd.DataFrame({"adv_upper": edges, "half_spread_bps": hs,
                         "inside_notional": inside})
