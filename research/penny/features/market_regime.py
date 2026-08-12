"""market_regime.py — market state and idiosyncrasy, from the universe itself.

Why not SPY
-----------
The daily cache holds no index symbols: the universe fetch was microcap-focused,
so for four rounds no market-regime feature was even available. Rather than add
a dependency, the market factor is computed **from the studied universe** — the
cross-section of liquid low-priced names. For this segment that is the more
relevant benchmark anyway: microcaps do not track the S&P, and what matters to a
reversal trade is whether the whole low-price complex is selling off or only
this name.

The feature that matters most
-----------------------------
``gap_vs_market`` separates an **idiosyncratic** gap-down from a **systematic**
one. A name down 6% on a day when the whole complex is down 5% has not really
been sold off; a name down 6% when the complex is flat has. Short-term reversal
is compensation for absorbing idiosyncratic selling pressure (Nagel 2012), so
the distinction is the mechanism, not a nicety.

Point-in-time
-------------
Two classes of quantity, and conflating them is the failure to avoid:

* **Trailing market state** uses sessions strictly before the candidate date.
  Shifted by one day, always.
* **Same-day open state** (market gap, breadth of gaps) is known at 09:30, an
  hour before the 09:35 decision, and is used unshifted. It is computed from
  *opening prints only* — never from any later bar.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_FEATURES: List[str] = [
    "mkt_ret_1d_prev",
    "mkt_ret_5d_prev",
    "mkt_vol_20d_prev",
    "mkt_breadth_prev",
    "mkt_drawdown_20d_prev",
    "mkt_gap_today",
    "mkt_gap_breadth_today",
    "gap_vs_market",
    "rel_strength_5d",
    "rel_strength_20d",
]


def build_market_factor(panel: pd.DataFrame, min_names: int = 30) -> pd.DataFrame:
    """One row per session describing the low-price complex.

    ``panel`` is the *universe* -- every liquid low-priced symbol-day, not just
    the candidates. Using the candidate set instead would define the market by
    the very selection under test.
    """
    need = {"date", "daily_ret", "gap"}
    missing = need - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing {sorted(missing)}")

    g = panel.groupby("date")
    mkt = g.agg(
        mkt_ret=("daily_ret", "median"),
        mkt_gap=("gap", "median"),
        n_names=("daily_ret", "size"),
        breadth=("daily_ret", lambda s: float((s > 0).mean())),
        gap_breadth=("gap", lambda s: float((s > 0).mean())),
    ).reset_index()

    # A handful of names is not a market. Blanking is honest; interpolating
    # would invent a factor on exactly the days it is least knowable.
    thin = mkt["n_names"] < min_names
    if thin.any():
        logger.info("market factor: %d of %d sessions below %d names -- blanked",
                    int(thin.sum()), len(mkt), min_names)
        mkt.loc[thin, ["mkt_ret", "mkt_gap", "breadth", "gap_breadth"]] = np.nan

    mkt = mkt.sort_values("date").reset_index(drop=True)

    # Trailing state: strictly prior sessions. The shift is the guarantee.
    prev = mkt["mkt_ret"].shift(1)
    mkt["mkt_ret_1d_prev"] = prev
    mkt["mkt_ret_5d_prev"] = prev.rolling(5, min_periods=3).sum()
    mkt["mkt_vol_20d_prev"] = prev.rolling(20, min_periods=10).std()
    mkt["mkt_breadth_prev"] = mkt["breadth"].shift(1)

    level = (1.0 + prev.fillna(0.0)).cumprod()
    mkt["mkt_drawdown_20d_prev"] = level / level.rolling(20, min_periods=5).max() - 1.0

    # Same-day, open only: known at 09:30, before the 09:35 decision.
    mkt["mkt_gap_today"] = mkt["mkt_gap"]
    mkt["mkt_gap_breadth_today"] = mkt["gap_breadth"]

    return mkt[["date"] + [c for c in REGIME_FEATURES if c in mkt.columns]
               + ["n_names"]]


def attach_regime(
    candidates: pd.DataFrame,
    market: pd.DataFrame,
    ret_5d_col: str = "ret_5d_prev",
    ret_20d_col: Optional[str] = None,
) -> pd.DataFrame:
    """Join the market factor and derive the idiosyncrasy features."""
    before = len(candidates)
    out = candidates.merge(market, on="date", how="left")
    assert len(out) == before, "regime merge changed the row count"

    # The mechanism feature: how much of this gap is the name, not the complex.
    out["gap_vs_market"] = out["gap"] - out["mkt_gap_today"]

    if ret_5d_col in out.columns:
        out["rel_strength_5d"] = out[ret_5d_col] - out["mkt_ret_5d_prev"]
    else:
        out["rel_strength_5d"] = np.nan

    if ret_20d_col and ret_20d_col in out.columns:
        out["rel_strength_20d"] = out[ret_20d_col] - 20 * out["mkt_ret_1d_prev"]
    else:
        out["rel_strength_20d"] = np.nan

    return out


def assert_regime_is_causal(market: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Tripwire: trailing market state must not equal the same session's value.

    The failure this guards is a missing ``shift(1)``, which would hand the
    model the market's own return for the day it is trying to trade -- the most
    flattering leak available, and invisible in any single number.
    """
    m = market.dropna(subset=["mkt_ret_1d_prev"]).sort_values("date")
    if len(m) < 3:
        return

    same_day = panel.groupby("date")["daily_ret"].median().rename("realised")
    joined = m.set_index("date").join(same_day, how="inner").dropna(
        subset=["mkt_ret_1d_prev", "realised"])
    if joined.empty:
        return

    identical = np.isclose(joined["mkt_ret_1d_prev"], joined["realised"], atol=1e-12)
    share = float(identical.mean())
    assert share < 0.20, (
        f"mkt_ret_1d_prev equals the same session's market return on "
        f"{100 * share:.0f}% of days -- the one-day shift is missing"
    )
