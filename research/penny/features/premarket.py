"""premarket.py — features from the 04:00-09:34 session.

Why this is the most promising untouched input
----------------------------------------------
For a gap-up microcap the setup forms before the bell. Overnight news lands,
the gap gets built on some amount of real volume, and the move is either
accumulated into the open or distributed into it. Rounds 1-2 saw five minutes
of that (09:25-09:29) and eight aggregates derived from them.

The median symbol-day now has ~96 pre-open bars instead of 5.

Design rule carried over from round 2
-------------------------------------
**Continuous only.** Binary EDGAR flags measured -0.64% against +1.05% for the
same information expressed as continuous recency. Nothing here returns a
boolean; everything is a magnitude, a ratio, or a position within a range.

Leakage boundary
----------------
Every feature is computed from bars strictly before 09:35. The premarket cache
is written with a hard assertion that it contains no bar at or after that
minute, and :func:`assert_no_post_decision_bars` re-checks it at feature-build
time. Two independent guards, because this is the one boundary that would make
every result meaningless.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DECISION_MINUTE = "09:35"

# Session boundaries in extended-hours trading. The early block is thin and
# dominated by news reactions; the 09:00-09:29 block is where the real
# pre-open auction interest accumulates.
EARLY_END = "07:00"
LATE_START = "09:00"


def _minute_to_index(minute: pd.Series) -> pd.Series:
    """HH:MM -> minutes since midnight, for time-weighted statistics."""
    parts = minute.str.split(":", expand=True).astype(int)
    return parts[0] * 60 + parts[1]


def assert_no_post_decision_bars(panel: pd.DataFrame, minute_col: str = "minute_et") -> None:
    """Second, independent check that nothing at/after 09:35 is present."""
    if panel.empty:
        return
    late = panel[panel[minute_col] >= DECISION_MINUTE]
    assert late.empty, (
        f"{len(late)} premarket bars at or after {DECISION_MINUTE}; "
        "the premarket feature set would be looking at the entry bar"
    )


def premarket_features(panel: pd.DataFrame, prev_close: Optional[pd.Series] = None) -> pd.DataFrame:
    """Per-symbol features from one session's pre-open bars.

    ``panel`` holds 04:00-09:34 bars for one session, many symbols.
    """
    if panel.empty:
        return pd.DataFrame()

    assert_no_post_decision_bars(panel)

    df = panel.sort_values(["symbol", "minute_et"]).copy()
    df["minute_idx"] = _minute_to_index(df["minute_et"])
    df["dollar_volume"] = df["close"] * df["volume"]

    rows: List[Dict] = []
    for symbol, g in df.groupby("symbol", sort=False):
        vol = g["volume"].to_numpy(dtype=float)
        dv = g["dollar_volume"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        idx = g["minute_idx"].to_numpy(dtype=float)

        total_dv = float(np.nansum(dv))
        total_vol = float(np.nansum(vol))
        vwap = total_dv / total_vol if total_vol > 0 else np.nan

        last = float(close[-1])
        pm_high, pm_low = float(np.nanmax(high)), float(np.nanmin(low))

        # Where in its own pre-open range the stock sits at 09:34. Near the
        # high is accumulation into the open; near the low is distribution.
        span = pm_high - pm_low
        close_position = (last - pm_low) / span if span > 0 else np.nan

        # Volume concentration: is the pre-open volume spread across the
        # session or crammed into the last half hour?
        late_mask = idx >= (9 * 60)
        late_dv = float(np.nansum(dv[late_mask]))
        early_dv = float(np.nansum(dv[idx < (7 * 60)]))

        # Time-weighted centre of volume. A high value means the interest
        # arrived late; a low value means it came on overnight news and faded.
        vol_centroid = float(np.nansum(idx * dv) / total_dv) if total_dv > 0 else np.nan

        # Trend of the last 30 pre-open minutes, normalised by price: is the
        # move still building into the bell or rolling over?
        tail = close[idx >= (9 * 60 + 5)]
        if len(tail) >= 3 and tail[0] > 0:
            late_drift = float(tail[-1] / tail[0] - 1.0)
        else:
            late_drift = np.nan

        rows.append(
            {
                "symbol": symbol,
                "pm_dollar_volume": total_dv,
                "pm_volume": total_vol,
                "pm_bars": int(len(g)),
                "pm_vwap": vwap,
                "pm_close": last,
                "pm_high": pm_high,
                "pm_low": pm_low,
                "pm_range": (pm_high - pm_low) / last if last > 0 else np.nan,
                "pm_close_position": close_position,
                "pm_dist_from_vwap": (last / vwap - 1.0) if vwap and vwap > 0 else np.nan,
                "pm_late_dv_share": late_dv / total_dv if total_dv > 0 else np.nan,
                "pm_early_dv_share": early_dv / total_dv if total_dv > 0 else np.nan,
                "pm_vol_centroid": vol_centroid,
                "pm_late_drift": late_drift,
                "pm_first_minute": float(idx[0]),
                "pm_minutes_traded": float(len(np.unique(idx))),
            }
        )

    out = pd.DataFrame(rows)

    if prev_close is not None and not out.empty:
        pc = prev_close.reindex(out["symbol"]).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            # The gap as it stood BEFORE the open -- distinct from the official
            # opening gap, and knowable earlier.
            out["pm_gap"] = out["pm_close"] / pc - 1.0
            out["pm_high_gap"] = out["pm_high"] / pc - 1.0
            # How much of the pre-open advance was given back by 09:34.
            out["pm_fade_from_high"] = out["pm_close"] / out["pm_high"] - 1.0
    return out


def add_relative_features(
    features: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    """Scale pre-open activity by the stock's own trailing norms.

    Absolute pre-open volume is not comparable across a $300k-ADV name and a
    $20m-ADV one. The ratio to trailing ADV is, and it is the classic microcap
    "in play" signal.
    """
    out = features.merge(
        daily[["symbol", "date", "adv_20", "prev_close"]],
        on=["symbol", "date"], how="left",
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pm_dv_to_adv"] = out["pm_dollar_volume"] / out["adv_20"].replace(0, np.nan)
        out["pm_rvol"] = np.log1p(out["pm_dv_to_adv"].clip(lower=0))
    return out.drop(columns=["adv_20", "prev_close"], errors="ignore")


PREMARKET_FEATURES: List[str] = [
    "pm_dollar_volume", "pm_bars", "pm_range", "pm_close_position",
    "pm_dist_from_vwap", "pm_late_dv_share", "pm_early_dv_share",
    "pm_vol_centroid", "pm_late_drift", "pm_first_minute", "pm_minutes_traded",
    "pm_gap", "pm_high_gap", "pm_fade_from_high", "pm_dv_to_adv", "pm_rvol",
]

# Ranked within the session: being the most-traded name pre-open is a
# different fact from trading $2m pre-open.
PREMARKET_RANK_FEATURES: List[str] = [
    "pm_dollar_volume", "pm_dv_to_adv", "pm_gap", "pm_range",
]
