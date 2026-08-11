"""quotes.py — measured execution costs from NBBO quotes.

Why this replaces two invented constants
----------------------------------------
Rounds 1 and 2 charged every trade ``2 x (118 bps half-spread + 100 bps
impact) = 437 bps``. Neither number was measured. The half-spread was inferred
from the minute high-low range with a floor; the impact term was a bare
coefficient with no volatility scaling at all.

A probe found measured half-spreads of 24-26 bps on real names at 09:35 --
roughly a fifth of the assumption. Since round 2's best model reached +1.51%
gross, the cost bar rather than the model may be what has been failing the
hypothesis. This module measures it.

What actually costs money
-------------------------
Two distinct things, and the old model conflated them:

1. **Crossing the spread.** Directly observable from the NBBO at the decision
   instant. Half the quoted spread, paid on entry and again on exit.
2. **Walking the book.** If the resting size at the inside is smaller than the
   order, the remainder fills worse. This is *not* captured by the spread, and
   in microcaps it dominates: a probe showed 22,300 shares at PLUG's inside
   versus 400 at NVAX's. A $10k order is free at one and walks the book at the
   other.

Impact is therefore modelled in the standard concave form

    impact_bps = Y * sigma_daily * sqrt(participation) * 10_000

(Almgren et al. 2005; Torre & Ferrari 1997), which scales with volatility --
the property the old constant lacked entirely, and the reason it was wrong in
both directions depending on the name.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUEST_PAUSE = 0.05

# Quote conditions that do not represent a firm, executable two-sided market.
# Crossed or locked books and zero-size quotes are artefacts, not prices.
MIN_PRICE = 0.01


@dataclass
class CostModel:
    """Measured execution costs, replacing the assumed constants."""

    half_spread_bps: float
    impact_bps: float
    depth_shortfall_bps: float = 0.0

    @property
    def round_trip_bps(self) -> float:
        return 2 * (self.half_spread_bps + self.impact_bps) + self.depth_shortfall_bps

    @property
    def round_trip(self) -> float:
        return self.round_trip_bps / 10_000


def fetch_quotes_for_minute(
    client,
    symbols: Sequence[str],
    day: str,
    minute: str = "09:35",
    window_seconds: int = 60,
) -> pd.DataFrame:
    """Every NBBO quote for these symbols inside one minute.

    Grouped by session like the bar fetch, so the whole measurement costs about
    one request per trading day rather than one per symbol-day.
    """
    from alpaca.data.requests import StockQuotesRequest

    start = pd.Timestamp(f"{day} {minute}", tz="America/New_York")
    end = start + pd.Timedelta(seconds=window_seconds)

    time.sleep(REQUEST_PAUSE)
    df = client.get_stock_quotes(
        StockQuotesRequest(
            symbol_or_symbols=list(symbols),
            start=start.tz_convert("UTC").to_pydatetime(),
            end=end.tz_convert("UTC").to_pydatetime(),
            feed="sip",
        )
    ).df

    if df.empty:
        return pd.DataFrame()

    out = df.reset_index()
    out["date"] = pd.Timestamp(day)
    return out


def summarise_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Per symbol-day half-spread and depth at the inside.

    Uses the **median** quote in the window rather than the first. Quotes at the
    open are noisy and frequently momentarily crossed or one-sided; a single
    snapshot would inherit that noise, and the first quote in particular tends
    to be the widest of the minute.
    """
    if quotes.empty:
        return pd.DataFrame()

    q = quotes.copy()
    for col in ("bid_price", "ask_price", "bid_size", "ask_size"):
        if col not in q.columns:
            raise KeyError(f"quote frame missing {col!r}")

    # Discard non-markets: zero/absent sides, crossed or locked books.
    valid = (
        (q["bid_price"] > MIN_PRICE)
        & (q["ask_price"] > MIN_PRICE)
        & (q["ask_price"] > q["bid_price"])
        & (q["bid_size"] > 0)
        & (q["ask_size"] > 0)
    )
    q = q[valid]
    if q.empty:
        return pd.DataFrame()

    q["mid"] = (q["bid_price"] + q["ask_price"]) / 2.0
    q["half_spread_bps"] = (q["ask_price"] - q["bid_price"]) / (2 * q["mid"]) * 10_000
    q["ask_notional"] = q["ask_price"] * q["ask_size"] * 100  # sizes are in round lots

    grouped = q.groupby(["symbol", "date"])
    out = grouped.agg(
        half_spread_bps=("half_spread_bps", "median"),
        half_spread_bps_p90=("half_spread_bps", lambda s: float(np.percentile(s, 90))),
        mid=("mid", "median"),
        ask_notional=("ask_notional", "median"),
        n_quotes=("half_spread_bps", "size"),
    ).reset_index()
    return out


def impact_bps(
    volatility_daily: pd.Series,
    participation: pd.Series,
    coefficient: float = 1.0,
) -> pd.Series:
    """Concave market impact, scaled by volatility.

    ``Y * sigma * sqrt(participation)``. The old model used a bare constant
    with no sigma term, which is wrong in both directions: it over-charged
    quiet names and under-charged violent ones, and the population here spans
    both.
    """
    sigma = pd.Series(volatility_daily).astype(float).clip(lower=0.0)
    part = pd.Series(participation).astype(float).clip(lower=0.0, upper=1.0)
    return coefficient * sigma * np.sqrt(part) * 10_000


def depth_shortfall_bps(
    order_notional: pd.Series,
    inside_notional: pd.Series,
    half_spread_bps: pd.Series,
    levels_penalty: float = 1.0,
) -> pd.Series:
    """Extra cost when the order is larger than the resting size at the inside.

    A deliberately crude model, and crude in the conservative direction: the
    portion that cannot fill at the inside is charged one additional
    half-spread per notional-multiple beyond it. Without free depth-of-book
    data the true shape is unknowable, but ignoring it entirely -- which is
    what rounds 1 and 2 did -- assumes infinite depth, and a 400-share inside
    quote says that assumption fails badly in this universe.
    """
    order = pd.Series(order_notional).astype(float)
    inside = pd.Series(inside_notional).astype(float).replace(0, np.nan)
    excess_multiple = (order / inside - 1.0).clip(lower=0.0)
    return (excess_multiple * levels_penalty * pd.Series(half_spread_bps).astype(float)).fillna(0.0)


def build_cost_table(
    quote_summary: pd.DataFrame,
    trades: pd.DataFrame,
    participation_cap: float = 0.01,
    impact_coefficient: float = 1.0,
) -> pd.DataFrame:
    """Per-trade measured cost, replacing the flat 437 bps."""
    df = trades.merge(quote_summary, on=["symbol", "date"], how="left", suffixes=("", "_q"))

    df["order_notional"] = df["adv_20"].astype(float) * participation_cap
    df["impact_bps"] = impact_bps(
        df["vol_20d"], pd.Series(participation_cap, index=df.index), impact_coefficient
    )
    df["depth_shortfall_bps"] = depth_shortfall_bps(
        df["order_notional"], df["ask_notional"], df["half_spread_bps"]
    )
    df["measured_round_trip_bps"] = (
        2 * (df["half_spread_bps"] + df["impact_bps"]) + df["depth_shortfall_bps"]
    )
    df["measured_cost"] = df["measured_round_trip_bps"] / 10_000
    return df


def bucket_summary(cost_table: pd.DataFrame, by: str, bins: Sequence[float]) -> pd.DataFrame:
    """Cost by price or liquidity bucket.

    Reported because a single average cost is misleading when the population
    spans $0.50 names quoting 200 bps wide and $9 names quoting 20.
    """
    df = cost_table.dropna(subset=["half_spread_bps", by]).copy()
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = pd.cut(df[by], bins=bins)
    return (
        df.groupby("bucket", observed=True)
        .agg(
            n=("half_spread_bps", "size"),
            half_spread_bps=("half_spread_bps", "median"),
            impact_bps=("impact_bps", "median"),
            depth_shortfall_bps=("depth_shortfall_bps", "median"),
            round_trip_bps=("measured_round_trip_bps", "median"),
        )
        .reset_index()
    )
