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

# Round-4 quote windows (America/New_York). See PREREGISTRATION_R4.md section 4.
#
# The distinction these three constants encode is the one round 3 lacked. Cost
# was evaluation-only there, so measuring it in the entry minute was harmless.
# The moment cost informs the decision -- which is the whole of round 4 -- a
# quantity measured at 09:35 is a quantity from after the entry instant, and
# using it to decide is look-ahead.
DECISION_MINUTE = "09:34"  # cost_hat: what was knowable before entering
ENTRY_MINUTE = "09:35"  # cost_realized, entry leg: what was actually paid
EXIT_MINUTE = "15:55"  # cost_realized, exit leg: measured, not assumed


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
    # The exit sells into the *bid*, so the exit leg's depth is bounded by the
    # bid book, not the ask. Round 3 measured only the ask and then charged the
    # exit as if the book were infinite; a long round trip touches both sides.
    q["bid_notional"] = q["bid_price"] * q["bid_size"] * 100

    grouped = q.groupby(["symbol", "date"])
    out = grouped.agg(
        half_spread_bps=("half_spread_bps", "median"),
        half_spread_bps_p90=("half_spread_bps", lambda s: float(np.percentile(s, 90))),
        mid=("mid", "median"),
        ask_notional=("ask_notional", "median"),
        bid_notional=("bid_notional", "median"),
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


def leg_cost_bps(
    order_notional: pd.Series,
    side_notional: pd.Series,
    half_spread_bps: pd.Series,
    volatility_daily: pd.Series,
    participation: pd.Series,
    impact_coefficient: float = 1.0,
    levels_penalty: float = 1.0,
) -> pd.Series:
    """Cost of **one** leg, charged against that leg's own measured book.

    Round 3 charged ``2 * (half_spread + impact) + depth`` -- depth once, for a
    round trip that crosses the book twice. That silently assumed the 15:55 exit
    fills entirely at the inside, while depth was 398 of the measured 728 bps,
    i.e. the single largest term was half-counted.

    Splitting the round trip into two independently-measured legs also fixes a
    second asymmetry: entry lifts the **ask** and exit hits the **bid**, and in
    this universe those two books are not the same size.
    """
    spread = pd.Series(half_spread_bps).astype(float)
    impact = impact_bps(volatility_daily, participation, impact_coefficient)
    depth = depth_shortfall_bps(order_notional, side_notional, spread, levels_penalty)
    return spread + impact + depth


def round_trip_cost_bps(
    order_notional: pd.Series,
    entry: pd.DataFrame,
    exit_: pd.DataFrame,
    volatility_daily: pd.Series,
    participation: pd.Series,
    impact_coefficient: float = 1.0,
    levels_penalty: float = 1.0,
) -> pd.DataFrame:
    """Fully-charged round trip: entry leg on the ask, exit leg on the bid.

    ``entry`` and ``exit_`` are aligned frames carrying ``half_spread_bps`` plus
    ``ask_notional`` and ``bid_notional`` respectively. Returns the two legs and
    their total so the split stays auditable -- if the exit turns out to be the
    expensive half, that should be visible rather than buried in one number.
    """
    order = pd.Series(order_notional).astype(float)

    entry_bps = leg_cost_bps(
        order, entry["ask_notional"], entry["half_spread_bps"],
        volatility_daily, participation, impact_coefficient, levels_penalty,
    )
    exit_bps = leg_cost_bps(
        order, exit_["bid_notional"], exit_["half_spread_bps"],
        volatility_daily, participation, impact_coefficient, levels_penalty,
    )
    return pd.DataFrame(
        {
            "entry_leg_bps": entry_bps,
            "exit_leg_bps": exit_bps,
            "round_trip_bps": entry_bps + exit_bps,
        },
        index=order.index,
    )


def trim_to_before(
    quotes: pd.DataFrame,
    minute: str = ENTRY_MINUTE,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Drop quotes at or after ``minute``, per session.

    Alpaca's ``end`` is an inclusive instant, not an exclusive bound -- the same
    trap that made rounds 1-2 silently drop each target day's own bar. Asking
    for ``[09:34:00, 09:35:00)`` therefore returns the quote stamped exactly
    09:35:00 when one exists, which is a quote from the entry instant itself.

    Trimming here and asserting afterwards is deliberate: the assertion then
    verifies the trim rather than merely restating the request.
    """
    if quotes.empty or timestamp_col not in quotes.columns:
        return quotes

    ts = pd.to_datetime(quotes[timestamp_col], utc=True).dt.tz_convert("America/New_York")
    cutoff = pd.to_datetime(ts.dt.date.astype(str) + f" {minute}").dt.tz_localize(
        "America/New_York"
    )
    return quotes[(ts < cutoff).to_numpy()]


def attach_costs(
    trades: pd.DataFrame,
    windows: pd.DataFrame,
    order_notional: float,
    impact_coefficient: float = 1.0,
    levels_penalty: float = 1.0,
) -> pd.DataFrame:
    """Join measured quotes and produce both ``cost_hat`` and ``cost_realized``.

    ``windows`` is the wide frame from ``measure_costs_r4.py``: one row per
    symbol-day carrying ``half_spread_bps_{decision,entry,exit}`` and the
    matching book sizes.

    Two costs come out, and keeping them distinct is the point of the round:

    ``cost_hat_bps``
        What the 09:34 book implied the round trip would cost. A decision input.
        Both legs are estimated from the decision book because that is all a
        decision at 09:35 could have known.
    ``cost_realized_bps``
        What it actually cost: entry leg against the 09:35 ask, exit leg against
        the 15:55 bid. Charged in evaluation, never an input.
    """
    df = trades.merge(windows, on=["symbol", "date"], how="left", validate="one_to_one")
    order = pd.Series(float(order_notional), index=df.index)
    part = (order / df["adv_20"].astype(float)).clip(0.0, 1.0)
    sigma = df["vol_20d"].astype(float)

    # --- decision-time estimate (09:34 book on both legs) --------------------
    hat_entry = leg_cost_bps(
        order, df.get("ask_notional_decision"), df.get("half_spread_bps_decision"),
        sigma, part, impact_coefficient, levels_penalty,
    )
    hat_exit = leg_cost_bps(
        order, df.get("bid_notional_decision"), df.get("half_spread_bps_decision"),
        sigma, part, impact_coefficient, levels_penalty,
    )
    df["cost_hat_bps"] = hat_entry + hat_exit
    df["cost_hat"] = df["cost_hat_bps"] / 10_000

    # Scale-free companions. Continuous only -- round 2 measured binary flags
    # costing -0.64% while continuous recency gained +1.05%.
    df["half_spread_bps_decision"] = df.get("half_spread_bps_decision")
    inside = df.get("ask_notional_decision")
    df["log_inside_notional_decision"] = np.log1p(
        pd.Series(inside, index=df.index).astype(float).clip(lower=0.0)
    )
    df["depth_ratio_decision"] = (
        pd.Series(inside, index=df.index).astype(float) / max(float(order_notional), 1.0)
    ).clip(upper=100.0)

    # --- realised charge (09:35 ask, 15:55 bid) -----------------------------
    legs = round_trip_cost_bps(
        order,
        pd.DataFrame({
            "half_spread_bps": df.get("half_spread_bps_entry"),
            "ask_notional": df.get("ask_notional_entry"),
        }, index=df.index),
        pd.DataFrame({
            "half_spread_bps": df.get("half_spread_bps_exit"),
            "bid_notional": df.get("bid_notional_exit"),
        }, index=df.index),
        sigma, part, impact_coefficient, levels_penalty,
    )
    df["entry_leg_bps"] = legs["entry_leg_bps"]
    df["exit_leg_bps"] = legs["exit_leg_bps"]
    df["cost_realized_bps"] = legs["round_trip_bps"]
    df["cost_realized"] = df["cost_realized_bps"] / 10_000
    return df


def assert_decision_quotes_are_pre_entry(
    quotes: pd.DataFrame,
    entry_minute: str = ENTRY_MINUTE,
    timestamp_col: str = "timestamp",
) -> None:
    """Tripwire: nothing feeding ``cost_hat`` may be observed at or after entry.

    Mirrors ``features.premarket.assert_no_post_decision_bars``. The failure it
    guards is flattering rather than obvious -- a cost estimate that peeks at the
    entry minute knows how wide the market actually opened, which is exactly the
    thing a decision rule would most like to know and could not.
    """
    if quotes.empty:
        return
    if timestamp_col not in quotes.columns:
        raise KeyError(f"quote frame missing {timestamp_col!r}")

    ts = pd.to_datetime(quotes[timestamp_col], utc=True).dt.tz_convert("America/New_York")
    cutoff = pd.to_datetime(ts.dt.date.astype(str) + f" {entry_minute}").dt.tz_localize(
        "America/New_York"
    )
    late = ts >= cutoff
    if bool(late.any()):
        worst = ts[late].max()
        raise AssertionError(
            f"cost_hat quotes must be strictly before {entry_minute}; "
            f"{int(late.sum())} quote(s) at or after entry, latest {worst}"
        )


def tradeable_mask(
    quote_summary: pd.DataFrame,
    order_notional: float,
    max_spread_bps: float,
) -> pd.Series:
    """Which candidates can actually be traded at this order size.

    A name with no firm inside quote, a book smaller than the order, or a
    ruinous spread is not a data gap to apologise for -- it is outside the
    strategy by construction, because you can only trade what you can quote.
    The count removed is reported by the caller rather than silently applied.
    """
    q = quote_summary
    has_quote = q["half_spread_bps"].notna() & q["ask_notional"].notna()
    deep_enough = q["ask_notional"].fillna(0.0) >= float(order_notional)
    tight_enough = q["half_spread_bps"] <= float(max_spread_bps)
    return (has_quote & deep_enough & tight_enough).fillna(False)


def bucket_summary(
    cost_table: pd.DataFrame,
    by: str,
    bins: Sequence[float],
    return_col: Optional[str] = None,
) -> pd.DataFrame:
    """Cost by price or liquidity bucket, optionally alongside gross return.

    Reported because a single average cost is misleading when the population
    spans $0.50 names quoting 200 bps wide and $9 names quoting 20.

    ``return_col`` puts cost and edge in the same table. Round 3 reported them
    separately and so never crossed them -- which is how a 6.3x cost span across
    the liquidity range went unnoticed for a whole round.
    """
    df = cost_table.dropna(subset=["half_spread_bps", by]).copy()
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = pd.cut(df[by], bins=bins)

    aggs = dict(
        n=("half_spread_bps", "size"),
        half_spread_bps=("half_spread_bps", "median"),
        impact_bps=("impact_bps", "median"),
        depth_shortfall_bps=("depth_shortfall_bps", "median"),
        round_trip_bps=("measured_round_trip_bps", "median"),
    )
    if return_col is not None and return_col in df.columns:
        aggs["gross_mean"] = (return_col, "mean")
        aggs["gross_median"] = (return_col, "median")

    return df.groupby("bucket", observed=True).agg(**aggs).reset_index()
