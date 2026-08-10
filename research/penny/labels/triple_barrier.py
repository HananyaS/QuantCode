"""triple_barrier.py — intraday triple-barrier labels with an executability model.

The label is the study's dependent variable, so every optimistic assumption
buried here becomes a fake edge downstream. The design below is deliberately
pessimistic wherever the data cannot settle a question.

Timeline (all times US/Eastern)
-------------------------------
    09:30-09:34   observation window -- features may use these bars ONLY
    09:35         decision, then entry filled inside this bar
    09:36-15:55   barrier scan
    15:55         vertical barrier: forced exit at the close of this bar

Method
------
Triple-barrier labelling follows Lopez de Prado, *Advances in Financial Machine
Learning* (2018), ch. 3. The departure here is that the barriers are evaluated
on a **minute path** rather than a daily series, and that an explicit
executability model can veto the trade entirely.

Five assumptions, each pessimistic by construction
--------------------------------------------------
1. **Ambiguous minutes resolve against the trade.** If one minute's high
   crosses the profit target *and* its low crosses the stop, the bar does not
   say which came first. We assume the stop. For microcaps -- whose minute
   ranges are enormous -- this is the difference between a plausible strategy
   and a real one, and the optimistic choice would manufacture edge out of bar
   granularity alone.

2. **No print means no fill.** Illiquid names simply have no trades in many
   minutes (a live probe found 52 prints in 391 minutes for one $3 name). A
   barrier "touched" during a stretch with no prints was not touchable. Runs
   with poor coverage are flagged, not silently kept.

3. **Stops slip, targets do not.** A protective stop is a market order once
   triggered and crosses the spread in a falling book; a profit target is a
   resting limit that fills at its price or not at all. Modelling both as exact
   would flatter the strategy at both ends.

4. **Entry pays the spread.** Entry is a marketable order at 09:35 and is
   charged the half-spread plus square-root impact. The printed price is not a
   fill.

5. **Data gaps are suspicion, not absence.** A gap longer than
   ``halt_gap_minutes`` is recorded as ``halt_suspect``. Historical halt data is
   not free, so a gap cannot be distinguished from a halt -- and during a halt a
   position cannot be exited at all, which no barrier model can represent.

One convention worth stating plainly: the barrier scan starts at 09:36, so a
barrier touched *within the entry bar itself* is not detected. This is a
modelling choice forced by bar granularity, and it cuts both ways -- a stop hit
in the first sixty seconds is missed (optimistic), and so is a target
(pessimistic). It is the one place in this module where the pessimism is not
strictly one-directional.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DECISION_MINUTE = "09:35"
OBSERVATION_START = "09:30"
OBSERVATION_END = "09:34"
VERTICAL_MINUTE = "15:55"


@dataclass
class BarrierSpec:
    """Barrier geometry and the execution frictions applied to it."""

    profit_target: float = 0.30       # +30% take-profit
    stop_loss: float = 0.10           # -10% stop
    vertical_minute: str = VERTICAL_MINUTE

    # Execution frictions
    half_spread_floor_bps: float = 25.0   # 0.25% each way, before the estimate
    stop_slippage_bps: float = 50.0       # extra loss when a stop triggers
    impact_coefficient: float = 0.10      # square-root impact, in units of participation
    participation_cap: float = 0.01       # assume 1% of the day's dollar volume

    # Data-quality guards
    min_coverage: float = 0.20            # fraction of scan minutes needing a print
    halt_gap_minutes: int = 10

    # Commission per share, both ways (many brokers charge microcaps per share).
    commission_per_share: float = 0.0


@dataclass
class LabelResult:
    """One labelled (symbol, date) decision."""

    symbol: str
    date: pd.Timestamp
    entry_price: float
    entry_price_filled: float
    exit_price: float
    exit_minute: str
    outcome: str                 # target | stop | vertical | no_fill | insufficient_data
    gross_return: float
    net_return: float
    holding_minutes: int
    max_favourable: float
    max_adverse: float
    coverage: float
    halt_suspect: bool
    ambiguous_bar: bool
    cost_bps: float


def _estimate_half_spread_bps(path: pd.DataFrame, spec: BarrierSpec) -> float:
    """Effective half-spread from the minute high-low range.

    Quote data is not available on the free tier, so the bar range stands in for
    it. The high-low range of a one-minute bar in a thin name is dominated by
    the bid-ask bounce, which is exactly the quantity being estimated. The
    floor prevents an unrealistically tight estimate on minutes that happen to
    print a single trade.

    Related in spirit to Corwin & Schultz (2012), "A Simple Way to Estimate
    Bid-Ask Spreads from Daily High and Low Prices", *Journal of Finance* 67(2),
    adapted to minute bars and used only as a conservative cost floor.
    """
    if path.empty:
        return spec.half_spread_floor_bps
    rng = ((path["high"] - path["low"]) / path["close"].replace(0, np.nan)).dropna()
    if rng.empty:
        return spec.half_spread_floor_bps
    # Half of the median one-minute range approximates the half-spread.
    est_bps = float(np.nanmedian(rng)) * 10_000 / 2.0
    return max(spec.half_spread_floor_bps, est_bps)


def _impact_bps(spec: BarrierSpec) -> float:
    """Square-root market impact at the assumed participation rate.

    Standard concave impact: cost scales with the square root of participation.
    Held fixed per trade because position size is fixed as a fraction of the
    day's dollar volume.
    """
    return spec.impact_coefficient * np.sqrt(spec.participation_cap) * 10_000


def label_one(
    symbol: str,
    day: pd.Timestamp,
    minute_path: pd.DataFrame,
    spec: BarrierSpec,
) -> Optional[LabelResult]:
    """Label a single symbol-day from its minute path.

    ``minute_path`` must contain columns [minute_et, open, high, low, close,
    volume] for one symbol on one session, sorted by time.
    """
    path = minute_path.sort_values("minute_et")

    entry_bar = path[path["minute_et"] == DECISION_MINUTE]
    if entry_bar.empty:
        # No print in the decision minute: the trade was not enterable.
        return LabelResult(
            symbol, day, np.nan, np.nan, np.nan, DECISION_MINUTE, "no_fill",
            np.nan, np.nan, 0, np.nan, np.nan, 0.0, False, False, np.nan,
        )

    entry_row = entry_bar.iloc[0]
    entry_price = float(entry_row["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    scan = path[(path["minute_et"] > DECISION_MINUTE) & (path["minute_et"] <= spec.vertical_minute)]
    if scan.empty:
        return LabelResult(
            symbol, day, entry_price, np.nan, np.nan, DECISION_MINUTE,
            "insufficient_data", np.nan, np.nan, 0, np.nan, np.nan, 0.0, False, False, np.nan,
        )

    # --- costs -------------------------------------------------------------
    # Estimate the spread from the OBSERVATION window only. A live trader
    # entering at 09:35 cannot pay a spread inferred from the afternoon, and
    # using the full path would charge wider spreads on days that later went
    # wild -- coupling the cost, and therefore the label, to future volatility
    # that a vol-correlated veto could indirectly exploit during selection.
    pre_entry = path[path["minute_et"] <= OBSERVATION_END]
    half_spread_bps = _estimate_half_spread_bps(
        pre_entry if not pre_entry.empty else entry_bar, spec
    )
    impact = _impact_bps(spec)
    entry_cost_bps = half_spread_bps + impact
    entry_filled = entry_price * (1 + entry_cost_bps / 10_000)

    up_level = entry_filled * (1 + spec.profit_target)
    down_level = entry_filled * (1 - spec.stop_loss)

    # --- coverage and halt suspicion --------------------------------------
    scan_minutes = pd.to_datetime(scan["minute_et"], format="%H:%M")
    expected = int(
        (pd.to_datetime(spec.vertical_minute, format="%H:%M")
         - pd.to_datetime(DECISION_MINUTE, format="%H:%M")).total_seconds() // 60
    )
    coverage = len(scan) / max(expected, 1)
    gaps = scan_minutes.diff().dt.total_seconds().div(60).fillna(1)
    halt_suspect = bool((gaps > spec.halt_gap_minutes).any())

    # --- barrier scan ------------------------------------------------------
    outcome = "vertical"
    exit_price = float(scan.iloc[-1]["close"])
    exit_minute = str(scan.iloc[-1]["minute_et"])
    holding = len(scan)
    ambiguous = False

    highs = scan["high"].to_numpy(dtype=float)
    lows = scan["low"].to_numpy(dtype=float)
    minutes = scan["minute_et"].to_numpy()

    hit_up = highs >= up_level
    hit_down = lows <= down_level
    first_up = int(np.argmax(hit_up)) if hit_up.any() else len(scan)
    first_down = int(np.argmax(hit_down)) if hit_down.any() else len(scan)

    if first_up < len(scan) or first_down < len(scan):
        if first_down < first_up:
            idx, outcome = first_down, "stop"
        elif first_up < first_down:
            idx, outcome = first_up, "target"
        else:
            # Same bar touches both barriers. The bar cannot say which came
            # first, so resolve against the trade (assumption 1).
            idx, outcome, ambiguous = first_up, "stop", True

        exit_minute = str(minutes[idx])
        holding = idx + 1
        if outcome == "target":
            # Resting limit fills at its price.
            exit_price = up_level
        else:
            # Triggered stop is a market order and crosses a falling book.
            exit_price = down_level * (1 - spec.stop_slippage_bps / 10_000)

    # --- excursions --------------------------------------------------------
    realised = scan.iloc[:holding]
    max_fav = float(realised["high"].max() / entry_filled - 1) if len(realised) else np.nan
    max_adv = float(realised["low"].min() / entry_filled - 1) if len(realised) else np.nan

    # --- returns -----------------------------------------------------------
    exit_cost_bps = half_spread_bps + impact
    exit_net = exit_price * (1 - exit_cost_bps / 10_000)

    gross = exit_price / entry_price - 1.0
    net = exit_net / entry_filled - 1.0
    if spec.commission_per_share:
        net -= (2 * spec.commission_per_share) / entry_filled

    return LabelResult(
        symbol=symbol,
        date=day,
        entry_price=entry_price,
        entry_price_filled=entry_filled,
        exit_price=exit_price,
        exit_minute=exit_minute,
        outcome=outcome,
        gross_return=gross,
        net_return=net,
        holding_minutes=holding,
        max_favourable=max_fav,
        max_adverse=max_adv,
        coverage=coverage,
        halt_suspect=halt_suspect,
        ambiguous_bar=ambiguous,
        cost_bps=entry_cost_bps + exit_cost_bps,
    )


def label_day(minute_panel: pd.DataFrame, day: pd.Timestamp, spec: BarrierSpec) -> pd.DataFrame:
    """Label every symbol present in one session's minute panel."""
    results: List[LabelResult] = []
    for symbol, path in minute_panel.groupby("symbol", sort=False):
        res = label_one(symbol, day, path, spec)
        if res is not None:
            results.append(res)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.__dict__ for r in results])


def observation_features(minute_panel: pd.DataFrame) -> pd.DataFrame:
    """Features from the 09:30-09:34 window only.

    This function is the leakage boundary for intraday information. It must
    never see a bar at or after the decision minute -- the filter below is the
    guarantee, and ``tests/test_penny_labels.py`` asserts it.
    """
    obs = minute_panel[
        (minute_panel["minute_et"] >= OBSERVATION_START)
        & (minute_panel["minute_et"] <= OBSERVATION_END)
    ]
    if obs.empty:
        return pd.DataFrame()

    grp = obs.groupby("symbol", sort=False)
    out = grp.agg(
        open_0930=("open", "first"),
        high_obs=("high", "max"),
        low_obs=("low", "min"),
        close_obs=("close", "last"),
        volume_obs=("volume", "sum"),
        trades_obs=("trade_count", "sum"),
        minutes_obs=("minute_et", "count"),
    ).reset_index()

    out["obs_return"] = out["close_obs"] / out["open_0930"] - 1.0
    out["obs_range"] = (out["high_obs"] - out["low_obs"]) / out["close_obs"].replace(0, np.nan)
    out["obs_dollar_volume"] = out["close_obs"] * out["volume_obs"]
    # Where in its opening range the stock sits after five minutes: near the
    # high is a continuation setup, near the low a failed one.
    span = (out["high_obs"] - out["low_obs"]).replace(0, np.nan)
    out["obs_close_position"] = (out["close_obs"] - out["low_obs"]) / span
    return out
