"""Per symbol-day cost inputs from the daily cache and the 09:34 quote summary.

Only rows dated before the session enter (point in time). Levels are on the traded
(unadjusted) share basis of the session (execution.md, Cost inputs; risk.md):

* ``sigma_daily``: sample std of the last ``window`` adjusted close-to-close returns
  (the ``bars.py`` ``vol_20d`` definition); a return statistic, so adjusted is fine;
* ``adv_shares``: mean adjusted share volume of the last ``window`` prior sessions,
  divided by the split-only factor ``s = volume / volume_unadj`` of the last prior row
  (controller ruling P2-5: Alpaca adjusts volume for splits only, so ``f`` is wrong).
"""
from __future__ import annotations

import math
from datetime import date

import pandas as pd

from research.intraday.exec.config import HOLDOUT_START
from research.intraday.exec.costs import CostInputs, StaticCostInputs

DAILY_REQUIRED = ("symbol", "date", "close", "volume", "volume_unadj")
SUMMARY_FIELDS = ("half_spread_bps", "ask_notional", "bid_notional")


def _require(frame: pd.DataFrame, cols: tuple[str, ...], what: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing columns {missing}")


def _daily_stats(prior: pd.DataFrame, window: int, min_sessions: int) -> pd.DataFrame:
    """Per-symbol sigma_daily and adv_shares from prior rows (sorted by symbol, date)."""
    grp = prior.groupby("symbol", sort=True)
    closes = grp.tail(window + 1)
    rets = closes.groupby("symbol")["close"].pct_change()
    sigma = rets.groupby(closes["symbol"]).std(ddof=1)
    mean_vol = grp.tail(window).groupby("symbol")["volume"].mean()
    last = grp.tail(1).set_index("symbol")
    s_prev = last["volume"] / last["volume_unadj"]
    out = pd.DataFrame({"sigma_daily": sigma, "adv_shares": mean_vol / s_prev,
                        "n": grp.size()})
    out = out[out["n"] >= min_sessions].drop(columns="n")
    finite = out.apply(lambda c: c.map(math.isfinite)).all(axis=1)
    return out[finite & (out["sigma_daily"] >= 0) & (out["adv_shares"] > 0)]


def _summary_for(decision_summary: pd.DataFrame | None, day: pd.Timestamp) -> pd.DataFrame:
    if decision_summary is None:
        return pd.DataFrame(columns=list(SUMMARY_FIELDS))
    _require(decision_summary, ("symbol", "date", *SUMMARY_FIELDS), "decision_summary")
    rows = decision_summary[pd.to_datetime(decision_summary["date"]) == day]
    if rows["symbol"].duplicated().any():
        raise ValueError(f"decision_summary has duplicate symbols on {day.date()}")
    return rows.set_index("symbol")[list(SUMMARY_FIELDS)]


def _opt(x: object) -> float | None:
    if x is None or pd.isna(x):
        return None
    return float(x)


def build_cost_inputs(daily: pd.DataFrame, decision_summary: pd.DataFrame | None,
                      session_date: date, *, holdout_start: date = HOLDOUT_START,
                      unseal: bool = False, window: int = 20,
                      min_sessions: int = 10) -> StaticCostInputs:
    if session_date >= holdout_start and not unseal:
        raise ValueError(
            f"session {session_date} is inside the sealed holdout (>= {holdout_start})")
    if window < 2 or min_sessions < 2:
        raise ValueError(f"window and min_sessions must be >= 2, got {window}, {min_sessions}")
    _require(daily, DAILY_REQUIRED, "daily")
    day = pd.Timestamp(session_date)
    prior = daily[pd.to_datetime(daily["date"]) < day].sort_values(["symbol", "date"])
    if prior.duplicated(["symbol", "date"]).any():
        raise ValueError("daily has duplicate (symbol, date) rows")
    stats = _daily_stats(prior, window, min_sessions)
    summ = _summary_for(decision_summary, day).reindex(stats.index)
    table: dict[tuple[str, date], CostInputs] = {}
    for sym, sig, adv, hs, ask, bid in zip(
            stats.index, stats["sigma_daily"], stats["adv_shares"],
            summ["half_spread_bps"], summ["ask_notional"], summ["bid_notional"]):
        table[(str(sym), session_date)] = CostInputs(
            sigma_daily=float(sig), adv_shares=float(adv), half_spread_bps=_opt(hs),
            ask_notional=_opt(ask), bid_notional=_opt(bid))
    return StaticCostInputs(table)
