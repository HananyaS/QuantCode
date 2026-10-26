"""Per symbol-day cost inputs from the daily cache and the 09:34 quote summary.

Only rows dated before the session enter (point in time). Levels are on the traded
(unadjusted) share basis of the session (execution.md, Cost inputs; risk.md):

* ``sigma_daily``: sample std of the last ``window`` adjusted close-to-close returns
  (the ``bars.py`` ``vol_20d`` definition); a return statistic, so adjusted is fine;
* ``adv_shares``: the mean of exactly ``window`` prior sessions' share volume on D's
  traded basis, the same semantics as ``risk/adv.py`` (implemented here, since agents
  never import each other): with ``traded = volume_unadj > 0`` and the split-only factor
  ``s = volume / volume_unadj`` on traded rows (controller ruling P2-5: Alpaca adjusts
  volume for splits only, so ``f`` is wrong), each traded session contributes
  ``volume_unadj * s / s_anchor`` and an untraded one 0, where ``s_anchor`` is the ``s``
  of the last prior session that traded.

A symbol with fewer than ``window`` prior sessions gets no entry, so a fill in it raises
``no cost inputs`` in the broker; a window that never traded, or a non-finite / non-positive
``s`` on a traded row, is a data defect and raises here.
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


def _adv_shares(w: pd.DataFrame) -> pd.Series:
    """risk/adv.py semantics, vectorised over symbols; ``w`` = each symbol's last window rows."""
    raw = w["volume_unadj"].astype(float)
    traded = raw > 0
    s = (w["volume"].astype(float) / raw).where(traded)
    bad = traded & ~((s > 0) & (s < math.inf))
    if bad.any():
        raise ValueError(f"split factor volume / volume_unadj not finite and > 0 on traded rows "
                         f"for {sorted(w.loc[bad, 'symbol'].unique())}")
    never = traded.groupby(w["symbol"]).sum() == 0
    if never.any():
        raise ValueError(f"no prior session in the window traded (volume_unadj > 0) for "
                         f"{sorted(never.index[never])}")
    s_anchor = s[traded].groupby(w.loc[traded, "symbol"]).last()
    contrib = (raw * s / w["symbol"].map(s_anchor)).where(traded, 0.0)
    return contrib.groupby(w["symbol"]).mean()


def _daily_stats(prior: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-symbol sigma_daily and adv_shares from prior rows (sorted by symbol, date)."""
    n = prior.groupby("symbol", sort=True).size()
    full = prior[prior["symbol"].map(n) >= window]
    grp = full.groupby("symbol", sort=True)
    closes = grp.tail(window + 1)
    rets = closes.groupby("symbol")["close"].pct_change()
    sigma = rets.groupby(closes["symbol"]).std(ddof=1)
    out = pd.DataFrame({"sigma_daily": sigma, "adv_shares": _adv_shares(grp.tail(window))})
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
                      unseal: bool = False, window: int = 20) -> StaticCostInputs:
    if session_date >= holdout_start and not unseal:
        raise ValueError(
            f"session {session_date} is inside the sealed holdout (>= {holdout_start})")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    _require(daily, DAILY_REQUIRED, "daily")
    day = pd.Timestamp(session_date)
    prior = daily[pd.to_datetime(daily["date"]) < day].sort_values(["symbol", "date"])
    if prior.duplicated(["symbol", "date"]).any():
        raise ValueError("daily has duplicate (symbol, date) rows")
    stats = _daily_stats(prior, window)
    summ = _summary_for(decision_summary, day).reindex(stats.index)
    table: dict[tuple[str, date], CostInputs] = {}
    for sym, sig, adv, hs, ask, bid in zip(
            stats.index, stats["sigma_daily"], stats["adv_shares"],
            summ["half_spread_bps"], summ["ask_notional"], summ["bid_notional"]):
        table[(str(sym), session_date)] = CostInputs(
            sigma_daily=float(sig), adv_shares=float(adv), half_spread_bps=_opt(hs),
            ask_notional=_opt(ask), bid_notional=_opt(bid))
    return StaticCostInputs(table)
