"""The three L1 cells (PREREGISTRATION_L1 §3) and their daily returns.

Each cell turns one session's bars into bracket orders, simulates them
(`bracket.simulate`) and returns Trade records. `daily()` turns trades into
daily returns on prior-day equity: the sum over the day's trades of
weight x net return. No position is held overnight; a day without a trade
returns 0 (idle cash earns nothing: conservative).

`long_short=True` and the other keyword variants exist only for the labelled,
non-selectable diagnostics in §6.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.liquid.bracket import Order, Result, simulate
from research.liquid.calendar import minute_index
from research.liquid.costs import CostSpec
from research.liquid.data import Minutes

COLLAR = 0.001          # marketable-limit collar: 10 bps through the decision close
LEVER_CAP = 4.0
C1_DECIDE_10 = minute_index("09:59")          # price at 10:00 = close of the 09:59 bar
C2_RANGE = 5                                   # bars 09:30..09:34
C2_RISK = 0.01
C2_TP_R = 10.0
C3_LOOKBACK = 14
C3_MIN_OBS = 10
C3_VOL_TARGET = 0.02
C3_CHECKS = tuple(minute_index(f"{h:02d}:{m:02d}") - 1
                  for h in range(10, 16) for m in (0, 30))      # 10:00..15:30 -> decide bars


@dataclass(frozen=True)
class Trade:
    date: pd.Timestamp
    cell: str
    side: int
    weight: float
    entry: float
    exit: float
    kind: str
    entry_bar: int
    exit_bar: int

    @property
    def gross(self) -> float:
        return self.side * (self.exit / self.entry - 1.0)

    def net(self, cost: CostSpec) -> float:
        ce, cx = cost.entry_frac(self.entry), cost.exit_frac(self.exit, self.kind)
        if self.side == 1:
            return (self.exit * (1 - cx) - self.entry * (1 + ce)) / self.entry
        return (self.entry * (1 - ce) - self.exit * (1 + cx)) / self.entry


def _limit(px: float, side: int) -> float:
    return px * (1 + COLLAR) if side == 1 else px * (1 - COLLAR)


def _trade(m: Minutes, i: int, cell: str, side: int, w: float, order: Order) -> Trade | None:
    r: Result = simulate(m.open[i], m.high[i], m.low[i], m.close[i], order, side)
    if not r.filled:
        return None
    return Trade(m.dates[i], cell, side, w, r.entry, r.exit, r.kind, r.entry_bar, r.exit_bar)


# ---------------------------------------------------------------- C1 -------
def c1(m: Minutes, long_short: bool = False) -> list[Trade]:
    """Gao et al. (2018): r1 = P(10:00)/prior close - 1 > 0 -> long the last half-hour."""
    out = []
    for i in range(len(m.dates)):
        pc = m.prev_close[i]
        if not m.valid[i] or not np.isfinite(pc):
            continue
        r1 = m.close[i, C1_DECIDE_10] / pc - 1.0
        side = 1 if r1 > 0 else (-1 if (r1 < 0 and long_short) else 0)
        if side == 0:
            continue
        last = m.last_bar(i)
        decide = last - 30                          # entry in the bar starting 30 min before the close
        t = _trade(m, i, "C1", side, 1.0, Order(decide, _limit(m.close[i, decide], side), last))
        if t:
            out.append(t)
    return out


# ---------------------------------------------------------------- C2 -------
def c2(m: Minutes, long_short: bool = False, cap: float = LEVER_CAP) -> list[Trade]:
    """Zarattini & Aziz (2023): direction of the 09:30-09:35 bar; stop at its far end; TP 10R."""
    out = []
    for i in range(len(m.dates)):
        if not m.valid[i]:
            continue
        o0 = m.open[i, 0]
        c4 = m.close[i, C2_RANGE - 1]
        lo, hi = np.min(m.low[i, :C2_RANGE]), np.max(m.high[i, :C2_RANGE])
        if c4 > o0:
            side, stop, risk = 1, lo, c4 - lo
        elif c4 < o0 and long_short:
            side, stop, risk = -1, hi, hi - c4
        else:
            continue                               # doji (c4 == o0), or a down bar long-only
        assert risk > 0
        tp = c4 + side * C2_TP_R * risk
        w = min(C2_RISK * c4 / risk, cap)
        order = Order(C2_RANGE - 1, _limit(c4, side), m.last_bar(i), tp, stop)
        t = _trade(m, i, "C2", side, w, order)
        if t:
            out.append(t)
    return out


# ---------------------------------------------------------------- C3 -------
def noise_area(m: Minutes) -> tuple[np.ndarray, np.ndarray]:
    """sigma (D, 390): mean |close(m)/open(09:30) - 1| over the previous 14 sessions; and
    the daily-return volatility (D,) over the previous 14 sessions (ddof=1)."""
    move = np.abs(m.close / m.open[:, :1] - 1.0)
    move[~m.valid] = np.nan
    have = np.isfinite(move)
    cs = np.vstack([np.zeros((1, move.shape[1])), np.cumsum(np.where(have, move, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, move.shape[1])), np.cumsum(have, axis=0)])
    d = len(m.dates)
    sigma = np.full_like(move, np.nan)
    for i in range(C3_LOOKBACK, d):
        s = cs[i] - cs[i - C3_LOOKBACK]
        n = cn[i] - cn[i - C3_LOOKBACK]
        sigma[i] = np.where(n >= C3_MIN_OBS, s / np.maximum(n, 1), np.nan)
    last_close = m.close[np.arange(d), m.n_bars - 1]
    ret = pd.Series(last_close / m.prev_close - 1.0)
    vol = ret.rolling(C3_LOOKBACK).std(ddof=1).shift(1).to_numpy()
    return sigma, vol


def session_vwap(m: Minutes, i: int) -> np.ndarray:
    """Cumulative session VWAP through each bar's close (bar vwap x volume)."""
    v = np.nan_to_num(m.volume[i])
    pv = np.cumsum(np.nan_to_num(m.vwap[i]) * v)
    cv = np.cumsum(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cv > 0, pv / cv, m.close[i])


def c3(m: Minutes, long_short: bool = False, cap: float = LEVER_CAP,
       resting_stop: bool = True) -> list[Trade]:
    """Zarattini, Aziz & Barbon (2024) noise area: enter at a half-hour check when the
    price is beyond the boundary and VWAP; stop at max(UB, VWAP), re-priced each check."""
    sigma, vol = noise_area(m)
    out = []
    for i in range(len(m.dates)):
        pc = m.prev_close[i]
        if not m.valid[i] or not np.isfinite(pc) or not np.isfinite(vol[i]) or vol[i] <= 0:
            continue
        last = m.last_bar(i)
        o0, c = m.open[i, 0], m.close[i]
        ub = max(o0, pc) * (1 + sigma[i])
        lb = min(o0, pc) * (1 - sigma[i])
        vw = session_vwap(m, i)
        checks = [d for d in C3_CHECKS if d < last and np.isfinite(sigma[i, d])]
        w = min(cap, C3_VOL_TARGET / vol[i])
        free_from = -1                               # flat by the close of this bar
        for d in checks:
            if d < free_from:
                continue
            if c[d] > ub[d] and c[d] > vw[d]:
                side = 1
            elif long_short and c[d] < lb[d] and c[d] < vw[d]:
                side = -1
            else:
                continue
            stop = np.full(last + 1, np.nan)
            later = [x for x in checks if x >= d]
            for a, b in zip(later, later[1:] + [last]):
                lvl = max(ub[a], vw[a]) if side == 1 else min(lb[a], vw[a])
                if resting_stop:
                    stop[a + 1: b + 1] = lvl                     # live from the bar after the check
                elif a > d and side * (c[a] - lvl) < 0 and a + 1 <= last:
                    stop[a + 1] = side * np.inf                  # check-time exit: market at the next open
            t = _trade(m, i, "C3", side, w, Order(d, _limit(c[d], side), last, np.nan, stop))
            if t:
                out.append(t)
                free_from = t.exit_bar
            # unfilled: try again at the next check
    return out


# ---------------------------------------------------------------- returns --
def daily(trades: list[Trade], dates: pd.DatetimeIndex, cost: CostSpec | None) -> pd.Series:
    """Daily return on prior-day equity; cost=None gives gross."""
    r = pd.Series(0.0, index=dates)
    for t in trades:
        r[t.date] += t.weight * (t.gross if cost is None else t.net(cost))
    return r


def trade_frame(trades: list[Trade], cost: CostSpec) -> pd.DataFrame:
    return pd.DataFrame([{**t.__dict__, "gross": t.gross, "net": t.net(cost)} for t in trades])
