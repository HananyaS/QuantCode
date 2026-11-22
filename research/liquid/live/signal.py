"""C3's decision, computed live with the research cell's own functions (PAPER_C3 §2).

`day_plan` appends a placeholder row for today to the SIP history and calls
`cells.noise_area`, which reads only earlier rows (tests/test_liquid.py), so
sigma and the volatility are exactly the research values. `check` applies the
registered rule at one check bar from today's bars up to that bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.liquid import cells
from research.liquid.calendar import FULL_BARS
from research.liquid.data import FIELDS, Minutes


@dataclass(frozen=True)
class DayPlan:
    date: pd.Timestamp
    n_bars: int
    prev_close: float
    sigma: np.ndarray          # (390,) mean absolute move from the open, previous 14 sessions
    vol: float                 # 14-day daily-return sd (ddof 1)
    weight: float              # min(4, 0.02 / vol)
    checks: tuple[int, ...]    # decide bars, ascending

    @property
    def last(self) -> int:
        return self.n_bars - 1

    @property
    def tradable(self) -> bool:
        return np.isfinite(self.prev_close) and np.isfinite(self.vol) and self.vol > 0 and bool(self.checks)


@dataclass(frozen=True)
class CheckResult:
    d: int
    price: float
    ub: float
    vwap: float
    enter: bool                # the rule's entry condition (caller adds "flat")
    stop: float                # max(UB, VWAP) at this check


def day_plan(hist: Minutes, today: pd.Timestamp, n_bars: int) -> DayPlan:
    """The plan for `today` from SIP history ending with the previous session."""
    assert len(hist.dates) and hist.dates[-1] < today, "history must end before today"
    blank = np.full((1, FULL_BARS), np.nan)
    blank[0, :n_bars] = 1.0                           # placeholder: noise_area reads only earlier rows
    last_close = hist.close[len(hist.dates) - 1, hist.n_bars[-1] - 1]
    m = Minutes(hist.symbol, hist.dates.append(pd.DatetimeIndex([today])),
                np.append(hist.n_bars, n_bars),
                *(np.vstack([getattr(hist, f), blank]) for f in FIELDS),
                np.append(hist.prev_close, last_close), np.append(hist.valid, True))
    sigma, vol = cells.noise_area(m)
    s, v = sigma[-1], float(vol[-1])
    last = n_bars - 1
    checks = tuple(d for d in cells.C3_CHECKS if d < last and np.isfinite(s[d]))
    w = min(cells.LEVER_CAP, cells.C3_VOL_TARGET / v) if np.isfinite(v) and v > 0 else np.nan
    return DayPlan(pd.Timestamp(today), n_bars, float(last_close), s, v, w, checks)


def vwap_through(vwap: np.ndarray, volume: np.ndarray, close: np.ndarray, d: int) -> float:
    """Session VWAP through bar d, as cells.session_vwap."""
    v = np.nan_to_num(volume[: d + 1])
    pv = float(np.sum(np.nan_to_num(vwap[: d + 1]) * v))
    cv = float(np.sum(v))
    return pv / cv if cv > 0 else float(close[d])


def check(plan: DayPlan, o0: float, close: np.ndarray, volume: np.ndarray, vwap: np.ndarray,
          d: int) -> CheckResult:
    """The registered C3 rule at decide bar d, from bars 0..d only."""
    assert d in plan.checks, d
    assert np.isfinite(close[d]), f"bar {d} missing"
    ub = max(o0, plan.prev_close) * (1 + plan.sigma[d])
    vw = vwap_through(vwap, volume, close, d)
    px = float(close[d])
    return CheckResult(d, px, float(ub), vw, px > ub and px > vw, float(max(ub, vw)))


def shares(equity: float, weight: float, price: float, buying_power: float) -> int:
    """Whole shares for notional equity x weight, capped by buying power."""
    notional = min(equity * weight, buying_power)
    return max(int(np.floor(notional / price)), 0)


def tick(price: float) -> float:
    """Alpaca price increment: 2 decimals at or above $1, 4 below (fact sheet item 7)."""
    return round(price, 2) if price >= 1.0 else round(price, 4)


def floor_tick(price: float) -> float:
    """Round down to the Alpaca increment. The runner uses this for the buy limit, the sell
    stop and the forced-exit test (G3 W1): prices trade on the cent grid, so for a grid price p,
    p <= L exactly when p <= floor(L), and the live order triggers or fills exactly where the
    research order does. Rounding to nearest diverged on 98 of 1,741 dev sessions."""
    scale = 100 if price >= 1.0 else 10_000
    return float(np.floor(price * scale + 1e-7)) / scale
