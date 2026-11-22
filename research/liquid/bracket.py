"""One bracket trade on one session's one-minute bars (PREREGISTRATION_L1 §2.3-2.4).

Long side. A short is simulated by mirroring prices (see `simulate`).

Entry: a marketable buy limit sent at the close of bar `decide`, working in
bar e = decide + 1 only.
- open[e] <= limit: filled at open[e];
- else low[e] < limit: filled at the limit;
- else unfilled and cancelled.

Exits, live from the entry bar e onward, checked bar by bar:
- stop: low[k] <= stop[k] -> filled at min(stop[k], open[k]); on the entry bar
  at min(stop[k], entry). The stop level for bar k must be known by the close
  of bar k-1 (the caller's job). NaN means no stop; +inf means "sell at this
  bar's open", a market exit decided at the previous close.
- take-profit: high[k] > tp (strictly: the queue ahead must trade first) ->
  filled at tp; on the entry bar at max(tp, entry).
- both on one bar: the stop.
- time: neither by the close of bar `last` -> filled at close[last] (market-on-close).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Order:
    decide: int                    # bar whose close sends the entry
    limit: float                   # buy limit
    last: int                      # session's last bar: time exit at its close
    take_profit: float = np.nan    # absolute price; NaN = none
    stop: float | np.ndarray = np.nan   # scalar, or per-bar level (len >= last + 1); NaN = none


@dataclass(frozen=True)
class Result:
    filled: bool
    entry: float = np.nan
    entry_bar: int = -1
    exit: float = np.nan
    exit_bar: int = -1
    kind: str = "unfilled"         # unfilled | tp | sl | time


def _stop_at(stop: float | np.ndarray, k: int) -> float:
    return float(stop[k]) if isinstance(stop, np.ndarray) else float(stop)


def simulate_long(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, order: Order) -> Result:
    e = order.decide + 1
    if e > order.last:
        return Result(False)
    if o[e] <= order.limit:
        entry = float(o[e])
    elif l[e] < order.limit:
        entry = float(order.limit)
    else:
        return Result(False)
    tp = float(order.take_profit)
    for k in range(e, order.last + 1):
        s = _stop_at(order.stop, k)
        hit_s = not np.isnan(s) and l[k] <= s          # +inf = exit at this bar's open
        hit_t = np.isfinite(tp) and h[k] > tp
        if hit_s:
            px = min(s, entry) if k == e else min(s, float(o[k]))
            return Result(True, entry, e, px, k, "sl")
        if hit_t:
            px = max(tp, entry) if k == e else tp
            return Result(True, entry, e, px, k, "tp")
    return Result(True, entry, e, float(c[order.last]), order.last, "time")


def simulate(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, order: Order,
             side: int = 1) -> Result:
    """side=+1 long; side=-1 short, run as a long on the mirrored series -price.

    Mirroring maps a sell limit, a buy-to-cover stop above and a take-profit
    below onto the long rules exactly. Prices in the order are given in real
    terms and mirrored here.
    """
    if side == 1:
        return simulate_long(o, h, l, c, order)
    assert side == -1
    st = -order.stop if isinstance(order.stop, np.ndarray) else -float(order.stop)
    m = Order(order.decide, -order.limit, order.last, -order.take_profit, st)
    r = simulate_long(-o, -l, -h, -c, m)
    if not r.filled:
        return r
    return Result(True, -r.entry, r.entry_bar, -r.exit, r.exit_bar, r.kind)
