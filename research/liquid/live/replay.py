"""Replay the real C3 Runner over recorded sessions, with fills by the research rules (G3 W1, B2).

`BarBroker` fills orders from one session's minute bars as `bracket.py` does:
- a buy limit works only in the bar it is sent in: at the open if open <= limit,
  else at the limit if low < limit;
- a stop is checked on each completed bar from the bar it is sent in: it fills at
  min(stop, open) once low <= stop;
- a market order fills at the open of the bar it is sent in.
The clock is simulated, so a whole session replays in milliseconds. A day's live
trades are then compared with `cells.c3` on (entry check, exit kind, exit bar).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquid import cells
from research.liquid.data import Minutes
from research.liquid.live import signal
from research.liquid.live.broker import FakeBroker
from research.liquid.live.feed import FakeFeed
from research.liquid.live.ledger import Ledger
from research.liquid.live.runner import ET, Runner, Session

HISTORY = 20


class SimClock:
    def __init__(self, start: datetime) -> None:
        self.t = start

    def now(self) -> datetime:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += timedelta(seconds=max(s, 0.001))


@dataclass
class BarBroker(FakeBroker):
    m: Minutes | None = None
    i: int = 0
    open_at: datetime | None = None
    fill_bar: dict[str, int] = field(default_factory=dict)
    _sb: dict[str, int] = field(default_factory=dict)        # bar each order was sent in
    _next: dict[str, int] = field(default_factory=dict)      # next bar a stop is checked on

    def _bar(self) -> int:
        k = int((self.clock.now() - self.open_at).total_seconds() // 60)
        return min(k, int(self.m.n_bars[self.i]) - 1)

    def _new(self, coid, side, kind, qty, limit=None, stop=None, status="new"):
        o = super()._new(coid, side, kind, qty, limit, stop, status)
        k = self._bar()
        self._sb[o.id], self._next[o.id] = k, k
        if kind == "market":
            self._fill_at(o.id, float(self.m.open[self.i, k]), k)
        return self.orders[o.id]

    def _fill_at(self, oid: str, price: float, bar: int) -> None:
        self.fill(oid, price)
        self.fill_bar[oid] = bar

    def _advance(self) -> None:
        if self.m is None:
            return
        k, i = self._bar(), self.i
        for oid, o in list(self.orders.items()):
            if not o.open or o.status == "pending_cancel":
                continue
            if o.kind == "limit" and self._next[oid] <= k and self._next[oid] == self._sb[oid]:
                b = self._sb[oid]
                self._next[oid] = b + 1
                if self.m.open[i, b] <= o.limit_price:
                    self._fill_at(oid, float(self.m.open[i, b]), b)
                elif self.m.low[i, b] < o.limit_price:
                    self._fill_at(oid, o.limit_price, b)
            elif o.kind == "stop":
                b = self._next[oid]
                while b < k:                                  # completed bars only
                    if self.m.low[i, b] <= o.stop_price:
                        self._fill_at(oid, float(min(o.stop_price, self.m.open[i, b])), b)
                        break
                    b += 1
                self._next[oid] = b + 1 if self.orders[oid].status == "filled" else b

    def get(self, order_id):
        self._advance()
        return super().get(order_id)

    def open_orders(self, symbol):
        self._advance()
        return super().open_orders(symbol)

    def cancel(self, order_id):
        self._advance()
        return super().cancel(order_id)

    def position_qty(self, symbol):
        self._advance()
        return super().position_qty(symbol)


def research_trades(m: Minutes, i: int, trades: list[cells.Trade] | None = None) -> list[tuple]:
    tr = trades if trades is not None else cells.c3(m)
    return [(t.entry_bar - 1, t.kind, t.exit_bar) for t in tr if t.date == m.dates[i]]


def replay_day(m: Minutes, i: int, today: Minutes | None = None, ledger_path: Path | None = None,
               **runner_kw) -> list[tuple]:
    """Run the Runner on session i of `m`. Fills come from `m`; the runner reads today's bars from
    `today` (default `m`, e.g. IEX bars for the fidelity calibration). Returns live
    (entry check, exit kind, exit bar) tuples."""
    day = m.dates[i]
    n = int(m.n_bars[i])
    open_at = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    clock = SimClock(open_at - timedelta(minutes=2))
    b = BarBroker(clock=clock, m=m, i=i, open_at=open_at)
    led = Ledger(ledger_path or Path(":memory:"))
    plan = signal.day_plan(m.slice(m.dates[max(i - HISTORY, 0)], m.dates[i - 1]), day, n)
    runner_kw.setdefault("poll_s", 30.0)
    Runner(b, FakeFeed(today if today is not None else m), clock, led, **runner_kw).run_session(
        Session(day, open_at, n), plan)
    out = []
    for t in led.trades():
        bar = b.fill_bar.get(t["exit_order_id"]) if t["exit_order_id"] else None
        out.append((int(t["entry_check"]), t["exit_kind"], bar))
    return out


def parity(m: Minutes, rows: list[int]) -> dict:
    """Replay each row and compare with cells.c3. Returns counts and the diverging days."""
    tr = cells.c3(m)
    same, diff, n_live, n_res = 0, [], 0, 0
    for i in rows:
        live, res = replay_day(m, i), research_trades(m, i, tr)
        n_live += len(live)
        n_res += len(res)
        if live == res:
            same += 1
        else:
            diff.append({"day": str(m.dates[i].date()), "live": live, "research": res})
    return {"sessions": len(rows), "identical": same, "diverging": diff, "live_trades": n_live,
            "research_trades": n_res}


def entry_jaccard(live: set, research: set) -> float:
    u = live | research
    return len(live & research) / len(u) if u else float("nan")


def valid_rows(m: Minutes, start: int = HISTORY) -> list[int]:
    return [i for i in range(start, len(m.dates)) if m.valid[i]]


__all__ = ["BarBroker", "SimClock", "replay_day", "research_trades", "parity", "entry_jaccard", "valid_rows",
           "np", "pd"]
