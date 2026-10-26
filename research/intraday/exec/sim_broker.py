"""SimBroker: the spec section 4 fill rules on bus Bars (execution.md, SimBroker fill rules).

Every returned event carries its cause's ``ts``: a bar fill is stamped ``bar.ts``;
tick, halt and kill outcomes carry that event's ``ts``. A decision at ``T`` fills at
the open of the first bar with ``bar_start >= T``.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from research.intraday.contracts.events import (
    Bar, ClockTick, Fill, Halt, Kill, OrderIntent, OrderUpdate, Quote,
)
from research.intraday.exec.book import Lot, WorkingOrder
from research.intraday.exec.broker import ExecEvent
from research.intraday.exec.config import ExecConfig, to_et
from research.intraday.exec.costs import CostInputsSource, fill_cost_bps
from research.intraday.exec.ids import client_order_id
from research.intraday.exec.sizing import fill_cap, fresh_quote

Side = Literal["buy", "sell"]
SOURCE_KEYS = ("cap:quote", "cap:bar", "cost:quote", "cost:summary", "cost:default")


class _BarBudget:
    """One fill-cap budget per side for one bar; computed lazily, once per side."""

    def __init__(self, bar: Bar, quote: Quote | None, config: ExecConfig) -> None:
        self.bar, self.quote, self._cfg = bar, quote, config
        self._left: dict[str, int] = {}
        self.source: dict[str, str] = {}

    def left(self, side: Side) -> int:
        if side not in self._left:
            cap, src = fill_cap(side, self.bar.bar_start, self.bar, self.quote, self._cfg)
            self._left[side], self.source[side] = cap, src
        return self._left[side]

    def take(self, side: Side, qty: int) -> None:
        if not 0 < qty <= self.left(side):
            raise AssertionError(f"fill of {qty} exceeds the {side} budget {self.left(side)}")
        self._left[side] -= qty


class SimBroker:
    def __init__(self, config: ExecConfig, cost_inputs: CostInputsSource) -> None:
        self._cfg = config
        self._inputs = cost_inputs
        self._seq = 0
        self._working: dict[str, WorkingOrder] = {}
        self._lots: dict[str, Lot] = {}
        self._quote: dict[str, Quote] = {}
        self._last_close: dict[str, float] = {}
        self._last_start: dict[str, datetime] = {}
        self._halted: set[str] = set()
        self._closed = False
        self._counts: dict[str, int] = {k: 0 for k in SOURCE_KEYS}

    # ---- queries ------------------------------------------------------------
    def position(self, symbol: str) -> int:
        return sum(lot.open_qty for lot in self._lots.values() if lot.symbol == symbol)

    def source_counts(self) -> dict[str, int]:
        return dict(self._counts)

    def sellable_qty(self, symbol: str) -> int:
        free = sum(lot.open_qty for lot in self._lots.values()
                   if lot.symbol == symbol and lot.exit is None)
        committed = sum(w.remaining for w in self._working.values()
                        if w.symbol == symbol and w.intent.side == "sell")
        return free - committed

    # ---- intents ------------------------------------------------------------
    def refusal(self, intent: OrderIntent) -> str | None:
        if self._closed:
            return "session closed"
        if intent.side == "buy" and intent.symbol in self._halted:
            return "symbol halted"
        return None

    def submit(self, intent: OrderIntent) -> None:
        if intent.intent_id in self._working:
            raise AssertionError(f"intent {intent.intent_id} submitted twice")
        buy = intent.side == "buy"
        deadline = intent.ts + timedelta(minutes=self._cfg.entry_max_wait_min) if buy else None
        self._working[intent.intent_id] = WorkingOrder(
            intent=intent, client_order_id=client_order_id(intent.intent_id, "entry" if buy else "sell"),
            remaining=intent.qty, not_before=intent.ts, deadline=deadline, seq=self._next_seq())

    # ---- market events ------------------------------------------------------
    def on_quote(self, quote: Quote) -> None:
        self._quote[quote.symbol] = quote

    def on_bar(self, bar: Bar) -> tuple[ExecEvent, ...]:
        sym = bar.symbol
        last = self._last_start.get(sym)
        if last is not None and bar.bar_start <= last:
            raise ValueError(f"bar for {sym} out of order: bar_start {bar.bar_start} <= {last}")
        out: list[ExecEvent] = []
        if sym not in self._halted and not self._closed:
            budget = _BarBudget(bar, fresh_quote(self._quote.get(sym), bar.bar_start, self._cfg),
                                self._cfg)
            out.extend(self._fill_buys(bar, budget))
        self._last_close[sym] = bar.close
        self._last_start[sym] = bar.bar_start
        return tuple(out)

    def on_tick(self, tick: ClockTick) -> tuple[ExecEvent, ...]:
        out: list[ExecEvent] = []
        for w in self._sorted_working("buy"):
            if w.deadline is not None and tick.ts > w.deadline:
                del self._working[w.intent.intent_id]
                out.append(self._update(tick.ts, w.intent.intent_id, "expired", "entry wait exceeded"))
        return tuple(out)

    def on_halt(self, halt: Halt) -> tuple[ExecEvent, ...]:
        return ()

    def on_kill(self, kill: Kill) -> tuple[ExecEvent, ...]:
        return ()

    # ---- fills --------------------------------------------------------------
    def _fill_buys(self, bar: Bar, budget: _BarBudget) -> list[ExecEvent]:
        out: list[ExecEvent] = []
        for w in self._symbol_working(bar.symbol, "buy"):
            if w.not_before > bar.bar_start:
                continue
            iid = w.intent.intent_id
            if w.deadline is None or bar.bar_start >= w.deadline:
                del self._working[iid]
                out.append(self._update(bar.ts, iid, "expired", "entry wait exceeded"))
                continue
            q = min(w.remaining, budget.left("buy"))
            del self._working[iid]
            if q == 0:
                out.append(self._update(bar.ts, iid, "cancelled", "size cap zero"))
                continue
            out.append(self._fill(budget, iid, "buy", q))
            it = w.intent
            self._lots[iid] = Lot(entry_id=iid, symbol=it.symbol, open_qty=q,
                                  take_profit=it.take_profit, stop_loss=it.stop_loss,
                                  time_stop=it.time_stop, seq=self._next_seq())
            if q < it.qty:
                out.append(self._update(bar.ts, iid, "cancelled",
                                        f"partial fill {q}/{it.qty}: remainder {it.qty - q} cancelled"))
        return out

    def _fill(self, budget: _BarBudget, intent_id: str, side: Side, qty: int) -> Fill:
        bar = budget.bar
        budget.take(side, qty)
        cost, src = self._cost(side, qty, bar.open, bar.symbol, bar.bar_start, budget.quote)
        self._counts[f"cap:{budget.source[side]}"] += 1
        self._counts[f"cost:{src}"] += 1
        return Fill(ts=bar.ts, intent_id=intent_id, symbol=bar.symbol, side=side, qty=qty,
                    price=bar.open, cost_bps=cost)

    def _cost(self, side: Side, qty: int, price: float, symbol: str, fill_instant: datetime,
              quote: Quote | None) -> tuple[float, str]:
        session = to_et(fill_instant).date()
        inputs = self._inputs.get(symbol, session)
        if inputs is None:
            raise ValueError(f"no cost inputs for {symbol} on {session}")
        return fill_cost_bps(side, qty, price, fill_instant, quote, inputs, self._cfg)

    # ---- helpers ------------------------------------------------------------
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _symbol_working(self, symbol: str, side: Side) -> list[WorkingOrder]:
        return sorted((w for w in self._working.values()
                       if w.symbol == symbol and w.intent.side == side), key=lambda w: w.seq)

    def _sorted_working(self, side: Side) -> list[WorkingOrder]:
        return sorted((w for w in self._working.values() if w.intent.side == side),
                      key=lambda w: (w.symbol, w.seq))

    @staticmethod
    def _update(ts: datetime, intent_id: str, status: str, detail: str) -> OrderUpdate:
        return OrderUpdate(ts=ts, intent_id=intent_id, status=status, detail=detail)
