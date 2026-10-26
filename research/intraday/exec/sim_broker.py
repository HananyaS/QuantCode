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
from research.intraday.exec.book import ExitReason, Lot, PendingExit, WorkingOrder
from research.intraday.exec.broker import ExecEvent
from research.intraday.exec.config import ExecConfig, to_et
from research.intraday.exec.costs import CostInputsSource, fill_cost_bps
from research.intraday.exec.ids import client_order_id
from research.intraday.exec.sizing import fill_cap, fresh_quote

Side = Literal["buy", "sell"]
SOURCE_KEYS = ("cap:quote", "cap:bar", "cost:quote", "cost:summary", "cost:default", "cost:refit")


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
        self._killed = False
        self._counts: dict[str, int] = {k: 0 for k in SOURCE_KEYS}
        self._exit_log: list[tuple[str, ExitReason]] = []

    # ---- queries ------------------------------------------------------------
    def position(self, symbol: str) -> int:
        return sum(lot.open_qty for lot in self._lots.values() if lot.symbol == symbol)

    def source_counts(self) -> dict[str, int]:
        return dict(self._counts)

    def exit_log(self) -> tuple[tuple[str, ExitReason], ...]:
        """(entry id, reason) of every exit decision, in decision order."""
        return tuple(self._exit_log)

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
            # feed.md emits each symbol's Bar before its Quote, so a quote stamped after
            # bar_start here is look-ahead and fresh_quote raises.
            budget = _BarBudget(bar, fresh_quote(self._quote.get(sym), bar.bar_start, self._cfg),
                                self._cfg)
            out.extend(self._fill_exits(bar, budget))
            out.extend(self._fill_sells(bar, budget))
            out.extend(self._fill_buys(bar, budget))
        if not self._closed:
            self._check_brackets(bar)
        self._last_close[sym] = bar.close
        self._last_start[sym] = bar.bar_start
        return tuple(out)

    def on_tick(self, tick: ClockTick) -> tuple[ExecEvent, ...]:
        out: list[ExecEvent] = []
        for w in self._sorted_working("buy"):
            if w.deadline is not None and tick.ts > w.deadline:
                del self._working[w.intent.intent_id]
                out.append(self._update(tick.ts, w.intent.intent_id, "expired", "entry wait exceeded"))
        for lot in self._sorted_lots():
            if lot.exit is None and lot.time_stop is not None and lot.time_stop <= tick.ts:
                self._decide_exit(lot, "time", tick.ts)
        if not self._closed and to_et(tick.ts).time() >= self._cfg.eod_et:
            out.extend(self._end_session(tick))
        return tuple(out)

    def on_halt(self, halt: Halt) -> tuple[ExecEvent, ...]:
        sym = halt.symbol
        if halt.halted:
            self._halted.add(sym)
            out: list[ExecEvent] = []
            for w in self._symbol_working(sym, "buy"):
                del self._working[w.intent.intent_id]
                out.append(self._update(halt.ts, w.intent.intent_id, "cancelled", "halt"))
            return tuple(out)
        self._halted.discard(sym)
        for lot in self._symbol_lots(sym):
            if lot.exit is not None:
                lot.exit.not_before = max(lot.exit.not_before, halt.ts)
        for w in self._symbol_working(sym, "sell"):
            w.not_before = max(w.not_before, halt.ts)
        return ()

    def on_kill(self, kill: Kill) -> tuple[ExecEvent, ...]:
        if self._killed:
            return ()
        self._killed = True
        out: list[ExecEvent] = []
        for w in self._sorted_working("buy"):
            del self._working[w.intent.intent_id]
            out.append(self._update(kill.ts, w.intent.intent_id, "cancelled", "kill"))
        for lot in self._sorted_lots():
            if lot.exit is None and not lot.stranded:
                self._decide_exit(lot, "kill", kill.ts)
        return tuple(out)

    def _end_session(self, tick: ClockTick) -> list[ExecEvent]:
        """Once: expire working orders, flatten every open lot at its symbol's last close."""
        self._closed = True
        out: list[ExecEvent] = []
        for side in ("buy", "sell"):
            for w in self._sorted_working(side):
                del self._working[w.intent.intent_id]
                out.append(self._update(tick.ts, w.intent.intent_id, "expired", "session end"))
        for lot in self._sorted_lots():
            if lot.stranded:
                continue
            if lot.symbol in self._halted:
                # expired here means the lot's EXIT went unfilled (halted at the close), not
                # that the entry expired: the lot stays open in the record, marked stranded.
                lot.stranded = True
                out.append(self._update(tick.ts, lot.entry_id, "expired",
                                        "exit unfilled: halted at session end"))
                continue
            out.append(self._eod_fill(lot, tick))
        return out

    def _eod_fill(self, lot: Lot, tick: ClockTick) -> Fill:
        """Full open qty at the last close (uncapped: the depth term charges the size)."""
        price = self._last_close.get(lot.symbol)
        if price is None:
            raise AssertionError(f"open lot {lot.entry_id} in {lot.symbol} has no bar close")
        qty = lot.open_qty
        quote = fresh_quote(self._quote.get(lot.symbol), tick.ts, self._cfg)
        cost, src = self._cost("sell", qty, price, lot.symbol, tick.ts, quote)
        self._counts[f"cost:{src}"] += 1
        self._reduce(lot, qty)
        return Fill(ts=tick.ts, intent_id=lot.entry_id, symbol=lot.symbol, side="sell", qty=qty,
                    price=price, cost_bps=cost)

    # ---- exits --------------------------------------------------------------
    def _decide_exit(self, lot: Lot, reason: ExitReason, not_before: datetime) -> None:
        if lot.exit is not None:
            raise AssertionError(f"lot {lot.entry_id} already has a pending exit")
        lot.exit = PendingExit(reason=reason, not_before=not_before,
                               client_order_id=client_order_id(lot.entry_id, "exit"))
        self._exit_log.append((lot.entry_id, reason))

    def _check_brackets(self, bar: Bar) -> None:
        """Stop first (conservative), then target; decided at bar.ts, the fill bar included."""
        for lot in self._symbol_lots(bar.symbol):
            if lot.exit is not None or lot.stranded:
                continue
            if lot.stop_loss is not None and bar.low <= lot.stop_loss:
                self._decide_exit(lot, "stop", bar.ts)
            elif lot.take_profit is not None and bar.high >= lot.take_profit:
                self._decide_exit(lot, "target", bar.ts)

    def _fill_exits(self, bar: Bar, budget: _BarBudget) -> list[ExecEvent]:
        out: list[ExecEvent] = []
        for lot in self._symbol_lots(bar.symbol):
            if lot.exit is None or lot.exit.not_before > bar.bar_start:
                continue
            q = min(lot.open_qty, budget.left("sell"))
            if q == 0:
                continue  # retry on the next bar
            out.append(self._fill(budget, lot.entry_id, "sell", q))
            self._reduce(lot, q)
        return out

    def _fill_sells(self, bar: Bar, budget: _BarBudget) -> list[ExecEvent]:
        out: list[ExecEvent] = []
        for w in self._symbol_working(bar.symbol, "sell"):
            if w.not_before > bar.bar_start:
                continue
            iid = w.intent.intent_id
            q = min(w.remaining, budget.left("sell"), self.position(bar.symbol))
            if q > 0:
                out.append(self._fill(budget, iid, "sell", q))
                w.remaining -= q
                self._reduce_oldest_first(bar.symbol, q)
            if w.remaining == 0:
                del self._working[iid]
            elif self.position(bar.symbol) == 0:
                del self._working[iid]
                out.append(self._update(bar.ts, iid, "cancelled", "position closed"))
        return out

    def _reduce(self, lot: Lot, qty: int) -> None:
        if not 0 < qty <= lot.open_qty:
            raise AssertionError(f"cannot sell {qty} of lot {lot.entry_id} ({lot.open_qty} open)")
        lot.open_qty -= qty
        if lot.open_qty == 0:
            del self._lots[lot.entry_id]

    def _reduce_oldest_first(self, symbol: str, qty: int) -> None:
        """A sell intent takes lots with no pending exit first (the qty it was checked
        against), oldest first, then the oldest of the lots already exiting."""
        lots = self._symbol_lots(symbol)
        for lot in [x for x in lots if x.exit is None] + [x for x in lots if x.exit is not None]:
            if qty == 0:
                break
            take = min(qty, lot.open_qty)
            self._reduce(lot, take)
            qty -= take
        if qty != 0:
            raise AssertionError(f"sell of {symbol} exceeded the position by {qty}")

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
        if self._cfg.cost_book == "refit" and not inputs.has_refit_book:
            raise ValueError(f"no refit cost book for {symbol} on {session}")
        return fill_cost_bps(side, qty, price, fill_instant, quote, inputs, self._cfg)

    # ---- helpers ------------------------------------------------------------
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _symbol_working(self, symbol: str, side: Side) -> list[WorkingOrder]:
        return sorted((w for w in self._working.values()
                       if w.symbol == symbol and w.intent.side == side), key=lambda w: w.seq)

    def _symbol_lots(self, symbol: str) -> list[Lot]:
        return sorted((lot for lot in self._lots.values() if lot.symbol == symbol),
                      key=lambda lot: lot.seq)

    def _sorted_lots(self) -> list[Lot]:
        return sorted(self._lots.values(), key=lambda lot: (lot.symbol, lot.seq))

    def _sorted_working(self, side: Side) -> list[WorkingOrder]:
        return sorted((w for w in self._working.values() if w.intent.side == side),
                      key=lambda w: (w.symbol, w.seq))

    @staticmethod
    def _update(ts: datetime, intent_id: str, status: str, detail: str) -> OrderUpdate:
        return OrderUpdate(ts=ts, intent_id=intent_id, status=status, detail=detail)
