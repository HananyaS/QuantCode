"""RiskAgent: the only agent that turns a Signal into an OrderIntent (risk.md)."""
from __future__ import annotations

from collections import Counter, deque
from datetime import date, timedelta
from typing import Callable

from research.intraday.contracts.events import (
    Bar, Event, Fill, Halt, Kill, OrderIntent, OrderUpdate, Quote, Signal,
)
from research.intraday.risk.books import Books
from research.intraday.risk.brackets import Bracket, make_bracket, mean_true_range, true_range
from research.intraday.risk.config import RiskConfig, SkipReason
from research.intraday.risk.session import et_date, session_bounds
from research.intraday.risk.sizing import CapName, size_order

_Handler = Callable[[Event], tuple[Event, ...]]
_CLOSING_STATUSES: frozenset[str] = frozenset({"rejected", "cancelled", "expired"})


class RiskAgent:
    """One agent per session. Pure: sees events, returns events, holds no peer."""

    def __init__(self, session_date: date, config: RiskConfig | None = None) -> None:
        self.name = "risk"
        self.subscribes: tuple[type[Event], ...] = (Bar, Quote, Signal, Fill, OrderUpdate, Halt, Kill)
        self._cfg = RiskConfig() if config is None else config
        self._session_date = session_date
        self._last_entry_utc, self._flatten_utc = session_bounds(session_date, self._cfg)
        self._last_bar: dict[str, Bar] = {}
        self._trs: dict[str, deque[float]] = {}
        self._quotes: dict[str, Quote] = {}
        self._entries: dict[str, int] = {}
        self._halted: set[str] = set()
        self._n = 0
        self._books = Books(self._cfg.start_equity_usd)
        self._killed = False
        self._tripped = False
        self._skips: Counter[str] = Counter()
        self._handlers: dict[type[Event], _Handler] = {
            Bar: self._on_bar, Quote: self._on_quote, Signal: self._on_signal, Fill: self._on_fill,
            OrderUpdate: self._on_update, Halt: self._on_halt, Kill: self._on_kill,
        }

    # -- public views -------------------------------------------------------------------------
    @property
    def skip_counts(self) -> dict[str, int]:
        return {k: v for k, v in sorted(self._skips.items()) if v}

    @property
    def books(self) -> Books:
        return self._books

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def loss_limit_tripped(self) -> bool:
        return self._tripped

    def on_event(self, event: Event) -> tuple[Event, ...]:
        handler = self._handlers.get(type(event))
        return () if handler is None else handler(event)

    # -- market data --------------------------------------------------------------------------
    def _require_session(self, event: Event) -> None:
        if et_date(event.ts) != self._session_date:
            raise ValueError(
                f"{type(event).__name__} at {event.ts.isoformat()} is not in session {self._session_date}")

    def _on_bar(self, event: Event) -> tuple[Event, ...]:
        assert isinstance(event, Bar)
        self._require_session(event)
        prev = self._last_bar.get(event.symbol)
        if prev is not None and event.ts <= prev.ts:
            raise ValueError(
                f"Bar for {event.symbol!r} at {event.ts.isoformat()} is not after {prev.ts.isoformat()}")
        trs = self._trs.setdefault(event.symbol, deque(maxlen=self._cfg.atr_period))
        trs.append(true_range(event, None if prev is None else prev.close))
        self._last_bar[event.symbol] = event
        self._check_loss_limit()
        return ()

    def _on_quote(self, event: Event) -> tuple[Event, ...]:
        assert isinstance(event, Quote)
        self._quotes[event.symbol] = event
        return ()

    # -- entries ------------------------------------------------------------------------------
    def _on_signal(self, event: Event) -> tuple[Event, ...]:
        assert isinstance(event, Signal)
        self._require_session(event)
        skip = self._gate(event)
        if skip is not None:
            self._skips[skip] += 1
            return ()
        bar = self._last_bar[event.symbol]
        sized = self._size_and_bracket(event, bar)
        if isinstance(sized, str):
            self._skips[sized] += 1
            return ()
        qty, cap, bracket = sized
        return (self._emit(event, bar.close, qty, cap, bracket),)

    def _gate(self, s: Signal) -> SkipReason | None:
        """Rules 1-7 of risk.md, in order; the first failure is the skip reason."""
        if self._killed:
            return "killed"
        if self._tripped:
            return "loss_limit"
        if s.symbol in self._halted:
            return "halted"
        if s.ts >= self._last_entry_utc:
            return "after_last_entry"
        if s.score < self._cfg.entry_threshold:
            return "below_threshold"
        return self._slot_gate(s)

    def _slot_gate(self, s: Signal) -> SkipReason | None:
        if self._books.is_busy(s.symbol):
            return "symbol_busy"
        if self._entries.get(s.symbol, 0) >= self._cfg.max_entries_per_symbol:
            return "symbol_entries"
        if self._books.occupied() >= self._cfg.max_open_positions:
            return "max_open"
        if s.symbol not in self._last_bar:
            return "no_bar"
        return None

    def _size_and_bracket(self, s: Signal, bar: Bar) -> tuple[int, CapName, Bracket] | SkipReason:
        """Rule 8: qty >= 1 and a valid bracket."""
        ref = bar.close
        qty, cap = size_order(
            ref_price=ref, available_cash=self._books.cash - self._books.reserved_cash(),
            signal_ts=s.ts, bar_volume=bar.volume, quote=self._quotes.get(s.symbol), cfg=self._cfg)
        if qty < 1:
            return "zero_qty"
        bracket = make_bracket(ref, mean_true_range(self._trs[s.symbol], self._cfg.atr_period), self._cfg)
        if isinstance(bracket, str):
            return bracket
        return qty, cap, bracket

    def _emit(self, s: Signal, ref: float, qty: int, cap: CapName, b: Bracket) -> OrderIntent:
        self._n += 1
        intent_id = f"{self._session_date:%Y%m%d}-{s.symbol}-{self._n}"
        hold = s.horizon_min if self._cfg.hold_min is None else self._cfg.hold_min
        time_stop = min(s.ts + timedelta(minutes=hold), self._flatten_utc)
        assert time_stop > s.ts
        reason = f"entry score={s.score:.6g} thr={self._cfg.entry_threshold:.6g} cap={cap}"
        intent = OrderIntent(
            ts=s.ts, intent_id=intent_id, symbol=s.symbol, side="buy", qty=qty, limit_price=None,
            take_profit=b.take_profit, stop_loss=b.stop_loss, time_stop=time_stop, reason=reason)
        self._books.open_intent(intent_id, s.symbol, qty, ref)
        self._entries[s.symbol] = self._entries.get(s.symbol, 0) + 1
        return intent

    # -- account ------------------------------------------------------------------------------
    def _on_fill(self, event: Event) -> tuple[Event, ...]:
        assert isinstance(event, Fill)
        self._books.apply_fill(event)
        self._check_loss_limit()
        return ()

    def _on_update(self, event: Event) -> tuple[Event, ...]:
        assert isinstance(event, OrderUpdate)
        self._books.require_known(event.intent_id)
        if event.status in _CLOSING_STATUSES:
            self._books.close_working(event.intent_id)
        return ()

    def _on_halt(self, event: Event) -> tuple[Event, ...]:
        return ()

    def _on_kill(self, event: Event) -> tuple[Event, ...]:
        return ()

    def _check_loss_limit(self) -> None:
        return None
