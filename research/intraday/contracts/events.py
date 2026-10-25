"""Frozen event contracts for the intraday engine.

Every event is immutable and validates its own invariants in ``__post_init__``.
Contracts validate; they never repair (no auto-sorting, no clamping).
All datetimes are timezone-aware UTC so codec output is canonical.

Note: with ``slots=True`` the class is rebuilt, so zero-arg ``super()`` breaks;
subclasses call ``Event.__post_init__(self)`` explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

SCHEMA_VERSION: int = 1

_ONE_MIN = timedelta(minutes=1)


def _require_utc(cls: str, name: str, dt: datetime) -> None:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError(f"{cls}.{name} must be a timezone-aware UTC datetime, got {dt!r}")


def _finite_positive(cls: str, name: str, x: float) -> None:
    if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0):
        raise ValueError(f"{cls}.{name} must be finite and > 0, got {x}")


def _non_negative(cls: str, name: str, x: float) -> None:
    if not (isinstance(x, (int, float)) and math.isfinite(x) and x >= 0):
        raise ValueError(f"{cls}.{name} must be finite and >= 0, got {x}")


def _non_empty(cls: str, name: str, s: str) -> None:
    if not isinstance(s, str) or not s:
        raise ValueError(f"{cls}.{name} must be a non-empty string, got {s!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    ts: datetime

    def __post_init__(self) -> None:
        _require_utc(type(self).__name__, "ts", self.ts)


@dataclass(frozen=True, slots=True, kw_only=True)
class ClockTick(Event):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidates(Event):
    screen_id: str
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        if list(self.symbols) != sorted(set(self.symbols)):
            raise ValueError(f"Candidates.symbols must be unique and sorted, got {self.symbols!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Bar(Event):
    symbol: str
    bar_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int
    feed: Literal["sip", "iex"]

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _require_utc("Bar", "bar_start", self.bar_start)
        if self.ts != self.bar_start + _ONE_MIN:
            raise ValueError(
                f"Bar.ts must equal bar_start + 1 minute (end-of-bar stamp), "
                f"got ts={self.ts!r} bar_start={self.bar_start!r}")
        for name in ("open", "high", "low", "close", "vwap"):
            _finite_positive("Bar", name, getattr(self, name))
        if self.low > min(self.open, self.close):
            raise ValueError(f"Bar.low must be <= min(open, close), got low={self.low}")
        if self.high < max(self.open, self.close):
            raise ValueError(f"Bar.high must be >= max(open, close), got high={self.high}")
        _non_negative("Bar", "volume", self.volume)
        _non_negative("Bar", "trade_count", self.trade_count)
        if self.feed not in ("sip", "iex"):
            raise ValueError(f"Bar.feed must be 'sip' or 'iex', got {self.feed!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Quote(Event):
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _finite_positive("Quote", "bid", self.bid)
        _finite_positive("Quote", "ask", self.ask)
        if self.bid > self.ask:
            raise ValueError(f"Quote.bid must be <= ask (crossed book), got bid={self.bid} ask={self.ask}")
        _non_negative("Quote", "bid_size", self.bid_size)
        _non_negative("Quote", "ask_size", self.ask_size)


@dataclass(frozen=True, slots=True, kw_only=True)
class Gap(Event):
    symbol: str
    missing_bar_start: datetime

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _require_utc("Gap", "missing_bar_start", self.missing_bar_start)


@dataclass(frozen=True, slots=True, kw_only=True)
class Window(Event):
    symbol: str
    bars: tuple[Bar, ...]

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        if len(self.bars) == 0:
            raise ValueError("Window.bars must be non-empty")
        if any(b.symbol != self.symbol for b in self.bars):
            raise ValueError(f"Window.bars must all have symbol {self.symbol!r}")
        starts = [b.bar_start for b in self.bars]
        if any(a >= b for a, b in zip(starts, starts[1:])):
            raise ValueError("Window.bars must have strictly increasing bar_start")
        if self.bars[-1].ts > self.ts:
            raise ValueError(
                f"Window.bars[-1].ts ({self.bars[-1].ts!r}) must be <= Window.ts ({self.ts!r})")


@dataclass(frozen=True, slots=True, kw_only=True)
class Signal(Event):
    symbol: str
    score: float
    horizon_min: int
    model_hash: str

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        if not (isinstance(self.score, (int, float)) and math.isfinite(self.score)):
            raise ValueError(f"Signal.score must be finite, got {self.score}")
        if not self.horizon_min > 0:
            raise ValueError(f"Signal.horizon_min must be > 0, got {self.horizon_min}")
        _non_empty("Signal", "model_hash", self.model_hash)


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderIntent(Event):
    intent_id: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    limit_price: float | None
    take_profit: float | None
    stop_loss: float | None
    time_stop: datetime | None
    reason: str

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _non_empty("OrderIntent", "intent_id", self.intent_id)
        if self.side not in ("buy", "sell"):
            raise ValueError(f"OrderIntent.side must be 'buy' or 'sell', got {self.side!r}")
        if not self.qty > 0:
            raise ValueError(f"OrderIntent.qty must be > 0, got {self.qty}")
        for name in ("limit_price", "take_profit", "stop_loss"):
            val = getattr(self, name)
            if val is not None:
                _finite_positive("OrderIntent", name, val)
        tp, sl = self.take_profit, self.stop_loss
        if tp is not None and sl is not None:
            if self.side == "buy" and not sl < tp:
                raise ValueError(
                    f"OrderIntent bracket for buy needs stop_loss < take_profit, got sl={sl} tp={tp}")
            if self.side == "sell" and not tp < sl:
                raise ValueError(
                    f"OrderIntent bracket for sell needs take_profit < stop_loss, got tp={tp} sl={sl}")
        if self.time_stop is not None:
            _require_utc("OrderIntent", "time_stop", self.time_stop)
            if self.time_stop <= self.ts:
                raise ValueError(
                    f"OrderIntent.time_stop must be > ts, got time_stop={self.time_stop!r} ts={self.ts!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderUpdate(Event):
    intent_id: str
    status: Literal["accepted", "rejected", "cancelled", "expired"]
    detail: str

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _non_empty("OrderUpdate", "intent_id", self.intent_id)
        if self.status not in ("accepted", "rejected", "cancelled", "expired"):
            raise ValueError(f"OrderUpdate.status invalid, got {self.status!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Fill(Event):
    intent_id: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    price: float
    cost_bps: float

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _non_empty("Fill", "intent_id", self.intent_id)
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Fill.side must be 'buy' or 'sell', got {self.side!r}")
        if not self.qty > 0:
            raise ValueError(f"Fill.qty must be > 0, got {self.qty}")
        _finite_positive("Fill", "price", self.price)
        _non_negative("Fill", "cost_bps", self.cost_bps)


@dataclass(frozen=True, slots=True, kw_only=True)
class Halt(Event):
    symbol: str
    halted: bool
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Kill(Event):
    reason: str


EVENT_TYPES: dict[str, type[Event]] = {
    cls.__name__: cls
    for cls in (ClockTick, Candidates, Bar, Quote, Gap, Window, Signal,
                OrderIntent, OrderUpdate, Fill, Halt, Kill)
}
