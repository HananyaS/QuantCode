"""Frozen event contracts for the intraday engine.

Every event is immutable and validates its own invariants in ``__post_init__``.
Contracts validate; they never repair (no auto-sorting, no clamping, no coercion).

Field types are strict, and every violation raises ``ValueError`` (never
``TypeError``/``AttributeError``):

* int fields need ``type(x) is int`` (bool and numpy ints are rejected);
* float fields need ``isinstance(x, float)`` (``np.float64`` is accepted; int
  and bool are rejected, so a producer passes ``0.0``, never ``0``);
* bool fields need ``type(x) is bool``;
* symbols and ids are non-empty ``str``; free text (reason, detail) is ``str``;
* tuple fields need ``type(v) is tuple`` with every element of the given class;
* datetimes need ``type(dt) is datetime and dt.tzinfo is timezone.utc``, which
  rejects naive values, ``pd.Timestamp`` and zero-offset zones such as
  ``ZoneInfo("Europe/London")``, so codec output is canonical.

Note: with ``slots=True`` the class is rebuilt, so zero-arg ``super()`` breaks;
subclasses call ``Event.__post_init__(self)`` explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

SCHEMA_VERSION: int = 1

_ONE_MIN = timedelta(minutes=1)


def _require_utc(cls: str, name: str, dt: Any) -> None:
    if not (type(dt) is datetime and dt.tzinfo is timezone.utc):
        raise ValueError(
            f"{cls}.{name} must be a datetime with tzinfo=timezone.utc, got {dt!r}")


def _int(cls: str, name: str, x: Any) -> None:
    if type(x) is not int:
        raise ValueError(f"{cls}.{name} must be an int, got {type(x).__name__} {x!r}")


def _float(cls: str, name: str, x: Any) -> None:
    if not isinstance(x, float):
        raise ValueError(f"{cls}.{name} must be a float, got {type(x).__name__} {x!r}")


def _bool(cls: str, name: str, x: Any) -> None:
    if type(x) is not bool:
        raise ValueError(f"{cls}.{name} must be a bool, got {type(x).__name__} {x!r}")


def _str(cls: str, name: str, s: Any) -> None:
    """Non-empty string: symbols and ids."""
    if not isinstance(s, str) or not s:
        raise ValueError(f"{cls}.{name} must be a non-empty string, got {s!r}")


def _text(cls: str, name: str, s: Any) -> None:
    """Free text (may be empty)."""
    if not isinstance(s, str):
        raise ValueError(f"{cls}.{name} must be a str, got {type(s).__name__} {s!r}")


def _tuple_of(cls: str, name: str, v: Any, elem: type) -> None:
    if type(v) is not tuple or not all(isinstance(e, elem) for e in v):
        raise ValueError(f"{cls}.{name} must be a tuple of {elem.__name__}, got {v!r}")


def _finite_positive(cls: str, name: str, x: Any) -> None:
    _float(cls, name, x)
    if not (math.isfinite(x) and x > 0):
        raise ValueError(f"{cls}.{name} must be finite and > 0, got {x}")


def _non_negative_int(cls: str, name: str, x: Any) -> None:
    _int(cls, name, x)
    if x < 0:
        raise ValueError(f"{cls}.{name} must be >= 0, got {x}")


def _positive_int(cls: str, name: str, x: Any) -> None:
    _int(cls, name, x)
    if x <= 0:
        raise ValueError(f"{cls}.{name} must be > 0, got {x}")


def _non_negative_float(cls: str, name: str, x: Any) -> None:
    _float(cls, name, x)
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{cls}.{name} must be finite and >= 0, got {x}")


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
        _str("Candidates", "screen_id", self.screen_id)
        _tuple_of("Candidates", "symbols", self.symbols, str)
        for sym in self.symbols:
            _str("Candidates", "symbols[]", sym)
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
        _str("Bar", "symbol", self.symbol)
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
        _non_negative_int("Bar", "volume", self.volume)
        _non_negative_int("Bar", "trade_count", self.trade_count)
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
        _str("Quote", "symbol", self.symbol)
        _finite_positive("Quote", "bid", self.bid)
        _finite_positive("Quote", "ask", self.ask)
        if self.bid > self.ask:
            raise ValueError(f"Quote.bid must be <= ask (crossed book), got bid={self.bid} ask={self.ask}")
        _non_negative_int("Quote", "bid_size", self.bid_size)
        _non_negative_int("Quote", "ask_size", self.ask_size)


@dataclass(frozen=True, slots=True, kw_only=True)
class Gap(Event):
    symbol: str
    missing_bar_start: datetime

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _str("Gap", "symbol", self.symbol)
        _require_utc("Gap", "missing_bar_start", self.missing_bar_start)
        if self.ts != self.missing_bar_start + _ONE_MIN:
            raise ValueError(
                f"Gap.ts must equal missing_bar_start + 1 minute (the missing bar's end), "
                f"got ts={self.ts!r} missing_bar_start={self.missing_bar_start!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Window(Event):
    symbol: str
    bars: tuple[Bar, ...]

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _str("Window", "symbol", self.symbol)
        _tuple_of("Window", "bars", self.bars, Bar)
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
        _str("Signal", "symbol", self.symbol)
        _float("Signal", "score", self.score)
        if not math.isfinite(self.score):
            raise ValueError(f"Signal.score must be finite, got {self.score}")
        _positive_int("Signal", "horizon_min", self.horizon_min)
        _str("Signal", "model_hash", self.model_hash)


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
        _str("OrderIntent", "intent_id", self.intent_id)
        _str("OrderIntent", "symbol", self.symbol)
        if self.side not in ("buy", "sell"):
            raise ValueError(f"OrderIntent.side must be 'buy' or 'sell', got {self.side!r}")
        _positive_int("OrderIntent", "qty", self.qty)
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
        _text("OrderIntent", "reason", self.reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderUpdate(Event):
    intent_id: str
    status: Literal["accepted", "rejected", "cancelled", "expired"]
    detail: str

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _str("OrderUpdate", "intent_id", self.intent_id)
        if self.status not in ("accepted", "rejected", "cancelled", "expired"):
            raise ValueError(f"OrderUpdate.status invalid, got {self.status!r}")
        _text("OrderUpdate", "detail", self.detail)


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
        _str("Fill", "intent_id", self.intent_id)
        _str("Fill", "symbol", self.symbol)
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Fill.side must be 'buy' or 'sell', got {self.side!r}")
        _positive_int("Fill", "qty", self.qty)
        _finite_positive("Fill", "price", self.price)
        _non_negative_float("Fill", "cost_bps", self.cost_bps)


@dataclass(frozen=True, slots=True, kw_only=True)
class Halt(Event):
    symbol: str
    halted: bool
    reason: str

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _str("Halt", "symbol", self.symbol)
        _bool("Halt", "halted", self.halted)
        _text("Halt", "reason", self.reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class Kill(Event):
    reason: str

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        _text("Kill", "reason", self.reason)


EVENT_TYPES: dict[str, type[Event]] = {
    cls.__name__: cls
    for cls in (ClockTick, Candidates, Bar, Quote, Gap, Window, Signal,
                OrderIntent, OrderUpdate, Fill, Halt, Kill)
}
