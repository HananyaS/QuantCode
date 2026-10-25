"""Valid-by-default factories for tests. Fixed constants only: no wall clock."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from research.intraday.contracts.events import (
    Bar, Fill, OrderIntent, Quote, Signal, Window,
)

T0: datetime = datetime(2026, 3, 2, 14, 31, tzinfo=timezone.utc)
_MIN = timedelta(minutes=1)


def make_bar(**overrides: Any) -> Bar:
    fields: dict[str, Any] = dict(
        ts=T0, symbol="TEST", bar_start=T0 - _MIN, open=1.00, high=1.05, low=0.98,
        close=1.02, volume=10_000, vwap=1.01, trade_count=120, feed="sip")
    fields.update(overrides)
    return Bar(**fields)


def make_quote(**overrides: Any) -> Quote:
    fields: dict[str, Any] = dict(
        ts=T0, symbol="TEST", bid=1.00, ask=1.01, bid_size=100, ask_size=100)
    fields.update(overrides)
    return Quote(**fields)


def make_window(n: int = 3, symbol: str = "TEST", end_ts: datetime = T0) -> Window:
    """n consecutive 1-minute bars, the last one ending at end_ts."""
    bars = tuple(
        make_bar(symbol=symbol, ts=end_ts - (n - 1 - i) * _MIN,
                 bar_start=end_ts - (n - i) * _MIN)
        for i in range(n))
    return Window(ts=end_ts, symbol=symbol, bars=bars)


def make_signal(**overrides: Any) -> Signal:
    fields: dict[str, Any] = dict(
        ts=T0, symbol="TEST", score=0.5, horizon_min=15, model_hash="abc123")
    fields.update(overrides)
    return Signal(**fields)


def make_order_intent(**overrides: Any) -> OrderIntent:
    fields: dict[str, Any] = dict(
        ts=T0, intent_id="intent-1", symbol="TEST", side="buy", qty=100,
        limit_price=None, take_profit=None, stop_loss=None, time_stop=None,
        reason="test")
    fields.update(overrides)
    return OrderIntent(**fields)


def make_fill(**overrides: Any) -> Fill:
    fields: dict[str, Any] = dict(
        ts=T0, intent_id="intent-1", symbol="TEST", side="buy", qty=100,
        price=1.01, cost_bps=5.0)
    fields.update(overrides)
    return Fill(**fields)
