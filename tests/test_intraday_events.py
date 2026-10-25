"""Invariant checks for research.intraday.contracts events."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from research.intraday.contracts import (
    EVENT_TYPES, SCHEMA_VERSION, Bar, Candidates, ClockTick, Event, Fill, Gap,
    Halt, Kill, OrderIntent, OrderUpdate, Quote, Signal, Window,
)
from research.intraday.contracts.fixtures import (
    T0, make_bar, make_fill, make_order_intent, make_quote, make_signal,
    make_window,
)

MIN = timedelta(minutes=1)


def all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        # slots=True rebuilds each class; the stale pre-slots original stays in
        # __subclasses__() but has no own __slots__, so skip it.
        if "__slots__" in sub.__dict__:
            out.add(sub)
        out |= all_subclasses(sub)
    return out


def test_schema_version() -> None:
    assert SCHEMA_VERSION == 1


def test_event_rejects_naive_and_non_utc_ts() -> None:
    with pytest.raises(ValueError, match="ClockTick.ts"):
        ClockTick(ts=datetime(2026, 3, 2, 14, 31))
    with pytest.raises(ValueError, match="ClockTick.ts"):
        ClockTick(ts=T0.astimezone(ZoneInfo("America/New_York")))
    assert ClockTick(ts=T0).ts == T0


def test_subclass_checks_ts_via_super() -> None:
    with pytest.raises(ValueError, match="Kill.ts"):
        Kill(ts=datetime(2026, 3, 2), reason="x")


def test_other_datetime_fields_must_be_utc() -> None:
    ny = ZoneInfo("America/New_York")
    with pytest.raises(ValueError, match="bar_start"):
        make_bar(bar_start=(T0 - MIN).astimezone(ny))
    with pytest.raises(ValueError, match="missing_bar_start"):
        Gap(ts=T0, symbol="TEST", missing_bar_start=datetime(2026, 3, 2, 14, 30))
    with pytest.raises(ValueError, match="time_stop"):
        make_order_intent(time_stop=(T0 + 5 * MIN).astimezone(ny))


def test_bar_ts_is_end_of_bar() -> None:
    with pytest.raises(ValueError, match="Bar.ts"):
        make_bar(ts=T0, bar_start=T0)
    assert make_bar(ts=T0, bar_start=T0 - MIN).ts == T0


def test_bar_rejects_nan() -> None:
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="Bar.close must be finite and > 0"):
            make_bar(close=bad)


def test_bar_rejects_inconsistent_ohlc() -> None:
    with pytest.raises(ValueError):
        make_bar(open=2.0, high=1.9, low=1.8, close=1.85)
    with pytest.raises(ValueError, match="Bar.low"):
        make_bar(low=1.03)
    with pytest.raises(ValueError, match="Bar.volume"):
        make_bar(volume=-1)
    assert make_bar().feed == "sip"


def test_quote_rejects_crossed_book() -> None:
    with pytest.raises(ValueError):
        make_quote(bid=1.01, ask=1.00)
    with pytest.raises(ValueError, match="Quote.bid"):
        make_quote(bid=0.0)
    with pytest.raises(ValueError, match="Quote.ask_size"):
        make_quote(ask_size=-1)
    make_quote(bid=1.0, ask=1.0)


def test_candidates_unique_and_sorted() -> None:
    assert Candidates(ts=T0, screen_id="s", symbols=("A", "B")).symbols == ("A", "B")
    with pytest.raises(ValueError, match="Candidates.symbols"):
        Candidates(ts=T0, screen_id="s", symbols=("B", "A"))
    with pytest.raises(ValueError, match="Candidates.symbols"):
        Candidates(ts=T0, screen_id="s", symbols=("A", "A"))


def test_window_rejects_future_and_unsorted_bars() -> None:
    w = make_window(n=3)
    assert len(w.bars) == 3 and w.bars[-1].ts == w.ts
    with pytest.raises(ValueError, match="Window.bars"):
        Window(ts=T0, symbol="TEST", bars=())
    with pytest.raises(ValueError, match="Window.bars"):  # future bar
        Window(ts=T0 - MIN, symbol="TEST", bars=w.bars)
    with pytest.raises(ValueError, match="Window.bars"):  # unsorted
        Window(ts=T0, symbol="TEST", bars=tuple(reversed(w.bars)))
    with pytest.raises(ValueError, match="Window.bars"):  # duplicate
        Window(ts=T0, symbol="TEST", bars=(w.bars[0], w.bars[0]))
    with pytest.raises(ValueError, match="Window.bars"):  # wrong symbol
        Window(ts=T0, symbol="OTHER", bars=w.bars)
    # gaps are legal
    gapped = (make_bar(bar_start=T0 - 5 * MIN, ts=T0 - 4 * MIN), make_bar())
    Window(ts=T0, symbol="TEST", bars=gapped)


def test_signal_invariants() -> None:
    make_signal()
    with pytest.raises(ValueError, match="Signal.score"):
        make_signal(score=float("nan"))
    with pytest.raises(ValueError, match="Signal.horizon_min"):
        make_signal(horizon_min=0)
    with pytest.raises(ValueError, match="Signal.model_hash"):
        make_signal(model_hash="")


def test_order_intent_bracket_and_time_stop_validation() -> None:
    make_order_intent()
    make_order_intent(side="buy", take_profit=1.1, stop_loss=0.9)
    make_order_intent(side="sell", take_profit=0.9, stop_loss=1.1)
    with pytest.raises(ValueError, match="OrderIntent"):
        make_order_intent(side="buy", take_profit=0.9, stop_loss=1.1)
    with pytest.raises(ValueError, match="OrderIntent"):
        make_order_intent(side="sell", take_profit=1.1, stop_loss=0.9)
    with pytest.raises(ValueError, match="OrderIntent.qty"):
        make_order_intent(qty=0)
    with pytest.raises(ValueError, match="OrderIntent.intent_id"):
        make_order_intent(intent_id="")
    for field in ("limit_price", "take_profit", "stop_loss"):
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match=f"OrderIntent.{field}"):
                make_order_intent(**{field: bad})
    with pytest.raises(ValueError, match="OrderIntent.time_stop"):
        make_order_intent(time_stop=T0)
    with pytest.raises(ValueError, match="OrderIntent.time_stop"):
        make_order_intent(time_stop=datetime(2026, 3, 2, 15, 0))
    make_order_intent(time_stop=T0 + MIN)


def test_order_update_fill_halt_kill() -> None:
    OrderUpdate(ts=T0, intent_id="i", status="accepted", detail="")
    make_fill()
    with pytest.raises(ValueError, match="Fill.qty"):
        make_fill(qty=0)
    with pytest.raises(ValueError, match="Fill.price"):
        make_fill(price=0.0)
    with pytest.raises(ValueError, match="Fill.cost_bps"):
        make_fill(cost_bps=-0.1)
    Halt(ts=T0, symbol="TEST", halted=True, reason="LULD")
    Kill(ts=T0, reason="manual")


def test_events_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_bar().close = 9.0  # type: ignore[misc]


def test_event_types_registry_covers_every_subclass() -> None:
    assert set(EVENT_TYPES.values()) == all_subclasses(Event)
    assert all(k == v.__name__ for k, v in EVENT_TYPES.items())
    assert Event not in EVENT_TYPES.values()
    assert len(EVENT_TYPES) == 12


# ---- strict field types (final-review fix A1-A3, A8) -----------------------
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.intraday.contracts.fixtures import (  # noqa: E402
    T_DEV, make_candidates, make_clock_tick, make_gap, make_halt, make_kill,
    make_order_update,
)


def test_new_fixtures_build_valid_events() -> None:
    assert isinstance(make_clock_tick(), ClockTick)
    assert isinstance(make_candidates(), Candidates)
    assert isinstance(make_gap(), Gap)
    assert isinstance(make_order_update(), OrderUpdate)
    assert isinstance(make_halt(), Halt)
    assert isinstance(make_kill(), Kill)
    assert make_clock_tick(ts=T_DEV).ts == T_DEV


def test_t_dev_is_pre_holdout_and_t0_unchanged() -> None:
    assert T_DEV == datetime(2024, 3, 4, 14, 31, tzinfo=timezone.utc)
    assert T0 == datetime(2026, 3, 2, 14, 31, tzinfo=timezone.utc)


def test_utc_must_be_the_timezone_utc_singleton() -> None:
    with pytest.raises(ValueError, match="ClockTick.ts"):
        ClockTick(ts=pd.Timestamp(T0))
    with pytest.raises(ValueError, match="ClockTick.ts"):
        ClockTick(ts=T0.astimezone(ZoneInfo("Europe/London")))  # offset 0 in March
    with pytest.raises(ValueError, match="ClockTick.ts"):
        ClockTick(ts=T0.replace(tzinfo=ZoneInfo("UTC")))
    with pytest.raises(ValueError, match="bar_start"):
        make_bar(bar_start=pd.Timestamp(T0 - MIN))


def test_gap_ts_is_missing_bar_start_plus_one_minute() -> None:
    make_gap(ts=T0, missing_bar_start=T0 - MIN)
    with pytest.raises(ValueError, match="Gap.ts"):
        make_gap(ts=T0, missing_bar_start=T0)
    with pytest.raises(ValueError, match="Gap.ts"):
        make_gap(ts=T0, missing_bar_start=T0 - 2 * MIN)


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "vwap"])
def test_bar_float_fields_reject_int_and_bool(field: str) -> None:
    for bad in (1, True, "1.0"):
        with pytest.raises(ValueError, match=f"Bar.{field}"):
            make_bar(**{field: bad})
    make_bar(**{field: np.float64(getattr(make_bar(), field))})


@pytest.mark.parametrize("field", ["volume", "trade_count"])
def test_bar_int_fields_reject_float_bool_numpy(field: str) -> None:
    for bad in (10.0, True, np.int64(10), "10"):
        with pytest.raises(ValueError, match=f"Bar.{field}"):
            make_bar(**{field: bad})


def test_symbols_must_be_non_empty_str() -> None:
    for bad in ("", 1, None):
        with pytest.raises(ValueError, match="Bar.symbol"):
            make_bar(symbol=bad)
        with pytest.raises(ValueError, match="Quote.symbol"):
            make_quote(symbol=bad)
        with pytest.raises(ValueError, match="Gap.symbol"):
            make_gap(symbol=bad)
        with pytest.raises(ValueError, match="Signal.symbol"):
            make_signal(symbol=bad)
        with pytest.raises(ValueError, match="OrderIntent.symbol"):
            make_order_intent(symbol=bad)
        with pytest.raises(ValueError, match="Fill.symbol"):
            make_fill(symbol=bad)
        with pytest.raises(ValueError, match="Halt.symbol"):
            make_halt(symbol=bad)


def test_quote_types() -> None:
    with pytest.raises(ValueError, match="Quote.bid"):
        make_quote(bid=1)
    with pytest.raises(ValueError, match="Quote.ask"):
        make_quote(ask=True)
    with pytest.raises(ValueError, match="Quote.bid_size"):
        make_quote(bid_size=100.0)
    with pytest.raises(ValueError, match="Quote.ask_size"):
        make_quote(ask_size=False)


def test_candidates_types() -> None:
    with pytest.raises(ValueError, match="Candidates.screen_id"):
        make_candidates(screen_id="")
    with pytest.raises(ValueError, match="Candidates.symbols"):
        make_candidates(symbols=["A", "B"])
    with pytest.raises(ValueError, match="Candidates.symbols"):
        make_candidates(symbols=("A", 1))
    with pytest.raises(ValueError, match="Candidates.symbols"):
        make_candidates(symbols=("", "A"))


def test_window_bars_must_be_tuple_of_bar() -> None:
    w = make_window(n=2)
    with pytest.raises(ValueError, match="Window.bars"):
        Window(ts=T0, symbol="TEST", bars=list(w.bars))
    with pytest.raises(ValueError, match="Window.bars"):
        Window(ts=T0, symbol="TEST", bars=(w.bars[0], make_quote()))
    with pytest.raises(ValueError, match="Window.symbol"):
        Window(ts=T0, symbol="", bars=w.bars)


def test_signal_types() -> None:
    with pytest.raises(ValueError, match="Signal.score"):
        make_signal(score=1)
    with pytest.raises(ValueError, match="Signal.score"):
        make_signal(score=True)
    make_signal(score=np.float64(0.25))
    with pytest.raises(ValueError, match="Signal.horizon_min"):
        make_signal(horizon_min=15.0)
    with pytest.raises(ValueError, match="Signal.model_hash"):
        make_signal(model_hash=1)


def test_order_intent_types() -> None:
    with pytest.raises(ValueError, match="OrderIntent.qty"):
        make_order_intent(qty=100.0)
    with pytest.raises(ValueError, match="OrderIntent.qty"):
        make_order_intent(qty=True)
    with pytest.raises(ValueError, match="OrderIntent.limit_price"):
        make_order_intent(limit_price=2)
    with pytest.raises(ValueError, match="OrderIntent.reason"):
        make_order_intent(reason=None)
    with pytest.raises(ValueError, match="OrderIntent.time_stop"):
        make_order_intent(time_stop=pd.Timestamp(T0 + MIN))


def test_order_update_types() -> None:
    with pytest.raises(ValueError, match="OrderUpdate.detail"):
        make_order_update(detail=None)
    with pytest.raises(ValueError, match="OrderUpdate.status"):
        make_order_update(status="filled")
    with pytest.raises(ValueError, match="OrderUpdate.intent_id"):
        make_order_update(intent_id=7)


def test_fill_types_cost_bps_zero_int_rejected() -> None:
    with pytest.raises(ValueError, match="Fill.cost_bps"):
        make_fill(cost_bps=0)
    make_fill(cost_bps=0.0)
    with pytest.raises(ValueError, match="Fill.price"):
        make_fill(price=1)
    with pytest.raises(ValueError, match="Fill.qty"):
        make_fill(qty=100.0)


def test_halt_and_kill_types() -> None:
    with pytest.raises(ValueError, match="Halt.halted"):
        make_halt(halted=1)
    with pytest.raises(ValueError, match="Halt.halted"):
        make_halt(halted=np.bool_(True))
    with pytest.raises(ValueError, match="Halt.reason"):
        make_halt(reason=None)
    with pytest.raises(ValueError, match="Kill.reason"):
        make_kill(reason=3)


def test_clock_tick_ts_type() -> None:
    with pytest.raises(ValueError, match="ClockTick.ts"):
        ClockTick(ts="2026-03-02T14:31:00+00:00")
