"""MonitorAgent tests for intraday-ops, Phase 2: halts, stale feed, order flow, latch."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Sequence

import pytest

from research.intraday.contracts import Bar, Candidates, ClockTick, Event, Gap, Halt
from research.intraday.contracts.fixtures import T_DEV, make_bar, make_candidates, make_gap
from research.intraday.ops.halts import HaltRow, RecordedHaltLog
from research.intraday.ops.monitor import MonitorAgent

MIN = timedelta(minutes=1)
OPEN = T_DEV - MIN                     # 2024-03-04 14:30 UTC = 09:30 EST, first bar_start


def at(m: int) -> datetime:
    return OPEN + m * MIN      # at(1) == T_DEV, the 09:31 tick


def tick(m: int) -> ClockTick:
    return ClockTick(ts=at(m))


def bar(sym: str, m: int) -> Bar:
    return make_bar(symbol=sym, ts=at(m), bar_start=at(m - 1))


def gap(sym: str, m: int) -> Gap:
    return make_gap(symbol=sym, ts=at(m), missing_bar_start=at(m - 1))


def cands(*syms: str, m: int = -5) -> Candidates:
    return make_candidates(ts=at(m), symbols=tuple(sorted(syms)))


def monitor(rows: Sequence[HaltRow] = (), **kw: Any) -> MonitorAgent:
    return MonitorAgent(RecordedHaltLog(tuple(rows)), **kw)


def feed(mon: MonitorAgent, *events: Event) -> list[Event]:
    return [out for e in events for out in mon.on_event(e)]


# --- Task 4: halt source and halt release ---

def test_halt_rows_released_at_first_tick_at_or_after_row_time() -> None:
    rows = [HaltRow(at(10) + timedelta(seconds=30), "AAA", True, "LUDP"),
            HaltRow(at(15), "AAA", False, "LUDP")]
    mon = monitor(rows)
    assert feed(mon, tick(10)) == []
    assert feed(mon, tick(11)) == [Halt(ts=at(11), symbol="AAA", halted=True, reason="LUDP")]
    assert feed(mon, tick(14)) == []
    assert feed(mon, tick(15)) == [Halt(ts=at(15), symbol="AAA", halted=False, reason="LUDP")]


def test_halt_ts_equals_tick_ts() -> None:
    mon = monitor([HaltRow(at(-20), "BBB", True, "T1")])     # before the first tick seen
    (h,) = feed(mon, tick(-5))
    assert isinstance(h, Halt) and h.ts == at(-5)


def test_resume_without_halt_skipped() -> None:
    mon = monitor([HaltRow(at(2), "AAA", False, "LUDP"),
                   HaltRow(at(3), "AAA", True, "LUDP"), HaltRow(at(3), "AAA", True, "LUDP")])
    assert feed(mon, tick(2)) == []
    assert [e.halted for e in feed(mon, tick(3))] == [True]
    assert mon.counts == {"skipped_resume": 1, "skipped_duplicate_halt": 1}


def test_out_of_order_halt_log_raises() -> None:
    mon = monitor([HaltRow(at(5), "AAA", True, "x"), HaltRow(at(3), "BBB", True, "x")])
    with pytest.raises(ValueError, match="out of time order"):
        feed(mon, tick(5))


def test_halt_row_validates_fields() -> None:
    for bad in (dict(ts=at(1).replace(tzinfo=None)), dict(symbol=""), dict(halted=1)):
        with pytest.raises(ValueError):
            HaltRow(**{**dict(ts=at(1), symbol="AAA", halted=True, reason="x"), **bad})
