"""MonitorAgent tests for intraday-ops, Phase 2: halts, stale feed, order flow, latch."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from research.intraday.contracts import Bar, Candidates, ClockTick, Event, Gap, Halt, Kill
from research.intraday.contracts.fixtures import (
    T_DEV, make_bar, make_candidates, make_fill, make_gap, make_order_intent, make_order_update,
)
from research.intraday.engine.replay import clock_ticks, run_replay
from research.intraday.ops.halts import HaltRow, RecordedHaltLog
from research.intraday.ops.monitor import MonitorAgent
from research.intraday.ops.recorder import read_log

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


# --- Task 5: stale-feed kill ---

def test_first_session_tick_does_not_kill() -> None:
    mon = monitor()
    assert feed(mon, tick(-5), cands("AAA"), tick(1)) == []   # no bar has arrived yet


def test_stale_feed_emits_kill() -> None:
    mon = monitor()
    evs: list[Event] = [tick(-5), cands("AAA", "BBB")]
    for m in range(1, 6):
        evs += [tick(m), bar("AAA", m), gap("BBB", m)]
    assert feed(mon, *evs) == []
    assert feed(mon, tick(6), bar("BBB", 6), tick(7)) == []
    out = feed(mon, tick(8))                                     # AAA last = at(5): at(7) - at(5) = 2
    assert out == [Kill(ts=at(8), reason=(f"stale_feed: AAA last={at(5).isoformat()} "
                                          f"at {at(8).isoformat()} (>= 2 min)"))]


def test_stale_check_at_tick_uses_previous_minute() -> None:
    mon = monitor(stale_after_min=1)       # evaluated against T it would kill at every tick
    evs = [tick(-5), cands("AAA")] + [e for m in range(1, 31) for e in (tick(m), bar("AAA", m))]
    assert feed(mon, *evs) == []
    out = feed(monitor(stale_after_min=2), tick(-5), cands("AAA"), tick(1), tick(2), tick(3))
    assert [(type(e), e.ts) for e in out] == [(Kill, at(3))]   # at(2) - at(0) = 2; not at tick(2)


def test_full_coverage_never_kills() -> None:
    mon = monitor()
    evs: list[Event] = [tick(m) for m in range(-5, 1)] + [cands("AAA", "BBB", m=0)]
    for m in range(1, 391):                # bars 09:30..15:59, ticks to 16:00
        evs += [tick(m), bar("AAA", m), gap("BBB", m) if m % 7 == 0 else bar("BBB", m)]
    evs += [tick(391), tick(392), tick(393)]   # after the session: no check
    assert feed(mon, *evs) == []


def test_stale_baseline_follows_dst() -> None:
    open_ = datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc)   # 09:30 EDT
    t = lambda m: ClockTick(ts=open_ + m * MIN)  # noqa: E731
    mon = monitor()
    c = make_candidates(ts=open_ - 5 * MIN, symbols=("AAA",))
    assert feed(mon, t(-5), c, t(1), t(2)) == []
    assert [type(e) for e in feed(mon, t(3))] == [Kill]


def test_late_candidates_use_their_ts_as_baseline() -> None:
    mon = monitor()
    assert feed(mon, tick(60), cands("AAA", m=60), tick(61), tick(62)) == []
    (k,) = feed(mon, tick(63))
    assert k.reason.startswith(f"stale_feed: AAA last={at(60).isoformat()}")


def test_second_candidates_raises() -> None:
    mon = monitor()
    feed(mon, tick(-5), cands("AAA"))
    with pytest.raises(ValueError):
        feed(mon, cands("BBB", m=-4))


def test_second_session_date_raises() -> None:
    mon = monitor()
    feed(mon, tick(1))
    with pytest.raises(ValueError, match="one session"):
        feed(mon, ClockTick(ts=at(1) + timedelta(days=1)))


# --- Task 6: order-flow consistency kill ---

def test_fill_for_unknown_intent_kills() -> None:
    (k,) = feed(monitor(), make_fill(ts=T_DEV, intent_id="ghost"))
    assert k == Kill(ts=T_DEV, reason="unknown_intent: ghost")


def test_overfill_kills() -> None:
    mon = monitor()
    assert feed(mon, make_order_intent(ts=T_DEV, qty=100), make_fill(ts=T_DEV, qty=60)) == []
    (k,) = feed(mon, make_fill(ts=T_DEV + MIN, qty=50))
    assert k == Kill(ts=T_DEV + MIN, reason="overfill: intent-1 filled 110 > qty 100")


def test_partial_fill_then_cancel_books_filled_qty() -> None:
    mon = monitor()
    assert feed(mon, make_order_intent(ts=T_DEV, qty=100),
                make_fill(ts=T_DEV + MIN, qty=40),                       # Fill before cancel,
                make_order_update(ts=T_DEV + MIN, status="cancelled"),  # same cascade
                make_fill(ts=T_DEV + 5 * MIN, side="sell", qty=40)) == []   # exit of the 40
    (k,) = feed(mon, make_fill(ts=T_DEV + 6 * MIN, side="sell", qty=1))
    assert k.reason == "short_sell: TEST position -1"


def test_entry_fill_after_cancel_kills() -> None:
    mon = monitor()
    feed(mon, make_order_intent(ts=T_DEV), make_order_update(ts=T_DEV, status="cancelled"))
    (k,) = feed(mon, make_fill(ts=T_DEV + MIN, qty=10))
    assert k.reason == "fill_after_close: intent-1"


def test_exit_fill_on_entry_intent_is_not_an_overfill() -> None:
    mon = monitor()
    assert feed(mon, make_order_intent(ts=T_DEV, qty=100),
                make_fill(ts=T_DEV + MIN, qty=100),
                make_fill(ts=T_DEV + 9 * MIN, side="sell", qty=100)) == []


def test_short_creating_sell_kills() -> None:
    mon = monitor()
    feed(mon, make_order_intent(ts=T_DEV, intent_id="s-1", side="sell", qty=10))
    (k,) = feed(mon, make_fill(ts=T_DEV + MIN, intent_id="s-1", side="sell", qty=10))
    assert k == Kill(ts=T_DEV + MIN, reason="short_sell: TEST position -10")


def test_duplicate_intent_id_kills() -> None:
    mon = monitor()
    assert feed(mon, make_order_intent(ts=T_DEV)) == []
    (k,) = feed(mon, make_order_intent(ts=T_DEV + MIN))
    assert k == Kill(ts=T_DEV + MIN, reason="duplicate_intent: intent-1")


# --- Task 7: kill latch, gap-inferred halts, replay determinism ---

def test_at_most_one_kill_and_silence_after() -> None:
    mon = monitor([HaltRow(at(5), "AAA", True, "x")])
    assert len(feed(mon, make_fill(ts=at(1), intent_id="ghost"))) == 1
    assert feed(mon, make_fill(ts=at(2), intent_id="ghost2"), cands("AAA", m=2),
                *[tick(m) for m in range(3, 10)]) == []          # no 2nd Kill, no Halt


def test_external_kill_silences_monitor() -> None:
    mon = monitor()
    assert feed(mon, Kill(ts=at(1), reason="operator")) == []
    assert feed(mon, make_fill(ts=at(2), intent_id="ghost")) == []


def test_gap_inference_off_by_default() -> None:
    mon = monitor()
    evs = [tick(-5), cands("AAA")] + [e for m in range(1, 8) for e in (tick(m), gap("AAA", m))]
    assert feed(mon, *evs) == []


def test_gap_inference_emits_halt_and_resume() -> None:
    mon = monitor(infer_halts_from_gaps=True, gap_halt_min=5)
    evs = [tick(-5), cands("AAA")] + [e for m in range(1, 6) for e in (tick(m), gap("AAA", m))]
    assert feed(mon, *evs) == [Halt(ts=at(5), symbol="AAA", halted=True,
                                    reason="inferred: 5 consecutive gaps")]
    assert feed(mon, tick(6), bar("AAA", 6)) == [
        Halt(ts=at(6), symbol="AAA", halted=False, reason="inferred: bar after gaps")]


def test_monitor_in_replay_identical_hash(tmp_path: Path) -> None:
    rows = (HaltRow(at(3), "AAA", True, "LUDP"), HaltRow(at(8), "AAA", False, "LUDP"))

    def sources() -> list:
        return [clock_ticks(at(-5), at(12)), [cands("AAA", "BBB")],
                [b for m in range(1, 12) for b in (bar("AAA", m), bar("BBB", m))]]
    h1 = run_replay([MonitorAgent(RecordedHaltLog(rows))], sources(), tmp_path / "a.jsonl")
    h2 = run_replay([MonitorAgent(RecordedHaltLog(rows))], sources(), tmp_path / "b.jsonl")
    assert h1 == h2
    halts = [e.event for e in read_log(tmp_path / "a.jsonl") if isinstance(e.event, Halt)]
    assert [(h.ts, h.halted) for h in halts] == [(at(3), True), (at(8), False)]
    assert not any(isinstance(e.event, Kill) for e in read_log(tmp_path / "a.jsonl"))


def test_halt_log_halt_takes_over_inferred_halt() -> None:
    mon = monitor([HaltRow(at(6), "AAA", True, "LUDP")], infer_halts_from_gaps=True)
    evs = [tick(-5), cands("AAA")] + [e for m in range(1, 6) for e in (tick(m), gap("AAA", m))]
    assert [h.reason for h in feed(mon, *evs)] == ["inferred: 5 consecutive gaps"]
    assert feed(mon, tick(6)) == []                     # duplicate halt: skipped, counted
    assert mon.counts["skipped_duplicate_halt"] == 1
    assert feed(mon, bar("AAA", 6)) == []               # the log, not a bar, ends this halt


# --- Fix round 1 ---

def test_bar_before_candidates_does_not_predate_baseline() -> None:
    mon = monitor()
    assert feed(mon, tick(-5), bar("AAA", -4), cands("AAA", m=-3), tick(1), tick(2)) == []
    (k,) = feed(mon, tick(3))                                    # at(2) - baseline at(0) = 2
    assert k.reason.startswith(f"stale_feed: AAA last={at(0).isoformat()}")


def test_gap_inference_fires_on_run_crossing_threshold_after_log_resume() -> None:
    rows = [HaltRow(at(3), "AAA", True, "LUDP"), HaltRow(at(6), "AAA", False, "LUDP")]
    mon = monitor(rows, infer_halts_from_gaps=True, gap_halt_min=5)
    evs = [tick(-5), cands("AAA")] + [e for m in range(1, 9) for e in (tick(m), gap("AAA", m))]
    assert [(h.ts, h.halted, h.reason) for h in feed(mon, *evs)] == [
        (at(3), True, "LUDP"), (at(6), False, "LUDP"),
        (at(6), True, "inferred: 5 consecutive gaps")]       # run is 6 at gap(6): >= 5
