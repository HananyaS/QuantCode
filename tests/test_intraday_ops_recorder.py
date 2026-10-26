"""Recorder (log format 2) and run-manifest tests for intraday-ops, Phase 2."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import pytest

from research.intraday.contracts import EVENT_TYPES, Agent, Bar, Envelope, Event
from research.intraday.contracts.codec import canonical_json
from research.intraday.contracts.fixtures import (
    T0, make_bar, make_candidates, make_clock_tick, make_fill, make_gap, make_halt,
    make_kill, make_order_intent, make_order_update, make_quote, make_signal,
    make_window,
)
from research.intraday.engine.bus import EXTERNAL, EventBus
from research.intraday.ops.recorder import (
    EXTERNAL_SOURCE, JsonlRecorder, external_envelopes, read_log,
)

MIN = timedelta(minutes=1)


class ToyAgent:  # emits one Signal per Bar
    name = "toy"
    subscribes: tuple[type[Event], ...] = (Bar,)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        return (make_signal(ts=event.ts, symbol=event.symbol),)


def _publish_all(path: Path, events: Sequence[Event], agents: Sequence[Agent] = ()) -> list[Envelope]:
    with JsonlRecorder(path) as rec:
        bus = EventBus(list(agents), sink=rec)
        return [env for e in events for env in bus.publish(e)]


def _rewrite(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# --- Task 1: strict reader, every-type round trip, nightly-replay extractor ---

def test_external_source_matches_bus() -> None:
    assert EXTERNAL_SOURCE == EXTERNAL  # engine.bus.EXTERNAL


def test_seq_contiguous_from_zero_in_log(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    bars = [make_bar(ts=T0 + i * MIN, bar_start=T0 + (i - 1) * MIN) for i in range(3)]
    _publish_all(p, bars, [ToyAgent()])
    assert [e.seq for e in read_log(p)] == list(range(6))
    lines = p.read_text(encoding="utf-8").splitlines()
    q = tmp_path / "hole.jsonl"
    _rewrite(q, [lines[0], *lines[2:]])            # drop seq 1
    with pytest.raises(ValueError, match="line 2"):
        list(read_log(q))


def test_cause_seq_points_to_earlier_line(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    envs = _publish_all(p, [make_bar()], [ToyAgent()])
    assert envs[1].cause_seq == 0 and envs[0].cause_seq is None
    lines = p.read_text(encoding="utf-8").splitlines()
    for bad in (1, -1):                                 # itself; before the log
        rec = json.loads(lines[1])
        rec["cause_seq"] = bad
        q = tmp_path / f"cause{bad}.jsonl"
        _rewrite(q, [lines[0], canonical_json(rec)])
        with pytest.raises(ValueError, match="line 2"):
            list(read_log(q))


def test_every_event_type_round_trips(tmp_path: Path) -> None:
    w = make_window(3)                      # bars end T0-2m, T0-1m, T0; logged first
    inline_w = make_window(2, symbol="ZZZ")  # its bars are never logged
    bracket = make_order_intent(intent_id="intent-2", limit_price=1.0, take_profit=1.2,
                                stop_loss=0.9, time_stop=T0 + 15 * MIN)
    events = [*w.bars, make_clock_tick(), make_candidates(), make_quote(), make_gap(),
              w, inline_w, make_signal(), make_order_intent(), bracket, make_order_update(),
              make_fill(), make_halt(), make_kill()]
    envs = _publish_all(tmp_path / "all.jsonl", events)
    assert {type(e.event).__name__ for e in envs} == set(EVENT_TYPES)
    assert list(read_log(tmp_path / "all.jsonl")) == envs


def test_external_events_extracted_in_order(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    b1, b2 = make_bar(symbol="AAA"), make_bar(symbol="BBB")
    _publish_all(p, [b1, b2], [ToyAgent()])
    ext = list(external_envelopes(p))
    assert [e.event for e in ext] == [b1, b2]
    assert [e.seq for e in ext] == [0, 2]
    assert all(e.source == EXTERNAL_SOURCE and e.cause_seq is None for e in ext)
