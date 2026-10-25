from __future__ import annotations

import json

import pytest

from research.intraday.contracts.codec import canonical_json, from_record, to_record
from research.intraday.contracts.events import (
    SCHEMA_VERSION, Candidates, ClockTick, Halt, Kill,
)
from research.intraday.contracts.fixtures import (
    T0, make_bar, make_fill, make_order_intent, make_quote, make_signal, make_window,
)


@pytest.mark.parametrize("event", [
    make_bar(), make_quote(), make_window(3), make_signal(), make_order_intent(),
    make_fill(), ClockTick(ts=T0), Kill(ts=T0, reason="x"),
    Halt(ts=T0, symbol="A", halted=True, reason="LUDP"),
    Candidates(ts=T0, screen_id="s", symbols=("A", "B")),
    make_order_intent(limit_price=1.5, take_profit=2.0, stop_loss=1.0,
                      time_stop=T0.replace(hour=15)),
])
def test_codec_round_trip_is_exact(event):
    assert from_record(json.loads(canonical_json(to_record(event)))) == event


def test_codec_preserves_awkward_floats():
    bar = make_bar(open=0.1 + 0.2, high=1.0000000000000002, low=0.30000000000000004, close=0.5)
    assert from_record(json.loads(canonical_json(to_record(bar)))).open == 0.1 + 0.2


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_bytes_are_stable():
    w = make_window(3)
    assert canonical_json(to_record(w)) == canonical_json(to_record(w))
    rec = to_record(w)
    assert rec["type"] == "Window" and rec["v"] == SCHEMA_VERSION
    assert canonical_json(rec).count('"type":"Window"') == 1
    assert all(b["type"] == "Bar" and b["v"] == SCHEMA_VERSION for b in rec["bars"])


def test_datetimes_use_utc_suffix():
    assert to_record(make_bar())["ts"] == "2026-03-02T14:31:00+00:00"


def test_from_record_rejects_unknown_type_and_version():
    rec = to_record(ClockTick(ts=T0))
    with pytest.raises(ValueError, match="Bogus"):
        from_record({**rec, "type": "Bogus"})
    with pytest.raises(ValueError, match="99") as ei:
        from_record({**rec, "v": 99})
    assert str(SCHEMA_VERSION) in str(ei.value)


def test_from_record_rejects_extra_and_missing_keys():
    rec = to_record(Kill(ts=T0, reason="x"))
    with pytest.raises(ValueError, match="Kill"):
        from_record({**rec, "extra": 1})
    missing = {k: v for k, v in rec.items() if k != "reason"}
    with pytest.raises(ValueError, match="Kill"):
        from_record(missing)


@pytest.mark.parametrize("bad", [None, [], "Kill", 3, [("type", "Kill")]])
def test_from_record_rejects_non_mapping(bad):
    with pytest.raises(ValueError, match="Mapping"):
        from_record(bad)


def test_round_trip_new_fixtures():
    from research.intraday.contracts.fixtures import (
        make_candidates, make_clock_tick, make_gap, make_halt, make_kill,
        make_order_update,
    )
    for event in (make_candidates(), make_clock_tick(), make_gap(), make_halt(),
                  make_kill(), make_order_update()):
        assert from_record(json.loads(canonical_json(to_record(event)))) == event


def test_decoded_datetimes_are_utc_singleton():
    from datetime import timezone
    out = from_record(json.loads(canonical_json(to_record(make_bar()))))
    assert out.ts.tzinfo is timezone.utc and out.bar_start.tzinfo is timezone.utc


def test_from_record_rejects_int_for_float_field():
    rec = to_record(make_fill(cost_bps=0.0))
    rec["cost_bps"] = 0
    with pytest.raises(ValueError, match="Fill.cost_bps"):
        from_record(rec)
