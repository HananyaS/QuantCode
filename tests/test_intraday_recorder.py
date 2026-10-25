from __future__ import annotations

from pathlib import Path

import pytest

from research.intraday.contracts import Bar, Event
from research.intraday.contracts.fixtures import make_bar, make_signal
from research.intraday.engine.bus import EventBus
from research.intraday.ops.recorder import JsonlRecorder, log_hash, read_log


class ToyAgent:
    name = "toy"
    subscribes: tuple[type[Event], ...] = (Bar,)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        return (make_signal(ts=event.ts),)


def _record(path: Path) -> list:
    with JsonlRecorder(path) as rec:
        bus = EventBus([ToyAgent()], sink=rec)
        return bus.publish(make_bar())


def test_recorder_round_trips_envelopes(tmp_path: Path) -> None:
    p = tmp_path / "day.jsonl"
    envelopes = _record(p)
    assert len(envelopes) == 2
    assert list(read_log(p)) == envelopes


def test_recorder_refuses_to_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "day.jsonl"
    p.write_text("")
    with pytest.raises(FileExistsError):
        JsonlRecorder(p)


def test_log_is_lf_utf8_without_bom(tmp_path: Path) -> None:
    p = tmp_path / "day.jsonl"
    _record(p)
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")


def test_log_hash_changes_with_content(tmp_path: Path) -> None:
    a, b, c = (tmp_path / n for n in ("a.jsonl", "b.jsonl", "c.jsonl"))
    _record(a)
    _record(b)
    with JsonlRecorder(c) as rec:
        EventBus([ToyAgent()], sink=rec).publish(make_bar(close=1.03))
    assert log_hash(a) == log_hash(b)
    assert log_hash(a) != log_hash(c)
    assert len(log_hash(a)) == 64


def test_write_after_close_raises(tmp_path: Path) -> None:
    rec = JsonlRecorder(tmp_path / "day.jsonl")
    rec.close()
    rec.close()  # idempotent
    env = EventBus([]).publish(make_bar())[0]
    with pytest.raises(ValueError):
        rec(env)


def test_read_log_rejects_malformed_line(tmp_path: Path) -> None:
    p = tmp_path / "day.jsonl"
    _record(p)
    good = p.read_text(encoding="utf-8").splitlines()
    for bad in ("", "{not json"):
        q = tmp_path / f"bad{len(bad)}.jsonl"
        q.write_text("\n".join([good[0], bad, good[1]]) + "\n", encoding="utf-8", newline="\n")
        with pytest.raises(ValueError, match="line 2"):
            list(read_log(q))
