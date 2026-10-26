"""Recorder (log format 2) and run-manifest tests for intraday-ops, Phase 2."""
from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import pytest

from research.intraday.contracts import (
    EVENT_TYPES, SCHEMA_VERSION, Agent, Bar, Envelope, Event, Window,
)
from research.intraday.contracts.codec import canonical_json
from research.intraday.contracts.fixtures import (
    T0, make_bar, make_candidates, make_clock_tick, make_fill, make_gap, make_halt,
    make_kill, make_order_intent, make_order_update, make_quote, make_signal,
    make_window,
)
from research.intraday.engine.bus import EXTERNAL, EventBus
from research.intraday.engine.replay import clock_ticks, run_replay
from research.intraday.ops.manifest import (
    config_hash, git_state, session_log_path, write_manifest,
)
from research.intraday.ops.recorder import (
    EXTERNAL_SOURCE, LOG_FORMAT, JsonlRecorder, external_envelopes, log_hash, read_log,
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


# --- Task 2: compact Window encoding (log format 2) ---

class WindowAgent:  # one symbol; emits the last k bars as a Window on each Bar
    name = "win"
    subscribes: tuple[type[Event], ...] = (Bar,)

    def __init__(self, k: int = 30) -> None:
        self._bars: deque[Bar] = deque(maxlen=k)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        assert isinstance(event, Bar)
        self._bars.append(event)
        return (Window(ts=event.ts, symbol=event.symbol, bars=tuple(self._bars)),)


def _lines(p: Path) -> list[dict]:
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]


def test_window_line_references_bar_seqs(tmp_path: Path) -> None:
    w = make_window(3)
    p = tmp_path / "w.jsonl"
    envs = _publish_all(p, [*w.bars, w])
    ev = _lines(p)[3]["event"]
    assert ev == {"bar_seqs": [0, 1, 2], "symbol": "TEST", "ts": T0.isoformat(),
                  "type": "Window", "v": SCHEMA_VERSION}
    assert list(read_log(p)) == envs


def test_window_with_unlogged_or_differing_bar_written_inline(tmp_path: Path) -> None:
    w = make_window(3)
    p = tmp_path / "missing.jsonl"
    envs = _publish_all(p, [w.bars[0], w.bars[1], w])        # bars[2] never logged
    assert "bars" in _lines(p)[2]["event"] and "bar_seqs" not in _lines(p)[2]["event"]
    assert list(read_log(p)) == envs
    q = tmp_path / "differs.jsonl"
    other = replace(w.bars[2], close=1.03)                    # same key, different value
    envs = _publish_all(q, [w.bars[0], w.bars[1], other, w])
    assert "bars" in _lines(q)[3]["event"]
    assert list(read_log(q)) == envs


def test_compact_ref_to_non_bar_or_later_line_rejected(tmp_path: Path) -> None:
    w = make_window(3)
    p = tmp_path / "w.jsonl"
    _publish_all(p, [*w.bars, make_signal(), w])              # Signal is seq 3, Window seq 4
    raw = p.read_text(encoding="utf-8").splitlines()
    for bad in ([0, 1, 3], [0, 1, 5]):                       # a Signal line; a later line
        rec = json.loads(raw[4])
        rec["event"]["bar_seqs"] = bad
        q = tmp_path / f"bad{bad[-1]}.jsonl"
        _rewrite(q, [*raw[:4], canonical_json(rec)])
        with pytest.raises(ValueError, match="line 5"):
            list(read_log(q))


def test_compact_window_lines_are_small(tmp_path: Path) -> None:
    bars = [make_bar(ts=T0 + i * MIN, bar_start=T0 + (i - 1) * MIN) for i in range(60)]
    p = tmp_path / "k30.jsonl"
    envs = _publish_all(p, bars, [WindowAgent(30)])
    win_lines = [s for s in p.read_text(encoding="utf-8").splitlines() if '"Window"' in s]
    assert len(win_lines) == 60
    assert max(len(s) for s in win_lines) < 400               # inline k=30 is ~7 KB
    assert list(read_log(p)) == envs


def test_replay_with_windows_identical_hash(tmp_path: Path) -> None:
    def src() -> list[Bar]:
        return [make_bar(ts=T0 + i * MIN, bar_start=T0 + (i - 1) * MIN) for i in range(40)]
    h1 = run_replay([WindowAgent(30)], [clock_ticks(T0, T0 + 40 * MIN), src()], tmp_path / "a.jsonl")
    h2 = run_replay([WindowAgent(30)], [clock_ticks(T0, T0 + 40 * MIN), src()], tmp_path / "b.jsonl")
    assert h1 == h2
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert sum(isinstance(e.event, Window) for e in read_log(tmp_path / "a.jsonl")) == 40


def test_log_format_is_two() -> None:
    assert LOG_FORMAT == 2


# --- Task 3: live flushing, log path layout, run manifest ---

def test_flush_makes_each_line_visible_before_close(tmp_path: Path) -> None:
    p = tmp_path / "live.jsonl"
    env = EventBus([]).publish(make_bar())[0]
    rec = JsonlRecorder(p, flush=True)
    rec(env)
    assert p.read_text(encoding="utf-8").count("\n") == 1
    rec.close()


def test_session_log_path_layout(tmp_path: Path) -> None:
    p = session_log_path(date(2024, 3, 4), "run-001", log_root=tmp_path)
    assert p == tmp_path / "2024-03-04" / "run-001.jsonl"
    assert p.parent.is_dir() and not p.exists()
    for bad in ("", "a/b", "a\\b", "..", "c:x"):
        with pytest.raises(ValueError):
            session_log_path(date(2024, 3, 4), bad, log_root=tmp_path)


def test_config_hash_is_key_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2.0}) == config_hash({"b": 2.0, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_manifest_records_sha_config_and_model_hash(tmp_path: Path) -> None:
    p = tmp_path / "run-001.jsonl"
    _publish_all(p, [make_bar()])
    cfg = {"stale_after_min": 2, "arm": "screen"}
    m = write_manifest(p, git_sha="a" * 40, git_dirty=False, config=cfg,
                       model_hash="abc123", unseal=False)
    assert m == tmp_path / "run-001.manifest.json"
    assert json.loads(m.read_text(encoding="utf-8")) == {
        "config_hash": config_hash(cfg), "git_dirty": False, "git_sha": "a" * 40,
        "log_format": 2, "log_hash": log_hash(p), "model_hash": "abc123",
        "run_id": "run-001", "unseal": False}
    with pytest.raises(FileExistsError):
        write_manifest(p, git_sha="a" * 40, git_dirty=False, config=cfg,
                       model_hash="abc123", unseal=False)
    with pytest.raises(ValueError):
        write_manifest(tmp_path / "x.jsonl", git_sha="nothex", git_dirty=False,
                       config=cfg, model_hash="abc123", unseal=False)


def test_git_state_reports_sha_and_dirty() -> None:
    sha, dirty = git_state(Path(__file__).resolve().parents[1])
    assert re.fullmatch(r"[0-9a-f]{40}", sha) and isinstance(dirty, bool)


# --- Fix round 1 ---

def test_git_state_rejects_non_sha_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess

    import research.intraday.ops.manifest as manifest_mod

    def fake_run(args: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="not-a-sha\n", stderr="")
    monkeypatch.setattr(manifest_mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="git sha"):
        git_state(tmp_path)
