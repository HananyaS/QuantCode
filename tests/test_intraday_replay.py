"""Replay driver: ordering, monotonicity, and byte-identical determinism."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from research.intraday.contracts.events import Bar, ClockTick, Event, Signal
from research.intraday.contracts.fixtures import T0, make_bar
from research.intraday.engine.replay import clock_ticks, merge_sources, run_replay
from research.intraday.ops.recorder import read_log

_MIN = timedelta(minutes=1)


class EchoAgent:
    name = "echo"
    subscribes = (Bar,)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        assert isinstance(event, Bar)
        return (Signal(ts=event.ts, symbol=event.symbol, score=event.close,
                       horizon_min=5, model_hash="echo-v1"),)


class BoomAgent:
    name = "boom"
    subscribes = (Bar,)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        raise RuntimeError("boom")


def _bar(symbol: str, i: int, start: datetime, close: float, spread: float = 0.01) -> Bar:
    bar_start = start + i * _MIN
    open_ = close * 0.999
    return make_bar(symbol=symbol, bar_start=bar_start, ts=bar_start + _MIN,
                    open=open_, high=max(open_, close) + spread,
                    low=min(open_, close) - spread, close=close, vwap=close)


def synthetic_bars(symbol: str, n: int, seed: int = 0,
                   start: datetime = T0) -> list[Bar]:
    rng = np.random.default_rng(seed)
    closes = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return [_bar(symbol, i, start, float(c)) for i, c in enumerate(closes)]


def _sources() -> list:
    return [clock_ticks(T0, T0 + 30 * _MIN), synthetic_bars("AAA", 30, 1),
            synthetic_bars("BBB", 30, 2)]


def test_clock_ticks_bounds_and_utc() -> None:
    ticks = list(clock_ticks(T0, T0 + 3 * _MIN))
    assert [t.ts for t in ticks] == [T0, T0 + _MIN, T0 + 2 * _MIN]
    assert all(isinstance(t, ClockTick) for t in ticks)
    assert list(clock_ticks(T0, T0)) == []
    with pytest.raises(ValueError):
        list(clock_ticks(T0.replace(tzinfo=None), T0 + _MIN))
    with pytest.raises(ValueError):
        list(clock_ticks(T0, (T0 + _MIN).replace(tzinfo=None)))
    with pytest.raises(ValueError):
        list(clock_ticks(T0, T0 + _MIN, step=timedelta(0)))
    with pytest.raises(ValueError):
        list(clock_ticks(T0 + _MIN, T0))


def test_merge_orders_by_ts_then_source_index() -> None:
    ticks = list(clock_ticks(T0 + _MIN, T0 + 3 * _MIN))
    bars = synthetic_bars("AAA", 2, 1)  # ts = T0+1, T0+2
    merged = list(merge_sources(ticks, bars))
    assert [type(e).__name__ for e in merged] == ["ClockTick", "Bar", "ClockTick", "Bar"]
    assert [e.ts for e in merged] == sorted(e.ts for e in merged)


def test_merge_rejects_non_monotone_source() -> None:
    bars = synthetic_bars("AAA", 3, 1)
    with pytest.raises(ValueError, match="source 1"):
        list(merge_sources(clock_ticks(T0, T0 + 5 * _MIN), [bars[0], bars[2], bars[1]]))


def test_same_day_replayed_twice_gives_identical_hash(tmp_path: Path) -> None:
    h1 = run_replay([EchoAgent()], _sources(), tmp_path / "a.jsonl")
    h2 = run_replay([EchoAgent()], _sources(), tmp_path / "b.jsonl")
    assert h1 == h2
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert len(list(read_log(tmp_path / "a.jsonl"))) == 30 + 60 + 60


def test_changed_input_changes_hash(tmp_path: Path) -> None:
    h1 = run_replay([EchoAgent()], _sources(), tmp_path / "a.jsonl")
    perturbed = synthetic_bars("AAA", 30, 1)
    b = perturbed[10]
    perturbed[10] = _bar("AAA", 10, T0, b.close + 1e-9)
    h2 = run_replay([EchoAgent()],
                    [clock_ticks(T0, T0 + 30 * _MIN), perturbed,
                     synthetic_bars("BBB", 30, 2)], tmp_path / "b.jsonl")
    assert h1 != h2


def test_replay_log_closed_on_agent_error(tmp_path: Path) -> None:
    path = tmp_path / "err.jsonl"
    with pytest.raises(RuntimeError, match="boom"):
        run_replay([BoomAgent()], [clock_ticks(T0, T0 + 5 * _MIN),
                                   synthetic_bars("AAA", 3, 1)], path)
    envs = list(read_log(path))
    assert envs and isinstance(envs[-1].event, Bar)
    path.unlink()  # on Windows this fails if the handle is still open
