"""Replay driver: clock ticks, source merging, and a one-call deterministic run.

Ordering convention: pass ``clock_ticks(...)`` as source 0. At equal ``ts`` a
lower source index publishes first, so time-driven logic (time stops, gap
checks) sees the tick before that minute's bars. See ADR-001.
"""
from __future__ import annotations

import heapq
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from research.intraday.contracts.events import ClockTick, Event
from research.intraday.contracts.protocols import Agent
from research.intraday.engine.bus import EventBus
from research.intraday.ops.recorder import JsonlRecorder, log_hash


def _is_utc_aware(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.utcoffset() == timedelta(0)


def clock_ticks(start: datetime, end: datetime,
                step: timedelta = timedelta(minutes=1)) -> Iterator[ClockTick]:
    """Ticks in [start, end). Both bounds must be UTC-aware; step must be > 0."""
    if not (_is_utc_aware(start) and _is_utc_aware(end)):
        raise ValueError("start and end must be UTC-aware")
    if step <= timedelta(0):
        raise ValueError(f"step must be positive, got {step}")
    if end < start:
        raise ValueError(f"end {end} < start {start}")
    ts = start
    while ts < end:
        yield ClockTick(ts=ts)
        ts += step


def _keyed(i: int, source: Iterable[Event]) -> Iterator[tuple[datetime, int, int, Event]]:
    last: datetime | None = None
    for k, event in enumerate(source):
        if last is not None and event.ts < last:
            raise ValueError(
                f"source {i} is not ts-monotone at position {k}: {event.ts} < {last}")
        last = event.ts
        yield (event.ts, i, k, event)


def merge_sources(*sources: Iterable[Event]) -> Iterator[Event]:
    """Lazy merge by (ts, source_index, position_in_source); events never compared."""
    keyed = [_keyed(i, s) for i, s in enumerate(sources)]
    for _, _, _, event in heapq.merge(*keyed, key=lambda t: t[:3]):
        yield event


def run_replay(agents: Sequence[Agent], sources: Sequence[Iterable[Event]],
               log_path: Path) -> str:
    """Replay sources through the agents, record the log, return its sha256."""
    with JsonlRecorder(log_path) as recorder:
        bus = EventBus(agents, sink=recorder)
        for event in merge_sources(*sources):
            bus.publish(event)
    return log_hash(log_path)
