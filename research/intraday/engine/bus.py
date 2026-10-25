"""Deterministic event bus.

Two rules are the no-look-ahead guarantee: an emitted event carries its
cause's exact ts, and external publishes are ts-monotone.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Callable, Sequence

from research.intraday.contracts.events import Event
from research.intraday.contracts.protocols import Agent, Envelope

EXTERNAL = "external"


class EventBus:
    def __init__(
        self,
        agents: Sequence[Agent],
        sink: Callable[[Envelope], None] | None = None,
        max_cascade: int = 10_000,
    ) -> None:
        names = [a.name for a in agents]
        if any(not n for n in names):
            raise ValueError("agent name must be non-empty")
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise ValueError(f"duplicate agent names: {dups}")
        self._agents: tuple[Agent, ...] = tuple(agents)
        self._sink = sink
        self._max_cascade = max_cascade
        self._seq = 0
        self._last_external_ts: datetime | None = None

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def publish(self, event: Event) -> list[Envelope]:
        last = self._last_external_ts
        if last is not None and event.ts < last:
            raise ValueError(
                f"non-monotone source: event ts {event.ts} < last external ts {last}")
        self._last_external_ts = event.ts
        queue: deque[Envelope] = deque([Envelope(self._next_seq(), None, EXTERNAL, event)])
        out: list[Envelope] = []
        while queue:
            env = queue.popleft()
            if self._sink is not None:
                self._sink(env)
            out.append(env)
            if len(out) > self._max_cascade:
                raise RuntimeError(f"cascade exceeded max_cascade={self._max_cascade}")
            for agent in self._agents:
                if env.source == agent.name or not isinstance(env.event, agent.subscribes):
                    continue
                for emitted in agent.on_event(env.event):
                    if emitted.ts != env.event.ts:
                        raise ValueError(
                            f"{agent.name} emitted {type(emitted).__name__} with ts "
                            f"{emitted.ts} != cause ts {env.event.ts}")
                    queue.append(Envelope(self._next_seq(), env.seq, agent.name, emitted))
        return out
