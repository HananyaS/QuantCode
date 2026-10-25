"""Agent protocol and bus envelope: the only coupling between agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from research.intraday.contracts.events import Event


class Agent(Protocol):
    """Pure object: sees events, returns events. Never sees peers or the bus."""

    name: str
    subscribes: tuple[type[Event], ...]

    def on_event(self, event: Event) -> tuple[Event, ...]: ...


@dataclass(frozen=True, slots=True)
class Envelope:
    seq: int
    cause_seq: int | None
    source: str  # emitting agent's name, or "external"
    event: Event
