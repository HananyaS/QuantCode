"""Halt sources for the MonitorAgent: rows of (ts, symbol, halted, reason).

Halt data is not an event on the way in. Phase 2 has one source, the recorded
halt log (empty for days before recording began); the live Nasdaq halt RSS
adapter and a halt-log file loader are Phase 5.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class HaltRow:
    ts: datetime
    symbol: str
    halted: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self.ts) is not datetime or self.ts.tzinfo is not timezone.utc:
            raise ValueError(f"HaltRow.ts must be a datetime with tzinfo=timezone.utc, got {self.ts!r}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError(f"HaltRow.symbol must be a non-empty str, got {self.symbol!r}")
        if type(self.halted) is not bool:
            raise ValueError(f"HaltRow.halted must be a bool, got {self.halted!r}")
        if not isinstance(self.reason, str):
            raise ValueError(f"HaltRow.reason must be a str, got {self.reason!r}")


class HaltSource(Protocol):
    def due(self, ts: datetime) -> tuple[HaltRow, ...]:
        """Every row not yet returned whose ts <= ``ts``, in source order."""
        ...


class RecordedHaltLog:
    """A fixed, recorded halt log. Not sorted here: the monitor checks the order."""

    def __init__(self, rows: Sequence[HaltRow]) -> None:
        self._rows: tuple[HaltRow, ...] = tuple(rows)
        assert all(isinstance(r, HaltRow) for r in self._rows), "rows must be HaltRow"
        self._cursor = 0

    def due(self, ts: datetime) -> tuple[HaltRow, ...]:
        start = self._cursor
        while self._cursor < len(self._rows) and self._rows[self._cursor].ts <= ts:
            self._cursor += 1
        return self._rows[start:self._cursor]
