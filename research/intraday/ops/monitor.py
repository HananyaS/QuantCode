"""MonitorAgent: halt release from an injected halt source, and the kill switch.

Pure ``on_event`` object; time comes only from ``ClockTick``. See
``research/intraday/docs/design/monitor.md``.
"""
from __future__ import annotations

from datetime import datetime, time

from research.intraday.contracts.events import (
    Bar, Candidates, ClockTick, Event, Fill, Gap, Halt, Kill, OrderIntent, OrderUpdate,
)
from research.intraday.ops.halts import HaltRow, HaltSource


class MonitorAgent:
    name = "monitor"
    subscribes: tuple[type[Event], ...] = (
        ClockTick, Candidates, Bar, Gap, OrderIntent, OrderUpdate, Fill, Kill)

    def __init__(self, halt_source: HaltSource, *, stale_after_min: int = 2,
                 session_open_et: time = time(9, 30), session_close_et: time = time(16, 0),
                 infer_halts_from_gaps: bool = False, gap_halt_min: int = 5) -> None:
        for label, val in (("stale_after_min", stale_after_min), ("gap_halt_min", gap_halt_min)):
            if type(val) is not int or val <= 0:
                raise ValueError(f"{label} must be a positive int, got {val!r}")
        if not session_open_et < session_close_et:
            raise ValueError(
                f"session_open_et {session_open_et} must be < session_close_et {session_close_et}")
        self._halt_source = halt_source
        self._stale_after_min = stale_after_min
        self._session_open_et = session_open_et
        self._session_close_et = session_close_et
        self._infer_halts = infer_halts_from_gaps
        self._gap_halt_min = gap_halt_min
        self._halted: dict[str, bool] = {}
        self._last_row_ts: datetime | None = None
        self._counts: dict[str, int] = {"skipped_resume": 0, "skipped_duplicate_halt": 0}

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        if isinstance(event, ClockTick):
            return self._release_halts(event.ts)
        return ()

    # --- halts ---

    def _release_halts(self, now: datetime) -> tuple[Halt, ...]:
        out: list[Halt] = []
        for row in self._halt_source.due(now):
            self._check_row_order(row, now)
            self._last_row_ts = row.ts
            halted = self._halted.get(row.symbol, False)
            if not row.halted and not halted:
                self._counts["skipped_resume"] += 1
                continue
            if row.halted and halted:
                self._counts["skipped_duplicate_halt"] += 1
                continue
            self._halted[row.symbol] = row.halted
            out.append(Halt(ts=now, symbol=row.symbol, halted=row.halted, reason=row.reason))
        return tuple(out)

    def _check_row_order(self, row: HaltRow, now: datetime) -> None:
        last = self._last_row_ts
        if (last is not None and row.ts < last) or row.ts > now:
            raise ValueError(
                f"halt log out of time order: row {row.symbol} ts={row.ts.isoformat()} "
                f"(last released {last.isoformat() if last else None}, tick {now.isoformat()})")
