"""MonitorAgent: halt release from an injected halt source, and the kill switch.

Pure ``on_event`` object; time comes only from ``ClockTick``. See
``research/intraday/docs/design/monitor.md``.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from research.intraday.contracts.events import (
    Bar, Candidates, ClockTick, Event, Fill, Gap, Halt, Kill, OrderIntent, OrderUpdate,
)
from research.intraday.ops.halts import HaltRow, HaltSource

_ET = ZoneInfo("America/New_York")
_MIN = timedelta(minutes=1)


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
        self._killed = False
        # Session (set by the first ClockTick or Candidates).
        self._session_date: date | None = None
        self._open_utc: datetime | None = None
        self._close_utc: datetime | None = None
        # Feed coverage.
        self._candidates: tuple[str, ...] | None = None
        self._baseline: datetime | None = None
        self._last_seen: dict[str, datetime] = {}

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        if self._killed:
            return ()
        out = self._dispatch(event)
        if any(isinstance(e, Kill) for e in out):
            self._killed = True
        return out

    def _dispatch(self, event: Event) -> tuple[Event, ...]:
        if isinstance(event, ClockTick):
            self._enter_session(event.ts)
            halts = self._release_halts(event.ts)
            kill = self._stale_check(event.ts)
            return halts + kill
        if isinstance(event, Candidates):
            return self._on_candidates(event)
        if isinstance(event, (Bar, Gap)):
            self._last_seen[event.symbol] = event.ts
        return ()

    # --- session and feed coverage ---

    def _enter_session(self, ts: datetime) -> None:
        day = ts.astimezone(_ET).date()
        if self._session_date is None:
            self._session_date = day
            self._open_utc = datetime.combine(day, self._session_open_et, _ET).astimezone(timezone.utc)
            self._close_utc = datetime.combine(day, self._session_close_et, _ET).astimezone(timezone.utc)
        elif day != self._session_date:
            raise ValueError(
                f"one run is one session: {ts.isoformat()} is ET date {day}, "
                f"session is {self._session_date}")

    def _on_candidates(self, event: Candidates) -> tuple[Event, ...]:
        if self._candidates is not None:
            raise ValueError(f"second Candidates at {event.ts.isoformat()}: exactly one per session")
        self._enter_session(event.ts)
        assert self._open_utc is not None
        self._candidates = event.symbols
        self._baseline = max(self._open_utc, event.ts)
        return ()

    def _stale_check(self, now: datetime) -> tuple[Kill, ...]:
        if not self._candidates or self._open_utc is None or self._close_utc is None:
            return ()
        if not self._open_utc + _MIN <= now <= self._close_utc:
            return ()
        assert self._baseline is not None
        ref = now - _MIN  # the minute ending at `now` arrives after the tick (ADR-001)
        limit = timedelta(minutes=self._stale_after_min)
        stale = [(s, self._last_seen.get(s, self._baseline)) for s in self._candidates]
        stale = [(s, last) for s, last in stale if ref - last >= limit]
        if not stale:
            return ()
        evidence = ", ".join(f"{s} last={last.isoformat()}" for s, last in stale)
        return (Kill(ts=now, reason=(f"stale_feed: {evidence} at {now.isoformat()} "
                                     f"(>= {self._stale_after_min} min)")),)

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
