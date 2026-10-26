"""FeedAgent: one ``Bar`` or ``Gap`` per candidate per session minute, released on the clock.

Pure agent over an injected ``BarSource`` (``ReplayFeed`` now, a stream buffer
later). On a tick at ``T`` it releases the minute that has just closed,
``bar_start = T - 1 min``, for each candidate in sorted order: the source's bar
or a ``Gap`` (never a filled bar), then the quote stamped ``T`` if one exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Mapping, Protocol

from research.intraday.contracts.events import Bar, Candidates, ClockTick, Event, Gap, Quote
from research.intraday.data.clock import et_date, et_to_utc
from research.intraday.data.rejects import Reject

_ONE_MIN = timedelta(minutes=1)


@dataclass(frozen=True)
class SessionData:
    bars: Mapping[tuple[str, datetime], Bar]      # key (symbol, bar_start)
    quotes: Mapping[tuple[str, datetime], Quote]  # key (symbol, ts)
    rejects: tuple[Reject, ...]
    n_quotes_skipped: int


class BarSource(Protocol):
    feed: str

    def open_session(self, session_date: date, symbols: tuple[str, ...],
                     screen_id: str) -> SessionData: ...


class FeedAgent:
    subscribes: tuple[type[Event], ...] = (Candidates, ClockTick)

    def __init__(self, source: BarSource, session_open_et: str = "09:30",
                 session_close_et: str = "16:00", name: str = "feed") -> None:
        self.source = source
        self.session_open_et = session_open_et
        self.session_close_et = session_close_et
        self.name = name
        self._data: SessionData | None = None
        self._symbols: tuple[str, ...] = ()
        self._open_utc: datetime | None = None
        self._close_utc: datetime | None = None

    @property
    def rejects(self) -> tuple[Reject, ...]:
        return () if self._data is None else self._data.rejects

    def on_event(self, event: Event) -> tuple[Event, ...]:
        if isinstance(event, Candidates):
            self._open(event)
            return ()
        if isinstance(event, ClockTick):
            return self._release(event.ts)
        return ()

    def _open(self, c: Candidates) -> None:
        if self._data is not None:
            raise ValueError(f"second Candidates in one session ({c.screen_id})")
        d = et_date(c.ts)
        self._data = self.source.open_session(d, c.symbols, c.screen_id)
        self._symbols = c.symbols
        self._open_utc = et_to_utc(d, self.session_open_et)
        self._close_utc = et_to_utc(d, self.session_close_et)

    def _release(self, t: datetime) -> tuple[Event, ...]:
        if self._data is None:
            return ()
        assert self._open_utc is not None and self._close_utc is not None
        bs = t - _ONE_MIN
        if not (self._open_utc <= bs < self._close_utc):
            return ()
        out: list[Event] = []
        for sym in self._symbols:  # the Candidates tuple is sorted and unique
            bar = self._data.bars.get((sym, bs))
            out.append(bar if bar is not None else Gap(ts=t, symbol=sym, missing_bar_start=bs))
            quote = self._data.quotes.get((sym, t))
            if quote is not None:
                out.append(quote)
        return tuple(out)
