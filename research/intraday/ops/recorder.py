"""Append-only JSONL event recorder: a bus sink, not an Agent.

An agent never sees ``seq``/``cause_seq``/``source``; a sink does. Bytes are
canonical (UTF-8, LF, no BOM) so ``log_hash`` is platform-independent.

Log format (``LOG_FORMAT``): 1 = Phase-1, inline Windows only; 2 = a Window may be
written as ``bar_seqs``. A Window is written compactly when, and only when, every
one of its bars equals the latest logged ``Bar`` line for its ``(symbol, bar_start)``;
its record then carries ``"bar_seqs"`` (the seqs of those lines) instead of
``"bars"``. Otherwise it is written inline. Every other line is byte-identical to
format 1, and ``read_log`` expands compact Windows, so ``read_log(write(x)) == x``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator

from research.intraday.contracts.codec import canonical_json, from_record, to_record
from research.intraday.contracts.events import Bar, Event, Window
from research.intraday.contracts.protocols import Envelope

LOG_FORMAT: int = 2

# Must equal engine.bus.EXTERNAL; ops may not import the engine (a test pins it).
EXTERNAL_SOURCE: str = "external"


class JsonlRecorder:
    def __init__(self, path: Path, *, flush: bool = False) -> None:
        # Mode "x": an existing log is never overwritten.
        self._fh = open(path, "x", encoding="utf-8", newline="\n")
        # Live runs flush every line so a crash loses at most the line in flight.
        self._flush = flush
        # (symbol, bar_start) -> (seq, Bar) of the latest logged Bar line.
        self._bars: dict[tuple[str, datetime], tuple[int, Bar]] = {}

    def _logged_seqs(self, window: Window) -> list[int] | None:
        """Seqs of the window's bars if every one equals its latest logged Bar, else None."""
        seqs: list[int] = []
        for bar in window.bars:
            hit = self._bars.get((bar.symbol, bar.bar_start))
            if hit is None or hit[1] != bar:
                return None
            seqs.append(hit[0])
        return seqs

    def _event_record(self, event: Event) -> dict[str, object]:
        record = to_record(event)
        if isinstance(event, Window):
            seqs = self._logged_seqs(event)
            if seqs is not None:
                del record["bars"]
                record["bar_seqs"] = seqs
        return record

    def __call__(self, env: Envelope) -> None:
        line = canonical_json({
            "seq": env.seq,
            "cause_seq": env.cause_seq,
            "source": env.source,
            "event": self._event_record(env.event),
        })
        self._fh.write(line + "\n")  # ValueError if closed
        if self._flush:
            self._fh.flush()
        if isinstance(env.event, Bar):
            self._bars[(env.event.symbol, env.event.bar_start)] = (env.seq, env.event)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlRecorder:
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()


def _check_order(env: Envelope, expected_seq: int) -> None:
    """A log line's seq is its 0-based line index; cause_seq names an earlier line."""
    if type(env.seq) is not int or env.seq != expected_seq:
        raise ValueError(f"seq {env.seq!r} != expected {expected_seq}")
    cause = env.cause_seq
    if cause is not None and (type(cause) is not int or not 0 <= cause < env.seq):
        raise ValueError(f"cause_seq {cause!r} does not name an earlier line (seq {env.seq})")


def _expand_window(event_rec: Any, bars_by_seq: dict[int, Bar]) -> Any:
    """A compact Window record -> its inline record; any other record unchanged."""
    if not isinstance(event_rec, dict) or "bar_seqs" not in event_rec:
        return event_rec
    if event_rec.get("type") != "Window" or "bars" in event_rec:
        raise ValueError("bar_seqs is valid only on a Window record without bars")
    seqs = event_rec["bar_seqs"]
    if not isinstance(seqs, list):
        raise ValueError(f"bar_seqs must be a list, got {type(seqs).__name__}")
    for s in seqs:
        if type(s) is not int or s not in bars_by_seq:
            raise ValueError(f"bar_seqs entry {s!r} is not the seq of an earlier Bar line")
    inline = {k: v for k, v in event_rec.items() if k != "bar_seqs"}
    inline["bars"] = [to_record(bars_by_seq[s]) for s in seqs]
    return inline


def _parse_line(line: str, lineno: int, bars_by_seq: dict[int, Bar]) -> Envelope:
    try:
        rec: Any = json.loads(line)
        env = Envelope(rec["seq"], rec["cause_seq"], rec["source"],
                       from_record(_expand_window(rec["event"], bars_by_seq)))
        _check_order(env, lineno - 1)
        return env
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed log line {lineno}: {exc}") from exc


def read_log(path: Path) -> Iterator[Envelope]:
    """Every envelope in log order, compact Windows expanded. ValueError on a malformed
    line, a seq gap, a cause_seq that does not name an earlier line, or a bar_seqs
    entry that is not an earlier Bar line."""
    bars_by_seq: dict[int, Bar] = {}
    with open(path, "r", encoding="utf-8", newline="\n") as fh:
        for lineno, raw in enumerate(fh, start=1):
            env = _parse_line(raw.rstrip("\n"), lineno, bars_by_seq)
            if isinstance(env.event, Bar):
                bars_by_seq[env.seq] = env.event
            yield env


def external_envelopes(path: Path) -> Iterator[Envelope]:
    """The externally published envelopes (the nightly-replay inputs), in log order."""
    return (env for env in read_log(path) if env.source == EXTERNAL_SOURCE)


def log_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
