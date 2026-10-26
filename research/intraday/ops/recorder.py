"""Append-only JSONL event recorder: a bus sink, not an Agent.

An agent never sees ``seq``/``cause_seq``/``source``; a sink does. Bytes are
canonical (UTF-8, LF, no BOM) so ``log_hash`` is platform-independent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator

from research.intraday.contracts.codec import canonical_json, from_record, to_record
from research.intraday.contracts.protocols import Envelope

# Must equal engine.bus.EXTERNAL; ops may not import the engine (a test pins it).
EXTERNAL_SOURCE: str = "external"


class JsonlRecorder:
    def __init__(self, path: Path) -> None:
        # Mode "x": an existing log is never overwritten.
        self._fh = open(path, "x", encoding="utf-8", newline="\n")

    def __call__(self, env: Envelope) -> None:
        line = canonical_json({
            "seq": env.seq,
            "cause_seq": env.cause_seq,
            "source": env.source,
            "event": to_record(env.event),
        })
        self._fh.write(line + "\n")  # ValueError if closed

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


def _parse_line(line: str, lineno: int) -> Envelope:
    try:
        rec: Any = json.loads(line)
        env = Envelope(rec["seq"], rec["cause_seq"], rec["source"],
                       from_record(rec["event"]))
        _check_order(env, lineno - 1)
        return env
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed log line {lineno}: {exc}") from exc


def read_log(path: Path) -> Iterator[Envelope]:
    """Every envelope in log order. ValueError on a malformed line, a seq gap, or a
    cause_seq that does not name an earlier line."""
    with open(path, "r", encoding="utf-8", newline="\n") as fh:
        for lineno, raw in enumerate(fh, start=1):
            yield _parse_line(raw.rstrip("\n"), lineno)


def external_envelopes(path: Path) -> Iterator[Envelope]:
    """The externally published envelopes (the nightly-replay inputs), in log order."""
    return (env for env in read_log(path) if env.source == EXTERNAL_SOURCE)


def log_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
