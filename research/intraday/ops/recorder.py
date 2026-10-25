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


def _parse_line(line: str, lineno: int) -> Envelope:
    try:
        rec: Any = json.loads(line)
        return Envelope(rec["seq"], rec["cause_seq"], rec["source"],
                        from_record(rec["event"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed log line {lineno}: {exc}") from exc


def read_log(path: Path) -> Iterator[Envelope]:
    with open(path, "r", encoding="utf-8", newline="\n") as fh:
        for lineno, raw in enumerate(fh, start=1):
            yield _parse_line(raw.rstrip("\n"), lineno)


def log_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
