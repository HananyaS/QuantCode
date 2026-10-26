"""Symbol-day reject records of the replay ETL, counted per arm (feed.md failure modes)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

RejectKind = Literal["coverage", "unadjust", "no_daily", "no_volume_factor"]


@dataclass(frozen=True)
class Reject:
    kind: RejectKind
    symbol: str
    session_date: date
    reason: str
    screen_id: str

    @property
    def arm(self) -> str:
        return self.screen_id.split(":")[0]


def counts_by_arm(rejects: Iterable[Reject], kind: RejectKind) -> dict[str, int]:
    """Number of rejects of ``kind`` per arm, keys sorted."""
    counts: dict[str, int] = {}
    for r in rejects:
        if r.kind == kind:
            counts[r.arm] = counts.get(r.arm, 0) + 1
    return {k: counts[k] for k in sorted(counts)}


def write_rejects_jsonl(path: Path, rejects: Iterable[Reject]) -> None:
    """One canonical JSON object per line, sorted by (session_date, kind, symbol)."""
    ordered = sorted(rejects, key=lambda r: (r.session_date, r.kind, r.symbol, r.screen_id))
    lines = [json.dumps({"kind": r.kind, "symbol": r.symbol,
                         "session_date": r.session_date.isoformat(), "reason": r.reason,
                         "screen_id": r.screen_id}, sort_keys=True, separators=(",", ":"))
             for r in ordered]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
