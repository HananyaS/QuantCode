"""ET <-> UTC helpers. ``et_to_utc`` is the one datetime constructor of the data slice."""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

ET: ZoneInfo = ZoneInfo("America/New_York")


def et_to_utc(d: date, hhmm: str) -> datetime:
    """ET wall-clock ``hhmm`` on ``d`` as a datetime with ``tzinfo is timezone.utc``."""
    out = datetime.combine(d, time.fromisoformat(hhmm), tzinfo=ET).astimezone(timezone.utc)
    assert out.tzinfo is timezone.utc
    return out


def et_date(ts: datetime) -> date:
    """The America/New_York calendar date of an aware timestamp."""
    return ts.astimezone(ET).date()


def et_hhmm(ts: datetime) -> str:
    """The America/New_York wall-clock minute of an aware timestamp, ``"HH:MM"``."""
    return ts.astimezone(ET).strftime("%H:%M")


def minute_index_to_utc(d: date, minutes: Iterable[str]) -> dict[str, datetime]:
    """One ``et_to_utc`` per distinct ``"HH:MM"``, in first-seen order (for vectorised mapping)."""
    out: dict[str, datetime] = {}
    for m in minutes:
        if m not in out:
            out[m] = et_to_utc(d, m)
    return out
