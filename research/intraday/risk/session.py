"""ET session bounds: built per session date with zoneinfo, so DST is handled."""
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from research.intraday.risk.config import RiskConfig

ET: ZoneInfo = ZoneInfo("America/New_York")


def et_date(ts: datetime) -> date:
    """The ET calendar date of a UTC timestamp."""
    assert ts.tzinfo is not None, "et_date needs an aware datetime"
    return ts.astimezone(ET).date()


def session_bounds(session_date: date, cfg: RiskConfig) -> tuple[datetime, datetime]:
    """(last_entry_utc, flatten_utc) for the session date."""
    last = datetime.combine(session_date, cfg.last_entry_et, ET).astimezone(timezone.utc)
    flat = datetime.combine(session_date, cfg.flatten_et, ET).astimezone(timezone.utc)
    assert last < flat
    return last, flat
