"""Regular-session close per date: 16:00 ET, or 13:00 ET on NYSE early-close (half) days.

The calendar is explicit (announced ahead by NYSE), never inferred from the
session's own bars, which keep extended-hours prints after a 13:00 close.
Audit 2026-10-26 on the minute cache (development dates < 2025-08-07): the
regular-session volume share at or after 13:00 is 0.35 median and >= 0.12 on
every day except exactly the 19 development dates below (0.007-0.108).
"""
from __future__ import annotations

from datetime import date

NYSE_EARLY_CLOSES: frozenset[date] = frozenset({
    date(2016, 11, 25),
    date(2017, 7, 3), date(2017, 11, 24),
    date(2018, 7, 3), date(2018, 11, 23), date(2018, 12, 24),
    date(2019, 7, 3), date(2019, 11, 29), date(2019, 12, 24),
    date(2020, 11, 27), date(2020, 12, 24),
    date(2021, 11, 26),
    date(2022, 11, 25),
    date(2023, 7, 3), date(2023, 11, 24),
    date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
    date(2025, 7, 3),
    # holdout / live dates: NYSE published calendar, not audited against the cache
    date(2025, 11, 28), date(2025, 12, 24),
    date(2026, 11, 27), date(2026, 12, 24),
})

REGULAR_CLOSE_ET: str = "16:00"
EARLY_CLOSE_ET: str = "13:00"


def session_close_et(d: date, half_days: frozenset[date] = NYSE_EARLY_CLOSES,
                     early_close_et: str = EARLY_CLOSE_ET,
                     regular_close_et: str = REGULAR_CLOSE_ET) -> str:
    """The ET wall-clock close of the regular session on ``d``."""
    return early_close_et if d in half_days else regular_close_et
