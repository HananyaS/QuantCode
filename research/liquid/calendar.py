"""Windows and the NYSE session shape for the liquid intraday track.

Dev is 2016-01-04..2022-12-30. The confirmation window (2023-01-03..2025-08-06)
is read once, on the user's go. Everything from 2025-08-07 is a holdout that
has not been fetched.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

DEV_START = pd.Timestamp("2016-01-04")
DEV_END = pd.Timestamp("2022-12-30")
CONFIRM_START = pd.Timestamp("2023-01-03")
CONFIRM_END = pd.Timestamp("2025-08-06")
HOLDOUT_START = pd.Timestamp("2025-08-07")

FULL_BARS = 390          # 09:30..15:59
HALF_BARS = 210          # 09:30..12:59 on a 13:00 close

# NYSE 13:00 closes inside the data (2016-01-04..2025-08-06), from the NYSE
# holiday schedules; data.build_cache checks each against the volume profile.
EARLY_CLOSES = frozenset(pd.Timestamp(d) for d in (
    "2016-11-25",
    "2017-07-03", "2017-11-24",
    "2018-07-03", "2018-11-23", "2018-12-24",
    "2019-07-03", "2019-11-29", "2019-12-24",
    "2020-11-27", "2020-12-24",
    "2021-11-26",
    "2022-11-25",
    "2023-07-03", "2023-11-24",
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03",
))


def session_bars(day: pd.Timestamp) -> int:
    """Regular-session one-minute bars on `day`."""
    return HALF_BARS if pd.Timestamp(day).normalize() in EARLY_CLOSES else FULL_BARS


def minute_index(hhmm: str) -> int:
    """Index of the bar that STARTS at HH:MM (09:30 -> 0, 15:59 -> 389)."""
    h, m = (int(x) for x in hhmm.split(":"))
    i = (h - 9) * 60 + (m - 30)
    assert 0 <= i < FULL_BARS, hhmm
    return i


def assert_through(dates: Iterable[pd.Timestamp], end: pd.Timestamp, what: str = "data") -> None:
    """Raise if any date is after `end`, the window's last session."""
    d = pd.DatetimeIndex(list(dates))
    if len(d) and d.max() > end:
        raise ValueError(f"{what} reaches {d.max().date()}, past {end.date()}")
