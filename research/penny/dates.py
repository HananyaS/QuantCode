"""dates.py -- the penny track's period boundaries, in one place (STRATEGY_SPEC §9.5).

Ten scripts used to carry their own copies of these two constants. A copy that
drifts by one day moves rows across the dev/holdout line without anyone seeing
it, so every boundary now comes from here.
"""
from __future__ import annotations

import pandas as pd

DEV_START = pd.Timestamp("2020-01-01")    # first signal date of the development period
SEAL = pd.Timestamp("2024-01-01")         # first signal date of the sealed holdout


def assert_dev(dates: pd.Series | pd.DatetimeIndex) -> None:
    """Raise if any date reaches the sealed period."""
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)))
    if len(d) and d.max() >= SEAL:
        raise ValueError(f"dev data reaches the sealed period: max {d.max().date()} >= {SEAL.date()}")
