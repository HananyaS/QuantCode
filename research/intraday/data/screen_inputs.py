"""ScreenInputs: the read-only adapter the ScreenerAgent asks for data before the screen."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from research.intraday.data.basis import price_factor
from research.intraday.data.cache import (
    minute_cache_membership, read_daily, read_premarket_day,
)
from research.intraday.data.holdout import HOLDOUT_START, check_holdout
from research.intraday.data.paths import DEFAULT_DATA_ROOT, DEFAULT_VOLUME_DAILY_ROOT

_ONE_MIN = timedelta(minutes=1)


class ScreenInputs(Protocol):
    def daily_before(self, session_date: date, lookback_days: int) -> pd.DataFrame: ...

    def session_factor(self, session_date: date) -> pd.Series: ...

    def premarket_before(self, session_date: date, cutoff: datetime) -> pd.DataFrame: ...

    def minute_cache_symbols(self, session_date: date) -> tuple[str, ...]: ...


class ReplayScreenInputs:
    """ScreenInputs over the caches; every method checks the holdout before reading."""

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT,
                 volume_root: Path = DEFAULT_VOLUME_DAILY_ROOT,
                 holdout_start: date = HOLDOUT_START, unseal: bool = False) -> None:
        self.data_root = Path(data_root)
        self.volume_root = Path(volume_root)
        self.holdout_start = holdout_start
        self.unseal = unseal

    def _check(self, session_date: date) -> None:
        check_holdout(session_date, self.holdout_start, self.unseal)

    def daily_before(self, session_date: date, lookback_days: int) -> pd.DataFrame:
        """Rows dated in ``[D - lookback_days, D)``."""
        self._check(session_date)
        start = session_date - timedelta(days=lookback_days)
        return read_daily(self.data_root, self.volume_root, start, session_date)

    def session_factor(self, session_date: date) -> pd.Series:
        """``f_D`` per symbol, name ``"f"``: the one session-date daily read, prices withheld."""
        self._check(session_date)
        day = read_daily(self.data_root, self.volume_root, session_date,
                         session_date + timedelta(days=1)).set_index("symbol")
        return price_factor(day["close_unadj"], day["close"])

    def premarket_before(self, session_date: date, cutoff: datetime) -> pd.DataFrame:
        """Pre-market bars that ended at or before ``cutoff``."""
        self._check(session_date)
        pm = read_premarket_day(self.data_root, session_date)
        keep = pm["bar_start"] + _ONE_MIN <= pd.Timestamp(cutoff)
        return pm.loc[keep].reset_index(drop=True)

    def minute_cache_symbols(self, session_date: date) -> tuple[str, ...]:
        self._check(session_date)
        return minute_cache_membership(self.data_root, session_date)
