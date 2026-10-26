"""ExecutionAgent configuration and the ET clock helpers (execution.md, Configuration)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

ET: ZoneInfo = ZoneInfo("America/New_York")
SUMMARY_READY_ET: time = time(9, 35)
"""The 09:34 quote summary covers quotes through 09:34 ET: causal from 09:35 ET."""
HOLDOUT_START: date = date(2025, 8, 7)


def to_et(ts: datetime) -> datetime:
    """UTC-aware datetime -> the same instant in America/New_York."""
    if not (type(ts) is datetime and ts.tzinfo is timezone.utc):
        raise ValueError(f"to_et needs a datetime with tzinfo=timezone.utc, got {ts!r}")
    return ts.astimezone(ET)


_FLOAT_FIELDS = ("quote_size_frac", "bar_participation_cap", "impact_coefficient",
                 "levels_penalty", "default_half_spread_bps")
_INT_FIELDS = ("quote_max_age_min", "entry_max_wait_min")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecConfig:
    quote_size_frac: float = 1.0
    quote_max_age_min: int = 5
    bar_participation_cap: float = 0.02
    entry_max_wait_min: int = 2
    eod_et: time = field(default=time(16, 1))
    impact_coefficient: float = 1.0
    levels_penalty: float = 1.0
    default_half_spread_bps: float = 100.0

    def __post_init__(self) -> None:
        for name in _FLOAT_FIELDS:
            v = getattr(self, name)
            if not isinstance(v, float) or not math.isfinite(v):
                raise ValueError(f"ExecConfig.{name} must be a finite float, got {v!r}")
        for name in _INT_FIELDS:
            v = getattr(self, name)
            if type(v) is not int:
                raise ValueError(f"ExecConfig.{name} must be an int, got {v!r}")
        for name in ("quote_size_frac", "bar_participation_cap"):
            v = getattr(self, name)
            if not 0.0 < v <= 1.0:
                raise ValueError(f"ExecConfig.{name} must be in (0, 1], got {v}")
        if self.quote_max_age_min < 0:
            raise ValueError(f"ExecConfig.quote_max_age_min must be >= 0, got {self.quote_max_age_min}")
        if self.entry_max_wait_min < 1:
            raise ValueError(f"ExecConfig.entry_max_wait_min must be >= 1, got {self.entry_max_wait_min}")
        for name in ("impact_coefficient", "levels_penalty"):
            if getattr(self, name) < 0:
                raise ValueError(f"ExecConfig.{name} must be >= 0, got {getattr(self, name)}")
        if self.default_half_spread_bps <= 0:
            raise ValueError(
                f"ExecConfig.default_half_spread_bps must be > 0, got {self.default_half_spread_bps}")
        if type(self.eod_et) is not time or self.eod_et.tzinfo is not None:
            raise ValueError(f"ExecConfig.eod_et must be a naive datetime.time, got {self.eod_et!r}")
