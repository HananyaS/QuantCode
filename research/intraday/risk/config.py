"""RiskAgent configuration: validated at construction, one exit-grid cell per instance."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal, get_args

ExitMode = Literal["time_stop", "fixed_pct", "atr"]
EXIT_MODES: tuple[str, ...] = get_args(ExitMode)

SkipReason = Literal[
    "killed", "loss_limit", "halted", "after_last_entry", "below_threshold", "symbol_busy",
    "symbol_entries", "max_open", "no_bar", "zero_qty", "atr_warmup", "bad_bracket"]

_EXIT_PARAMS: dict[str, tuple[str, ...]] = {
    "time_stop": (),
    "fixed_pct": ("tp_pct", "sl_pct"),
    "atr": ("tp_atr", "sl_atr"),
}
_ALL_EXIT_PARAMS: tuple[str, ...] = ("tp_pct", "sl_pct", "tp_atr", "sl_atr")


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskConfig:
    start_equity_usd: float = 25_000.0
    position_notional_usd: float = 2_000.0
    max_open_positions: int = 5
    max_entries_per_symbol: int = 1
    entry_threshold: float = 0.5
    exit_mode: ExitMode = "time_stop"
    tp_pct: float | None = None
    sl_pct: float | None = None
    tp_atr: float | None = None
    sl_atr: float | None = None
    atr_period: int = 14
    hold_min: int | None = None
    quote_size_frac: float = 1.0
    quote_max_age_min: int = 5
    bar_participation_cap: float = 0.02
    daily_loss_limit_pct: float = 0.02
    last_entry_et: time = time(15, 30)
    flatten_et: time = time(15, 55)

    def __post_init__(self) -> None:
        self._check_times()
        self._check_scalars()
        self._check_exit_params()

    def _check_times(self) -> None:
        for name in ("last_entry_et", "flatten_et"):
            t = getattr(self, name)
            if not isinstance(t, time) or t.tzinfo is not None:
                raise ValueError(f"RiskConfig.{name} must be a naive datetime.time (ET wall clock), got {t!r}")
        if not self.last_entry_et < self.flatten_et:
            raise ValueError(
                f"RiskConfig.last_entry_et must be before flatten_et, "
                f"got last_entry_et={self.last_entry_et} flatten_et={self.flatten_et}")

    def _check_scalars(self) -> None:
        for name in ("start_equity_usd", "position_notional_usd"):
            if not getattr(self, name) > 0:
                raise ValueError(f"RiskConfig.{name} must be > 0, got {getattr(self, name)}")
        for name in ("max_open_positions", "max_entries_per_symbol", "atr_period", "quote_max_age_min"):
            if getattr(self, name) < 1:
                raise ValueError(f"RiskConfig.{name} must be >= 1, got {getattr(self, name)}")
        if self.hold_min is not None and self.hold_min < 1:
            raise ValueError(f"RiskConfig.hold_min must be None or >= 1, got {self.hold_min}")
        for name in ("quote_size_frac", "bar_participation_cap"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"RiskConfig.{name} must be in (0, 1], got {getattr(self, name)}")
        if not 0 < self.daily_loss_limit_pct < 1:
            raise ValueError(f"RiskConfig.daily_loss_limit_pct must be in (0, 1), got {self.daily_loss_limit_pct}")

    def _check_exit_params(self) -> None:
        if self.exit_mode not in EXIT_MODES:
            raise ValueError(f"RiskConfig.exit_mode must be one of {EXIT_MODES}, got {self.exit_mode!r}")
        wanted = _EXIT_PARAMS[self.exit_mode]
        for name in _ALL_EXIT_PARAMS:
            val = getattr(self, name)
            if name in wanted and val is None:
                raise ValueError(f"RiskConfig.{name} is required for exit_mode={self.exit_mode!r}")
            if name not in wanted and val is not None:
                raise ValueError(f"RiskConfig.{name} must be None for exit_mode={self.exit_mode!r}, got {val}")
        for name in wanted:
            if not getattr(self, name) > 0:
                raise ValueError(f"RiskConfig.{name} must be > 0, got {getattr(self, name)}")
        if self.exit_mode == "fixed_pct" and not self.sl_pct < 1:  # type: ignore[operator]
            raise ValueError(f"RiskConfig.sl_pct must be < 1 (a stop above zero), got {self.sl_pct}")
