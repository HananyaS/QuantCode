"""ScreenerAgent: one causal ``Candidates`` per session at ``screen_time_et`` (screener.md).

Pure agent: all I/O goes through the injected ``ScreenInputs`` adapter. The
integrator builds ``ReplayScreenInputs(Path(cfg.data_root), Path(cfg.volume_daily_root),
cfg.holdout_start, cfg.unseal)`` from the same config.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from research.intraday.contracts.events import Candidates, ClockTick, Event
from research.intraday.data.clock import et_date, et_to_utc
from research.intraday.data.errors import LookAheadError
from research.intraday.data.holdout import HOLDOUT_START, check_holdout
from research.intraday.data.paths import DEFAULT_DATA_ROOT, DEFAULT_VOLUME_DAILY_ROOT
from research.intraday.data.ranking import RANKABLE_FEATURES, convert_premarket, rank_symbols
from research.intraday.data.screen_inputs import ScreenInputs
from research.intraday.data.universe import UniverseResult, causal_universe, intersect_cache

logger = logging.getLogger(__name__)

_ARMS = ("screen", "no_selection")
_ONE_MIN = timedelta(minutes=1)


@dataclass(frozen=True)
class ScreenerConfig:
    screen_time_et: str = "09:25"
    premarket_start_et: str = "04:00"
    arm: str = "screen"
    min_price: float = 1.0
    max_price: float = 20.0
    min_adv_usd: float = 1_000_000.0
    adv_window: int = 20
    rank_by: tuple[str, ...] = ("pm_gap", "pm_dollar_volume")
    top_n: int = 20
    exclude_split_days: bool = True
    split_log_step_threshold: float = math.log(1.25)
    unadjust_tolerance: float = 0.02
    holdout_start: date = HOLDOUT_START
    unseal: bool = False
    data_root: str = str(DEFAULT_DATA_ROOT)
    volume_daily_root: str = str(DEFAULT_VOLUME_DAILY_ROOT)
    restrict_to_minute_cache: bool = True
    daily_lookback_days: int = 45
    max_prior_gap_days: int = 5
    max_late_minutes: int = 5

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected one of {_ARMS}")
        if self.top_n <= 0:
            raise ValueError(f"top_n must be > 0, got {self.top_n}")
        if not self.rank_by:
            raise ValueError("rank_by must name at least one feature")
        unknown = [r for r in self.rank_by if r not in RANKABLE_FEATURES]
        if unknown:
            raise ValueError(f"rank_by names unknown features {unknown}")


def config_hash(config: ScreenerConfig) -> str:
    """sha256 of the canonical JSON of the configuration."""
    blob = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"),
                      default=date.isoformat)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScreenReport:
    session_date: date
    arm: str
    screen_id: str
    n_universe: int
    n_removed_by_cache: int
    n_split_excluded: int
    n_unconvertible: int
    n_no_premarket: int
    n_selected: int


class ScreenerAgent:
    subscribes: tuple[type[Event], ...] = (ClockTick,)

    def __init__(self, inputs: ScreenInputs, config: ScreenerConfig = ScreenerConfig(),
                 name: str = "screener") -> None:
        self.inputs = inputs
        self.config = config
        self.name = name
        self.session_date: date | None = None
        self.first_tick_seen = False
        self.fired = False
        self.report: ScreenReport | None = None

    def on_event(self, event: Event) -> tuple[Event, ...]:
        if not isinstance(event, ClockTick):
            return ()
        self._track_session(event.ts)
        assert self.session_date is not None
        if self.fired or event.ts < et_to_utc(self.session_date, self.config.screen_time_et):
            return ()
        return (self._fire(event.ts),)

    def _track_session(self, ts: datetime) -> None:
        d = et_date(ts)
        if not self.first_tick_seen:
            self.first_tick_seen = True
            self.session_date = d
            latest = (et_to_utc(d, self.config.screen_time_et)
                      + timedelta(minutes=self.config.max_late_minutes))
            if ts > latest:
                raise ValueError(
                    f"run started late: first tick {ts.isoformat()} is after "
                    f"{self.config.screen_time_et} ET + {self.config.max_late_minutes} min")
        elif d != self.session_date:
            raise ValueError(f"second session date {d} in a run for {self.session_date}")

    def _fire(self, ts: datetime) -> Candidates:
        cfg, d = self.config, self.session_date
        assert d is not None
        check_holdout(d, cfg.holdout_start, cfg.unseal)
        daily = self.inputs.daily_before(d, cfg.daily_lookback_days)
        if (daily["date"] >= pd.Timestamp(d)).any():
            raise LookAheadError(f"daily rows on or after the session date {d}")
        universe = causal_universe(
            daily, d, self.inputs.session_factor(d), min_price=cfg.min_price,
            max_price=cfg.max_price, min_adv_usd=cfg.min_adv_usd, adv_window=cfg.adv_window,
            exclude_split_days=cfg.exclude_split_days,
            split_log_step_threshold=cfg.split_log_step_threshold,
            max_prior_gap_days=cfg.max_prior_gap_days)
        symbols, removed = universe.symbols, 0
        if cfg.restrict_to_minute_cache:
            symbols, removed = intersect_cache(symbols, self.inputs.minute_cache_symbols(d))
        n_unconv = n_no_pm = 0
        if cfg.arm == "no_selection":
            selected = symbols
        else:
            selected, n_unconv, n_no_pm = self._rank(d, ts, symbols, universe)
        screen_id = f"{cfg.arm}:{config_hash(cfg)[:12]}:{d:%Y-%m-%d}"
        self.fired = True
        self.report = ScreenReport(
            session_date=d, arm=cfg.arm, screen_id=screen_id, n_universe=len(universe.symbols),
            n_removed_by_cache=removed, n_split_excluded=universe.n_split_excluded,
            n_unconvertible=n_unconv, n_no_premarket=n_no_pm, n_selected=len(selected))
        logger.info("screen fired: %s", self.report)
        return Candidates(ts=ts, screen_id=screen_id, symbols=tuple(selected))

    def _rank(self, d: date, ts: datetime, symbols: tuple[str, ...],
              universe: UniverseResult) -> tuple[tuple[str, ...], int, int]:
        cfg = self.config
        pm = self.inputs.premarket_before(d, ts)
        if (pm["bar_start"] + _ONE_MIN > pd.Timestamp(ts)).any():
            raise LookAheadError(f"pre-market bar ending after the screen tick {ts.isoformat()}")
        pm = pm[pm["minute_et"] >= cfg.premarket_start_et]
        prior = universe.prior.loc[list(symbols)]
        traded, dropped = convert_premarket(pm, prior, cfg.unadjust_tolerance)
        selected, n_no_pm = rank_symbols(symbols, traded, prior["close_unadj"], cfg.rank_by,
                                         cfg.top_n)
        return selected, len(dropped), n_no_pm
