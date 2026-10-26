"""Session runner: one recorded trading session through every service (Phase 2 integration).

Wiring (engine.md agent order, ADR-001): screener, feed, scorer (stands in for state +
predictor), risk, execution, monitor; ``clock_ticks`` over ``[09:25, 16:02)`` ET is the
only source. The recorder is ``run_replay``'s sink; the manifest is written after the
recorder has closed, and the feed's symbol-day rejects beside the log.

This module only wires public service APIs; it changes no agent behaviour.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from research.intraday.contracts.events import Bar, Event, Signal
from research.intraday.contracts.protocols import Agent
from research.intraday.data.cache import minute_cache_membership, read_daily
from research.intraday.data.clock import et_date, et_to_utc
from research.intraday.data.feed import FeedAgent
from research.intraday.data.holdout import HOLDOUT_START, check_holdout
from research.intraday.data.rejects import write_rejects_jsonl
from research.intraday.data.replay_feed import ReplayFeed
from research.intraday.data.replay_quotes import read_quote_summaries
from research.intraday.data.screen_inputs import ReplayScreenInputs
from research.intraday.data.screener import ScreenerAgent, ScreenerConfig, config_hash
from research.intraday.engine.replay import clock_ticks, run_replay
from research.intraday.exec import ExecConfig, ExecutionAgent, SimBroker, StaticCostInputs
from research.intraday.exec.cost_inputs import build_cost_inputs
from research.intraday.ops.halts import RecordedHaltLog
from research.intraday.ops.manifest import git_state, session_log_path, write_manifest
from research.intraday.ops.monitor import MonitorAgent
from research.intraday.risk import RiskAgent, RiskConfig

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
TICKS_START_ET: str = "09:25"
TICKS_END_ET: str = "16:02"  # exclusive: the 16:01 tick runs the end-of-session pass
COST_WINDOW: int = 20
COST_LOOKBACK_DAYS: int = 45  # calendar days: > COST_WINDOW + 1 sessions
_ONE_MIN = timedelta(minutes=1)


class StandInTestScorer:
    """TEST STAND-IN for StateAgent + PredictorAgent. Never used for research.

    Emits one ``Signal`` (score 1.0) per symbol, on the first ``Bar`` that starts at or
    after ``start_et`` with ``close > open``. A fixed rule, so the integration replay can
    exercise risk and execution before the real model exists.
    """

    model_hash = "test-stand-in:first-green-bar"

    def __init__(self, start_et: str = "09:35", horizon_min: int = 390,
                 name: str = "scorer") -> None:
        self.name = name
        self.subscribes: tuple[type[Event], ...] = (Bar,)
        self._start_et = start_et
        self._horizon_min = horizon_min
        self._start_utc: datetime | None = None
        self._fired: set[str] = set()

    def on_event(self, event: Event) -> tuple[Event, ...]:
        if not isinstance(event, Bar) or event.symbol in self._fired:
            return ()
        if self._start_utc is None:
            self._start_utc = et_to_utc(et_date(event.ts), self._start_et)
        if event.bar_start < self._start_utc or not event.close > event.open:
            return ()
        self._fired.add(event.symbol)
        return (Signal(ts=event.ts, symbol=event.symbol, score=1.0,
                       horizon_min=self._horizon_min, model_hash=self.model_hash),)


def exec_daily_frame(data_root: Path, volume_daily_root: Path, d: date) -> pd.DataFrame:
    """Prior daily rows in exec's schema, for the symbols the minute cache covers on ``d``
    (every possible candidate; a symbol outside it has no bars, so it never fills).

    Seam rename: the data package's breakout pair ``volume_v``/``volume_unadj_v`` becomes
    exec's ``volume``/``volume_unadj`` (one source, so ``s = volume / volume_unadj`` is the
    split-only factor); ``close`` stays the penny adjusted close (a return statistic).
    Rows without the volume source are dropped.

    exec's builder raises for the whole frame on one bad symbol, so two kinds of symbol
    are left out here and get no cost inputs: a traded row whose adjusted volume is not
    > 0 (a deep reverse split rounds it to 0, so there is no split factor), and a window
    of ``COST_WINDOW`` sessions that never traded. A fill in one then raises
    ``no cost inputs`` in the broker (loud, the run is void), never a guess."""
    daily = read_daily(Path(data_root), Path(volume_daily_root),
                       d - timedelta(days=COST_LOOKBACK_DAYS), d)
    out = pd.DataFrame({"symbol": daily["symbol"], "date": daily["date"],
                        "close": daily["close"], "volume": daily["volume_v"],
                        "volume_unadj": daily["volume_unadj_v"]})
    out = out.dropna(subset=["close", "volume", "volume_unadj"])
    out = out[out["symbol"].isin(minute_cache_membership(Path(data_root), d))]
    no_factor = (out["volume_unadj"] > 0) & ~(out["volume"] > 0)
    out = out[~out["symbol"].isin(out.loc[no_factor, "symbol"].unique())]
    out = out.sort_values(["symbol", "date"], kind="mergesort")
    traded = (out.groupby("symbol").tail(COST_WINDOW)
              .assign(t=lambda w: w["volume_unadj"] > 0).groupby("symbol")["t"].any())
    out = out[out["symbol"].map(traded)]
    assert (pd.to_datetime(out["date"]) < pd.Timestamp(d)).all(), "cost inputs read the session"
    return out.reset_index(drop=True)


def decision_summary(data_root: Path, d: date) -> pd.DataFrame:
    """The 09:34 quote summary rows for ``d`` (exec asserts they are used from 09:35 only)."""
    q = read_quote_summaries(Path(data_root), d,
                             minute_cache_membership(Path(data_root), d))
    q = q[q["window"] == "decision"]
    return q[["symbol", "date", "half_spread_bps", "ask_notional", "bid_notional"]]


def build_cost_source(data_root: Path, volume_daily_root: Path, d: date, *,
                      holdout_start: date, unseal: bool) -> StaticCostInputs:
    return build_cost_inputs(exec_daily_frame(data_root, volume_daily_root, d),
                             decision_summary(data_root, d), d, holdout_start=holdout_start,
                             unseal=unseal, window=COST_WINDOW)


def _jsonable(obj: Mapping[str, Any]) -> dict[str, Any]:
    def conv(v: Any) -> Any:
        if isinstance(v, (date, time)):
            return v.isoformat()
        if isinstance(v, tuple):
            return [conv(x) for x in v]
        return v
    return {k: conv(v) for k, v in obj.items()}


def run_session(session_date: date, *, arm: str, data_root: Path, volume_daily_root: Path,
                out_dir: Path, scorer: Agent, run_id: str | None = None,
                screener_config: ScreenerConfig | None = None,
                risk_config: RiskConfig | None = None, exec_config: ExecConfig | None = None,
                holdout_start: date = HOLDOUT_START, unseal: bool = False,
                repo: Path = REPO_ROOT) -> Path:
    """Replay one session through fresh agents (``scorer`` must be fresh too); return the log.

    Refuses a date ``>= holdout_start`` unless ``unseal`` before anything is read or
    written. Writes ``<out_dir>/<date>/<run_id>.jsonl``, its manifest and its rejects.
    """
    check_holdout(session_date, holdout_start, unseal)
    if scorer.name in ("screener", "feed", "risk", "execution", "monitor"):
        raise ValueError(f"scorer name {scorer.name!r} collides with a service agent")
    data_root, volume_daily_root = Path(data_root), Path(volume_daily_root)
    s_cfg = replace(screener_config or ScreenerConfig(), arm=arm, data_root=str(data_root),
                    volume_daily_root=str(volume_daily_root), holdout_start=holdout_start,
                    unseal=unseal)
    r_cfg = risk_config or RiskConfig()
    x_cfg = exec_config or ExecConfig()
    feed_source = ReplayFeed(data_root, volume_daily_root, s_cfg.unadjust_tolerance,
                             holdout_start=holdout_start, unseal=unseal)
    close_et = feed_source.close_et(session_date)
    screener = ScreenerAgent(
        ReplayScreenInputs(data_root, volume_daily_root, holdout_start, unseal), s_cfg)
    feed = FeedAgent(feed_source, session_close_et=close_et)
    risk = RiskAgent(session_date, r_cfg)
    costs = build_cost_source(data_root, volume_daily_root, session_date,
                              holdout_start=holdout_start, unseal=unseal)
    execution = ExecutionAgent(SimBroker(x_cfg, costs))
    monitor = MonitorAgent(RecordedHaltLog(()), session_close_et=time.fromisoformat(close_et))
    agents: list[Agent] = [screener, feed, scorer, risk, execution, monitor]

    log_path = session_log_path(session_date, run_id or f"{arm}-{session_date:%Y%m%d}",
                                Path(out_dir))
    ticks = clock_ticks(et_to_utc(session_date, TICKS_START_ET),
                        et_to_utc(session_date, TICKS_END_ET))
    run_replay(agents, [ticks], log_path)  # the recorder is closed when this returns

    write_rejects_jsonl(log_path.with_name(f"{log_path.stem}.rejects.jsonl"), feed.rejects)
    model_hash = str(getattr(scorer, "model_hash", scorer.name))
    config = {
        "session_date": session_date.isoformat(), "arm": arm, "unseal": unseal,
        "screener_config_hash": config_hash(s_cfg), "scorer": scorer.name,
        "risk": _jsonable(asdict(r_cfg)), "exec": _jsonable(asdict(x_cfg)),
        "ticks_et": [TICKS_START_ET, TICKS_END_ET], "cost_window": COST_WINDOW,
    }
    sha, dirty = git_state(repo)
    write_manifest(log_path, git_sha=sha, git_dirty=dirty, config=config,
                   model_hash=model_hash, unseal=unseal)
    return log_path
