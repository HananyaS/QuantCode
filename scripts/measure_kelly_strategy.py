"""measure_kelly_strategy.py — fast, deterministic strategy-quality metric
for autoresearch, for the Kelly-criterion leverage-sizing layer
(agents/timing/kelly_position_agent.py).

Prints a single number to stdout: mean out-of-sample walk-forward Sharpe
across folds (KellyBacktestAgent's returns), penalized by cross-fold Sharpe
std — same formula as scripts/measure_leveraged_strategy.py and
scripts/measure_timing_strategy.py. Higher is better.

Scope deliberately EXCLUDES worst_case_daily_move and ruin_buffer — these
are safety constants (the ruin-floor cap), not signal parameters, and
fitting them to historical backtest score would create exactly the wrong
incentive (loosen the safety margin to chase a better number). Only the
regime/vol/hysteresis parameters that determine WHEN to rotate are in
scope.

Never hits the network — requires data/cache/QQQ.parquet already covering
configs/kelly_timing.yaml's date range.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

from agents.timing.kelly_backtest_agent import KellyBacktestAgent
from agents.timing.kelly_position_agent import KellyPositionAgent
from utils.data_cache import load_or_fetch
from utils.synthetic_leveraged_series import build_synthetic_universe
from utils.timing_walk_forward import timing_walk_forward_validate

_CACHE_DIR = Path("data/cache")


def _no_network_fetch(missing, start, end):
    return {}


def _load_universe(cfg: dict) -> dict:
    u = cfg["universe"]
    fetched = load_or_fetch(
        [u["underlying_ticker"]], u["start_date"], u["end_date"], _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    assert u["underlying_ticker"] in fetched, (
        f"Missing cached coverage for {u['underlying_ticker']} — fetch via UniverseAgent once first"
    )
    return build_synthetic_universe(fetched[u["underlying_ticker"]]["Close"])


def main() -> None:
    with open("configs/kelly_timing.yaml") as fh:
        cfg = yaml.safe_load(fh)

    universe_data = _load_universe(cfg)
    ctx = {"universe_data": universe_data}

    cv, rg, ks = cfg["conditional_vol"], cfg["regime"], cfg["kelly_sizing"]
    ctx = KellyPositionAgent(
        signal_ticker=cfg["universe"]["underlying_ticker"],
        vol_method=cv["method"], vol_decay=cv["ewma_decay"],
        garch_refit_every=cv["garch_refit_every"], garch_min_train_obs=cv["garch_min_train_obs"],
        mu_decay=rg["mu_decay"], adx_period=rg["adx_period"], adx_threshold=rg["adx_threshold"],
        fractional_kelly=ks["fractional_kelly"], max_leverage=ks["max_leverage"],
        worst_case_daily_move=ks["worst_case_daily_move"], ruin_buffer=ks["ruin_buffer"],
        vol_spike_threshold=ks["vol_spike_threshold"], drawdown_limit=ks["drawdown_limit"],
        entry_margin=ks["entry_margin"], min_observations=ks["min_observations"],
    ).run(ctx)

    ctx = KellyBacktestAgent(
        tickers=("QQQ", "QLD", "TQQQ"), transaction_cost_bps=cfg["backtest"]["transaction_cost_bps"],
        benchmark_ticker="TQQQ",
    ).run(ctx)

    wf = cfg["walk_forward"]
    result = timing_walk_forward_validate(
        ctx["timing_returns"], n_splits=wf["n_splits"],
        periods_per_year=cfg["evaluation"]["periods_per_year"], purge_days=wf["purge_days"],
    )
    mean_sharpe = result["test_sharpe"].mean()
    std_sharpe = result["test_sharpe"].std()
    score = mean_sharpe - wf["overfit_penalty"] * (0.0 if pd.isna(std_sharpe) else std_sharpe)
    print(f"{score:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
