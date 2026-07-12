"""measure_leveraged_strategy.py — fast, deterministic strategy-quality
metric for autoresearch, for the leveraged-ETF position-sizing layer
(agents/timing/leveraged_position_agent.py).

Prints a single number to stdout: mean out-of-sample walk-forward Sharpe
across folds (LeveragedBacktestAgent's returns), penalized by cross-fold
Sharpe std — same formula as scripts/measure_timing_strategy.py, applied to
the leveraged position instead of the single-instrument one.

The regime SIGNAL itself (configs/timing.yaml's `signal` section) is held
fixed at its already-tuned value — this script only searches the sizing
layer (leverage_sizing section) that sits downstream of it. Never hits the
network — requires data/cache/ to already cover QQQ/^VIX/QLD/TQQQ.
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

from agents.timing.leveraged_backtest_agent import LeveragedBacktestAgent
from agents.timing.leveraged_position_agent import LeveragedPositionAgent
from agents.timing.timing_signal_agent import TimingSignalAgent
from utils.data_cache import load_or_fetch
from utils.timing_walk_forward import timing_walk_forward_validate

_CACHE_DIR = Path("data/cache")


def _no_network_fetch(missing, start, end):
    return {}


def _load_universe(cfg: dict) -> dict:
    u = cfg["universe"]
    lb = cfg["leverage_backtest"]
    fetched = load_or_fetch(
        u["tickers"], u["start_date"], u["end_date"], _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    for t in lb["tickers"]:
        if t in fetched:
            continue
        fetched.update(load_or_fetch(
            [t], lb["eval_start"], lb["eval_end"], _no_network_fetch, cache_dir=_CACHE_DIR,
        ))
    missing = set(u["tickers"] + lb["tickers"]) - set(fetched)
    assert not missing, f"Missing cached coverage for {missing} — fetch via UniverseAgent once first"
    return fetched


def main() -> None:
    with open("configs/timing.yaml") as fh:
        cfg = yaml.safe_load(fh)

    universe_data = _load_universe(cfg)
    ctx = {"universe_data": universe_data}

    # Fixed, already-tuned regime signal — NOT part of this search.
    s = cfg["signal"]
    ctx = TimingSignalAgent(
        ticker=s["ticker"], sma_window=s["sma_window"], mom_window=s.get("mom_window"),
        vol_window=s.get("vol_window"), vol_threshold=s.get("vol_threshold"),
        vix_threshold=s.get("vix_threshold"), combine=s["combine"],
    ).run(ctx)

    ls = cfg["leverage_sizing"]
    ctx = LeveragedPositionAgent(
        signal_ticker=ls["signal_ticker"], vol_window=ls["vol_window"],
        target_vol=ls["target_vol"], max_leverage=ls["max_leverage"],
    ).run(ctx)

    lb = cfg["leverage_backtest"]
    ctx = LeveragedBacktestAgent(
        tickers=tuple(lb["tickers"]), transaction_cost_bps=lb["transaction_cost_bps"],
        benchmark_ticker=lb["benchmark_ticker"],
    ).run(ctx)

    eval_returns = ctx["timing_returns"].loc[lb["eval_start"]:lb["eval_end"]]

    lwf = cfg["leverage_walk_forward"]
    result = timing_walk_forward_validate(
        eval_returns, n_splits=lwf["n_splits"], periods_per_year=cfg["evaluation"]["periods_per_year"],
        purge_days=lwf["purge_days"],
    )
    mean_sharpe = result["test_sharpe"].mean()
    std_sharpe = result["test_sharpe"].std()
    score = mean_sharpe - lwf["overfit_penalty"] * (0.0 if pd.isna(std_sharpe) else std_sharpe)
    print(f"{score:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
