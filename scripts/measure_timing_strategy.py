"""measure_timing_strategy.py — fast, deterministic strategy-quality metric
for autoresearch, for the QQQ timing signal (agents/timing).

Prints a single number to stdout: mean out-of-sample walk-forward Sharpe
across folds, penalized by the cross-fold Sharpe std (the overfitting-
analogue signal for a rule-based, unfit signal — see
utils/timing_walk_forward.py's module docstring for why this differs from
the cross-sectional IC-gap penalty in scripts/measure_strategy.py). Higher
is better.

Never hits the network — requires data/cache/ to already cover
configs/timing.yaml's configured tickers and date range (run the QQQ/VIX
UniverseAgent fetch once first if the cache is empty).
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

from agents.timing.timing_backtest_agent import TimingBacktestAgent
from agents.timing.timing_signal_agent import TimingSignalAgent
from utils.data_cache import load_or_fetch
from utils.timing_walk_forward import timing_walk_forward_validate

_CACHE_DIR = Path("data/cache")


def _no_network_fetch(missing, start, end):
    return {}


def _load_universe(cfg: dict) -> dict:
    u = cfg["universe"]
    tickers = u["tickers"]
    fetched = load_or_fetch(
        tickers, u["start_date"], u["end_date"], _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    missing = set(tickers) - set(fetched)
    assert not missing, (
        f"Missing cached coverage for {missing} over [{u['start_date']}, {u['end_date']}] "
        f"— fetch via UniverseAgent once first"
    )

    common = None
    for df in fetched.values():
        common = df.index if common is None else common.intersection(df.index)
    assert common is not None and len(common) >= u["min_history_days"], (
        f"Only {0 if common is None else len(common)} common dates after alignment "
        f"(need >= {u['min_history_days']})"
    )
    return {t: df.loc[common] for t, df in fetched.items()}


def main() -> None:
    with open("configs/timing.yaml") as fh:
        cfg = yaml.safe_load(fh)

    universe_data = _load_universe(cfg)
    ctx = {"universe_data": universe_data}

    s = cfg["signal"]
    ctx = TimingSignalAgent(
        ticker=s["ticker"],
        sma_window=s["sma_window"],
        mom_window=s.get("mom_window"),
        vol_window=s.get("vol_window"),
        vol_threshold=s.get("vol_threshold"),
        vix_threshold=s.get("vix_threshold"),
        combine=s["combine"],
    ).run(ctx)

    b = cfg["backtest"]
    ctx = TimingBacktestAgent(
        ticker=s["ticker"], transaction_cost_bps=b["transaction_cost_bps"],
    ).run(ctx)

    wf = cfg["walk_forward"]
    result = timing_walk_forward_validate(
        ctx["timing_returns"],
        n_splits=wf["n_splits"],
        periods_per_year=cfg["evaluation"]["periods_per_year"],
        purge_days=wf["purge_days"],
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
