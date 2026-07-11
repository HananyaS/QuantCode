"""holdout_check_timing.py — out-of-sample holdout check for the QQQ timing
signal found by scripts/measure_timing_strategy.py's autoresearch run.

Why this is needed
-------------------
The autoresearch loop scored every candidate using a walk-forward Sharpe
computed across 6 folds spanning the FULL 1999-2024 history. That means
every fold's data — including the most recent 2020-2024 fold — informed
which config got kept at every iteration. That's real, if mild, look-ahead
leakage in the *parameter selection* itself (the underlying signal is
rule-based causal, but the choice of sma_window/vol_threshold/vix_threshold
was made with knowledge of how those choices performed in every era,
including the most recent one). A "walk-forward" search over the whole
history is not the same thing as a genuine held-out test.

This script runs two independent checks that don't have that problem:

1. **Clean temporal train/holdout split.** Re-select the best config among
   the same candidates tried during the original search, but using ONLY
   pre-2020 data to score them (dev period). Then evaluate that
   independently-selected config purely on 2020-2024 (holdout period) —
   data the selection step never saw. Also evaluates the ORIGINAL
   full-history-search winner on the same holdout, so the two can be
   compared directly: did full-history search pick something that only
   looks good because it saw the holdout, or does dev-only selection agree?

2. **Genuine forward/blind test.** Fetches real QQQ/^VIX data for the
   period after the search's data cutoff (2025-01-01 onward) — data that
   did not exist when the search ran, so leakage is structurally
   impossible. This window is short and low-powered, but it's the only
   truly blind evidence available.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.timing.timing_backtest_agent import TimingBacktestAgent
from agents.timing.timing_evaluation_agent import TimingEvaluationAgent
from agents.timing.timing_signal_agent import TimingSignalAgent
from agents.universe_agent import UniverseAgent
from utils.data_cache import load_or_fetch
from utils.timing_walk_forward import timing_walk_forward_validate

_CACHE_DIR = Path("data/cache")
_DEV_END = "2019-12-31"
_HOLDOUT_START = "2020-01-01"
_HOLDOUT_END = "2024-12-27"

# The progression of "kept" configs from the actual autoresearch run, plus
# the Faber baseline, in the order they were discovered — this is the same
# candidate space the original search explored, just re-scored on dev-only
# data instead of the full history.
_CANDIDATES = {
    "baseline_sma_only_200": dict(sma_window=200, mom_window=None, vol_window=None,
                                   vol_threshold=None, vix_threshold=None, combine="sma_only"),
    "sma200_vol20_030_allagree": dict(sma_window=200, mom_window=None, vol_window=20,
                                       vol_threshold=0.30, vix_threshold=None, combine="all_agree"),
    "sma200_vol20_030_vix30_majority": dict(sma_window=200, mom_window=None, vol_window=20,
                                             vol_threshold=0.30, vix_threshold=30.0, combine="majority"),
    "sma200_vol20_030_vix25_majority": dict(sma_window=200, mom_window=None, vol_window=20,
                                             vol_threshold=0.30, vix_threshold=25.0, combine="majority"),
    "sma200_vol20_030_vix20_majority": dict(sma_window=200, mom_window=None, vol_window=20,
                                             vol_threshold=0.30, vix_threshold=20.0, combine="majority"),
    "sma150_vol20_030_vix20_majority": dict(sma_window=150, mom_window=None, vol_window=20,
                                             vol_threshold=0.30, vix_threshold=20.0, combine="majority"),
    "sma175_vol20_030_vix20_majority": dict(sma_window=175, mom_window=None, vol_window=20,
                                             vol_threshold=0.30, vix_threshold=20.0, combine="majority"),
    "sma190_vol20_030_vix20_majority": dict(sma_window=190, mom_window=None, vol_window=20,
                                             vol_threshold=0.30, vix_threshold=20.0, combine="majority"),
}
_FULL_HISTORY_WINNER = "sma190_vol20_030_vix20_majority"


def _no_network_fetch(missing, start, end):
    return {}


def _load_full_history() -> dict:
    fetched = load_or_fetch(
        ["QQQ", "^VIX"], "1999-04-01", _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    common = None
    for df in fetched.values():
        common = df.index if common is None else common.intersection(df.index)
    return {t: df.loc[common] for t, df in fetched.items()}


def _run_signal_and_backtest(universe_data: dict, params: dict) -> dict:
    ctx = {"universe_data": universe_data}
    ctx = TimingSignalAgent(ticker="QQQ", **params).run(ctx)
    ctx = TimingBacktestAgent(ticker="QQQ", transaction_cost_bps=2.0).run(ctx)
    return ctx


def _dev_score(returns: pd.Series, n_splits: int = 4) -> float:
    """Same scoring formula as measure_timing_strategy.py, restricted to
    whatever date range `returns` already covers (the caller slices to dev)."""
    result = timing_walk_forward_validate(returns, n_splits=n_splits, periods_per_year=252)
    mean_sharpe = result["test_sharpe"].mean()
    std_sharpe = result["test_sharpe"].std()
    return float(mean_sharpe - 0.5 * (0.0 if pd.isna(std_sharpe) else std_sharpe))


def _metrics_for_window(ctx: dict, start: str, end: str) -> dict:
    window_ctx = {
        "timing_returns": ctx["timing_returns"].loc[start:end],
        "timing_equity": (1 + ctx["timing_returns"].loc[start:end]).cumprod(),
        "benchmark_returns": ctx["benchmark_returns"].loc[start:end],
        "benchmark_equity": (1 + ctx["benchmark_returns"].loc[start:end]).cumprod(),
        "timing_signal": ctx["timing_signal"].loc[start:end],
    }
    return TimingEvaluationAgent(periods_per_year=252).run(window_ctx)["timing_metrics"]


def _print_metrics(label: str, m: dict) -> None:
    print(f"  {label}:")
    print(f"    strategy  Sharpe={m['strategy_sharpe']:+.3f}  CAGR={m['strategy_cagr']:+.1%}  "
          f"MaxDD={m['strategy_max_drawdown']:.1%}  time_in_mkt={m['time_in_market']:.1%}")
    print(f"    buy&hold  Sharpe={m['benchmark_sharpe']:+.3f}  CAGR={m['benchmark_cagr']:+.1%}  "
          f"MaxDD={m['benchmark_max_drawdown']:.1%}")


def part1_clean_holdout_split() -> None:
    print("=" * 78)
    print("PART 1 — Clean temporal train/holdout split")
    print(f"  Dev (selection) period:  1999-04-01 -> {_DEV_END}")
    print(f"  Holdout period:          {_HOLDOUT_START} -> {_HOLDOUT_END}  (never used for selection)")
    print("=" * 78)

    universe_data = _load_full_history()

    dev_scores = {}
    for name, params in _CANDIDATES.items():
        ctx = _run_signal_and_backtest(universe_data, params)
        dev_returns = ctx["timing_returns"].loc[:_DEV_END]
        dev_scores[name] = _dev_score(dev_returns)

    print("\nDev-only score per candidate (selection uses ONLY pre-2020 data):")
    for name, score in sorted(dev_scores.items(), key=lambda kv: -kv[1]):
        flag = "  <- full-history search also picked this" if name == _FULL_HISTORY_WINNER else ""
        print(f"    {score:+.4f}  {name}{flag}")

    dev_winner = max(dev_scores, key=dev_scores.get)
    print(f"\nDev-only winner: {dev_winner}")
    print(f"Full-history-search winner: {_FULL_HISTORY_WINNER}")
    agree = dev_winner == _FULL_HISTORY_WINNER
    print(f"Agreement: {'YES — same config' if agree else 'NO — different config'}")

    print("\nHoldout-period (2020-2024) performance, config selected WITHOUT seeing holdout data:")
    ctx_dev_winner = _run_signal_and_backtest(universe_data, _CANDIDATES[dev_winner])
    m = _metrics_for_window(ctx_dev_winner, _HOLDOUT_START, _HOLDOUT_END)
    _print_metrics(f"dev-selected config ({dev_winner})", m)

    print("\nHoldout-period (2020-2024) performance, full-history-search winner (for comparison):")
    ctx_full_winner = _run_signal_and_backtest(universe_data, _CANDIDATES[_FULL_HISTORY_WINNER])
    m2 = _metrics_for_window(ctx_full_winner, _HOLDOUT_START, _HOLDOUT_END)
    _print_metrics(f"full-history winner ({_FULL_HISTORY_WINNER})", m2)


def part2_genuine_forward_test() -> None:
    print()
    print("=" * 78)
    print("PART 2 — Genuine forward/blind test (data that didn't exist during the search)")
    print("=" * 78)

    try:
        agent = UniverseAgent(
            tickers=["QQQ", "^VIX"], start_date="2024-01-01", end_date="2026-07-11",
            benchmark="QQQ", min_history_days=30, min_assets=1, data_source="yfinance",
        )
        ctx_raw = agent.run({})
    except Exception as exc:
        print(f"  SKIPPED — live fetch failed: {exc}")
        return

    universe_data = ctx_raw["universe_data"]
    span = universe_data["QQQ"].index
    print(f"  Fetched {span.min().date()} -> {span.max().date()} ({len(span)} bars)")

    forward_start = "2025-01-01"
    if span.max() < pd.Timestamp(forward_start) + pd.Timedelta(days=60):
        print("  SKIPPED — not enough post-cutoff data yet for a meaningful blind window")
        return

    ctx = _run_signal_and_backtest(universe_data, _CANDIDATES[_FULL_HISTORY_WINNER])
    forward_end = str(span.max().date())
    m = _metrics_for_window(ctx, forward_start, forward_end)
    print(f"\n  Blind window: {forward_start} -> {forward_end} "
          f"(config: {_FULL_HISTORY_WINNER}, frozen, zero refitting)")
    _print_metrics("forward/blind", m)
    print("\n  Caveat: this window is short (~1.5 years) and low-powered — treat as a")
    print("  directional sanity check, not statistical proof either way.")


if __name__ == "__main__":
    try:
        part1_clean_holdout_split()
        part2_genuine_forward_test()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
