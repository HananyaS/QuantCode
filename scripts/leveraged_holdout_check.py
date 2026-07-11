"""leveraged_holdout_check.py — out-of-sample holdout check for the naive
TQQQ/QLD overlay (scripts/leveraged_overlay_check.py).

The overlay itself introduces no new leakage risk — the signal's parameters
were chosen entirely from QQQ/VIX data and never touched TQQQ/QLD prices.
But the underlying signal DOES carry the leakage flagged in
scripts/holdout_check_timing.py: the full-history-search winner
(sma_window=190) was selected using a walk-forward score spanning
1999-2024, so its parameter choice was informed by how it performed in
2020-2024 — and TQQQ/QLD's price action in that same window is extremely
correlated with QQQ's (same underlying index, leveraged), so that leakage
carries over to a TQQQ backtest of the same period.

This script re-runs the same two checks as holdout_check_timing.py, this
time evaluating the leveraged instruments:

1. Clean temporal split — apply BOTH the dev-only-selected config
   (sma_window=175, chosen using ONLY pre-2020 QQQ data) and the
   full-history-search winner (sma_window=190) to TQQQ/QLD, scored
   separately on the pre-2020 "dev" window and the untouched 2020-2024
   holdout. If the dev-selected config still beats leveraged buy-and-hold
   on the holdout, that's leakage-free evidence the leveraged-overlay
   effect is real, not an artifact of the original signal search peeking
   at the same period TQQQ is later tested on.

2. Genuine blind forward test — live-fetch real TQQQ/QLD (+QQQ/VIX for the
   signal) from 2025-01-01 onward, data that did not exist when either
   config was chosen.
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

_CACHE_DIR = Path("data/cache")
_DEV_END = "2019-12-31"
_HOLDOUT_START = "2020-01-01"
_HOLDOUT_END = "2024-12-27"
_LEVERAGED_TICKERS = ["TQQQ", "QLD"]
_LEVERAGED_INCEPTION = {"TQQQ": "2010-02-11", "QLD": "2006-06-21"}

_CONFIGS = {
    "dev_selected_sma175": dict(sma_window=175, mom_window=None, vol_window=20,
                                 vol_threshold=0.30, vix_threshold=20.0, combine="majority"),
    "full_history_sma190": dict(sma_window=190, mom_window=None, vol_window=20,
                                 vol_threshold=0.30, vix_threshold=20.0, combine="majority"),
}


def _no_network_fetch(missing, start, end):
    return {}


def _load_historical_universe() -> dict:
    fetched = load_or_fetch(
        ["QQQ", "^VIX"], "1999-04-01", _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    for t in _LEVERAGED_TICKERS:
        fetched.update(load_or_fetch(
            [t], _LEVERAGED_INCEPTION[t], _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR,
        ))
    return fetched


def _metrics_for_window(ctx: dict, start: str, end: str) -> dict:
    sl = slice(start, end)
    window_ctx = {
        "timing_returns": ctx["timing_returns"].loc[sl],
        "timing_equity": (1 + ctx["timing_returns"].loc[sl]).cumprod(),
        "benchmark_returns": ctx["benchmark_returns"].loc[sl],
        "benchmark_equity": (1 + ctx["benchmark_returns"].loc[sl]).cumprod(),
        "timing_signal": ctx["timing_signal"].loc[sl],
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
    print("PART 1 — Clean temporal split, leveraged overlay")
    print(f"  Dev window (pre-signal-selection cutoff): -> {_DEV_END}")
    print(f"  Holdout window (never seen by dev-only selection): {_HOLDOUT_START} -> {_HOLDOUT_END}")
    print("=" * 78)

    universe_data = _load_historical_universe()

    for cfg_name, params in _CONFIGS.items():
        signal_ctx = {"universe_data": universe_data}
        signal_ctx = TimingSignalAgent(ticker="QQQ", **params).run(signal_ctx)
        signal = signal_ctx["timing_signal"]

        print(f"\n--- signal config: {cfg_name} ---")
        for lev_ticker in _LEVERAGED_TICKERS:
            lev_dates = universe_data[lev_ticker].index
            dev_start = str(lev_dates.min().date())

            ctx = {"universe_data": universe_data, "timing_signal": signal}
            ctx = TimingBacktestAgent(ticker=lev_ticker, transaction_cost_bps=2.0).run(ctx)

            if pd.Timestamp(dev_start) < pd.Timestamp(_DEV_END):
                m_dev = _metrics_for_window(ctx, dev_start, _DEV_END)
                _print_metrics(f"{lev_ticker} dev ({dev_start} -> {_DEV_END})", m_dev)
            else:
                print(f"  {lev_ticker}: no dev-period data (inception {dev_start} is after dev cutoff)")

            m_holdout = _metrics_for_window(ctx, _HOLDOUT_START, _HOLDOUT_END)
            _print_metrics(f"{lev_ticker} holdout ({_HOLDOUT_START} -> {_HOLDOUT_END})", m_holdout)


def part2_genuine_forward_test() -> None:
    print()
    print("=" * 78)
    print("PART 2 — Genuine forward/blind test, leveraged overlay")
    print("=" * 78)

    try:
        agent = UniverseAgent(
            tickers=["QQQ", "^VIX"] + _LEVERAGED_TICKERS, start_date="2024-01-01", end_date="2026-07-11",
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

    signal_ctx = {"universe_data": universe_data}
    signal_ctx = TimingSignalAgent(ticker="QQQ", **_CONFIGS["full_history_sma190"]).run(signal_ctx)
    signal = signal_ctx["timing_signal"]
    forward_end = str(span.max().date())

    for lev_ticker in _LEVERAGED_TICKERS:
        ctx = {"universe_data": universe_data, "timing_signal": signal}
        ctx = TimingBacktestAgent(ticker=lev_ticker, transaction_cost_bps=2.0).run(ctx)
        m = _metrics_for_window(ctx, forward_start, forward_end)
        print(f"\n  Blind window: {forward_start} -> {forward_end}  (ticker={lev_ticker}, "
              f"signal=full_history_sma190, frozen)")
        _print_metrics(lev_ticker, m)

    print("\n  Caveat: this window is short (~1.5 years) and low-powered — treat as a")
    print("  directional sanity check, not statistical proof either way.")


if __name__ == "__main__":
    try:
        part1_clean_holdout_split()
        part2_genuine_forward_test()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
