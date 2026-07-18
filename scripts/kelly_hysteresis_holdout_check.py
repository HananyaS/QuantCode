"""kelly_hysteresis_holdout_check.py — leakage-free holdout check for the
Kelly hysteresis-parameter autoresearch winner (fractional_kelly=0.75,
ewma_decay=0.90, mu_decay=0.92, adx_threshold=20, drawdown_limit=0.20,
vol_spike_threshold=0.35).

Unlike the earlier vol-sized-layer search, THIS search genuinely selected
parameters using a walk-forward score computed across the full 1999-2024
history — so unlike the (unfounded) worry about the raw Kelly-formula
defaults, this one has real selection-leakage risk and this check matters.

Mirrors scripts/holdout_check_timing.py / leveraged_sizing_holdout_check.py:

1. Clean temporal split — re-select among the actual "kept" progression
   from the real search (autoresearch-results.tsv), using ONLY pre-2020
   data. Evaluate both the dev-only winner and the full-history winner on
   the untouched 2020-2024 holdout.
2. Genuine blind forward test — live-fetch real 2025+ QQQ data (didn't
   exist when the search ran) and evaluate the frozen full-history winner.

IMPORTANT: this search ran on code that has since been fixed for a same-day
lookahead bug (see agents/timing/kelly_position_agent.py, commit 99d6491);
this script uses the CURRENT (fixed) code throughout, consistent with the
corrected re-run the search itself was restarted from.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.timing.kelly_backtest_agent import KellyBacktestAgent
from agents.timing.kelly_evaluation_agent import KellyEvaluationAgent
from agents.timing.kelly_position_agent import KellyPositionAgent
from agents.universe_agent import UniverseAgent
from utils.data_cache import load_or_fetch
from utils.synthetic_leveraged_series import build_synthetic_universe
from utils.timing_walk_forward import timing_walk_forward_validate

_CACHE_DIR = Path("data/cache")
_DEV_END = "2019-12-31"
_HOLDOUT_START = "2020-01-01"
_HOLDOUT_END = "2024-12-27"

_BASE = dict(
    signal_ticker="QQQ", vol_method="ewma",
    garch_refit_every=20, garch_min_train_obs=250,
    adx_period=14, max_leverage=3.0, worst_case_daily_move=0.20, ruin_buffer=0.20,
    entry_margin=0.3, min_observations=60,
)

# The actual "kept" progression from the RE-RUN search (after fixing the
# symmetric-hysteresis bug, commit 65a971c), in the order discovered -- not
# cherry-picked after the fact. Supersedes the pre-fix progression this
# script originally checked (fractional_kelly/vol_decay tuning from that
# run no longer applies; mu_decay turned out to be the dominant lever this
# time).
_CANDIDATES = {
    "baseline": dict(fractional_kelly=0.5, vol_decay=0.94, mu_decay=0.94,
                      adx_threshold=25.0, drawdown_limit=0.15, vol_spike_threshold=0.40),
    "v1_adx20": dict(fractional_kelly=0.5, vol_decay=0.94, mu_decay=0.94,
                      adx_threshold=20.0, drawdown_limit=0.15, vol_spike_threshold=0.40),
    "v2_adx15": dict(fractional_kelly=0.5, vol_decay=0.94, mu_decay=0.94,
                      adx_threshold=15.0, drawdown_limit=0.15, vol_spike_threshold=0.40),
    "v3_vspike030": dict(fractional_kelly=0.5, vol_decay=0.94, mu_decay=0.94,
                          adx_threshold=15.0, drawdown_limit=0.15, vol_spike_threshold=0.30),
    "v4_vspike025": dict(fractional_kelly=0.5, vol_decay=0.94, mu_decay=0.94,
                          adx_threshold=15.0, drawdown_limit=0.15, vol_spike_threshold=0.25),
    "v5_mudecay097": dict(fractional_kelly=0.5, vol_decay=0.94, mu_decay=0.97,
                           adx_threshold=15.0, drawdown_limit=0.15, vol_spike_threshold=0.25),
    "v6_mudecay099": dict(fractional_kelly=0.5, vol_decay=0.94, mu_decay=0.99,
                           adx_threshold=15.0, drawdown_limit=0.15, vol_spike_threshold=0.25),
    "v7_mudecay0995": dict(fractional_kelly=0.5, vol_decay=0.94, mu_decay=0.995,
                            adx_threshold=15.0, drawdown_limit=0.15, vol_spike_threshold=0.25),
    "v8_final": dict(fractional_kelly=0.5, vol_decay=0.94, mu_decay=0.995,
                      adx_threshold=15.0, drawdown_limit=0.15, vol_spike_threshold=0.35),
}
_FULL_HISTORY_WINNER = "v8_final"


def _no_network_fetch(missing, start, end):
    return {}


def _load_historical_universe() -> dict:
    fetched = load_or_fetch(["QQQ"], "1999-04-01", _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR)
    assert "QQQ" in fetched, "QQQ cache missing — run UniverseAgent once first"
    return build_synthetic_universe(fetched["QQQ"]["Close"])


def _run_backtest(universe_data: dict, params: dict) -> dict:
    ctx = {"universe_data": universe_data}
    ctx = KellyPositionAgent(**_BASE, **params).run(ctx)
    ctx = KellyBacktestAgent(
        tickers=("QQQ", "QLD", "TQQQ"), transaction_cost_bps=2.0, benchmark_ticker="TQQQ",
    ).run(ctx)
    return ctx


def _dev_score(returns: pd.Series, n_splits: int = 4) -> float:
    result = timing_walk_forward_validate(returns, n_splits=n_splits, periods_per_year=252)
    mean_sharpe = result["test_sharpe"].mean()
    std_sharpe = result["test_sharpe"].std()
    return float(mean_sharpe - 0.5 * (0.0 if pd.isna(std_sharpe) else std_sharpe))


def _metrics_for_window(ctx: dict, start: str, end: str) -> dict:
    sl = slice(start, end)
    window_ctx = {
        "universe_data": ctx["universe_data"],
        "timing_returns": ctx["timing_returns"].loc[sl],
        "timing_equity": (1 + ctx["timing_returns"].loc[sl]).cumprod(),
        "benchmark_returns": ctx["benchmark_returns"].loc[sl],
        "benchmark_equity": (1 + ctx["benchmark_returns"].loc[sl]).cumprod(),
        "leverage_position": ctx["leverage_position"].loc[sl],
    }
    return KellyEvaluationAgent(periods_per_year=252, underlying_ticker="QQQ").run(window_ctx)["kelly_metrics"]


def _print_metrics(label: str, m: dict) -> None:
    print(f"  {label}:")
    print(f"    strategy  Sharpe={m['strategy_sharpe']:+.3f}  Sortino={m['strategy_sortino']:+.3f}  "
          f"CAGR={m['strategy_cagr']:+.1%}  MaxDD={m['strategy_max_drawdown']:.1%}  "
          f"avg_lev={m['avg_leverage']:.2f}x")
    print(f"    buy&hold  Sharpe={m['benchmark_sharpe']:+.3f}  Sortino={m['benchmark_sortino']:+.3f}  "
          f"CAGR={m['benchmark_cagr']:+.1%}  MaxDD={m['benchmark_max_drawdown']:.1%}")


def part1_clean_holdout_split() -> None:
    print("=" * 78)
    print("PART 1 — Clean temporal train/holdout split, Kelly hysteresis params")
    print(f"  Dev (selection) period:  1999-04-01 -> {_DEV_END}")
    print(f"  Holdout period:          {_HOLDOUT_START} -> {_HOLDOUT_END} (never used for selection)")
    print("=" * 78)

    universe_data = _load_historical_universe()

    dev_scores = {}
    ctx_by_name = {}
    for name, params in _CANDIDATES.items():
        ctx = _run_backtest(universe_data, params)
        ctx_by_name[name] = ctx
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

    print(f"\nHoldout-period ({_HOLDOUT_START} -> {_HOLDOUT_END}) performance, "
          f"config selected WITHOUT seeing holdout data:")
    m_dev = _metrics_for_window(ctx_by_name[dev_winner], _HOLDOUT_START, _HOLDOUT_END)
    _print_metrics(f"dev-selected ({dev_winner})", m_dev)

    print(f"\nHoldout-period performance, full-history-search winner (for comparison):")
    m_full = _metrics_for_window(ctx_by_name[_FULL_HISTORY_WINNER], _HOLDOUT_START, _HOLDOUT_END)
    _print_metrics(f"full-history winner ({_FULL_HISTORY_WINNER})", m_full)

    print(f"\nHoldout-period performance, baseline defaults (for comparison):")
    m_baseline = _metrics_for_window(ctx_by_name["baseline"], _HOLDOUT_START, _HOLDOUT_END)
    _print_metrics("baseline defaults", m_baseline)


def part2_genuine_forward_test() -> None:
    print()
    print("=" * 78)
    print("PART 2 — Genuine forward/blind test, Kelly hysteresis params")
    print("=" * 78)

    try:
        agent = UniverseAgent(
            tickers=["QQQ"], start_date="2022-01-01", end_date="2026-07-11",
            benchmark="QQQ", min_history_days=30, min_assets=1, data_source="yfinance",
        )
        ctx_raw = agent.run({})
    except Exception as exc:
        print(f"  SKIPPED — live fetch failed: {exc}")
        return

    qqq_close = ctx_raw["universe_data"]["QQQ"]["Close"]
    universe_data = build_synthetic_universe(qqq_close)
    span = qqq_close.index
    print(f"  Fetched {span.min().date()} -> {span.max().date()} ({len(span)} bars)")

    forward_start = "2025-01-01"
    if span.max() < pd.Timestamp(forward_start) + pd.Timedelta(days=60):
        print("  SKIPPED — not enough post-cutoff data yet")
        return
    forward_end = str(span.max().date())

    ctx_final = _run_backtest(universe_data, _CANDIDATES[_FULL_HISTORY_WINNER])
    m_final = _metrics_for_window(ctx_final, forward_start, forward_end)
    print(f"\n  Blind window: {forward_start} -> {forward_end}  (config: {_FULL_HISTORY_WINNER}, frozen)")
    _print_metrics("Kelly-tuned", m_final)

    ctx_base = _run_backtest(universe_data, _CANDIDATES["baseline"])
    m_base = _metrics_for_window(ctx_base, forward_start, forward_end)
    _print_metrics("baseline defaults (for comparison)", m_base)

    print("\n  Caveat: short (~1.5yr), low-powered window — directional signal only.")


if __name__ == "__main__":
    try:
        part1_clean_holdout_split()
        part2_genuine_forward_test()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
