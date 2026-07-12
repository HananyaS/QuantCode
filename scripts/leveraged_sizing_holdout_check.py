"""leveraged_sizing_holdout_check.py — leakage-free holdout check for the
tuned leverage-sizing config found by the leverage-sizing autoresearch run
(scripts/measure_leveraged_strategy.py), which converged on
vol_window=12, target_vol=0.65, max_leverage=3.0 (configs/timing.yaml).

Why this check is necessary
-----------------------------
The autoresearch search scored every candidate sizing config using a
walk-forward Sharpe computed across the FULL 2010-2024 history (5 folds via
utils/timing_walk_forward.py). That means every fold — including the most
recent, most volatile one (2020-2024: COVID crash + 2022 bear) — informed
which sizing parameters got kept at every iteration. A walk-forward score
over the whole history is not a genuine held-out test: it's possible the
search simply found parameters that happen to fit 2020-2024's specific vol
pattern well, not a parameterization that generalizes.

This mirrors the exact structure already used for the regime signal itself
(scripts/holdout_check_timing.py) and for the naive binary overlay
(scripts/leveraged_holdout_check.py) — same two independent checks:

1. Clean temporal train/holdout split. Re-select the best sizing config
   among the same "kept" candidates the real search actually favored, using
   ONLY pre-2020 data to score them (dev period). Evaluate that
   independently-selected config purely on 2020-2024 (holdout period) —
   data the selection step never saw — and compare directly against the
   full-history-search winner AND the naive overlay on the same untouched
   window. If the dev-only winner still beats naive on Sharpe/CAGR/MaxDD
   simultaneously on the holdout, that's leakage-free evidence the sizing
   improvement is real, not an artifact of tuning against the same period
   it's later "tested" on.

2. Genuine forward/blind test. Live-fetches real QQQ/^VIX/QLD/TQQQ data
   from 2025-01-01 onward — data that did not exist when the search ran,
   so leakage is structurally impossible.
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
_TQQQ_INCEPTION = "2010-02-11"

# The actual "kept" progression from the real leverage-sizing autoresearch
# run (autoresearch-results.tsv), in the order discovered — NOT cherry-picked
# after the fact. Mirrors scripts/holdout_check_timing.py's _CANDIDATES
# convention: only genuinely-evaluated-and-kept configs, plus the baseline.
_CANDIDATES = {
    "baseline_vw20_tv030": dict(vol_window=20, target_vol=0.30, max_leverage=3.0),
    "vw20_tv045": dict(vol_window=20, target_vol=0.45, max_leverage=3.0),
    "vw20_tv060": dict(vol_window=20, target_vol=0.60, max_leverage=3.0),
    "vw10_tv060": dict(vol_window=10, target_vol=0.60, max_leverage=3.0),
    "vw10_tv070": dict(vol_window=10, target_vol=0.70, max_leverage=3.0),
    "vw12_tv065": dict(vol_window=12, target_vol=0.65, max_leverage=3.0),
}
_FULL_HISTORY_WINNER = "vw12_tv065"


def _no_network_fetch(missing, start, end):
    return {}


def _load_signal_config() -> dict:
    with open("configs/timing.yaml") as fh:
        cfg = yaml.safe_load(fh)
    return cfg["signal"]


def _load_historical_universe() -> dict:
    fetched = load_or_fetch(
        ["QQQ", "^VIX"], "1999-04-01", _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    fetched.update(load_or_fetch(["QLD"], "2006-06-21", _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR))
    fetched.update(load_or_fetch(["TQQQ"], _TQQQ_INCEPTION, _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR))
    return fetched


def _build_signal(universe_data: dict, signal_cfg: dict) -> pd.Series:
    ctx = {"universe_data": universe_data}
    ctx = TimingSignalAgent(
        ticker=signal_cfg["ticker"], sma_window=signal_cfg["sma_window"],
        mom_window=signal_cfg.get("mom_window"), vol_window=signal_cfg.get("vol_window"),
        vol_threshold=signal_cfg.get("vol_threshold"), vix_threshold=signal_cfg.get("vix_threshold"),
        combine=signal_cfg["combine"],
    ).run(ctx)
    return ctx["timing_signal"]


def _build_sized_returns(universe_data: dict, signal: pd.Series, sizing_params: dict) -> dict:
    pos_ctx = {"universe_data": universe_data, "timing_signal": signal}
    pos_ctx = LeveragedPositionAgent(signal_ticker="QQQ", **sizing_params).run(pos_ctx)
    ctx = {"universe_data": universe_data, "leverage_position": pos_ctx["leverage_position"]}
    ctx = LeveragedBacktestAgent(
        tickers=("QQQ", "QLD", "TQQQ"), transaction_cost_bps=2.0, benchmark_ticker="TQQQ",
    ).run(ctx)
    ctx["timing_signal"] = signal
    return ctx


def _dev_score(returns: pd.Series, n_splits: int = 4) -> float:
    result = timing_walk_forward_validate(returns, n_splits=n_splits, periods_per_year=252)
    mean_sharpe = result["test_sharpe"].mean()
    std_sharpe = result["test_sharpe"].std()
    return float(mean_sharpe - 0.5 * (0.0 if pd.isna(std_sharpe) else std_sharpe))


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
    print("PART 1 — Clean temporal train/holdout split, leverage-sizing layer")
    print(f"  Dev (selection) period:  {_TQQQ_INCEPTION} -> {_DEV_END}")
    print(f"  Holdout period:          {_HOLDOUT_START} -> {_HOLDOUT_END}  (never used for selection)")
    print("=" * 78)

    signal_cfg = _load_signal_config()
    universe_data = _load_historical_universe()
    signal = _build_signal(universe_data, signal_cfg)

    dev_scores = {}
    ctx_by_name = {}
    for name, params in _CANDIDATES.items():
        ctx = _build_sized_returns(universe_data, signal, params)
        ctx_by_name[name] = ctx
        dev_returns = ctx["timing_returns"].loc[_TQQQ_INCEPTION:_DEV_END]
        dev_scores[name] = _dev_score(dev_returns)

    print("\nDev-only score per candidate (selection uses ONLY pre-2020 data):")
    for name, score in sorted(dev_scores.items(), key=lambda kv: -kv[1]):
        flag = "  <- full-history search also picked this" if name == _FULL_HISTORY_WINNER else ""
        print(f"    {score:+.4f}  {name} ({_CANDIDATES[name]}){flag}")

    dev_winner = max(dev_scores, key=dev_scores.get)
    print(f"\nDev-only winner: {dev_winner}  {_CANDIDATES[dev_winner]}")
    print(f"Full-history-search winner: {_FULL_HISTORY_WINNER}  {_CANDIDATES[_FULL_HISTORY_WINNER]}")
    agree = dev_winner == _FULL_HISTORY_WINNER
    print(f"Agreement: {'YES — same config' if agree else 'NO — different config'}")

    print(f"\nHoldout-period ({_HOLDOUT_START} -> {_HOLDOUT_END}) performance, "
          f"config selected WITHOUT seeing holdout data:")
    m_dev_winner = _metrics_for_window(ctx_by_name[dev_winner], _HOLDOUT_START, _HOLDOUT_END)
    _print_metrics(f"dev-selected sizing ({dev_winner})", m_dev_winner)

    print(f"\nHoldout-period performance, full-history-search winner (for comparison):")
    m_full_winner = _metrics_for_window(ctx_by_name[_FULL_HISTORY_WINNER], _HOLDOUT_START, _HOLDOUT_END)
    _print_metrics(f"full-history winner ({_FULL_HISTORY_WINNER})", m_full_winner)

    # The critical comparison: does the LEAKAGE-FREE dev-selected sizing
    # config still beat the naive binary overlay on the untouched holdout?
    naive_ctx = {"universe_data": universe_data, "timing_signal": signal}
    naive_ctx = TimingBacktestAgent(ticker="TQQQ", transaction_cost_bps=2.0).run(naive_ctx)
    m_naive = _metrics_for_window(naive_ctx, _HOLDOUT_START, _HOLDOUT_END)
    print(f"\nHoldout-period performance, naive TQQQ-or-cash overlay (for comparison):")
    _print_metrics("naive overlay", m_naive)

    print("\n--- Verdict: does the dev-selected (leakage-free) sizing beat naive on the holdout? ---")
    beats_sharpe = m_dev_winner["strategy_sharpe"] > m_naive["strategy_sharpe"]
    beats_cagr = m_dev_winner["strategy_cagr"] > m_naive["strategy_cagr"]
    beats_dd = m_dev_winner["strategy_max_drawdown"] > m_naive["strategy_max_drawdown"]
    print(f"  Sharpe:  {'BEATS' if beats_sharpe else 'DOES NOT BEAT'} naive "
          f"({m_dev_winner['strategy_sharpe']:+.3f} vs {m_naive['strategy_sharpe']:+.3f})")
    print(f"  CAGR:    {'BEATS' if beats_cagr else 'DOES NOT BEAT'} naive "
          f"({m_dev_winner['strategy_cagr']:+.1%} vs {m_naive['strategy_cagr']:+.1%})")
    print(f"  MaxDD:   {'BEATS' if beats_dd else 'DOES NOT BEAT'} naive "
          f"({m_dev_winner['strategy_max_drawdown']:.1%} vs {m_naive['strategy_max_drawdown']:.1%})")


def part2_genuine_forward_test() -> None:
    print()
    print("=" * 78)
    print("PART 2 — Genuine forward/blind test, leverage-sizing layer")
    print("=" * 78)

    signal_cfg = _load_signal_config()

    try:
        agent = UniverseAgent(
            tickers=["QQQ", "^VIX", "QLD", "TQQQ"], start_date="2024-01-01", end_date="2026-07-11",
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

    signal = _build_signal(universe_data, signal_cfg)
    forward_end = str(span.max().date())

    sized_ctx = _build_sized_returns(universe_data, signal, _CANDIDATES[_FULL_HISTORY_WINNER])
    m_sized = _metrics_for_window(sized_ctx, forward_start, forward_end)

    naive_ctx = {"universe_data": universe_data, "timing_signal": signal}
    naive_ctx = TimingBacktestAgent(ticker="TQQQ", transaction_cost_bps=2.0).run(naive_ctx)
    m_naive = _metrics_for_window(naive_ctx, forward_start, forward_end)

    print(f"\n  Blind window: {forward_start} -> {forward_end} "
          f"(sizing frozen: {_CANDIDATES[_FULL_HISTORY_WINNER]}, zero refitting)")
    _print_metrics("tuned vol-sized", m_sized)
    _print_metrics("naive overlay (for comparison)", m_naive)

    print("\n  Caveat: this window is short (~1.5 years) and low-powered — treat as a")
    print("  directional sanity check, not statistical proof either way.")


if __name__ == "__main__":
    try:
        part1_clean_holdout_split()
        part2_genuine_forward_test()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
