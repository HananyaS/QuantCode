"""vol_sized_overlay_check.py — compares the vol-sized leveraged position
layer (LeveragedPositionAgent/LeveragedBacktestAgent) against the naive
binary QQQ-signal-drives-TQQQ-or-cash overlay
(scripts/leveraged_overlay_check.py) and plain TQQQ buy-and-hold.

Reads the regime signal (`signal`) and sizing (`leverage_sizing`) configs
live from configs/timing.yaml rather than hardcoding them, so this always
reflects whatever autoresearch has most recently tuned
(scripts/measure_leveraged_strategy.py is the Verify metric for the sizing
search) instead of going stale.
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
from utils.data_cache import load_or_fetch

_CACHE_DIR = Path("data/cache")
_HOLDOUT_END = "2024-12-27"


def _no_network_fetch(missing, start, end):
    return {}


def _load_universe() -> dict:
    fetched = load_or_fetch(["QQQ", "^VIX"], "1999-04-01", _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR)
    fetched.update(load_or_fetch(["QLD"], "2006-06-21", _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR))
    fetched.update(load_or_fetch(["TQQQ"], "2010-02-11", _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR))
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


def _print_row(label: str, m: dict) -> None:
    print(f"  {label:<28} Sharpe={m['strategy_sharpe']:+.3f}  CAGR={m['strategy_cagr']:+6.1%}  "
          f"MaxDD={m['strategy_max_drawdown']:7.1%}  time_in_mkt={m['time_in_market']:.1%}")


def main() -> None:
    with open("configs/timing.yaml") as fh:
        cfg = yaml.safe_load(fh)
    s, ls, lb = cfg["signal"], cfg["leverage_sizing"], cfg["leverage_backtest"]

    universe_data = _load_universe()

    signal_ctx = {"universe_data": universe_data}
    signal_ctx = TimingSignalAgent(
        ticker=s["ticker"], sma_window=s["sma_window"], mom_window=s.get("mom_window"),
        vol_window=s.get("vol_window"), vol_threshold=s.get("vol_threshold"),
        vix_threshold=s.get("vix_threshold"), combine=s["combine"],
    ).run(signal_ctx)
    signal = signal_ctx["timing_signal"]

    # Vol-sized layer
    pos_ctx = {"universe_data": universe_data, "timing_signal": signal}
    pos_ctx = LeveragedPositionAgent(
        signal_ticker=ls["signal_ticker"], vol_window=ls["vol_window"],
        target_vol=ls["target_vol"], max_leverage=ls["max_leverage"],
    ).run(pos_ctx)
    vol_sized_ctx = {"universe_data": universe_data, "leverage_position": pos_ctx["leverage_position"]}
    vol_sized_ctx = LeveragedBacktestAgent(
        tickers=tuple(lb["tickers"]), transaction_cost_bps=lb["transaction_cost_bps"],
        benchmark_ticker=lb["benchmark_ticker"],
    ).run(vol_sized_ctx)
    vol_sized_ctx["timing_signal"] = signal  # for time_in_market reporting

    # Naive binary overlay (for direct comparison)
    naive_ctx = {"universe_data": universe_data, "timing_signal": signal}
    naive_ctx = TimingBacktestAgent(ticker="TQQQ", transaction_cost_bps=2.0).run(naive_ctx)

    windows = [
        ("Full history (2010-2024)", "2010-02-11", _HOLDOUT_END),
        ("Dev period (2010-2019)", "2010-02-11", "2019-12-31"),
        ("Holdout (2020-2024)", "2020-01-01", _HOLDOUT_END),
    ]

    for label, start, end in windows:
        print("=" * 78)
        print(label)
        print("=" * 78)
        m_vol_sized = _metrics_for_window(vol_sized_ctx, start, end)
        m_naive = _metrics_for_window(naive_ctx, start, end)
        _print_row("vol-sized (QQQ/QLD/TQQQ)", m_vol_sized)
        _print_row("naive (TQQQ-or-cash)", m_naive)
        print(f"  {'TQQQ buy&hold':<28} Sharpe={m_naive['benchmark_sharpe']:+.3f}  "
              f"CAGR={m_naive['benchmark_cagr']:+6.1%}  MaxDD={m_naive['benchmark_max_drawdown']:7.1%}")
        print()

    avg_leverage = pos_ctx["leverage_position"].apply(
        lambda row: {"QQQ": 1.0, "QLD": 2.0, "TQQQ": 3.0}[row["ticker"]] * row["fraction"], axis=1,
    )
    print(f"Average realized leverage when position is non-zero: "
          f"{avg_leverage[avg_leverage > 0].mean():.2f}x")
    ticker_share = pos_ctx["leverage_position"].loc[pos_ctx["leverage_position"]["fraction"] > 0, "ticker"].value_counts(normalize=True)
    print("Ticker share of invested days:")
    print(ticker_share.to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
