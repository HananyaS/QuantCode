"""leveraged_overlay_check.py — naive leveraged-ETF overlay check.

Applies the QQQ/VIX-derived timing signal (the winning config from
scripts/measure_timing_strategy.py's autoresearch run: sma_window=190,
vol_window=20, vol_threshold=0.30, vix_threshold=20, combine=majority) to
TQQQ (3x) and QLD (2x) instead of QQQ itself — i.e. "when the signal says
long, hold the leveraged ETF; when flat, hold cash."

This is deliberately the NAIVE version: constant leveraged exposure whenever
the signal is long, no vol-aware position sizing between QQQ/QLD/TQQQ. The
POC plan (docs/poc-plans/breakout-and-qqq-timing.md, Idea 2 Phase 4)
explicitly flags that decay must be handled deliberately, not assumed
away — this script measures what happens if you DON'T do that yet, as the
baseline the more careful version should beat.

Leveraged ETFs reset to constant leverage daily; decay ~ L(L-1)/2 * sigma^2
per period (Cheng & Madhavan 2009). Since the vol-regime and VIX-regime
components already push the signal flat exactly when realized/implied vol
is elevated, the timing signal is — even naively applied — dodging some of
the periods where decay is worst. Whether that's enough to make constant 3x
exposure viable is the empirical question here.
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
from utils.data_cache import load_or_fetch

_CACHE_DIR = Path("data/cache")
_WINNING_CONFIG = dict(
    sma_window=190, mom_window=None, vol_window=20,
    vol_threshold=0.30, vix_threshold=20.0, combine="majority",
)
_LEVERAGED_TICKERS = ["TQQQ", "QLD"]


def _no_network_fetch(missing, start, end):
    return {}


def _load_universe() -> dict:
    """Load QQQ/^VIX (long history) and the leveraged ETFs (shorter,
    inception-limited history) WITHOUT inner-join alignment — each ticker
    keeps its own native date range so QQQ's signal can be computed over
    its full history while only being backtested against TQQQ/QLD over
    the dates those instruments actually existed."""
    signal_tickers = ["QQQ", "^VIX"]
    fetched = load_or_fetch(
        signal_tickers, "1999-04-01", "2024-12-27", _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    # Leveraged ETFs post-date QQQ/VIX's start and each other (QLD inception
    # 2006-06-21, TQQQ inception 2010-02-11) — fetched individually with
    # each one's own start so _covers() doesn't reject them for lacking
    # pre-inception history.
    fetched.update(load_or_fetch(["QLD"], "2006-06-21", "2024-12-27", _no_network_fetch, cache_dir=_CACHE_DIR))
    fetched.update(load_or_fetch(["TQQQ"], "2010-02-11", "2024-12-27", _no_network_fetch, cache_dir=_CACHE_DIR))
    missing = set(signal_tickers + _LEVERAGED_TICKERS) - set(fetched)
    assert not missing, f"Missing cached coverage for {missing}"
    return fetched


def _print_metrics(label: str, m: dict) -> None:
    print(f"  {label}:")
    print(f"    strategy  Sharpe={m['strategy_sharpe']:+.3f}  CAGR={m['strategy_cagr']:+.1%}  "
          f"MaxDD={m['strategy_max_drawdown']:.1%}  time_in_mkt={m['time_in_market']:.1%}")
    print(f"    buy&hold  Sharpe={m['benchmark_sharpe']:+.3f}  CAGR={m['benchmark_cagr']:+.1%}  "
          f"MaxDD={m['benchmark_max_drawdown']:.1%}")


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


def main() -> None:
    universe_data = _load_universe()

    # Signal generated once from QQQ/VIX over their full available history —
    # causal, so it's valid to apply to any sub-window a leveraged ETF covers.
    signal_ctx = {"universe_data": universe_data}
    signal_ctx = TimingSignalAgent(ticker="QQQ", **_WINNING_CONFIG).run(signal_ctx)
    signal = signal_ctx["timing_signal"]

    for lev_ticker in _LEVERAGED_TICKERS:
        print("=" * 78)
        print(f"{lev_ticker} — QQQ-derived signal applied naively (constant exposure when long)")
        print("=" * 78)

        ctx = {"universe_data": universe_data, "timing_signal": signal}
        ctx = TimingBacktestAgent(ticker=lev_ticker, transaction_cost_bps=2.0).run(ctx)

        lev_dates = universe_data[lev_ticker].index
        full_start, full_end = str(lev_dates.min().date()), str(lev_dates.max().date())

        m_full = _metrics_for_window(ctx, full_start, full_end)
        print(f"\n  Full available history ({full_start} -> {full_end}):")
        _print_metrics(lev_ticker, m_full)

        m_holdout = _metrics_for_window(ctx, "2020-01-01", "2024-12-27")
        print(f"\n  2020-2024 window (COVID crash + 2022 bear):")
        _print_metrics(lev_ticker, m_holdout)
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
