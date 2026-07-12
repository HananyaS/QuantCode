"""kelly_backtest_run.py — end-to-end Kelly-criterion leverage backtest,
reporting standard metrics plus the Kelly-specific diagnostics (Sortino,
realized-vs-theoretical g(L), naive-vs-actual) across the full historical
window plus a pre-2020/2020-2024 regime breakdown, followed by a genuine
blind forward test on live-fetched 2025+ data.

Uses synthetic QLD/TQQQ (utils/synthetic_leveraged_series.py) built from
QQQ's full 1999+ real return history, NOT real cached TQQQ/QLD (inception-
limited to 2010/2006) — this is what lets the backtest cover the dot-com
crash. Never hits the network for the historical part; the blind-test part
does a single live fetch of the most recent ~2.5 years.
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
from agents.timing.kelly_evaluation_agent import KellyEvaluationAgent
from agents.timing.kelly_position_agent import KellyPositionAgent
from agents.timing.timing_backtest_agent import TimingBacktestAgent
from agents.timing.timing_signal_agent import TimingSignalAgent
from agents.universe_agent import UniverseAgent
from utils.data_cache import load_or_fetch
from utils.synthetic_leveraged_series import build_synthetic_universe

_CACHE_DIR = Path("data/cache")


def _no_network_fetch(missing, start, end):
    return {}


def _load_config() -> dict:
    with open("configs/kelly_timing.yaml") as fh:
        return yaml.safe_load(fh)


def _build_agent(cfg: dict) -> KellyPositionAgent:
    cv, rg, ks = cfg["conditional_vol"], cfg["regime"], cfg["kelly_sizing"]
    return KellyPositionAgent(
        signal_ticker=cfg["universe"]["underlying_ticker"],
        vol_method=cv["method"], vol_decay=cv["ewma_decay"],
        garch_refit_every=cv["garch_refit_every"], garch_min_train_obs=cv["garch_min_train_obs"],
        mu_decay=rg["mu_decay"], adx_period=rg["adx_period"], adx_threshold=rg["adx_threshold"],
        fractional_kelly=ks["fractional_kelly"], max_leverage=ks["max_leverage"],
        worst_case_daily_move=ks["worst_case_daily_move"], ruin_buffer=ks["ruin_buffer"],
        vol_spike_threshold=ks["vol_spike_threshold"], drawdown_limit=ks["drawdown_limit"],
        entry_margin=ks["entry_margin"], min_observations=ks["min_observations"],
    )


def _run_backtest(universe_data: dict, cfg: dict) -> dict:
    ctx = {"universe_data": universe_data}
    ctx = _build_agent(cfg).run(ctx)
    ctx = KellyBacktestAgent(
        tickers=("QQQ", "QLD", "TQQQ"), transaction_cost_bps=cfg["backtest"]["transaction_cost_bps"],
        benchmark_ticker="TQQQ",
    ).run(ctx)
    return ctx


def _metrics_for_window(ctx: dict, cfg: dict, start: str, end: str) -> dict:
    sl = slice(start, end)
    window_ctx = {
        "universe_data": ctx["universe_data"],
        "timing_returns": ctx["timing_returns"].loc[sl],
        "timing_equity": (1 + ctx["timing_returns"].loc[sl]).cumprod(),
        "benchmark_returns": ctx["benchmark_returns"].loc[sl],
        "benchmark_equity": (1 + ctx["benchmark_returns"].loc[sl]).cumprod(),
        "leverage_position": ctx["leverage_position"].loc[sl],
    }
    return KellyEvaluationAgent(
        periods_per_year=cfg["evaluation"]["periods_per_year"], underlying_ticker="QQQ",
    ).run(window_ctx)["kelly_metrics"]


def _print_metrics(label: str, m: dict) -> None:
    print(f"  {label}:")
    print(f"    strategy  Sharpe={m['strategy_sharpe']:+.3f}  Sortino={m['strategy_sortino']:+.3f}  "
          f"CAGR={m['strategy_cagr']:+.1%}  MaxDD={m['strategy_max_drawdown']:.1%}  "
          f"time_in_mkt={m['time_in_market']:.1%}  avg_lev={m['avg_leverage']:.2f}x")
    print(f"    buy&hold  Sharpe={m['benchmark_sharpe']:+.3f}  Sortino={m['benchmark_sortino']:+.3f}  "
          f"CAGR={m['benchmark_cagr']:+.1%}  MaxDD={m['benchmark_max_drawdown']:.1%}")
    print(f"    g(L): theoretical={m['theoretical_g']:.6f}/day  realized={m['realized_g']:.6f}/day  "
          f"gap={m['g_gap']:+.6f}/day")
    print(f"    naive_cum_return={m['naive_cum_return']:+.1%}  actual_cum_return={m['actual_cum_return']:+.1%}  "
          f"divergence={m['naive_vs_actual_divergence']:+.1%}")


def historical_backtest() -> None:
    cfg = _load_config()
    u = cfg["universe"]
    fetched = load_or_fetch(
        [u["underlying_ticker"]], u["start_date"], u["end_date"], _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    assert u["underlying_ticker"] in fetched, "QQQ cache missing — run UniverseAgent once first"
    qqq_close = fetched[u["underlying_ticker"]]["Close"]

    universe_data = build_synthetic_universe(qqq_close)
    ctx = _run_backtest(universe_data, cfg)

    windows = [
        ("Full history (1999-2024)", "1999-04-01", u["end_date"]),
        ("Dev-like period (2000-2019)", "2000-01-01", "2019-12-31"),
        ("Recent period (2020-2024)", "2020-01-01", u["end_date"]),
        ("Dot-com stress (2000-2002)", "2000-01-01", "2002-12-31"),
        ("GFC stress (2007-2009)", "2007-10-01", "2009-03-31"),
    ]
    for label, start, end in windows:
        print("=" * 78)
        print(label)
        print("=" * 78)
        m = _metrics_for_window(ctx, cfg, start, end)
        _print_metrics("Kelly-sized", m)
        print()


def blind_forward_test() -> None:
    print("=" * 78)
    print("Genuine blind forward test (real 2025+ data, frozen config)")
    print("=" * 78)
    cfg = _load_config()
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
    ctx = _run_backtest(universe_data, cfg)

    span = qqq_close.index
    forward_start = "2025-01-01"
    if span.max() < pd.Timestamp(forward_start) + pd.Timedelta(days=60):
        print("  SKIPPED — not enough post-cutoff data yet")
        return
    forward_end = str(span.max().date())
    m = _metrics_for_window(ctx, cfg, forward_start, forward_end)
    _print_metrics(f"Blind {forward_start} -> {forward_end}", m)
    print("\n  Caveat: short (~1.5yr), low-powered window — directional signal only.")


if __name__ == "__main__":
    try:
        historical_backtest()
        blind_forward_test()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
