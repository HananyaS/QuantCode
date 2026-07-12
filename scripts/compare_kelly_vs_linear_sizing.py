"""compare_kelly_vs_linear_sizing.py — head-to-head comparison of the two
leverage-sizing approaches built this session, on IDENTICAL price data:

1. Kelly-criterion sizing (agents/timing/kelly_position_agent.py), tuned
   via autoresearch (configs/kelly_timing.yaml) and leakage-checked
   (scripts/kelly_hysteresis_holdout_check.py).
2. Linear vol-sized layer (agents/timing/leveraged_position_agent.py, the
   QQQ regime signal from agents/timing/timing_signal_agent.py), tuned via
   autoresearch (configs/timing.yaml) and leakage-checked
   (scripts/leveraged_sizing_holdout_check.py).

Both previously validated on DIFFERENT price series: the linear layer used
real cached TQQQ/QLD (inception-limited to 2010/2006), the Kelly module
used synthetic QLD/TQQQ built from QQQ's full 1999+ history. This script
runs BOTH on the SAME synthetic universe (utils/synthetic_leveraged_series.py)
so the comparison isn't confounded by which price series each was built
against — and, as a side effect, tests the linear layer against the
dot-com crash it was never exposed to before.
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
from agents.timing.leveraged_backtest_agent import LeveragedBacktestAgent
from agents.timing.leveraged_position_agent import LeveragedPositionAgent
from agents.timing.timing_signal_agent import TimingSignalAgent
from agents.universe_agent import UniverseAgent
from utils.data_cache import load_or_fetch
from utils.synthetic_leveraged_series import build_synthetic_universe

_CACHE_DIR = Path("data/cache")
_HOLDOUT_END = "2024-12-27"


def _no_network_fetch(missing, start, end):
    return {}


def _load_kelly_config() -> dict:
    with open("configs/kelly_timing.yaml") as fh:
        return yaml.safe_load(fh)


def _load_linear_config() -> dict:
    with open("configs/timing.yaml") as fh:
        return yaml.safe_load(fh)


def _run_kelly(universe_data: dict, cfg: dict) -> dict:
    cv, rg, ks = cfg["conditional_vol"], cfg["regime"], cfg["kelly_sizing"]
    ctx = {"universe_data": universe_data}
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
    return ctx


def _run_linear(universe_data: dict, cfg: dict) -> dict:
    # universe_data must include real '^VIX' alongside the synthetic
    # QQQ/QLD/TQQQ -- the actual tuned linear config's winning combine mode
    # (majority, vix_threshold=20) depends on it; swapping in a degraded
    # VIX-free variant wouldn't be a fair test of what was actually tuned.
    assert "^VIX" in universe_data, "compare_kelly_vs_linear_sizing: '^VIX' required for the linear signal"
    s, ls, lb = cfg["signal"], cfg["leverage_sizing"], cfg["leverage_backtest"]
    signal_ctx = {"universe_data": universe_data}
    signal_ctx = TimingSignalAgent(
        ticker=s["ticker"], sma_window=s["sma_window"], mom_window=s.get("mom_window"),
        vol_window=s.get("vol_window"), vol_threshold=s.get("vol_threshold"),
        vix_threshold=s.get("vix_threshold"), combine=s["combine"],
    ).run(signal_ctx)
    signal = signal_ctx["timing_signal"]

    pos_ctx = {"universe_data": universe_data, "timing_signal": signal}
    pos_ctx = LeveragedPositionAgent(
        signal_ticker=ls["signal_ticker"], vol_window=ls["vol_window"],
        target_vol=ls["target_vol"], max_leverage=ls["max_leverage"],
    ).run(pos_ctx)
    ctx = {"universe_data": universe_data, "leverage_position": pos_ctx["leverage_position"]}
    ctx = LeveragedBacktestAgent(
        tickers=tuple(lb["tickers"]), transaction_cost_bps=lb["transaction_cost_bps"],
        benchmark_ticker=lb["benchmark_ticker"],
    ).run(ctx)
    ctx["timing_signal"] = signal
    return ctx


def _kelly_metrics(ctx: dict, cfg: dict, start: str, end: str) -> dict:
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


def _print_row(label: str, m: dict) -> None:
    print(f"  {label:<20} Sharpe={m['strategy_sharpe']:+.3f}  Sortino={m['strategy_sortino']:+.3f}  "
          f"CAGR={m['strategy_cagr']:+7.1%}  MaxDD={m['strategy_max_drawdown']:7.1%}  "
          f"avg_lev={m['avg_leverage']:.2f}x")


def main() -> None:
    kelly_cfg = _load_kelly_config()
    linear_cfg = _load_linear_config()

    fetched = load_or_fetch(["QQQ", "^VIX"], "1999-04-01", _HOLDOUT_END, _no_network_fetch, cache_dir=_CACHE_DIR)
    assert "QQQ" in fetched and "^VIX" in fetched, "QQQ/^VIX cache missing"
    universe_data = build_synthetic_universe(fetched["QQQ"]["Close"])
    universe_data["^VIX"] = fetched["^VIX"]

    kelly_ctx = _run_kelly(universe_data, kelly_cfg)
    linear_ctx = _run_linear(universe_data, linear_cfg)

    windows = [
        ("Full history (1999-2024)", "1999-04-01", _HOLDOUT_END),
        ("Dev-like (2000-2019)", "2000-01-01", "2019-12-31"),
        ("Recent (2020-2024)", "2020-01-01", _HOLDOUT_END),
        ("Dot-com stress (2000-2002)", "2000-01-01", "2002-12-31"),
        ("GFC stress (2007-2009)", "2007-10-01", "2009-03-31"),
    ]
    for label, start, end in windows:
        print("=" * 78)
        print(label)
        print("=" * 78)
        m_kelly = _kelly_metrics(kelly_ctx, kelly_cfg, start, end)
        m_linear = _kelly_metrics(linear_ctx, kelly_cfg, start, end)
        _print_row("Kelly-tuned", m_kelly)
        _print_row("Linear vol-sized", m_linear)
        print(f"  {'TQQQ buy&hold':<20} Sharpe={m_kelly['benchmark_sharpe']:+.3f}  "
              f"CAGR={m_kelly['benchmark_cagr']:+7.1%}  MaxDD={m_kelly['benchmark_max_drawdown']:7.1%}")
        print()

    print("=" * 78)
    print("Genuine blind forward test (real 2025+ data, both configs frozen)")
    print("=" * 78)
    try:
        agent = UniverseAgent(
            tickers=["QQQ", "^VIX"], start_date="2022-01-01", end_date="2026-07-11",
            benchmark="QQQ", min_history_days=30, min_assets=1, data_source="yfinance",
        )
        ctx_raw = agent.run({})
        qqq_close = ctx_raw["universe_data"]["QQQ"]["Close"]
        blind_universe = build_synthetic_universe(qqq_close)
        blind_universe["^VIX"] = ctx_raw["universe_data"]["^VIX"]

        kelly_blind = _run_kelly(blind_universe, kelly_cfg)
        linear_blind = _run_linear(blind_universe, linear_cfg)

        forward_start, forward_end = "2025-01-01", str(qqq_close.index.max().date())
        m_kelly_blind = _kelly_metrics(kelly_blind, kelly_cfg, forward_start, forward_end)
        m_linear_blind = _kelly_metrics(linear_blind, kelly_cfg, forward_start, forward_end)
        print(f"  Window: {forward_start} -> {forward_end}")
        _print_row("Kelly-tuned", m_kelly_blind)
        _print_row("Linear vol-sized", m_linear_blind)
        print(f"  {'TQQQ buy&hold':<20} Sharpe={m_kelly_blind['benchmark_sharpe']:+.3f}  "
              f"CAGR={m_kelly_blind['benchmark_cagr']:+7.1%}  MaxDD={m_kelly_blind['benchmark_max_drawdown']:7.1%}")
    except Exception as exc:
        print(f"  SKIPPED — live fetch failed: {exc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
