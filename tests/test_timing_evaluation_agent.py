"""Tests for agents/timing/timing_evaluation_agent.py — Sharpe/CAGR/MaxDD/
time-in-market metrics for a single-instrument timing strategy vs benchmark.
"""
import numpy as np
import pandas as pd
import pytest

from agents.timing.timing_evaluation_agent import TimingEvaluationAgent


def _make_context(strat_rets, bh_rets, signal_values):
    dates = pd.bdate_range("2020-01-01", periods=len(strat_rets))
    strat = pd.Series(strat_rets, index=dates)
    bh = pd.Series(bh_rets, index=dates)
    signal = pd.Series(signal_values, index=dates)
    return {
        "timing_returns": strat,
        "timing_equity": (1 + strat).cumprod(),
        "benchmark_returns": bh,
        "benchmark_equity": (1 + bh).cumprod(),
        "timing_signal": signal,
    }


def test_zero_volatility_zero_return_gives_zero_sharpe_not_nan():
    n = 30
    ctx = _make_context([0.0] * n, [0.0] * n, [0] * n)
    metrics = TimingEvaluationAgent(periods_per_year=252).run(ctx)["timing_metrics"]
    assert metrics["strategy_sharpe"] == 0.0
    assert not np.isnan(metrics["strategy_sharpe"])


def test_max_drawdown_is_negative_and_correct_for_known_path():
    # Equity path: 1.0 -> 1.2 -> 0.6 -> 0.9  => max DD = 0.6/1.2 - 1 = -0.5
    rets = [0.0, 0.2, -0.5, 0.5]
    ctx = _make_context(rets, rets, [1, 1, 1, 1])
    metrics = TimingEvaluationAgent(periods_per_year=252).run(ctx)["timing_metrics"]
    assert metrics["strategy_max_drawdown"] == pytest.approx(-0.5, abs=1e-6)


def test_time_in_market_matches_signal_mean():
    signal = [1, 1, 0, 0, 1]
    rets = [0.01, 0.01, 0.0, 0.0, 0.01]
    ctx = _make_context(rets, rets, signal)
    metrics = TimingEvaluationAgent(periods_per_year=252).run(ctx)["timing_metrics"]
    assert metrics["time_in_market"] == pytest.approx(3 / 5)


def test_metrics_reported_for_both_strategy_and_benchmark():
    n = 50
    rng = np.random.RandomState(0)
    strat = rng.normal(0.001, 0.01, n).tolist()
    bh = rng.normal(0.0008, 0.015, n).tolist()
    ctx = _make_context(strat, bh, [1] * n)
    metrics = TimingEvaluationAgent(periods_per_year=252).run(ctx)["timing_metrics"]

    for key in ("strategy_sharpe", "strategy_cagr", "strategy_max_drawdown",
                "benchmark_sharpe", "benchmark_cagr", "benchmark_max_drawdown",
                "time_in_market"):
        assert key in metrics
