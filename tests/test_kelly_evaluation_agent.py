"""Tests for agents/timing/kelly_evaluation_agent.py — standard metrics
plus Sortino, realized-vs-theoretical g(L) decomposition, and naive-vs-
actual (L x underlying) comparison.
"""
import numpy as np
import pandas as pd
import pytest

from agents.timing.kelly_evaluation_agent import KellyEvaluationAgent


def _make_context(underlying_closes, strat_rets, positions):
    n = len(underlying_closes)
    dates = pd.bdate_range("2020-01-01", periods=n)
    underlying = pd.Series(underlying_closes, index=dates, name="Close")
    underlying_df = pd.DataFrame({
        "Open": underlying, "High": underlying, "Low": underlying, "Close": underlying, "Volume": 1_000_000,
    })
    strat = pd.Series(strat_rets, index=dates)
    position = pd.DataFrame(positions, index=dates)
    return {
        "universe_data": {"QQQ": underlying_df},
        "timing_returns": strat,
        "timing_equity": (1 + strat).cumprod(),
        "benchmark_returns": underlying.pct_change().fillna(0.0),
        "benchmark_equity": (1 + underlying.pct_change().fillna(0.0)).cumprod(),
        "leverage_position": position,
    }


def test_output_has_expected_keys():
    n = 60
    underlying = [100.0 * (1.001**i) for i in range(n)]
    strat_rets = [0.001] * n
    positions = {"ticker": ["TQQQ"] * n, "fraction": [1.0] * n}
    ctx = _make_context(underlying, strat_rets, positions)
    metrics = KellyEvaluationAgent(periods_per_year=252, underlying_ticker="QQQ").run(ctx)["kelly_metrics"]

    for key in (
        "strategy_sharpe", "strategy_sortino", "strategy_cagr", "strategy_max_drawdown",
        "benchmark_sharpe", "benchmark_sortino", "benchmark_cagr", "benchmark_max_drawdown",
        "time_in_market", "avg_leverage", "realized_underlying_mu", "realized_underlying_sigma_sq",
        "theoretical_g", "realized_g", "g_gap", "naive_cum_return", "actual_cum_return",
        "naive_vs_actual_divergence",
    ):
        assert key in metrics


def test_sortino_zero_downside_gives_zero_not_inf_or_nan():
    n = 40
    underlying = [100.0] * n
    strat_rets = [0.001] * n  # always positive -> zero downside deviation
    positions = {"ticker": ["QQQ"] * n, "fraction": [1.0] * n}
    ctx = _make_context(underlying, strat_rets, positions)
    metrics = KellyEvaluationAgent(periods_per_year=252, underlying_ticker="QQQ").run(ctx)["kelly_metrics"]
    assert metrics["strategy_sortino"] == 0.0


def test_sortino_only_penalizes_downside_not_upside_volatility():
    n = 100
    underlying = [100.0] * n
    rng = np.random.RandomState(0)
    # All positive returns but with varying magnitude (upside vol only).
    strat_rets = np.abs(rng.normal(0.002, 0.01, n)).tolist()
    positions = {"ticker": ["QQQ"] * n, "fraction": [1.0] * n}
    ctx = _make_context(underlying, strat_rets, positions)
    metrics = KellyEvaluationAgent(periods_per_year=252, underlying_ticker="QQQ").run(ctx)["kelly_metrics"]
    assert metrics["strategy_sortino"] == 0.0  # no downside periods at all


def test_avg_leverage_reflects_position_blend():
    n = 20
    underlying = [100.0 * (1.0005**i) for i in range(n)]
    strat_rets = [0.001] * n
    positions = {"ticker": ["TQQQ"] * n, "fraction": [0.5] * n}  # 1.5x average
    ctx = _make_context(underlying, strat_rets, positions)
    metrics = KellyEvaluationAgent(periods_per_year=252, underlying_ticker="QQQ").run(ctx)["kelly_metrics"]
    assert metrics["avg_leverage"] == pytest.approx(1.5)


def test_naive_vs_actual_diverges_in_choppy_round_trip():
    # Underlying round-trips (+10% then back to start) -> naive 3x*0cumret=0,
    # but actual compounded 3x path decays below zero net -- classic
    # leveraged-decay demonstration, this time surfaced through the
    # evaluation agent's own reporting rather than a standalone script.
    underlying = [100, 110, 100]
    strat_rets = [0.0, 0.30, -100 / 110 * 3]  # exact 3x daily compounding of underlying's own moves
    positions = {"ticker": ["TQQQ"] * 3, "fraction": [1.0] * 3}
    ctx = _make_context(underlying, strat_rets, positions)
    metrics = KellyEvaluationAgent(periods_per_year=252, underlying_ticker="QQQ").run(ctx)["kelly_metrics"]

    assert metrics["naive_cum_return"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["actual_cum_return"] < metrics["naive_cum_return"] - 0.01


def test_theoretical_g_uses_kelly_sizing_formula():
    from utils.kelly_sizing import expected_log_growth

    n = 100
    rng = np.random.RandomState(2)
    rets = rng.normal(0.0006, 0.01, n)
    underlying = (100 * np.cumprod(1 + rets)).tolist()
    strat_rets = [0.001] * n
    positions = {"ticker": ["QLD"] * n, "fraction": [1.0] * n}  # constant 2x
    ctx = _make_context(underlying, strat_rets, positions)
    metrics = KellyEvaluationAgent(periods_per_year=252, underlying_ticker="QQQ").run(ctx)["kelly_metrics"]

    expected = expected_log_growth(
        metrics["avg_leverage"], metrics["realized_underlying_mu"], metrics["realized_underlying_sigma_sq"],
    )
    assert metrics["theoretical_g"] == pytest.approx(expected)
