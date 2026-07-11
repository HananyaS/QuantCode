"""Tests for agents/timing/timing_backtest_agent.py — single-instrument
time-series backtest (not cross-sectional; no PortfolioAgent involved).
"""
import numpy as np
import pandas as pd
import pytest

from agents.timing.timing_backtest_agent import TimingBacktestAgent


def _make_context(closes, signal_values):
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    close = pd.Series(closes, index=dates, name="Close")
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000})
    signal = pd.Series(signal_values, index=dates)
    return {"universe_data": {"QQQ": df}, "timing_signal": signal}


def test_signal_is_shifted_no_lookahead():
    # Price doubles on day index 2 (from 100 -> 200). A signal that only
    # turns on AT that same date must NOT capture that day's return.
    closes = [100, 100, 200, 200, 200]
    signal = [0, 0, 1, 1, 1]
    ctx = _make_context(closes, signal)

    agent = TimingBacktestAgent(ticker="QQQ", transaction_cost_bps=0.0)
    ctx = agent.run(ctx)

    strat_ret = ctx["timing_returns"]
    # Return on day index 2 (100% jump) must NOT be captured since signal
    # only just turned on that day (no lookahead -> uses prior day's signal).
    assert strat_ret.iloc[2] == 0.0
    # Day index 3 onward, signal was already 1 -> should track price moves.
    assert strat_ret.iloc[3] == pytest.approx(0.0)  # flat price day 2->3


def test_flat_signal_produces_zero_returns():
    closes = [100, 105, 95, 110, 90]
    signal = [0, 0, 0, 0, 0]
    ctx = _make_context(closes, signal)

    agent = TimingBacktestAgent(ticker="QQQ", transaction_cost_bps=0.0)
    ctx = agent.run(ctx)

    assert (ctx["timing_returns"] == 0.0).all()


def test_always_long_matches_buy_and_hold_returns():
    closes = [100, 105, 95, 110, 90]
    signal = [1, 1, 1, 1, 1]
    ctx = _make_context(closes, signal)

    agent = TimingBacktestAgent(ticker="QQQ", transaction_cost_bps=0.0)
    ctx = agent.run(ctx)

    close = pd.Series(closes, index=ctx["timing_returns"].index)
    expected_bh = close.pct_change().fillna(0.0)
    pd.testing.assert_series_equal(
        ctx["timing_returns"], expected_bh, check_names=False,
    )


def test_transaction_cost_applied_on_signal_change():
    closes = [100, 100, 100, 100, 100]
    signal = [0, 1, 1, 0, 0]  # two transitions: enter at idx1, exit at idx3
    ctx = _make_context(closes, signal)

    agent = TimingBacktestAgent(ticker="QQQ", transaction_cost_bps=10.0)  # 10 bps = 0.001
    ctx = agent.run(ctx)

    strat_ret = ctx["timing_returns"]
    # Cost is charged on the bar where the position actually changes
    # (i.e. where the shifted/applied signal differs from its prior value).
    n_transitions = (pd.Series(signal).diff().fillna(0) != 0).sum()
    assert n_transitions == 2
    total_cost_drag = -strat_ret[strat_ret < 0].sum()
    assert total_cost_drag == pytest.approx(2 * 0.001, abs=1e-9)


def test_equity_curves_start_at_one():
    closes = [100, 105, 110]
    signal = [1, 1, 1]
    ctx = _make_context(closes, signal)

    agent = TimingBacktestAgent(ticker="QQQ", transaction_cost_bps=0.0)
    ctx = agent.run(ctx)

    assert ctx["timing_equity"].iloc[0] == pytest.approx(1.0)
    assert ctx["benchmark_equity"].iloc[0] == pytest.approx(1.0)
