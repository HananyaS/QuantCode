"""Tests for agents/timing/kelly_backtest_agent.py.

Critical distinction from LeveragedBacktestAgent: KellyPositionAgent's
output position is ALREADY a forecast-for-t (its mu_hat/sigma_sq_hat inputs
are pre-shifted at the source — see kelly_position_agent.py's module
docstring). KellyBacktestAgent must therefore apply the position for date t
against date t's OWN return directly, with NO additional shift — applying
LeveragedBacktestAgent's shift(1) on top would double-lag and desynchronize
from what KellyPositionAgent actually decided.
"""
import pandas as pd
import pytest

from agents.timing.kelly_backtest_agent import KellyBacktestAgent


def _price_series(closes, dates):
    close = pd.Series(closes, index=dates, name="Close")
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000})


def _make_context(qqq, qld, tqqq, positions):
    n = len(qqq)
    dates = pd.bdate_range("2020-01-01", periods=n)
    universe_data = {
        "QQQ": _price_series(qqq, dates),
        "QLD": _price_series(qld, dates),
        "TQQQ": _price_series(tqqq, dates),
    }
    position = pd.DataFrame(positions, index=dates)
    return {"universe_data": universe_data, "leverage_position": position}


def test_same_day_position_applies_to_same_day_return_no_extra_lag():
    # Position at index 2 is already "decided using info through index 1"
    # (KellyPositionAgent's own forecast-shift convention) -- so it must
    # capture index 2's OWN return directly, unlike LeveragedBacktestAgent
    # which would need an extra shift to avoid lookahead on a
    # same-day-decided position.
    tqqq = [100, 100, 200, 200, 200]
    qqq = [100] * 5
    qld = [100] * 5
    positions = {"ticker": ["QQQ", "QQQ", "TQQQ", "TQQQ", "TQQQ"], "fraction": [0.0, 0.0, 1.0, 1.0, 1.0]}
    ctx = _make_context(qqq, qld, tqqq, positions)

    agent = KellyBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="TQQQ")
    ctx = agent.run(ctx)

    # index 2's 100% jump IS captured since the position for index 2 was
    # already forecast-decided using data through index 1.
    assert ctx["timing_returns"].iloc[2] == pytest.approx(1.0)


def test_zero_fraction_gives_zero_returns():
    qqq = [100, 105, 95, 110, 90]
    positions = {"ticker": ["QQQ"] * 5, "fraction": [0.0] * 5}
    ctx = _make_context(qqq, qqq, qqq, positions)

    agent = KellyBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="QQQ")
    ctx = agent.run(ctx)
    assert (ctx["timing_returns"] == 0.0).all()


def test_full_fraction_single_ticker_matches_buy_and_hold():
    tqqq = [100, 105, 95, 110, 90]
    qqq = [100] * 5
    qld = [100] * 5
    positions = {"ticker": ["TQQQ"] * 5, "fraction": [1.0] * 5}
    ctx = _make_context(qqq, qld, tqqq, positions)

    agent = KellyBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="TQQQ")
    ctx = agent.run(ctx)

    expected = pd.Series(tqqq, index=ctx["timing_returns"].index).pct_change().fillna(0.0)
    pd.testing.assert_series_equal(ctx["timing_returns"], expected, check_names=False)


def test_transaction_cost_on_ticker_switch():
    qqq = [100] * 5
    qld = [100] * 5
    tqqq = [100] * 5
    positions = {"ticker": ["QQQ", "QQQ", "TQQQ", "TQQQ", "TQQQ"], "fraction": [1.0] * 5}
    ctx = _make_context(qqq, qld, tqqq, positions)

    agent = KellyBacktestAgent(transaction_cost_bps=10.0, benchmark_ticker="TQQQ")  # 10bps=0.001
    ctx = agent.run(ctx)

    strat_ret = ctx["timing_returns"]
    total_cost = -strat_ret[strat_ret < 0].sum()
    # entry into QQQ at index0 (turnover 1.0) + switch QQQ->TQQQ at index2 (turnover 2.0) = 3.0
    assert total_cost == pytest.approx(3.0 * 0.001, abs=1e-9)


def test_equity_curves_start_at_one():
    qqq = [100, 105, 110]
    positions = {"ticker": ["QQQ"] * 3, "fraction": [1.0] * 3}
    ctx = _make_context(qqq, qqq, qqq, positions)

    agent = KellyBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="QQQ")
    ctx = agent.run(ctx)
    assert ctx["timing_equity"].iloc[0] == pytest.approx(1.0)
    assert ctx["benchmark_equity"].iloc[0] == pytest.approx(1.0)


def test_benchmark_uses_specified_ticker():
    qqq = [100, 110, 121]
    tqqq = [100, 130, 169]
    positions = {"ticker": ["QQQ"] * 3, "fraction": [1.0] * 3}
    ctx = _make_context(qqq, qqq, tqqq, positions)

    agent = KellyBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="TQQQ")
    ctx = agent.run(ctx)
    expected_bh = pd.Series(tqqq, index=ctx["benchmark_returns"].index).pct_change().fillna(0.0)
    pd.testing.assert_series_equal(ctx["benchmark_returns"], expected_bh, check_names=False)
