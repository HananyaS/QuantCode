"""Tests for agents/timing/leveraged_backtest_agent.py — backtests a
per-date (ticker, fraction) position blend across QQQ/QLD/TQQQ + cash,
with turnover-based transaction costs and a 1-day execution lag.
"""
import pandas as pd
import pytest

from agents.timing.leveraged_backtest_agent import LeveragedBacktestAgent


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


def test_full_fraction_single_ticker_matches_buy_and_hold():
    # Constant TQQQ,1.0 position throughout -> since there's no transition,
    # the 1-day execution lag is invisible and strategy returns exactly
    # match TQQQ's own buy-and-hold returns (mirrors
    # TimingBacktestAgent's test_always_long_matches_buy_and_hold_returns).
    tqqq = [100, 100, 200, 200, 200]
    qqq = [100] * 5
    qld = [100] * 5
    positions = {"ticker": ["TQQQ"] * 5, "fraction": [1.0] * 5}
    ctx = _make_context(qqq, qld, tqqq, positions)

    agent = LeveragedBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="TQQQ")
    ctx = agent.run(ctx)

    expected = pd.Series(tqqq, index=ctx["timing_returns"].index).pct_change().fillna(0.0)
    pd.testing.assert_series_equal(ctx["timing_returns"], expected, check_names=False)


def test_position_change_is_lagged_no_lookahead():
    # Position switches to TQQQ on index 2 itself; the 100% jump also
    # happens ON index 2 -> execution lag means it must NOT be captured
    # (the switch only takes effect starting index 3).
    tqqq = [100, 100, 200, 200, 200]
    qqq = [100] * 5
    qld = [100] * 5
    positions = {"ticker": ["QQQ", "QQQ", "TQQQ", "TQQQ", "TQQQ"],
                 "fraction": [0.0, 0.0, 1.0, 1.0, 1.0]}
    ctx = _make_context(qqq, qld, tqqq, positions)

    agent = LeveragedBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="TQQQ")
    ctx = agent.run(ctx)

    assert ctx["timing_returns"].iloc[2] == 0.0


def test_zero_fraction_gives_zero_returns():
    qqq = [100, 105, 95, 110, 90]
    positions = {"ticker": ["QQQ"] * 5, "fraction": [0.0] * 5}
    ctx = _make_context(qqq, qqq, qqq, positions)

    agent = LeveragedBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="QQQ")
    ctx = agent.run(ctx)

    assert (ctx["timing_returns"] == 0.0).all()


def test_transaction_cost_on_ticker_switch_charges_full_turnover():
    qqq = [100] * 5
    qld = [100] * 5
    tqqq = [100] * 5
    # Switch from QQQ,1.0 -> TQQQ,1.0 at index 2: full round-trip turnover = 2.0
    positions = {"ticker": ["QQQ", "QQQ", "TQQQ", "TQQQ", "TQQQ"], "fraction": [1.0] * 5}
    ctx = _make_context(qqq, qld, tqqq, positions)

    agent = LeveragedBacktestAgent(transaction_cost_bps=10.0, benchmark_ticker="TQQQ")  # 10bps = 0.001
    ctx = agent.run(ctx)

    strat_ret = ctx["timing_returns"]
    # Prices are flat, so all return drag comes from transaction costs.
    total_cost = -strat_ret[strat_ret < 0].sum()
    # One turnover event of magnitude 2.0 (sell 1.0 QQQ, buy 1.0 TQQQ) plus
    # the initial entry into QQQ (magnitude 1.0) = 3.0 total turnover units.
    assert total_cost == pytest.approx(3.0 * 0.001, abs=1e-9)


def test_fraction_change_same_ticker_charges_partial_turnover():
    qqq = [100] * 5
    positions = {"ticker": ["TQQQ"] * 5, "fraction": [0.5, 0.5, 0.8, 0.8, 0.8]}
    ctx = _make_context(qqq, qqq, qqq, positions)

    agent = LeveragedBacktestAgent(transaction_cost_bps=10.0, benchmark_ticker="TQQQ")
    ctx = agent.run(ctx)

    strat_ret = ctx["timing_returns"]
    total_cost = -strat_ret[strat_ret < 0].sum()
    # Entry to 0.5 (turnover 0.5) + bump to 0.8 (turnover 0.3) = 0.8 total.
    assert total_cost == pytest.approx(0.8 * 0.001, abs=1e-9)


def test_benchmark_equity_uses_specified_benchmark_ticker():
    qqq = [100, 110, 121]
    tqqq = [100, 130, 169]
    positions = {"ticker": ["QQQ"] * 3, "fraction": [1.0] * 3}
    ctx = _make_context(qqq, qqq, tqqq, positions)

    agent = LeveragedBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="TQQQ")
    ctx = agent.run(ctx)

    expected_bh = pd.Series(tqqq, index=ctx["benchmark_returns"].index).pct_change().fillna(0.0)
    pd.testing.assert_series_equal(ctx["benchmark_returns"], expected_bh, check_names=False)


def test_equity_curves_start_at_one():
    qqq = [100, 105, 110]
    positions = {"ticker": ["QQQ"] * 3, "fraction": [1.0] * 3}
    ctx = _make_context(qqq, qqq, qqq, positions)

    agent = LeveragedBacktestAgent(transaction_cost_bps=0.0, benchmark_ticker="QQQ")
    ctx = agent.run(ctx)

    assert ctx["timing_equity"].iloc[0] == pytest.approx(1.0)
    assert ctx["benchmark_equity"].iloc[0] == pytest.approx(1.0)
