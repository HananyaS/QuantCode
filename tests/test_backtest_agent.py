"""Tests for BacktestAgent — P&L computation, equity curve, edge cases."""
import numpy as np
import pandas as pd
import pytest

from agents.backtest_agent import BacktestAgent


def _make_predictions(index: pd.DatetimeIndex, values) -> dict:
    return {"values": pd.Series(values, index=index, name="prediction")}


def _make_context(ohlcv: pd.DataFrame, pred_values) -> dict:
    pred_index = ohlcv.index[:-10]  # leave headroom for next-day return
    return {
        "data": ohlcv,
        "predictions": _make_predictions(pred_index, pred_values),
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_backtest_keys_present(sample_ohlcv):
    ctx = BacktestAgent().run(_make_context(sample_ohlcv, np.ones(len(sample_ohlcv) - 10)))
    assert "equity_curve" in ctx["backtest"]
    assert "returns" in ctx["backtest"]
    assert "n_trades" in ctx["backtest"]
    assert "win_rate" in ctx["backtest"]


def test_equity_curve_is_series(sample_ohlcv):
    ctx = BacktestAgent().run(_make_context(sample_ohlcv, np.ones(len(sample_ohlcv) - 10)))
    assert isinstance(ctx["backtest"]["equity_curve"], pd.Series)


def test_equity_starts_near_initial_capital(sample_ohlcv):
    """First equity value should be close to initial_capital × (1 + first_return - tc)."""
    capital = 50_000.0
    ctx = BacktestAgent(initial_capital=capital).run(
        _make_context(sample_ohlcv, np.ones(len(sample_ohlcv) - 10))
    )
    equity = ctx["backtest"]["equity_curve"]
    # Within 5% of initial capital on first step
    assert abs(equity.iloc[0] - capital) / capital < 0.05


def test_equity_is_positive(sample_ohlcv):
    ctx = BacktestAgent().run(_make_context(sample_ohlcv, np.ones(len(sample_ohlcv) - 10)))
    assert (ctx["backtest"]["equity_curve"] > 0).all()


def test_win_rate_in_range(sample_ohlcv):
    ctx = BacktestAgent().run(_make_context(sample_ohlcv, np.ones(len(sample_ohlcv) - 10)))
    wr = ctx["backtest"]["win_rate"]
    assert 0.0 <= wr <= 1.0


# ---------------------------------------------------------------------------
# All-flat strategy
# ---------------------------------------------------------------------------

def test_all_flat_zero_returns(sample_ohlcv):
    """All-zero predictions → strategy returns should be 0 (no positions taken)."""
    n = len(sample_ohlcv) - 10
    ctx = BacktestAgent(transaction_cost=0).run(_make_context(sample_ohlcv, np.zeros(n)))
    returns = ctx["backtest"]["returns"]
    assert (returns == 0).all(), "All-flat strategy should produce zero returns"


def test_all_flat_n_trades_zero(sample_ohlcv):
    n = len(sample_ohlcv) - 10
    ctx = BacktestAgent().run(_make_context(sample_ohlcv, np.zeros(n)))
    assert ctx["backtest"]["n_trades"] == 0


# ---------------------------------------------------------------------------
# All-long strategy
# ---------------------------------------------------------------------------

def test_all_long_tracks_market(sample_ohlcv):
    """All-long (no cost) strategy returns should equal market returns on aligned dates."""
    n = len(sample_ohlcv) - 10
    pred_index = sample_ohlcv.index[:n]
    ctx = BacktestAgent(transaction_cost=0).run({
        "data": sample_ohlcv,
        "predictions": _make_predictions(pred_index, np.ones(n)),
    })
    # Market return = sum of (close[t+1]/close[t] - 1) for dates in pred_index (excl last)
    close = sample_ohlcv["Close"]
    next_ret = close.pct_change().shift(-1).reindex(pred_index).dropna()
    strat_ret = ctx["backtest"]["returns"]
    # They must be equal (we exclude the last NaN row)
    pd.testing.assert_series_equal(strat_ret, next_ret, check_names=False)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_transaction_cost_reduces_return(sample_ohlcv):
    """Strategy with cost must earn less than strategy without cost."""
    n = len(sample_ohlcv) - 10
    ctx_no_cost = BacktestAgent(transaction_cost=0).run(
        _make_context(sample_ohlcv, np.ones(n))
    )
    ctx_with_cost = BacktestAgent(transaction_cost=0.01).run(
        _make_context(sample_ohlcv, np.ones(n))
    )
    final_no_cost = ctx_no_cost["backtest"]["equity_curve"].iloc[-1]
    final_with_cost = ctx_with_cost["backtest"]["equity_curve"].iloc[-1]
    assert final_with_cost < final_no_cost


def test_missing_data_raises():
    ctx = {"data": None, "predictions": {"values": pd.Series(dtype=float)}}
    with pytest.raises(AssertionError):
        BacktestAgent().run(ctx)


def test_missing_predictions_raises(sample_ohlcv):
    ctx = {"data": sample_ohlcv, "predictions": None}
    with pytest.raises(AssertionError):
        BacktestAgent().run(ctx)


def test_invalid_initial_capital():
    with pytest.raises(AssertionError):
        BacktestAgent(initial_capital=-1)


def test_invalid_transaction_cost():
    with pytest.raises(AssertionError):
        BacktestAgent(transaction_cost=1.5)
