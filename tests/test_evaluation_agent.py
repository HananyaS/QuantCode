"""Tests for EvaluationAgent — metrics computation, benchmark comparison, edge cases."""
import numpy as np
import pandas as pd
import pytest

from agents.evaluation_agent import EvaluationAgent


def _make_backtest(returns: pd.Series, initial: float = 100_000.0) -> dict:
    equity = (1 + returns).cumprod() * initial
    n_trades = int((returns != 0).sum())
    active = returns[returns != 0]
    win_rate = float((active > 0).sum() / len(active)) if len(active) else 0.0
    return {
        "returns": returns,
        "equity_curve": equity,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "initial_capital": initial,
    }


def _make_context(returns: pd.Series, predictions: dict = None, benchmark=None) -> dict:
    return {
        "backtest": _make_backtest(returns),
        "predictions": predictions,
        "benchmark_data": benchmark,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_metrics_keys_present(sample_ohlcv):
    dates = pd.bdate_range("2021-01-01", periods=100)
    returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 100), index=dates)
    ctx = EvaluationAgent().run(_make_context(returns))
    for key in ("sharpe", "max_drawdown", "annualized_return", "win_rate", "n_trades", "final_equity"):
        assert key in ctx["metrics"]


def test_sharpe_is_float(sample_ohlcv):
    dates = pd.bdate_range("2021-01-01", periods=100)
    returns = pd.Series(np.random.default_rng(2).normal(0.001, 0.01, 100), index=dates)
    ctx = EvaluationAgent().run(_make_context(returns))
    assert isinstance(ctx["metrics"]["sharpe"], float)


def test_max_drawdown_non_positive():
    dates = pd.bdate_range("2021-01-01", periods=200)
    returns = pd.Series(np.random.default_rng(3).normal(0, 0.01, 200), index=dates)
    ctx = EvaluationAgent().run(_make_context(returns))
    assert ctx["metrics"]["max_drawdown"] <= 0.0


def test_positive_returns_positive_sharpe():
    dates = pd.bdate_range("2021-01-01", periods=252)
    returns = pd.Series(np.full(252, 0.001), index=dates)  # constant positive return
    ctx = EvaluationAgent().run(_make_context(returns))
    assert ctx["metrics"]["sharpe"] > 0


def test_negative_returns_negative_sharpe():
    dates = pd.bdate_range("2021-01-01", periods=252)
    returns = pd.Series(np.full(252, -0.001), index=dates)
    ctx = EvaluationAgent().run(_make_context(returns))
    assert ctx["metrics"]["sharpe"] < 0


# ---------------------------------------------------------------------------
# Sharpe edge cases
# ---------------------------------------------------------------------------

def test_zero_std_returns_zero_sharpe():
    """Constant returns have zero std → Sharpe should be 0, not NaN or error."""
    dates = pd.bdate_range("2021-01-01", periods=50)
    returns = pd.Series(np.zeros(50), index=dates)
    ctx = EvaluationAgent().run(_make_context(returns))
    assert ctx["metrics"]["sharpe"] == 0.0


def test_single_return_zero_sharpe():
    dates = pd.bdate_range("2021-01-01", periods=1)
    returns = pd.Series([0.01], index=dates)
    ctx = EvaluationAgent().run(_make_context(returns))
    assert ctx["metrics"]["sharpe"] == 0.0


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

def test_benchmark_metrics_added_when_present(sample_ohlcv):
    dates = sample_ohlcv.index[:100]
    returns = pd.Series(np.random.default_rng(4).normal(0, 0.01, 100), index=dates)
    ctx = EvaluationAgent().run(_make_context(returns, benchmark=sample_ohlcv))
    assert "benchmark" in ctx["metrics"]
    assert "sharpe" in ctx["metrics"]["benchmark"]
    assert "max_drawdown" in ctx["metrics"]["benchmark"]


def test_benchmark_not_added_when_absent():
    dates = pd.bdate_range("2021-01-01", periods=50)
    returns = pd.Series(np.zeros(50), index=dates)
    ctx = EvaluationAgent().run(_make_context(returns))
    assert "benchmark" not in ctx["metrics"]


# ---------------------------------------------------------------------------
# Prediction accuracy forwarded
# ---------------------------------------------------------------------------

def test_accuracy_forwarded_from_predictions():
    dates = pd.bdate_range("2021-01-01", periods=50)
    returns = pd.Series(np.zeros(50), index=dates)
    preds = {
        "values": pd.Series(np.zeros(50), index=dates),
        "train_accuracy": 0.7,
        "test_accuracy": 0.6,
        "feature_importances": {},
    }
    ctx = EvaluationAgent().run(_make_context(returns, predictions=preds))
    assert ctx["metrics"]["train_accuracy"] == pytest.approx(0.7)
    assert ctx["metrics"]["test_accuracy"] == pytest.approx(0.6)
