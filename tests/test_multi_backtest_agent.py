"""Tests for MultiAssetBacktestAgent."""
import numpy as np
import pandas as pd
import pytest

from agents.multi_backtest_agent import MultiAssetBacktestAgent


def _make_universe(n_days: int = 100, tickers=None) -> dict:
    if tickers is None:
        tickers = ["A", "B", "C"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    universe = {}
    for i, t in enumerate(tickers):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n_days)))
        spread = np.abs(rng.normal(0, 0.005, size=n_days))
        universe[t] = pd.DataFrame(
            {
                "Open": close,
                "High": close * (1 + spread),
                "Low": close * (1 - spread),
                "Close": close,
                "Volume": 1_000_000.0,
            },
            index=dates,
        )
    return universe


def _make_weights(universe: dict, top_k: int = 2, n_dates: int = 60) -> pd.DataFrame:
    """Equal-weight top-k portfolio over last n_dates of universe."""
    tickers = list(universe.keys())
    dates = next(iter(universe.values())).index[-n_dates:]
    k = min(top_k, len(tickers))
    w = pd.DataFrame(0.0, index=dates, columns=tickers)
    for d in dates:
        w.loc[d, tickers[:k]] = 1.0 / k
    return w


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_context_key_written():
    universe = _make_universe()
    weights = _make_weights(universe)
    ctx = MultiAssetBacktestAgent().run({"universe_data": universe, "portfolio_weights": weights})
    assert ctx["multi_backtest"] is not None


def test_backtest_keys():
    universe = _make_universe()
    weights = _make_weights(universe)
    ctx = MultiAssetBacktestAgent().run({"universe_data": universe, "portfolio_weights": weights})
    bt = ctx["multi_backtest"]
    for key in ("equity_curve", "returns", "turnover", "n_rebalances", "initial_capital"):
        assert key in bt, f"Missing key: {key}"


def test_equity_starts_near_initial_capital():
    IC = 50_000.0
    universe = _make_universe()
    weights = _make_weights(universe)
    ctx = MultiAssetBacktestAgent(initial_capital=IC).run(
        {"universe_data": universe, "portfolio_weights": weights}
    )
    equity = ctx["multi_backtest"]["equity_curve"]
    # First equity value should be close to IC (small first-day return ± cost)
    assert abs(equity.iloc[0] - IC) / IC < 0.05


def test_equity_always_positive():
    universe = _make_universe()
    weights = _make_weights(universe)
    ctx = MultiAssetBacktestAgent().run({"universe_data": universe, "portfolio_weights": weights})
    assert (ctx["multi_backtest"]["equity_curve"] > 0).all()


def test_returns_length_matches_equity():
    universe = _make_universe()
    weights = _make_weights(universe)
    ctx = MultiAssetBacktestAgent().run({"universe_data": universe, "portfolio_weights": weights})
    bt = ctx["multi_backtest"]
    assert len(bt["returns"]) == len(bt["equity_curve"])


# ---------------------------------------------------------------------------
# Transaction cost
# ---------------------------------------------------------------------------

def test_zero_tc_gives_higher_equity():
    universe = _make_universe(n_days=100)
    weights = _make_weights(universe, n_dates=60)
    ctx_no_tc = MultiAssetBacktestAgent(transaction_cost=0.0).run(
        {"universe_data": universe, "portfolio_weights": weights}
    )
    ctx_with_tc = MultiAssetBacktestAgent(transaction_cost=0.01).run(
        {"universe_data": universe, "portfolio_weights": weights}
    )
    assert (
        ctx_no_tc["multi_backtest"]["equity_curve"].iloc[-1]
        >= ctx_with_tc["multi_backtest"]["equity_curve"].iloc[-1]
    )


def test_turnover_is_nonnegative():
    universe = _make_universe()
    weights = _make_weights(universe)
    ctx = MultiAssetBacktestAgent().run({"universe_data": universe, "portfolio_weights": weights})
    assert (ctx["multi_backtest"]["turnover"] >= 0).all()


# ---------------------------------------------------------------------------
# Flat portfolio → zero return (approximately)
# ---------------------------------------------------------------------------

def test_fully_flat_portfolio_zero_return():
    """If all weights are 0, net returns should be 0 (only small tc on first entry)."""
    universe = _make_universe(n_days=80)
    dates = next(iter(universe.values())).index[-40:]
    tickers = list(universe.keys())
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    ctx = MultiAssetBacktestAgent(transaction_cost=0.0).run(
        {"universe_data": universe, "portfolio_weights": weights}
    )
    # With zero weights and zero tc, all returns must be 0
    assert (ctx["multi_backtest"]["returns"] == 0.0).all()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_raises_if_universe_data_missing():
    with pytest.raises(AssertionError, match="universe_data"):
        MultiAssetBacktestAgent().run({"portfolio_weights": pd.DataFrame()})


def test_raises_if_portfolio_weights_missing():
    universe = _make_universe()
    with pytest.raises(AssertionError, match="portfolio_weights"):
        MultiAssetBacktestAgent().run({"universe_data": universe})
