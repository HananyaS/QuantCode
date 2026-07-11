"""Tests for MultiAssetBacktestAgent's slippage-aware cost model.

The flat `transaction_cost` bps assumption doesn't reflect that trading a
large fraction of an illiquid name's daily dollar volume should cost more
than trading a small fraction of a liquid name's. `slippage_coef` adds a
per-asset, per-day cost component scaled by sqrt(participation rate) =
sqrt(notional traded / trailing avg dollar volume), on top of the flat cost.
"""
import numpy as np
import pandas as pd
import pytest

from agents.multi_backtest_agent import MultiAssetBacktestAgent


def _make_universe(n_days: int = 100, tickers=None, volume: float = 1_000_000.0) -> dict:
    if tickers is None:
        tickers = ["A", "B", "C"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    universe = {}
    for t in tickers:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n_days)))
        spread = np.abs(rng.normal(0, 0.005, size=n_days))
        universe[t] = pd.DataFrame(
            {
                "Open": close,
                "High": close * (1 + spread),
                "Low": close * (1 - spread),
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )
    return universe


def _make_weights(universe: dict, top_k: int = 2, n_dates: int = 60) -> pd.DataFrame:
    tickers = list(universe.keys())
    dates = next(iter(universe.values())).index[-n_dates:]
    k = min(top_k, len(tickers))
    w = pd.DataFrame(0.0, index=dates, columns=tickers)
    for d in dates:
        w.loc[d, tickers[:k]] = 1.0 / k
    return w


# ---------------------------------------------------------------------------
# Backward compatibility: slippage_coef=0 must match the old flat-cost formula
# ---------------------------------------------------------------------------

def test_default_slippage_coef_matches_flat_cost_baseline():
    universe = _make_universe(n_days=100)
    weights = _make_weights(universe, n_dates=60)

    ctx = MultiAssetBacktestAgent(transaction_cost=0.01, slippage_coef=0.0).run(
        {"universe_data": universe, "portfolio_weights": weights}
    )
    bt = ctx["multi_backtest"]

    # Manually recompute the old flat-cost formula: turnover * transaction_cost
    close_df = pd.DataFrame({t: d["Close"] for t, d in universe.items()})[weights.columns]
    next_ret = close_df.pct_change().shift(-1)
    aligned_w = weights.reindex(next_ret.dropna(how="all").index).dropna(how="all")
    next_ret_aligned = next_ret.reindex(aligned_w.index).fillna(0.0)
    gross = (aligned_w * next_ret_aligned).sum(axis=1)
    wd = aligned_w.diff()
    wd.iloc[0] = aligned_w.iloc[0]
    turnover = wd.abs().sum(axis=1)
    expected_net = gross - turnover * 0.01

    pd.testing.assert_series_equal(bt["returns"], expected_net, check_names=False)


# ---------------------------------------------------------------------------
# Slippage increases cost for low-volume (high participation) names
# ---------------------------------------------------------------------------

def test_low_volume_name_costs_more_than_high_volume_name():
    tickers = ["LOWVOL", "HIGHVOL"]
    low_vol_universe = _make_universe(n_days=100, tickers=["X"], volume=1_000.0)
    high_vol_universe = _make_universe(n_days=100, tickers=["X"], volume=1_000_000_000.0)

    weights = _make_weights(low_vol_universe, top_k=1, n_dates=60)  # single ticker X, full weight

    ctx_low = MultiAssetBacktestAgent(
        transaction_cost=0.0, slippage_coef=0.1, initial_capital=1_000_000.0
    ).run({"universe_data": low_vol_universe, "portfolio_weights": weights})
    ctx_high = MultiAssetBacktestAgent(
        transaction_cost=0.0, slippage_coef=0.1, initial_capital=1_000_000.0
    ).run({"universe_data": high_vol_universe, "portfolio_weights": weights})

    assert (
        ctx_low["multi_backtest"]["equity_curve"].iloc[-1]
        < ctx_high["multi_backtest"]["equity_curve"].iloc[-1]
    ), "Trading a large fraction of a thin name's volume should cost more"


def test_slippage_coef_zero_volume_no_error():
    """Zero-volume days must not raise (division-by-zero guard)."""
    universe = _make_universe(n_days=30, tickers=["A"], volume=0.0)
    weights = _make_weights(universe, top_k=1, n_dates=20)
    ctx = MultiAssetBacktestAgent(slippage_coef=0.1).run(
        {"universe_data": universe, "portfolio_weights": weights}
    )
    assert np.isfinite(ctx["multi_backtest"]["equity_curve"]).all()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_slippage_costs_reported_in_context():
    universe = _make_universe(n_days=100)
    weights = _make_weights(universe, n_dates=60)
    ctx = MultiAssetBacktestAgent(slippage_coef=0.05).run(
        {"universe_data": universe, "portfolio_weights": weights}
    )
    assert "slippage_costs" in ctx["multi_backtest"]
    assert (ctx["multi_backtest"]["slippage_costs"] >= 0).all()


def test_raises_on_negative_slippage_coef():
    with pytest.raises(AssertionError):
        MultiAssetBacktestAgent(slippage_coef=-0.1)
