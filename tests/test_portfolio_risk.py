"""Tests for PortfolioAgent's risk-management additions:
max-drawdown circuit breaker, volatility-targeted sizing, correlation exposure caps.
"""
import numpy as np
import pandas as pd
import pytest

from agents.portfolio_agent import PortfolioAgent


def _make_flat_universe(prices: dict, n_days: int) -> dict:
    """Build a universe from Dict[ticker, np.ndarray of length n_days] close prices."""
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    universe = {}
    for ticker, close in prices.items():
        close = np.asarray(close, dtype=float)
        universe[ticker] = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.001,
                "Low": close * 0.999,
                "Close": close,
                "Volume": 1_000_000.0,
            },
            index=dates,
        )
    return universe


def _scores(dates, per_date_scores: dict) -> pd.Series:
    """per_date_scores: Dict[ticker, List[float] of length len(dates)]."""
    tickers = list(per_date_scores.keys())
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    s = pd.Series(0.0, index=idx)
    for ticker, vals in per_date_scores.items():
        for d, v in zip(dates, vals):
            s.loc[(d, ticker)] = v
    return s


# ---------------------------------------------------------------------------
# Max-drawdown circuit breaker
# ---------------------------------------------------------------------------

def test_breaker_blocks_new_entry_after_drawdown():
    """Only 1 slot (max_positions=1): A holds it until B's score overtakes A's,
    which rank-exits A and opens the slot on the same day the crash also
    breaches the drawdown limit — so B's would-be entry is what gets blocked.
    """
    n_days = 10
    a_close = np.full(n_days, 100.0)
    a_close[3:] = 40.0  # -60% crash starting index 3
    b_close = np.full(n_days, 100.0)
    universe = _make_flat_universe({"A": a_close, "B": b_close}, n_days)
    dates = universe["A"].index

    a_scores = [5.0] * n_days
    b_scores = [0.0] * 3 + [10.0] * (n_days - 3)  # B overtakes A on the crash day
    scores = _scores(dates, {"A": a_scores, "B": b_scores})

    ctx = PortfolioAgent(
        max_positions=1, entry_rank=1, exit_rank=1, min_score=-100.0,
        trailing_stop_atr_mult=1000.0, atr_period=3,
        max_drawdown_limit=0.10,
    ).run({"cs_predictions": scores, "universe_data": universe})

    w = ctx["portfolio_weights"]
    assert w.loc[dates[3], "B"] == 0.0, "New entry must be blocked once drawdown limit is breached"


def test_without_breaker_entry_allowed_baseline():
    """Same setup, no max_drawdown_limit — confirms the setup would otherwise let B enter."""
    n_days = 10
    a_close = np.full(n_days, 100.0)
    a_close[3:] = 40.0
    b_close = np.full(n_days, 100.0)
    universe = _make_flat_universe({"A": a_close, "B": b_close}, n_days)
    dates = universe["A"].index

    a_scores = [5.0] * n_days
    b_scores = [0.0] * 3 + [10.0] * (n_days - 3)
    scores = _scores(dates, {"A": a_scores, "B": b_scores})

    ctx = PortfolioAgent(
        max_positions=1, entry_rank=1, exit_rank=1, min_score=-100.0,
        trailing_stop_atr_mult=1000.0, atr_period=3,
        max_drawdown_limit=None,
    ).run({"cs_predictions": scores, "universe_data": universe})

    w = ctx["portfolio_weights"]
    assert w.loc[dates[3], "B"] > 0.0


def test_de_risk_on_breach_force_exits_positions():
    """B's score never overtakes A's, so only the drawdown breach (not a rank
    exit) can explain A being force-exited."""
    n_days = 10
    a_close = np.full(n_days, 100.0)
    a_close[3:] = 70.0  # -30% crash, breaches 10% DD limit but not the (disabled) trailing stop
    b_close = np.full(n_days, 100.0)
    universe = _make_flat_universe({"A": a_close, "B": b_close}, n_days)
    dates = universe["A"].index

    scores = _scores(dates, {"A": [5.0] * n_days, "B": [0.0] * n_days})

    ctx = PortfolioAgent(
        # atr_period=1 makes ATR valid at entry (day0) — with atr_period=3
        # ATR would be NaN at entry and fall back to a fixed 5% stop, which
        # this crash would trip on its own, confounding the assertion.
        max_positions=1, entry_rank=1, exit_rank=2, min_score=-100.0,
        trailing_stop_atr_mult=1000.0, atr_period=1,
        max_drawdown_limit=0.10, de_risk_on_breach=True,
    ).run({"cs_predictions": scores, "universe_data": universe})

    w = ctx["portfolio_weights"]
    assert w.loc[dates[3], "A"] == 0.0, "de_risk_on_breach must force-exit existing positions"
    reasons = {t["action"] for t in ctx["portfolio_trades"]}
    assert "circuit_breaker_exit" in reasons


def test_breaker_requires_valid_fraction():
    with pytest.raises(AssertionError):
        PortfolioAgent(max_drawdown_limit=1.5)
    with pytest.raises(AssertionError):
        PortfolioAgent(max_drawdown_limit=0.0)


# ---------------------------------------------------------------------------
# Volatility-targeted position sizing
# ---------------------------------------------------------------------------

def test_vol_target_sizing_favors_lower_vol_asset():
    n_days = 80
    rng = np.random.default_rng(0)
    # A: low vol, B: high vol, both flat drift so weights differences come from vol only
    a_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=n_days)))
    b_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, size=n_days)))
    universe = _make_flat_universe({"A": a_close, "B": b_close}, n_days)
    dates = universe["A"].index[-10:]

    scores = _scores(dates, {"A": [5.0] * len(dates), "B": [5.0] * len(dates)})

    ctx = PortfolioAgent(
        max_positions=2, entry_rank=2, exit_rank=2, min_score=-100.0,
        trailing_stop_atr_mult=1000.0, atr_period=5,
        vol_target_sizing=True, vol_window=20,
    ).run({"cs_predictions": scores, "universe_data": universe})

    w = ctx["portfolio_weights"]
    last_day = w.index[-1]
    assert w.loc[last_day, "A"] > w.loc[last_day, "B"], (
        "Lower-volatility asset should receive a larger weight under vol targeting"
    )


def test_vol_target_weights_sum_to_one():
    n_days = 80
    rng = np.random.default_rng(1)
    a_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n_days)))
    b_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=n_days)))
    universe = _make_flat_universe({"A": a_close, "B": b_close}, n_days)
    dates = universe["A"].index[-10:]
    scores = _scores(dates, {"A": [5.0] * len(dates), "B": [5.0] * len(dates)})

    ctx = PortfolioAgent(
        max_positions=2, entry_rank=2, exit_rank=2, min_score=-100.0,
        trailing_stop_atr_mult=1000.0, atr_period=5,
        vol_target_sizing=True, vol_window=20,
    ).run({"cs_predictions": scores, "universe_data": universe})

    w = ctx["portfolio_weights"]
    held_days = w[(w > 0).sum(axis=1) > 0]
    np.testing.assert_allclose(held_days.sum(axis=1).values, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Correlation exposure caps
# ---------------------------------------------------------------------------

def test_correlation_cap_blocks_highly_correlated_entry():
    n_days = 80
    rng = np.random.default_rng(2)
    base_ret = rng.normal(0.0005, 0.01, size=n_days)
    a_close = 100.0 * np.exp(np.cumsum(base_ret))
    b_close = 100.0 * np.exp(np.cumsum(base_ret + rng.normal(0, 0.0001, size=n_days)))  # near-identical to A
    c_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n_days)))  # independent
    universe = _make_flat_universe({"A": a_close, "B": b_close, "C": c_close}, n_days)
    dates = universe["A"].index[-5:]

    # A already held (score high on all test dates); B and C both attractive candidates
    scores = _scores(dates, {
        "A": [10.0] * len(dates),
        "B": [9.0] * len(dates),
        "C": [8.0] * len(dates),
    })

    ctx = PortfolioAgent(
        max_positions=3, entry_rank=3, exit_rank=3, min_score=-100.0,
        trailing_stop_atr_mult=1000.0, atr_period=5,
        max_correlation=0.9, corr_window=60,
    ).run({"cs_predictions": scores, "universe_data": universe})

    w = ctx["portfolio_weights"]
    last_day = w.index[-1]
    assert w.loc[last_day, "A"] > 0, "A should be held"
    assert w.loc[last_day, "B"] == 0.0, "B is highly correlated with A and should be blocked"
    assert w.loc[last_day, "C"] > 0.0, "C is uncorrelated and should still be allowed in"


def test_no_correlation_cap_when_param_none():
    n_days = 80
    rng = np.random.default_rng(2)
    base_ret = rng.normal(0.0005, 0.01, size=n_days)
    a_close = 100.0 * np.exp(np.cumsum(base_ret))
    b_close = 100.0 * np.exp(np.cumsum(base_ret + rng.normal(0, 0.0001, size=n_days)))
    universe = _make_flat_universe({"A": a_close, "B": b_close}, n_days)
    dates = universe["A"].index[-5:]

    scores = _scores(dates, {"A": [10.0] * len(dates), "B": [9.0] * len(dates)})

    ctx = PortfolioAgent(
        max_positions=2, entry_rank=2, exit_rank=2, min_score=-100.0,
        trailing_stop_atr_mult=1000.0, atr_period=5,
        max_correlation=None,
    ).run({"cs_predictions": scores, "universe_data": universe})

    w = ctx["portfolio_weights"]
    last_day = w.index[-1]
    assert w.loc[last_day, "B"] > 0.0
