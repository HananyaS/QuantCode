"""Tests for PortfolioAgent — stateful with trailing stops and dual-rank exit."""
import numpy as np
import pandas as pd
import pytest

from agents.portfolio_agent import PortfolioAgent


def _make_universe(tickers=None, n_days=100, seed=42):
    if tickers is None:
        tickers = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(seed)
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


def _make_predictions(universe, n_test_days=50, seed=7):
    tickers = list(universe.keys())
    all_dates = next(iter(universe.values())).index
    test_dates = all_dates[-n_test_days:]
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([test_dates, tickers], names=["date", "ticker"])
    return pd.Series(rng.standard_normal(len(idx)), index=idx, name="score")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_context_keys_written():
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(max_positions=3, entry_rank=3, exit_rank=4).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    assert ctx["portfolio_weights"] is not None
    assert ctx["portfolio_trades"] is not None


def test_weights_shape():
    tickers = ["A", "B", "C", "D", "E"]
    universe = _make_universe(tickers=tickers)
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(max_positions=3, entry_rank=3, exit_rank=4).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    assert ctx["portfolio_weights"].shape[1] == len(tickers)


def test_no_negative_weights():
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(max_positions=3, entry_rank=3, exit_rank=4).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    assert (ctx["portfolio_weights"] >= 0).all().all()


def test_max_positions_respected():
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(max_positions=2, entry_rank=2, exit_rank=4).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    held_per_day = (ctx["portfolio_weights"] > 0).sum(axis=1)
    assert (held_per_day <= 2).all()


def test_equal_weight_across_held():
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(max_positions=3, entry_rank=3, exit_rank=4).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    w = ctx["portfolio_weights"]
    for _, row in w.iterrows():
        nonzero = row[row > 0]
        if len(nonzero) > 0:
            np.testing.assert_allclose(nonzero.values, nonzero.values[0], atol=1e-9)


# ---------------------------------------------------------------------------
# Min-score threshold: fully flat on bearish days
# ---------------------------------------------------------------------------

def test_min_score_prevents_entry():
    """When min_score is very high, no entries should occur."""
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(
        max_positions=3, entry_rank=3, exit_rank=4, min_score=999.0
    ).run({"cs_predictions": preds, "universe_data": universe})
    # No entries → all weights 0
    assert (ctx["portfolio_weights"].sum(axis=1) == 0).all()


# ---------------------------------------------------------------------------
# Dual rank threshold
# ---------------------------------------------------------------------------

def test_held_stock_survives_between_entry_and_exit_rank():
    """A stock ranked between entry_rank and exit_rank should NOT be exited."""
    tickers = ["A", "B", "C", "D", "E"]
    universe = _make_universe(tickers=tickers, n_days=80)
    dates = next(iter(universe.values())).index[-30:]

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    scores = pd.Series(0.0, index=idx, name="score")

    # Day 0: A is #1 → enters
    for t in tickers:
        scores.loc[(dates[0], t)] = 5.0 if t == "A" else 1.0

    # Day 1+: A drops to rank #3 (between entry_rank=2 and exit_rank=4) → should NOT exit
    for d in dates[1:]:
        scores.loc[(d, "A")] = 3.0  # rank ~3
        scores.loc[(d, "B")] = 5.0
        scores.loc[(d, "C")] = 4.0
        scores.loc[(d, "D")] = 2.0
        scores.loc[(d, "E")] = 1.0

    ctx = PortfolioAgent(
        max_positions=2, entry_rank=2, exit_rank=4,
        min_score=-100.0, trailing_stop_atr_mult=100.0,  # disable trailing stop
    ).run({"cs_predictions": scores, "universe_data": universe})

    # A should still be held on day 1+ (rank 3 < exit_rank 4)
    w = ctx["portfolio_weights"]
    assert w.loc[dates[5], "A"] > 0, "A should survive between entry and exit rank"


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------

def test_trailing_stop_triggers():
    """Trade log contains trailing_stop exits."""
    universe = _make_universe(n_days=100)
    preds = _make_predictions(universe, n_test_days=50)
    ctx = PortfolioAgent(
        max_positions=3, entry_rank=3, exit_rank=5,
        trailing_stop_atr_mult=0.01,  # very tight stop → triggers easily
    ).run({"cs_predictions": preds, "universe_data": universe})

    trail_exits = [t for t in ctx["portfolio_trades"] if t["action"] == "trailing_stop"]
    assert len(trail_exits) > 0, "Expected at least one trailing stop exit"


def test_trade_log_has_required_fields():
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(max_positions=3, entry_rank=3, exit_rank=4).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    for trade in ctx["portfolio_trades"]:
        assert "date" in trade
        assert "ticker" in trade
        assert "action" in trade
        assert trade["action"] in ("entry", "trailing_stop", "rank_exit")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_raises_if_predictions_missing():
    with pytest.raises(AssertionError):
        PortfolioAgent().run({"universe_data": {}})


def test_raises_if_universe_data_missing():
    with pytest.raises(AssertionError):
        PortfolioAgent().run({"cs_predictions": pd.Series(dtype=float)})
