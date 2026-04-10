"""Tests for PortfolioAgent — ATR-% stops, low-based trigger, z-score normalisation,
score-weighted allocation."""
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
    for t in tickers:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n_days)))
        spread = np.abs(rng.normal(0, 0.005, size=n_days))
        universe[t] = pd.DataFrame(
            {
                "Open":   close,
                "High":   close * (1 + spread),
                "Low":    close * (1 - spread),
                "Close":  close,
                "Volume": 1_000_000.0,
            },
            index=dates,
        )
    return universe


def _make_predictions(universe, n_test_days=50, seed=7):
    tickers   = list(universe.keys())
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


def test_equal_weight_when_score_weighting_off():
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(max_positions=3, entry_rank=3, exit_rank=4,
                         score_weighting=False).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    w = ctx["portfolio_weights"]
    for _, row in w.iterrows():
        nonzero = row[row > 0]
        if len(nonzero) > 0:
            np.testing.assert_allclose(nonzero.values, nonzero.values[0], atol=1e-9)


# ---------------------------------------------------------------------------
# ATR-% trailing stop: stop distance is percentage-based
# ---------------------------------------------------------------------------

def test_stop_level_is_pct_below_entry():
    """Initial stop_level should be entry_price * (1 - atr_mult * atr_pct)."""
    universe = _make_universe(tickers=["A"], n_days=60)
    dates = next(iter(universe.values())).index[-10:]
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"])
    # Strong signal every day so A always enters
    preds = pd.Series(5.0, index=idx, name="score")

    ctx = PortfolioAgent(max_positions=1, entry_rank=1, exit_rank=2,
                         trailing_stop_atr_mult=1.0, atr_period=5).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    entry_trade = next(t for t in ctx["portfolio_trades"] if t["action"] == "entry")
    entry_price = entry_trade["price"]
    stop_level  = entry_trade["stop_level"]
    # Stop must be strictly below entry price and within reasonable % range
    assert stop_level < entry_price
    pct_below = (entry_price - stop_level) / entry_price
    assert 0 < pct_below < 0.5, f"Stop {pct_below:.2%} below entry looks wrong"


# ---------------------------------------------------------------------------
# Low-based stop check
# ---------------------------------------------------------------------------

def test_trailing_stop_uses_low_not_close():
    """Force Low below stop but keep Close above stop — stop should still trigger."""
    tickers = ["A", "B", "C"]
    universe = _make_universe(tickers=tickers, n_days=80)
    dates = next(iter(universe.values())).index[-20:]

    # Patch ticker A: close stays flat, but Low is very low on day 5
    # so that if we only checked close the stop would NOT trigger.
    df_a = universe["A"].copy().loc[dates]
    entry_close = float(df_a["Close"].iloc[0])
    # Make close always = entry price (never moves → HWM = entry = stop stays low)
    df_a["Close"] = entry_close
    df_a["High"]  = entry_close * 1.001
    # On day 5 push Low far below where the stop would be set
    df_a.loc[dates[5], "Low"] = entry_close * 0.50  # -50% intraday low
    df_a["Open"] = entry_close
    universe["A"] = universe["A"].copy()
    universe["A"].loc[dates] = df_a

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    scores = pd.Series(0.0, index=idx, name="score")
    for d in dates:
        scores.loc[(d, "A")] = 10.0   # A always top-ranked
        scores.loc[(d, "B")] = 1.0
        scores.loc[(d, "C")] = 0.5

    ctx = PortfolioAgent(
        max_positions=1, entry_rank=1, exit_rank=2,
        min_score=-100.0, trailing_stop_atr_mult=0.1, atr_period=5,
    ).run({"cs_predictions": scores, "universe_data": universe})

    stop_exits = [t for t in ctx["portfolio_trades"] if t["action"] == "trailing_stop"]
    assert len(stop_exits) > 0, "Expected trailing stop triggered by intraday Low"


# ---------------------------------------------------------------------------
# Z-score normalisation
# ---------------------------------------------------------------------------

def test_min_score_applied_to_zscore():
    """With min_score=999 (impossible z-score) no entries should occur."""
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(
        max_positions=3, entry_rank=3, exit_rank=4, min_score=999.0
    ).run({"cs_predictions": preds, "universe_data": universe})
    assert (ctx["portfolio_weights"].sum(axis=1) == 0).all()


def test_trade_log_has_score_z_field():
    """All trade records should carry score_z, not raw score."""
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(max_positions=3, entry_rank=3, exit_rank=4).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    for trade in ctx["portfolio_trades"]:
        assert "score_z" in trade, f"Missing score_z in trade: {trade}"


# ---------------------------------------------------------------------------
# Score-weighted allocation
# ---------------------------------------------------------------------------

def test_score_weighted_sums_to_one():
    """Weights must still sum to 1 when score_weighting is enabled."""
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(
        max_positions=3, entry_rank=3, exit_rank=4,
        score_weighting=True, weighting_temperature=0.5,
    ).run({"cs_predictions": preds, "universe_data": universe})
    w = ctx["portfolio_weights"]
    row_sums = w.sum(axis=1)
    # Days with positions held must sum to ~1.0
    held_days = row_sums[row_sums > 0]
    np.testing.assert_allclose(held_days.values, 1.0, atol=1e-9)


def test_score_weighted_not_uniform():
    """With score_weighting=True and low temperature, weights should NOT be equal."""
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(
        max_positions=3, entry_rank=3, exit_rank=4,
        score_weighting=True, weighting_temperature=0.1,  # very concentrated
    ).run({"cs_predictions": preds, "universe_data": universe})
    w = ctx["portfolio_weights"]
    # Find at least one day with >1 position and non-uniform weights
    non_uniform_found = False
    for _, row in w.iterrows():
        nonzero = row[row > 0]
        if len(nonzero) > 1:
            if nonzero.max() - nonzero.min() > 1e-6:
                non_uniform_found = True
                break
    assert non_uniform_found, "Expected non-uniform weights with score_weighting=True"


def test_high_temperature_approaches_uniform():
    """Very high temperature should produce near-uniform weights."""
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(
        max_positions=3, entry_rank=3, exit_rank=4,
        score_weighting=True, weighting_temperature=1000.0,
    ).run({"cs_predictions": preds, "universe_data": universe})
    w = ctx["portfolio_weights"]
    for _, row in w.iterrows():
        nonzero = row[row > 0]
        if len(nonzero) > 1:
            np.testing.assert_allclose(nonzero.values, nonzero.values[0], atol=1e-3)


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

    for t in tickers:
        scores.loc[(dates[0], t)] = 5.0 if t == "A" else 1.0

    for d in dates[1:]:
        scores.loc[(d, "A")] = 3.0
        scores.loc[(d, "B")] = 5.0
        scores.loc[(d, "C")] = 4.0
        scores.loc[(d, "D")] = 2.0
        scores.loc[(d, "E")] = 1.0

    ctx = PortfolioAgent(
        max_positions=2, entry_rank=2, exit_rank=4,
        min_score=-100.0, trailing_stop_atr_mult=100.0,
    ).run({"cs_predictions": scores, "universe_data": universe})

    w = ctx["portfolio_weights"]
    assert w.loc[dates[5], "A"] > 0, "A should survive between entry and exit rank"


# ---------------------------------------------------------------------------
# Trade log fields
# ---------------------------------------------------------------------------

def test_trade_log_has_required_fields():
    universe = _make_universe()
    preds = _make_predictions(universe)
    ctx = PortfolioAgent(max_positions=3, entry_rank=3, exit_rank=4).run(
        {"cs_predictions": preds, "universe_data": universe}
    )
    for trade in ctx["portfolio_trades"]:
        assert "date"    in trade
        assert "ticker"  in trade
        assert "action"  in trade
        assert "score_z" in trade
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
