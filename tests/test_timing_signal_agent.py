"""Tests for agents/timing/timing_signal_agent.py — binary regime signal generation.

Convention matches agents/cs_feature_agent.py: rolling windows use
min_periods=window (no partial-window leakage) and every value at date t
uses only data through date t (causal).
"""
import numpy as np
import pandas as pd
import pytest

from agents.timing.timing_signal_agent import TimingSignalAgent


def _trending_price(n=400, start=100.0, daily_drift=0.001, seed=0) -> pd.Series:
    rng = np.random.RandomState(seed)
    rets = daily_drift + rng.normal(0, 0.01, n)
    prices = start * np.cumprod(1 + rets)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(prices, index=dates, name="Close")


def _make_context(price: pd.Series, vix: pd.Series = None) -> dict:
    df = pd.DataFrame({"Open": price, "High": price, "Low": price, "Close": price, "Volume": 1_000_000})
    universe_data = {"QQQ": df}
    if vix is not None:
        vix_df = pd.DataFrame({"Open": vix, "High": vix, "Low": vix, "Close": vix, "Volume": 0})
        universe_data["^VIX"] = vix_df
    return {"universe_data": universe_data}


def test_sma_only_signal_is_long_when_price_above_sma():
    price = _trending_price(n=400, daily_drift=0.002)  # strong uptrend
    ctx = _make_context(price)
    agent = TimingSignalAgent(ticker="QQQ", sma_window=200, combine="sma_only")
    ctx = agent.run(ctx)

    signal = ctx["timing_signal"]
    # Strong uptrend: by the end, price should be well above its 200d SMA -> long
    assert signal.iloc[-1] == 1


def test_sma_only_signal_is_flat_in_downtrend():
    price = _trending_price(n=400, daily_drift=-0.002)  # strong downtrend
    ctx = _make_context(price)
    agent = TimingSignalAgent(ticker="QQQ", sma_window=200, combine="sma_only")
    ctx = agent.run(ctx)

    signal = ctx["timing_signal"]
    assert signal.iloc[-1] == 0


def test_signal_has_nan_free_warmup_region_defaulted_to_flat():
    price = _trending_price(n=400)
    ctx = _make_context(price)
    agent = TimingSignalAgent(ticker="QQQ", sma_window=200, combine="sma_only")
    ctx = agent.run(ctx)

    signal = ctx["timing_signal"]
    assert signal.notna().all()
    # Before the SMA window is filled, there's no valid signal -> must default to flat (0), not long.
    assert (signal.iloc[:199] == 0).all()


def test_no_lookahead_signal_at_t_unchanged_by_future_data():
    price = _trending_price(n=400)
    ctx_full = _make_context(price)
    ctx_truncated = _make_context(price.iloc[:300])

    agent = TimingSignalAgent(ticker="QQQ", sma_window=200, combine="sma_only")
    sig_full = agent.run(ctx_full)["timing_signal"]
    sig_truncated = agent.run(ctx_truncated)["timing_signal"]

    pd.testing.assert_series_equal(
        sig_full.iloc[:300], sig_truncated, check_names=False,
    )


def test_all_agree_combine_requires_every_component_long():
    price = _trending_price(n=400, daily_drift=0.002)
    ctx = _make_context(price)
    agent = TimingSignalAgent(
        ticker="QQQ", sma_window=200, mom_window=126, combine="all_agree",
    )
    ctx = agent.run(ctx)
    components = ctx["timing_signal_components"]
    signal = ctx["timing_signal"]

    expected = (components.fillna(0).sum(axis=1) == components.shape[1]).astype(int)
    pd.testing.assert_series_equal(signal, expected, check_names=False)


def test_majority_combine():
    price = _trending_price(n=400, daily_drift=0.0005, seed=1)
    ctx = _make_context(price)
    agent = TimingSignalAgent(
        ticker="QQQ", sma_window=200, mom_window=126, vol_window=20,
        vol_threshold=0.40, combine="majority",
    )
    ctx = agent.run(ctx)
    components = ctx["timing_signal_components"]
    signal = ctx["timing_signal"]

    majority_needed = components.shape[1] / 2
    expected = (components.fillna(0).sum(axis=1) > majority_needed).astype(int)
    pd.testing.assert_series_equal(signal, expected, check_names=False)


def test_vix_component_requires_vix_in_universe_data():
    price = _trending_price(n=400)
    ctx = _make_context(price)  # no ^VIX
    agent = TimingSignalAgent(ticker="QQQ", sma_window=200, vix_threshold=30.0, combine="sma_only")
    with pytest.raises(AssertionError, match="\\^VIX"):
        agent.run(ctx)


def test_vix_component_goes_defensive_above_threshold():
    price = _trending_price(n=400, daily_drift=0.002)  # uptrend -> sma component long
    vix = pd.Series(15.0, index=price.index)
    vix.iloc[-30:] = 45.0  # VIX spike at the end
    ctx = _make_context(price, vix=vix)

    agent = TimingSignalAgent(
        ticker="QQQ", sma_window=200, vix_threshold=30.0, combine="all_agree",
    )
    ctx = agent.run(ctx)
    signal = ctx["timing_signal"]
    assert signal.iloc[-1] == 0  # VIX spike should veto the long signal under all_agree
