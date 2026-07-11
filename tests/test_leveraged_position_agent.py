"""Tests for agents/timing/leveraged_position_agent.py — continuous
vol-targeted leverage sizing, mapped onto a QQQ(1x)/QLD(2x)/TQQQ(3x) +
cash blend (Moreira & Muir 2017 vol-scaling rationale, made deployable with
only the three real ETFs a retail account can actually hold).
"""
import numpy as np
import pandas as pd
import pytest

from agents.timing.leveraged_position_agent import LeveragedPositionAgent


def _make_context(closes, signal_values):
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    close = pd.Series(closes, index=dates, name="Close")
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000})
    signal = pd.Series(signal_values, index=dates)
    return {"universe_data": {"QQQ": df}, "timing_signal": signal}


def _low_vol_prices(n=300, daily_drift=0.0005):
    # Constant daily return -> zero realized volatility.
    return [100.0 * (1 + daily_drift) ** i for i in range(n)]


def _high_vol_prices(n=300, seed=0, daily_vol=0.06):
    rng = np.random.RandomState(seed)
    rets = rng.normal(0.0, daily_vol, n)
    prices = 100.0 * np.cumprod(1 + rets)
    return prices.tolist()


def test_flat_signal_gives_zero_fraction_regardless_of_vol():
    closes = _low_vol_prices()
    ctx = _make_context(closes, [0] * len(closes))
    agent = LeveragedPositionAgent(vol_window=20, target_vol=0.18, max_leverage=3.0)
    ctx = agent.run(ctx)

    position = ctx["leverage_position"]
    assert (position["fraction"] == 0.0).all()


def test_warmup_region_is_flat_cash():
    closes = _low_vol_prices()
    ctx = _make_context(closes, [1] * len(closes))
    agent = LeveragedPositionAgent(vol_window=20, target_vol=0.18, max_leverage=3.0)
    ctx = agent.run(ctx)

    position = ctx["leverage_position"]
    assert (position["fraction"].iloc[:19] == 0.0).all()


def test_zero_vol_long_regime_selects_top_tier_near_full_fraction():
    closes = _low_vol_prices(n=300)
    ctx = _make_context(closes, [1] * len(closes))
    agent = LeveragedPositionAgent(vol_window=20, target_vol=0.18, max_leverage=3.0)
    ctx = agent.run(ctx)

    position = ctx["leverage_position"]
    last = position.iloc[-1]
    assert last["ticker"] == "TQQQ"
    assert last["fraction"] == pytest.approx(1.0, abs=1e-6)


def test_high_vol_long_regime_reduces_to_qqq_partial_fraction():
    closes = _high_vol_prices(n=300, daily_vol=0.08)  # ~127% annualized realized vol
    ctx = _make_context(closes, [1] * len(closes))
    agent = LeveragedPositionAgent(vol_window=20, target_vol=0.18, max_leverage=3.0)
    ctx = agent.run(ctx)

    position = ctx["leverage_position"]
    last = position.iloc[-1]
    assert last["ticker"] == "QQQ"
    assert 0.0 < last["fraction"] < 1.0


def test_leverage_fraction_never_exceeds_one():
    closes = _high_vol_prices(n=300, daily_vol=0.005)  # very low vol -> large raw target
    ctx = _make_context(closes, [1] * len(closes))
    agent = LeveragedPositionAgent(vol_window=20, target_vol=0.18, max_leverage=3.0)
    ctx = agent.run(ctx)

    position = ctx["leverage_position"]
    assert (position["fraction"] <= 1.0 + 1e-9).all()
    assert (position["fraction"] >= 0.0).all()


def test_moderate_vol_maps_to_ceiling_tier_with_partial_fraction():
    # Construct a realized vol such that target_vol / vol == 2.5 exactly,
    # landing strictly between the QLD(2x) and TQQQ(3x) tiers. The agent
    # should pick the smallest tier >= target (TQQQ) scaled down to hit it
    # exactly, not the largest tier <= target (which could never reach 2.5x).
    target_vol = 0.20
    desired_leverage = 2.5
    implied_daily_vol = (target_vol / desired_leverage) / np.sqrt(252)

    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.RandomState(3)
    rets = rng.choice([implied_daily_vol, -implied_daily_vol], size=n)
    prices = 100.0 * np.cumprod(1 + rets)
    close = pd.Series(prices, index=dates)
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000})
    ctx = {"universe_data": {"QQQ": df}, "timing_signal": pd.Series(1, index=dates)}

    agent = LeveragedPositionAgent(vol_window=20, target_vol=target_vol, max_leverage=3.0)
    ctx = agent.run(ctx)
    last = ctx["leverage_position"].iloc[-1]

    assert last["ticker"] == "TQQQ"
    assert last["fraction"] == pytest.approx(desired_leverage / 3.0, rel=0.05)


def test_no_lookahead_position_at_t_unchanged_by_future_data():
    closes = _high_vol_prices(n=300, seed=7)
    ctx_full = _make_context(closes, [1] * len(closes))
    ctx_truncated = _make_context(closes[:250], [1] * 250)

    agent = LeveragedPositionAgent(vol_window=20, target_vol=0.18, max_leverage=3.0)
    pos_full = agent.run(ctx_full)["leverage_position"]
    pos_truncated = agent.run(ctx_truncated)["leverage_position"]

    pd.testing.assert_frame_equal(
        pos_full.iloc[:250].reset_index(drop=True),
        pos_truncated.reset_index(drop=True),
    )
