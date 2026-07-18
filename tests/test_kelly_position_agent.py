"""Tests for agents/timing/kelly_position_agent.py — Kelly-criterion
leverage sizing with entry/exit hysteresis, orchestrating
utils/conditional_vol.py + utils/regime_classifier.py + utils/kelly_sizing.py
into a per-date (ticker, fraction) position (same schema as
LeveragedPositionAgent).
"""
import numpy as np
import pandas as pd
import pytest

from agents.timing.kelly_position_agent import KellyPositionAgent


def _trending_ohlc(n=400, daily_drift=0.0025, vol=0.006, seed=0, start="2015-01-01"):
    rng = np.random.RandomState(seed)
    rets = daily_drift + rng.normal(0, vol, n)
    close = pd.Series(100.0 * np.cumprod(1 + rets), index=pd.bdate_range(start, periods=n))
    high, low = close * 1.004, close * 0.996
    return high, low, close


def _choppy_ohlc(n=400, seed=1, start="2015-01-01"):
    rng = np.random.RandomState(seed)
    rets = rng.normal(0, 0.012, n)
    rets = rets - pd.Series(rets).rolling(5, min_periods=1).mean().values * 0.85
    close = pd.Series(100.0 * np.cumprod(1 + rets), index=pd.bdate_range(start, periods=n))
    high, low = close * 1.01, close * 0.99
    return high, low, close


def _universe(high, low, close):
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": 1_000_000})
    qld = close  # not used by KellyPositionAgent's own math (only for downstream backtest), flat OK here
    tqqq = close
    return {
        "QQQ": df,
        "QLD": pd.DataFrame({"Open": qld, "High": qld, "Low": qld, "Close": qld, "Volume": 1_000_000}),
        "TQQQ": pd.DataFrame({"Open": tqqq, "High": tqqq, "Low": tqqq, "Close": tqqq, "Volume": 1_000_000}),
    }


def _agent(**overrides):
    defaults = dict(
        signal_ticker="QQQ", vol_method="ewma", vol_decay=0.94, mu_decay=0.94,
        adx_period=14, adx_threshold=25.0, fractional_kelly=0.5, max_leverage=3.0,
        worst_case_daily_move=0.20, ruin_buffer=0.20, vol_spike_threshold=0.40,
        drawdown_limit=0.15, entry_margin=0.3,
    )
    defaults.update(overrides)
    return KellyPositionAgent(**defaults)


def test_warmup_region_is_flat_cash():
    high, low, close = _trending_ohlc(n=300)
    ctx = {"universe_data": _universe(high, low, close)}
    ctx = _agent().run(ctx)
    position = ctx["leverage_position"]
    assert (position["fraction"].iloc[:60] == 0.0).all()


def test_output_schema_matches_leveraged_position_agent():
    high, low, close = _trending_ohlc(n=300)
    ctx = {"universe_data": _universe(high, low, close)}
    ctx = _agent().run(ctx)
    position = ctx["leverage_position"]
    assert list(position.columns) == ["ticker", "fraction"]
    assert len(position) == len(close)
    assert set(position["ticker"].unique()).issubset({"QQQ", "QLD", "TQQQ"})


def test_decision_log_has_expected_columns():
    high, low, close = _trending_ohlc(n=300)
    ctx = {"universe_data": _universe(high, low, close)}
    ctx = _agent().run(ctx)
    log = ctx["kelly_decision_log"]
    for col in ("mu_hat", "sigma_sq_hat", "l_star", "target_leverage", "regime", "ticker", "fraction"):
        assert col in log.columns


def test_strong_uptrend_eventually_enters_leveraged_position():
    high, low, close = _trending_ohlc(n=400, daily_drift=0.003, vol=0.005, seed=7)
    ctx = {"universe_data": _universe(high, low, close)}
    ctx = _agent(adx_threshold=15.0).run(ctx)
    position = ctx["leverage_position"]
    late_exposure = position["ticker"].iloc[-30:].isin(["QLD", "TQQQ"])
    assert late_exposure.any()


def test_position_never_implies_leverage_beyond_ruin_cap():
    high, low, close = _trending_ohlc(n=400, daily_drift=0.004, vol=0.003, seed=11)
    ctx = {"universe_data": _universe(high, low, close)}
    tier_leverage = {"QQQ": 1.0, "QLD": 2.0, "TQQQ": 3.0}
    ctx = _agent(worst_case_daily_move=0.20, ruin_buffer=0.20, adx_threshold=10.0).run(ctx)
    position = ctx["leverage_position"]
    max_safe = (1 - 0.20) / 0.20  # = 4.0, so cap never binds for QQQ/QLD/TQQQ tiers (max 3.0) --
    # use a tighter buffer to actually exercise the cap:
    effective_leverage = position.apply(lambda r: tier_leverage[r["ticker"]] * r["fraction"], axis=1)
    assert (effective_leverage <= 3.0 + 1e-9).all()


def test_ruin_cap_actually_binds_with_tight_buffer():
    high, low, close = _trending_ohlc(n=400, daily_drift=0.004, vol=0.002, seed=12)
    ctx = {"universe_data": _universe(high, low, close)}
    tier_leverage = {"QQQ": 1.0, "QLD": 2.0, "TQQQ": 3.0}
    # worst_case_daily_move=0.40, buffer=0.20 -> max safe leverage = 0.8/0.40 = 2.0
    ctx = _agent(worst_case_daily_move=0.40, ruin_buffer=0.20, adx_threshold=10.0,
                 fractional_kelly=1.0, max_leverage=3.0).run(ctx)
    position = ctx["leverage_position"]
    effective_leverage = position.apply(lambda r: tier_leverage[r["ticker"]] * r["fraction"], axis=1)
    assert (effective_leverage <= 2.0 + 1e-9).all()


def test_hysteresis_avoids_churn_on_small_fluctuations():
    high, low, close = _trending_ohlc(n=400, daily_drift=0.0025, vol=0.005, seed=20)
    ctx = {"universe_data": _universe(high, low, close)}
    ctx = _agent(entry_margin=0.5, adx_threshold=15.0).run(ctx)
    position = ctx["leverage_position"]
    # Count how often the (ticker, fraction) pair actually changes day over day.
    changes = (position["ticker"] != position["ticker"].shift(1)) | (
        (position["fraction"] - position["fraction"].shift(1)).abs() > 1e-9
    )
    n_changes = changes.iloc[60:].sum()  # after warmup
    # With a wide entry margin, a smoothly trending series should not churn
    # every single day.
    assert n_changes < len(position) * 0.5


def test_choppy_regime_caps_exposure_at_or_below_qqq_level():
    high, low, close = _choppy_ohlc(n=400, seed=30)
    ctx = {"universe_data": _universe(high, low, close)}
    tier_leverage = {"QQQ": 1.0, "QLD": 2.0, "TQQQ": 3.0}
    ctx = _agent(adx_threshold=25.0).run(ctx)
    position = ctx["leverage_position"]
    effective_leverage = position.apply(lambda r: tier_leverage[r["ticker"]] * r["fraction"], axis=1)
    log = ctx["kelly_decision_log"]
    choppy_mask = (log["regime"] == "choppy").values
    assert choppy_mask[60:].any(), "fixture should produce some choppy-labeled days"
    assert (effective_leverage[choppy_mask] <= 1.0 + 1e-9).all()


def test_drawdown_breach_forces_flat():
    # Strong uptrend to build a leveraged position, then a sharp crash that
    # breaches drawdown_limit on the held instrument -> must force fraction to 0.
    n = 300
    high, low, close = _trending_ohlc(n=n, daily_drift=0.003, vol=0.004, seed=42)
    crash_idx = n - 20
    close = close.copy()
    close.iloc[crash_idx:] = close.iloc[crash_idx] * np.linspace(1.0, 0.7, n - crash_idx)
    high = close * 1.004
    low = close * 0.996

    ctx = {"universe_data": _universe(high, low, close)}
    ctx = _agent(adx_threshold=15.0, drawdown_limit=0.15).run(ctx)
    position = ctx["leverage_position"]
    assert position["fraction"].iloc[-1] == pytest.approx(0.0)


def test_drawdown_check_does_not_use_same_day_own_price():
    # The decision for date t must depend ONLY on data through t-1 -- the
    # drawdown check must NOT use date t's own closing price (which already
    # reflects date t's own return) to decide date t's position. Two price
    # paths identical through index k-1 but diverging sharply AT index k
    # (one crashes, one doesn't) must produce an IDENTICAL decision at
    # index k; they may differ from index k+1 onward, once k's price is
    # historical.
    n = 300
    crash_idx = 250
    high_a, low_a, close_a = _trending_ohlc(n=n, daily_drift=0.003, vol=0.003, seed=99)

    close_b = close_a.copy()
    close_b.iloc[crash_idx] = close_a.iloc[crash_idx - 1] * 0.80  # -20% single-day crash at index k
    high_b, low_b = close_b * 1.004, close_b * 0.996

    ctx_a = {"universe_data": _universe(high_a, low_a, close_a)}
    ctx_b = {"universe_data": _universe(high_b, low_b, close_b)}

    agent_kwargs = dict(adx_threshold=10.0, drawdown_limit=0.10, vol_spike_threshold=0.40)
    pos_a = _agent(**agent_kwargs).run(ctx_a)["leverage_position"]
    pos_b = _agent(**agent_kwargs).run(ctx_b)["leverage_position"]

    # Prefix through index k-1 is identical by construction, so decisions
    # through k-1 trivially match. The critical check is index k itself:
    row_a = pos_a.iloc[crash_idx]
    row_b = pos_b.iloc[crash_idx]
    assert row_a["ticker"] == row_b["ticker"]
    assert row_a["fraction"] == pytest.approx(row_b["fraction"])


def test_signal_turning_negative_reduces_exposure_even_without_explicit_trigger():
    # A real gap found while dry-running this module live: once leveraged,
    # the position could ONLY shrink via vol-spike, drawdown-breach, or the
    # chop-cap-at-1x rule. If the Kelly-optimal target itself turns
    # negative (mu_hat < 0 -> l_star < 0 -> fractional_kelly clips to 0)
    # while regime stays "trending" and neither vol-spike nor drawdown-
    # breach has fired yet, the strategy must still de-risk toward that
    # degraded target -- holding a leveraged position purely because no
    # explicit trigger happened to fire is not growth-optimal, it's a gap
    # in the hysteresis logic.
    n = 300
    # Build a price path that trends up long enough to enter a leveraged
    # position, then reverses into a persistent downtrend (still smooth/
    # low-vol enough to stay "trending" per ADX and NOT trip the vol-spike
    # or drawdown-limit thresholds) so mu_hat goes negative while nothing
    # else forces a de-risk.
    up_days = 220
    down_days = n - up_days
    rng = np.random.RandomState(77)
    up_rets = 0.003 + rng.normal(0, 0.002, up_days)
    down_rets = -0.003 + rng.normal(0, 0.002, down_days)
    rets = np.concatenate([up_rets, down_rets])
    close = pd.Series(100.0 * np.cumprod(1 + rets), index=pd.bdate_range("2015-01-01", periods=n))
    high, low = close * 1.003, close * 0.997

    ctx = {"universe_data": _universe(high, low, close)}
    agent = _agent(adx_threshold=15.0, vol_spike_threshold=0.60, drawdown_limit=0.60, entry_margin=0.3)
    ctx = agent.run(ctx)
    position = ctx["leverage_position"]
    log = ctx["kelly_decision_log"]
    tier_leverage = {"QQQ": 1.0, "QLD": 2.0, "TQQQ": 3.0}
    effective_leverage = position.apply(lambda r: tier_leverage[r["ticker"]] * r["fraction"], axis=1)

    # Confirm the fixture actually built a leveraged position during the
    # uptrend and a negative mu_hat during the downtrend (both needed for
    # this to be a meaningful test, not a vacuous pass).
    assert effective_leverage.iloc[:up_days].max() > 1.0, "fixture should build leverage during the uptrend"
    assert log["mu_hat"].iloc[-1] < 0, "fixture should produce negative mu_hat by the end of the downtrend"

    # The position late in the downtrend must have come down toward zero,
    # not remained stuck at its uptrend-era leveraged level.
    assert effective_leverage.iloc[-1] < 1.0


def test_no_lookahead_prefix_unchanged_by_future_data():
    high, low, close = _trending_ohlc(n=400, seed=50)
    ctx_full = {"universe_data": _universe(high, low, close)}
    ctx_trunc = {"universe_data": _universe(high.iloc[:300], low.iloc[:300], close.iloc[:300])}

    agent = _agent(adx_threshold=15.0)
    pos_full = agent.run(ctx_full)["leverage_position"]
    pos_trunc = _agent(adx_threshold=15.0).run(ctx_trunc)["leverage_position"]

    pd.testing.assert_frame_equal(
        pos_full.iloc[:300].reset_index(drop=True), pos_trunc.reset_index(drop=True),
    )
