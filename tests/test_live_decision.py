"""Tests for utils/live_decision.py — converting a single-row timing-agent
decision into the full ticker->weight mapping ExecutionAgent needs, and
extending a universe with a next-session placeholder bar so forecast-for-t
agents (KellyPositionAgent) can emit a decision for the session actually
being traded.
"""
import numpy as np
import pandas as pd
import pytest

from utils.live_decision import append_next_session_bar, position_row_to_weights


def test_selected_ticker_gets_its_fraction():
    weights = position_row_to_weights("TQQQ", 0.83, tracked_tickers=("QQQ", "QLD", "TQQQ"))
    assert weights["TQQQ"] == pytest.approx(0.83)


def test_untraded_tickers_are_explicitly_zeroed():
    # Critical: ExecutionAgent only sells a position if its ticker appears
    # in the weights row with a lower/zero target -- an untraded ticker
    # missing from the dict entirely would never get sold if a prior day's
    # position needs to be unwound.
    weights = position_row_to_weights("TQQQ", 1.0, tracked_tickers=("QQQ", "QLD", "TQQQ"))
    assert weights["QQQ"] == 0.0
    assert weights["QLD"] == 0.0


def test_flat_position_zeros_every_tracked_ticker():
    weights = position_row_to_weights("QQQ", 0.0, tracked_tickers=("QQQ", "QLD", "TQQQ"))
    assert all(v == 0.0 for v in weights.values())


def test_rejects_ticker_outside_tracked_set():
    with pytest.raises(AssertionError):
        position_row_to_weights("SPY", 1.0, tracked_tickers=("QQQ", "QLD", "TQQQ"))


# ----------------------------------------------------------------------
# append_next_session_bar
# ----------------------------------------------------------------------

def _ohlcv_frame(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": close, "High": close * 1.004, "Low": close * 0.996,
        "Close": close, "Volume": 1_000_000,
    })


def _mini_universe(n: int = 300, final_bar_crash: float | None = None) -> dict:
    rng = np.random.RandomState(0)
    rets = 0.0025 + rng.normal(0, 0.006, n)
    if final_bar_crash is not None:
        rets[-1] = final_bar_crash
    close = pd.Series(100.0 * np.cumprod(1 + rets), index=pd.bdate_range("2015-01-01", periods=n))
    return {t: _ohlcv_frame(close) for t in ("QQQ", "QLD", "TQQQ")}


def test_append_next_session_bar_appends_nan_row_without_mutating_input():
    universe = _mini_universe()
    original_lengths = {t: len(df) for t, df in universe.items()}
    next_session = universe["QQQ"].index[-1] + pd.tseries.offsets.BDay(1)

    extended = append_next_session_bar(universe, next_session, tickers=("QQQ", "QLD", "TQQQ"))

    for t in ("QQQ", "QLD", "TQQQ"):
        assert len(universe[t]) == original_lengths[t], "input universe must not be mutated"
        assert len(extended[t]) == original_lengths[t] + 1
        assert extended[t].index[-1] == pd.Timestamp(next_session)
        assert extended[t].index.is_monotonic_increasing
        assert extended[t]["Close"].iloc[-1] != extended[t]["Close"].iloc[-1]  # NaN
        # Every real bar's VALUES are untouched. check_freq/check_dtype
        # relaxed: concat drops bdate_range freq metadata and upcasts int
        # Volume to float when a NaN row joins -- neither is observable by
        # any agent (Volume is unused in the Kelly path, freq by nothing).
        pd.testing.assert_frame_equal(
            extended[t].iloc[:-1], universe[t], check_freq=False, check_dtype=False,
        )


def test_append_next_session_bar_rejects_non_future_date():
    universe = _mini_universe()
    last_bar = universe["QQQ"].index[-1]
    with pytest.raises(AssertionError):
        append_next_session_bar(universe, last_bar, tickers=("QQQ", "QLD", "TQQQ"))


def test_extended_kelly_decision_sees_final_bar_crash():
    """THE reason append_next_session_bar exists: KellyPositionAgent's
    forecast-for-t convention means the row at the last completed bar t was
    decided from data through t-1 -- executing it during session t+1
    discards bar t's information entirely. A -25% final bar must flatten
    the NEXT session's decision (drawdown/vol triggers), even though the
    last real row's decision -- decided before the crash -- stays leveraged.
    """
    from agents.timing.kelly_position_agent import KellyPositionAgent

    universe = _mini_universe(n=300, final_bar_crash=-0.25)
    next_session = universe["QQQ"].index[-1] + pd.tseries.offsets.BDay(1)
    agent_kwargs = dict(
        signal_ticker="QQQ", vol_method="ewma", vol_decay=0.94, mu_decay=0.94,
        adx_period=14, adx_threshold=15.0, fractional_kelly=0.5, max_leverage=3.0,
        worst_case_daily_move=0.20, ruin_buffer=0.20, vol_spike_threshold=0.40,
        drawdown_limit=0.15, entry_margin=0.3,
    )

    plain = KellyPositionAgent(**agent_kwargs).run({"universe_data": universe})
    last_real_row = plain["leverage_position"].iloc[-1]
    tier_leverage = {"QQQ": 1.0, "QLD": 2.0, "TQQQ": 3.0}
    exposure_before = tier_leverage[last_real_row["ticker"]] * last_real_row["fraction"]
    assert exposure_before > 0, (
        "fixture must hold a position at the crash bar (decided pre-crash) "
        "for the staleness demonstration to mean anything"
    )

    extended = append_next_session_bar(universe, next_session, tickers=("QQQ", "QLD", "TQQQ"))
    ctx = KellyPositionAgent(**agent_kwargs).run({"universe_data": extended})
    next_row = ctx["leverage_position"].iloc[-1]
    assert ctx["leverage_position"].index[-1] == pd.Timestamp(next_session)
    assert next_row["fraction"] == 0.0, (
        "the next-session decision must reflect the crash bar and de-risk to flat"
    )


def test_extended_kelly_mu_uses_all_data_through_last_bar():
    """Structural check of the staleness fix: the appended row's mu_hat must
    equal the EWM of returns through the final REAL bar -- i.e. the freshest
    information available, which the last real row's own mu_hat (shifted,
    data through t-1) deliberately excludes.
    """
    from agents.timing.kelly_position_agent import KellyPositionAgent

    universe = _mini_universe(n=300)
    next_session = universe["QQQ"].index[-1] + pd.tseries.offsets.BDay(1)
    extended = append_next_session_bar(universe, next_session, tickers=("QQQ", "QLD", "TQQQ"))

    ctx = KellyPositionAgent(
        signal_ticker="QQQ", vol_method="ewma", vol_decay=0.94, mu_decay=0.94,
    ).run({"universe_data": extended})
    log = ctx["kelly_decision_log"]

    expected_mu = (
        universe["QQQ"]["Close"].pct_change().ewm(alpha=1 - 0.94, adjust=False).mean().iloc[-1]
    )
    assert log["mu_hat"].iloc[-1] == pytest.approx(expected_mu, rel=1e-12)
