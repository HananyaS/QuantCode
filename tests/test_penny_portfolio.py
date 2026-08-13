"""What the account would have done, rather than what the average trade did.

Six rounds reported per-trade expectancy, which weights every trade equally. An
account weights every *day* equally. The two diverge whenever candidate count
covaries with candidate quality — and in a reversal strategy it plausibly does,
since candidates spike on turbulent days.

The flattering failure guarded here: dropping days the strategy sat out. That
reports returns only on the days it chose to act, which inflates any rule that
abstains during bad conditions — and abstention is exactly what the
expected-value rules in this study do.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.eval.portfolio import (
    compare_weightings,
    daily_returns,
    evaluate_portfolio,
)


def _trades(rows):
    """(date, net_return) pairs, optionally with a rank."""
    df = pd.DataFrame(rows, columns=["date", "net_return"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _steady(n_days=60, per_day=5, ret=0.01, start="2024-01-02"):
    days = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame({
        "date": np.repeat(days, per_day),
        "net_return": ret,
    })


# ---- Equal weighting within a day ----


def test_daily_return_is_the_equal_weight_mean_of_that_day():
    t = _trades([("2024-01-02", 0.10), ("2024-01-02", -0.02),
                 ("2024-01-03", 0.04)])
    d = daily_returns(t).set_index("date")
    assert d.loc["2024-01-02", "ret"] == pytest.approx(0.04)
    assert d.loc["2024-01-03", "ret"] == pytest.approx(0.04)


def test_position_count_is_recorded_per_day():
    t = _trades([("2024-01-02", 0.01)] * 7 + [("2024-01-03", 0.01)])
    d = daily_returns(t).set_index("date")
    assert d.loc["2024-01-02", "n_positions"] == 7
    assert d.loc["2024-01-03", "n_positions"] == 1


# ---- Sat-out days are flat, not absent ----


def test_days_with_no_trade_earn_zero_when_a_calendar_is_supplied():
    cal = pd.bdate_range("2024-01-02", periods=5)
    t = _trades([("2024-01-02", 0.10), ("2024-01-04", -0.10)])
    d = daily_returns(t, all_sessions=cal)
    assert len(d) == 5
    assert d["ret"].tolist() == pytest.approx([0.10, 0.0, -0.10, 0.0, 0.0])


def test_abstaining_on_bad_days_does_not_inflate_the_mean():
    """The failure this module exists to prevent."""
    cal = pd.bdate_range("2024-01-02", periods=10)
    # Acts on two days only, both winners; sits out the other eight.
    t = _trades([("2024-01-02", 0.05), ("2024-01-03", 0.05)])
    without_cal = evaluate_portfolio(t.assign(x=1), all_sessions=None)
    with_cal = evaluate_portfolio(t, all_sessions=cal)
    assert with_cal is not None
    assert with_cal.mean_daily < 0.05, "sat-out days were dropped"
    assert with_cal.mean_daily == pytest.approx(0.01)
    assert with_cal.pct_days_active == pytest.approx(0.2)


def test_pct_days_active_reports_how_often_the_rule_trades():
    cal = pd.bdate_range("2024-01-02", periods=10)
    t = _trades([("2024-01-02", 0.01), ("2024-01-09", 0.01)])
    res = evaluate_portfolio(t, all_sessions=cal)
    assert res.pct_days_active == pytest.approx(0.2)


# ---- Capital realism ----


def test_max_positions_caps_simultaneous_holdings():
    t = _trades([("2024-01-02", 0.01)] * 50)
    d = daily_returns(t, max_positions=10)
    assert d.iloc[0]["n_positions"] == 10


def test_the_cap_keeps_the_highest_ranked_names():
    t = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 4),
        "net_return": [0.10, -0.10, 0.20, -0.20],
        "score": [0.9, 0.1, 0.8, 0.2],
    })
    d = daily_returns(t, max_positions=2, rank_col="score")
    # Keeps score 0.9 and 0.8 -> returns 0.10 and 0.20
    assert d.iloc[0]["ret"] == pytest.approx(0.15)


def test_a_cap_above_the_day_count_changes_nothing():
    t = _trades([("2024-01-02", 0.02)] * 3)
    a = daily_returns(t)
    b = daily_returns(t, max_positions=99)
    pd.testing.assert_frame_equal(a, b)


# ---- Path statistics ----


def test_equity_compounds_rather_than_sums():
    t = _steady(n_days=10, per_day=1, ret=0.10)
    res = evaluate_portfolio(t)
    assert res.total_return == pytest.approx(1.10 ** 10 - 1.0, rel=1e-9)
    assert res.total_return > 10 * 0.10, "returns were summed, not compounded"


def test_drawdown_is_measured_peak_to_trough_and_is_never_positive():
    t = _trades([("2024-01-0%d" % d, r) for d, r in
                 zip(range(2, 7), [0.20, -0.30, -0.10, 0.05, 0.05])])
    res = evaluate_portfolio(t)
    assert res.max_drawdown < 0
    assert res.max_drawdown == pytest.approx((0.7 * 0.9) - 1.0, abs=1e-9)


def test_a_flat_series_has_zero_drawdown():
    res = evaluate_portfolio(_steady(n_days=20, per_day=1, ret=0.0))
    assert res.max_drawdown == pytest.approx(0.0)


def test_sharpe_is_annualised_from_daily():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=500)
    t = pd.DataFrame({"date": days, "net_return": rng.normal(0.001, 0.01, 500)})
    res = evaluate_portfolio(t)
    expected = (res.mean_daily / res.sd_daily) * np.sqrt(252)
    assert res.sharpe == pytest.approx(expected)


def test_sharpe_is_nan_rather_than_infinite_on_a_constant_series():
    res = evaluate_portfolio(_steady(n_days=20, per_day=1, ret=0.01))
    assert np.isnan(res.sharpe)


def test_losing_streak_is_reported():
    t = _trades([("2024-01-%02d" % d, r) for d, r in
                 zip(range(2, 9), [0.05, -0.01, -0.01, -0.01, -0.01, 0.05, 0.05])])
    assert evaluate_portfolio(t).max_consecutive_losing_days == 4


# ---- Per-trade against per-day ----


def test_weightings_agree_when_every_day_has_the_same_count():
    t = _steady(n_days=20, per_day=4, ret=0.01)
    c = compare_weightings(t)
    assert c["gap"] == pytest.approx(0.0, abs=1e-12)


def test_weightings_diverge_when_busy_days_are_worse():
    """Per-trade averaging over-weights whatever the busy days did."""
    rows = [("2024-01-02", -0.10)] * 20 + [("2024-01-03", 0.10)] * 2
    c = compare_weightings(_trades(rows))
    assert c["per_trade_mean"] < 0, "per-trade is dragged by the busy losing day"
    assert c["per_day_mean"] == pytest.approx(0.0)
    assert c["gap"] > 0


def test_weightings_report_the_count_return_correlation():
    rows = ([("2024-01-02", 0.10)] * 10 + [("2024-01-03", -0.10)] * 2
            + [("2024-01-04", 0.05)] * 6)
    c = compare_weightings(_trades(rows))
    assert c["corr_count_vs_return"] > 0
    assert c["n_days"] == 3


# ---- Degenerate input ----


def test_empty_input_returns_none_rather_than_raising():
    assert evaluate_portfolio(pd.DataFrame()) is None


def test_too_few_days_returns_none():
    assert evaluate_portfolio(_trades([("2024-01-02", 0.01)])) is None


def test_all_returns_missing_returns_none():
    t = _trades([("2024-01-0%d" % d, np.nan) for d in range(2, 9)])
    assert evaluate_portfolio(t) is None


def test_missing_returns_are_dropped_not_zeroed():
    t = _trades([("2024-01-02", 0.10), ("2024-01-02", np.nan),
                 ("2024-01-03", 0.10), ("2024-01-04", 0.10),
                 ("2024-01-05", 0.10), ("2024-01-08", 0.10)])
    d = daily_returns(t).set_index("date")
    assert d.loc["2024-01-02", "ret"] == pytest.approx(0.10)
    assert d.loc["2024-01-02", "n_positions"] == 1
