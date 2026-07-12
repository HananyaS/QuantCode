"""Tests for utils/synthetic_leveraged_series.py — exact daily-reset NAV
recursion V_{t+1} = V_t * max(1 + L*r_t - daily_expense, 0), NOT the naive
(wrong) L * cumulative_underlying_return approximation.
"""
import numpy as np
import pandas as pd
import pytest

from utils.synthetic_leveraged_series import (
    build_synthetic_leveraged_close,
    build_synthetic_universe,
    calibrate_total_annual_drag,
)


def _underlying(closes, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.Series(closes, index=dates, dtype=float)


def test_first_bar_equals_starting_underlying_price():
    underlying = _underlying([100, 102, 98, 105])
    synth = build_synthetic_leveraged_close(underlying, leverage=3.0, expense_ratio_annual=0.0)
    assert synth.iloc[0] == pytest.approx(100.0)


def test_matches_exact_recursion_for_known_path_no_expense():
    underlying = _underlying([100, 110, 99, 108.9])  # +10%, -10%, +10%
    synth = build_synthetic_leveraged_close(underlying, leverage=2.0, expense_ratio_annual=0.0)
    # day1: r=+10% -> factor = 1+2*0.10 = 1.20 -> NAV = 100*1.20 = 120
    # day2: r=(99-110)/110 = -10% -> factor = 1+2*(-0.10) = 0.80 -> NAV = 120*0.80 = 96
    # day3: r=(108.9-99)/99 = +10% -> factor = 1.20 -> NAV = 96*1.20 = 115.2
    assert synth.iloc[1] == pytest.approx(120.0, rel=1e-6)
    assert synth.iloc[2] == pytest.approx(96.0, rel=1e-6)
    assert synth.iloc[3] == pytest.approx(115.2, rel=1e-6)


def test_diverges_from_naive_multiplicative_approximation_in_choppy_market():
    # Classic leveraged-ETF decay demonstration: underlying round-trips to
    # its starting price (+10% then -100/110 ~ -9.09%), so naive L*cumret
    # says the leveraged fund should also be flat, but the exact recursion
    # shows real decay.
    underlying = _underlying([100, 110, 100])
    synth = build_synthetic_leveraged_close(underlying, leverage=3.0, expense_ratio_annual=0.0)

    underlying_cumret = underlying.iloc[-1] / underlying.iloc[0] - 1  # ~0.0
    naive_leveraged_price = underlying.iloc[0] * (1 + 3.0 * underlying_cumret)
    actual_leveraged_price = synth.iloc[-1]

    assert underlying_cumret == pytest.approx(0.0, abs=1e-9)
    assert naive_leveraged_price == pytest.approx(100.0, abs=1e-6)
    # The actual 3x fund is NOT flat -- it has decayed due to volatility drag.
    assert actual_leveraged_price < naive_leveraged_price - 1.0


def test_ruin_floor_is_absorbing():
    # A single-day move worse than -1/L wipes the fund to exactly 0, and it
    # must STAY at 0 even if the underlying fully recovers afterward.
    underlying = _underlying([100, 50, 100])  # -50% then +100%
    synth = build_synthetic_leveraged_close(underlying, leverage=3.0, expense_ratio_annual=0.0)
    # day1: r=-50%, L=3 -> 1+3*(-0.50) = -0.5 -> floored to 0
    assert synth.iloc[1] == pytest.approx(0.0, abs=1e-9)
    # day2: NAV already 0 -> stays 0 regardless of underlying's recovery
    assert synth.iloc[2] == pytest.approx(0.0, abs=1e-9)


def test_expense_drag_reduces_nav_relative_to_zero_expense():
    rng = np.random.RandomState(0)
    n = 500
    rets = rng.normal(0.0005, 0.01, n)
    underlying = _underlying((100 * np.cumprod(1 + rets)).tolist())

    no_expense = build_synthetic_leveraged_close(underlying, leverage=2.0, expense_ratio_annual=0.0)
    with_expense = build_synthetic_leveraged_close(underlying, leverage=2.0, expense_ratio_annual=0.0095)

    assert with_expense.iloc[-1] < no_expense.iloc[-1]


def test_expense_drag_matches_expected_annualized_magnitude_over_flat_underlying():
    # With a perfectly flat (zero-return, zero-vol) underlying, ONLY the
    # expense drag should reduce NAV -- an exact, easily-verified case.
    n = 253  # 252 trading days of drag (day 0 has no elapsed return)
    underlying = _underlying([100.0] * n)
    synth = build_synthetic_leveraged_close(underlying, leverage=3.0, expense_ratio_annual=0.0095)

    daily_drag = 0.0095 / 252
    expected = 100.0 * (1 - daily_drag) ** (n - 1)
    assert synth.iloc[-1] == pytest.approx(expected, rel=1e-6)


def test_build_synthetic_universe_passes_through_real_qqq_unmodified():
    underlying = _underlying([100, 102, 98, 105])
    universe = build_synthetic_universe(underlying)
    pd.testing.assert_series_equal(universe["QQQ"]["Close"], underlying, check_names=False)


def test_build_synthetic_universe_includes_qld_and_tqqq_with_close_column():
    underlying = _underlying([100, 102, 98, 105, 110])
    universe = build_synthetic_universe(underlying)
    assert set(universe.keys()) == {"QQQ", "QLD", "TQQQ"}
    for ticker, df in universe.items():
        assert "Close" in df.columns
        assert len(df) == len(underlying)


def test_calibrate_total_annual_drag_recovers_known_synthetic_drag():
    # Build a "real" series with a KNOWN total drag (0.05 = 5%), then check
    # calibration recovers that number from the underlying + "real" pair —
    # this is what confirms the calibration function (used against actual
    # cached TQQQ/QLD data to find the true empirical drag, since the
    # published expense ratio alone badly understates it — real TQQQ's
    # implied annual drag runs far above its stated 0.95% expense ratio due
    # to unmodeled financing/borrowing costs) is itself correct.
    rng = np.random.RandomState(2)
    n = 600
    rets = rng.normal(0.0006, 0.011, n)
    underlying = _underlying((100 * np.cumprod(1 + rets)).tolist())

    known_drag = 0.05
    synthetic_real = build_synthetic_leveraged_close(underlying, leverage=3.0, expense_ratio_annual=known_drag)

    recovered = calibrate_total_annual_drag(underlying, synthetic_real, leverage=3.0)
    assert recovered == pytest.approx(known_drag, abs=1e-4)


def test_calibrate_total_annual_drag_zero_when_no_extra_drag():
    rng = np.random.RandomState(3)
    n = 400
    rets = rng.normal(0.0004, 0.009, n)
    underlying = _underlying((100 * np.cumprod(1 + rets)).tolist())
    synthetic_real = build_synthetic_leveraged_close(underlying, leverage=2.0, expense_ratio_annual=0.0)

    recovered = calibrate_total_annual_drag(underlying, synthetic_real, leverage=2.0)
    assert recovered == pytest.approx(0.0, abs=1e-6)


def test_build_synthetic_universe_tqqq_leverage_exceeds_qld():
    rng = np.random.RandomState(1)
    n = 300
    rets = rng.normal(0.0008, 0.012, n)
    underlying = _underlying((100 * np.cumprod(1 + rets)).tolist())
    universe = build_synthetic_universe(underlying)

    qld_ret = universe["QLD"]["Close"].iloc[-1] / universe["QLD"]["Close"].iloc[0] - 1
    tqqq_ret = universe["TQQQ"]["Close"].iloc[-1] / universe["TQQQ"]["Close"].iloc[0] - 1
    underlying_ret = underlying.iloc[-1] / underlying.iloc[0] - 1
    # In a strong enough uptrend both should beat the underlying and TQQQ > QLD.
    assert underlying_ret > 0
    assert tqqq_ret > qld_ret
