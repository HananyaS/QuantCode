"""Tests for utils/kelly_sizing.py — pure Kelly-criterion leverage math.

Formula note (see docs/research-log/kelly-leverage-sizing.md for full
derivation): g(L) = L*mu - 0.5*L^2*sigma^2, NOT the literal
L*mu - 0.5*L*(L-1)*sigma^2 one-liner sometimes quoted — that form omits the
L=1 Jensen drag (mu - 0.5*sigma^2) every buy-and-hold position has. Both
forms agree on the optimum L* = mu/sigma^2; they disagree on the absolute
g(L) value away from the optimum, which matters for the L=1 sanity check.
"""
import math

import pytest

from utils.kelly_sizing import (
    expected_log_growth,
    excess_leverage_decay,
    fractional_kelly,
    kelly_optimal_leverage,
    map_to_instrument_blend,
    ruin_floor_cap,
)


def test_kelly_optimal_leverage_matches_mu_over_sigma_sq():
    assert kelly_optimal_leverage(mu=0.0008, sigma_sq=0.0004) == pytest.approx(2.0)


def test_kelly_optimal_leverage_zero_sigma_returns_zero_not_inf():
    assert kelly_optimal_leverage(mu=0.001, sigma_sq=0.0) == 0.0


def test_kelly_optimal_leverage_negative_mu_is_negative():
    # A negative expected drift implies a negative optimal leverage (short) —
    # the sizing layer must clip this to zero downstream (no shorting), but
    # the raw Kelly formula itself should not silently floor it.
    assert kelly_optimal_leverage(mu=-0.001, sigma_sq=0.0004) < 0


def test_expected_log_growth_at_l_equals_1_is_jensen_term_only():
    mu, sigma_sq = 0.0006, 0.0003
    g1 = expected_log_growth(leverage=1.0, mu=mu, sigma_sq=sigma_sq)
    assert g1 == pytest.approx(mu - 0.5 * sigma_sq)


def test_expected_log_growth_at_l_equals_1_has_zero_excess_decay():
    mu, sigma_sq = 0.0006, 0.0003
    assert excess_leverage_decay(leverage=1.0, sigma_sq=sigma_sq) == pytest.approx(0.0)


def test_excess_leverage_decay_matches_cheng_madhavan_formula():
    # decay = L(L-1)/2 * sigma^2
    L, sigma_sq = 3.0, 0.0004
    expected = L * (L - 1) / 2 * sigma_sq
    assert excess_leverage_decay(leverage=L, sigma_sq=sigma_sq) == pytest.approx(expected)


def test_expected_log_growth_decomposes_into_jensen_plus_excess_decay():
    mu, sigma_sq, L = 0.0007, 0.00035, 2.5
    g = expected_log_growth(leverage=L, mu=mu, sigma_sq=sigma_sq)
    jensen_term = L * (mu - 0.5 * sigma_sq)
    decay = excess_leverage_decay(leverage=L, sigma_sq=sigma_sq)
    assert g == pytest.approx(jensen_term - decay)


def test_expected_log_growth_peaks_at_kelly_optimal_leverage():
    mu, sigma_sq = 0.0006, 0.00025
    l_star = kelly_optimal_leverage(mu, sigma_sq)
    g_star = expected_log_growth(l_star, mu, sigma_sq)
    for delta in (-0.5, -0.1, 0.1, 0.5):
        assert expected_log_growth(l_star + delta, mu, sigma_sq) < g_star


def test_half_kelly_gives_75_percent_of_growth():
    mu, sigma_sq = 0.0009, 0.0003
    l_star = kelly_optimal_leverage(mu, sigma_sq)
    g_full = expected_log_growth(l_star, mu, sigma_sq)
    g_half = expected_log_growth(0.5 * l_star, mu, sigma_sq)
    assert g_half / g_full == pytest.approx(0.75, rel=1e-9)


def test_fractional_kelly_applies_multiplier():
    assert fractional_kelly(l_star=2.0, fraction=0.5) == pytest.approx(1.0)


def test_fractional_kelly_clips_negative_to_zero():
    # Negative L* (negative expected drift) must clip to 0 (no shorting
    # support in this module — flat/cash is the floor).
    assert fractional_kelly(l_star=-1.5, fraction=0.5) == 0.0


def test_fractional_kelly_handles_nan_sigma_gracefully():
    # near-zero / degenerate conditional variance estimates must degrade
    # gracefully (flat), never raise or return inf/nan.
    l_star = kelly_optimal_leverage(mu=0.001, sigma_sq=1e-12)
    result = fractional_kelly(l_star=l_star, fraction=0.5, max_leverage=3.0)
    assert math.isfinite(result)
    assert 0.0 <= result <= 3.0


def test_ruin_floor_cap_blocks_leverage_within_buffer_of_ruin():
    # Ruin threshold for leverage L is a single-day move of -1/L. With a
    # 20% buffer, max allowed L is such that -1/L is 20% further away than
    # the worst realistic one-day move assumed here (e.g. -20% for equities
    # in a crash) -- i.e. cap so that 1/L * (1 - buffer) >= worst_case_move.
    capped = ruin_floor_cap(leverage=5.0, worst_case_daily_move=0.20, buffer=0.20)
    # 1/L must be >= worst_case_move / (1 - buffer) = 0.20 / 0.80 = 0.25 -> L <= 4.0
    assert capped == pytest.approx(4.0)


def test_ruin_floor_cap_no_op_when_leverage_already_safe():
    capped = ruin_floor_cap(leverage=1.5, worst_case_daily_move=0.20, buffer=0.20)
    assert capped == pytest.approx(1.5)


def test_ruin_floor_cap_never_negative():
    capped = ruin_floor_cap(leverage=-1.0, worst_case_daily_move=0.20, buffer=0.20)
    assert capped >= 0.0


def test_map_to_instrument_blend_exact_tier_match():
    ticker, fraction = map_to_instrument_blend(2.0, tiers=((1.0, "QQQ"), (2.0, "QLD"), (3.0, "TQQQ")))
    assert ticker == "QLD"
    assert fraction == pytest.approx(1.0)


def test_map_to_instrument_blend_interpolates_between_tiers():
    # 1.4x should use the smallest tier whose own leverage >= target (QLD,
    # 2x), scaled down to hit 1.4x exactly via partial-cash blend.
    ticker, fraction = map_to_instrument_blend(1.4, tiers=((1.0, "QQQ"), (2.0, "QLD"), (3.0, "TQQQ")))
    assert ticker == "QLD"
    assert fraction == pytest.approx(0.7)


def test_map_to_instrument_blend_zero_leverage_is_flat():
    ticker, fraction = map_to_instrument_blend(0.0, tiers=((1.0, "QQQ"), (2.0, "QLD"), (3.0, "TQQQ")))
    assert fraction == pytest.approx(0.0)
