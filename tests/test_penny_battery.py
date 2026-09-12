"""Tests for research/penny/eval/battery.py.

The failure this module exists to prevent
------------------------------------------
Fourteen rounds compared strategies on a Sharpe ratio with no interval, no
deflation for the size of the search, and no check that its variance assumption
holds. Each metric here is pinned against a case where the right answer is known
in closed form, because a performance metric that is subtly wrong is worse than
no metric at all -- it is wrong with authority.

The anchors that matter most:
  * `sharpe_se` must collapse to the textbook normal formula on normal data
  * `expected_max_drawdown` must reproduce 1.2533 * sigma * sqrt(T) driftless
  * `hill_tail_index` must recover a Pareto index it was not told
  * `average_uniqueness` must return 1/k for k fully overlapping labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from research.penny.eval import battery as bt


def _normal(n=2000, mu=0.0005, sd=0.01, seed=0):
    return np.random.default_rng(seed).normal(mu, sd, n)


# ---- Tier 1: Sharpe ------------------------------------------------------
def test_sharpe_standard_error_matches_the_normal_formula_on_normal_data():
    """With g3=0 and g4=3 the estimator must reduce to sqrt((1+SR^2/2)/T)."""
    r = _normal(n=5000, seed=1)
    sr = bt.sharpe_ratio(r) / np.sqrt(bt.TRADING_DAYS)      # per-period
    expected = np.sqrt((1 + 0.5 * sr**2) / len(r)) * np.sqrt(bt.TRADING_DAYS)
    assert bt.sharpe_se(r) == pytest.approx(expected, rel=0.10), (
        "on normal data the non-normal SE must agree with the normal one")


def test_the_standard_error_shrinks_with_the_square_root_of_sample_size():
    short, long = _normal(n=500, seed=2), _normal(n=8000, seed=2)
    assert bt.sharpe_se(short) > 3 * bt.sharpe_se(long), (
        "16x the sample must roughly quarter the standard error")


def test_negative_skew_widens_the_standard_error_at_a_matched_sharpe():
    """A left tail makes the Sharpe less trustworthy -- holding the Sharpe fixed.

    The comparison has to be made at a matched mean and standard deviation.
    Simply adding large losses also moves the Sharpe itself, and the Sharpe
    enters the correction quadratically, so an unmatched comparison can move
    either way and tests nothing about skew.
    """
    rng = np.random.default_rng(3)
    sym = rng.normal(0.001, 0.01, 4000)
    skewed = sym.copy()
    skewed[::50] -= 0.08                        # rare large losses
    # rescale to the symmetric series' exact mean and sd
    skewed = (skewed - skewed.mean()) / skewed.std(ddof=1)
    skewed = skewed * sym.std(ddof=1) + sym.mean()

    assert sps.skew(skewed, bias=False) < -0.5, "the fixture must be left-skewed"
    assert bt.sharpe_ratio(skewed) == pytest.approx(bt.sharpe_ratio(sym), rel=1e-6)
    assert bt.sharpe_se(skewed) > bt.sharpe_se(sym), (
        "at the same Sharpe, negative skew must widen the interval")


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    a = bt.expected_max_sharpe(10, 1.0)
    b = bt.expected_max_sharpe(1000, 1.0)
    assert 0 < a < b, "a bigger search produces a bigger best-of-noise Sharpe"
    assert bt.expected_max_sharpe(1, 1.0) == 0.0
    assert bt.expected_max_sharpe(100, 0.0) == 0.0


def test_min_track_record_length_is_infinite_when_there_is_no_edge():
    flat = _normal(n=2000, mu=0.0, seed=4)
    assert bt.min_track_record_length(flat, benchmark_sr=2.0) == float("inf")


def test_min_track_record_length_falls_as_the_edge_grows():
    weak = _normal(n=3000, mu=0.0002, seed=5)
    strong = _normal(n=3000, mu=0.0015, seed=5)
    assert bt.min_track_record_length(strong) < bt.min_track_record_length(weak)


def test_autocorrelation_adjustment_penalises_a_positively_correlated_series():
    rng = np.random.default_rng(6)
    e = rng.normal(0.0005, 0.01, 4000)
    ar = np.zeros_like(e)
    for i in range(1, len(e)):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    ar += 0.0005
    assert bt.sharpe_autocorr_adjusted(ar, q=5) < bt.sharpe_ratio(ar), (
        "positive autocorrelation makes the naive sqrt(q) scaling optimistic")


# ---- Tier 2: shape of the loss ------------------------------------------
def test_expected_max_drawdown_reproduces_the_driftless_closed_form():
    """E[MDD] = sqrt(pi/2) * sigma * sqrt(T) for a driftless random walk.

    This is the one case Magdon-Ismail & Atiya give exactly, so it is the
    anchor for the Monte Carlo that replaces their series elsewhere.
    """
    sigma, T = 0.01, 500
    got = bt.expected_max_drawdown(0.0, sigma, T, n_sims=40_000, seed=7)
    want = np.sqrt(np.pi / 2) * sigma * np.sqrt(T)
    assert got == pytest.approx(want, rel=0.05), f"{got} vs {want}"


def test_expected_max_drawdown_shrinks_with_positive_drift():
    a = bt.expected_max_drawdown(0.0, 0.01, 500, n_sims=20_000, seed=8)
    b = bt.expected_max_drawdown(0.002, 0.01, 500, n_sims=20_000, seed=8)
    assert b < a, "drift pulls the path away from its running maximum"


def test_sortino_exceeds_sharpe_when_the_downside_is_tame():
    rng = np.random.default_rng(9)
    r = np.abs(rng.normal(0, 0.01, 3000)) * 1.0 - 0.002   # right-skewed
    assert bt.sortino_ratio(r) > bt.sharpe_ratio(r), (
        "penalising only downside deviation must flatter a right-skewed book")


def test_omega_is_one_at_the_break_even_threshold():
    r = np.array([-0.02, -0.01, 0.01, 0.02])
    assert bt.omega_ratio(r, threshold=0.0) == pytest.approx(1.0)


def test_omega_rises_as_the_threshold_falls():
    r = _normal(n=2000, seed=10)
    assert bt.omega_ratio(r, -0.01) > bt.omega_ratio(r, 0.01)


def test_ulcer_index_is_zero_for_a_monotonic_curve_and_positive_otherwise():
    assert bt.ulcer_index(np.array([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)
    assert bt.ulcer_index(np.array([1.0, 1.5, 0.9, 1.2])) > 0


def test_cvar_is_never_above_var():
    r = _normal(n=3000, seed=11)
    assert bt.conditional_var(r, 0.05) <= bt.value_at_risk(r, 0.05)


def test_drawdown_durations_count_only_underwater_days():
    eq = np.array([1.0, 1.1, 1.0, 0.9, 1.2, 1.3])   # 2 days under water
    d = bt.drawdown_durations(eq)
    assert d["max_underwater"] == 2
    assert d["pct_underwater"] == pytest.approx(2 / 6)


def test_hill_estimator_recovers_a_pareto_tail_index():
    """X = (1-U)^(-1/alpha) is Pareto with tail index alpha."""
    rng = np.random.default_rng(12)
    for alpha in (1.5, 3.0):
        x = (1 - rng.random(20_000)) ** (-1.0 / alpha)
        got = bt.hill_tail_index(x, k=2000)["alpha"]
        assert got == pytest.approx(alpha, rel=0.15), f"alpha={alpha} got {got}"


def test_hill_estimator_flags_a_thin_tail_as_high_alpha():
    r = _normal(n=20_000, mu=0.0, sd=1.0, seed=13)
    assert bt.hill_tail_index(r)["alpha"] > 2.0, (
        "a Gaussian has finite variance; alpha must land above 2")


def test_hill_estimator_refuses_a_sample_too_small_to_speak():
    assert np.isnan(bt.hill_tail_index(np.array([1.0, 2.0, 3.0]))["alpha"])


# ---- Tier 3: breadth -----------------------------------------------------
def test_effective_n_equals_n_for_equal_weights():
    assert bt.effective_n(np.full(60, 1 / 60)) == pytest.approx(60.0)


def test_effective_n_collapses_under_concentration():
    w = np.array([0.9, 0.05, 0.05])
    assert bt.effective_n(w) < 1.5, "one dominant position is barely a portfolio"


def test_correlation_adjusted_breadth_spans_n_to_one():
    assert bt.correlation_adjusted_breadth(60, 0.0) == pytest.approx(60.0)
    assert bt.correlation_adjusted_breadth(60, 1.0) == pytest.approx(1.0)
    assert bt.correlation_adjusted_breadth(60, 0.2) < 6.0, (
        "modest correlation destroys most of the nominal breadth")


def test_transfer_coefficient_is_one_when_the_book_is_unconstrained():
    w = np.array([0.1, -0.2, 0.3, 0.05, -0.25])
    assert bt.transfer_coefficient(w, w) == pytest.approx(1.0)


def test_transfer_coefficient_is_undefined_without_dispersion():
    ideal = np.array([0.1, -0.2, 0.3, 0.05])
    flat = np.full(4, 0.25)                 # equal weight: nothing transferred
    assert np.isnan(bt.transfer_coefficient(ideal, flat))


def test_non_overlapping_labels_are_perfectly_unique():
    cal = pd.bdate_range("2021-01-04", periods=40)
    trades = pd.DataFrame({
        "date": [cal[0], cal[10], cal[20]],
        "exit_date": [cal[5], cal[15], cal[25]],
    })
    u = bt.average_uniqueness(trades, cal)
    assert u["avg_uniqueness"] == pytest.approx(1.0)
    assert u["effective_n_trades"] == pytest.approx(3.0)


def test_k_fully_overlapping_labels_each_have_uniqueness_one_over_k():
    """This is the whole point: 60 concurrent holds are not 60 observations."""
    cal = pd.bdate_range("2021-01-04", periods=40)
    k = 5
    trades = pd.DataFrame({"date": [cal[0]] * k, "exit_date": [cal[10]] * k})
    u = bt.average_uniqueness(trades, cal)
    assert u["avg_uniqueness"] == pytest.approx(1 / k)
    assert u["effective_n_trades"] == pytest.approx(1.0), (
        "five identical labels carry the information of one")


def test_average_uniqueness_handles_an_empty_trade_list():
    cal = pd.bdate_range("2021-01-04", periods=10)
    u = bt.average_uniqueness(pd.DataFrame(columns=["date", "exit_date"]), cal)
    assert u["nominal_n_trades"] == 0


# ---- Tier 4: is the backtest real ---------------------------------------
def test_pbo_is_near_one_half_when_every_config_is_noise():
    """Selecting the best of N noise strategies must carry no information."""
    rng = np.random.default_rng(14)
    R = pd.DataFrame(rng.normal(0, 0.01, size=(600, 20)))
    out = bt.pbo_cscv(R, n_splits=8)
    assert 0.25 < out["pbo"] < 0.75, out


def test_pbo_is_low_when_one_config_is_genuinely_better():
    rng = np.random.default_rng(15)
    R = pd.DataFrame(rng.normal(0, 0.01, size=(600, 20)))
    R[7] = rng.normal(0.004, 0.01, 600)          # a real, persistent edge
    out = bt.pbo_cscv(R, n_splits=8)
    assert out["pbo"] < 0.25, out


def test_pbo_reports_nan_rather_than_guessing_on_a_single_config():
    R = pd.DataFrame(np.random.default_rng(16).normal(0, 0.01, size=(400, 1)))
    assert np.isnan(bt.pbo_cscv(R)["pbo"])


def test_the_stationary_bootstrap_returns_valid_wrapped_indices():
    idx = bt.stationary_bootstrap_indices(100, mean_block=10, n_boot=50, seed=17)
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_the_bootstrap_interval_brackets_the_point_estimate():
    r = _normal(n=1500, seed=18)
    point, lo, hi = bt.bootstrap_metric_ci(r, bt.sharpe_ratio, n_boot=200)
    assert lo < point < hi


# ---- Tier 5: execution ---------------------------------------------------
def test_corwin_schultz_returns_a_non_negative_fractional_spread():
    rng = np.random.default_rng(19)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    high, low = mid * 1.02, mid * 0.98
    s = bt.corwin_schultz_spread(high, low)
    assert 0.0 <= s < 0.5


def test_roll_spread_recovers_bid_ask_bounce():
    """A pure bounce series has a known negative serial covariance."""
    rng = np.random.default_rng(20)
    true_spread, mid = 0.02, 10.0
    p = mid + rng.choice([-1, 1], 5000) * true_spread / 2
    est = bt.roll_spread(p)
    assert est == pytest.approx(true_spread / mid, rel=0.25), est


def test_roll_spread_is_nan_when_the_covariance_is_positive():
    """Momentum in the price changes, not bid-ask bounce -- Roll is undefined.

    Returning NaN rather than clamping to zero matters: a clamped Roll estimate
    reads as "no spread" when what actually happened is that the estimator does
    not apply.
    """
    rng = np.random.default_rng(23)
    d = np.zeros(500)
    for i in range(1, len(d)):
        d[i] = 0.8 * d[i - 1] + rng.normal(0, 0.01)   # positively autocorrelated
    assert np.isnan(bt.roll_spread(10.0 + np.cumsum(d)))


def test_square_root_impact_scales_with_the_square_root_of_participation():
    a = bt.square_root_impact(0.05, 0.01)
    b = bt.square_root_impact(0.05, 0.04)
    assert b == pytest.approx(2 * a), "4x participation must double the impact"
    assert bt.square_root_impact(0.05, 0.0) == 0.0


def test_amihud_rises_as_dollar_volume_falls():
    r = np.full(100, 0.02)
    thick = bt.amihud_illiquidity(r, np.full(100, 1e7))
    thin = bt.amihud_illiquidity(r, np.full(100, 1e5))
    assert thin > thick


# ---- assembly ------------------------------------------------------------
def test_the_tearsheet_assembles_every_tier():
    cal = pd.bdate_range("2021-01-04", periods=800)
    rng = np.random.default_rng(21)
    daily = pd.Series(rng.normal(0.0006, 0.012, 800), index=cal)
    trades = pd.DataFrame({
        "date": cal[::10][:60], "exit_date": cal[60::10][:60],
        "net": rng.normal(0.05, 0.4, 60), "weight": np.full(60, 1 / 60),
    })
    t = bt.tearsheet(daily, trades=trades, calendar=cal,
                     n_trials=500, trial_sr_sd=0.6, bootstrap=False)
    for key in ("sharpe", "sharpe_se", "expected_max_sharpe", "sortino",
                "omega", "calmar", "ulcer_index", "cvar_95", "max_drawdown",
                "expected_max_drawdown", "avg_uniqueness", "effective_n",
                "tail_alpha_gains", "max_underwater"):
        assert key in t, f"{key} missing from the tearsheet"
    assert np.isfinite(t["sharpe"]) and np.isfinite(t["sharpe_se"])


def test_the_tearsheet_survives_a_short_series_without_raising():
    cal = pd.bdate_range("2021-01-04", periods=30)
    daily = pd.Series(np.random.default_rng(22).normal(0, 0.01, 30), index=cal)
    t = bt.tearsheet(daily, bootstrap=False)
    assert "sharpe" in t
