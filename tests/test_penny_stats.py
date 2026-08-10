"""Tests for the inference layer.

These statistics exist to stop a search over thousands of veto sets from
manufacturing a result. If they are wrong in the permissive direction the whole
study silently becomes a random-number generator, so each correction is tested
for the property it is supposed to provide -- not merely for running.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.eval.stats import (
    benjamini_hochberg,
    cluster_bootstrap_mean,
    daily_equity_curve,
    deflated_sharpe_ratio,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


# ---- Clustered bootstrap ----


def test_bootstrap_interval_covers_a_known_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(0.05, 0.20, 2000)
    days = np.repeat(np.arange(200), 10)
    est = cluster_bootstrap_mean(values, days, n_boot=1000, seed=1)
    assert est.lo < 0.05 < est.hi
    assert est.n == 2000
    assert est.n_days == 200


def test_clustering_widens_the_interval_versus_pretending_independence():
    """Same-day trades share a market factor.

    Treating them as independent understates the standard error and
    manufactures significance -- the single easiest way to fake a result here.
    """
    rng = np.random.default_rng(5)
    n_days, per_day = 100, 20
    day_effect = rng.normal(0, 0.10, n_days)          # strong common factor
    values, days = [], []
    for d in range(n_days):
        values.extend(day_effect[d] + rng.normal(0, 0.01, per_day))
        days.extend([d] * per_day)

    clustered = cluster_bootstrap_mean(values, days, n_boot=1500, seed=2)
    pseudo_iid = cluster_bootstrap_mean(values, np.arange(len(values)), n_boot=1500, seed=2)

    clustered_width = clustered.hi - clustered.lo
    iid_width = pseudo_iid.hi - pseudo_iid.lo
    assert clustered_width > 3 * iid_width, (
        "clustered CI must be much wider when a day factor dominates"
    )


def test_bootstrap_on_a_true_null_does_not_reject():
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.1, 1500)
    days = np.repeat(np.arange(150), 10)
    est = cluster_bootstrap_mean(values, days, n_boot=1000, seed=4)
    assert not est.excludes_zero
    assert est.p_value > 0.05


def test_bootstrap_handles_empty_input():
    est = cluster_bootstrap_mean([], [], n_boot=10)
    assert est.n == 0
    assert np.isnan(est.mean)


# ---- Benjamini-Hochberg ----


def test_bh_rejects_nothing_when_all_hypotheses_are_null():
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1, 1000)
    rejected = benjamini_hochberg(p, q=0.05)
    assert rejected.sum() <= 0.05 * len(p)


def test_bh_finds_strong_signals_among_noise():
    p = np.concatenate([np.full(10, 1e-6), np.random.default_rng(1).uniform(0.2, 1.0, 990)])
    rejected = benjamini_hochberg(p, q=0.05)
    assert rejected[:10].all(), "clear signals must survive correction"


def test_bh_is_stricter_than_an_uncorrected_threshold():
    rng = np.random.default_rng(11)
    p = rng.uniform(0, 1, 500)
    assert benjamini_hochberg(p, q=0.05).sum() < (p < 0.05).sum()


def test_bh_on_empty_input():
    assert len(benjamini_hochberg([], q=0.05)) == 0


# ---- Sharpe family ----


def test_sharpe_of_a_constant_series_is_undefined():
    assert np.isnan(sharpe_ratio([0.01] * 50))


def test_sharpe_sign_follows_the_mean():
    rng = np.random.default_rng(2)
    assert sharpe_ratio(rng.normal(0.001, 0.01, 500)) > 0
    assert sharpe_ratio(rng.normal(-0.001, 0.01, 500)) < 0


def test_psr_falls_when_returns_are_negatively_skewed():
    """Skew and kurtosis should be penalised, not ignored."""
    rng = np.random.default_rng(9)
    symmetric = rng.normal(0.001, 0.01, 1000)
    skewed = symmetric.copy()
    skewed[:20] -= 0.10          # a few large losses
    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(symmetric)


def test_deflated_sharpe_falls_as_the_search_grows():
    """The core anti-overfitting property.

    The same track record is less impressive if it was the best of 10,000
    attempts than if it was the only attempt.
    """
    rng = np.random.default_rng(13)
    returns = rng.normal(0.0015, 0.01, 750)
    few = deflated_sharpe_ratio(returns, n_trials=1)
    many = deflated_sharpe_ratio(returns, n_trials=10_000)
    assert many < few


def test_deflated_sharpe_kills_a_marginal_result_under_heavy_search():
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0004, 0.01, 500)   # weak edge
    assert deflated_sharpe_ratio(returns, n_trials=5_000) < 0.95


# ---- Drawdown ----


def test_max_drawdown_of_a_monotonic_curve_is_zero():
    assert max_drawdown(np.cumprod(1 + np.full(50, 0.01))) == pytest.approx(0.0)


def test_max_drawdown_measures_peak_to_trough():
    equity = np.array([1.0, 2.0, 1.0, 1.5])
    assert max_drawdown(equity) == pytest.approx(-0.5)


# ---- Daily aggregation ----


def test_daily_curve_equal_weights_within_a_day():
    """A day with 40 signals must not take 40x the risk of a day with one."""
    trades = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05")] * 4 + [pd.Timestamp("2026-01-06")],
            "symbol": list("ABCDE"),
            "net_return": [0.10, 0.10, 0.10, 0.10, 0.20],
        }
    )
    curve = daily_equity_curve(trades)
    assert len(curve) == 2
    assert curve.iloc[0]["ret"] == pytest.approx(0.10)   # mean, not sum
    assert curve.iloc[1]["ret"] == pytest.approx(0.20)
    assert curve.iloc[-1]["equity"] == pytest.approx(1.10 * 1.20)


def test_daily_curve_respects_a_position_cap():
    trades = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05")] * 5,
            "symbol": list("ABCDE"),
            "net_return": [1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    capped = daily_equity_curve(trades, max_positions_per_day=2)
    assert capped.iloc[0]["n_trades"] == 2


def test_daily_curve_on_empty_input():
    empty = pd.DataFrame(columns=["date", "symbol", "net_return"])
    assert daily_equity_curve(empty).empty


# ---- Vectorised bootstrap must equal the naive one ----


def test_vectorised_bootstrap_matches_the_naive_estimator():
    """The fast path resamples per-day (sum, count) pairs.

    That is algebraically identical to concatenating each drawn day's values,
    not an approximation -- this pins that equivalence.
    """
    rng = np.random.default_rng(99)
    n_days, per_day = 40, 6
    values = rng.normal(0.01, 0.05, n_days * per_day)
    days = np.repeat(np.arange(n_days), per_day)

    fast = cluster_bootstrap_mean(values, days, n_boot=3000, seed=5)

    # Naive reference with the identical draw sequence.
    unique_days, day_index = np.unique(days, return_inverse=True)
    by_day = {d: values[days == d] for d in unique_days}
    ref_rng = np.random.default_rng(5)
    picked = ref_rng.integers(0, len(unique_days), size=(3000, len(unique_days)))
    naive = np.array([
        np.concatenate([by_day[unique_days[i]] for i in row]).mean() for row in picked
    ])

    assert fast.mean == pytest.approx(values.mean())
    assert fast.lo == pytest.approx(np.percentile(naive, 2.5), abs=1e-12)
    assert fast.hi == pytest.approx(np.percentile(naive, 97.5), abs=1e-12)


def test_bootstrap_handles_unequal_trades_per_day():
    """Days carry different trade counts; the estimator must weight by count."""
    values = np.array([1.0, 1.0, 1.0, 0.0])
    days = np.array(["a", "a", "a", "b"])
    est = cluster_bootstrap_mean(values, days, n_boot=500, seed=3)
    assert est.mean == pytest.approx(0.75)
    assert est.n == 4
    assert est.n_days == 2
