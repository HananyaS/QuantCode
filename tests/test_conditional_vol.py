"""Tests for utils/conditional_vol.py — causal, forward-looking one-step-
ahead conditional variance forecasts (EWMA baseline + GJR-GARCH option).
"""
import numpy as np
import pandas as pd
import pytest

from utils.conditional_vol import ewma_variance, gjr_garch_variance


def _returns(n=300, seed=0, scale=0.01):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(0, scale, n), index=dates)


def test_ewma_variance_is_causal_forecast_for_next_day():
    # sigma_hat_t^2 must be usable to forecast day t's return using only
    # data through t-1 -- i.e. shifted by construction, never same-day.
    rets = _returns()
    var = ewma_variance(rets, decay=0.94)
    # First value must be NaN (no prior data to form a forecast from).
    assert pd.isna(var.iloc[0])
    assert var.notna().iloc[1:].all()


def test_ewma_variance_no_lookahead():
    rets = _returns(n=300)
    full = ewma_variance(rets, decay=0.94)
    truncated = ewma_variance(rets.iloc[:250], decay=0.94)
    pd.testing.assert_series_equal(full.iloc[:250], truncated, check_names=False)


def test_ewma_variance_reacts_more_to_recent_shock_with_lower_decay():
    # A sudden large-return day should raise the NEXT day's forecast more
    # under a lower decay (faster-reacting) EWMA than a higher decay one.
    n = 100
    dates = pd.bdate_range("2015-01-01", periods=n)
    rets = pd.Series(0.001, index=dates)
    rets.iloc[50] = 0.10  # shock

    fast = ewma_variance(rets, decay=0.80)
    slow = ewma_variance(rets, decay=0.97)
    assert fast.iloc[51] > slow.iloc[51]


def test_ewma_variance_always_non_negative():
    rets = _returns(n=200, seed=3)
    var = ewma_variance(rets, decay=0.94)
    assert (var.dropna() >= 0).all()


def test_gjr_garch_variance_is_causal_and_finite():
    rets = _returns(n=400, seed=1, scale=0.015)
    var = gjr_garch_variance(rets, refit_every=50, min_train_obs=100)
    valid = var.dropna()
    assert len(valid) > 0
    assert np.isfinite(valid).all()
    assert (valid >= 0).all()


def test_gjr_garch_variance_no_lookahead_across_refit_boundary():
    # A forecast made using a refit that only had access to data through
    # day t must not change when future data (beyond day t) is added.
    rets = _returns(n=400, seed=2, scale=0.012)
    full = gjr_garch_variance(rets, refit_every=50, min_train_obs=100)
    truncated = gjr_garch_variance(rets.iloc[:300], refit_every=50, min_train_obs=100)
    # Compare the overlapping, non-NaN prefix.
    common = full.iloc[:300].dropna()
    truncated_common = truncated.loc[common.index]
    pd.testing.assert_series_equal(common, truncated_common, check_names=False, rtol=1e-6)


def test_gjr_garch_asymmetric_response_to_negative_shock():
    # The defining GJR feature: a negative return shock must raise the
    # NEXT forecast more than a positive shock of equal magnitude
    # (leverage effect).
    n = 300
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.RandomState(5)
    base = rng.normal(0, 0.01, n)

    rets_neg_shock = pd.Series(base.copy(), index=dates)
    rets_neg_shock.iloc[200] = -0.08

    rets_pos_shock = pd.Series(base.copy(), index=dates)
    rets_pos_shock.iloc[200] = 0.08

    var_neg = gjr_garch_variance(rets_neg_shock, refit_every=250, min_train_obs=150)
    var_pos = gjr_garch_variance(rets_pos_shock, refit_every=250, min_train_obs=150)

    assert var_neg.iloc[201] > var_pos.iloc[201]


def test_gjr_garch_insufficient_data_returns_all_nan_not_raise():
    rets = _returns(n=20, seed=4)
    var = gjr_garch_variance(rets, refit_every=50, min_train_obs=100)
    assert var.isna().all()
