"""Tests for utils/regime_classifier.py — deterministic MA-slope/ADX-style
regime classifier + conditional mu estimator.

Both `conditional_mu` and `classify_regime` follow the SAME forecast-for-t
convention as utils/conditional_vol.py (a value at t depends only on data
through t-1) — this is a deliberate consistency choice so KellyPositionAgent
can combine mu_hat_t and sigma_hat_t^2 directly without additional
downstream shifting (see docs/research-log/kelly-leverage-sizing.md).
"""
import numpy as np
import pandas as pd
import pytest

from utils.regime_classifier import average_directional_index, classify_regime, conditional_mu


def _trending_ohlc(n=300, daily_drift=0.003, seed=0):
    rng = np.random.RandomState(seed)
    rets = daily_drift + rng.normal(0, 0.004, n)
    close = 100.0 * np.cumprod(1 + rets)
    dates = pd.bdate_range("2015-01-01", periods=n)
    close = pd.Series(close, index=dates)
    high = close * 1.005
    low = close * 0.995
    return high, low, close


def _choppy_ohlc(n=300, seed=1):
    rng = np.random.RandomState(seed)
    # Mean-reverting: alternating sign shocks around a flat level.
    rets = rng.normal(0, 0.012, n)
    rets = rets - pd.Series(rets).rolling(5, min_periods=1).mean().values * 0.8
    close = 100.0 * np.cumprod(1 + rets)
    dates = pd.bdate_range("2015-01-01", periods=n)
    close = pd.Series(close, index=dates)
    high = close * 1.01
    low = close * 0.99
    return high, low, close


def test_conditional_mu_is_causal_forecast():
    _, _, close = _trending_ohlc()
    rets = close.pct_change().dropna()
    mu = conditional_mu(rets, decay=0.94)
    assert pd.isna(mu.iloc[0])
    assert mu.notna().iloc[1:].all()


def test_conditional_mu_no_lookahead():
    _, _, close = _trending_ohlc(n=300)
    rets = close.pct_change().dropna()
    full = conditional_mu(rets, decay=0.94)
    truncated = conditional_mu(rets.iloc[:200], decay=0.94)
    pd.testing.assert_series_equal(full.iloc[:200], truncated, check_names=False)


def test_conditional_mu_positive_in_strong_uptrend():
    _, _, close = _trending_ohlc(n=300, daily_drift=0.004)
    rets = close.pct_change().dropna()
    mu = conditional_mu(rets, decay=0.90)
    assert mu.iloc[-1] > 0


def test_adx_higher_for_trending_than_choppy_series():
    h_trend, l_trend, c_trend = _trending_ohlc(n=250, daily_drift=0.003)
    h_chop, l_chop, c_chop = _choppy_ohlc(n=250)

    adx_trend = average_directional_index(h_trend, l_trend, c_trend, period=14)
    adx_chop = average_directional_index(h_chop, l_chop, c_chop, period=14)

    assert adx_trend.iloc[-20:].mean() > adx_chop.iloc[-20:].mean()


def test_classify_regime_is_causal_forecast():
    h, l, c = _trending_ohlc(n=250)
    regime = classify_regime(h, l, c, adx_period=14, adx_threshold=25.0)
    assert regime.iloc[0] is None or pd.isna(regime.iloc[0])
    assert regime.iloc[30:].notna().all()


def test_classify_regime_no_lookahead():
    h, l, c = _trending_ohlc(n=300)
    full = classify_regime(h, l, c, adx_period=14, adx_threshold=25.0)
    h2, l2, c2 = h.iloc[:250], l.iloc[:250], c.iloc[:250]
    truncated = classify_regime(h2, l2, c2, adx_period=14, adx_threshold=25.0)
    pd.testing.assert_series_equal(full.iloc[:250], truncated, check_names=False)


def test_classify_regime_labels_are_trending_or_choppy():
    h, l, c = _trending_ohlc(n=250)
    regime = classify_regime(h, l, c, adx_period=14, adx_threshold=25.0)
    labels = set(regime.dropna().unique())
    assert labels.issubset({"trending", "choppy"})


def test_classify_regime_strong_trend_is_labeled_trending():
    h, l, c = _trending_ohlc(n=250, daily_drift=0.004, seed=9)
    regime = classify_regime(h, l, c, adx_period=14, adx_threshold=20.0)
    assert (regime.iloc[-20:] == "trending").mean() > 0.5
