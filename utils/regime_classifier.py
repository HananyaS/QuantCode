"""regime_classifier.py — deterministic, causal trend/regime classification.

`conditional_mu` and `classify_regime` both follow the same forecast-for-t
convention as utils/conditional_vol.py (a value at date t depends only on
data through t-1) — deliberate consistency so KellyPositionAgent can
combine mu_hat_t and sigma_hat_t^2 directly to compute L*_t without an
additional downstream execution-lag shift (see
docs/research-log/kelly-leverage-sizing.md for why: KellyBacktestAgent does
NOT re-shift its input position, unlike LeveragedBacktestAgent, because the
lag is already baked into these forecasts).

Deterministic MA-slope/ADX chosen over an HMM regime-switching model to
avoid a second new ML dependency and keep regime detection exactly
reproducible for walk-forward review — consistent with this repo's existing
signal (agents/timing/timing_signal_agent.py), which is also rule-based.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing is itself an EMA with alpha = 1/period."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def conditional_mu(returns: pd.Series, decay: float = 0.94) -> pd.Series:
    """EWMA one-step-ahead forecast of expected daily return (conditional
    mu), NOT a long-run unconditional sample mean — locally weighted so it
    adapts to the current regime instead of converging glacially slowly.
    """
    contemporaneous = returns.astype(float).ewm(alpha=1 - decay, adjust=False).mean()
    return contemporaneous.shift(1)


def average_directional_index(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
) -> pd.Series:
    """Wilder's ADX — a same-day (non-forecast) technical indicator; the
    forecast-for-t shift is applied by classify_regime, not here, so this
    remains a reusable raw building block."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index,
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1,
    ).max(axis=1)

    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return _wilder_smooth(dx.fillna(0), period)


def classify_regime(
    high: pd.Series, low: pd.Series, close: pd.Series,
    adx_period: int = 14, adx_threshold: float = 25.0,
) -> pd.Series:
    """Forecast-for-t regime label: "trending" if ADX (computed through
    t-1) >= adx_threshold, else "choppy". Object-dtype Series with None for
    the unresolvable warmup region.
    """
    adx = average_directional_index(high, low, close, period=adx_period)
    raw_label = pd.Series(
        np.where(adx >= adx_threshold, "trending", "choppy"), index=close.index, dtype=object,
    )
    raw_label[adx.isna()] = None
    return raw_label.shift(1)
