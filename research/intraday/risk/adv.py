"""Average daily share volume on the session's traded basis.

Controller ruling P2-5 (design docs updated in e09aed7 on research/intraday): Alpaca
``adjustment="all"`` adjusts share volume for splits only, not dividends, so the conversion uses
the split-only factor ``s = volume / volume_unadj``. Each prior session's raw volume is moved to
D's basis by ``volume_unadj * s_day / s_{D-1}`` (equivalently ``volume / s_{D-1}``), so a split
inside the window does not mix two share bases. D's own row never enters.
"""
from __future__ import annotations

from datetime import date

import math

import pandas as pd

_REQUIRED: tuple[str, ...] = ("date", "volume", "volume_unadj")


def _prior_window(daily: pd.DataFrame, session_date: date, window: int) -> pd.DataFrame:
    missing = [c for c in _REQUIRED if c not in daily.columns]
    if missing:
        raise ValueError(f"adv_shares needs columns {missing} (split-only factor s = volume / volume_unadj)")
    if "symbol" in daily.columns and daily["symbol"].nunique(dropna=False) > 1:
        raise ValueError("adv_shares takes one symbol's rows, got several symbols")
    dates = pd.to_datetime(daily["date"])
    prior = daily.loc[dates < pd.Timestamp(session_date)].assign(_d=dates).sort_values("_d")
    if len(prior) < window:
        raise ValueError(f"adv_shares needs {window} prior sessions before {session_date}, got {len(prior)}")
    return prior.iloc[-window:]


def adv_shares(daily: pd.DataFrame, session_date: date, window: int = 20) -> float:
    """Mean of the last `window` prior sessions' share volume, on `session_date`'s traded basis."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    w = _prior_window(daily, session_date, window)
    vol = w["volume"].astype(float)
    raw = w["volume_unadj"].astype(float)
    s = vol / raw.where(raw > 0)
    if not (s.notna().all() and (s > 0).all() and (s < math.inf).all()):
        raise ValueError("adv_shares: split factor volume / volume_unadj must be finite and > 0 on every prior session")
    on_d_basis = raw * s / float(s.iloc[-1])
    out = float(on_d_basis.mean())
    assert math.isfinite(out) and out >= 0
    return out
