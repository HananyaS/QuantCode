"""Adjusted -> traded basis: factors, conversion and split flags (pure, vectorised).

The caches were fetched with Alpaca ``adjustment="all"``. That adjusts **prices**
for splits and dividends but **share volume only for splits** (decision D1 of
the phase-2 data plan). Measured on the caches, 2024-03-04: BRSP has
``f = close_unadj / close = 1.345`` (cumulative dividends) while its adjusted
volume equals its raw volume (1,034,853); ENVB has ``volume / volume_unadj =
0.00556`` (a 1:180 reverse split). Dividing volume by ``f`` misstates raw volume
by more than 2 % on 35-58 % of daily symbol-days. So:

* prices: ``raw = adjusted * f`` with ``f = close_unadj / close`` (``price_factor``);
* share volume: ``raw = adjusted / s`` with ``s = volume / volume_unadj``, the
  split-only factor (``volume_factor``).

The split flag reads the step of the price factor between consecutive days,
as the causal event factor of ``research/penny/etl/bars.py:226-238`` does (that
file uses the reciprocal ``close / close_unadj``; ``abs(log)`` of the step is the
same either way).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "vwap")


def _ratio(num: pd.Series, den: pd.Series, name: str) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").astype(float)
    den = pd.to_numeric(den, errors="coerce").astype(float)
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (num / den).where(ok)
    return out.rename(name)


def price_factor(close_unadj: pd.Series, close: pd.Series) -> pd.Series:
    """``f = close_unadj / close``; NaN where either is non-finite or <= 0."""
    return _ratio(close_unadj, close, "f")


def volume_factor(volume: pd.Series, volume_unadj: pd.Series) -> pd.Series:
    """``s = volume / volume_unadj`` (adjusted / raw, split-only); NaN where invalid."""
    return _ratio(volume, volume_unadj, "s")


def factor_consistent(a: pd.Series, b: pd.Series, tol: float) -> pd.Series:
    """``abs(a / b - 1) <= tol`` on ``a``'s index; NaN or missing -> False."""
    b_al = b.reindex(a.index).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = (a.astype(float) / b_al - 1.0).abs()
    return (dev <= tol).fillna(False).astype(bool)


def split_flags(f_prev: pd.Series, f_session: pd.Series, threshold: float) -> pd.Series:
    """``abs(log(f_prev / f_session)) > threshold`` on ``f_prev``'s index.

    A symbol missing from ``f_session`` or with a NaN factor is not flagged:
    if the session row is missing, no split is known.
    """
    f_d = f_session.reindex(f_prev.index).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.log(f_prev.astype(float) / f_d).abs()
    return (step > threshold).fillna(False).astype(bool)


def to_traded_basis(bars: pd.DataFrame, f: pd.Series, s: pd.Series) -> pd.DataFrame:
    """Prices x ``f[symbol]``, volume / ``s[symbol]`` (float, unrounded) on a copy of ``bars``."""
    fr = bars["symbol"].map(f).astype(float)
    sr = bars["symbol"].map(s).astype(float)
    bad = ~(np.isfinite(fr) & np.isfinite(sr) & (fr > 0) & (sr > 0))
    if bad.any():
        missing = sorted(bars.loc[bad, "symbol"].unique())
        raise ValueError(f"no finite price/volume factor for symbols {missing}")
    out = bars.copy()
    for col in PRICE_COLUMNS:
        out[col] = out[col].astype(float) * fr
    out["volume"] = out["volume"].astype(float) / sr
    return out
