"""Pre-market bars to the traded basis, the reused pre-market features, and the top-n rank.

Conversion uses the prior session's factors only (``f_{D-1}`` for prices, the
split-only ``s_{D-1}`` for share volume, decision D1); the session's own daily
row is never read here.
"""
from __future__ import annotations

import pandas as pd

from research.penny.features.premarket import premarket_features

from research.intraday.data.basis import factor_consistent, to_traded_basis

RANKABLE_FEATURES: tuple[str, ...] = (
    "pm_dollar_volume", "pm_volume", "pm_bars", "pm_vwap", "pm_close", "pm_high", "pm_low",
    "pm_range", "pm_close_position", "pm_dist_from_vwap", "pm_late_dv_share",
    "pm_early_dv_share", "pm_vol_centroid", "pm_late_drift", "pm_first_minute",
    "pm_minutes_traded", "pm_gap", "pm_high_gap", "pm_fade_from_high",
)


def convert_premarket(pm: pd.DataFrame, prior: pd.DataFrame,
                      tol: float) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Traded-basis pre-market rows for symbols with a trustworthy prior factor.

    Returns the converted rows and the sorted symbols dropped as unconvertible
    (pre-market bars present, but no finite ``s`` or ``f_v`` disagreeing with ``f``).
    """
    s_ok = prior["s"].notna() & (prior["s"] > 0)
    trusted = prior.index[(s_ok & factor_consistent(prior["f_v"], prior["f"], tol)).to_numpy()]
    in_prior = pm["symbol"].isin(prior.index)
    ok = pm["symbol"].isin(trusted)
    dropped = tuple(sorted(set(pm.loc[in_prior & ~ok, "symbol"].astype(str))))
    rows = pm.loc[ok].reset_index(drop=True)
    return to_traded_basis(rows, prior["f"], prior["s"]), dropped


def rank_symbols(universe: tuple[str, ...], pm_traded: pd.DataFrame,
                 prev_close_unadj: pd.Series, rank_by: tuple[str, ...],
                 top_n: int) -> tuple[tuple[str, ...], int]:
    """Top ``top_n`` of ``universe`` by ``rank_by`` (descending, ties by symbol), sorted.

    Also returns the count of universe symbols with no pre-market bars.
    """
    panel = pm_traded[pm_traded["symbol"].isin(universe)]
    if panel.empty:
        return (), len(universe)
    feats = premarket_features(panel, prev_close=prev_close_unadj)
    n_missing = len(universe) - int(feats["symbol"].nunique())
    ordered = feats.sort_values([*rank_by, "symbol"],
                                ascending=[False] * len(rank_by) + [True],
                                na_position="last", kind="mergesort")
    selected = tuple(sorted(str(s) for s in ordered["symbol"].head(top_n)))
    return selected, n_missing
