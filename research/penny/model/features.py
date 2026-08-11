"""features.py — the model's feature matrix.

What changed from round 1
-------------------------
Round 1 discretised twelve continuous, mutually correlated signals into
independent binary thresholds and searched over their subsets. That threw away
almost all the information: ``obs_range`` carries an IC of -0.18 against the
barrier-free return, but "is obs_range above 0.20" carries a fraction of it,
and combining several such binaries adds little because they all proxy the same
latent quantity -- how extreme the name is.

Here the features stay continuous, and two genuinely new kinds are added:

**Cross-sectional ranks.** A 30% gap means something different on a day with
three candidates than on a day with forty. Absolute thresholds cannot express
that; a within-day rank can. This also makes the feature stationary across the
regime shift documented in the log (3.5 candidates/session pre-2020 versus 15.5
in 2025-26).

**Trailing regime state.** Computed from a rolling window that ends *before*
the current session, never from the day itself -- otherwise the model would
know how busy today was before deciding whether to trade it.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Continuous features carried straight through. Every one is trailing/shifted
# or uses only bars strictly before the 09:35 decision.
BASE_FEATURES: List[str] = [
    "gap",
    "rvol_prev",
    "adv_5",
    "adv_20",
    "adv_60",
    "max_ret_20d",
    "vol_20d",
    "hl_range_20d",
    "consec_up",
    "ret_1d_prev",
    "ret_5d_prev",
    "close_unadj_prev",
    "split_factor_180d",
    # Opening-window observations (09:30-09:34 only)
    "obs_return",
    "obs_range",
    "obs_close_position",
    "obs_dollar_volume",
    "minutes_obs",
    "trades_obs",
]

# Ranked within each session. Chosen because their meaning is relative: being
# the biggest gap of the day is a different fact from gapping 30%.
RANK_FEATURES: List[str] = [
    "gap",
    "obs_dollar_volume",
    "obs_range",
    "rvol_prev",
    "adv_20",
    "obs_return",
]

EDGAR_FEATURES: List[str] = [
    "edgar_days_since_shelf",
    "edgar_days_since_takedown",
    "edgar_deficiency_180d",
    "edgar_dilutive_8k_180d",
    "edgar_filing_count_180d",
    "edgar_has_coverage",
]


def add_cross_sectional_ranks(
    df: pd.DataFrame, columns: Sequence[str] = RANK_FEATURES, date_col: str = "date"
) -> pd.DataFrame:
    """Within-day percentile rank of each named column.

    Uses only same-session information, all of which is already known at 09:35,
    so this introduces no look-ahead: the ranks are over *candidates*, and
    candidacy is itself determined by 09:30 information.
    """
    out = df.copy()
    grouped = out.groupby(date_col, sort=False)
    for col in columns:
        if col not in out.columns:
            continue
        out[f"xs_rank_{col}"] = grouped[col].rank(pct=True, method="average")
    # How crowded the session is -- known at 09:35, since candidacy is fixed at
    # the open. Distinct from the trailing regime features below.
    out["xs_n_candidates"] = grouped[date_col].transform("size")
    return out


def add_regime_features(
    df: pd.DataFrame, windows: Sequence[int] = (20, 60), date_col: str = "date"
) -> pd.DataFrame:
    """Trailing measures of how active the microcap tape has been.

    Strictly trailing: each window is shifted by one session, so a decision
    never sees the day it is being made on. Without the shift these would
    encode today's candidate count, which is knowable -- but the trailing
    version is what generalises, and mixing the two would blur the distinction.
    """
    out = df.copy()

    per_day = (
        out.groupby(date_col)
        .agg(n_candidates=("symbol", "size"), mean_gap=("gap", "mean"))
        .sort_index()
    )
    for window in windows:
        per_day[f"regime_n_cand_{window}d"] = (
            per_day["n_candidates"].shift(1).rolling(window, min_periods=max(5, window // 4)).mean()
        )
        per_day[f"regime_gap_{window}d"] = (
            per_day["mean_gap"].shift(1).rolling(window, min_periods=max(5, window // 4)).mean()
        )

    regime_cols = [c for c in per_day.columns if c.startswith("regime_")]
    return out.merge(
        per_day[regime_cols], left_on=date_col, right_index=True, how="left"
    )


def build_matrix(
    trades: pd.DataFrame,
    edgar: Optional[pd.DataFrame] = None,
    include_edgar: bool = True,
) -> pd.DataFrame:
    """Assemble the full feature matrix."""
    df = trades.copy()

    if include_edgar and edgar is not None and not edgar.empty:
        before = len(df)
        df = df.merge(edgar, on=["symbol", "date"], how="left", validate="one_to_one")
        assert len(df) == before, "EDGAR merge changed the row count"
        coverage = df["edgar_has_coverage"].fillna(False).mean()
        logger.info("edgar coverage: %.1f%% of trades", 100 * coverage)

    df = add_cross_sectional_ranks(df)
    df = add_regime_features(df)
    return df


def feature_columns(df: pd.DataFrame, include_edgar: bool = True) -> List[str]:
    """The columns the model may see.

    An explicit allowlist rather than "everything numeric". The trade table
    also carries outcome-derived columns (``net_return``, ``outcome``,
    ``mfe_full``) and diagnostic ones prefixed ``fut_``/``diag_``; a
    dtype-based selection would sweep those in and produce a spectacular,
    entirely fake result.
    """
    cols = [c for c in BASE_FEATURES if c in df.columns]
    cols += [c for c in df.columns if c.startswith("xs_rank_")]
    cols += ["xs_n_candidates"] if "xs_n_candidates" in df.columns else []
    cols += [c for c in df.columns if c.startswith("regime_")]
    if include_edgar:
        cols += [c for c in EDGAR_FEATURES if c in df.columns]

    banned = _forbidden_columns(df)
    leaked = sorted(set(cols) & banned)
    assert not leaked, f"outcome-derived columns reached the feature list: {leaked}"
    return cols


def _forbidden_columns(df: pd.DataFrame) -> set:
    """Anything that encodes the outcome or the future."""
    explicit = {
        "net_return", "gross_return", "outcome", "exit_price", "exit_minute",
        "holding_minutes", "max_favourable", "max_adverse", "cost_bps",
        "entry_price_filled", "ret_c2c", "mfe_full", "mae_full",
        "minutes_to_mfe", "minutes_to_mae", "mfe_before_mae", "n_scan_minutes",
        "coverage", "halt_suspect", "ambiguous_bar", "meta_label",
    }
    prefixed = {c for c in df.columns if c.startswith(("fut_", "diag_"))}
    return explicit | prefixed
