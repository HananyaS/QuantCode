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
    # Pre-open activity is strongly relative: being the most-traded name before
    # the bell is a different fact from trading $2m before the bell.
    "pm_dollar_volume",
    "pm_dv_to_adv",
    "pm_gap",
]

# Continuous only. Round 2 measured the binary flags at -0.64% against +1.05%
# for the same information as days-since, so the booleans are deliberately
# excluded rather than merely deprioritised.
EDGAR_FEATURES: List[str] = [
    "edgar_days_since_shelf",
    "edgar_days_since_takedown",
    "edgar_days_since_deficiency",
    "edgar_days_since_dilutive_8k",
    "edgar_takedown_count_180d",
    "edgar_filing_count_180d",
]

# FINRA short volume, lagged one session (the file publishes after the close).
SHORT_FEATURES: List[str] = [
    "short_volume_ratio",
    "short_ratio_chg_5d",
    "short_ratio_chg_20d",
]

# Float proxy from XBRL, dated by the filing that disclosed it.
FLOAT_FEATURES: List[str] = [
    "shares_outstanding",
    "shares_growth_qoq",
    "shares_growth_yoy",
    "float_dollar",
    "gap_to_float",
]

# Round 4: microstructure state at the DECISION instant, from the 09:34 window.
#
# Rounds 1-3 banned every quote-derived column outright. That was hygiene rather
# than a leakage argument -- the module measuring them says plainly that the
# NBBO is "directly observable from the decision instant" -- and it had a real
# cost: the model could not learn to avoid expensive names, while cost turned
# out to vary 6.3x across the candidate universe.
#
# The distinction that makes these admissible is the window, not the quantity.
# The 09:34 book is knowable before entering. The 09:35 entry book and the 15:55
# exit book are not, and stay forbidden below.
COST_FEATURES: List[str] = [
    "half_spread_bps_decision",
    "log_inside_notional_decision",
    "depth_ratio_decision",
    "cost_hat_bps",
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
    premarket: Optional[pd.DataFrame] = None,
    short_volume: Optional[pd.DataFrame] = None,
    float_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble the full feature matrix.

    Every join is validated one-to-one. A silent fan-out would duplicate trades
    and inflate every subsequent statistic without raising anything.
    """
    df = trades.copy()

    sources = [
        ("edgar", edgar if include_edgar else None, "edgar_has_coverage"),
        ("premarket", premarket, None),
        ("short volume", short_volume, "short_has_coverage"),
        ("float", float_data, "shares_has_coverage"),
    ]

    for name, source, coverage_col in sources:
        if source is None or source.empty:
            if source is not None:
                logger.warning("%s: empty, skipping", name)
            continue
        before = len(df)
        overlap = (set(source.columns) & set(df.columns)) - {"symbol", "date"}
        df = df.merge(
            source.drop(columns=list(overlap)), on=["symbol", "date"],
            how="left", validate="one_to_one",
        )
        assert len(df) == before, f"{name} merge changed the row count"
        if coverage_col and coverage_col in df.columns:
            logger.info("%s coverage: %.1f%% of trades",
                        name, 100 * df[coverage_col].fillna(False).mean())
        else:
            probe = [c for c in source.columns if c not in ("symbol", "date")]
            if probe:
                logger.info("%s coverage: %.1f%% of trades",
                            name, 100 * df[probe[0]].notna().mean())

    df = add_cross_sectional_ranks(df)
    df = add_regime_features(df)
    return df


def feature_columns(
    df: pd.DataFrame,
    include_edgar: bool = True,
    include_premarket: bool = True,
    include_short: bool = True,
    include_float: bool = True,
    include_cost: bool = False,
) -> List[str]:
    """The columns the model may see.

    An explicit allowlist rather than "everything numeric". The trade table
    also carries outcome-derived columns (``net_return``, ``outcome``,
    ``mfe_full``) and diagnostic ones prefixed ``fut_``/``diag_``; a
    dtype-based selection would sweep those in and produce a spectacular,
    entirely fake result.
    """
    from research.penny.features.premarket import PREMARKET_FEATURES

    cols = [c for c in BASE_FEATURES if c in df.columns]
    cols += [c for c in df.columns if c.startswith("xs_rank_")]
    cols += ["xs_n_candidates"] if "xs_n_candidates" in df.columns else []
    cols += [c for c in df.columns if c.startswith("regime_")]
    if include_edgar:
        cols += [c for c in EDGAR_FEATURES if c in df.columns]
    if include_premarket:
        cols += [c for c in PREMARKET_FEATURES if c in df.columns]
    if include_short:
        cols += [c for c in SHORT_FEATURES if c in df.columns]
    if include_float:
        cols += [c for c in FLOAT_FEATURES if c in df.columns]
    if include_cost:
        cols += [c for c in COST_FEATURES if c in df.columns]

    cols = list(dict.fromkeys(cols))
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
        "net_at_cost", "model_score", "measured_cost", "measured_round_trip_bps",
        "seq_score", "fold", "cost_2000", "cost_5000", "cost_10000",
        # Round-3 quote columns: measured over [09:35:00, 09:36:00), i.e. from
        # the entry minute itself. Admissible as a charge, never as an input.
        "half_spread_bps", "ask_notional", "bid_notional",
        # Round-4 realised cost. `cost_realized` is what the trade actually
        # paid, measured at entry and at 15:55; a model that saw it would know
        # how the day's liquidity turned out. Its decision-time twin,
        # `cost_hat_bps` from the 09:34 book, is allowed -- see COST_FEATURES.
        "half_spread_bps_entry", "ask_notional_entry", "bid_notional_entry",
        "half_spread_bps_exit", "ask_notional_exit", "bid_notional_exit",
        "entry_leg_bps", "exit_leg_bps", "round_trip_bps",
        "cost_realized", "cost_realized_bps", "p_calib", "ev",
        "payoff_win_mean", "payoff_loss_mean",
        # Coverage flags are booleans about data availability, not signals.
        # Left out deliberately: a model could otherwise learn "symbols the SEC
        # has no record of behave differently", which is a statement about the
        # dataset rather than the market.
        "edgar_has_coverage", "short_has_coverage", "shares_has_coverage",
        # Raw pre-open levels the scale-free features already encode; keeping
        # them would let the model key on absolute price.
        "pm_close", "pm_high", "pm_low", "pm_vwap", "pm_volume",
    }
    prefixed = {c for c in df.columns if c.startswith(("fut_", "diag_"))}
    return explicit | prefixed
