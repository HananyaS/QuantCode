"""candidates.py — which symbol-days enter the study at all.

The candidate rule defines the population, and therefore defines what any
result is a statement *about*. It is not a filter applied for convenience: it
is the denominator. Reporting "E[net return] > 0" without saying which
population was screened is meaningless.

Every condition here must be knowable at 09:30 ET, before the 09:35 decision:

* ``gap`` uses the opening print against the previous close -- available at 09:30.
* ``adv_20``, ``rvol_prev``, ``max_ret_20d`` are trailing and shifted by one day.
* the price screen uses the **previous** unadjusted close, not today's.

The rule intentionally selects "in play" names -- a large opening gap on a
liquid-enough microcap -- because that is the population the user actually
observes exploding, and because a study over all microcap-days would be
dominated by names that never move and would answer a different question.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CandidateSpec:
    """The screen defining the studied population."""

    min_gap: float = 0.10             # opening gap of at least +10%
    min_price: float = 0.50
    max_price: float = 10.00
    min_adv_20: float = 100_000.0     # trailing 20-day dollar volume
    min_prior_bars: int = 30          # enough history for trailing features
    max_per_day: Optional[int] = None # cap for tractability; None = no cap


def build_candidates(panel: pd.DataFrame, spec: CandidateSpec) -> pd.DataFrame:
    """Select the symbol-days that enter the study.

    ``panel`` must already carry the cross-day features from
    :func:`research.penny.etl.bars.add_cross_day_features` plus the unadjusted
    price columns.
    """
    df = panel.sort_values(["symbol", "date"]).copy()
    grp = df.groupby("symbol", sort=False)

    # Previous unadjusted close: the price screen must not use today's close,
    # which is not known at 09:30, nor the adjusted close, which misstates what
    # the stock actually traded at.
    df["close_unadj_prev"] = grp["close_unadj"].shift(1)
    df["bar_index"] = grp.cumcount()

    conditions = {
        "has_history": df["bar_index"] >= spec.min_prior_bars,
        "price_in_range": df["close_unadj_prev"].between(spec.min_price, spec.max_price),
        "liquid_enough": df["adv_20"] >= spec.min_adv_20,
        "gapped_up": df["gap"] >= spec.min_gap,
        "finite": np.isfinite(df["gap"]) & df["adv_20"].notna(),
    }

    mask = pd.Series(True, index=df.index)
    counts = {}
    for name, cond in conditions.items():
        mask &= cond.fillna(False)
        counts[name] = int(mask.sum())

    logger.info("candidate screen (cumulative):")
    for name, n in counts.items():
        logger.info("  after %-16s %8d rows", name, n)

    out = df[mask].copy()

    if spec.max_per_day is not None:
        # Rank by dollar volume so the cap keeps the most tradable names rather
        # than an arbitrary alphabetical slice.
        out = (
            out.sort_values(["date", "adv_20"], ascending=[True, False])
            .groupby("date", group_keys=False)
            .head(spec.max_per_day)
        )
        logger.info("  after max_per_day=%s  %8d rows", spec.max_per_day, len(out))

    logger.info(
        "candidates: %d symbol-days over %d sessions (%.1f/day), %d unique symbols",
        len(out), out["date"].nunique(),
        len(out) / max(out["date"].nunique(), 1), out["symbol"].nunique(),
    )
    return out.reset_index(drop=True)


def add_delisting_distance(candidates: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Days from each candidate date to that symbol's final observed bar.

    Feeds the ``veto_delisted_soon`` DIAGNOSTIC only. It uses future
    information by construction and must never enter a tradable result; its
    purpose is to bound how much of any measured edge is simply the avoidance
    of terminal names, which a live system would need filings to detect.
    """
    last_bar = panel.groupby("symbol")["date"].max().rename("last_bar")
    out = candidates.merge(last_bar, on="symbol", how="left")
    out["days_to_last_bar"] = (out["last_bar"] - out["date"]).dt.days
    return out
