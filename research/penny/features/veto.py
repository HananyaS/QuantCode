"""veto.py — candidate veto conditions, each with a stated prior.

The design principle
--------------------
A veto is a *negative* condition: when it is present, the trade is rejected.
This is the asymmetry the whole study rests on. Predicting which microcap will
explode is very hard. Recognising which ones are structurally poisoned is much
easier, and the published literature hands over most of the conditions for
free.

Every veto below is registered with the claim it encodes and the source it
comes from. A veto with no prior is a fishing expedition dressed up as a
hypothesis: it would still be searched over, and would still inflate the
multiple-testing correction for everything else.

Sources
-------
* Bali, Cakici & Whitelaw (2011), "Maxing Out: Stocks as Lotteries and the
  Cross-Section of Expected Returns", *JFE* 99(2) -- extreme recent maximum
  daily return predicts LOWER subsequent returns.
* Eraker & Ready (2015), "Do investors overpay for stocks with lottery-like
  payoffs?", *JFE* 115(3) -- OTC/microcap returns are negative on average and
  violently right-skewed.
* Avramov, Chordia & Goyal (2006), "Liquidity and Autocorrelations in
  Individual Stock Returns", *JF* -- reversals are largest in high-turnover,
  low-liquidity names, and vanish net of costs.
* Frieder & Zittrain (2007), "Spam Works" -- buying into a promotion loses
  ~5.5% over two days.
* Hou, Xue & Zhang (2020), "Replicating Anomalies", *RFS* -- microcap effects
  are largely untradeable; illiquidity is the mechanism, not a nuisance.
* Corwin & Schultz (2012), *JF* 67(2) -- high-low range as a spread proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class VetoDefinition:
    """One candidate veto: what it flags, why, and where the claim comes from."""

    name: str
    claim: str
    source: str
    family: str
    build: Callable[[pd.DataFrame, Dict], pd.Series]

    def apply(self, df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
        return self.build(df, params or {}).fillna(False).astype(bool)


# --------------------------------------------------------------------------
# Family: lottery / overextension
# --------------------------------------------------------------------------


def _veto_high_max(df: pd.DataFrame, p: Dict) -> pd.Series:
    threshold = p.get("max_ret_threshold", 0.50)
    return df["max_ret_20d"] > threshold


def _veto_extended_run(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df["consec_up"] >= p.get("consec_up_threshold", 3)


def _veto_huge_gap(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df["gap"] > p.get("gap_max", 1.00)


def _veto_prior_day_spike(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df["ret_1d_prev"] > p.get("prior_spike", 0.50)


# --------------------------------------------------------------------------
# Family: liquidity / executability
# --------------------------------------------------------------------------


def _veto_thin_adv(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df["adv_20"] < p.get("min_adv", 1_000_000)


def _veto_wide_spread(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df["hl_range_20d"] > p.get("max_hl_range", 0.20)


def _veto_sub_dollar(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df["close_unadj_prev"] < p.get("min_price", 1.00)


def _veto_thin_open(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df["obs_dollar_volume"] < p.get("min_open_dollar", 250_000)


def _veto_sparse_prints(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df["minutes_obs"] < p.get("min_minutes_obs", 5)


# --------------------------------------------------------------------------
# Family: structure / dilution risk
# --------------------------------------------------------------------------


def _veto_reverse_split_recent(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df.get("reverse_split_180d", pd.Series(False, index=df.index)).astype(bool)


def _veto_extreme_adjustment(df: pd.DataFrame, p: Dict) -> pd.Series:
    """Large adjusted/unadjusted divergence implies heavy split activity.

    A cheap structural proxy that needs no filings: a name whose adjusted and
    raw prices diverge by a large factor has been repeatedly split, which in
    this universe almost always means reverse splits to maintain listing.
    """
    ratio = df["close"] / df["close_unadj"].replace(0, np.nan)
    return (ratio > p.get("max_adj_ratio", 2.0)) | (ratio < 1 / p.get("max_adj_ratio", 2.0))


def _veto_delisted_soon(df: pd.DataFrame, p: Dict) -> pd.Series:
    """Flags trades within N days of the symbol's final observed bar.

    NOT usable in production -- it peeks at the future -- and is included ONLY
    as an upper-bound diagnostic: it measures how much of any edge comes from
    avoiding terminal names, which a live system would have to detect from
    filings instead. Excluded from the tradable search by default.
    """
    horizon = p.get("delist_horizon", 60)
    return (df["days_to_last_bar"] <= horizon) & df["days_to_last_bar"].notna()


# --------------------------------------------------------------------------
# Family: short-side pressure
# --------------------------------------------------------------------------


def _veto_high_short_volume(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df.get("short_volume_ratio", pd.Series(np.nan, index=df.index)) > p.get(
        "max_short_ratio", 0.60
    )


def _veto_low_short_volume(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df.get("short_volume_ratio", pd.Series(np.nan, index=df.index)) < p.get(
        "min_short_ratio", 0.20
    )


# --------------------------------------------------------------------------
# Family: opening-window behaviour
# --------------------------------------------------------------------------


def _veto_weak_open(df: pd.DataFrame, p: Dict) -> pd.Series:
    """Fading in the opening window: closing near the low of the first 5 min."""
    return df["obs_close_position"] < p.get("min_close_position", 0.35)


def _veto_open_reversal(df: pd.DataFrame, p: Dict) -> pd.Series:
    return df["obs_return"] < p.get("min_obs_return", -0.02)


REGISTRY: List[VetoDefinition] = [
    VetoDefinition(
        "veto_high_max", "A large trailing max daily return predicts lower forward returns.",
        "Bali, Cakici & Whitelaw (2011), JFE 99(2)", "lottery", _veto_high_max,
    ),
    VetoDefinition(
        "veto_extended_run", "Multi-day consecutive advances are mean-reverting in microcaps.",
        "Avramov, Chordia & Goyal (2006), JF", "lottery", _veto_extended_run,
    ),
    VetoDefinition(
        "veto_huge_gap", "Extreme opening gaps are dominated by promotion and dilution supply.",
        "Frieder & Zittrain (2007), Hastings Comm. & Ent. L.J.", "lottery", _veto_huge_gap,
    ),
    VetoDefinition(
        "veto_prior_day_spike", "Buying the day after a large spike is the classic MAX trap.",
        "Bali, Cakici & Whitelaw (2011)", "lottery", _veto_prior_day_spike,
    ),
    VetoDefinition(
        "veto_thin_adv", "Below a dollar-volume floor, costs exceed any edge.",
        "Novy-Marx & Velikov (2016), RFS 29(1)", "liquidity", _veto_thin_adv,
    ),
    VetoDefinition(
        "veto_wide_spread", "A wide high-low range proxies a spread that eats the trade.",
        "Corwin & Schultz (2012), JF 67(2)", "liquidity", _veto_wide_spread,
    ),
    VetoDefinition(
        "veto_sub_dollar", "Sub-$1 names carry delisting risk and the widest relative spreads.",
        "SEC DERA (2016), Outcomes of Investing in OTC Stocks", "liquidity", _veto_sub_dollar,
    ),
    VetoDefinition(
        "veto_thin_open", "A thin opening window means the entry price is not attainable at size.",
        "Hou, Xue & Zhang (2020), RFS", "liquidity", _veto_thin_open,
    ),
    VetoDefinition(
        "veto_sparse_prints", "Few prints in the opening window means no reliable fill.",
        "Executability; measured directly in this study", "liquidity", _veto_sparse_prints,
    ),
    VetoDefinition(
        "veto_extreme_adjustment", "Heavy split activity marks recycled, serially diluted shells.",
        "Hillion & Vermaelen (2004), JFE 71(2), death-spiral financing", "structure",
        _veto_extreme_adjustment,
    ),
    VetoDefinition(
        "veto_high_short_volume", "Very high short volume signals informed downside pressure.",
        "Boehmer, Jones & Zhang (2008), JF 63(2)", "short", _veto_high_short_volume,
    ),
    VetoDefinition(
        "veto_low_short_volume", "Very low short volume can mark a no-borrow, unshortable squeeze.",
        "Beneish, Lee & Nichols (2015), JAE 60(1)", "short", _veto_low_short_volume,
    ),
    VetoDefinition(
        "veto_weak_open", "Closing near the low of the opening range marks a failed continuation.",
        "Opening-range convention; tested here, not assumed", "opening", _veto_weak_open,
    ),
    VetoDefinition(
        "veto_open_reversal", "A negative opening window contradicts the continuation premise.",
        "Opening-range convention; tested here, not assumed", "opening", _veto_open_reversal,
    ),
]

# Peeks at the future. Diagnostic only -- never included in a tradable result.
LOOKAHEAD_DIAGNOSTICS: List[VetoDefinition] = [
    VetoDefinition(
        "veto_delisted_soon",
        "Upper bound on how much edge comes purely from avoiding terminal names.",
        "Diagnostic only -- uses future information", "diagnostic", _veto_delisted_soon,
    ),
]


def build_all(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
    include_diagnostics: bool = False,
) -> pd.DataFrame:
    """Attach every registered veto column to ``df``."""
    out = df.copy()
    definitions = list(REGISTRY) + (LOOKAHEAD_DIAGNOSTICS if include_diagnostics else [])
    for definition in definitions:
        try:
            out[definition.name] = definition.apply(out, params)
        except KeyError as exc:
            # A missing input is a real gap, not something to silently skip.
            out[definition.name] = False
            out.attrs.setdefault("veto_build_errors", {})[definition.name] = str(exc)
    return out


def registry_frame(include_diagnostics: bool = False) -> pd.DataFrame:
    """The registry as a documentation table."""
    definitions = list(REGISTRY) + (LOOKAHEAD_DIAGNOSTICS if include_diagnostics else [])
    return pd.DataFrame(
        [
            {"veto": d.name, "family": d.family, "claim": d.claim, "source": d.source}
            for d in definitions
        ]
    )


def veto_names(include_diagnostics: bool = False) -> List[str]:
    definitions = list(REGISTRY) + (LOOKAHEAD_DIAGNOSTICS if include_diagnostics else [])
    return [d.name for d in definitions]
