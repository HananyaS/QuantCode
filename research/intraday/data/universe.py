"""Causal universe filter on daily rows strictly before the session, and the cache intersection.

Every level is on the traded basis: the price filter reads ``close_unadj`` of the
prior session; ADV is the median of ``close_unadj x volume_unadj_v`` (raw x raw).
The split flag reads only the factor step ``f_{D-1} / f_D`` (decision in screener.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from research.intraday.data.basis import price_factor, split_flags, volume_factor
from research.intraday.data.errors import LookAheadError, MissingDataError


@dataclass(frozen=True)
class UniverseResult:
    symbols: tuple[str, ...]
    prior: pd.DataFrame  # index symbol; close_unadj, f, f_v, s of the prior_date row
    prior_date: date
    n_split_excluded: int


def _prior_rows(daily: pd.DataFrame, prior_ts: pd.Timestamp) -> pd.DataFrame:
    """Each symbol's last row, kept only if it is on the market's prior session (D4)."""
    last = daily.sort_values(["symbol", "date"], kind="mergesort").groupby("symbol").tail(1)
    return last[last["date"] == prior_ts].set_index("symbol")


def _adv(daily: pd.DataFrame, adv_window: int) -> pd.Series:
    """Median raw dollar volume of the last ``adv_window`` rows; NaN unless all are finite."""
    ordered = daily.sort_values(["symbol", "date"], kind="mergesort")
    tail = ordered.groupby("symbol").tail(adv_window)
    dv = (tail["close_unadj"] * tail["volume_unadj_v"]).where(lambda x: np.isfinite(x))
    agg = dv.groupby(tail["symbol"]).agg(["median", "count"])
    return agg["median"].where(agg["count"] >= adv_window)


def causal_universe(daily: pd.DataFrame, session_date: date, f_session: pd.Series, *,
                    min_price: float, max_price: float, min_adv_usd: float, adv_window: int,
                    exclude_split_days: bool, split_log_step_threshold: float,
                    max_prior_gap_days: int) -> UniverseResult:
    """Symbols tradeable on ``session_date`` from daily rows strictly before it."""
    d_ts = pd.Timestamp(session_date)
    if (daily["date"] >= d_ts).any():
        raise LookAheadError(f"daily rows on or after the session date {session_date}")
    if daily.empty:
        raise MissingDataError(f"no daily rows before {session_date}")
    prior_ts = daily["date"].max()
    if prior_ts.date() < session_date - timedelta(days=max_prior_gap_days):
        raise MissingDataError(
            f"last daily row {prior_ts.date()} is more than {max_prior_gap_days} days "
            f"before {session_date}: the daily cache is missing")
    prior = _prior_rows(daily, prior_ts)
    price_ok = (prior["close_unadj"] >= min_price) & (prior["close_unadj"] <= max_price)
    adv = _adv(daily, adv_window).reindex(prior.index)
    keep = price_ok & (adv >= min_adv_usd)
    out = pd.DataFrame({
        "close_unadj": prior["close_unadj"].astype(float),
        "f": price_factor(prior["close_unadj"], prior["close"]),
        "f_v": price_factor(prior["close_unadj_v"], prior["close_v"]),
        "s": volume_factor(prior["volume_v"], prior["volume_unadj_v"]),
    })[keep.to_numpy()]
    n_split = 0
    if exclude_split_days:
        flags = split_flags(out["f"], f_session, split_log_step_threshold)
        n_split = int(flags.sum())
        out = out[~flags.to_numpy()]
    out = out.sort_index()
    return UniverseResult(symbols=tuple(str(s) for s in out.index), prior=out,
                          prior_date=prior_ts.date(), n_split_excluded=n_split)


def intersect_cache(symbols: tuple[str, ...],
                    cache_symbols: tuple[str, ...]) -> tuple[tuple[str, ...], int]:
    """Sorted intersection with the cache membership, and the count removed."""
    cached = frozenset(cache_symbols)
    kept = tuple(sorted(s for s in symbols if s in cached))
    return kept, len(symbols) - len(kept)
