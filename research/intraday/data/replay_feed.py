"""ReplayFeed: cached (adjusted) SIP minute bars converted to the traded basis (feed.md).

Per symbol-day, first matching rule wins: coverage (zero rows), no_daily,
unadjust (factor_mismatch / open_mismatch), no_volume_factor (decision D1), else
bars with prices x ``f`` and share volume / ``s``. A rejected symbol-day emits no
bars and no quotes; the FeedAgent turns its minutes into ``Gap``s.

Prices always use ``f_D`` (D's minute bars sit on D's dividend basis). Share volume
uses ``s`` from D's own daily row (``volume_factor_basis="same_day"``, the default)
or from the market's prior session (``"prior"``, rulings P3-16/17, used by R16).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from research.intraday.contracts.events import Bar, Quote
from research.intraday.data.basis import factor_consistent, price_factor, volume_factor
from research.intraday.data.cache import (
    check_unique, minute_cache_membership, read_daily, read_minute_day,
)
from research.intraday.data.clock import et_to_utc, minute_index_to_utc
from research.intraday.data.feed import SessionData
from research.intraday.data.holdout import HOLDOUT_START, check_holdout
from research.intraday.data.paths import DEFAULT_DATA_ROOT, DEFAULT_VOLUME_DAILY_ROOT
from research.intraday.data.rejects import Reject, RejectKind
from research.intraday.data.sessions import EARLY_CLOSE_ET, NYSE_EARLY_CLOSES, session_close_et
from research.intraday.data.universe import _prior_rows
from research.intraday.data.replay_quotes import read_quote_summaries, synthetic_quotes

_ONE_MIN = timedelta(minutes=1)
_PRICES = ("open", "high", "low", "close", "vwap")
VolumeFactorBasis = Literal["same_day", "prior"]
_VOLUME_BASES: tuple[str, ...] = ("same_day", "prior")
_PRIOR_LOOKBACK_DAYS = 10   # calendar days searched for the market's prior session


def _require_finite_counts(rows: pd.DataFrame) -> None:
    """``volume`` and ``trade_count`` must be finite before ``int()``; name the bad rows."""
    bad = ~(np.isfinite(rows["volume"].astype(float))
            & np.isfinite(rows["trade_count"].astype(float)))
    if bad.any():
        where = list(zip(rows.loc[bad, "symbol"].astype(str), rows.loc[bad, "minute_et"].astype(str)))
        raise ValueError(f"non-finite volume or trade_count in the minute cache at "
                         f"(symbol, minute_et) {where[:5]}")


class ReplayFeed:
    feed = "sip"

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT,
                 volume_root: Path = DEFAULT_VOLUME_DAILY_ROOT,
                 unadjust_tolerance: float = 0.02, holdout_start: date = HOLDOUT_START,
                 unseal: bool = False, session_open_et: str = "09:30",
                 session_close_et: str = "16:00", quote_size_scale: float = 1.0,
                 half_days: frozenset[date] = NYSE_EARLY_CLOSES,
                 early_close_et: str = EARLY_CLOSE_ET, *,
                 volume_factor_basis: VolumeFactorBasis = "same_day") -> None:
        if volume_factor_basis not in _VOLUME_BASES:
            raise ValueError(f"volume_factor_basis must be one of {_VOLUME_BASES}, "
                             f"got {volume_factor_basis!r}")
        self.volume_factor_basis = volume_factor_basis
        self.data_root = Path(data_root)
        self.volume_root = Path(volume_root)
        self.tol = unadjust_tolerance
        self.holdout_start = holdout_start
        self.unseal = unseal
        self.session_open_et = session_open_et
        self.session_close_et = session_close_et
        self.quote_size_scale = quote_size_scale
        self.half_days = frozenset(half_days)
        self.early_close_et = early_close_et

    def close_et(self, d: date) -> str:
        """The session's ET close: ``early_close_et`` on a half day, else ``session_close_et``."""
        return session_close_et(d, self.half_days, self.early_close_et, self.session_close_et)

    def open_session(self, session_date: date, symbols: tuple[str, ...],
                     screen_id: str) -> SessionData:
        check_holdout(session_date, self.holdout_start, self.unseal)
        symbols = tuple(sorted(set(symbols)))
        minute = read_minute_day(self.data_root, session_date, symbols)
        check_unique(minute, ["symbol", "minute_et"], f"minute cache {session_date}")
        close_et = self.close_et(session_date)
        factors = self._factors(session_date, symbols)
        verdict = self._verdicts(session_date, symbols, minute, factors, close_et)
        rejects = tuple(Reject(kind=k, symbol=s, session_date=session_date, reason=r,
                               screen_id=screen_id)
                        for s, (k, r) in sorted(verdict.items()))
        kept = tuple(s for s in symbols if s not in verdict)
        bars = self._bars(session_date, minute[minute["symbol"].isin(kept)], factors, close_et)
        quotes, n_skipped = self._quotes(session_date, kept)
        return SessionData(bars=bars, quotes=quotes, rejects=rejects, n_quotes_skipped=n_skipped,
                           session_close=et_to_utc(session_date, close_et))

    def _factors(self, d: date, symbols: tuple[str, ...]) -> pd.DataFrame:
        """Index ``symbols``: ``open_unadj, f, f_open`` from D's row; ``f_v, s, f_vref``
        (the penny ``f`` of the row ``s`` came from) from D's row or the prior session's."""
        daily = read_daily(self.data_root, self.volume_root, d, d + timedelta(days=1))
        daily = daily[daily["symbol"].isin(symbols)].set_index("symbol")
        out = pd.DataFrame({
            "open_unadj": daily["open_unadj"].astype(float),
            "f": price_factor(daily["close_unadj"], daily["close"]),
            "f_open": price_factor(daily["open_unadj"], daily["open"]),
        }).reindex(list(symbols))
        vol = daily if self.volume_factor_basis == "same_day" else self._prior_daily(d, symbols)
        out["f_v"] = price_factor(vol["close_unadj_v"], vol["close_v"]).reindex(out.index)
        out["s"] = volume_factor(vol["volume_v"], vol["volume_unadj_v"]).reindex(out.index)
        out["f_vref"] = price_factor(vol["close_unadj"], vol["close"]).reindex(out.index)
        return out

    def _prior_daily(self, d: date, symbols: tuple[str, ...]) -> pd.DataFrame:
        """Rows of the market's last session strictly before ``d`` (``universe._prior_rows``)."""
        daily = read_daily(self.data_root, self.volume_root,
                           d - timedelta(days=_PRIOR_LOOKBACK_DAYS), d)
        assert (daily["date"] < pd.Timestamp(d)).all(), "prior read reached the session date"
        if daily.empty:
            return daily.set_index("symbol")
        prior = _prior_rows(daily, daily["date"].max())
        return prior[prior.index.isin(symbols)]

    def _first_session_open(self, minute: pd.DataFrame, close_et: str) -> pd.Series:
        # regular-session rows only: a symbol with rows only outside it gets Gaps (D5)
        session = minute[(minute["minute_et"] >= self.session_open_et)
                         & (minute["minute_et"] < close_et)]
        first = session.sort_values(["symbol", "minute_et"], kind="mergesort")
        return first.groupby("symbol")["open"].first().astype(float)

    def _verdicts(self, d: date, symbols: tuple[str, ...], minute: pd.DataFrame,
                  fac: pd.DataFrame, close_et: str) -> dict[str, tuple[RejectKind, str]]:
        """symbol -> (kind, reason) for every rejected symbol-day; first rule wins."""
        n_rows = minute.groupby("symbol").size().reindex(list(symbols), fill_value=0)
        requested = frozenset(minute_cache_membership(self.data_root, d))
        f_ok = np.isfinite(fac["f"])
        open_conv = self._first_session_open(minute, close_et).reindex(fac.index) * fac["f"]
        open_dev = (open_conv / fac["open_unadj"] - 1.0).abs()
        rules: list[tuple[pd.Series, RejectKind, str]] = [
            (~f_ok, "no_daily", "no_daily_row_or_factor"),
            (~factor_consistent(fac["f_open"], fac["f"], self.tol), "unadjust", "factor_mismatch"),
            (open_dev > self.tol, "unadjust", "open_mismatch"),
            (~np.isfinite(fac["s"]), "no_volume_factor", "no_split_factor"),
            (~factor_consistent(fac["f_v"], fac["f_vref"], self.tol), "no_volume_factor",
             "volume_source_mismatch"),
        ]
        out: dict[str, tuple[RejectKind, str]] = {}
        for sym in symbols:
            if n_rows[sym] == 0:
                reason = "requested_no_rows" if sym in requested else "not_requested"
                out[sym] = ("coverage", reason)
                continue
            for mask, kind, reason in rules:
                if bool(mask[sym]):
                    out[sym] = (kind, reason)
                    break
        return out

    def _bars(self, d: date, minute: pd.DataFrame, fac: pd.DataFrame,
              close_et: str) -> dict[tuple[str, datetime], Bar]:
        rows = minute[(minute["minute_et"] >= self.session_open_et)
                      & (minute["minute_et"] < close_et)]
        rows = rows.sort_values(["symbol", "minute_et"], kind="mergesort").copy()
        _require_finite_counts(rows)
        f = rows["symbol"].map(fac["f"]).astype(float)
        for col in _PRICES:
            rows[col] = rows[col].astype(float) * f
        rows["volume"] = (rows["volume"].astype(float)
                          / rows["symbol"].map(fac["s"]).astype(float)).round()
        starts = minute_index_to_utc(d, rows["minute_et"].astype(str))
        out: dict[tuple[str, datetime], Bar] = {}
        for r in rows.itertuples(index=False):
            bs = starts[str(r.minute_et)]
            sym = str(r.symbol)
            out[(sym, bs)] = Bar(
                ts=bs + _ONE_MIN, symbol=sym, bar_start=bs, open=float(r.open),
                high=float(r.high), low=float(r.low), close=float(r.close),
                volume=int(r.volume), vwap=float(r.vwap), trade_count=int(r.trade_count),
                feed="sip")
        return out

    def _quotes(self, d: date,
                kept: tuple[str, ...]) -> tuple[dict[tuple[str, datetime], Quote], int]:
        summary = read_quote_summaries(self.data_root, d, kept)
        return synthetic_quotes(summary, d, self.quote_size_scale)


@dataclass(frozen=True)
class SymbolDayVerdicts:
    factors: pd.DataFrame                           # ReplayFeed._factors, index = symbols
    verdicts: dict[str, tuple[RejectKind, str]]     # rejected symbols only; first rule wins


def symbol_day_verdicts(feed: ReplayFeed, d: date,
                        symbols: tuple[str, ...]) -> SymbolDayVerdicts:
    """The feed's per-symbol-day factors and reject verdicts for ``d`` (no bars built)."""
    check_holdout(d, feed.holdout_start, feed.unseal)
    symbols = tuple(sorted(set(symbols)))
    minute = read_minute_day(feed.data_root, d, symbols)
    check_unique(minute, ["symbol", "minute_et"], f"minute cache {d}")
    factors = feed._factors(d, symbols)
    return SymbolDayVerdicts(factors=factors,
                             verdicts=feed._verdicts(d, symbols, minute, factors, feed.close_et(d)))
