"""Market data for the C3 runner (PAPER_C3 §2.1). The only module importing alpaca.data.

- today(...): today's one-minute bars from IEX (real time on the free plan),
  bars 0..upto only. A minute with no IEX trade carries the previous close
  forward, as research does. Nothing past `upto` is filled.
- history(...): SIP bars of earlier sessions (>= 15 minutes old, so allowed on
  the free plan), laid out exactly as the research cache (`data._day_grid`).
- FakeFeed replays a Minutes object for tests and parity checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from research.liquid.calendar import FULL_BARS
from research.liquid.data import FIELDS, Minutes, _day_grid

ET = "America/New_York"


@dataclass(frozen=True)
class TodayBars:
    upto: int
    open0: float
    close: np.ndarray          # (390,) NaN past `upto`
    volume: np.ndarray
    vwap: np.ndarray
    complete: bool             # bar 0 present and every bar 0..upto present or carried
    has_upto: bool = True      # the decision bar itself arrived (not carried) (G3 W3)
    n_carried: int = 0         # bars 1..upto carried forward

    def last_price(self) -> float:
        return float(self.close[self.upto])


def today_from_rows(rows: pd.DataFrame, upto: int) -> TodayBars:
    """rows: minute_et 'HH:MM' plus FIELDS. Uses bars 0..upto only."""
    g = np.full((len(FIELDS), FULL_BARS), np.nan)
    if len(rows):
        hh = rows["minute_et"].str.slice(0, 2).astype(int)
        mm = rows["minute_et"].str.slice(3, 5).astype(int)
        idx = ((hh - 9) * 60 + (mm - 30)).to_numpy()
        keep = (idx >= 0) & (idx <= upto)
        for j, f in enumerate(FIELDS):
            g[j, idx[keep]] = rows[f].to_numpy(float)[keep]
    ok = bool(np.isfinite(g[3, 0]))
    has_upto = bool(np.isfinite(g[3, upto]))
    carried = 0
    if ok:
        for i in range(1, upto + 1):
            if np.isnan(g[3, i]):
                p = g[3, i - 1]
                g[:, i] = (p, p, p, p, 0.0, p)
                carried += 1
    return TodayBars(upto, float(g[0, 0]), g[3].copy(), g[4].copy(), g[5].copy(), ok, has_upto, carried)


SIP_DELAY_MIN = 16            # SIP bars this old are free on the basic plan (15 min, plus a margin)


def merge_today(sip: pd.DataFrame, iex: pd.DataFrame, sip_through: int) -> pd.DataFrame:
    """SIP rows for bars 0..sip_through, IEX rows after (the hybrid live feed, G3 B2)."""
    def idx(df):
        if not len(df):
            return np.array([], int)
        hh = df["minute_et"].str.slice(0, 2).astype(int)
        mm = df["minute_et"].str.slice(3, 5).astype(int)
        return ((hh - 9) * 60 + (mm - 30)).to_numpy()
    a = sip[idx(sip) <= sip_through] if len(sip) else sip
    b = iex[idx(iex) > sip_through] if len(iex) else iex
    cols = ["minute_et", *FIELDS]
    return pd.concat([a[cols], b[cols]], ignore_index=True) if len(a) or len(b) else \
        pd.DataFrame(columns=cols)


class Feed(Protocol):
    def today(self, symbol: str, day: pd.Timestamp, upto: int) -> TodayBars: ...
    def history(self, symbol: str, sessions: list[tuple[pd.Timestamp, int]]) -> Minutes: ...


# ----------------------------------------------------------------- Alpaca ---
class AlpacaFeed:
    def __init__(self, key: str, secret: str) -> None:
        from alpaca.data.historical import StockHistoricalDataClient
        from research.liquid.live.broker import with_timeout
        self._c = StockHistoricalDataClient(key, secret)
        with_timeout(self._c)

    def _bars(self, symbol: str, start: datetime, end: datetime, feed: str,
              adjustment: str = "all") -> pd.DataFrame:
        """adjustment 'all' for history matches the dividend-adjusted research cache (G3 W2);
        today's bars carry no later corporate action, so 'all' and 'raw' agree on them."""
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end,
                               feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP,
                               adjustment=Adjustment(adjustment))
        df = self._c.get_stock_bars(req).df
        if df is None or not len(df):
            return pd.DataFrame(columns=["date", "minute_et", *FIELDS])
        df = df.reset_index()
        ts = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
        df["date"] = ts.dt.tz_localize(None).dt.normalize()
        df["minute_et"] = ts.dt.strftime("%H:%M")
        return df[["date", "minute_et", *FIELDS]]

    def today(self, symbol: str, day: pd.Timestamp, upto: int) -> TodayBars:
        """Bars 0..upto: SIP for bars at least SIP_DELAY_MIN old, IEX for the rest (hybrid)."""
        start = pd.Timestamp(day).tz_localize(ET) + timedelta(hours=9, minutes=30)
        end = start + timedelta(minutes=upto + 1)
        sip_through = upto - SIP_DELAY_MIN
        now = pd.Timestamp.now(tz=ET)
        sip_through = min(sip_through, int((now - start).total_seconds() // 60) - SIP_DELAY_MIN - 1)
        iex = self._bars(symbol, start.to_pydatetime(), end.to_pydatetime(), "iex")
        if sip_through < 0:
            return today_from_rows(iex, upto)
        sip = self._bars(symbol, start.to_pydatetime(), (start + timedelta(minutes=sip_through + 1)).to_pydatetime(),
                         "sip")
        return today_from_rows(merge_today(sip, iex, sip_through), upto)

    def history(self, symbol: str, sessions: list[tuple[pd.Timestamp, int]],
                adjustment: str = "all") -> Minutes:
        days = [pd.Timestamp(d) for d, _ in sessions]
        start = days[0].tz_localize(ET) + timedelta(hours=9, minutes=30)
        end = days[-1].tz_localize(ET) + timedelta(hours=16)
        df = self._bars(symbol, start.to_pydatetime(), end.to_pydatetime(), "sip", adjustment)
        return minutes_from_frame(symbol, df, sessions)

    def iex_history(self, symbol: str, sessions: list[tuple[pd.Timestamp, int]],
                    adjustment: str = "all") -> Minutes:
        """IEX bars of past sessions, laid out like `history` (for the fidelity calibration)."""
        days = [pd.Timestamp(d) for d, _ in sessions]
        start = days[0].tz_localize(ET) + timedelta(hours=9, minutes=30)
        end = days[-1].tz_localize(ET) + timedelta(hours=16)
        df = self._bars(symbol, start.to_pydatetime(), end.to_pydatetime(), "iex", adjustment)
        return minutes_from_frame(symbol, df, sessions)

    def mid_at(self, symbol: str, ts: datetime, window_s: float = 3.0) -> float:
        """SIP NBBO midpoint in force at `ts` (free once 15 minutes old). NaN if none."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockQuotesRequest
        for w in (window_s, 30.0):
            req = StockQuotesRequest(symbol_or_symbols=symbol, start=ts - timedelta(seconds=w), end=ts,
                                     feed=DataFeed.SIP)
            df = self._c.get_stock_quotes(req).df
            if df is not None and len(df):
                q = df[(df["bid_price"] > 0) & (df["ask_price"] >= df["bid_price"])]
                if len(q):
                    r = q.iloc[-1]
                    return float((r["bid_price"] + r["ask_price"]) / 2)
        return float("nan")


def minutes_from_frame(symbol: str, df: pd.DataFrame, sessions: list[tuple[pd.Timestamp, int]]) -> Minutes:
    """SIP rows for several sessions -> Minutes, built like the research cache.
    Bar counts come from the broker's calendar (`sessions`), not the research list."""
    grids, nbs, val = [], [], []
    for d, n in sessions:
        g, nb, v, _ = _day_grid(pd.Timestamp(d), df[df["date"] == pd.Timestamp(d)], n_bars=n)
        grids.append(g); nbs.append(nb); val.append(v)
    G = np.stack(grids)
    nb = np.array(nbs)
    last_close = G[np.arange(len(sessions)), 3, nb - 1]
    prev = np.concatenate([[np.nan], last_close[:-1]])
    return Minutes(symbol, pd.DatetimeIndex([pd.Timestamp(d) for d, _ in sessions]), nb,
                   *(G[:, j, :] for j in range(len(FIELDS))), prev, np.array(val))


# ------------------------------------------------------------------- fake ---
class FakeFeed:
    """Serves bars from a Minutes object (research cache) for tests and parity runs."""

    def __init__(self, m: Minutes) -> None:
        self.m = m

    def _row(self, day: pd.Timestamp) -> int:
        return int(np.flatnonzero(self.m.dates == pd.Timestamp(day))[0])

    def today(self, symbol: str, day: pd.Timestamp, upto: int) -> TodayBars:
        i = self._row(day)
        c = np.full(FULL_BARS, np.nan); v = np.full(FULL_BARS, np.nan); w = np.full(FULL_BARS, np.nan)
        c[: upto + 1] = self.m.close[i, : upto + 1]
        v[: upto + 1] = self.m.volume[i, : upto + 1]
        w[: upto + 1] = self.m.vwap[i, : upto + 1]
        return TodayBars(upto, float(self.m.open[i, 0]), c, v, w, bool(self.m.valid[i]))

    def history(self, symbol: str, sessions: list[tuple[pd.Timestamp, int]]) -> Minutes:
        days = [pd.Timestamp(d) for d, _ in sessions]
        return self.m.slice(days[0], days[-1])
