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
    if ok:
        for i in range(1, upto + 1):
            if np.isnan(g[3, i]):
                p = g[3, i - 1]
                g[:, i] = (p, p, p, p, 0.0, p)
    return TodayBars(upto, float(g[0, 0]), g[3].copy(), g[4].copy(), g[5].copy(), ok)


class Feed(Protocol):
    def today(self, symbol: str, day: pd.Timestamp, upto: int) -> TodayBars: ...
    def history(self, symbol: str, sessions: list[tuple[pd.Timestamp, int]]) -> Minutes: ...


# ----------------------------------------------------------------- Alpaca ---
class AlpacaFeed:
    def __init__(self, key: str, secret: str) -> None:
        from alpaca.data.historical import StockHistoricalDataClient
        self._c = StockHistoricalDataClient(key, secret)

    def _bars(self, symbol: str, start: datetime, end: datetime, feed: str) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end,
                               feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP)
        df = self._c.get_stock_bars(req).df
        if df is None or not len(df):
            return pd.DataFrame(columns=["date", "minute_et", *FIELDS])
        df = df.reset_index()
        ts = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
        df["date"] = ts.dt.tz_localize(None).dt.normalize()
        df["minute_et"] = ts.dt.strftime("%H:%M")
        return df[["date", "minute_et", *FIELDS]]

    def today(self, symbol: str, day: pd.Timestamp, upto: int) -> TodayBars:
        start = pd.Timestamp(day).tz_localize(ET) + timedelta(hours=9, minutes=30)
        end = start + timedelta(minutes=upto + 1)
        return today_from_rows(self._bars(symbol, start.to_pydatetime(), end.to_pydatetime(), "iex"), upto)

    def history(self, symbol: str, sessions: list[tuple[pd.Timestamp, int]]) -> Minutes:
        days = [pd.Timestamp(d) for d, _ in sessions]
        start = days[0].tz_localize(ET) + timedelta(hours=9, minutes=30)
        end = days[-1].tz_localize(ET) + timedelta(hours=16)
        df = self._bars(symbol, start.to_pydatetime(), end.to_pydatetime(), "sip")
        return minutes_from_frame(symbol, df, sessions)


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
