"""One-minute regular-session bars for SPY/QQQ as (sessions x 390) arrays.

Source: `data/research_poolpoc/etf/minute/dt=YYYY-MM-DD/etf.parquet`, Alpaca
SIP bars 04:00-16:05 ET, dividend-adjusted (SPY's 2016-01-04 open reads 170.23
against a raw ~201). Adjustment is a constant within a session, so intraday
returns are exact; across sessions the adjusted series is the total-return path.

Layout: bar i starts at 09:30 + i minutes. A session has 390 bars, or 210 on a
13:00 close; slots past the session's last bar are NaN. A regular-session
minute with no trades gets open = high = low = close = the previous close,
volume 0, vwap = the previous close. A session is `valid` only if its 09:30 bar
exists and at least 95% of its bars do; cells stay flat on invalid sessions.

The cache is built per window end, so a dev build never opens a later file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquid.calendar import EARLY_CLOSES, FULL_BARS, assert_through, session_bars

logger = logging.getLogger(__name__)

MINUTE_DIR = Path("data/research_poolpoc/etf/minute")
CACHE_DIR = Path("data/research_liquid")
FIELDS = ("open", "high", "low", "close", "volume", "vwap")
MIN_COVERAGE = 0.95
EARLY_CLOSE_VOL_RATIO = 0.2      # median minute volume after 13:00 / before; dev: early <= 0.09, normal >= 0.28


@dataclass(frozen=True)
class Minutes:
    symbol: str
    dates: pd.DatetimeIndex
    n_bars: np.ndarray            # (D,) 390 or 210
    open: np.ndarray              # (D, 390), NaN past the session
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vwap: np.ndarray
    prev_close: np.ndarray        # (D,) close of the previous session's last bar; NaN on the first
    valid: np.ndarray             # (D,) bool

    def __post_init__(self) -> None:
        d = len(self.dates)
        for f in FIELDS:
            assert getattr(self, f).shape == (d, FULL_BARS), f
        assert self.n_bars.shape == self.prev_close.shape == self.valid.shape == (d,)

    def last_bar(self, i: int) -> int:
        return int(self.n_bars[i]) - 1

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> "Minutes":
        k = (self.dates >= start) & (self.dates <= end)
        return Minutes(self.symbol, self.dates[k], self.n_bars[k],
                       *(getattr(self, f)[k] for f in FIELDS), self.prev_close[k], self.valid[k])


def _day_grid(day: pd.Timestamp, rows: pd.DataFrame,
              n_bars: int | None = None) -> tuple[np.ndarray, int, bool, float]:
    """One session's (6, 390) grid, its bar count, validity and late-volume ratio.

    `n_bars` overrides the research calendar (the live runner passes Alpaca's)."""
    n = session_bars(day) if n_bars is None else n_bars
    g = np.full((len(FIELDS), FULL_BARS), np.nan)
    hh = rows["minute_et"].str.slice(0, 2).astype(int)
    mm = rows["minute_et"].str.slice(3, 5).astype(int)
    idx = ((hh - 9) * 60 + (mm - 30)).to_numpy()
    full = (idx >= 0) & (idx < FULL_BARS)
    vol_all = np.zeros(FULL_BARS)
    vol_all[idx[full]] = rows["volume"].to_numpy()[full]
    # median per-minute volume 13:05-15:54 over 10:00-12:54; a sum is swamped by a 13:00 auction print
    early, late = np.median(vol_all[30:205]), np.median(vol_all[215:385])
    ratio = late / early if early > 0 else np.nan
    keep = (idx >= 0) & (idx < n)
    for j, f in enumerate(FIELDS):
        g[j, idx[keep]] = rows[f].to_numpy(float)[keep]
    present = ~np.isnan(g[3, :n])
    valid = bool(present[0]) and present.mean() >= MIN_COVERAGE
    if present[0]:
        for i in range(1, n):                       # no trade that minute: carry the close
            if np.isnan(g[3, i]):
                p = g[3, i - 1]
                g[:, i] = (p, p, p, p, 0.0, p)
    return g, n, valid, ratio


def build_cache(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    """Read the day files in [start, end] and write one .npz for `symbol`."""
    days = sorted(pd.Timestamp(p.name[3:]) for p in MINUTE_DIR.glob("dt=*"))
    days = [d for d in days if start <= d <= end]
    assert_through(days, end, "minute files")
    grids, nb, val, ratios = [], [], [], []
    for d in days:
        df = pd.read_parquet(MINUTE_DIR / f"dt={d:%Y-%m-%d}" / "etf.parquet",
                             columns=["symbol", "minute_et", *FIELDS])
        g, n, v, r = _day_grid(d, df[df["symbol"] == symbol])
        grids.append(g); nb.append(n); val.append(v); ratios.append(r)
    ratios = np.array(ratios)
    listed = np.array([d in EARLY_CLOSES for d in days])
    looks_early = ratios < EARLY_CLOSE_VOL_RATIO
    bad = np.flatnonzero(listed != looks_early)
    if len(bad):
        raise ValueError(f"{symbol}: early-close list disagrees with volume on "
                         f"{[(days[i].date(), round(float(ratios[i]), 3)) for i in bad]}")
    G = np.stack(grids)                                            # (D, 6, 390)
    nb = np.array(nb)
    last_close = G[np.arange(len(days)), 3, nb - 1]
    prev = np.concatenate([[np.nan], last_close[:-1]])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}_rth_{start:%Y%m%d}_{end:%Y%m%d}.npz"
    np.savez_compressed(path, dates=np.array(days, dtype="datetime64[ns]"), n_bars=nb,
                        grid=G, prev_close=prev, valid=np.array(val))
    logger.info("%s: %d sessions, %d invalid, %d early closes -> %s",
                symbol, len(days), int((~np.array(val)).sum()), int(listed.sum()), path)
    return path


def load(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Minutes:
    """Bars for `symbol` on [start, end] from a cache whose own end is <= `end`'s window."""
    path = CACHE_DIR / f"{symbol}_rth_{start:%Y%m%d}_{end:%Y%m%d}.npz"
    if not path.exists():
        build_cache(symbol, start, end)
    z = np.load(path)
    dates = pd.DatetimeIndex(z["dates"])
    assert_through(dates, end, f"{symbol} cache")
    G = z["grid"]
    return Minutes(symbol, dates, z["n_bars"], *(G[:, j, :] for j in range(len(FIELDS))),
                   z["prev_close"], z["valid"])
