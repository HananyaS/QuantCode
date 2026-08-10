"""bars.py — daily bar ETL, cached per year.

Why both raw and adjusted bars are fetched
------------------------------------------
They answer different questions and using one for both is a silent corruption:

* **Raw (unadjusted)** is what the price screen must use. A stock trading at $2
  in 2022 that later reverse-splits 1-for-20 appears at $40 in an adjusted
  series. Screening a penny-stock study on adjusted prices excludes that name
  on exactly the days it *was* a penny stock, and admits it on days it was not.
  Raw is also the correct basis for same-session P&L, because entry and exit
  are both real prices on the same day.

* **Adjusted** is what any cross-day feature must use -- gap versus prior
  close, trailing ADV, 20-day MAX. On a split date the raw series shows a
  fabricated overnight move that an unguarded gap feature would read as a real
  one, and reverse splits are endemic in this universe.

Storage is one parquet per calendar year in long format, which keeps a single
year loadable in memory while allowing a multi-year panel to be assembled
lazily.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.penny.etl import alpaca_fetch

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data/research_penny")
DAILY_DIR = DATA_ROOT / "daily"

_KEEP_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


def _year_path(year: int) -> Path:
    return DAILY_DIR / f"bars_{year}.parquet"


def fetch_year(
    client,
    symbols: List[str],
    year: int,
    end_cap: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Fetch one calendar year of raw and adjusted daily bars, merged.

    Returns a frame carrying adjusted OHLCV plus ``close_unadj`` / ``open_unadj``
    / ``high_unadj`` / ``low_unadj`` from the raw series, and the rejected
    symbols per adjustment mode.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    import time

    start = f"{year}-01-01"
    end = end_cap or f"{year}-12-31"

    def make_request(adjustment: str):
        def request(batch: List[str]) -> pd.DataFrame:
            time.sleep(alpaca_fetch.REQUEST_PAUSE_SECONDS)
            return client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end,
                    adjustment=adjustment,
                    feed="sip",
                )
            ).df

        return request

    results: Dict[str, pd.DataFrame] = {}
    rejects: Dict[str, List[str]] = {}

    for adjustment in ("all", "raw"):
        frames: List[pd.DataFrame] = []
        rejected: List[str] = []
        request = make_request(adjustment)
        batches = [
            symbols[i : i + alpaca_fetch.DEFAULT_BATCH_SIZE]
            for i in range(0, len(symbols), alpaca_fetch.DEFAULT_BATCH_SIZE)
        ]
        for idx, batch in enumerate(batches, start=1):
            if idx == 1 or idx % 10 == 0:
                logger.info("  %d %s: batch %d/%d", year, adjustment, idx, len(batches))
            df, dropped = alpaca_fetch.fetch_with_rejects(request, batch)
            rejected.extend(dropped)
            if df is not None and not df.empty:
                frames.append(df)

        rejects[adjustment] = rejected
        if not frames:
            results[adjustment] = pd.DataFrame()
            continue

        out = pd.concat(frames).reset_index()
        ts = out["timestamp"]
        if getattr(ts.dt, "tz", None) is not None:
            out["date"] = ts.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        else:
            out["date"] = ts.dt.normalize()
        results[adjustment] = out.drop(columns=["timestamp"])

    adj, raw = results.get("all"), results.get("raw")
    if adj is None or adj.empty:
        return pd.DataFrame(), rejects

    adj = adj[[c for c in _KEEP_COLS if c in adj.columns]]

    if raw is not None and not raw.empty:
        raw_cols = raw[["symbol", "date", "open", "high", "low", "close"]].rename(
            columns={
                "open": "open_unadj", "high": "high_unadj",
                "low": "low_unadj", "close": "close_unadj",
            }
        )
        merged = adj.merge(raw_cols, on=["symbol", "date"], how="left")
    else:
        merged = adj.copy()
        for col in ("open", "high", "low", "close"):
            merged[f"{col}_unadj"] = pd.NA

    return merged.sort_values(["symbol", "date"]).reset_index(drop=True), rejects


def load_years(years: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load cached years into one panel."""
    frames = []
    for year in years:
        path = _year_path(year)
        if not path.exists():
            logger.warning("bars: %s missing, skipping", path)
            continue
        frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])


def write_year(df: pd.DataFrame, year: int) -> Path:
    path = _year_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    return path


def add_cross_day_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Causal cross-day features, computed on the ADJUSTED series.

    Every column here is observable strictly before the session it describes
    opens, except ``gap`` and ``open_*`` which use only the opening print --
    available at 09:30, before the 09:35 decision point.

    Uses adjusted prices throughout: on a split date the raw series contains a
    fabricated overnight move that ``gap`` would otherwise read as real.
    """
    df = panel.sort_values(["symbol", "date"]).copy()
    grp = df.groupby("symbol", sort=False)

    df["prev_close"] = grp["close"].shift(1)
    df["prev_volume"] = grp["volume"].shift(1)
    df["prev_dollar_volume"] = df["prev_close"] * df["prev_volume"]

    # Gap uses the opening print against the previous close -- known at 09:30.
    df["gap"] = df["open"] / df["prev_close"] - 1.0

    # Both shifts must live INSIDE the transform. A trailing `.shift(1)` on the
    # concatenated result shifts across the whole frame, so each symbol's first
    # row would inherit the previous symbol's last value -- a cross-symbol,
    # cross-time bleed from years in the future.
    df["ret_1d_prev"] = grp["close"].transform(lambda s: s.pct_change().shift(1))
    df["ret_5d_prev"] = grp["close"].transform(lambda s: s.pct_change(5).shift(1))

    # Trailing dollar volume, shifted so a row never sees its own volume.
    # Uses transform on a precomputed column rather than groupby.apply, which
    # returns a frame rather than a series and cannot be assigned back.
    df["_dollar_volume_today"] = df["close"] * df["volume"]
    for win in (5, 20, 60):
        df[f"adv_{win}"] = df.groupby("symbol", sort=False)["_dollar_volume_today"].transform(
            lambda s, w=win: s.shift(1).rolling(w, min_periods=max(3, w // 4)).mean()
        )

    df["rvol_prev"] = df["prev_dollar_volume"] / df["adv_20"]

    # MAX effect (Bali, Cakici & Whitelaw 2011): the largest single-day return
    # over the trailing month predicts *lower* subsequent returns.
    df["daily_ret"] = grp["close"].pct_change()
    df["max_ret_20d"] = grp["daily_ret"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).max()
    )

    # Realised volatility and the Parkinson high-low estimator.
    df["vol_20d"] = grp["daily_ret"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).std()
    )

    # Corwin-Schultz style spread proxy: the high-low range carries the
    # effective spread when quotes are unavailable, which they are here.
    # (`mode.use_inf_as_na` is removed in pandas 3.x; replace infinities directly.)
    hl = ((df["high"] - df["low"]) / df["close"]).replace([np.inf, -np.inf], np.nan)
    df["hl_range_prev"] = hl.groupby(df["symbol"]).shift(1)
    df["hl_range_20d"] = grp["hl_range_prev"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    # Consecutive up days. `cumcount` counts ROWS in each reset-group, not True
    # values -- and a NaN yesterday is False under both `> 0` and `<= 0`, so it
    # neither qualifies nor resets yet still increments the count. That inflates
    # every streak after any data gap. `cumsum` over the boolean counts only
    # genuine up days, and an unknown yesterday correctly resets to zero.
    def _streak(s: pd.Series) -> pd.Series:
        up = s.shift(1) > 0
        return up.groupby((~up).cumsum()).cumsum()

    df["consec_up"] = grp["daily_ret"].transform(_streak)

    # Causal split activity. With adjustment="all" the adjusted series is
    # anchored to the FETCH date, so close/close_unadj on day t is the
    # cumulative factor for corporate actions occurring AFTER t. Using that
    # ratio directly is a look-ahead: it announces future reverse splits, which
    # in this universe follow price collapse. Differencing it recovers the
    # per-day event factor, which IS knowable (ex-dates are announced ahead),
    # and the trailing product then measures splits that have ALREADY happened.
    ratio = (df["close"] / df["close_unadj"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    df["_adj_ratio"] = ratio
    prev_ratio = ratio.groupby(df["symbol"]).shift(1)
    event_factor = (prev_ratio / ratio).fillna(1.0)
    # The shift reintroduces a NaN at each symbol's first row; np.prod over a
    # window containing it returns NaN, which a trailing fillna(1.0) would then
    # disguise as "no split activity" for every early row.
    df["split_factor_180d"] = event_factor.groupby(df["symbol"]).transform(
        lambda s: s.shift(1).fillna(1.0).rolling(180, min_periods=1).apply(np.prod, raw=True)
    )

    return df
