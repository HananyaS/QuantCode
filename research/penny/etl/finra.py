"""finra.py — consolidated NMS daily short-sale volume.

Free, and history runs back to 2018-08-01. Never wired into the study until
now, which is why two short-volume vetoes in round 1 were inert and had to be
excluded from the search.

The lag is the whole point
--------------------------
FINRA publishes day-*t* short volume **after** day *t* closes -- by 18:00 ET on
the trade date. A feature keyed on day *t* therefore cannot inform a 09:35
decision on day *t*; it can only inform day *t+1*.

This is the same class of error as keying EDGAR on ``filingDate`` instead of
``acceptanceDateTime``, and it is worth being paranoid about because it is
invisible in results: same-day short volume correlates strongly with same-day
returns, so the leak would look like a superb feature.

Every function here shifts by one session, and
:func:`assert_short_volume_is_lagged` exists to make that structural rather
than a matter of remembering.

Interpretation caveat: this is short *volume*, not short *interest*. A high
ratio need not mean bearish positioning -- market makers report hedging flow as
short sales, so in a heavily-quoted name the ratio partly measures liquidity
provision. It is used here as a feature, not as a sentiment reading.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"
HTTP_TIMEOUT = 45
REQUEST_PAUSE = 0.05

# Consolidated NMS short-volume files begin here.
HISTORY_START = date(2018, 8, 1)


def fetch_day(day: date, http_get: Callable = requests.get) -> Optional[pd.DataFrame]:
    """One session's consolidated short volume, or None if not published."""
    if day < HISTORY_START:
        return None

    time.sleep(REQUEST_PAUSE)
    resp = http_get(DAILY_URL.format(yyyymmdd=day.strftime("%Y%m%d")), timeout=HTTP_TIMEOUT)
    if resp.status_code != 200 or not resp.text.strip():
        return None

    df = pd.read_csv(io.StringIO(resp.text), sep="|")
    if "Symbol" not in df.columns:
        return None

    df = df[df["Symbol"].notna()].copy()
    df["date"] = pd.Timestamp(day)
    keep = ["date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume"]
    df = df[[c for c in keep if c in df.columns]]
    return df.rename(columns={
        "Symbol": "symbol", "ShortVolume": "short_volume",
        "ShortExemptVolume": "short_exempt_volume", "TotalVolume": "total_volume",
    })


def fetch_range(
    start: date,
    end: date,
    symbols: Optional[Sequence[str]] = None,
    http_get: Callable = requests.get,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Short volume across a date range, optionally filtered to a symbol set.

    Filtering during the fetch keeps memory and disk small: the raw files carry
    ~12,000 symbols a day, of which this study needs ~3,000.
    """
    wanted = {s.upper() for s in symbols} if symbols else None
    frames: List[pd.DataFrame] = []
    missing = 0

    day = max(start, HISTORY_START)
    i = 0
    while day <= end:
        if day.weekday() < 5:
            i += 1
            part = fetch_day(day, http_get=http_get)
            if part is None:
                missing += 1
            else:
                if wanted is not None:
                    part = part[part["symbol"].str.upper().isin(wanted)]
                if not part.empty:
                    frames.append(part)
            if i % progress_every == 0:
                logger.info("finra: %s | %d files, %s rows",
                            day, len(frames), f"{sum(len(f) for f in frames):,}")
        day += timedelta(days=1)

    if missing:
        logger.info("finra: %d sessions had no published file (holidays/gaps)", missing)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_lagged_features(
    candidates: pd.DataFrame,
    short_volume: pd.DataFrame,
    windows: Sequence[int] = (5, 20),
) -> pd.DataFrame:
    """Point-in-time short-volume features, lagged one session.

    ``shift(1)`` inside each symbol is what makes this causal: a 09:35 decision
    on day *t* sees day *t-1*'s ratio, because day *t*'s file does not exist
    until after the close.
    """
    if short_volume.empty:
        logger.warning("finra: no short-volume data; features will be null")
        out = candidates[["symbol", "date"]].copy()
        out["short_volume_ratio"] = np.nan
        for w in windows:
            out[f"short_ratio_chg_{w}d"] = np.nan
        out["short_has_coverage"] = False
        return out

    sv = short_volume.copy()
    sv["symbol"] = sv["symbol"].str.upper()
    sv = sv.sort_values(["symbol", "date"])

    total = sv["total_volume"].replace(0, np.nan)
    sv["ratio_raw"] = (sv["short_volume"] + sv.get("short_exempt_volume", 0)) / total

    grp = sv.groupby("symbol", sort=False)
    # The lag. Everything downstream reads only the shifted column.
    sv["short_volume_ratio"] = grp["ratio_raw"].shift(1)
    for w in windows:
        trailing = grp["ratio_raw"].transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
        )
        sv[f"short_ratio_chg_{w}d"] = sv["short_volume_ratio"] - trailing

    covered = set(sv["symbol"].unique())
    cols = ["symbol", "date", "short_volume_ratio"] + [f"short_ratio_chg_{w}d" for w in windows]

    out = candidates[["symbol", "date"]].copy()
    out["symbol_u"] = out["symbol"].str.upper()
    merged = out.merge(
        sv[cols].rename(columns={"symbol": "symbol_u"}),
        on=["symbol_u", "date"], how="left",
    )
    merged["short_has_coverage"] = merged["symbol_u"].isin(covered)
    return merged.drop(columns=["symbol_u"])


def assert_short_volume_is_lagged(
    features: pd.DataFrame, raw: pd.DataFrame, symbol: str, day: pd.Timestamp
) -> None:
    """Fail if a symbol-day's own short volume reached its own features.

    Structural rather than remembered. FINRA publishes after the close, so
    same-day short volume is future information -- and because it correlates
    with same-day returns, the leak would present as an excellent feature.
    """
    same_day = raw[(raw["symbol"].str.upper() == symbol.upper()) & (raw["date"] == day)]
    if same_day.empty:
        return

    row = features[(features["symbol"] == symbol) & (features["date"] == day)]
    if row.empty or row["short_volume_ratio"].isna().all():
        return

    total = same_day["total_volume"].iloc[0]
    if not total:
        return
    own = (same_day["short_volume"].iloc[0] + same_day.get("short_exempt_volume", pd.Series([0])).iloc[0]) / total
    observed = float(row["short_volume_ratio"].iloc[0])
    assert not np.isclose(observed, own, atol=1e-9), (
        f"{symbol} on {day:%Y-%m-%d}: feature equals that day's OWN short ratio "
        f"({observed:.6f}) -- FINRA publishes after the close, so this is a leak"
    )
