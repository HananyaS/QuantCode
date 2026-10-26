"""Parquet readers over the cached data (read-only; never writes, never downloads).

Layout (see ``paths.py``): penny daily ``bars_<year>.parquet`` (adjusted OHLCV plus
``*_unadj`` prices), the breakout daily (same plus ``volume_unadj``, the source of
the split-only volume factor), per-day minute and pre-market parquet files, and the
per-day ``requested.json`` (cache membership, decision D2).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from research.intraday.data.clock import minute_index_to_utc
from research.intraday.data.paths import (
    daily_file, minute_file, premarket_file, requested_file, volume_daily_file,
)

DAILY_COLUMNS: tuple[str, ...] = (
    "symbol", "date", "open", "close", "open_unadj", "close_unadj", "volume")
VOLUME_COLUMNS: tuple[str, ...] = (
    "symbol", "date", "close", "close_unadj", "volume", "volume_unadj")
_VOLUME_RENAME: dict[str, str] = {
    "close": "close_v", "close_unadj": "close_unadj_v", "volume": "volume_v",
    "volume_unadj": "volume_unadj_v"}
READ_DAILY_COLUMNS: tuple[str, ...] = DAILY_COLUMNS + tuple(_VOLUME_RENAME.values())
MINUTE_COLUMNS: tuple[str, ...] = (
    "symbol", "date", "minute_et", "open", "high", "low", "close", "volume",
    "trade_count", "vwap")
_TEXT_COLUMNS = frozenset({"symbol", "minute_et", "window"})


def empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    """An empty frame with the cache's column dtypes."""
    data: dict[str, pd.Series] = {}
    for c in columns:
        if c in _TEXT_COLUMNS:
            data[c] = pd.Series(dtype=object)
        elif c == "date":
            data[c] = pd.Series(dtype="datetime64[ns]")
        else:
            data[c] = pd.Series(dtype=float)
    return pd.DataFrame(data)


def _read_years(paths: list[Path], columns: tuple[str, ...], start: date,
                end: date) -> pd.DataFrame:
    filters = [("date", ">=", pd.Timestamp(start)), ("date", "<", pd.Timestamp(end))]
    parts = [pd.read_parquet(p, columns=list(columns), filters=filters)
             for p in paths if p.is_file()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return empty_frame(columns)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    return df


def check_unique(df: pd.DataFrame, keys: list[str], what: str) -> None:
    """Raise ``ValueError`` ("duplicate ...") if ``keys`` repeat in ``df``."""
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"duplicate {tuple(keys)} rows in {what}: {sample}")


def read_daily(data_root: Path, volume_root: Path, start: date, end: date) -> pd.DataFrame:
    """Daily rows ``start <= date < end``: penny daily left-joined to the volume source."""
    years = range(start.year, (end - timedelta(days=1)).year + 1)
    penny = _read_years([daily_file(data_root, y) for y in years], DAILY_COLUMNS, start, end)
    vol = _read_years([volume_daily_file(volume_root, y) for y in years], VOLUME_COLUMNS,
                      start, end)
    check_unique(penny, ["symbol", "date"], "penny daily")
    check_unique(vol, ["symbol", "date"], "volume daily")
    vol = vol.rename(columns=_VOLUME_RENAME)
    out = penny.merge(vol, on=["symbol", "date"], how="left", validate="one_to_one")
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    return out[list(READ_DAILY_COLUMNS)]


def minute_cache_membership(data_root: Path, d: date) -> tuple[str, ...]:
    """Sorted symbols requested for ``d`` (``requested.json``), else those present, else ()."""
    req = requested_file(data_root, d)
    if req.is_file():
        symbols = json.loads(req.read_text(encoding="utf-8"))
        assert isinstance(symbols, list), f"{req}: expected a JSON list"
        return tuple(sorted({str(s) for s in symbols}))
    path = minute_file(data_root, d)
    if path.is_file():
        present = pd.read_parquet(path, columns=["symbol"])["symbol"]
        return tuple(sorted(present.astype(str).unique()))
    return ()


def read_minute_day(data_root: Path, d: date, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Rows of ``minute.parquet`` for ``symbols`` (empty frame if the file is missing)."""
    path = minute_file(data_root, d)
    if not path.is_file() or not symbols:
        return empty_frame(MINUTE_COLUMNS)
    df = pd.read_parquet(path, filters=[("symbol", "in", list(symbols))])
    return df.reset_index(drop=True)


def read_premarket_day(data_root: Path, d: date) -> pd.DataFrame:
    """``premarket.parquet`` plus ``bar_start`` (``datetime64[ns, UTC]``)."""
    path = premarket_file(data_root, d)
    df = pd.read_parquet(path) if path.is_file() else empty_frame(MINUTE_COLUMNS)
    mapping = minute_index_to_utc(d, df["minute_et"].astype(str).unique())
    starts = df["minute_et"].map(mapping)
    df["bar_start"] = pd.to_datetime(starts, utc=True).astype("datetime64[ns, UTC]")
    return df.reset_index(drop=True)
