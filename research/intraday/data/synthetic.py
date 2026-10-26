"""Tiny deterministic synthetic cache writers (test support; no randomness)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from research.intraday.data.paths import (
    daily_file, minute_file, premarket_file, quote_summary_files, requested_file,
    volume_daily_file,
)

_PENNY_DAILY = ("symbol", "date", "open", "high", "low", "close", "volume", "trade_count",
                "vwap", "open_unadj", "high_unadj", "low_unadj", "close_unadj")
_VOLUME_DAILY = ("symbol", "date", "open", "high", "low", "close", "volume", "vwap",
                 "open_unadj", "high_unadj", "low_unadj", "close_unadj", "volume_unadj",
                 "trade_count")


def daily_row(symbol: str, d: date, close: float, f: float = 1.0, s: float | None = None,
              volume: float = 1e6, open_: float | None = None) -> dict[str, Any]:
    """One adjusted daily row; raw = adjusted x ``f``; ``volume_unadj = volume / s``.

    ``s`` defaults to ``f``.
    """
    s = f if s is None else s
    o = close if open_ is None else open_
    hi, lo = max(o, close), min(o, close)
    return dict(symbol=symbol, date=pd.Timestamp(d), open=o, high=hi, low=lo, close=close,
                volume=float(volume), trade_count=100.0, vwap=close,
                open_unadj=o * f, high_unadj=hi * f, low_unadj=lo * f, close_unadj=close * f,
                volume_unadj=float(volume) / s)


def _write_by_year(rows: pd.DataFrame, columns: tuple[str, ...],
                   path_of: Callable[[int], Path]) -> None:
    df = rows.copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    for year, part in df.groupby(df["date"].dt.year, sort=True):
        path = path_of(int(year))
        path.parent.mkdir(parents=True, exist_ok=True)
        part[[c for c in columns if c in part.columns]].to_parquet(path, index=False)


def write_daily(root: Path, rows: pd.DataFrame) -> None:
    _write_by_year(rows, _PENNY_DAILY, lambda y: daily_file(root, y))


def write_volume_daily(volume_root: Path, rows: pd.DataFrame) -> None:
    _write_by_year(rows, _VOLUME_DAILY, lambda y: volume_daily_file(volume_root, y))


def minute_rows(symbol: str, d: date, minutes: Sequence[str], price: float = 2.0,
                volume: float = 1000.0) -> pd.DataFrame:
    """Flat bars in the cache's dtypes (``volume``, ``trade_count`` float64)."""
    n = len(minutes)
    return pd.DataFrame({
        "symbol": [symbol] * n,
        "date": pd.Series([pd.Timestamp(d)] * n, dtype="datetime64[ms]"),
        "minute_et": list(minutes),
        "open": [float(price)] * n, "high": [float(price)] * n, "low": [float(price)] * n,
        "close": [float(price)] * n, "volume": [float(volume)] * n,
        "trade_count": [10.0] * n, "vwap": [float(price)] * n,
    })


def _write(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)


def write_minute(root: Path, d: date, rows: pd.DataFrame,
                 requested: tuple[str, ...] | None) -> None:
    _write(minute_file(root, d), rows)
    if requested is not None:
        requested_file(root, d).write_text(json.dumps(sorted(requested)), encoding="utf-8")


def write_premarket(root: Path, d: date, rows: pd.DataFrame) -> None:
    _write(premarket_file(root, d), rows)


def write_quote_summaries(root: Path, r4: pd.DataFrame, rev: pd.DataFrame) -> None:
    p_r4, p_rev = quote_summary_files(root)
    _write(p_r4, r4)
    _write(p_rev, rev)
