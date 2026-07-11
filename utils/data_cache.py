"""data_cache.py — local parquet cache for per-ticker OHLCV data.

Storage layout
--------------
  data/cache/{TICKER}.parquet   (one file per ticker, all dates ever fetched)

On each run the cache is consulted first.  Only tickers whose cached data
does NOT fully cover [start, end] are re-downloaded.  Downloaded data is
merged with the existing parquet (deduped, sorted) so subsequent runs are
instant for the same or narrower date range.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")


def load_or_fetch(
    tickers: List[str],
    start: str,
    end: str,
    fetch_fn: Callable[[List[str], str, str], Dict[str, pd.DataFrame]],
    cache_dir: Path = _CACHE_DIR,
    source_name: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Return OHLCV data for *tickers* over [start, end], using cache where possible.

    Args:
        tickers    : Ticker symbols to load.
        start      : Start date "YYYY-MM-DD" (inclusive).
        end        : End date "YYYY-MM-DD" (inclusive).
        fetch_fn   : Callable(tickers, start, end) -> Dict[ticker, DataFrame].
                     Called only for tickers that need downloading.
        cache_dir  : Root directory for parquet files.
        source_name: If given, tags newly-fetched rows with a "source"
                     column (e.g. "tiingo", "alpaca") for provenance —
                     useful once multiple providers are merged into one
                     ticker's cache. Omit to preserve the original
                     column-free schema exactly (backward compatible).

    Returns:
        Dict mapping ticker -> DataFrame sliced to [start, end].
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    result: Dict[str, pd.DataFrame] = {}
    need_fetch: List[str] = []

    # ── Phase 1: check cache ──────────────────────────────────────────────
    for ticker in tickers:
        cached = _read_cache(ticker, cache_dir)
        if cached is not None and _covers(cached, start_dt, end_dt):
            result[ticker] = cached.loc[start_dt:end_dt]
        else:
            need_fetch.append(ticker)

    if need_fetch:
        logger.info(
            "DataCache: %d / %d tickers need downloading",
            len(need_fetch), len(tickers),
        )
    else:
        logger.info("DataCache: all %d tickers loaded from cache", len(tickers))
        return result

    # ── Phase 2: fetch missing ────────────────────────────────────────────
    fetched = fetch_fn(need_fetch, start, end)

    # ── Phase 3: merge, save, slice ───────────────────────────────────────
    for ticker in need_fetch:
        new_df = fetched.get(ticker)
        if new_df is None or new_df.empty:
            continue

        if source_name is not None:
            new_df = new_df.copy()
            new_df["source"] = source_name

        existing = _read_cache(ticker, cache_dir)
        if existing is not None:
            if "source" in new_df.columns and "source" not in existing.columns:
                existing = existing.copy()
                existing["source"] = None
            merged = pd.concat([existing, new_df])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()
        else:
            merged = new_df.sort_index()

        _write_cache(ticker, merged, cache_dir)
        result[ticker] = merged.loc[start_dt:end_dt]

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cache_path(ticker: str, cache_dir: Path) -> Path:
    # Replace characters unsafe for filenames (e.g. BRK/B -> BRK_B)
    safe = ticker.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe}.parquet"


def _read_cache(ticker: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    path = _cache_path(ticker, cache_dir)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:
        logger.warning("DataCache: corrupt cache for %s, will re-download — %s", ticker, exc)
        path.unlink(missing_ok=True)
        return None


def _write_cache(ticker: str, df: pd.DataFrame, cache_dir: Path) -> None:
    path = _cache_path(ticker, cache_dir)
    df.to_parquet(path, engine="pyarrow")


def _covers(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Return True if df's date range fully covers [start, end]."""
    if df.empty:
        return False
    return df.index.min() <= start and df.index.max() >= end


def find_gaps(df: pd.DataFrame, start: str, end: str) -> List[pd.Timestamp]:
    """Return business days in [start, end] missing from df's index.

    `_covers()` above only checks that the cached min/max bounds span the
    requested range — it says nothing about holes *inside* that range. This
    matters once multiple providers with different missing-data patterns
    are merged into one ticker's cache. Uses a plain business-day calendar
    (not a real NYSE holiday calendar) as a deliberately simple
    approximation — good enough to flag suspicious holes for manual review,
    not a certifiably exact trading-day count.
    """
    expected = pd.bdate_range(start, end)
    present = set(df.index)
    return [d for d in expected if d not in present]
