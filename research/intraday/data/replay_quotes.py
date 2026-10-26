"""Synthetic ``Quote`` objects from the cached per-window NBBO summaries (decision D3).

The summaries (``research/penny/etl/quotes.py::summarise_quotes``) hold, per
symbol-day and 60 s window, the median mid, half-spread and inside notional.
Quotes are raw (unadjusted) prices and need no conversion. The notional was
formed as ``price x size x 100`` ("sizes are in round lots", ``quotes.py:139-143``),
so ``notional / price`` is shares; ``size_scale`` exists for when that unit is
verified against current Alpaca docs (default 1.0, unverified: see the data report).
Each quote is stamped at its window's end.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from research.penny.etl.quotes import DECISION_MINUTE, ENTRY_MINUTE, EXIT_MINUTE

from research.intraday.contracts.events import Quote
from research.intraday.data.cache import check_unique
from research.intraday.data.clock import et_to_utc
from research.intraday.data.paths import quote_summary_files

QUOTE_WINDOWS_ET: dict[str, str] = {
    "decision": DECISION_MINUTE, "entry": ENTRY_MINUTE, "exit": EXIT_MINUTE}
WINDOW_SECONDS: int = 60  # fetch_quotes_for_minute(..., window_seconds=60)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "symbol", "date", "window", "mid", "half_spread_bps", "bid_notional", "ask_notional")


def read_quote_summaries(data_root: Path, d: date, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Both summary files' rows for ``d`` and ``symbols``; a duplicate (symbol, window) raises."""
    parts = []
    if symbols:
        filters = [("date", "==", pd.Timestamp(d)), ("symbol", "in", list(symbols))]
        parts = [pd.read_parquet(p, columns=list(SUMMARY_COLUMNS), filters=filters)
                 for p in quote_summary_files(data_root) if p.is_file()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=object if c in ("symbol", "window") else float)
                             for c in SUMMARY_COLUMNS})
    df = pd.concat(parts, ignore_index=True)
    check_unique(df, ["symbol", "window"], f"quote summaries for {d}")
    return df.sort_values(["symbol", "window"], kind="mergesort").reset_index(drop=True)


def _quote(symbol: str, ts: datetime, mid: float, hs: float, bid_notional: float,
           ask_notional: float, size_scale: float) -> Quote | None:
    values = (mid, hs, bid_notional, ask_notional)
    if not all(math.isfinite(v) for v in values):
        return None
    bid, ask = mid * (1 - hs / 1e4), mid * (1 + hs / 1e4)
    if not (bid > 0 and ask > 0):
        return None
    bid_size = int(round(bid_notional / bid * size_scale))
    ask_size = int(round(ask_notional / ask * size_scale))
    if bid_size <= 0 or ask_size <= 0:
        return None
    try:
        return Quote(ts=ts, symbol=symbol, bid=float(bid), ask=float(ask),
                     bid_size=bid_size, ask_size=ask_size)
    except ValueError:
        return None


def synthetic_quotes(summary: pd.DataFrame, d: date,
                     size_scale: float = 1.0) -> tuple[dict[tuple[str, datetime], Quote], int]:
    """Quotes keyed ``(symbol, window end)``, and the count of rows skipped as non-markets."""
    stamps = {w: et_to_utc(d, m) + timedelta(seconds=WINDOW_SECONDS)
              for w, m in QUOTE_WINDOWS_ET.items()}
    out: dict[tuple[str, datetime], Quote] = {}
    skipped = 0
    ordered = summary.sort_values(["symbol", "window"], kind="mergesort")
    for row in ordered.itertuples(index=False):
        ts = stamps[str(row.window)]
        q = _quote(str(row.symbol), ts, float(row.mid), float(row.half_spread_bps),
                   float(row.bid_notional), float(row.ask_notional), size_scale)
        if q is None:
            skipped += 1
        else:
            out[(q.symbol, ts)] = q
    return out, skipped
