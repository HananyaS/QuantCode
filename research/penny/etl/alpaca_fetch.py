"""alpaca_fetch.py — resilient bar fetching against the free SIP tier.

Two properties matter here and neither is optional:

1. ``feed="sip"``. IEX alone is ~2.5% of consolidated volume, which would make
   relative volume -- the most load-bearing feature in the study -- meaningless.
   The free tier serves SIP for any window ending at least 15 minutes ago.

2. Rejected symbols must not poison the batch. The universe deliberately
   includes delisted names, and Alpaca's asset registry carries CUSIPs, rights
   and preferred lines it will not price. A single rejected symbol fails the
   entire multi-symbol request, so an unguarded fetch silently loses hundreds
   of good symbols per batch.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_INVALID_SYMBOL_RE = re.compile(r"invalid symbol:\s*([^\s\"'},]+)", re.IGNORECASE)

MAX_REJECT_DROPS = 60
MAX_BISECT_DEPTH = 7
DEFAULT_BATCH_SIZE = 200

# Free-tier courtesy pause between requests; the documented ceiling is 200/min.
REQUEST_PAUSE_SECONDS = 0.05


def make_data_client(api_key: str, secret_key: str):
    """Build the read-only market-data client.

    Deliberately never returns a TradingClient: research code must not be able
    to place an order even by accident.
    """
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def fetch_with_rejects(
    request: Callable[[List[str]], pd.DataFrame],
    batch: List[str],
    depth: int = 0,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Run ``request``, isolating and discarding symbols the API rejects.

    Alpaca names the offending symbol, so the cheap path parses it, drops it and
    retries -- O(bad symbols) requests. When a failure cannot be attributed to a
    named symbol the batch is bisected instead, so one opaque error costs a
    handful of extra calls rather than the whole batch.
    """
    remaining = list(batch)
    rejected: List[str] = []

    for _ in range(MAX_REJECT_DROPS):
        if not remaining:
            return None, rejected
        try:
            return request(remaining), rejected
        except Exception as exc:  # noqa: BLE001
            match = _INVALID_SYMBOL_RE.search(str(exc))
            if not match:
                break
            bad = match.group(1)
            if bad not in remaining:
                break
            remaining.remove(bad)
            rejected.append(bad)
    else:
        return None, rejected + remaining

    if len(remaining) == 1 or depth >= MAX_BISECT_DEPTH:
        return None, rejected + remaining

    mid = len(remaining) // 2
    frames: List[pd.DataFrame] = []
    for half in (remaining[:mid], remaining[mid:]):
        df, dropped = fetch_with_rejects(request, half, depth + 1)
        rejected.extend(dropped)
        if df is not None and not df.empty:
            frames.append(df)
    return (pd.concat(frames) if frames else None), rejected


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the (symbol, timestamp) MultiIndex into tidy columns."""
    out = df.reset_index()
    ts = out["timestamp"]
    if getattr(ts.dt, "tz", None) is not None:
        out["timestamp"] = ts.dt.tz_convert("America/New_York")
    else:
        out["timestamp"] = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    return out


def fetch_daily(
    client,
    symbols: List[str],
    start: str,
    end: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_every: int = 10,
) -> Tuple[pd.DataFrame, List[str]]:
    """Adjusted daily bars for many symbols over one window.

    Returns (tidy frame with a `date` column, rejected symbols).
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    def request(batch: List[str]) -> pd.DataFrame:
        time.sleep(REQUEST_PAUSE_SECONDS)
        return client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment="all",
                feed="sip",
            )
        ).df

    frames: List[pd.DataFrame] = []
    rejected: List[str] = []
    batches = [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]

    for idx, batch in enumerate(batches, start=1):
        if idx % progress_every == 0 or idx == 1:
            logger.info("daily %s..%s: batch %d/%d", start, end, idx, len(batches))
        df, dropped = fetch_with_rejects(request, batch)
        rejected.extend(dropped)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), rejected

    out = _normalise(pd.concat(frames))
    out["date"] = out["timestamp"].dt.normalize().dt.tz_localize(None)
    return out.drop(columns=["timestamp"]), rejected


def fetch_minute_for_day(
    client,
    symbols: List[str],
    day: str,
    start_et: str = "09:25",
    end_et: str = "16:05",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[pd.DataFrame, List[str]]:
    """Minute bars for a set of symbols on a single session.

    Grouping the request by *day* rather than by symbol is what makes intraday
    labelling affordable: one request covers every candidate that day, so the
    whole study costs roughly one request per trading day instead of one per
    symbol-day.

    The window starts before 09:30 so the opening auction print is included,
    and ends after 16:00 to capture the closing cross.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    start_ts = pd.Timestamp(f"{day} {start_et}", tz="America/New_York")
    end_ts = pd.Timestamp(f"{day} {end_et}", tz="America/New_York")

    def request(batch: List[str]) -> pd.DataFrame:
        time.sleep(REQUEST_PAUSE_SECONDS)
        return client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start_ts.tz_convert("UTC").to_pydatetime(),
                end=end_ts.tz_convert("UTC").to_pydatetime(),
                adjustment="all",
                feed="sip",
            )
        ).df

    frames: List[pd.DataFrame] = []
    rejected: List[str] = []
    for i in range(0, len(symbols), batch_size):
        df, dropped = fetch_with_rejects(request, symbols[i : i + batch_size])
        rejected.extend(dropped)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), rejected

    out = _normalise(pd.concat(frames))
    out["date"] = pd.Timestamp(day)
    out["minute_et"] = out["timestamp"].dt.strftime("%H:%M")
    return out, rejected
