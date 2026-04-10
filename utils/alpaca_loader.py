"""alpaca_loader.py — bulk historical OHLCV download via Alpaca Data API.

Why Alpaca instead of yfinance
-------------------------------
yfinance relies on an unofficial Yahoo Finance endpoint that aggressively
rate-limits bulk requests (429 errors become frequent with > ~50 tickers).
Alpaca provides an official, documented REST API backed by IEX/SIP feeds
with generous rate limits suitable for universe-scale downloads.

Timestamp handling
------------------
Alpaca returns daily bar timestamps as timezone-aware UTC datetimes
representing midnight US/Eastern (e.g. 2024-01-02 05:00:00+00:00 = midnight
Eastern on Jan 2).  We convert to US/Eastern and strip timezone to produce
a clean, tz-naive DatetimeIndex aligned with standard market conventions.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)

_EASTERN_TZ = "America/New_York"
_REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
_COL_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def fetch_universe_bars(
    tickers: List[str],
    start: str,
    end: str,
    api_key: str,
    secret_key: str,
    batch_size: int = 200,
    feed: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Download adjusted daily OHLCV bars for a list of tickers from Alpaca.

    Args:
        tickers    : List of Alpaca-format symbols (e.g. "BRK/B", "AAPL").
        start      : Start date string "YYYY-MM-DD" (inclusive).
        end        : End date string "YYYY-MM-DD" (inclusive).
        api_key    : Alpaca API key.
        secret_key : Alpaca secret key.
        batch_size : Max symbols per API request (default 200).
        feed       : Data feed override ("iex", "sip", "delayed_sip").
                     Leave None to use the account default.

    Returns:
        Dict mapping ticker → pd.DataFrame with DatetimeIndex and columns
        [Open, High, Low, Close, Volume].  Tickers with no data are absent.
    """
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    results: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    for batch_idx, batch in enumerate(batches):
        logger.info(
            "AlpacaLoader: batch %d/%d  (%d symbols)",
            batch_idx + 1,
            len(batches),
            len(batch),
        )
        try:
            request_kwargs: dict = dict(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment="all",
            )
            if feed:
                request_kwargs["feed"] = feed

            bars = client.get_stock_bars(StockBarsRequest(**request_kwargs))
            raw_df = bars.df
        except Exception as exc:
            # Batch rejected (e.g. one invalid symbol): retry individually
            logger.warning(
                "AlpacaLoader: batch %d failed (%s) — retrying %d tickers one-by-one",
                batch_idx + 1, exc, len(batch),
            )
            raw_df = _retry_individually(client, batch, start, end, feed)
            if raw_df.empty:
                failed.extend(batch)
                continue

        if raw_df.empty:
            logger.warning("AlpacaLoader: empty response for batch %d", batch_idx + 1)
            continue

        # raw_df index: MultiIndex (symbol, timestamp [UTC, tz-aware])
        for symbol in raw_df.index.get_level_values("symbol").unique():
            try:
                ticker_df = raw_df.xs(symbol, level="symbol").copy()
                ticker_df = _normalise(ticker_df)
                results[symbol] = ticker_df
            except Exception as exc:
                logger.warning("AlpacaLoader: failed to process %s — %s", symbol, exc)
                failed.append(symbol)

    if failed:
        logger.warning(
            "AlpacaLoader: %d symbols had errors or no data: %s",
            len(failed),
            failed[:10],
        )

    logger.info(
        "AlpacaLoader: downloaded %d / %d symbols", len(results), len(tickers)
    )
    return results


def _retry_individually(
    client: StockHistoricalDataClient,
    tickers: List[str],
    start: str,
    end: str,
    feed: Optional[str],
) -> pd.DataFrame:
    """Retry a failed batch one ticker at a time; concatenate successes."""
    parts = []
    for ticker in tickers:
        try:
            kw: dict = dict(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment="all",
            )
            if feed:
                kw["feed"] = feed
            bars = client.get_stock_bars(StockBarsRequest(**kw))
            if not bars.df.empty:
                parts.append(bars.df)
        except Exception as exc:
            logger.warning("AlpacaLoader: skipping %s — %s", ticker, exc)
    return pd.concat(parts) if parts else pd.DataFrame()


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Alpaca response DataFrame to a clean OHLCV DataFrame."""
    # Convert UTC timestamps → US/Eastern date, strip timezone
    df.index = pd.to_datetime(
        df.index.tz_convert(_EASTERN_TZ).date
    )
    df.index.name = None

    # Rename lowercase alpaca columns to capitalised OHLCV
    df = df.rename(columns=_COL_MAP)

    # Keep only standard OHLCV columns
    df = df[[c for c in _REQUIRED_COLS if c in df.columns]]

    # Ensure float dtype for all columns
    df = df.astype(float)

    return df
