"""tiingo_loader.py — deep historical OHLCV backfill via the Tiingo API.

Why Tiingo (and not Stooq)
-----------------------------
Stooq — this project's originally planned primary deep-history source — is,
as of this writing, gated behind a JavaScript proof-of-work bot-challenge
across its entire site (bulk archive AND per-ticker CSV endpoint), not
scriptable via plain HTTP anymore. Tiingo's free tier (50 symbols/hour,
30+ years of history, delisted-ticker coverage with end dates) is a normal
authenticated REST API and became the primary source instead.

Adjustment convention
-----------------------
Uses Tiingo's `adj*` fields (CRSP-style split+dividend adjusted), matching
`utils/alpaca_loader.py`'s `adjustment="all"` convention, so data from both
sources is comparable when merged/cross-validated
(scripts/validate_data_sources.py).

Rate limiting
--------------
Free tier: 50 symbols/hour. `fetch_bars` paces requests accordingly via a
simple sleep-based throttle — intended for a one-time/infrequent bulk
backfill (e.g. the ~842-ticker point-in-time universe from
utils/sp500_membership.py), not a per-run API call pattern.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests

from utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
_REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
_COL_MAP = {
    "adjOpen": "Open",
    "adjHigh": "High",
    "adjLow": "Low",
    "adjClose": "Close",
    "adjVolume": "Volume",
}
_FREE_TIER_SYMBOLS_PER_HOUR = 50


def fetch_bars(
    tickers: List[str],
    start: str,
    end: str,
    api_key: str,
    pace: bool = True,
    http_get: Callable = requests.get,
) -> Dict[str, pd.DataFrame]:
    """Download adjusted daily OHLCV bars for a list of tickers from Tiingo.

    Args:
        tickers: Ticker symbols.
        start, end: "YYYY-MM-DD" date strings (inclusive).
        api_key: Tiingo API token.
        pace: If True (default), sleep between requests to respect the free
            tier's 50-symbols/hour limit. Set False only for tests/small runs.
        http_get: Injectable GET callable (for testing); defaults to `requests.get`.

    Returns:
        Dict mapping ticker -> DataFrame with tz-naive DatetimeIndex and
        columns [Open, High, Low, Close, Volume] (adjusted). Tickers with no
        data, or that error, are absent (logged, not raised) — a bulk
        backfill shouldn't abort over one bad symbol.
    """
    results: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    interval_seconds = 3600.0 / _FREE_TIER_SYMBOLS_PER_HOUR

    for i, ticker in enumerate(tickers):
        if pace and i > 0:
            time.sleep(interval_seconds)
        try:
            df = _fetch_one(ticker, start, end, api_key, http_get)
        except Exception as exc:
            logger.warning("TiingoLoader: failed to fetch %s — %s", ticker, exc)
            df = None

        if df is not None and not df.empty:
            results[ticker] = df
        else:
            failed.append(ticker)

    if failed:
        logger.warning(
            "TiingoLoader: %d/%d tickers had errors or no data: %s",
            len(failed), len(tickers), failed[:10],
        )
    logger.info("TiingoLoader: downloaded %d / %d tickers", len(results), len(tickers))
    return results


def _fetch_one(
    ticker: str, start: str, end: str, api_key: str, http_get: Callable,
) -> Optional[pd.DataFrame]:
    url = _BASE_URL.format(ticker=ticker.lower())
    response = http_get(
        url, params={"startDate": start, "endDate": end, "token": api_key}, timeout=30,
    )
    response.raise_for_status()
    records = response.json()
    if not records:
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df = df.rename(columns=_COL_MAP)
    df = df[[c for c in _REQUIRED_COLS if c in df.columns]]
    return df.astype(float)
