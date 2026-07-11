"""UniverseAgent: assembles and validates OHLCV data for an asset universe.

Data sources
------------
data_source="alpaca"  (default)
    Uses the official Alpaca Data API via alpaca-py.  Requires api_key and
    secret_key.  Supports bulk downloads without hitting Yahoo Finance
    rate limits.

data_source="yfinance"
    Falls back to yfinance for small universes or offline testing.

Survivorship-bias note
-----------------------
When tickers="sp500" the current S&P 500 constituent list is fetched from
Wikipedia.  Companies removed from the index during the study period (due to
acquisition, bankruptcy, or demotion) are NOT included, introducing a mild
upward performance bias.  This limitation is documented here and cannot be
eliminated without a point-in-time index membership dataset.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
_MIN_ROWS = 30


class UniverseAgent(BaseAgent):
    """Downloads and aligns OHLCV data for a universe of tickers.

    Writes
    ------
    context['universe']        : List[str]
    context['universe_data']   : Dict[str, pd.DataFrame]  — aligned OHLCV
    context['benchmark_data']  : pd.DataFrame

    Args:
        tickers      : Explicit list of symbols, or "sp500" to auto-fetch.
        start_date   : Inclusive start date ("YYYY-MM-DD").
        end_date     : Inclusive end date ("YYYY-MM-DD").
        benchmark    : Benchmark ticker (default "SPY").
        min_assets   : Minimum valid assets required before raising.
        data_source  : "alpaca" (default) or "yfinance".
        alpaca_key   : Alpaca API key (required when data_source="alpaca").
        alpaca_secret: Alpaca secret key.
        alpaca_feed  : Optional feed override ("iex", "sip").
    """

    def __init__(
        self,
        tickers: List[str] | str,
        start_date: str,
        end_date: str,
        benchmark: str = "SPY",
        min_assets: int = 5,
        min_history_days: int = 1260,
        data_source: str = "alpaca",
        alpaca_key: Optional[str] = None,
        alpaca_secret: Optional[str] = None,
        alpaca_feed: Optional[str] = None,
    ) -> None:
        self.tickers = tickers  # may be a list or the string "sp500"
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.min_assets = min_assets
        self.min_history_days = min_history_days  # drop tickers with shorter history
        self.data_source = data_source
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret
        self.alpaca_feed = alpaca_feed

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        tickers = self._resolve_tickers()

        raw = self._download_all(tickers)
        valid: Dict[str, pd.DataFrame] = {}
        for ticker, df in raw.items():
            try:
                self._validate(df, ticker, self.min_history_days)
                valid[ticker] = df
            except Exception as exc:
                logger.warning("UniverseAgent: dropping %s — %s", ticker, exc)

        assert len(valid) >= self.min_assets, (
            f"Only {len(valid)} valid tickers after validation "
            f"(need >= {self.min_assets})"
        )

        aligned = self._align(valid)

        bm_raw = self._download_single(self.benchmark)
        self._validate(bm_raw, self.benchmark)

        n_dates = len(next(iter(aligned.values())))
        logger.info(
            "UniverseAgent: %d tickers x %d days  source=%s  dropped=%d",
            len(aligned),
            n_dates,
            self.data_source,
            len(raw) - len(valid),
        )

        context["universe"] = list(aligned.keys())
        context["universe_data"] = aligned
        context["benchmark_data"] = bm_raw
        return context

    # ------------------------------------------------------------------
    # Ticker resolution
    # ------------------------------------------------------------------

    def _resolve_tickers(self) -> List[str]:
        if isinstance(self.tickers, str) and self.tickers.lower() == "sp500":
            logger.info("UniverseAgent: fetching S&P 500 constituent list from Wikipedia")
            from utils.sp500 import get_sp500_tickers
            tickers = get_sp500_tickers()
            logger.info("UniverseAgent: %d S&P 500 tickers retrieved", len(tickers))
            return tickers
        assert isinstance(self.tickers, list) and len(self.tickers) > 0, (
            "tickers must be a non-empty list or the string 'sp500'"
        )
        return list(self.tickers)

    # ------------------------------------------------------------------
    # Data download
    # ------------------------------------------------------------------

    def _download_all(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        from utils.data_cache import load_or_fetch
        return load_or_fetch(
            tickers=tickers,
            start=self.start_date,
            end=self.end_date,
            fetch_fn=self._raw_fetch,
        )

    def _download_single(self, ticker: str) -> pd.DataFrame:
        results = self._download_all([ticker])
        assert ticker in results, f"Benchmark {ticker} not returned by data source"
        return results[ticker]

    def _raw_fetch(
        self, tickers: List[str], start: str, end: str
    ) -> Dict[str, pd.DataFrame]:
        """Direct download (no cache) — used as callback by data_cache."""
        if self.data_source == "alpaca":
            return self._download_alpaca(tickers, start, end)
        return self._download_yfinance(tickers, start, end)

    def _download_alpaca(
        self, tickers: List[str], start: str, end: str
    ) -> Dict[str, pd.DataFrame]:
        assert self.alpaca_key and self.alpaca_secret, (
            "alpaca_key and alpaca_secret are required when data_source='alpaca'"
        )
        from utils.alpaca_loader import fetch_universe_bars
        return fetch_universe_bars(
            tickers=tickers,
            start=start,
            end=end,
            api_key=self.alpaca_key,
            secret_key=self.alpaca_secret,
            feed=self.alpaca_feed,
        )

    _YFINANCE_BATCH_SIZE = 50

    def _download_yfinance(
        self, tickers: List[str], start: str, end: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch via yfinance's own multi-ticker batch API.

        A prior implementation spawned a thread per ticker, each independently
        calling `yf.download(single_ticker)`. yfinance's internal session/cache
        state is NOT safely shared across manually-spawned threads this way —
        in production this returned IDENTICAL (cross-contaminated) data for
        every requested ticker. `yf.download(tickers=[...])` is yfinance's own
        tested multi-ticker code path and is safe to call directly.
        """
        import yfinance as yf

        results: Dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), self._YFINANCE_BATCH_SIZE):
            batch = tickers[i : i + self._YFINANCE_BATCH_SIZE]
            try:
                raw = yf.download(
                    tickers=batch, start=start, end=end,
                    progress=False, auto_adjust=True,
                    group_by="ticker", threads=True,
                )
            except Exception as exc:
                logger.warning("UniverseAgent(yfinance): batch failed — %s", exc)
                continue

            if raw is None or raw.empty:
                continue

            for t in batch:
                try:
                    df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                    df = df.dropna(how="all")
                    if not df.empty:
                        results[t] = df
                except KeyError:
                    logger.warning("UniverseAgent(yfinance): %s missing from batch response", t)
        return results

    # ------------------------------------------------------------------
    # Validation & alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame, ticker: str, min_rows: int = _MIN_ROWS) -> None:
        assert not df.empty, f"{ticker}: empty DataFrame"
        missing = _REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"{ticker}: missing columns {missing}"
        assert isinstance(df.index, pd.DatetimeIndex), f"{ticker}: index must be DatetimeIndex"
        assert df.index.is_monotonic_increasing, f"{ticker}: index not sorted"
        assert df["Close"].notna().all(), f"{ticker}: NaN in Close"
        assert (df["Close"] > 0).all(), f"{ticker}: non-positive Close"
        assert len(df) >= min_rows, (
            f"{ticker}: only {len(df)} rows (need >= {min_rows})"
        )

    @staticmethod
    def _align(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Inner-join all assets on common trading dates."""
        common: Optional[pd.Index] = None
        for df in data.values():
            common = df.index if common is None else common.intersection(df.index)
        assert common is not None and len(common) >= _MIN_ROWS, (
            f"Only {0 if common is None else len(common)} common dates after alignment"
        )
        return {ticker: df.loc[common].copy() for ticker, df in data.items()}
