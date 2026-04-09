"""DataAgent: fetches, validates, and stores OHLCV data."""
from typing import Optional

import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class DataAgent(BaseAgent):
    """Loads OHLCV data for a single ticker and validates it.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        start_date: ISO date string for the start of the range.
        end_date: ISO date string for the end of the range.
        context_key: Which context key to write the data into (default "data").
        data_override: Optional pre-built DataFrame; skips the network fetch.
    """

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        context_key: str = "data",
        data_override: Optional[pd.DataFrame] = None,
    ) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.context_key = context_key
        self._data_override = data_override

    def run(self, context: dict) -> dict:
        """Fetch data, validate it, and store in context[context_key]."""
        logger.info(
            "DataAgent: fetching %s [%s → %s] → context['%s']",
            self.ticker,
            self.start_date,
            self.end_date,
            self.context_key,
        )
        data = (
            self._data_override.copy()
            if self._data_override is not None
            else self._fetch_data()
        )
        self._validate(data)
        context[self.context_key] = data
        logger.info("DataAgent: stored %d rows for %s", len(data), self.ticker)
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_data(self) -> pd.DataFrame:
        """Download OHLCV data via yfinance."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required: pip install yfinance") from exc

        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for ticker '{self.ticker}'")

        # yfinance >= 0.2 may return MultiIndex columns when multiple tickers requested
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    def _validate(self, data: pd.DataFrame) -> None:
        """Assert data integrity; raises AssertionError on failure."""
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in data.columns]
        assert not missing, f"Missing OHLCV columns: {missing}"

        assert isinstance(data.index, pd.DatetimeIndex), (
            "Index must be a DatetimeIndex"
        )
        assert data.index.is_monotonic_increasing, "Index must be sorted ascending"
        assert not data["Close"].isna().any(), "Close price contains NaN values"
        assert (data["Close"] > 0).all(), "All Close prices must be positive"
        assert len(data) >= 30, (
            f"Insufficient rows ({len(data)}); need at least 30"
        )
        logger.info("DataAgent: validation passed for %s", self.ticker)
