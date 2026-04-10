"""CrossSectionalFeatureAgent: per-asset technical features + cross-sectional normalization.

Leakage guarantee
-----------------
* All per-asset features use only rolling/shift operations with strictly
  past data (min_periods = window, no center=True, no look-ahead).
* Cross-sectional rank and z-score are computed *within* a single calendar
  date across assets — they use no future dates.
* warm-up rows (where any feature is NaN) are dropped after stacking.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class CrossSectionalFeatureAgent(BaseAgent):
    """Builds a (date, ticker) MultiIndex feature DataFrame.

    Per-asset features
    ------------------
    ret_Nd       : N-day simple return
    vol_20d      : 20-day realised volatility (std of daily returns)
    rsi          : 14-period Wilder RSI
    sma_N_ratio  : close / N-day SMA

    Cross-sectional features (added when cross_sectional=True)
    -----------------------------------------------------------
    rank_{feat}   : percentile rank across assets, per date  [0, 1]
    zscore_{feat} : z-score across assets, per date

    Writes
    ------
    context['cs_features'] : pd.DataFrame with MultiIndex(date, ticker)
    """

    def __init__(
        self,
        returns_windows: List[int] | None = None,
        vol_window: int = 20,
        rsi_period: int = 14,
        sma_windows: List[int] | None = None,
        cross_sectional: bool = True,
    ) -> None:
        self.returns_windows = returns_windows or [1, 5, 10]
        self.vol_window = vol_window
        self.rsi_period = rsi_period
        self.sma_windows = sma_windows or [10, 20]
        self.cross_sectional = cross_sectional

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        assert context.get("universe_data") is not None, (
            "CrossSectionalFeatureAgent requires context['universe_data']"
        )
        universe_data: Dict[str, pd.DataFrame] = context["universe_data"]

        per_asset = {
            ticker: self._compute_asset_features(df)
            for ticker, df in universe_data.items()
        }

        # Stack: (ticker, date) -> (date, ticker)
        stacked = (
            pd.concat(per_asset, names=["ticker", "date"])
            .swaplevel()
            .sort_index()
        )
        stacked = stacked.dropna()
        assert len(stacked) > 0, "No feature rows remain after dropping NaN warm-up rows"
        self._validate_no_leakage(stacked)

        if self.cross_sectional:
            stacked = self._add_cross_sectional(stacked)

        context["cs_features"] = stacked
        n_dates = stacked.index.get_level_values("date").nunique()
        n_assets = stacked.index.get_level_values("ticker").nunique()
        logger.info(
            "CSFeatureAgent: %d features x %d dates x %d assets",
            stacked.shape[1],
            n_dates,
            n_assets,
        )
        return context

    # ------------------------------------------------------------------
    # Per-asset feature computation
    # ------------------------------------------------------------------

    def _compute_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        feats = pd.DataFrame(index=df.index)

        # Momentum returns
        for w in self.returns_windows:
            feats[f"ret_{w}d"] = close.pct_change(w)

        # Realised volatility
        feats[f"vol_{self.vol_window}d"] = (
            close.pct_change()
            .rolling(window=self.vol_window, min_periods=self.vol_window)
            .std()
        )

        # RSI
        feats["rsi"] = self._rsi(close, self.rsi_period)

        # SMA ratios
        for w in self.sma_windows:
            sma = close.rolling(window=w, min_periods=w).mean()
            feats[f"sma_{w}_ratio"] = close / sma

        return feats

    # ------------------------------------------------------------------
    # Cross-sectional normalization
    # ------------------------------------------------------------------

    def _add_cross_sectional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append rank and z-score columns, computed per date across assets."""
        base_cols = list(df.columns)

        rank_parts: List[pd.Series] = []
        zscore_parts: List[pd.Series] = []

        for col in base_cols:
            by_date = df[col].groupby(level="date")
            rank_parts.append(
                by_date.transform(lambda x: x.rank(pct=True)).rename(f"rank_{col}")
            )
            zscore_parts.append(
                by_date.transform(self._zscore).rename(f"zscore_{col}")
            )

        return pd.concat([df] + rank_parts + zscore_parts, axis=1)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss.where(loss != 0, np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _zscore(x: pd.Series) -> pd.Series:
        std = x.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / std

    @staticmethod
    def _validate_no_leakage(df: pd.DataFrame) -> None:
        assert "label" not in df.columns, "Feature matrix must not contain the label column"
        assert isinstance(df.index, pd.MultiIndex), "Expected MultiIndex (date, ticker)"
        dates = df.index.get_level_values("date")
        assert np.issubdtype(dates.dtype, np.datetime64), (
            f"date level must be datetime, got {dates.dtype}"
        )
