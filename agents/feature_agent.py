"""FeatureAgent: computes backward-looking technical indicators from OHLCV data."""
from typing import List

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureAgent(BaseAgent):
    """Computes technical features; all rolling operations are strictly backward-looking.

    Args:
        sma_windows: List of SMA periods for price-ratio features.
        rsi_period: Look-back window for RSI computation.
        bb_window: Look-back window for Bollinger Bands.
        momentum_period: Look-back period for momentum feature.
    """

    def __init__(
        self,
        sma_windows: List[int] = None,
        rsi_period: int = 14,
        bb_window: int = 20,
        momentum_period: int = 5,
    ) -> None:
        self.sma_windows = sma_windows if sma_windows is not None else [5, 10, 20]
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.momentum_period = momentum_period

    def run(self, context: dict) -> dict:
        """Compute features, drop warm-up NaN rows, store in context['features']."""
        assert context.get("data") is not None, (
            "FeatureAgent requires context['data']"
        )
        data = context["data"]
        features = self._compute_features(data)
        features = features.dropna()
        assert len(features) > 0, (
            "No valid feature rows after dropping NaN — "
            "data may be too short or all prices are constant"
        )
        self._validate(features)
        context["features"] = features
        logger.info(
            "FeatureAgent: %d features × %d rows (dropped %d warm-up rows)",
            features.shape[1],
            features.shape[0],
            len(data) - len(features),
        )
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix; every column uses only past prices."""
        close = data["Close"]
        volume = data["Volume"]
        feats = pd.DataFrame(index=data.index)

        # Price / SMA ratios (close relative to its own moving average)
        for w in self.sma_windows:
            sma = close.rolling(window=w, min_periods=w).mean()
            feats[f"SMA_{w}_ratio"] = close / sma

        # RSI
        feats["RSI"] = self._rsi(close, self.rsi_period)

        # Bollinger Band position (0 = lower band, 1 = upper band)
        feats["BB_pos"] = self._bollinger_pos(close, self.bb_window)

        # Momentum: return over the last N days
        feats[f"MOM_{self.momentum_period}"] = (
            close / close.shift(self.momentum_period) - 1
        )

        # Lagged 1-day return (already realised — no leakage)
        feats["RET_1"] = close.pct_change()

        # Volume relative to its 10-day average
        vol_ma = volume.rolling(window=10, min_periods=10).mean()
        feats["VOL_ratio"] = volume / vol_ma.where(vol_ma != 0, np.nan)

        return feats

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        """Wilder's RSI — strictly backward-looking."""
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss.where(loss != 0, np.nan)
        return (100 - 100 / (1 + rs)).rename("RSI")

    @staticmethod
    def _bollinger_pos(close: pd.Series, window: int) -> pd.Series:
        """Position of close within Bollinger Bands (capped to [0, 1])."""
        sma = close.rolling(window=window, min_periods=window).mean()
        std = close.rolling(window=window, min_periods=window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        band_width = (upper - lower).where((upper - lower) > 0, np.nan)
        return ((close - lower) / band_width).rename("BB_pos")

    def _validate(self, features: pd.DataFrame) -> None:
        """Sanity checks on the output feature matrix."""
        assert "label" not in features.columns, (
            "Feature matrix must not contain the label column"
        )
        assert isinstance(features.index, pd.DatetimeIndex), (
            "Feature index must be DatetimeIndex"
        )
        assert features.index.is_monotonic_increasing, (
            "Feature index must be sorted ascending"
        )
