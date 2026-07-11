"""TimingSignalAgent: binary long/flat regime signal for a single instrument.

Unlike the cross-sectional pipeline (agents/cs_feature_agent.py), this is a
single-instrument, rule-based signal — no model fitting. Each component is a
simple, literature-grounded trend/regime rule (Faber's 10-month SMA,
absolute momentum, realized-vol regime, VIX regime); components combine into
one binary signal via a configurable vote rule.

Causality: every rolling/pct_change computation at date t uses only data
through date t (min_periods=window, standard pandas rolling semantics — no
explicit shift needed here). The one-day execution lag (using yesterday's
signal to trade today) is applied downstream in TimingBacktestAgent, not
here — this agent's output is the *decision*, not the *position*.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent

_COMBINE_MODES = ("sma_only", "all_agree", "majority")
_TRADING_DAYS_PER_YEAR = 252


class TimingSignalAgent(BaseAgent):
    """Generates a binary (1=long, 0=flat) regime signal for one instrument.

    Writes
    ------
    context['timing_signal']            : pd.Series (0/1), indexed by date
    context['timing_signal_components'] : pd.DataFrame, one 0/1 column per
                                           enabled component (diagnostics)

    Args:
        ticker        : Key into context['universe_data'] for the price series.
        sma_window    : Trailing SMA window (trading days). Faber's published
                         rule uses 10 months ≈ 200 trading days.
        mom_window    : Absolute-momentum lookback (trading days). None disables
                         the momentum component.
        vol_window    : Realized-volatility lookback (trading days). Must be
                         paired with vol_threshold; None disables the component.
        vol_threshold : Annualized realized-vol level; component votes long
                         only when realized vol is below this.
        vix_threshold : VIX level; component votes long only when VIX close is
                         below this. Requires '^VIX' in context['universe_data'].
        combine       : "sma_only" (ignore other components for the final
                         signal), "all_agree" (all enabled components must
                         vote long), "majority" (more than half must vote long).
    """

    def __init__(
        self,
        ticker: str = "QQQ",
        sma_window: int = 200,
        mom_window: Optional[int] = None,
        vol_window: Optional[int] = None,
        vol_threshold: Optional[float] = None,
        vix_threshold: Optional[float] = None,
        combine: str = "sma_only",
    ) -> None:
        assert combine in _COMBINE_MODES, f"combine must be one of {_COMBINE_MODES}, got {combine!r}"
        self.ticker = ticker
        self.sma_window = sma_window
        self.mom_window = mom_window
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.vix_threshold = vix_threshold
        self.combine = combine

    def run(self, context: dict) -> dict:
        assert self.ticker in context["universe_data"], (
            f"TimingSignalAgent: {self.ticker!r} missing from universe_data"
        )
        price = context["universe_data"][self.ticker]["Close"]

        components = {"sma": self._sma_signal(price)}
        if self.mom_window is not None:
            components["momentum"] = self._momentum_signal(price)
        if self.vol_window is not None and self.vol_threshold is not None:
            components["vol_regime"] = self._vol_signal(price)
        if self.vix_threshold is not None:
            assert "^VIX" in context["universe_data"], (
                "TimingSignalAgent: vix_threshold is set but '^VIX' is missing from universe_data"
            )
            vix = context["universe_data"]["^VIX"]["Close"]
            components["vix_regime"] = self._vix_signal(vix, price.index)

        components_df = pd.DataFrame(components)
        signal = self._combine(components_df)

        context["timing_signal"] = signal
        context["timing_signal_components"] = components_df
        return context

    # ------------------------------------------------------------------
    # Components (each returns a 0/1 int Series, warmup defaults to 0)
    # ------------------------------------------------------------------

    def _sma_signal(self, price: pd.Series) -> pd.Series:
        sma = price.rolling(window=self.sma_window, min_periods=self.sma_window).mean()
        return (price > sma).astype(int)

    def _momentum_signal(self, price: pd.Series) -> pd.Series:
        mom = price.pct_change(periods=self.mom_window)
        return (mom > 0).astype(int)

    def _vol_signal(self, price: pd.Series) -> pd.Series:
        daily_ret = price.pct_change()
        realized_vol = (
            daily_ret.rolling(window=self.vol_window, min_periods=self.vol_window).std()
            * np.sqrt(_TRADING_DAYS_PER_YEAR)
        )
        return (realized_vol < self.vol_threshold).astype(int)

    def _vix_signal(self, vix: pd.Series, target_index: pd.Index) -> pd.Series:
        aligned = vix.reindex(target_index).ffill()
        return (aligned < self.vix_threshold).astype(int)

    # ------------------------------------------------------------------
    # Combination
    # ------------------------------------------------------------------

    def _combine(self, components_df: pd.DataFrame) -> pd.Series:
        n = components_df.shape[1]
        if self.combine == "sma_only":
            return components_df["sma"].astype(int)
        if self.combine == "all_agree":
            return (components_df.fillna(0).sum(axis=1) == n).astype(int)
        # "majority"
        return (components_df.fillna(0).sum(axis=1) > n / 2).astype(int)
