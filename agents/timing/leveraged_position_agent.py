"""LeveragedPositionAgent: continuous vol-targeted leverage sizing, mapped
onto a QQQ(1x)/QLD(2x)/TQQQ(3x) + cash blend.

Rationale (Moreira & Muir 2017): scaling exposure inversely with realized
volatility improves risk-adjusted returns. Applied here to make the naive
binary QQQ-signal-drives-TQQQ-or-cash overlay
(scripts/leveraged_overlay_check.py) less all-or-nothing: instead of always
using the full 3x instrument whenever the regime signal is "on," reduce
toward QLD/QQQ (or a partial-cash blend of the nearest tier) as realized
volatility rises — since leveraged-ETF decay scales with vol squared
(L(L-1)/2 * sigma^2), this specifically targets the mechanism that hurt
buy-and-hold TQQQ/QLD worst.

Retail-deployability constraint: a real account can't hold synthetic
fractional leverage, only the three actual ETFs. So a continuous target
leverage is achieved by picking the SMALLEST tier whose own leverage is >=
the target, then holding only a `fraction` of capital in it (rest cash) —
e.g. a 2.5x target maps to TQQQ (3x) at 83.3% invested, 16.7% cash, which
exactly reproduces 2.5x average exposure without needing margin.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent

_TRADING_DAYS_PER_YEAR = 252
_EPS = 1e-9

_DEFAULT_TIERS: Tuple[Tuple[float, str], ...] = ((1.0, "QQQ"), (2.0, "QLD"), (3.0, "TQQQ"))


class LeveragedPositionAgent(BaseAgent):
    """Maps a binary regime signal + realized volatility to a per-date
    (ticker, fraction) leveraged position.

    Writes
    ------
    context['leverage_position'] : pd.DataFrame with columns ['ticker', 'fraction'],
                                    indexed by date.

    Args:
        signal_ticker : Key into context['universe_data'] used to compute
                         realized volatility (the same instrument the
                         regime signal, context['timing_signal'], was
                         derived from).
        vol_window    : Realized-vol lookback (trading days).
        target_vol    : Annualized volatility target — target_leverage =
                         target_vol / realized_vol, clipped to max_leverage.
        max_leverage  : Upper bound on target leverage; should not exceed
                         the largest tier's own leverage.
        tiers         : Ascending (leverage, ticker) pairs of investable
                         instruments. Default: QQQ(1x)/QLD(2x)/TQQQ(3x).
    """

    def __init__(
        self,
        signal_ticker: str = "QQQ",
        vol_window: int = 20,
        target_vol: float = 0.18,
        max_leverage: float = 3.0,
        tiers: Tuple[Tuple[float, str], ...] = _DEFAULT_TIERS,
    ) -> None:
        assert max_leverage <= max(lev for lev, _ in tiers) + _EPS, (
            "max_leverage must not exceed the largest tier's own leverage"
        )
        self.signal_ticker = signal_ticker
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.tiers = tuple(sorted(tiers, key=lambda pair: pair[0]))

    def run(self, context: dict) -> dict:
        assert self.signal_ticker in context["universe_data"], (
            f"LeveragedPositionAgent: {self.signal_ticker!r} missing from universe_data"
        )
        assert "timing_signal" in context, "LeveragedPositionAgent: context missing 'timing_signal'"

        price = context["universe_data"][self.signal_ticker]["Close"]
        signal = context["timing_signal"].reindex(price.index).fillna(0).astype(int)

        realized_vol = self._realized_vol(price)
        raw_target = self._target_leverage(realized_vol)
        effective_leverage = raw_target * signal

        tickers, fractions = self._map_to_tiers(effective_leverage)
        position = pd.DataFrame({"ticker": tickers, "fraction": fractions}, index=price.index)

        context["leverage_position"] = position
        return context

    def _realized_vol(self, price: pd.Series) -> pd.Series:
        daily_ret = price.pct_change()
        return (
            daily_ret.rolling(window=self.vol_window, min_periods=self.vol_window).std()
            * np.sqrt(_TRADING_DAYS_PER_YEAR)
        )

    def _target_leverage(self, realized_vol: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = self.target_vol / realized_vol
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return raw.clip(lower=0.0, upper=self.max_leverage)

    def _map_to_tiers(self, effective_leverage: pd.Series) -> Tuple[List[str], List[float]]:
        tickers: List[str] = []
        fractions: List[float] = []
        for lev in effective_leverage:
            ticker, fraction = self._select_tier(lev)
            tickers.append(ticker)
            fractions.append(fraction)
        return tickers, fractions

    def _select_tier(self, effective_leverage: float) -> Tuple[str, float]:
        for tier_leverage, ticker in self.tiers:
            if effective_leverage <= tier_leverage + _EPS:
                fraction = 0.0 if tier_leverage == 0 else effective_leverage / tier_leverage
                return ticker, min(max(fraction, 0.0), 1.0)
        # effective_leverage exceeds every tier (shouldn't happen given the
        # max_leverage <= top-tier-leverage invariant) — fall back to the
        # top tier fully invested.
        top_leverage, top_ticker = self.tiers[-1]
        return top_ticker, 1.0
