"""KellyPositionAgent: theoretically-grounded leverage sizing via the Kelly
criterion, with entry/exit hysteresis and per-decision logging.

Not a naive "pick TQQQ when bullish" signal — leverage is driven by the
ratio of conditional drift to conditional variance (L* = mu_hat/sigma_sq_hat,
utils/kelly_sizing.py), sized down by a fractional-Kelly safety multiplier,
capped against ruin proximity, and only rotated on a meaningful trigger
(never churned by noise).

Causality convention (IMPORTANT — see docs/research-log/kelly-leverage-sizing.md)
-------------------------------------------------------------------------------
mu_hat_t and sigma_sq_hat_t (from utils/regime_classifier.py and
utils/conditional_vol.py) are already forecasts FOR date t using only data
through t-1 (pre-shifted at the source). The position this agent decides
for date t is therefore ALREADY the correct position to hold DURING t — the
downstream backtest agent (agents/timing/kelly_backtest_agent.py) must NOT
apply an additional shift(1), unlike LeveragedBacktestAgent (which expects
a non-shifted, same-day-decided position and applies the lag itself).
Mixing the two conventions would double-lag and desynchronize this agent's
output from LeveragedBacktestAgent's contract — use KellyBacktestAgent, not
LeveragedBacktestAgent, downstream of this agent.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.conditional_vol import ewma_variance, gjr_garch_variance
from utils.kelly_sizing import fractional_kelly, kelly_optimal_leverage, map_to_instrument_blend, ruin_floor_cap
from utils.regime_classifier import classify_regime, conditional_mu

_TRADING_DAYS_PER_YEAR = 252
_DEFAULT_TIERS: Tuple[Tuple[float, str], ...] = ((1.0, "QQQ"), (2.0, "QLD"), (3.0, "TQQQ"))


class KellyPositionAgent(BaseAgent):
    """Sizes a per-date (ticker, fraction) leveraged position via the Kelly
    criterion applied to conditional (time-varying) drift/variance
    estimates, with entry/exit hysteresis.

    Writes
    ------
    context['leverage_position']  : pd.DataFrame(['ticker', 'fraction']),
                                     schema-compatible with LeveragedPositionAgent.
    context['kelly_decision_log'] : pd.DataFrame — mu_hat, sigma_sq_hat,
                                     l_star, target_leverage, regime, ticker,
                                     fraction per date, for walk-forward review.

    Entry/exit rule (all thresholds are constructor args — no magic numbers)
    --------------------------------------------------------------------------
    - Drawdown breach on the held instrument (> drawdown_limit) OR a
      conditional-vol spike (annualized sigma_hat > vol_spike_threshold)
      forces the position fully flat (cash) — the two most acute risks.
    - Regime flips to "choppy" (and neither of the above fired) caps
      exposure at <= 1.0x (QQQ-level) — chop alone means no leverage edge,
      not necessarily "flee the underlying too."
    - Otherwise (trending, vol not spiking, no drawdown breach): the
      position only INCREASES when the fractional-Kelly target exceeds the
      currently-held exposure by more than `entry_margin` — avoids
      noise-driven churn. If it doesn't clear that margin, HOLD.
    """

    def __init__(
        self,
        signal_ticker: str = "QQQ",
        vol_method: str = "ewma",
        vol_decay: float = 0.94,
        garch_refit_every: int = 20,
        garch_min_train_obs: int = 250,
        mu_decay: float = 0.94,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        fractional_kelly: float = 0.5,
        max_leverage: float = 3.0,
        worst_case_daily_move: float = 0.20,
        ruin_buffer: float = 0.20,
        vol_spike_threshold: float = 0.40,
        drawdown_limit: float = 0.15,
        entry_margin: float = 0.3,
        min_observations: int = 60,
        tiers: Tuple[Tuple[float, str], ...] = _DEFAULT_TIERS,
    ) -> None:
        assert vol_method in ("ewma", "gjr_garch"), f"vol_method must be 'ewma' or 'gjr_garch', got {vol_method!r}"
        self.signal_ticker = signal_ticker
        self.vol_method = vol_method
        self.vol_decay = vol_decay
        self.garch_refit_every = garch_refit_every
        self.garch_min_train_obs = garch_min_train_obs
        self.mu_decay = mu_decay
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.fractional_kelly_mult = fractional_kelly
        self.max_leverage = max_leverage
        self.worst_case_daily_move = worst_case_daily_move
        self.ruin_buffer = ruin_buffer
        self.vol_spike_threshold = vol_spike_threshold
        self.drawdown_limit = drawdown_limit
        self.entry_margin = entry_margin
        self.min_observations = min_observations
        self.tiers = tuple(sorted(tiers, key=lambda pair: pair[0]))
        self._tier_leverage = {ticker: lev for lev, ticker in self.tiers}

    def run(self, context: dict) -> dict:
        assert self.signal_ticker in context["universe_data"], (
            f"KellyPositionAgent: {self.signal_ticker!r} missing from universe_data"
        )
        df = context["universe_data"][self.signal_ticker]
        high, low, close = df["High"], df["Low"], df["Close"]
        returns = close.pct_change()

        mu_hat = conditional_mu(returns, decay=self.mu_decay)
        sigma_sq_hat = self._conditional_vol(returns)
        regime = classify_regime(high, low, close, adx_period=self.adx_period, adx_threshold=self.adx_threshold)

        position, log_rows = self._simulate(close, mu_hat, sigma_sq_hat, regime, context["universe_data"])

        context["leverage_position"] = position
        context["kelly_decision_log"] = pd.DataFrame(log_rows, index=close.index)
        return context

    def _conditional_vol(self, returns: pd.Series) -> pd.Series:
        if self.vol_method == "ewma":
            return ewma_variance(returns, decay=self.vol_decay)
        return gjr_garch_variance(
            returns, refit_every=self.garch_refit_every, min_train_obs=self.garch_min_train_obs,
        )

    def _simulate(
        self,
        close: pd.Series,
        mu_hat: pd.Series,
        sigma_sq_hat: pd.Series,
        regime: pd.Series,
        universe_data: dict,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        dates = close.index
        tickers: List[str] = []
        fractions: List[float] = []
        log_rows: List[dict] = []

        held_ticker = self.tiers[0][1]  # "QQQ" tier as the flat/zero-leverage placeholder ticker
        held_fraction = 0.0
        watermark: Optional[float] = None

        held_price_cache = {t: context_df["Close"] for t, context_df in universe_data.items()}

        for i, date in enumerate(dates):
            mu_t, sigma_sq_t, regime_t = mu_hat.get(date), sigma_sq_hat.get(date), regime.get(date)
            valid = (
                i >= self.min_observations
                and mu_t is not None and sigma_sq_t is not None and regime_t is not None
                and not pd.isna(mu_t) and not pd.isna(sigma_sq_t)
            )

            current_exposure = self._tier_leverage.get(held_ticker, 1.0) * held_fraction

            if not valid:
                target_leverage = 0.0
                l_star = float("nan")
                de_risk_reason = "warmup"
            else:
                l_star = kelly_optimal_leverage(mu_t, sigma_sq_t)
                l_frac = fractional_kelly(l_star, fraction=self.fractional_kelly_mult, max_leverage=self.max_leverage)
                l_capped = ruin_floor_cap(l_frac, self.worst_case_daily_move, self.ruin_buffer)

                annualized_vol = float(np.sqrt(max(sigma_sq_t, 0.0)) * np.sqrt(_TRADING_DAYS_PER_YEAR))
                vol_spiking = annualized_vol > self.vol_spike_threshold

                dd = 0.0
                if held_fraction > 0 and watermark:
                    held_price_now = held_price_cache[held_ticker].get(date)
                    if held_price_now is not None and not pd.isna(held_price_now):
                        dd = held_price_now / watermark - 1.0
                drawdown_breach = dd < -self.drawdown_limit

                if vol_spiking or drawdown_breach:
                    target_leverage = 0.0
                    de_risk_reason = "vol_spike" if vol_spiking else "drawdown_breach"
                elif regime_t == "choppy":
                    target_leverage = min(current_exposure, 1.0)
                    de_risk_reason = "choppy"
                else:
                    if l_capped > current_exposure + self.entry_margin:
                        target_leverage = l_capped
                    else:
                        target_leverage = current_exposure
                    de_risk_reason = None

            new_ticker, new_fraction = map_to_instrument_blend(target_leverage, tiers=self.tiers)

            entered_new_position = (new_ticker != held_ticker) or (
                held_fraction <= 0.0 < new_fraction
            )
            if new_fraction <= 0.0:
                watermark = None
            elif entered_new_position:
                watermark = held_price_cache[new_ticker].get(date)
            else:
                price_now = held_price_cache[new_ticker].get(date)
                if price_now is not None and not pd.isna(price_now) and watermark is not None:
                    watermark = max(watermark, price_now)

            held_ticker, held_fraction = new_ticker, new_fraction
            tickers.append(new_ticker)
            fractions.append(new_fraction)
            log_rows.append({
                "mu_hat": mu_t, "sigma_sq_hat": sigma_sq_t, "l_star": l_star,
                "target_leverage": target_leverage, "regime": regime_t,
                "ticker": new_ticker, "fraction": new_fraction,
            })

        position = pd.DataFrame({"ticker": tickers, "fraction": fractions}, index=dates)
        return position, log_rows
