"""KellyEvaluationAgent: standard metrics plus the Kelly-specific diagnostics
the module needs to be explainable rather than a black box:

- Sortino ratio (downside-only risk-adjusted return, complementing Sharpe).
- theoretical_g vs realized_g: the growth rate expected_log_growth() (see
  utils/kelly_sizing.py) predicts for the strategy's realized average
  leverage and the underlying's realized mu/sigma^2, compared against the
  strategy's ACTUAL realized log-growth rate.
- naive_cum_return vs actual_cum_return: what a naive `avg_leverage x
  underlying_cumulative_return` calculation would imply, vs. what the
  strategy's real (exact-recursion, compounded) cumulative return actually
  was — the concrete numeric proof that leveraged daily-reset products are
  NOT simply L times the underlying's cumulative return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.kelly_sizing import expected_log_growth

_DEFAULT_TIER_LEVERAGE = {"QQQ": 1.0, "QLD": 2.0, "TQQQ": 3.0}


class KellyEvaluationAgent(BaseAgent):
    """Computes summary + Kelly-diagnostic metrics from KellyBacktestAgent's
    outputs.

    Writes
    ------
    context['kelly_metrics'] : dict — strategy_*/benchmark_* Sharpe/Sortino/
        CAGR/MaxDD, time_in_market, avg_leverage, realized_underlying_mu,
        realized_underlying_sigma_sq, theoretical_g, realized_g, g_gap,
        naive_cum_return, actual_cum_return, naive_vs_actual_divergence.

    Args:
        periods_per_year : Annualization factor (252 for daily bars).
        underlying_ticker: Key into context['universe_data'] for the
            reference underlying used to compute realized mu/sigma^2 for
            the theoretical g(L) prediction.
        tier_leverage     : Mapping of ticker -> its own daily leverage
            factor, for computing the strategy's realized average leverage
            from context['leverage_position'].
    """

    def __init__(
        self,
        periods_per_year: int = 252,
        underlying_ticker: str = "QQQ",
        tier_leverage: dict = None,
    ) -> None:
        self.periods_per_year = periods_per_year
        self.underlying_ticker = underlying_ticker
        self.tier_leverage = tier_leverage or _DEFAULT_TIER_LEVERAGE

    def run(self, context: dict) -> dict:
        strat_ret = context["timing_returns"]
        strat_eq = context["timing_equity"]
        bh_ret = context["benchmark_returns"]
        bh_eq = context["benchmark_equity"]
        position = context["leverage_position"]

        effective_leverage = position.apply(
            lambda r: self.tier_leverage.get(r["ticker"], 1.0) * r["fraction"], axis=1,
        )
        avg_leverage = float(effective_leverage.mean())

        underlying_close = context["universe_data"][self.underlying_ticker]["Close"].reindex(strat_ret.index)
        underlying_ret = underlying_close.pct_change().fillna(0.0)
        realized_mu = float(underlying_ret.mean())
        realized_sigma_sq = float(underlying_ret.var())

        theoretical_g = expected_log_growth(avg_leverage, realized_mu, realized_sigma_sq)
        log_ret = np.log1p(strat_ret.clip(lower=-0.999999))
        realized_g = float(log_ret.mean())

        underlying_cum_return = float(underlying_close.iloc[-1] / underlying_close.iloc[0] - 1)
        naive_cum_return = avg_leverage * underlying_cum_return
        actual_cum_return = float(strat_eq.iloc[-1] / strat_eq.iloc[0] - 1)

        metrics = {
            "strategy_sharpe": self._sharpe(strat_ret),
            "strategy_sortino": self._sortino(strat_ret),
            "strategy_cagr": self._cagr(strat_eq),
            "strategy_max_drawdown": self._max_drawdown(strat_eq),
            "benchmark_sharpe": self._sharpe(bh_ret),
            "benchmark_sortino": self._sortino(bh_ret),
            "benchmark_cagr": self._cagr(bh_eq),
            "benchmark_max_drawdown": self._max_drawdown(bh_eq),
            "time_in_market": float((effective_leverage > 0).mean()),
            "avg_leverage": avg_leverage,
            "realized_underlying_mu": realized_mu,
            "realized_underlying_sigma_sq": realized_sigma_sq,
            "theoretical_g": theoretical_g,
            "realized_g": realized_g,
            "g_gap": realized_g - theoretical_g,
            "naive_cum_return": naive_cum_return,
            "actual_cum_return": actual_cum_return,
            "naive_vs_actual_divergence": actual_cum_return - naive_cum_return,
        }
        context["kelly_metrics"] = metrics
        return context

    def _sharpe(self, returns: pd.Series) -> float:
        sigma = returns.std()
        if sigma == 0 or np.isnan(sigma):
            return 0.0
        return float(returns.mean() / sigma * np.sqrt(self.periods_per_year))

    def _sortino(self, returns: pd.Series) -> float:
        downside = returns.clip(upper=0.0)
        downside_std = np.sqrt((downside**2).mean())
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return float(returns.mean() / downside_std * np.sqrt(self.periods_per_year))

    def _cagr(self, equity: pd.Series) -> float:
        n_periods = len(equity) - 1
        if n_periods <= 0:
            return 0.0
        years = n_periods / self.periods_per_year
        if years <= 0 or equity.iloc[0] <= 0:
            return 0.0
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        running_max = equity.cummax()
        dd = equity / running_max - 1.0
        return float(dd.min())
