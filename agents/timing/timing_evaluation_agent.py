"""TimingEvaluationAgent: Sharpe/CAGR/MaxDD/time-in-market for a
single-instrument timing strategy vs its buy-and-hold benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent


class TimingEvaluationAgent(BaseAgent):
    """Computes summary metrics from TimingBacktestAgent's outputs.

    Writes
    ------
    context['timing_metrics'] : dict with strategy_* and benchmark_* keys
        plus time_in_market.

    Args:
        periods_per_year: Annualization factor (252 for daily bars).
    """

    def __init__(self, periods_per_year: int = 252) -> None:
        self.periods_per_year = periods_per_year

    def run(self, context: dict) -> dict:
        strat_ret = context["timing_returns"]
        strat_eq = context["timing_equity"]
        bh_ret = context["benchmark_returns"]
        bh_eq = context["benchmark_equity"]
        signal = context["timing_signal"]

        metrics = {
            "strategy_sharpe": self._sharpe(strat_ret),
            "strategy_cagr": self._cagr(strat_eq),
            "strategy_max_drawdown": self._max_drawdown(strat_eq),
            "benchmark_sharpe": self._sharpe(bh_ret),
            "benchmark_cagr": self._cagr(bh_eq),
            "benchmark_max_drawdown": self._max_drawdown(bh_eq),
            "time_in_market": float(signal.mean()),
        }
        context["timing_metrics"] = metrics
        return context

    def _sharpe(self, returns: pd.Series) -> float:
        sigma = returns.std()
        if sigma == 0 or np.isnan(sigma):
            return 0.0
        return float(returns.mean() / sigma * np.sqrt(self.periods_per_year))

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
