"""MultiAssetEvaluationAgent: risk-adjusted metrics for the multi-asset portfolio.

Sharpe sanity check
-------------------
A Sharpe > 2.0 on daily returns is almost always a sign of look-ahead bias
or a bug.  A WARNING is logged so the developer investigates before acting
on the results.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)

_SHARPE_WARNING_THRESHOLD = 2.0


class MultiAssetEvaluationAgent(BaseAgent):
    """Computes Sharpe, max drawdown, annualised return and SPY/QQQ comparison.

    Writes
    ------
    context['multi_metrics'] : Dict with keys:
        sharpe, max_drawdown, annualized_return,
        n_rebalances, avg_daily_turnover, final_equity,
        ic (from cs_model if available),
        benchmark (sub-dict: sharpe, max_drawdown, annualized_return)
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        assert context.get("multi_backtest") is not None, (
            "MultiAssetEvaluationAgent requires context['multi_backtest']"
        )

        bt = context["multi_backtest"]
        returns: pd.Series = bt["returns"]
        equity: pd.Series = bt["equity_curve"]

        metrics: Dict = {
            "sharpe": self._sharpe(returns),
            "max_drawdown": self._max_drawdown(equity),
            "annualized_return": self._annualized_return(equity),
            "n_rebalances": bt["n_rebalances"],
            "avg_daily_turnover": float(bt["turnover"].mean()),
            "final_equity": float(equity.iloc[-1]) if len(equity) else 0.0,
        }

        if context.get("cs_model"):
            metrics["ic"] = context["cs_model"].get("ic", float("nan"))

        if context.get("benchmark_data") is not None:
            metrics["benchmark"] = self._benchmark_metrics(
                context["benchmark_data"], returns.index
            )

        self._sanity_check(metrics)

        context["multi_metrics"] = metrics
        logger.info(
            "MultiEvaluationAgent: sharpe=%.3f  max_dd=%.3f  ann_ret=%.3f  IC=%.4f",
            metrics["sharpe"],
            metrics["max_drawdown"],
            metrics["annualized_return"],
            metrics.get("ic", float("nan")),
        )
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        daily_rf = self.risk_free_rate / 252
        excess = returns - daily_rf
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def _max_drawdown(self, equity: pd.Series) -> float:
        if len(equity) < 2:
            return 0.0
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return float(drawdown.min())

    def _annualized_return(self, equity: pd.Series) -> float:
        if len(equity) < 2 or equity.iloc[0] <= 0:
            return 0.0
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        n_years = len(equity) / 252
        return float((1 + total_return) ** (1 / n_years) - 1)

    def _benchmark_metrics(
        self, benchmark_data: pd.DataFrame, test_index: pd.Index
    ) -> Dict:
        bm_close = benchmark_data["Close"].reindex(test_index).dropna()
        bm_returns = bm_close.pct_change().dropna()
        bm_equity = (1 + bm_returns).cumprod() * 100_000.0
        return {
            "sharpe": self._sharpe(bm_returns),
            "max_drawdown": self._max_drawdown(bm_equity),
            "annualized_return": self._annualized_return(bm_equity),
        }

    @staticmethod
    def _sanity_check(metrics: Dict) -> None:
        sharpe = metrics.get("sharpe", 0.0)
        if sharpe > _SHARPE_WARNING_THRESHOLD:
            logger.warning(
                "MultiEvaluationAgent: Sharpe=%.2f exceeds %.1f — "
                "likely leakage or data error; investigate before trusting results",
                sharpe,
                _SHARPE_WARNING_THRESHOLD,
            )
