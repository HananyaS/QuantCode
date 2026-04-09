"""EvaluationAgent: computes risk-adjusted performance metrics and SPY comparison."""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationAgent(BaseAgent):
    """Computes Sharpe ratio, max drawdown, annualised return, and SPY benchmark.

    Args:
        risk_free_rate: Annualised risk-free rate used in Sharpe computation.
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self.risk_free_rate = risk_free_rate

    def run(self, context: dict) -> dict:
        """Compute all metrics and store in context['metrics']."""
        assert context.get("backtest") is not None, (
            "EvaluationAgent requires context['backtest']"
        )

        backtest = context["backtest"]
        returns: pd.Series = backtest["returns"]
        equity: pd.Series = backtest["equity_curve"]

        metrics: Dict = {
            "sharpe": self._sharpe(returns),
            "max_drawdown": self._max_drawdown(equity),
            "annualized_return": self._annualized_return(equity),
            "win_rate": backtest["win_rate"],
            "n_trades": backtest["n_trades"],
            "final_equity": float(equity.iloc[-1]) if len(equity) else 0.0,
        }

        if context.get("predictions"):
            metrics["train_accuracy"] = context["predictions"]["train_accuracy"]
            metrics["test_accuracy"] = context["predictions"]["test_accuracy"]

        if context.get("benchmark_data") is not None:
            metrics["benchmark"] = self._benchmark_metrics(
                context["benchmark_data"], returns.index
            )

        context["metrics"] = metrics
        logger.info(
            "EvaluationAgent: sharpe=%.3f  max_dd=%.3f  ann_ret=%.3f",
            metrics["sharpe"],
            metrics["max_drawdown"],
            metrics["annualized_return"],
        )
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sharpe(self, returns: pd.Series) -> float:
        """Annualised Sharpe ratio (assumes daily returns, 252 trading days/year)."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        daily_rf = self.risk_free_rate / 252
        excess = returns - daily_rf
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def _max_drawdown(self, equity: pd.Series) -> float:
        """Maximum peak-to-trough drawdown (negative value)."""
        if len(equity) < 2:
            return 0.0
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return float(drawdown.min())

    def _annualized_return(self, equity: pd.Series) -> float:
        """Compound annualised growth rate."""
        if len(equity) < 2 or equity.iloc[0] <= 0:
            return 0.0
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        n_years = len(equity) / 252
        return float((1 + total_return) ** (1 / n_years) - 1)

    def _benchmark_metrics(
        self, benchmark_data: pd.DataFrame, test_index: pd.Index
    ) -> Dict:
        """Compute the same metrics for SPY over the same test window."""
        bm_close = benchmark_data["Close"].reindex(test_index).dropna()
        bm_returns = bm_close.pct_change().dropna()
        bm_equity = (1 + bm_returns).cumprod() * 100_000.0
        return {
            "sharpe": self._sharpe(bm_returns),
            "max_drawdown": self._max_drawdown(bm_equity),
            "annualized_return": self._annualized_return(bm_equity),
        }
