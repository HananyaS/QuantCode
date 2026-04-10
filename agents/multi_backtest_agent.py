"""MultiAssetBacktestAgent: simulates a daily-rebalanced, long-only portfolio.

Execution assumption
--------------------
Weights determined at close of day t are executed at close of day t
(market-on-close orders).  The portfolio therefore earns the close-to-close
return from t to t+1.  This is consistent with the single-asset BacktestAgent.

Transaction cost model
----------------------
tc_cost[t] = sum_i |w[t,i] - w[t-1,i]| * transaction_cost
On the first day the "previous" weight is 0 for all assets.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiAssetBacktestAgent(BaseAgent):
    """Computes daily portfolio P&L from portfolio weights and OHLCV data.

    Writes
    ------
    context['multi_backtest'] : Dict with keys:
        equity_curve   : pd.Series  — portfolio value over time
        returns        : pd.Series  — net daily portfolio returns
        turnover       : pd.Series  — daily one-way turnover
        n_rebalances   : int
        initial_capital: float
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
    ) -> None:
        assert initial_capital > 0, "initial_capital must be positive"
        assert 0 <= transaction_cost < 1, "transaction_cost must be in [0, 1)"
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        assert context.get("universe_data") is not None, (
            "MultiAssetBacktestAgent requires context['universe_data']"
        )
        assert context.get("portfolio_weights") is not None, (
            "MultiAssetBacktestAgent requires context['portfolio_weights']"
        )

        universe_data: Dict[str, pd.DataFrame] = context["universe_data"]
        weights: pd.DataFrame = context["portfolio_weights"]

        # Price matrix: date × ticker
        close_df = pd.DataFrame(
            {ticker: data["Close"] for ticker, data in universe_data.items()}
        )
        close_df = close_df[weights.columns]  # align columns to portfolio tickers

        # Next-day return for each asset: (close[t+1] - close[t]) / close[t]
        # shift(-1) aligns so that next_ret.loc[t] is the overnight return
        next_ret = close_df.pct_change().shift(-1)

        # Restrict to dates where weights are defined and next-day returns exist
        aligned_weights = weights.reindex(next_ret.dropna(how="all").index)
        aligned_weights = aligned_weights.dropna(how="all")
        next_ret_aligned = next_ret.reindex(aligned_weights.index).fillna(0.0)

        # Gross portfolio return = sum_i w[t,i] * next_ret[t,i]
        gross_returns = (aligned_weights * next_ret_aligned).sum(axis=1)

        # Transaction costs
        weight_diff = aligned_weights.diff()
        weight_diff.iloc[0] = aligned_weights.iloc[0]  # first day: enter from zero
        turnover = weight_diff.abs().sum(axis=1)
        tc_costs = turnover * self.transaction_cost

        net_returns = gross_returns - tc_costs
        equity = (1 + net_returns).cumprod() * self.initial_capital

        logger.info(
            "MultiBacktestAgent: %d dates | final_equity=%.2f | avg_daily_turnover=%.2f%%",
            len(net_returns),
            float(equity.iloc[-1]) if len(equity) else 0.0,
            float(turnover.mean() * 100),
        )

        context["multi_backtest"] = {
            "equity_curve": equity,
            "returns": net_returns,
            "turnover": turnover,
            "n_rebalances": int((turnover > 0).sum()),
            "initial_capital": self.initial_capital,
        }
        return context
