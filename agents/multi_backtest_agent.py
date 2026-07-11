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

Slippage (market impact) model
--------------------------------
The flat `transaction_cost` bps assumption doesn't capture that trading a
large fraction of an illiquid name's daily volume should cost more than
trading a small fraction of a liquid one. When `slippage_coef > 0`, each
asset's per-day cost rate gets an additional term scaled by the square root
of its participation rate (a standard, simple market-impact functional
form):

  participation[t,i] = notional_traded[t,i] / trailing_avg_dollar_volume[t,i]
  slippage_rate[t,i]  = slippage_coef * sqrt(participation[t,i])
  cost_rate[t,i]       = transaction_cost + slippage_rate[t,i]
  cost[t]              = sum_i |w[t,i] - w[t-1,i]| * cost_rate[t,i]

notional_traded uses `initial_capital` as a fixed reference (not the
compounding equity curve, which would create a circular dependency) — an
approximation acceptable for research-stage cost sensitivity testing.
With slippage_coef=0 (default) this reduces exactly to the flat-cost model.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
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
        slippage_coef: float = 0.0,
        dollar_volume_window: int = 20,
    ) -> None:
        assert initial_capital > 0, "initial_capital must be positive"
        assert 0 <= transaction_cost < 1, "transaction_cost must be in [0, 1)"
        assert slippage_coef >= 0, "slippage_coef must be >= 0"
        assert dollar_volume_window >= 1
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage_coef = slippage_coef
        self.dollar_volume_window = dollar_volume_window

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
        next_ret_aligned = next_ret.reindex(aligned_weights.index)

        # Only zero-fill returns for tickers with zero weight (unaffected).
        # Tickers with non-zero weight but missing return data are a data error.
        has_weight = aligned_weights > 0
        missing_with_weight = has_weight & next_ret_aligned.isna()
        n_missing = int(missing_with_weight.sum().sum())
        if n_missing > 0:
            logger.warning(
                "MultiBacktestAgent: %d (date, ticker) pairs have weight > 0 "
                "but missing return data — treating as 0 return", n_missing,
            )
        next_ret_aligned = next_ret_aligned.fillna(0.0)

        # Gross portfolio return = sum_i w[t,i] * next_ret[t,i]
        gross_returns = (aligned_weights * next_ret_aligned).sum(axis=1)

        # Transaction costs (flat bps + optional volatility-of-liquidity slippage)
        weight_diff = aligned_weights.diff()
        weight_diff.iloc[0] = aligned_weights.iloc[0]  # first day: enter from zero
        turnover = weight_diff.abs().sum(axis=1)

        slippage_costs = self._compute_slippage(universe_data, weight_diff)
        tc_costs = turnover * self.transaction_cost + slippage_costs

        net_returns = gross_returns - tc_costs
        equity = (1 + net_returns).cumprod() * self.initial_capital

        logger.info(
            "MultiBacktestAgent: %d dates | final_equity=%.2f | avg_daily_turnover=%.2f%% | "
            "avg_slippage_bps=%.2f",
            len(net_returns),
            float(equity.iloc[-1]) if len(equity) else 0.0,
            float(turnover.mean() * 100),
            float(slippage_costs.mean() * 10_000),
        )

        context["multi_backtest"] = {
            "equity_curve": equity,
            "returns": net_returns,
            "turnover": turnover,
            "slippage_costs": slippage_costs,
            "n_rebalances": int((turnover > 0).sum()),
            "initial_capital": self.initial_capital,
        }
        return context

    # ------------------------------------------------------------------
    # Slippage / market-impact cost
    # ------------------------------------------------------------------

    def _compute_slippage(
        self,
        universe_data: Dict[str, pd.DataFrame],
        weight_diff: pd.DataFrame,
    ) -> pd.Series:
        """Per-day sqrt-participation slippage cost, as a fraction of initial_capital.

        Returns a Series of zeros (cheaply) when slippage_coef == 0.
        """
        if self.slippage_coef == 0.0:
            return pd.Series(0.0, index=weight_diff.index)

        dollar_volume = pd.DataFrame(
            {
                ticker: (df["Close"] * df["Volume"]).rolling(
                    window=self.dollar_volume_window, min_periods=1
                ).mean()
                for ticker, df in universe_data.items()
            }
        )
        dollar_volume = dollar_volume.reindex(index=weight_diff.index, columns=weight_diff.columns)

        notional_traded = weight_diff.abs() * self.initial_capital
        participation = notional_traded / dollar_volume.replace(0.0, np.nan)
        participation = participation.clip(upper=1.0).fillna(0.0)

        slippage_rate = self.slippage_coef * np.sqrt(participation)
        per_asset_cost = weight_diff.abs() * slippage_rate
        return per_asset_cost.sum(axis=1)
