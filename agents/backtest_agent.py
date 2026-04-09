"""BacktestAgent: runs a long/flat strategy on the test window predictions."""
import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class BacktestAgent(BaseAgent):
    """Translates predictions into a daily P&L stream and equity curve.

    Strategy:
        - prediction[t] == 1 → long from close[t] to close[t+1]
        - prediction[t] == 0 → flat (no position)
    Transaction costs are applied proportionally to the absolute position change.

    Args:
        initial_capital: Starting portfolio value in dollars.
        transaction_cost: One-way cost as a fraction of trade value.
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

    def run(self, context: dict) -> dict:
        """Compute strategy returns and equity curve; store in context['backtest']."""
        assert context.get("data") is not None, (
            "BacktestAgent requires context['data']"
        )
        assert context.get("predictions") is not None, (
            "BacktestAgent requires context['predictions']"
        )

        close = context["data"]["Close"]
        predictions = context["predictions"]["values"]

        net_returns, position = self._compute_returns(close, predictions)
        equity = self._equity_curve(net_returns)
        n_trades, win_rate = self._trade_stats(position, net_returns)

        context["backtest"] = {
            "equity_curve": equity,
            "returns": net_returns,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "initial_capital": self.initial_capital,
        }
        logger.info(
            "BacktestAgent: %d trades | win_rate=%.1f%% | final_equity=%.2f",
            n_trades,
            100 * win_rate,
            float(equity.iloc[-1]) if len(equity) else 0.0,
        )
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_returns(
        self, close: pd.Series, predictions: pd.Series
    ) -> tuple:
        """Align predictions with overnight returns; apply transaction costs."""
        # next_return[t] = (close[t+1] - close[t]) / close[t]
        # This is the return earned by holding from t's close to (t+1)'s close.
        next_return = close.pct_change().shift(-1)

        # Restrict to prediction dates and drop the last date (no next-day return)
        aligned_return = next_return.reindex(predictions.index).dropna()
        position = predictions.reindex(aligned_return.index).fillna(0)

        raw_returns = position * aligned_return

        # Transaction cost applied on each position change
        pos_change = position.diff().abs().fillna(position.abs())
        tc = pos_change * self.transaction_cost
        net_returns = raw_returns - tc

        return net_returns, position

    def _equity_curve(self, net_returns: pd.Series) -> pd.Series:
        """Compound daily returns starting from initial_capital."""
        if net_returns.empty:
            return pd.Series(dtype=float)
        return (1 + net_returns).cumprod() * self.initial_capital

    def _trade_stats(
        self, position: pd.Series, net_returns: pd.Series
    ) -> tuple:
        """Compute number of trades and win rate."""
        n_trades = int(position.diff().abs().fillna(position.abs()).gt(0).sum())
        active = net_returns[position != 0]
        win_rate = float((active > 0).sum() / len(active)) if len(active) > 0 else 0.0
        return n_trades, win_rate
