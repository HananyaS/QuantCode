"""TimingBacktestAgent: single-instrument time-series backtest.

Deliberately NOT the cross-sectional PortfolioAgent/MultiAssetBacktestAgent
path — there's exactly one position (long the underlying or flat), so a full
portfolio simulation is unnecessary machinery. The one thing that matters is
getting execution timing right: today's *position* is yesterday's *signal*
(the agent that generated the signal at date t cannot also trade at date t's
open using date t's own close).
"""
from __future__ import annotations

import pandas as pd

from agents.base_agent import BaseAgent


class TimingBacktestAgent(BaseAgent):
    """Backtests a binary timing_signal against one instrument's price series.

    Writes
    ------
    context['timing_returns']     : pd.Series — realized strategy returns
    context['timing_equity']      : pd.Series — strategy equity curve (starts at 1.0)
    context['benchmark_returns']  : pd.Series — buy-and-hold returns
    context['benchmark_equity']   : pd.Series — buy-and-hold equity curve (starts at 1.0)

    Args:
        ticker              : Key into context['universe_data'] for the price series.
        transaction_cost_bps: Round-trip-agnostic cost charged (in bps of
                               notional) on every bar where the applied
                               position changes.
    """

    def __init__(self, ticker: str = "QQQ", transaction_cost_bps: float = 0.0) -> None:
        self.ticker = ticker
        self.transaction_cost_bps = transaction_cost_bps

    def run(self, context: dict) -> dict:
        assert self.ticker in context["universe_data"], (
            f"TimingBacktestAgent: {self.ticker!r} missing from universe_data"
        )
        assert "timing_signal" in context, "TimingBacktestAgent: context missing 'timing_signal'"

        close = context["universe_data"][self.ticker]["Close"]
        signal = context["timing_signal"].reindex(close.index)

        price_ret = close.pct_change().fillna(0.0)
        # Execution lag: the position held DURING bar t is the signal decided
        # at the END of bar t-1 — never the same bar's own signal.
        position = signal.shift(1).fillna(0.0)

        gross_ret = position * price_ret
        turnover = position.diff().abs().fillna(position.abs())
        cost = turnover * (self.transaction_cost_bps / 10_000.0)
        strat_ret = gross_ret - cost

        context["timing_returns"] = strat_ret
        context["timing_equity"] = (1 + strat_ret).cumprod()
        context["benchmark_returns"] = price_ret
        context["benchmark_equity"] = (1 + price_ret).cumprod()
        return context
