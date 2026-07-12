"""KellyBacktestAgent: backtests KellyPositionAgent's per-date (ticker,
fraction) position.

Deliberately does NOT apply an execution-lag shift, unlike
LeveragedBacktestAgent. KellyPositionAgent's position for date t is decided
from mu_hat_t/sigma_sq_hat_t/regime_t, which are ALREADY forecasts using
only data through t-1 (pre-shifted at the source in utils/conditional_vol.py
and utils/regime_classifier.py) — so the position for t is already the
correct position to hold DURING t. Applying an additional shift here would
double-lag and desynchronize this backtest from what KellyPositionAgent
actually decided. See kelly_position_agent.py's module docstring.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from agents.base_agent import BaseAgent

_DEFAULT_TICKERS: Tuple[str, ...] = ("QQQ", "QLD", "TQQQ")


class KellyBacktestAgent(BaseAgent):
    """Backtests context['leverage_position'] (from KellyPositionAgent)
    against real or synthetic price data for each candidate instrument.

    Writes
    ------
    context['timing_returns']     : pd.Series — realized blended strategy returns
    context['timing_equity']      : pd.Series — strategy equity curve (starts at 1.0)
    context['benchmark_returns']  : pd.Series — buy-and-hold returns for benchmark_ticker
    context['benchmark_equity']   : pd.Series — buy-and-hold equity curve (starts at 1.0)

    Args:
        tickers               : Instruments the position may reference —
                                 each must be a key in context['universe_data'].
        transaction_cost_bps  : Cost (bps of notional) charged on the L1
                                 turnover of the (non-lagged) exposure
                                 vector each bar.
        benchmark_ticker      : Instrument to buy-and-hold as comparison.
    """

    def __init__(
        self,
        tickers: Tuple[str, ...] = _DEFAULT_TICKERS,
        transaction_cost_bps: float = 0.0,
        benchmark_ticker: str = "TQQQ",
    ) -> None:
        self.tickers = tuple(tickers)
        self.transaction_cost_bps = transaction_cost_bps
        self.benchmark_ticker = benchmark_ticker

    def run(self, context: dict) -> dict:
        assert "leverage_position" in context, "KellyBacktestAgent: context missing 'leverage_position'"
        position = context["leverage_position"]
        for t in self.tickers:
            assert t in context["universe_data"], f"KellyBacktestAgent: {t!r} missing from universe_data"
        assert self.benchmark_ticker in context["universe_data"], (
            f"KellyBacktestAgent: benchmark {self.benchmark_ticker!r} missing from universe_data"
        )

        dates = position.index
        exposure = pd.DataFrame(0.0, index=dates, columns=self.tickers)
        for t in self.tickers:
            mask = position["ticker"] == t
            exposure.loc[mask, t] = position.loc[mask, "fraction"].values

        price_returns = pd.DataFrame(index=dates)
        for t in self.tickers:
            close = context["universe_data"][t]["Close"].reindex(dates)
            price_returns[t] = close.pct_change().fillna(0.0)

        # NO shift here — exposure for date t already represents a decision
        # made using data through t-1 (baked into KellyPositionAgent's
        # forecast-for-t inputs). Applying today's exposure to today's
        # return is the correct, non-lookahead computation in this
        # convention (see module docstring).
        gross_ret = (exposure * price_returns).sum(axis=1)
        turnover = exposure.diff().abs().sum(axis=1)
        turnover.iloc[0] = exposure.iloc[0].abs().sum()
        cost = turnover * (self.transaction_cost_bps / 10_000.0)
        strat_ret = gross_ret - cost

        benchmark_close = context["universe_data"][self.benchmark_ticker]["Close"].reindex(dates)
        benchmark_ret = benchmark_close.pct_change().fillna(0.0)

        context["timing_returns"] = strat_ret
        context["timing_equity"] = (1 + strat_ret).cumprod()
        context["benchmark_returns"] = benchmark_ret
        context["benchmark_equity"] = (1 + benchmark_ret).cumprod()
        return context
