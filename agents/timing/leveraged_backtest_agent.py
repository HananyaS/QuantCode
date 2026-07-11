"""LeveragedBacktestAgent: backtests a per-date (ticker, fraction) position
blend across multiple instruments (QQQ/QLD/TQQQ) + cash.

Generalizes TimingBacktestAgent (which only ever holds one fixed
instrument) to LeveragedPositionAgent's continuous vol-sized output, where
the traded instrument itself can change over time. Turnover — and
therefore transaction cost — is measured as the L1 distance between
consecutive held (lagged) exposure vectors, which correctly captures both
a full ticker switch (e.g. QQQ,1.0 -> TQQQ,1.0 costs 2.0 units: sell all
QQQ, buy all TQQQ) and a same-ticker fraction change (e.g. TQQQ 0.5 -> 0.8
costs 0.3 units).
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from agents.base_agent import BaseAgent

_DEFAULT_TICKERS: Tuple[str, ...] = ("QQQ", "QLD", "TQQQ")


class LeveragedBacktestAgent(BaseAgent):
    """Backtests context['leverage_position'] (from LeveragedPositionAgent)
    against real price data for each candidate instrument.

    Writes
    ------
    context['timing_returns']     : pd.Series — realized blended strategy returns
    context['timing_equity']      : pd.Series — strategy equity curve (starts at 1.0)
    context['benchmark_returns']  : pd.Series — buy-and-hold returns for benchmark_ticker
    context['benchmark_equity']   : pd.Series — buy-and-hold equity curve (starts at 1.0)

    Args:
        tickers              : The full set of instruments the position may
                                reference — each must be a key in
                                context['universe_data'].
        transaction_cost_bps : Cost (bps of notional) charged on the L1
                                turnover of the held (lagged) exposure
                                vector each bar.
        benchmark_ticker      : Instrument to buy-and-hold as the comparison
                                benchmark.
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
        assert "leverage_position" in context, "LeveragedBacktestAgent: context missing 'leverage_position'"
        position = context["leverage_position"]
        for t in self.tickers:
            assert t in context["universe_data"], f"LeveragedBacktestAgent: {t!r} missing from universe_data"
        assert self.benchmark_ticker in context["universe_data"], (
            f"LeveragedBacktestAgent: benchmark {self.benchmark_ticker!r} missing from universe_data"
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

        # Execution lag: the exposure HELD during bar t is what was decided
        # at the end of bar t-1 — never the same bar's own decision.
        lagged_exposure = exposure.shift(1).fillna(0.0)

        gross_ret = (lagged_exposure * price_returns).sum(axis=1)
        turnover = lagged_exposure.diff().abs().sum(axis=1)
        turnover.iloc[0] = lagged_exposure.iloc[0].abs().sum()
        cost = turnover * (self.transaction_cost_bps / 10_000.0)
        strat_ret = gross_ret - cost

        benchmark_close = context["universe_data"][self.benchmark_ticker]["Close"].reindex(dates)
        benchmark_ret = benchmark_close.pct_change().fillna(0.0)

        context["timing_returns"] = strat_ret
        context["timing_equity"] = (1 + strat_ret).cumprod()
        context["benchmark_returns"] = benchmark_ret
        context["benchmark_equity"] = (1 + benchmark_ret).cumprod()
        return context
