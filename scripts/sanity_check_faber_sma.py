"""One-off sanity check (not part of the agent pipeline): replicate Faber's
10-month SMA timing rule on QQQ and compare vs buy-and-hold.

Methodology (Faber, "A Quantitative Approach to Tactical Asset Allocation"):
  - Use month-end closing prices.
  - Signal = long if price > trailing 10-month SMA (computed on month-end
    closes, i.e. ~200 trading days), else flat (cash, 0% return).
  - Rebalance monthly, decide using only data available at that month-end
    (no lookahead — the SMA at month t uses closes through month t).

This is a published, well-known result: the rule should reduce max drawdown
substantially (it dodges most of the 2000-2002 and 2008 crashes) at a
similar or better Sharpe than buy-and-hold, typically at a lower CAGR. If we
can't reproduce that qualitative pattern, that's a methodology red flag
worth chasing before building anything more elaborate on top of this data.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from agents.universe_agent import UniverseAgent

SMA_MONTHS = 10


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    mu, sigma = returns.mean(), returns.std()
    return float(mu / sigma * np.sqrt(periods_per_year)) if sigma > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    n_periods = len(equity) - 1
    years = n_periods / periods_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def main() -> None:
    agent = UniverseAgent(
        tickers=["QQQ"], start_date="1999-01-01", end_date="2024-12-31",
        benchmark="QQQ", min_history_days=30, min_assets=1, data_source="yfinance",
    )
    ctx = agent.run({})
    daily = ctx["universe_data"]["QQQ"]["Close"]

    monthly = daily.resample("ME").last()
    sma = monthly.rolling(SMA_MONTHS).mean()

    # Signal at month t uses SMA computed through month t (available at
    # month-end close, no lookahead); applied to month t+1's return.
    long_signal = (monthly > sma).astype(int)
    monthly_ret = monthly.pct_change()
    strat_ret = (long_signal.shift(1) * monthly_ret).dropna()
    bh_ret = monthly_ret.dropna()

    strat_equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + bh_ret).cumprod()

    print(f"Period: {monthly.index[0].date()} -> {monthly.index[-1].date()}  "
          f"({len(monthly)} months)")
    print()
    print(f"{'Metric':<20}{'Faber 10mo SMA':>18}{'Buy & Hold':>18}")
    print(f"{'CAGR':<20}{cagr(strat_equity, 12):>17.1%} {cagr(bh_equity, 12):>17.1%}")
    print(f"{'Sharpe (ann.)':<20}{sharpe(strat_ret, 12):>18.2f}{sharpe(bh_ret, 12):>18.2f}")
    print(f"{'Max Drawdown':<20}{max_drawdown(strat_equity):>17.1%} {max_drawdown(bh_equity):>17.1%}")
    print(f"{'Time in market':<20}{long_signal.mean():>17.1%}")

    dotcom = strat_equity.loc["2000-01-01":"2002-12-31"]
    dotcom_bh = bh_equity.loc["2000-01-01":"2002-12-31"]
    if len(dotcom) > 1:
        print()
        print(f"Dot-com window (2000-2002) — strategy: "
              f"{dotcom.iloc[-1] / dotcom.iloc[0] - 1:+.1%}  "
              f"buy-hold: {dotcom_bh.iloc[-1] / dotcom_bh.iloc[0] - 1:+.1%}")

    gfc = strat_equity.loc["2007-10-01":"2009-03-31"]
    gfc_bh = bh_equity.loc["2007-10-01":"2009-03-31"]
    if len(gfc) > 1:
        print(f"GFC window (2007-2009)    — strategy: "
              f"{gfc.iloc[-1] / gfc.iloc[0] - 1:+.1%}  "
              f"buy-hold: {gfc_bh.iloc[-1] / gfc_bh.iloc[0] - 1:+.1%}")


if __name__ == "__main__":
    main()
