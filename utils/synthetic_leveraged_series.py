"""synthetic_leveraged_series.py — exact daily-reset NAV recursion for
constructing synthetic QLD(2x)/TQQQ(3x) price series from real QQQ returns.

Why synthetic instead of real cached TQQQ/QLD prices
-------------------------------------------------------
Real TQQQ/QLD history is inception-limited (TQQQ: 2010-02-11, QLD:
2006-06-21) — far short of QQQ's 1999+ history, which matters for stress-
testing the dot-com crash and other tail regimes. Synthetic construction
also lets expense-ratio drag be modeled as an explicit, isolated cost term
rather than left implicit (and entangled with real tracking error) inside
whatever price series a data vendor returns.

The formula
-----------
V_{t+1} = V_t * max(1 + L*r_t - daily_expense, 0)

NEVER `L * cumulative_underlying_return` — that naive multiplicative
approximation is wrong for daily-reset leveraged products and silently
misstates every result (see test_diverges_from_naive_multiplicative_approximation
in tests/test_synthetic_leveraged_series.py for a worked example). The
max(..., 0) floor is absorbing: once a fund is wiped out by a single-day
move worse than -1/L, it stays at zero regardless of any subsequent
recovery in the underlying — matching a real leveraged fund's mechanics
(a reset/wind-down event, not a recoverable drawdown).
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

_TRADING_DAYS_PER_YEAR = 252

# (leverage, ticker, total annual cost-of-carry) — NOT just the published
# prospectus expense ratio (0.95% for both QLD/TQQQ). Cross-validating the
# zero-cost recursion against real cached TQQQ/QLD price history
# (calibrate_total_annual_drag, below) shows the published expense ratio
# badly understates the true long-run drag: real leveraged ETFs finance
# their exposure via swaps/futures whose implicit borrowing cost tracks
# short-term interest rates, adding a cost the prospectus figure doesn't
# capture. Empirically calibrated full-history averages used here:
#   TQQQ 2010-2026: 5.42%/yr total drag (vs. 0.95% stated)
#   QLD  2006-2026: 3.26%/yr total drag (vs. 0.95% stated)
# IMPORTANT CAVEAT: this drag is strongly rate-regime-dependent, not a
# stable constant — calibrated by decade: ~1.9%/yr in the 2010-2015
# near-zero-rate era vs. ~11%/yr in the 2022-2026 high-rate era. A single
# full-history average is a reasonable default for a "typical" backtest but
# will UNDERSTATE current costs in a high-rate regime and OVERSTATE them in
# a low-rate one — see docs/research-log/kelly-leverage-sizing.md.
_DEFAULT_TIERS: Tuple[Tuple[float, str, float], ...] = (
    (1.0, "QQQ", 0.0020),
    (2.0, "QLD", 0.0326),
    (3.0, "TQQQ", 0.0542),
)


def build_synthetic_leveraged_close(
    underlying_close: pd.Series,
    leverage: float,
    expense_ratio_annual: float,
    trading_days_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Construct a synthetic daily-reset leveraged Close series from a real
    underlying Close series via the exact NAV recursion.

    Args:
        underlying_close: Real daily Close prices of the underlying (e.g. QQQ).
        leverage: Daily leverage factor L (e.g. 2.0 for QLD, 3.0 for TQQQ).
        expense_ratio_annual: Annualized expense ratio (e.g. 0.0095 for 0.95%),
            charged as a daily drag inside the same max(..., 0) floor as the
            leveraged market return.
        trading_days_per_year: Divisor for converting the annual expense
            ratio to a daily charge.

    Returns:
        Synthetic Close series, same index as underlying_close, starting at
        underlying_close's own first value.
    """
    underlying_return = underlying_close.pct_change()
    daily_expense = expense_ratio_annual / trading_days_per_year

    fund_daily_return = leverage * underlying_return - daily_expense
    factor = (1 + fund_daily_return).clip(lower=0.0)
    factor.iloc[0] = 1.0  # no elapsed time on the first bar

    nav = underlying_close.iloc[0] * factor.cumprod()
    return nav


def calibrate_total_annual_drag(
    underlying_close: pd.Series,
    real_leveraged_close: pd.Series,
    leverage: float,
    trading_days_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> float:
    """Back out the TRUE total annual cost-of-carry a real leveraged fund
    bears, by comparing its real daily returns against L times the real
    underlying's daily returns:

        implied_daily_drag_t ~= L*underlying_return_t - real_fund_return_t

    Averaged over the overlapping window and annualized. This is what
    should be used instead of the published prospectus expense ratio —
    see _DEFAULT_TIERS' comment for why the two differ substantially.
    """
    common = underlying_close.index.intersection(real_leveraged_close.index)
    underlying_ret = underlying_close.loc[common].pct_change()
    real_ret = real_leveraged_close.loc[common].pct_change()

    aligned = pd.concat([underlying_ret, real_ret], axis=1).dropna()
    implied_daily_drag = leverage * aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(implied_daily_drag.mean() * trading_days_per_year)


def build_synthetic_universe(
    underlying_close: pd.Series,
    tiers: Tuple[Tuple[float, str, float], ...] = _DEFAULT_TIERS,
) -> Dict[str, pd.DataFrame]:
    """Build a universe_data-shaped dict (ticker -> OHLCV DataFrame with at
    least a 'Close' column) for every tier. The leverage=1.0 tier is a
    verbatim pass-through of `underlying_close` (its real expense ratio is
    already embedded in real historical returns — applying a second,
    synthetic expense-drag layer on top would double-count it). Only tiers
    with leverage > 1.0 go through the synthetic recursion.
    """
    universe: Dict[str, pd.DataFrame] = {}
    for leverage, ticker, expense_ratio in tiers:
        if leverage == 1.0:
            close = underlying_close
        else:
            close = build_synthetic_leveraged_close(underlying_close, leverage, expense_ratio)
        universe[ticker] = pd.DataFrame({
            "Open": close, "High": close, "Low": close, "Close": close,
            "Volume": 0,
        })
    return universe
