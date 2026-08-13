"""portfolio.py — what the account would have done.

Why per-trade expectancy is the wrong object
--------------------------------------------
Six rounds have reported the mean return *per trade*. That weights every trade
equally. An account weights every **day** equally: on a session with one
candidate you are wholly in one name, and on a session with fifty you are
diversified across fifty. The two numbers differ whenever the number of
candidates per day correlates with the return of those candidates — which in a
reversal strategy it plausibly does, since candidate count spikes on turbulent
days.

This is not a way to manufacture an edge out of a flat one. Diversification
reduces the *variance* of the daily series, not its mean, and the day-clustered
bootstrap already used throughout the study accounts for within-day clustering.
What a portfolio layer adds is:

* the **actual object of interest** — a daily return series and its equity
  curve, rather than a per-trade average nobody could have traded;
* **capital realism** — a cap on simultaneous positions, since an account cannot
  take 300 names on a busy day;
* **path** — Sharpe, drawdown and the losing-streak profile, which decide
  whether a thin positive mean is survivable in practice.

Days with no qualifying trade earn zero, not NaN. Dropping them would report the
strategy's return only on the days it chose to act, which flatters any rule that
abstains during bad conditions -- and abstention is exactly what the
expected-value rules in this study do.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class PortfolioResult:
    """Daily series plus the statistics an account is judged on."""

    daily: pd.Series
    equity: pd.Series
    n_days: int
    n_trades: int
    mean_daily: float
    sd_daily: float
    sharpe: float
    total_return: float
    max_drawdown: float
    hit_rate_days: float
    worst_day: float
    best_day: float
    max_consecutive_losing_days: int
    avg_positions: float
    pct_days_active: float

    def to_dict(self) -> Dict[str, float]:
        return {
            k: v for k, v in self.__dict__.items()
            if not isinstance(v, (pd.Series, pd.DataFrame))
        }


def _max_consecutive(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return int(best)


def daily_returns(
    trades: pd.DataFrame,
    return_col: str = "net_return",
    date_col: str = "date",
    max_positions: Optional[int] = None,
    rank_col: Optional[str] = None,
    all_sessions: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Equal-weight daily return of the selected trades.

    ``max_positions`` caps simultaneous holdings; when it binds, the kept names
    are the highest ``rank_col`` (falling back to an arbitrary but deterministic
    order). ``all_sessions`` supplies the full trading calendar so that days the
    strategy sat out are counted as flat rather than skipped.
    """
    if trades.empty:
        return pd.DataFrame(columns=[date_col, "ret", "n_positions"])

    df = trades.dropna(subset=[return_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=[date_col, "ret", "n_positions"])

    if max_positions is not None:
        if rank_col and rank_col in df.columns:
            df = df.sort_values([date_col, rank_col], ascending=[True, False])
        else:
            df = df.sort_values([date_col, return_col.replace(return_col, date_col)])
        df = df.groupby(date_col, group_keys=False).head(max_positions)

    out = df.groupby(date_col).agg(
        ret=(return_col, "mean"), n_positions=(return_col, "size")
    ).reset_index()

    if all_sessions is not None:
        cal = pd.DataFrame({date_col: pd.to_datetime(pd.Index(all_sessions).unique())})
        out = cal.merge(out, on=date_col, how="left")
        out["ret"] = out["ret"].fillna(0.0)
        out["n_positions"] = out["n_positions"].fillna(0).astype(int)

    return out.sort_values(date_col).reset_index(drop=True)


def evaluate_portfolio(
    trades: pd.DataFrame,
    return_col: str = "net_return",
    date_col: str = "date",
    max_positions: Optional[int] = None,
    rank_col: Optional[str] = None,
    all_sessions: Optional[Sequence] = None,
) -> Optional[PortfolioResult]:
    """Daily series, equity curve and the statistics an account is judged on."""
    daily = daily_returns(trades, return_col, date_col, max_positions,
                          rank_col, all_sessions)
    if daily.empty or len(daily) < 5:
        return None

    r = daily["ret"].astype(float)
    # Compounded, not summed: an account compounds, and on a series with this
    # much variance the two diverge materially over a decade.
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    sd = float(r.std())
    # `sd > 0` is not a sufficient guard: a constant series accumulates
    # floating-point noise and lands at ~1e-19, which yields a Sharpe of 1e16.
    # A ratio against a standard deviation at machine-epsilon scale is not a
    # number anyone should act on, so it is NaN.
    sd_is_meaningful = sd > 1e-12
    active = daily["n_positions"] > 0

    return PortfolioResult(
        daily=r,
        equity=equity,
        n_days=int(len(daily)),
        n_trades=int(daily["n_positions"].sum()),
        mean_daily=float(r.mean()),
        sd_daily=sd,
        sharpe=(float(r.mean() / sd * np.sqrt(TRADING_DAYS))
                if sd_is_meaningful else float("nan")),
        total_return=float(equity.iloc[-1] - 1.0),
        max_drawdown=float(dd.min()),
        hit_rate_days=float((r > 0).mean()),
        worst_day=float(r.min()),
        best_day=float(r.max()),
        max_consecutive_losing_days=_max_consecutive((r < 0).to_numpy()),
        avg_positions=float(daily.loc[active, "n_positions"].mean())
        if active.any() else 0.0,
        pct_days_active=float(active.mean()),
    )


def compare_weightings(
    trades: pd.DataFrame,
    return_col: str = "net_return",
    date_col: str = "date",
) -> Dict[str, float]:
    """Per-trade mean against per-day mean, and the gap between them.

    A large gap means candidate count covaries with candidate quality, so the
    per-trade average the study has reported for six rounds is not what an
    account would have earned.
    """
    df = trades.dropna(subset=[return_col])
    if df.empty:
        return {}
    per_trade = float(df[return_col].mean())
    by_day = df.groupby(date_col)[return_col].mean()
    per_day = float(by_day.mean())
    counts = df.groupby(date_col).size()
    return {
        "per_trade_mean": per_trade,
        "per_day_mean": per_day,
        "gap": per_day - per_trade,
        "corr_count_vs_return": float(counts.corr(by_day)),
        "n_days": int(len(by_day)),
        "median_positions": float(counts.median()),
    }
