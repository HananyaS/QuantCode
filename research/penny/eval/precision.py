"""precision.py — the KPIs a precision strategy is actually judged on.

Rounds 1-3 reported mean net return, a confidence interval, and the fraction of
candidates retained. That set is built for a recall question: it asks how good
the average trade is and how many were taken.

The strategy this round is testing takes very few trades and needs them to be
right. The numbers that decide whether such a thing is usable are different:
how often it wins, how much a win pays relative to a loss, how often it trades
at all, and how bad the worst stretch gets. A rule with a positive mean and
eleven consecutive losers is not deployable regardless of its interval.

``retained`` is deliberately absent as a headline. It is an *output* of the
expected-value rule, not a setting, and reporting it as a KPI invites tuning it.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def max_consecutive(flags: np.ndarray) -> int:
    """Longest run of True in a chronologically ordered boolean array."""
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return int(best)


def precision_kpis(
    trades: pd.DataFrame,
    return_col: str = "net_return",
    date_col: str = "date",
) -> Dict[str, float]:
    """Everything needed to judge a low-recall rule, from a selected trade set.

    ``trades`` holds only the taken trades. Rows with a missing return are
    dropped and counted, because a trade that could not be priced is not
    evidence either way and silently keeping it as a zero would flatter a rule
    that abstains on hard names.
    """
    if trades.empty or return_col not in trades.columns:
        return {"n": 0}

    df = trades.dropna(subset=[return_col]).sort_values(date_col)
    dropped = len(trades) - len(df)
    if df.empty:
        return {"n": 0, "n_unpriced": int(dropped)}

    r = df[return_col].astype(float)
    wins, losses = r[r > 0], r[r <= 0]

    by_day = df.groupby(date_col)[return_col].mean()
    months = df[date_col].dt.to_period("M")
    by_month = df.groupby(months)[return_col].mean()
    span_months = max(months.nunique(), 1)

    gross_win = float(wins.sum())
    gross_loss = float(losses.abs().sum())

    return {
        "n": int(len(df)),
        "n_unpriced": int(dropped),
        "n_days": int(df[date_col].nunique()),
        "expectancy": float(r.mean()),
        "median": float(r.median()),
        "hit_rate": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.abs().mean()) if len(losses) else 0.0,
        # A profit factor of exactly inf (no losers) is a sample-size artefact,
        # not a perfect strategy; reported as NaN so it cannot be ranked on.
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("nan"),
        "payoff_ratio": (
            float(wins.mean() / losses.abs().mean())
            if len(wins) and len(losses) and losses.abs().mean() > 0 else float("nan")
        ),
        "trades_per_month": float(len(df) / span_months),
        "worst_day": float(by_day.min()),
        "worst_month": float(by_month.min()),
        "best_day": float(by_day.max()),
        "max_consecutive_losers": max_consecutive((r <= 0).to_numpy()),
        "pct_days_traded": float(df[date_col].nunique()),
    }


def kpi_table(rows: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Stack per-configuration KPI dicts into one comparable frame."""
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).T
    out.index.name = "configuration"
    return out.reset_index()
