"""go_live_gate.py — numeric go/no-go criteria from paper-trading history.

Why this exists
----------------
ROADMAP.md Phase 4: no phase advances to real capital without an explicit,
numeric gate. This is decision support only — it evaluates a StateStore's
recorded paper-trading history against configurable thresholds and returns
a pass/fail verdict with reasons. It never places, authorizes, or blocks a
trade itself; a human still decides whether and when to actually flip to a
live-money account.

Metric methodology deliberately mirrors MultiAssetEvaluationAgent
(agents/multi_evaluation_agent.py) — same Sharpe/max-drawdown formulas — so
paper-trading gate metrics are directly comparable to backtest metrics.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.state_store import StateStore

# Order statuses that represent something going wrong, not a working risk
# control (e.g. skipped_pdt_guard is the PDT guard doing its job, not a failure).
_PROBLEM_STATUSES = {"rejected", "canceled", "skipped_no_price", "error"}


def _sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(252))


def _max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def evaluate_go_live(
    store: StateStore,
    min_days: int = 63,
    min_sharpe: float = 0.5,
    max_drawdown: float = 0.20,
    max_problem_orders: int = 0,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate paper-trading history against go-live gate criteria.

    Args:
        store: StateStore with recorded account snapshots and orders.
        min_days: Minimum number of recorded paper-trading days (63 trading
            days ≈ 3 months, matching ROADMAP.md Phase 3's target window).
        min_sharpe: Minimum acceptable Sharpe ratio over the recorded history.
        max_drawdown: Maximum acceptable drawdown magnitude (e.g. 0.20 = 20%).
        max_problem_orders: Maximum tolerated count of non-benign problem
            order statuses (rejected/canceled/skipped_no_price/error).
        risk_free_rate: Annualized risk-free rate for Sharpe.

    Returns:
        Dict with keys: passed (bool), reasons (List[str]), n_days,
        sharpe, max_drawdown, n_problem_orders.
    """
    snapshots = store.snapshots()
    n_days = len(snapshots)
    reasons = []

    if n_days < min_days:
        reasons.append(f"only {n_days} paper-trading days recorded, need >= {min_days} days")

    sharpe = float("nan")
    max_dd = float("nan")
    if n_days >= 2:
        equity = pd.Series(
            snapshots["equity"].values,
            index=pd.to_datetime(snapshots["run_date"]),
        ).sort_index()
        returns = equity.pct_change().dropna()
        sharpe = _sharpe(returns, risk_free_rate)
        max_dd = _max_drawdown(equity)

        if sharpe < min_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} below minimum {min_sharpe:.2f}")
        if max_dd < -max_drawdown:
            reasons.append(f"max drawdown {max_dd:.2%} exceeds limit {max_drawdown:.2%}")
    else:
        reasons.append("insufficient snapshot history to compute Sharpe/drawdown")

    all_orders = store.all_orders()
    n_problem = int(all_orders["status"].isin(_PROBLEM_STATUSES).sum()) if len(all_orders) else 0
    if n_problem > max_problem_orders:
        reasons.append(
            f"{n_problem} problem order(s) recorded, tolerance is {max_problem_orders}"
        )

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "n_days": n_days,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_problem_orders": n_problem,
    }
