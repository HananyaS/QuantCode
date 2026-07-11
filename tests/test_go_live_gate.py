"""Tests for utils/go_live_gate.py — numeric go/no-go criteria from paper-trading history.

Per ROADMAP.md Phase 4: no phase advances to real capital without an
explicit, numeric gate. This evaluates a StateStore's recorded paper-trading
history (account snapshots + orders) against configurable thresholds and
returns a pass/fail verdict with reasons — decision support only, it never
places or authorizes any trade itself.
"""
import pandas as pd
import pytest

from utils.go_live_gate import evaluate_go_live
from utils.state_store import StateStore


@pytest.fixture
def store(tmp_path):
    return StateStore(db_path=str(tmp_path / "live_state.db"))


def _seed_growing_equity(store, n_days=70, start=100_000.0, daily_return=0.001):
    """Alternates two positive daily returns around `daily_return` — a
    constant return has zero std (undefined Sharpe), so this keeps std > 0
    while still trending up, giving a well-defined, strongly positive Sharpe.
    """
    equity = start
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    for i, d in enumerate(dates):
        store.record_account_snapshot(str(d.date()), equity=equity, cash=equity * 0.2, positions={})
        step = daily_return * (1.6 if i % 2 == 0 else 0.4)
        equity *= (1 + step)
    return dates


# ---------------------------------------------------------------------------
# Insufficient history
# ---------------------------------------------------------------------------

def test_fails_when_too_few_days(store):
    _seed_growing_equity(store, n_days=10)
    result = evaluate_go_live(store, min_days=63)
    assert result["passed"] is False
    assert any("days" in r for r in result["reasons"])


def test_fails_when_no_history_at_all(store):
    result = evaluate_go_live(store, min_days=63)
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# Sharpe / drawdown thresholds
# ---------------------------------------------------------------------------

def test_passes_with_strong_steady_growth(store):
    _seed_growing_equity(store, n_days=70, daily_return=0.001)
    result = evaluate_go_live(store, min_days=63, min_sharpe=0.5, max_drawdown=0.20)
    assert result["passed"] is True
    assert result["reasons"] == []
    assert result["sharpe"] > 0.5


def test_fails_on_low_sharpe(store):
    # Flat-ish equity with noise -> near-zero Sharpe
    dates = pd.bdate_range("2024-01-02", periods=70)
    equity = 100_000.0
    for i, d in enumerate(dates):
        equity *= (1 + (0.001 if i % 2 == 0 else -0.001))
        store.record_account_snapshot(str(d.date()), equity=equity, cash=0.0, positions={})
    result = evaluate_go_live(store, min_days=63, min_sharpe=1.0)
    assert result["passed"] is False
    assert any("sharpe" in r.lower() for r in result["reasons"])


def test_fails_on_excessive_drawdown(store):
    dates = pd.bdate_range("2024-01-02", periods=70)
    equity_vals = [100_000.0 * (1.01 ** i) for i in range(35)]
    equity_vals += [equity_vals[-1] * (0.6 ** ((i - 34) / 35)) for i in range(35, 70)]  # sharp decline
    for d, e in zip(dates, equity_vals):
        store.record_account_snapshot(str(d.date()), equity=e, cash=0.0, positions={})
    result = evaluate_go_live(store, min_days=63, min_sharpe=-100.0, max_drawdown=0.10)
    assert result["passed"] is False
    assert any("drawdown" in r.lower() for r in result["reasons"])


# ---------------------------------------------------------------------------
# Problem orders
# ---------------------------------------------------------------------------

def test_fails_on_problem_orders_exceeding_tolerance(store):
    _seed_growing_equity(store, n_days=70, daily_return=0.001)
    store.record_order("2024-01-05", "AAPL", "buy", 10, None, "rejected")
    result = evaluate_go_live(store, min_days=63, min_sharpe=0.5, max_problem_orders=0)
    assert result["passed"] is False
    assert any("problem order" in r.lower() for r in result["reasons"])


def test_pdt_guard_skips_are_not_problem_orders(store):
    """skipped_pdt_guard is a working risk control, not a failure."""
    _seed_growing_equity(store, n_days=70, daily_return=0.001)
    store.record_order("2024-01-05", "AAPL", "sell", 0, None, "skipped_pdt_guard")
    result = evaluate_go_live(store, min_days=63, min_sharpe=0.5, max_problem_orders=0)
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

def test_result_contains_expected_keys(store):
    _seed_growing_equity(store, n_days=70)
    result = evaluate_go_live(store)
    for key in ("passed", "reasons", "n_days", "sharpe", "max_drawdown", "n_problem_orders"):
        assert key in result
