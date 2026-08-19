"""Tests for the once-per-trading-day preflight gate.

These pin the behaviour that stopped the duplicate phone notifications:
the scheduled workflow fires twice every weekday and knows nothing about
the NYSE calendar, so the *code* has to decide when a firing is real. A
firing that has nothing to do must do nothing AND say nothing.
"""
from __future__ import annotations

import pytest

from scripts.preflight_check import Preflight, decide, _already_ran
from utils.state_store import StateStore


# ---------------------------------------------------------------------------
# The decision table
# ---------------------------------------------------------------------------

def test_first_firing_of_an_open_session_trades_and_notifies():
    pf = decide(is_trading_day=True, already_ran=False, market_open=True)
    assert pf.trade is True
    assert pf.notify is True


def test_backup_firing_after_a_completed_session_is_completely_silent():
    """The duplicate-notification bug: the backup cron used to push a
    meaningless 'succeeded' every single weekday."""
    pf = decide(is_trading_day=True, already_ran=True, market_open=True)
    assert pf.trade is False
    assert pf.notify is False


@pytest.mark.parametrize("already_ran", [False, True])
def test_non_trading_day_is_completely_silent(already_ran):
    """Weekends are excluded by cron, but market holidays are not -- cron
    fires Mon-Fri regardless. Those firings used to send two pushes."""
    pf = decide(is_trading_day=False, already_ran=already_ran, market_open=False)
    assert pf.trade is False
    assert pf.notify is False


def test_missed_session_after_the_close_skips_but_does_notify():
    """A day that never ran and can no longer run is worth hearing about --
    but must not trade, or the order queues overnight on a stale signal."""
    pf = decide(is_trading_day=True, already_ran=False, market_open=False)
    assert pf.trade is False
    assert pf.notify is True


def test_completed_session_stays_silent_even_after_the_close():
    """`already_ran` must be checked before `market_open`: a normal day
    whose backup firing lands late is done, not missed."""
    pf = decide(is_trading_day=True, already_ran=True, market_open=False)
    assert pf.trade is False
    assert pf.notify is False


@pytest.mark.parametrize("is_trading_day", [False, True])
@pytest.mark.parametrize("already_ran", [False, True])
@pytest.mark.parametrize("market_open", [False, True])
def test_manual_run_always_executes_and_reports(is_trading_day, already_ran, market_open):
    """The quiet rule governs the schedule, not a human pressing the button
    and waiting on the answer."""
    pf = decide(
        is_trading_day=is_trading_day,
        already_ran=already_ran,
        market_open=market_open,
        manual=True,
    )
    assert pf.trade is True
    assert pf.notify is True


def test_decision_carries_a_human_readable_reason():
    for pf in (
        decide(is_trading_day=True, already_ran=False, market_open=True),
        decide(is_trading_day=True, already_ran=True, market_open=True),
        decide(is_trading_day=False, already_ran=False, market_open=False),
        decide(is_trading_day=True, already_ran=False, market_open=False),
    ):
        assert isinstance(pf, Preflight)
        assert pf.reason.strip(), "every decision must explain itself in the log"


# ---------------------------------------------------------------------------
# "Did today's run already happen?"
# ---------------------------------------------------------------------------

def test_has_snapshot_on_is_true_even_when_no_orders_were_placed(tmp_path):
    """A no-change day submits no orders, so `has_orders_on` cannot tell a
    completed run from one that never happened. `has_snapshot_on` can --
    which is why the preflight keys on it."""
    store = StateStore(db_path=str(tmp_path / "s.db"))
    store.record_account_snapshot("2026-08-19", equity=25_000.0, cash=25_000.0, positions={})

    assert store.has_orders_on("2026-08-19") is False
    assert store.has_snapshot_on("2026-08-19") is True


def test_has_snapshot_on_is_false_for_an_unrecorded_session(tmp_path):
    store = StateStore(db_path=str(tmp_path / "s.db"))
    store.record_account_snapshot("2026-08-19", equity=25_000.0, cash=25_000.0, positions={})

    assert store.has_snapshot_on("2026-08-20") is False


def _seed(path, run_date):
    StateStore(db_path=str(path)).record_account_snapshot(
        run_date, equity=25_000.0, cash=25_000.0, positions={},
    )


@pytest.mark.parametrize(
    "seed_kelly, seed_linear, expected",
    [
        (True, True, True),      # both done -> the day is complete
        (True, False, False),    # Linear died -> backup must still run
        (False, True, False),    # Kelly died -> backup must still run
        (False, False, False),   # nothing ran yet
    ],
)
def test_already_ran_requires_both_strategies(tmp_path, monkeypatch, seed_kelly, seed_linear, expected):
    """Only a fully successful primary earns the backup's silence."""
    kelly_db, linear_db = tmp_path / "k.db", tmp_path / "l.db"
    if seed_kelly:
        _seed(kelly_db, "2026-08-19")
    if seed_linear:
        _seed(linear_db, "2026-08-19")

    monkeypatch.setattr("scripts.preflight_check._KELLY_DB", str(kelly_db))
    monkeypatch.setattr("scripts.preflight_check._LINEAR_DB", str(linear_db))

    assert _already_ran("2026-08-19") is expected
