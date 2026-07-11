"""Tests for utils/state_store.py — persistent live-trading position/order ledger."""
import pandas as pd
import pytest

from utils.state_store import StateStore


@pytest.fixture
def store(tmp_path):
    return StateStore(db_path=str(tmp_path / "live_state.db"))


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def test_record_and_read_order(store):
    store.record_order("2024-01-02", "AAPL", "buy", 10, "order-1", "filled")
    df = store.orders_for_date("2024-01-02")
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["side"] == "buy"
    assert df.iloc[0]["qty"] == 10


def test_orders_for_date_filters_by_date(store):
    store.record_order("2024-01-02", "AAPL", "buy", 10, "order-1", "filled")
    store.record_order("2024-01-03", "MSFT", "buy", 5, "order-2", "filled")
    df = store.orders_for_date("2024-01-02")
    assert list(df["ticker"]) == ["AAPL"]


def test_tickers_bought_on_only_includes_buys(store):
    store.record_order("2024-01-02", "AAPL", "buy", 10, "order-1", "filled")
    store.record_order("2024-01-02", "MSFT", "sell", 5, "order-2", "filled")
    bought = store.tickers_bought_on("2024-01-02")
    assert bought == {"AAPL"}


def test_orders_for_date_empty_when_no_orders(store):
    df = store.orders_for_date("2099-01-01")
    assert len(df) == 0


def test_all_orders_returns_every_recorded_order(store):
    store.record_order("2024-01-02", "AAPL", "buy", 10, "order-1", "filled")
    store.record_order("2024-01-03", "MSFT", "sell", 5, "order-2", "filled")
    df = store.all_orders()
    assert len(df) == 2
    assert set(df["ticker"]) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# Account snapshots
# ---------------------------------------------------------------------------

def test_record_and_read_account_snapshot(store):
    store.record_account_snapshot("2024-01-02", equity=100_000.0, cash=20_000.0,
                                   positions={"AAPL": 5000.0})
    snap = store.latest_snapshot()
    assert snap["run_date"] == "2024-01-02"
    assert snap["equity"] == 100_000.0
    assert snap["cash"] == 20_000.0
    assert snap["positions"] == {"AAPL": 5000.0}


def test_latest_snapshot_returns_none_when_empty(store):
    assert store.latest_snapshot() is None


def test_record_account_snapshot_upserts_same_date(store):
    store.record_account_snapshot("2024-01-02", equity=100_000.0, cash=20_000.0, positions={})
    store.record_account_snapshot("2024-01-02", equity=101_000.0, cash=21_000.0, positions={"AAPL": 1000.0})
    df = store.snapshots()
    assert len(df) == 1
    assert df.iloc[0]["equity"] == 101_000.0


def test_snapshots_sorted_by_date_ascending(store):
    store.record_account_snapshot("2024-01-03", equity=2.0, cash=1.0, positions={})
    store.record_account_snapshot("2024-01-02", equity=1.0, cash=1.0, positions={})
    df = store.snapshots()
    assert list(df["run_date"]) == ["2024-01-02", "2024-01-03"]
