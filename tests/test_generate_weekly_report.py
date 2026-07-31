"""Tests for scripts/generate_weekly_report.py — the Saturday email digest
built from the live-trading state ledgers.

Pins the week-windowing logic (Mon-Fri of the week containing the last
snapshot), the weekly-return base (last snapshot BEFORE the week, falling
back to inception capital), and that the email HTML is email-client-safe
(no SVG -- Gmail strips it -- and no external assets).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_weekly_report import build_weekly_email_html, week_summary
from utils.state_store import StateStore


@pytest.fixture
def two_week_store(tmp_path):
    """Snapshots spanning two calendar weeks; the report week is the second
    (Mon 2026-07-27 .. Fri 2026-07-31), with the prior Friday as the base.
    """
    store = StateStore(db_path=str(tmp_path / "kelly.db"))
    store.record_account_snapshot("2026-07-24", equity=25_000.0, cash=25_000.0, positions={})
    store.record_account_snapshot("2026-07-28", equity=25_465.52, cash=8_500.0,
                                  positions={"TQQQ": 16_900.0})
    store.record_account_snapshot("2026-07-29", equity=24_751.33, cash=8_500.0,
                                  positions={"TQQQ": 16_200.0})
    store.record_account_snapshot("2026-07-31", equity=26_100.0, cash=8_500.0,
                                  positions={"TQQQ": 17_600.0})
    store.record_order("2026-07-28", "TQQQ", "buy", 265, "o1", "filled")
    store.record_order("2026-07-31", "TQQQ", "sell", 57, "o2", "filled")
    return store


@pytest.fixture
def empty_store(tmp_path):
    return StateStore(db_path=str(tmp_path / "empty.db"))


def test_week_window_is_mon_fri_of_last_snapshot(two_week_store):
    s = week_summary("Kelly", two_week_store)
    assert s["week_start"] == "2026-07-27"
    assert s["week_end"] == "2026-07-31"
    assert [r["date"] for r in s["daily"]] == ["2026-07-28", "2026-07-29", "2026-07-31"]


def test_week_return_uses_prior_week_close_as_base(two_week_store):
    s = week_summary("Kelly", two_week_store)
    assert s["week_return_pct"] == pytest.approx(100 * (26_100.0 / 25_000.0 - 1))


def test_week_orders_filtered_to_window(tmp_path):
    store = StateStore(db_path=str(tmp_path / "k.db"))
    store.record_account_snapshot("2026-07-31", equity=25_000.0, cash=25_000.0, positions={})
    store.record_order("2026-07-24", "TQQQ", "buy", 100, "old", "filled")   # prior week
    store.record_order("2026-07-30", "TQQQ", "buy", 5, "in-week", "filled")
    s = week_summary("Kelly", store)
    assert len(s["orders"]) == 1
    assert s["orders"].iloc[0]["order_id"] == "in-week"


def test_week_return_falls_back_to_inception_capital(tmp_path):
    # First-ever week: no snapshot before Monday -- base must be the known
    # $25,000 inception capital, not the week's own (post-gain) first row.
    store = StateStore(db_path=str(tmp_path / "k.db"))
    store.record_account_snapshot("2026-07-28", equity=25_539.77, cash=9_056.77,
                                  positions={"TQQQ": 16_483.0})
    s = week_summary("Kelly", store)
    assert s["week_return_pct"] == pytest.approx(100 * (25_539.77 / 25_000.0 - 1))


def test_empty_ledger_is_graceful(empty_store):
    s = week_summary("Linear", empty_store)
    assert s["week_end"] is None
    assert s["daily"] == []


def test_email_html_contains_key_numbers(two_week_store, empty_store):
    html = build_weekly_email_html({"Kelly": two_week_store, "Linear": empty_store})
    assert "26,100.00" in html
    assert "Kelly" in html and "Linear" in html
    assert "SELL" in html and "BUY" in html
    assert "hananyas.github.io/QuantCode/live-report" in html  # dashboard link


def test_email_html_is_email_client_safe(two_week_store):
    html = build_weekly_email_html({"Kelly": two_week_store})
    assert "<svg" not in html          # Gmail strips SVG entirely
    assert "<script" not in html
    assert "https://cdn" not in html   # no external assets
