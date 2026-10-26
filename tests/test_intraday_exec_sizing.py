"""Fill size cap and quote freshness (plan Task 3)."""
from __future__ import annotations

from datetime import timedelta

import pytest

from research.intraday.contracts.fixtures import T_DEV, make_bar, make_quote
from research.intraday.exec.config import ExecConfig
from research.intraday.exec.sizing import fill_cap

M = timedelta(minutes=1)


def test_fresh_quote_caps_by_side_size() -> None:
    b = make_bar(ts=T_DEV + 2 * M, bar_start=T_DEV + M, volume=10_000)
    q = make_quote(ts=T_DEV, ask_size=40, bid_size=70)
    assert fill_cap("buy", b.bar_start, b, q, ExecConfig()) == (40, "quote")
    assert fill_cap("sell", b.bar_start, b, q, ExecConfig()) == (70, "quote")


def test_stale_quote_falls_back_to_bar_volume() -> None:
    b = make_bar(ts=T_DEV + 7 * M, bar_start=T_DEV + 6 * M, volume=10_000)
    q = make_quote(ts=T_DEV, ask_size=40)            # age 6 min > 5
    assert fill_cap("buy", b.bar_start, b, q, ExecConfig()) == (200, "bar")
    q5 = make_quote(ts=T_DEV + M, ask_size=40)        # age exactly 5 min: fresh
    assert fill_cap("buy", b.bar_start, b, q5, ExecConfig()) == (40, "quote")


def test_quote_after_fill_instant_raises() -> None:
    b = make_bar(ts=T_DEV + 2 * M, bar_start=T_DEV + M)
    with pytest.raises(AssertionError):
        fill_cap("buy", b.bar_start, b, make_quote(ts=b.ts), ExecConfig())


def test_cap_is_int_and_floors() -> None:
    b = make_bar(ts=T_DEV + M, bar_start=T_DEV, volume=149)
    cap, _ = fill_cap("buy", b.bar_start, b, None, ExecConfig())
    assert cap == 2 and type(cap) is int
