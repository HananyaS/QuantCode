from __future__ import annotations

import pytest

from research.intraday.contracts.events import Fill
from research.intraday.contracts.fixtures import T_DEV, make_fill
from research.intraday.risk.books import Books


def fill(side: str, qty: int, price: float, bps: float, iid: str = "i1", sym: str = "AAA") -> Fill:
    return make_fill(ts=T_DEV, intent_id=iid, symbol=sym, side=side, qty=qty, price=price, cost_bps=bps)

def test_partial_buys_then_sell_match_cash_formula() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02)
    b.apply_fill(fill("buy", 120, 1.03, 10.0)); b.apply_fill(fill("buy", 80, 1.04, 10.0))
    paid = 120 * 1.03 * 1.001 + 80 * 1.04 * 1.001
    b.apply_fill(fill("sell", 150, 1.10, 20.0))
    got = 150 * 1.10 * 0.998
    assert b.cash == pytest.approx(25_000.0 - paid + got, rel=1e-12)
    assert b.realised_pnl == pytest.approx(got - paid * 150 / 200, rel=1e-12)
    assert b.positions() == {"AAA": (50, pytest.approx(paid * 50 / 200, rel=1e-12))}
    assert not b.reserved_cash() and b.occupied() == 1

def test_unknown_or_mismatched_fill_raises() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02)
    with pytest.raises(ValueError, match="unknown intent_id"):
        b.apply_fill(fill("buy", 1, 1.0, 0.0, iid="nope"))
    with pytest.raises(ValueError):
        b.apply_fill(fill("buy", 1, 1.0, 0.0, sym="BBB"))

def test_overfill_and_fill_after_cancel_raise() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02)
    with pytest.raises(ValueError):
        b.apply_fill(fill("buy", 201, 1.0, 0.0))
    b.apply_fill(fill("buy", 120, 1.0, 0.0)); b.close_working("i1")
    with pytest.raises(ValueError):
        b.apply_fill(fill("buy", 1, 1.0, 0.0))

def test_oversell_raises() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02); b.apply_fill(fill("buy", 50, 1.0, 0.0))
    with pytest.raises(ValueError):
        b.apply_fill(fill("sell", 51, 1.0, 0.0))

def test_partial_then_close_working_keeps_position() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02)
    b.apply_fill(fill("buy", 120, 1.0, 0.0))
    assert b.occupied() == 1 and b.reserved_cash() == pytest.approx(80 * 1.02)
    b.close_working("i1")
    assert b.position_qty("AAA") == 120 and b.is_busy("AAA") and b.reserved_cash() == 0.0
    b.close_working("i1")   # known, no longer working: no-op
    with pytest.raises(ValueError):
        b.close_working("nope")

def test_marked_equity() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 100, 1.0); b.apply_fill(fill("buy", 100, 1.0, 0.0))
    assert b.marked_equity({"AAA": 0.8}) == pytest.approx(25_000.0 - 20.0)
    with pytest.raises(ValueError):
        b.marked_equity({})

def test_duplicate_open_intent_raises() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 100, 1.0)
    with pytest.raises(ValueError):
        b.open_intent("i1", "BBB", 100, 1.0)
