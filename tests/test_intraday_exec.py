"""ExecutionAgent + SimBroker: the design's named tests (execution.md, Test plan; plan Tasks 5-7)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from research.intraday.contracts.events import Bar, Fill, OrderUpdate
from research.intraday.contracts.fixtures import (
    T_DEV, make_bar, make_clock_tick, make_halt, make_kill, make_order_intent, make_quote,
)
from research.intraday.exec import CostInputs, ExecConfig, ExecutionAgent, SimBroker, StaticCostInputs
from research.intraday.exec.costs import leg_cost_bps_scalar
from research.penny.etl.quotes import leg_cost_bps

M = timedelta(minutes=1)
SESSION = date(2024, 3, 4)


def at(k: int) -> datetime:
    """at(0) = 09:31 ET, at(3) = 09:34, at(390) = 16:01."""
    return T_DEV + k * M


def bar(k: int, symbol: str = "TEST", **ov: Any) -> Bar:
    """bar_start = at(k), ts = at(k + 1)."""
    return make_bar(symbol=symbol, ts=at(k + 1), bar_start=at(k), **ov)


INPUTS = CostInputs(sigma_daily=0.05, adv_shares=1_000_000.0)


def make_exec(inputs: dict[str, CostInputs] | None = None, **cfg: Any) -> tuple[ExecutionAgent, SimBroker]:
    table = {(s, SESSION): ci for s, ci in (inputs or {"TEST": INPUTS, "AAA": INPUTS}).items()}
    broker = SimBroker(ExecConfig(**cfg), StaticCostInputs(table))
    return ExecutionAgent(broker), broker


# ---- Task 5: entries --------------------------------------------------------
def test_new_intent_accepted_with_intent_ts() -> None:
    ex, _ = make_exec()
    assert ex.on_event(make_order_intent(ts=at(1))) == (
        OrderUpdate(ts=at(1), intent_id="intent-1", status="accepted", detail=""),)


def test_entry_fills_at_next_bar_open_not_causing_bar() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))
    assert ex.on_event(bar(0)) == ()                      # bar_start 09:31 < intent ts: never fills it
    (f,) = ex.on_event(bar(1, open=1.10, high=1.12, low=1.08, close=1.11))
    assert (f.ts, f.side, f.qty, f.price) == (at(2), "buy", 100, 1.10)


def test_gap_minute_moves_fill_to_next_bar() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))              # minute at(1) is a Gap: no bar
    (f,) = ex.on_event(bar(2, open=1.03, high=1.05, close=1.04))
    assert (f.ts, f.price) == (at(3), 1.03)


def test_entry_expires_after_max_wait() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))              # eligible bar_starts: at(1), at(2)
    assert ex.on_event(make_clock_tick(ts=at(3))) == ()
    assert ex.on_event(make_clock_tick(ts=at(4))) == (
        OrderUpdate(ts=at(4), intent_id="intent-1", status="expired", detail="entry wait exceeded"),)
    ex2, _ = make_exec()
    ex2.on_event(make_order_intent(ts=at(1)))
    assert [e.status for e in ex2.on_event(bar(3))] == ["expired"]   # bar path, no fill


def test_fill_capped_by_shares_at_best_quote() -> None:
    ex, _ = make_exec()
    ex.on_event(make_quote(ts=at(1), ask_size=40))
    ex.on_event(make_order_intent(ts=at(1)))
    f, u = ex.on_event(bar(1))
    assert (f.qty, u.status) == (40, "cancelled")


def test_partial_entry_remainder_cancelled() -> None:
    ex, broker = make_exec()
    ex.on_event(make_order_intent(ts=at(1), qty=300))     # bar cap 200
    f, u = ex.on_event(bar(1))
    assert f.qty == 200 and u.detail == "partial fill 200/300: remainder 100 cancelled"
    assert broker.position("TEST") == 200 and ex.on_event(bar(2)) == ()


def test_partial_fill_precedes_remainder_cancel() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1), qty=300))
    out = ex.on_event(bar(1))
    assert [type(e) for e in out] == [Fill, OrderUpdate] and all(e.ts == at(2) for e in out)


def test_zero_cap_entry_cancelled() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))
    assert ex.on_event(bar(1, volume=49)) == (
        OrderUpdate(ts=at(2), intent_id="intent-1", status="cancelled", detail="size cap zero"),)


def test_cost_bps_matches_leg_cost_bps() -> None:
    ex, broker = make_exec()
    ex.on_event(make_quote(ts=at(1), bid=1.00, ask=1.01, ask_size=100))
    ex.on_event(make_order_intent(ts=at(1)))
    f, = ex.on_event(bar(1, open=1.00))
    hs = 0.01 / (2 * 1.005) * 1e4
    want = leg_cost_bps(pd.Series([100 * 1.00]), pd.Series([1.01 * 100]), pd.Series([hs]),
                        pd.Series([0.05]), pd.Series([100 / 1e6])).iloc[0]   # penny frame function
    assert f.cost_bps == pytest.approx(want, rel=1e-12)
    assert broker.source_counts()["cost:quote"] == 1 and broker.source_counts()["cap:quote"] == 1


def test_duplicate_intent_rejected_no_second_fill() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))
    assert ex.on_event(make_order_intent(ts=at(1))) == (
        OrderUpdate(ts=at(1), intent_id="intent-1", status="rejected", detail="duplicate intent_id"),)
    assert [e.qty for e in ex.on_event(bar(1))] == [100]


def test_out_of_order_bar_raises() -> None:
    ex, _ = make_exec()
    ex.on_event(bar(2))
    with pytest.raises(ValueError):
        ex.on_event(bar(2))


def test_missing_cost_inputs_raises() -> None:
    ex, _ = make_exec(inputs={"AAA": INPUTS})
    ex.on_event(make_order_intent(ts=at(1)))              # symbol TEST has no inputs
    with pytest.raises(ValueError, match="no cost inputs for TEST"):
        ex.on_event(bar(1))
