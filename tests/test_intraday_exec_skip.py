"""Exec per-symbol skip (ledger P2-13): with ``skip_missing_cost_inputs`` the broker cancels an
entry whose symbol has no cost inputs instead of raising; the default still raises."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from research.intraday.contracts.events import Fill, OrderUpdate
from research.intraday.contracts.fixtures import T_DEV, make_bar, make_clock_tick, make_order_intent
from research.intraday.exec import CostInputs, ExecConfig, ExecutionAgent, SimBroker, StaticCostInputs

M = timedelta(minutes=1)
SESSION = date(2024, 3, 4)
INPUTS = CostInputs(sigma_daily=0.05, adv_shares=1_000_000.0)


def at(k: int) -> Any:
    return T_DEV + k * M


def bar(k: int, symbol: str = "TEST") -> Any:
    return make_bar(symbol=symbol, ts=at(k + 1), bar_start=at(k))


def make_exec(**cfg: Any) -> tuple[ExecutionAgent, SimBroker]:
    broker = SimBroker(ExecConfig(**cfg), StaticCostInputs({("AAA", SESSION): INPUTS}))
    return ExecutionAgent(broker), broker


def test_default_still_raises() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))
    with pytest.raises(ValueError, match="no cost inputs for TEST"):
        ex.on_event(bar(1))


def test_skip_cancels_without_fill_and_records_reason() -> None:
    ex, broker = make_exec(skip_missing_cost_inputs=True)
    ex.on_event(make_order_intent(ts=at(1)))
    out = ex.on_event(bar(1))
    assert out == (OrderUpdate(ts=at(2), intent_id="intent-1", status="cancelled",
                               detail="no cost inputs for TEST on 2024-03-04"),)
    assert broker.position("TEST") == 0
    assert ex.on_event(bar(2)) == ()                      # order is gone: nothing later fills


def test_skip_is_per_symbol_other_symbols_trade() -> None:
    ex, broker = make_exec(skip_missing_cost_inputs=True)
    ex.on_event(make_order_intent(ts=at(1), intent_id="i-t", symbol="TEST"))
    ex.on_event(make_order_intent(ts=at(1), intent_id="i-a", symbol="AAA"))
    assert [type(e) for e in ex.on_event(bar(1, "TEST"))] == [OrderUpdate]
    (f,) = ex.on_event(bar(1, "AAA"))
    assert isinstance(f, Fill) and f.intent_id == "i-a"
    assert broker.position("AAA") == 100 and broker.position("TEST") == 0


def test_skip_session_survives_to_close_and_is_deterministic() -> None:
    runs = []
    for _ in range(2):
        ex, _ = make_exec(skip_missing_cost_inputs=True)
        ex.on_event(make_order_intent(ts=at(1)))
        runs.append((ex.on_event(bar(1)), ex.on_event(make_clock_tick(ts=at(390)))))
    assert runs[0] == runs[1]


def test_flag_must_be_bool() -> None:
    with pytest.raises(ValueError, match="skip_missing_cost_inputs"):
        ExecConfig(skip_missing_cost_inputs=1)  # type: ignore[arg-type]
