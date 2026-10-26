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


# ---- Task 6: exits ----------------------------------------------------------
def enter(ex: ExecutionAgent, symbol: str = "TEST", intent_id: str = "intent-1", **bracket: Any) -> Fill:
    ex.on_event(make_order_intent(ts=at(1), symbol=symbol, intent_id=intent_id, **bracket))
    (f,) = ex.on_event(bar(1, symbol=symbol, open=1.00, high=1.01, low=0.99, close=1.00))
    assert isinstance(f, Fill) and f.side == "buy" and f.qty == 100
    return f


def test_stop_triggered_on_entry_fill_bar() -> None:
    ex, broker = make_exec()
    ex.on_event(make_order_intent(ts=at(1), stop_loss=0.95, take_profit=1.20))
    (f,) = ex.on_event(bar(1, open=1.00, high=1.01, low=0.93, close=0.96))   # buy only, no same-bar sell
    assert f.side == "buy" and broker.exit_log() == (("intent-1", "stop"),)
    (s,) = ex.on_event(bar(2, open=0.96, high=0.97, low=0.95, close=0.96))
    assert (s.side, s.price, s.ts) == ("sell", 0.96, at(3))


def test_zero_cap_exit_retries_next_bar() -> None:
    ex, _ = make_exec(); enter(ex, time_stop=at(5))
    ex.on_event(make_clock_tick(ts=at(5)))
    assert ex.on_event(bar(5, volume=0)) == ()               # cap 0: no fill, no update
    (f,) = ex.on_event(bar(6))
    assert (f.side, f.qty) == ("sell", 100)


def test_stop_fills_at_next_bar_open_including_gap_down() -> None:
    ex, _ = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20)
    assert ex.on_event(bar(2, open=0.99, high=1.00, low=0.94, close=0.95)) == ()   # trigger only
    (f,) = ex.on_event(bar(3, open=0.80, high=0.82, low=0.78, close=0.81))
    assert (f.ts, f.side, f.qty, f.price, f.intent_id) == (at(4), "sell", 100, 0.80, "intent-1")


def test_stop_never_fills_at_stop_level() -> None:
    ex, _ = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20)
    ex.on_event(bar(2, open=0.99, high=1.00, low=0.90, close=0.96))
    (f,) = ex.on_event(bar(3, open=0.97, high=0.98, low=0.94, close=0.95))
    assert f.price == 0.97 and f.price != 0.95


def test_take_profit_fills_at_next_bar_open() -> None:
    ex, broker = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20)
    ex.on_event(bar(2, open=1.10, high=1.21, low=1.09, close=1.19))
    (f,) = ex.on_event(bar(3, open=1.18, high=1.19, low=1.15, close=1.16))
    assert f.price == 1.18 and broker.exit_log() == (("intent-1", "target"),)


def test_stop_wins_when_bar_hits_both() -> None:
    ex, broker = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20)
    ex.on_event(bar(2, open=1.00, high=1.25, low=0.90, close=1.00))
    assert broker.exit_log() == (("intent-1", "stop"),)


def test_time_stop_exit_at_bar_starting_at_tick() -> None:
    ex, _ = make_exec(); enter(ex, time_stop=at(10))
    assert ex.on_event(make_clock_tick(ts=at(10))) == ()
    assert ex.on_event(bar(9)) == ()                       # bar_start at(9) < decision at(10)
    (f,) = ex.on_event(bar(10, open=1.04, high=1.05, close=1.04))
    assert (f.ts, f.price, f.side) == (at(11), 1.04, "sell")


def test_exit_remainder_stays_pending() -> None:
    ex, broker = make_exec(); enter(ex, time_stop=at(5))
    ex.on_event(make_clock_tick(ts=at(5)))
    (f1,) = ex.on_event(bar(5, volume=2_500))              # sell cap 50
    (f2,) = ex.on_event(bar(6))
    assert (f1.qty, f2.qty, broker.position("TEST")) == (50, 50, 0)


def test_oversized_sell_rejected() -> None:
    ex, _ = make_exec(); enter(ex)
    assert ex.on_event(make_order_intent(ts=at(3), intent_id="s-1", side="sell", qty=150))[0].detail == "sell exceeds position"
    assert ex.on_event(make_order_intent(ts=at(3), intent_id="s-2", side="sell", qty=60))[0].status == "accepted"
    (f,) = ex.on_event(bar(3))
    assert (f.intent_id, f.side, f.qty) == ("s-2", "sell", 60)


def test_fill_at_0934_open_does_not_use_summary() -> None:
    ci = CostInputs(sigma_daily=0.05, adv_shares=1e6, half_spread_bps=50.0, ask_notional=5e3, bid_notional=5e3)
    ex, broker = make_exec(inputs={"TEST": ci})
    ex.on_event(make_order_intent(ts=at(3)))
    (f,) = ex.on_event(bar(3, open=1.00))                   # bar_start 09:34, Fill.ts 09:35
    assert f.cost_bps == pytest.approx(leg_cost_bps_scalar(100.0, None, 100.0, 0.05, 1e-4, 1.0, 1.0))
    assert broker.source_counts()["cost:default"] == 1


def test_0934_summary_not_used_before_0935() -> None:
    ci = CostInputs(sigma_daily=0.05, adv_shares=1e6, half_spread_bps=50.0, ask_notional=5e3, bid_notional=5e3)
    ex, broker = make_exec(inputs={"TEST": ci})
    ex.on_event(make_order_intent(ts=at(2)))
    ex.on_event(bar(2))                                     # bar_start 09:33: default
    ex.on_event(make_order_intent(ts=at(4), intent_id="intent-2"))
    (f,) = ex.on_event(bar(4, open=1.00))                   # bar_start 09:35: summary
    assert f.cost_bps == pytest.approx(leg_cost_bps_scalar(100.0, 5e3, 50.0, 0.05, 1e-4, 1.0, 1.0))
    assert (broker.source_counts()["cost:default"], broker.source_counts()["cost:summary"]) == (1, 1)


# ---- Task 7: halts, kill, end of session ------------------------------------
def test_no_fill_while_halted() -> None:
    ex, _ = make_exec(); enter(ex, time_stop=at(5))
    ex.on_event(make_halt(ts=at(4), halted=True))
    ex.on_event(make_clock_tick(ts=at(5)))
    assert ex.on_event(bar(5)) == () and ex.on_event(bar(6)) == ()


def test_entry_cancelled_on_halt() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))
    assert ex.on_event(make_halt(ts=at(1), halted=True)) == (
        OrderUpdate(ts=at(1), intent_id="intent-1", status="cancelled", detail="halt"),)
    assert ex.on_event(make_order_intent(ts=at(2), intent_id="intent-2"))[0].detail == "broker refusal: symbol halted"


def test_exit_fills_after_resume_at_first_bar_open() -> None:
    ex, _ = make_exec(); enter(ex, time_stop=at(3))
    ex.on_event(make_clock_tick(ts=at(3)))                   # exit decided at(3)
    ex.on_event(make_halt(ts=at(3), halted=True))
    ex.on_event(make_halt(ts=at(9), halted=False, reason=""))
    assert ex.on_event(bar(8)) == ()                          # bar_start at(8) < resume at(9)
    (f,) = ex.on_event(bar(9, open=0.70, high=0.75, low=0.69, close=0.72))
    assert (f.ts, f.price) == (at(10), 0.70)


def test_kill_cancels_entries_and_exits_positions() -> None:
    ex, _ = make_exec()
    enter(ex, symbol="AAA")                                   # open lot in AAA
    ex.on_event(make_order_intent(ts=at(2), intent_id="intent-2"))   # working buy in TEST
    assert ex.on_event(make_kill(ts=at(2))) == (
        OrderUpdate(ts=at(2), intent_id="intent-2", status="cancelled", detail="kill"),)
    (f,) = ex.on_event(bar(2, symbol="AAA", open=0.98, high=0.99, low=0.97, close=0.98))
    assert (f.symbol, f.side, f.qty, f.price) == ("AAA", "sell", 100, 0.98)


def test_intent_after_kill_rejected() -> None:
    ex, _ = make_exec()
    ex.on_event(make_kill(ts=at(1)))
    assert ex.on_event(make_order_intent(ts=at(1))) == (
        OrderUpdate(ts=at(1), intent_id="intent-1", status="rejected", detail="after kill"),)


def test_end_of_session_closes_open_positions() -> None:
    ex, _ = make_exec()
    enter(ex, symbol="AAA"); enter(ex, symbol="TEST", intent_id="intent-2")
    ex.on_event(bar(2, symbol="TEST", close=1.03, high=1.05))
    out = ex.on_event(make_clock_tick(ts=at(390)))            # 16:01 ET
    assert [(e.symbol, e.qty, e.price, e.ts) for e in out] == [("AAA", 100, 1.00, at(390)), ("TEST", 100, 1.03, at(390))]
    assert ex.on_event(make_order_intent(ts=at(390), intent_id="late"))[0].detail == "broker refusal: session closed"


def test_halted_position_expires_at_session_end() -> None:
    ex, broker = make_exec(); enter(ex)
    ex.on_event(make_halt(ts=at(300), halted=True))
    assert ex.on_event(make_clock_tick(ts=at(390))) == (OrderUpdate(
        ts=at(390), intent_id="intent-1", status="expired", detail="exit unfilled: halted at session end"),)
    assert broker.position("TEST") == 100


def test_one_exit_per_lot_when_triggers_overlap() -> None:
    ex, _ = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20, time_stop=at(3))
    ex.on_event(bar(2, open=0.99, high=1.00, low=0.90, close=0.96))   # stop decided at(3)
    ex.on_event(make_clock_tick(ts=at(3))); ex.on_event(make_kill(ts=at(3)))
    sells = [e for k in (3, 4, 5) for e in ex.on_event(bar(k)) if isinstance(e, Fill)]
    assert sum(f.qty for f in sells) == 100 and len(sells) == 1


# ---- fix round 1 (controller review M2) ---------------------------------------
def test_sell_reduces_lots_without_pending_exit_first() -> None:
    ex, broker = make_exec()
    ex.on_event(make_order_intent(ts=at(1), intent_id="intent-1", time_stop=at(3)))
    ex.on_event(make_order_intent(ts=at(1), intent_id="intent-2"))
    assert [e.intent_id for e in ex.on_event(bar(1))] == ["intent-1", "intent-2"]
    assert ex.on_event(make_order_intent(ts=at(2), intent_id="s-1", side="sell", qty=100))[0].status == "accepted"
    ex.on_event(make_clock_tick(ts=at(3)))                   # intent-1's time exit, not before at(3)
    (s,) = ex.on_event(bar(2))                               # the sell fills; the exit cannot yet
    assert (s.intent_id, s.qty) == ("s-1", 100)
    (x,) = ex.on_event(bar(3))                               # intent-1 keeps its lot and its exit
    assert (x.intent_id, x.side, x.qty) == ("intent-1", "sell", 100) and broker.position("TEST") == 0


# ---- UP-2: refit cost book through the broker ---------------------------------
REFIT_CI = CostInputs(sigma_daily=0.05, adv_shares=1e6, hs_open_bps=80.0, hs_close_bps=40.0,
                      ask_notional_open=1_000.0, ask_notional_close=3_000.0,
                      bid_notional_open=2_000.0, bid_notional_close=600.0)


def test_refit_book_cost_through_broker() -> None:
    ex, broker = make_exec(inputs={"TEST": REFIT_CI}, cost_book="refit")
    ex.on_event(make_order_intent(ts=at(4)))
    (f,) = ex.on_event(bar(4, open=1.00))                   # bar_start 09:35: the open end of the book
    assert f.cost_bps == pytest.approx(leg_cost_bps_scalar(100.0, 1_000.0, 80.0, 0.05, 1e-4, 1.0, 1.0))
    assert broker.source_counts()["cost:refit"] == 1


def test_no_refit_book_for_symbol_day_raises() -> None:
    ex, _ = make_exec(cost_book="refit")                    # INPUTS carry no refit book
    ex.on_event(make_order_intent(ts=at(4)))
    with pytest.raises(ValueError, match="no refit cost book for TEST"):
        ex.on_event(bar(4))
