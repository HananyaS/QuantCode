"""The C3 dashboard's data layer: the Drive payload, check labels and the live tables (no network)."""
import math
from datetime import datetime

import pandas as pd
import pytest

from research.liquid.live import dashboard as D
from research.liquid.live import report

ET = D.ET


def test_wrap_round_trip_survives_markdown_escaping() -> None:
    doc = {"status": {"columns": ["a_b"], "rows": [{"a_b": 1.5}]}, "list": [1, None, "x+y"]}
    text = D.wrap(doc)
    escaped = "".join("\\" + ch if ch in "{}[]_+*#-.!()" else ch for ch in text)   # what the text reader may do
    assert D.unwrap(text) == doc
    assert D.unwrap(escaped) == doc


def test_wrap_refuses_nan() -> None:
    with pytest.raises(ValueError):
        D.wrap({"x": float("nan")})


@pytest.mark.parametrize("bar, want", [(0, "09:30"), (29, "09:59"), (389, "15:59"), (209, "12:59")])
def test_hhmm(bar: int, want: str) -> None:
    assert D.hhmm(bar) == want


def test_check_time_is_the_close_of_the_decision_bar() -> None:
    assert D.check_time(29) == "10:00"
    assert [D.check_time(d) for d in D.cells.C3_CHECKS][-1] == "15:30"


def test_num_is_json_safe() -> None:
    assert D.num(float("nan")) is None and D.num(float("inf")) is None and D.num(None) is None
    assert D.num(1.23456, 2) == 1.23 and D.num("2.5") == 2.5 and D.num("x") is None


def test_table_keeps_column_order_and_rejects_unknown_columns() -> None:
    t = D.table([{"b": 1, "a": 2}], ["a", "b"])
    assert t["columns"] == ["a", "b"] and list(t["rows"][0]) == ["a", "b"]
    with pytest.raises(AssertionError):
        D.table([{"c": 1}], ["a"])


def ev(kind: str, **payload) -> dict:
    return {"kind": kind, "payload": payload}


@pytest.mark.parametrize("payload, events, want", [
    ({"d": 59, "price": 101.0, "ub": 100.0, "vwap": 99.0, "enter": True, "in_position": False},
     [ev("entry_filled", d=59)], "enter"),
    ({"d": 59, "price": 101.0, "ub": 100.0, "vwap": 99.0, "enter": True, "in_position": False},
     [ev("entry_unfilled", d=59)], "enter_unfilled"),
    ({"d": 89, "price": 101.0, "ub": 100.0, "vwap": 99.0, "enter": True, "in_position": True},
     [ev("stop_replaced", d=89)], "hold"),
    ({"d": 89, "price": 99.0, "ub": 100.0, "vwap": 99.5, "enter": False, "in_position": True},
     [ev("market_exit_sent", d=89)], "exit"),
    ({"d": 29, "price": 99.0, "ub": 100.0, "vwap": 98.0, "enter": False, "in_position": False}, [], "inside"),
    ({"d": 29, "price": 101.0, "ub": 100.0, "vwap": 102.0, "enter": False, "in_position": False}, [], "below_vwap"),
])
def test_check_action(payload: dict, events: list[dict], want: str) -> None:
    assert D.check_action(payload, events) == want


def one_day() -> tuple[list[str], dict, dict, dict, dict]:
    day = "2026-11-24"
    trade = {"id": 1, "day": day, "entry_check": 59, "qty": 100, "decision_price": 600.0, "limit_price": 600.6,
             "entry_price": 600.1, "entry_ts": "2026-11-24T10:30:05-05:00", "stop_at_entry": 599.5,
             "exit_price": 603.0, "exit_ts": "2026-11-24T15:59:02-05:00", "exit_kind": "time", "exit_check": None,
             "stop_at_exit": 601.0, "weight": 1.2, "equity": 100_000.0, "entry_filled_at": "2026-11-24T15:30:06Z",
             "exit_filled_at": "2026-11-24T20:59:03Z"}
    events = [ev("plan", weight=1.2, vol=0.0166),
              ev("check", d=29, price=598.0, ub=599.0, vwap=597.0, enter=False, stop=None, in_position=False),
              ev("check", d=59, price=600.0, ub=599.2, vwap=598.0, enter=True, stop=599.2, in_position=False),
              ev("entry_filled", d=59, price=600.1, qty=100),
              ev("check", d=89, price=601.0, ub=599.4, vwap=599.0, enter=True, stop=599.4, in_position=True),
              ev("stop_replaced", d=89, level=599.4)]
    rep = {"costs": [{"trade_id": 1, "usable": True, "cost_bp": 1.1, "exec_entry_bp": 0.4, "exec_exit_bp": 0.5,
                      "fees_bp": 0.2, "latency_bp": 0.3, "model_bp": 0.9}],
           "entries": {"live": [59], "research": [59, 149]},
           "research_trades": [{"entry_check": 59, "kind": "time", "exit_bar": 389, "entry": 600.0, "exit": 603.0},
                               {"entry_check": 149, "kind": "sl", "exit_bar": 160, "entry": 602.0, "exit": 601.5}],
           "safety": {"ok": True, "max_unprotected_s": 2.0, "kills": []}}
    market = {"spy_ret": 0.004, "qqq_ret": 0.006, "arrays": {}}
    return [day], {day: [trade]}, {day: events}, {day: rep}, {day: market}


def test_live_tables_assemble_one_day() -> None:
    days, trades, events, reps, market = one_day()
    t = D.live_tables(days, trades, events, reps, market)
    (tr,), (dy,) = t["trades"], t["days"]
    fees = report.fees_owed(pd.Timestamp(days[0]), 100, 600.1, 603.0)
    assert tr["pnl"] == pytest.approx(100 * (603.0 - 600.1) - fees, abs=0.01)
    assert tr["check"] == "10:30" and tr["research_match"] is True and tr["cost_bp"] == 1.1
    assert tr["ub_at_check"] == 599.2 and tr["vwap_at_check"] == 598.0
    assert tr["entry_time"] == "10:30:06" and tr["exit_time"] == "15:59:03"
    assert dy["ret"] == pytest.approx(tr["pnl"] / 100_000.0, abs=1e-7)
    assert dy["matched"] == 1 and dy["jaccard"] == pytest.approx(0.5)
    assert dy["research_trades"] == 2 and dy["weight"] == 1.2
    want = 1.2 * (D.cells.Trade(pd.Timestamp(days[0]), "C3", 1, 1.2, 600.0, 603.0, "time", 0, 0).net(D.costs.BASE)
                  + D.cells.Trade(pd.Timestamp(days[0]), "C3", 1, 1.2, 602.0, 601.5, "sl", 0, 0).net(D.costs.BASE))
    assert dy["research_ret"] == pytest.approx(want, abs=1e-7)
    assert [c["action"] for c in t["checks"]] == ["inside", "enter", "hold"]
    assert [c["research_enters"] for c in t["checks"]] == [False, True, False]


def test_status_row_before_the_first_session() -> None:
    empty = {"days": [], "trades": [], "checks": []}
    s = D.status_row(empty, {}, now=datetime(2026, 11, 23, 19, 0, tzinfo=ET), kill_text=None,
                     next_session={"day": "2026-11-24", "open": "2026-11-24T09:30:00-05:00"}, equity=100_000.0,
                     code={"sha": "abc1234def", "dirty": False}, host="h", timer_next=None)
    assert s["phase"] == "waiting" and s["sessions"] == 0 and s["round_trips"] == 0
    assert s["ret_total"] is None and s["g_cost"] == "pending" and s["next_session"] == "2026-11-24"
    assert s["code_sha"] == "abc1234" and s["breakeven_bp"] == report.BREAKEVEN_BP
    assert D.wrap({"status": D.table([s])})                          # JSON-safe


def test_status_row_after_a_day_and_a_kill() -> None:
    days, trades, events, reps, market = one_day()
    t = D.live_tables(days, trades, events, reps, market)
    s = D.status_row(t, {"g_cost": {"status": "pending", "usable_round_trips": 1,
                                    "running": {"cost_mean_bp": 1.1, "cost_ci95": [0.5, 1.7]}}},
                     now=datetime(2026, 11, 24, 16, 30, tzinfo=ET), kill_text="KILL: test", next_session=None,
                     equity=None, code={}, host="h", timer_next=None)
    assert s["phase"] == "stopped" and s["kill"] is True and s["sessions"] == 1 and s["round_trips"] == 1
    assert s["ret_total"] == pytest.approx(t["days"][0]["ret"], abs=1e-6)
    assert s["spy_total"] == pytest.approx(0.004) and s["cost_lo_bp"] == 0.5 and s["usable_trips"] == 1
    assert not math.isnan(s["win_rate"])
