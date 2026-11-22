"""C3 live runner after the G3 review (research/intraday/docs/critiques/G3-C3-paper.md, PAPER_C3 A1)."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from research.liquid import calendar as cal
from research.liquid.data import FIELDS, FULL_BARS, Minutes
from research.liquid.live import replay, report, safety, signal
from research.liquid.live.broker import FakeBroker, OrderState
from research.liquid.live.feed import FakeFeed, merge_today, today_from_rows
from research.liquid.live.ledger import Ledger
from research.liquid.live.runner import Runner, Session

ET = ZoneInfo("America/New_York")
CACHE = Path("data/research_liquid/SPY_rth_20160104_20221230.npz")


# ------------------------------------------------------------------ helpers
def _minutes() -> Minutes:
    """15 quiet sessions, then a session that jumps to 100.8 at 09:50 (C3 enters at 10:00)."""
    days = pd.bdate_range("2020-01-02", periods=16)
    d = len(days)
    nb = np.full(d, FULL_BARS)
    g = {f: np.full((d, FULL_BARS), 100.0) for f in ("open", "high", "low", "close", "vwap")}
    g["volume"] = np.full((d, FULL_BARS), 1000.0)
    for i in range(15):
        for f in ("close", "high", "low", "open", "vwap"):
            g[f][i, :] = 100.0 * (1 + 0.001 * (1 + i % 2))
        g["open"][i, 0] = 100.0
    for f in ("close", "high", "low", "open", "vwap"):
        g[f][15, 20:] = 100.8
    last = g["close"][np.arange(d), nb - 1]
    prev = np.concatenate([[np.nan], last[:-1]])
    return Minutes("SPY", days, nb, g["open"], g["high"], g["low"], g["close"], g["volume"], g["vwap"],
                   prev, np.ones(d, bool))


class Clock:
    def __init__(self, start: datetime, events=()):
        self.t, self.events = start, sorted(events, key=lambda e: e[0])

    def now(self) -> datetime:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += timedelta(seconds=max(s, 0.001))
        while self.events and self.events[0][0] <= self.t:
            self.events.pop(0)[1]()


def _at(m: Minutes, hh: int, mm: int, ss: int = 0) -> datetime:
    day = m.dates[-1]
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=ET)


def _run(tmp_path, broker, m=None, feed=None, events=(), led=None, **kw):
    m = m if m is not None else _minutes()
    i = len(m.dates) - 1
    day = m.dates[i]
    sess = Session(day, _at(m, 9, 30), FULL_BARS)
    plan = signal.day_plan(m.slice(m.dates[0], m.dates[i - 1]), day, FULL_BARS)
    clock = Clock(sess.open - timedelta(minutes=5), events)
    broker.clock = clock
    led = led or Ledger(tmp_path / "led.sqlite")
    Runner(broker, feed or FakeFeed(m), clock, led, poll_s=30.0, **kw).run_session(sess, plan)
    return broker, led


def _fb(**kw) -> FakeBroker:
    return FakeBroker(autofill_limit=kw.pop("autofill_limit", 100.81), autofill_market=100.7, **kw)


# ------------------------------------------------------------------ W1: prices floored to the cent
def test_floor_tick_rounds_down_on_the_cent_grid():
    assert signal.floor_tick(412.3456) == 412.34
    assert signal.floor_tick(412.35) == 412.35 and signal.floor_tick(100.9008) == 100.90
    assert signal.floor_tick(0.123456) == 0.1234
    for p in (10.07, 99.99, 412.29):                   # grid prices stay put despite float error
        assert signal.floor_tick(p) == p


@pytest.mark.skipif(not CACHE.exists(), reason="dev cache not built")
def test_replayed_runner_trades_exactly_as_the_research_cell():
    """The real Runner, filled by the research rules, on every 5th dev session."""
    from research.liquid.data import load
    m = load("SPY", cal.DEV_START, cal.DEV_END)
    rows = replay.valid_rows(m)[::5]
    tr = replay.cells.c3(m)
    n = 0
    for i in rows:
        live, res = replay.replay_day(m, i), replay.research_trades(m, i, tr)
        assert [(e, b) for e, _, b in live] == [(e, b) for e, _, b in res], m.dates[i]
        for (_, kl, b), (_, kr, _) in zip(live, res):
            last = int(m.n_bars[i]) - 1                # live cancels its stop at the last minute's start
            assert kl == kr or (kl, kr, b) == ("time", "sl", last), m.dates[i]
        n += len(live)
    assert len(rows) > 300 and n > 150


# ------------------------------------------------------------------ B3: flatten
def _sim_time():
    t = [0.0]
    return (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + s))


def test_flatten_waits_for_a_pending_cancel_before_closing():
    b = FakeBroker(autofill_market=100.0, cancel_delay=3, reject_oversell=True)
    b.pos = 100
    stop = b.stop_sell("SPY", 100, 99.0, "s")
    b.cancel(stop.id)                                       # pending_cancel still holds the shares
    with pytest.raises(RuntimeError, match="insufficient qty"):
        b.close_position("SPY")                             # the pre-G3 flatten did exactly this
    b2 = FakeBroker(autofill_market=100.0, cancel_delay=3, reject_oversell=True)
    b2.pos = 100
    b2.stop_sell("SPY", 100, 99.0, "s")
    now, sleep = _sim_time()
    closes = safety.flatten(b2, "SPY", now, sleep)
    assert b2.pos == 0 and len(closes) == 1 and closes[0].status == "filled"
    assert not b2.open_orders("SPY")


def test_flatten_gives_up_loudly_when_orders_never_cancel():
    b = FakeBroker(autofill_market=100.0, cancel_delay=10**9)
    b.pos = 5
    b.stop_sell("SPY", 5, 99.0, "s")
    now, sleep = _sim_time()
    with pytest.raises(safety.FlattenError):
        safety.flatten(b, "SPY", now, sleep, timeout_s=10)


# ------------------------------------------------------------------ W5: races
class RaceBroker(FakeBroker):
    """The stop fills just before our cancel arrives; the cancel then fails (Alpaca: 422)."""
    def cancel(self, order_id):
        o = self.orders[order_id]
        if o.kind == "stop" and o.open:
            self.fill(order_id, 100.55)
        return super().cancel(order_id)


def test_a_stop_filling_during_the_close_out_cancel_is_booked_as_the_stop(tmp_path):
    b, led = _run(tmp_path, RaceBroker(autofill_limit=100.81, autofill_market=100.7))
    t = led.trades()
    assert len(t) == 1 and t[0]["exit_kind"] == "sl" and t[0]["exit_price"] == 100.55
    assert not led.events(kind="kill") and led.events(kind="cancel_raced") and b.pos == 0
    assert "market_sell" not in [c[0] for c in b.calls]


class RefuseReplace(FakeBroker):
    def replace_stop(self, order_id, stop, coid):
        self.calls.append(("replace_stop", order_id, stop, coid))
        raise RuntimeError("stop price must be below the market")


def test_a_refused_replace_exits_at_market_instead_of_killing(tmp_path):
    b, led = _run(tmp_path, RefuseReplace(autofill_limit=100.81, autofill_market=100.7))
    t = led.trades()
    assert t[0]["exit_kind"] == "sl" and t[0]["exit_check"] is not None
    assert all(x["entry_check"] != t[0]["exit_check"] for x in t)      # no re-entry at that check
    assert not led.events(kind="kill") and led.events(kind="replace_failed")


def test_a_partly_filled_stop_is_cancelled_and_replaced_for_the_rest(tmp_path):
    b = _fb()
    m = _minutes()
    at = _at(m, 10, 15)
    b, led = _run(tmp_path, b, m=m, events=[(at, lambda: b.fill(b.open_stop().id, 100.5, qty=10))])
    stops = [c for c in b.calls if c[0] == "stop_sell"]
    qty = led.trades()[0]["qty"]
    assert [s[1] for s in stops] == [qty, qty - 10]                    # new stop for the remainder
    first = b.by_coid(stops[0][3])
    assert first.status == "canceled" and first.filled_qty == 10       # cancelled, never patched
    assert first.id not in {c[1] for c in b.calls if c[0] == "replace_stop"}
    t = led.trades()[0]
    assert t["exit_kind"] == "time"
    assert t["exit_price"] == pytest.approx((10 * 100.5 + (qty - 10) * 100.7) / qty)
    assert led.events(kind="stop_partial")[0]["filled"] == 10


# ------------------------------------------------------------------ W4, W8: entry
def test_a_buying_power_rejection_resizes_on_the_limit_price(tmp_path):
    b, led = _run(tmp_path, _fb(enforce_buying_power=True))
    buys = [c for c in b.calls if c[0] == "limit_buy"]
    assert len(buys) == 2 and buys[1][3].endswith("-r")
    assert buys[1][1] == int(np.floor(0.99 * 400_000 / buys[1][2]))
    assert buys[1][1] * buys[1][2] <= 400_000 and led.trades()[0]["qty"] == buys[1][1]


def test_a_partial_entry_is_protected_at_once(tmp_path):
    b, led = _run(tmp_path, _fb(autofill_limit_qty=7))
    i_cancel = next(k for k, c in enumerate(b.calls) if c[0] == "cancel")
    i_stop = next(k for k, c in enumerate(b.calls) if c[0] == "stop_sell")
    assert i_cancel < i_stop and b.calls[i_stop][1] == 7
    e = led.events(kind="entry_filled")[0]
    assert e["partial"] is True


# ------------------------------------------------------------------ W3: decision bar
class LaggyFeed(FakeFeed):
    def __init__(self, m, missing: dict[int, int]):
        super().__init__(m)
        self.missing = dict(missing)

    def today(self, symbol, day, upto):
        t = super().today(symbol, day, upto)
        if self.missing.get(upto, 0) > 0:
            self.missing[upto] -= 1
            return replace(t, has_upto=False)
        return t


def test_a_late_decision_bar_is_waited_for(tmp_path):
    m = _minutes()
    b, led = _run(tmp_path, _fb(), m=m, feed=LaggyFeed(m, {29: 3}))
    assert led.trades()[0]["entry_check"] == 29 and not led.events(kind="check_skipped")


def test_a_decision_bar_that_never_arrives_skips_the_check(tmp_path):
    m = _minutes()
    b, led = _run(tmp_path, _fb(), m=m, feed=LaggyFeed(m, {29: 10**6}))
    sk = led.events(kind="check_skipped")
    assert [e["d"] for e in sk] == [29] and "not published" in sk[0]["reason"]
    assert led.trades()[0]["entry_check"] == 59


def test_today_bars_flag_a_missing_decision_bar_and_count_carries():
    rows = pd.DataFrame({"minute_et": ["09:30", "09:31", "09:33"], **{f: [1.0, 2.0, 3.0] for f in FIELDS}})
    t = today_from_rows(rows, upto=4)
    assert t.complete and not t.has_upto and t.n_carried == 2 and t.close[4] == 3.0
    assert today_from_rows(rows, upto=3).has_upto


def test_hybrid_today_uses_sip_for_old_minutes_and_iex_for_recent():
    lab = [f"{9 + (30 + j) // 60:02d}:{(30 + j) % 60:02d}" for j in range(40)]
    sip = pd.DataFrame({"minute_et": lab, **{f: [100.0] * 40 for f in FIELDS}})
    iex = pd.DataFrame({"minute_et": lab, **{f: [200.0] * 40 for f in FIELDS}})
    t = today_from_rows(merge_today(sip, iex, sip_through=13), upto=29)
    assert (t.close[:14] == 100.0).all() and (t.close[14:30] == 200.0).all() and t.open0 == 100.0


# ------------------------------------------------------------------ B4: kill, halt, crash
class BoomFeed(FakeFeed):
    def today(self, symbol, day, upto):
        if upto == 89:
            raise RuntimeError("feed down")
        return super().today(symbol, day, upto)


def test_an_unexpected_error_flattens_leaves_kill_and_books_the_real_fill(tmp_path):
    kill = tmp_path / "KILL"
    m = _minutes()
    with pytest.raises(RuntimeError, match="feed down"):
        _run(tmp_path, _fb(), m=m, feed=BoomFeed(m), kill_file=kill)
    assert "feed down" in kill.read_text()
    t = Ledger(tmp_path / "led.sqlite").trades()[0]
    assert t["exit_kind"] == "kill" and t["exit_price"] == 100.7 and t["exit_order_id"]


def test_a_leftover_position_leaves_kill(tmp_path):
    kill = tmp_path / "KILL"
    b = _fb()
    b.pos = 50
    _run(tmp_path, b, kill_file=kill)
    assert b.pos == 0 and "reconcile start" in kill.read_text()


def test_an_unclosed_trade_from_an_earlier_day_blocks_the_session(tmp_path):
    led = Ledger(tmp_path / "led.sqlite")
    led.open_trade(day="2019-12-31", entry_check=29, qty=5, decision_price=1.0, limit_price=1.0, entry_price=1.0,
                   entry_ts="x", stop_at_entry=0.9, weight=1.0, equity=1.0)
    b, led = _run(tmp_path, _fb(), led=led)
    assert [c[0] for c in b.calls] == ["flatten"]
    assert "unclosed" in led.events(kind="kill")[0]["reason"]


def test_heartbeat_is_written_through_the_session(tmp_path):
    hb = tmp_path / "hb.json"
    _run(tmp_path, _fb(), heartbeat=hb)
    ts, state = safety.read_heartbeat(hb)
    assert state == "done" and ts is not None


def test_guard_acts_only_when_the_runner_is_gone_or_the_close_out_failed():
    o = datetime(2026, 11, 23, 9, 30, tzinfo=ET)
    close = o + timedelta(minutes=389)
    t = o + timedelta(hours=2)
    fresh, stale = t - timedelta(seconds=5), t - timedelta(seconds=200)
    assert safety.guard_action(t, fresh, 100, 1, o, close) is None
    assert "stale" in safety.guard_action(t, stale, 100, 1, o, close)
    assert "stale" in safety.guard_action(t, None, 100, 0, o, close)
    assert safety.guard_action(t, stale, 0, 0, o, close) is None                 # nothing to protect
    assert safety.guard_action(o - timedelta(minutes=5), None, 100, 1, o, close) is None   # before the open
    late = close + timedelta(seconds=50)
    assert "after the close-out" in safety.guard_action(late, late, 10, 0, o, close)


def test_a_second_instance_is_refused(tmp_path):
    p = tmp_path / "run.lock"
    with safety.single_instance(p):
        with pytest.raises(SystemExit, match="second instance"):
            with safety.single_instance(p):
                pass
    with safety.single_instance(p):                     # released on exit
        pass


# ------------------------------------------------------------------ B1: cost gate
def _day_minutes(day="2026-11-23") -> Minutes:
    g = {f: np.full((1, FULL_BARS), 100.0) for f in FIELDS}
    g["close"][0, FULL_BARS - 1] = 101.0
    g["open"][0, 90] = 99.60
    return Minutes("SPY", pd.DatetimeIndex([day]), np.array([FULL_BARS]), *(g[f] for f in FIELDS),
                   np.array([100.0]), np.array([True]))


def _trade(**kw) -> dict:
    t = {"id": 1, "day": "2026-11-23", "entry_check": 29, "qty": 100, "entry_price": 100.02,
         "exit_price": 100.97, "exit_kind": "time", "exit_check": None, "stop_at_exit": None,
         "exit_filled_at": None, "exit_ts": None}
    return t | kw


def test_trip_costs_split_execution_latency_and_fees():
    m = _day_minutes()
    r = report.trip_costs(_trade(), m, 0, mid_entry=100.01, mid_exit=100.98)
    fee = (0.000003 * 200 + 20.6e-6 * 100 * 100.97) / (100 * 100.02) * 1e4
    assert r["exec_entry_bp"] == pytest.approx(1.0) and r["exec_exit_bp"] == pytest.approx(1.0)
    assert r["latency_bp"] == pytest.approx(1.0 + 2.0) and r["fees_bp"] == pytest.approx(fee)
    assert r["cost_bp"] == pytest.approx(2.0 + fee)
    s = report.trip_costs(_trade(exit_kind="sl", exit_price=99.48, stop_at_exit=99.50,
                                 exit_filled_at="2026-11-23T16:00:20+00:00"), m, 0, 100.01, np.nan)
    assert s["usable"] and s["resting_stop"] and s["exec_exit_bp"] == pytest.approx(2.0)
    assert s["latency_bp"] == pytest.approx(1.0)
    assert not report.trip_costs(_trade(exit_kind="kill"), m, 0, 100.01, 100.98)["usable"]
    assert not report.trip_costs(_trade(), m, 0, np.nan, 100.98)["usable"]


def _rows(n_days: int, per_day: int, cost: float, latency: float = 0.0) -> list[dict]:
    return [{"day": f"2027-{1 + d // 28:02d}-{1 + d % 28:02d}", "trade_id": d * 10 + k, "usable": True,
             "cost_bp": cost, "latency_bp": latency, "model_bp": 0.5}
            for d in range(n_days) for k in range(per_day)]


def test_g_cost_passes_fails_and_extends_by_the_registered_rule():
    assert report.g_cost(_rows(5, 2, 0.5))["status"] == "pending"
    assert report.g_cost(_rows(10, 2, 3.0))["status"] == "FAIL"                  # interim 99% look
    assert report.g_cost(_rows(10, 2, 0.5))["status"] == "pending"               # interim never passes
    g = report.g_cost(_rows(20, 2, 0.5))
    assert g["status"] == "PASS" and len(g["decision_days"]) == 20
    assert report.g_cost(_rows(20, 2, 2.0))["status"] == "FAIL"
    assert report.g_cost(_rows(20, 2, 1.2, latency=0.8))["status"] == "extended to 80"   # bar = 1.66 - 0.8
    assert report.g_cost(_rows(40, 2, 1.2, latency=0.8))["status"] == "INCONCLUSIVE"
    # the decision is taken once, on the first 40: later trips do not move it
    later = _rows(20, 2, 0.5) + [r | {"day": "2028-01-01", "trade_id": 9999 + k, "cost_bp": 50.0}
                                 for k, r in enumerate(_rows(1, 5, 0))]
    assert report.g_cost(later)["status"] == "PASS"


def test_g_fidelity_counts_missed_research_trades():
    e = {"2027-01-04": {"live": [29], "research": [29, 89]}, "2027-01-05": {"live": [], "research": [149]}}
    g = report.g_fidelity(e, ["2027-01-04", "2027-01-05"])
    assert g["window"]["jaccard"] == pytest.approx(1 / 3) and g["status"] == "FAIL"
    assert report.g_fidelity(e, None)["status"] == "pending"


def test_day_safety_flags_unprotected_time_unknown_fills_and_kills():
    o = lambda i, coid, q: OrderState(i, coid, "buy", "limit", q, "filled", q, 100.0)   # noqa: E731
    trades = [{"id": 1, "entry_filled_at": "2026-11-23T15:00:06+00:00", "entry_ts": None,
               "exit_filled_at": "2026-11-23T20:59:01+00:00", "exit_ts": None, "exit_kind": "time"}]
    ev = [{"kind": "entry_filled", "order_id": "a", "ts": "2026-11-23T10:00:07-05:00"},
          {"kind": "stop_placed", "trade_id": 1, "ts": "2026-11-23T10:00:08-05:00"},
          {"kind": "exit", "order_id": "b", "ts": "2026-11-23T15:59:02-05:00"},
          {"kind": "reconcile", "when": "end", "position": 0, "open_orders": 0, "ts": "x"}]
    closed = [o("a", "c3-2026-11-23-29-entry", 5), o("b", "c3-x", 5), o("z", "c3-drill-1", 1)]
    s = report.day_safety("2026-11-23", ev, trades, closed)
    assert s["ok"] and s["max_unprotected_s"] == pytest.approx(2.0)
    late = [e | {"ts": "2026-11-23T10:01:30-05:00"} if e["kind"] == "stop_placed" else e for e in ev]
    assert not report.day_safety("2026-11-23", late, trades, closed)["ok"]
    extra = closed + [o("c", "manual", 3)]
    s2 = report.day_safety("2026-11-23", ev, trades, extra)
    assert not s2["ok"] and s2["broker_fills_not_in_ledger"] == ["c"]
    s3 = report.day_safety("2026-11-23", ev + [{"kind": "kill", "reason": "x", "ts": "x"}], trades, closed)
    assert not s3["ok"]


def test_evening_report_on_a_full_session(tmp_path):
    m = _minutes()
    b, led = _run(tmp_path, _fb())
    day = m.dates[-1]
    d = str(day.date())
    live = led.trades(d)
    i = len(m.dates) - 1
    mids = {t["id"]: (float(m.open[i, t["entry_check"] + 1]), float(m.close[i, FULL_BARS - 1])) for t in live}
    rep = report.day_report(day, m, live, led.events(d), b.closed_orders("SPY"), mids)
    assert rep["entries"]["live"] == rep["entries"]["research"] == [29]
    assert rep["safety"]["ok"], rep["safety"]
    assert rep["costs"][0]["usable"] and rep["costs"][0]["latency_bp"] == pytest.approx(0.0)
    s = report.summary([rep])
    assert s["g_cost"]["status"] == "pending" and s["g_safety"]["status"] == "pass"
    assert s["g_fidelity"]["running"]["jaccard"] == 1.0


def test_fake_broker_defaults_are_unchanged():
    b = FakeBroker()
    o = b.stop_sell("SPY", 5, 99.0, "s")
    b.cancel(o.id)
    assert b.get(o.id).status == "canceled"
    with pytest.raises(RuntimeError):
        b.cancel(o.id)                                    # cancelling a closed order fails, as at Alpaca
