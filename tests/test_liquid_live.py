"""C3 live runner (research/liquid/PAPER_C3.md): decision parity with research, session scenarios."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from research.liquid import calendar as cal
from research.liquid import cells
from research.liquid.data import FULL_BARS, Minutes
from research.liquid.live import signal
from research.liquid.live.broker import FakeBroker, load_keys
from research.liquid.live.feed import FakeFeed, today_from_rows
from research.liquid.live.ledger import Ledger
from research.liquid.live.runner import Runner, Session

ET = ZoneInfo("America/New_York")
CACHE = Path("data/research_liquid/SPY_rth_20160104_20221230.npz")


# ------------------------------------------------------------- parity (real dev data)
@pytest.mark.skipif(not CACHE.exists(), reason="dev cache not built")
def test_live_plan_and_checks_equal_the_research_cell():
    from research.liquid.data import load
    m = load("SPY", cal.DEV_START, cal.DEV_END)
    sigma, vol = cells.noise_area(m)
    n_days = 0
    for i in range(20, len(m.dates), 25):
        if not m.valid[i] or not np.isfinite(vol[i]):
            continue
        n_days += 1
        n = int(m.n_bars[i])
        plan = signal.day_plan(m.slice(m.dates[i - 20], m.dates[i - 1]), m.dates[i], n)
        np.testing.assert_allclose(plan.sigma[:n], sigma[i, :n], rtol=0, atol=1e-15, equal_nan=True)
        assert plan.vol == pytest.approx(vol[i], rel=1e-12)
        assert plan.prev_close == m.prev_close[i]
        assert plan.weight == pytest.approx(min(4.0, 0.02 / vol[i]))
        ub = max(m.open[i, 0], m.prev_close[i]) * (1 + sigma[i])
        vw = cells.session_vwap(m, i)
        for d in plan.checks:
            r = signal.check(plan, m.open[i, 0], m.close[i], m.volume[i], m.vwap[i], d)
            assert r.enter == bool(m.close[i, d] > ub[d] and m.close[i, d] > vw[d])
            assert r.stop == pytest.approx(max(ub[d], vw[d]), rel=1e-12)
    assert n_days > 60


def test_today_bars_use_nothing_after_the_check():
    rows = pd.DataFrame({"minute_et": ["09:30", "09:31", "09:33", "09:40"],
                         "open": [1, 2, 3, 4.], "high": [1, 2, 3, 4.], "low": [1, 2, 3, 4.],
                         "close": [1, 2, 3, 4.], "volume": [10, 10, 10, 10.], "vwap": [1, 2, 3, 4.]})
    t = today_from_rows(rows, upto=3)
    assert t.complete and t.close[2] == 2.0 and t.volume[2] == 0.0 and t.close[3] == 3.0
    assert np.isnan(t.close[4:]).all()


def test_tick_and_shares():
    assert signal.tick(412.3456) == 412.35 and signal.tick(0.123456) == 0.1235
    assert signal.shares(100_000, 2.5, 400.0, 1e9) == 625
    assert signal.shares(100_000, 2.5, 400.0, 100_000) == 250


# ------------------------------------------------------------- synthetic sessions
def _minutes() -> Minutes:
    days = pd.bdate_range("2020-01-02", periods=16)
    d = len(days)
    nb = np.full(d, FULL_BARS)
    g = {f: np.full((d, FULL_BARS), 100.0) for f in ("open", "high", "low", "close", "vwap")}
    g["volume"] = np.full((d, FULL_BARS), 1000.0)
    for i in range(15):
        for f in ("close", "high", "low", "open", "vwap"):
            g[f][i, :] = 100.0 * (1 + 0.001 * (1 + i % 2))
        g["open"][i, 0] = 100.0
    i = 15
    for f in ("close", "high", "low", "open", "vwap"):
        g[f][i, 20:] = 100.8
    last = g["close"][np.arange(d), nb - 1]
    prev = np.concatenate([[np.nan], last[:-1]])
    return Minutes("SPY", days, nb, g["open"], g["high"], g["low"], g["close"], g["volume"], g["vwap"],
                   prev, np.ones(d, bool))


class ScriptedClock:
    def __init__(self, start: datetime, events=()):
        self.t, self.events = start, sorted(events, key=lambda e: e[0])

    def now(self) -> datetime:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += timedelta(seconds=max(s, 0.001))
        while self.events and self.events[0][0] <= self.t:
            self.events.pop(0)[1]()


def _run(tmp_path, m=None, broker=None, events=(), kill=None):
    m = m if m is not None else _minutes()
    i = len(m.dates) - 1
    day = m.dates[i]
    sess = Session(day, datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET), FULL_BARS)
    plan = signal.day_plan(m.slice(m.dates[0], m.dates[i - 1]), day, FULL_BARS)
    broker = broker or FakeBroker(autofill_limit=100.81, autofill_market=100.7)
    led = Ledger(tmp_path / "led.sqlite")
    clock = ScriptedClock(sess.open - timedelta(minutes=5), events)
    r = Runner(broker, FakeFeed(m), clock, led, kill_file=kill, poll_s=30.0)
    r.run_session(sess, plan)
    return broker, led, plan, sess


def _kinds(b: FakeBroker) -> list[str]:
    return [c[0] for c in b.calls]


def test_session_enters_at_ten_trails_the_stop_and_exits_at_the_close(tmp_path):
    b, led, plan, sess = _run(tmp_path)
    k = _kinds(b)
    assert k[0] == "limit_buy" and k[1] == "stop_sell" and k[-2:] == ["cancel", "market_sell"]
    qty, limit = b.calls[0][1], b.calls[0][2]
    assert limit == round(100.8 * 1.001, 2)
    assert qty == int(np.floor(min(100_000 * plan.weight, 400_000) / 100.8))
    t = led.trades()
    assert len(t) == 1 and t[0]["entry_check"] == 29 and t[0]["exit_kind"] == "time"
    assert t[0]["exit_price"] == 100.7 and b.pos == 0
    assert "replace_stop" in k                                 # VWAP rises toward 100.8, so the stop moves
    assert [e["when"] for e in led.events(kind="reconcile")] == ["start", "end"]


def test_unfilled_entry_is_cancelled_after_one_bar(tmp_path):
    b, led, *_ = _run(tmp_path, broker=FakeBroker(autofill_limit=None, autofill_market=100.7))
    assert led.trades() == []
    assert led.events(kind="entry_unfilled")
    first = b.calls[0]
    assert first[0] == "limit_buy" and b.calls[1] == ("cancel", b.calls[1][1])


def test_stop_fill_between_checks_then_re_entry(tmp_path):
    b = FakeBroker(autofill_limit=100.81, autofill_market=100.7)
    day = _minutes().dates[-1]
    at = datetime(day.year, day.month, day.day, 10, 15, tzinfo=ET)
    b2, led, *_ = _run(tmp_path, broker=b, events=[(at, lambda: b.fill(b.open_stop().id, 100.5))])
    t = led.trades()
    assert t[0]["exit_kind"] == "sl" and t[0]["exit_price"] == 100.5 and t[0]["exit_check"] is None
    assert len(t) == 2 and t[1]["entry_check"] == 59            # re-entered at the 10:30 check


def test_new_stop_above_price_exits_at_market_and_does_not_re_enter_that_check(tmp_path):
    m = _minutes()
    m.close[15, 59] = 100.0                                     # 10:30 price below max(UB, VWAP)
    b, led, *_ = _run(tmp_path, m=m)
    t = led.trades()
    assert t[0]["exit_kind"] == "sl" and t[0]["exit_check"] == 59
    assert all(x["entry_check"] != 59 for x in t)


def test_partial_entry_fill_stops_only_the_filled_shares(tmp_path):
    b = FakeBroker(autofill_limit=100.81, autofill_limit_qty=7, autofill_market=100.7)
    b, led, *_ = _run(tmp_path, broker=b)
    stop = next(c for c in b.calls if c[0] == "stop_sell")
    assert stop[1] == 7 and led.trades()[0]["qty"] == 7


def test_leftover_position_at_the_open_flattens_and_trades_nothing(tmp_path):
    b = FakeBroker(autofill_limit=100.81, autofill_market=100.7)
    b.pos = 50
    b, led, *_ = _run(tmp_path, broker=b)
    assert _kinds(b) == ["flatten"] and b.pos == 0 and led.events(kind="kill")


def test_kill_file_flattens_mid_session(tmp_path):
    kill = tmp_path / "KILL"
    day = _minutes().dates[-1]
    at = datetime(day.year, day.month, day.day, 11, 0, tzinfo=ET)
    b, led, *_ = _run(tmp_path, kill=kill, events=[(at, lambda: kill.write_text("stop"))])
    assert "flatten" in _kinds(b) and b.pos == 0
    assert led.trades()[0]["exit_kind"] == "kill"


# ------------------------------------------------------------- keys
def test_keys_must_be_the_dedicated_pair(tmp_path):
    env = tmp_path / ".env"
    env.write_text("ALPACA_API_KEY_KELLY=abc\nALPACA_SECRET_KEY_KELLY=s\n")
    with pytest.raises(SystemExit):
        load_keys(env)
    env.write_text("ALPACA_API_KEY_KELLY=abc\nALPACA_API_KEY_C3=abc\nALPACA_SECRET_KEY_C3=s\n")
    with pytest.raises(SystemExit):
        load_keys(env)
    env.write_text("ALPACA_API_KEY_KELLY=abc\nALPACA_API_KEY_C3=xyz\nALPACA_SECRET_KEY_C3=s\n")
    assert load_keys(env) == ("xyz", "s")


def test_late_start_skips_past_checks(tmp_path):
    m = _minutes()
    i = len(m.dates) - 1
    day = m.dates[i]
    sess = Session(day, datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET), FULL_BARS)
    plan = signal.day_plan(m.slice(m.dates[0], m.dates[i - 1]), day, FULL_BARS)
    b = FakeBroker(autofill_limit=100.81, autofill_market=100.7)
    led = Ledger(tmp_path / "led.sqlite")
    clock = ScriptedClock(datetime(day.year, day.month, day.day, 11, 10, tzinfo=ET))
    Runner(b, FakeFeed(m), clock, led, poll_s=30.0).run_session(sess, plan)
    missed = [e["d"] for e in led.events(kind="check_missed")]
    assert missed == [29, 59, 89]                       # 10:00, 10:30, 11:00 were already past
    assert led.trades()[0]["entry_check"] == 119        # first live decision is 11:30
