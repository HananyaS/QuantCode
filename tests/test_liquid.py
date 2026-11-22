"""Liquid intraday track (PREREGISTRATION_L1): bracket fills, costs, calendar, the three cells."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.liquid import calendar as cal
from research.liquid import cells
from research.liquid.bracket import Order, simulate
from research.liquid.costs import BASE, DOUBLE, HALF_SPREAD, SEC_RATE, STOP_SLIP, CostSpec
from research.liquid.data import FULL_BARS, Minutes


# ------------------------------------------------------------------ helpers
def bars(n: int = 10, px: float = 100.0):
    o = np.full(n, px); h = np.full(n, px); l = np.full(n, px); c = np.full(n, px)
    return o, h, l, c


def minutes(days: list[str], px: float = 100.0) -> Minutes:
    dates = pd.DatetimeIndex(pd.to_datetime(days))
    d = len(dates)
    nb = np.array([cal.session_bars(x) for x in dates])
    grid = {f: np.full((d, FULL_BARS), px) for f in ("open", "high", "low", "close", "vwap")}
    grid["volume"] = np.full((d, FULL_BARS), 1000.0)
    for i, n in enumerate(nb):
        for f in grid:
            grid[f][i, n:] = np.nan
    prev = np.concatenate([[np.nan], np.full(d - 1, px)])
    return Minutes("TST", dates, nb, grid["open"], grid["high"], grid["low"], grid["close"],
                   grid["volume"], grid["vwap"], prev, np.ones(d, bool))


def set_bar(m: Minutes, i: int, k: int, o=None, h=None, l=None, c=None) -> None:
    for arr, v in ((m.open, o), (m.high, h), (m.low, l), (m.close, c)):
        if v is not None:
            arr[i, k] = v


def refresh_prev(m: Minutes) -> Minutes:
    last = m.close[np.arange(len(m.dates)), m.n_bars - 1]
    prev = np.concatenate([[np.nan], last[:-1]])
    return Minutes(m.symbol, m.dates, m.n_bars, m.open, m.high, m.low, m.close, m.volume,
                   m.vwap, prev, m.valid)


# ------------------------------------------------------------------ calendar
def test_minute_index_and_sessions():
    assert cal.minute_index("09:30") == 0
    assert cal.minute_index("15:59") == 389
    assert cal.session_bars(pd.Timestamp("2019-12-24")) == 210
    assert cal.session_bars(pd.Timestamp("2019-12-23")) == 390


def test_assert_through_refuses_later_dates():
    cal.assert_through([pd.Timestamp("2022-12-30")], cal.DEV_END)
    with pytest.raises(ValueError):
        cal.assert_through([pd.Timestamp("2023-01-03")], cal.DEV_END)


# ------------------------------------------------------------------ bracket
def test_entry_at_open_then_time_exit():
    o, h, l, c = bars()
    c[9] = 101.0
    r = simulate(o, h, l, c, Order(decide=2, limit=100.1, last=9))
    assert (r.filled, r.entry, r.entry_bar, r.exit, r.exit_bar, r.kind) == (True, 100.0, 3, 101.0, 9, "time")


def test_entry_at_limit_inside_the_bar_or_unfilled():
    o, h, l, c = bars()
    o[3], l[3] = 100.5, 99.9
    r = simulate(o, h, l, c, Order(2, 100.1, 9))
    assert r.filled and r.entry == 100.1
    l[3] = 100.1                                   # touches but does not trade through
    assert not simulate(o, h, l, c, Order(2, 100.1, 9)).filled


def test_take_profit_needs_trade_through():
    o, h, l, c = bars()
    h[5] = 102.0
    assert simulate(o, h, l, c, Order(2, 100.1, 9, take_profit=102.0)).kind == "time"
    h[5] = 102.01
    r = simulate(o, h, l, c, Order(2, 100.1, 9, take_profit=102.0))
    assert (r.kind, r.exit, r.exit_bar) == ("tp", 102.0, 5)


def test_stop_gap_fills_at_open_and_tie_goes_to_stop():
    o, h, l, c = bars()
    o[6], l[6] = 98.0, 97.5
    r = simulate(o, h, l, c, Order(2, 100.1, 9, stop=99.0))
    assert (r.kind, r.exit, r.exit_bar) == ("sl", 98.0, 6)
    o, h, l, c = bars()
    h[6], l[6] = 103.0, 98.5
    r = simulate(o, h, l, c, Order(2, 100.1, 9, take_profit=102.0, stop=99.0))
    assert (r.kind, r.exit) == ("sl", 99.0)


def test_entry_bar_below_stop_exits_at_entry():
    o, h, l, c = bars()
    o[3], l[3] = 98.0, 98.0
    r = simulate(o, h, l, c, Order(2, 100.1, 9, stop=99.0))
    assert (r.kind, r.entry, r.exit, r.exit_bar) == ("sl", 98.0, 98.0, 3)


def test_per_bar_stop_takes_effect_on_its_bar():
    o, h, l, c = bars()
    l[5] = 99.4
    stop = np.full(10, np.nan)
    stop[6:] = 99.5                                # set at bar 5's close, live from bar 6
    assert simulate(o, h, l, c, Order(2, 100.1, 9, stop=stop)).kind == "time"
    l[6] = 99.4
    r = simulate(o, h, l, c, Order(2, 100.1, 9, stop=stop))
    assert (r.kind, r.exit_bar, r.exit) == ("sl", 6, 99.5)


def test_short_is_the_mirror():
    o, h, l, c = bars()
    l[5] = 97.9                                    # TP below trades through
    r = simulate(o, h, l, c, Order(2, 99.9, 9, take_profit=98.0, stop=101.0), side=-1)
    assert (r.kind, r.entry, r.exit) == ("tp", 100.0, 98.0)
    o, h, l, c = bars()
    o[6], h[6] = 101.5, 101.6                      # gap up through the buy stop
    r = simulate(o, h, l, c, Order(2, 99.9, 9, take_profit=98.0, stop=101.0), side=-1)
    assert (r.kind, r.exit) == ("sl", 101.5)


def test_no_entry_after_the_last_bar():
    o, h, l, c = bars()
    assert not simulate(o, h, l, c, Order(9, 100.1, 9)).filled


# ------------------------------------------------------------------ costs
def test_costs_by_exit_kind():
    p = 100.0
    assert BASE.entry_frac(p) == pytest.approx((HALF_SPREAD + 3e-6) / p)
    tp, sl, tm = (BASE.exit_frac(p, k) for k in ("tp", "sl", "time"))
    assert tp == pytest.approx((3e-6 + 0.000195) / p + SEC_RATE)
    assert tm - tp == pytest.approx(HALF_SPREAD / p)
    assert sl - tm == pytest.approx(STOP_SLIP / p)
    assert DOUBLE.exit_frac(p, "sl") - tp == pytest.approx(2 * (HALF_SPREAD + STOP_SLIP) / p)
    assert CostSpec(commission=0.0005, spread=False).entry_frac(p) == pytest.approx((0.0005 + 3e-6) / p)


def test_trade_net_long_and_short():
    t = cells.Trade(pd.Timestamp("2020-01-02"), "X", 1, 1.0, 100.0, 101.0, "tp", 1, 5)
    ce, cx = BASE.entry_frac(100.0), BASE.exit_frac(101.0, "tp")
    assert t.gross == pytest.approx(0.01)
    assert t.net(BASE) == pytest.approx((101 * (1 - cx) - 100 * (1 + ce)) / 100)
    s = cells.Trade(pd.Timestamp("2020-01-02"), "X", -1, 1.0, 100.0, 99.0, "tp", 1, 5)
    assert s.gross == pytest.approx(0.01)


# ------------------------------------------------------------------ cells
DAYS = ["2020-01-02", "2020-01-03"]


def test_c1_long_when_first_half_hour_up_and_causal():
    m = minutes(DAYS)
    set_bar(m, 1, 29, c=100.5)                     # 10:00 price above the prior close
    set_bar(m, 1, 389, c=101.0)
    t = cells.c1(m)
    assert len(t) == 1 and (t[0].entry_bar, t[0].exit_bar, t[0].kind, t[0].exit) == (360, 389, "time", 101.0)
    m2 = minutes(DAYS)
    set_bar(m2, 1, 29, c=100.5)
    set_bar(m2, 1, 30, c=50.0)                     # later bars must not change the decision
    set_bar(m2, 1, 389, c=200.0)
    assert len(cells.c1(m2)) == 1
    set_bar(m, 1, 29, c=99.5)
    assert cells.c1(m) == []
    assert len(cells.c1(m, long_short=True)) == 1 and cells.c1(m, long_short=True)[0].side == -1


def test_c1_half_day_uses_the_last_half_hour():
    m = minutes(["2019-12-23", "2019-12-24"])
    set_bar(m, 1, 29, c=100.5)
    t = cells.c1(m)
    assert (t[0].entry_bar, t[0].exit_bar) == (180, 209)


def test_c2_orb_bracket():
    m = minutes(DAYS)
    set_bar(m, 1, 0, o=100.0, l=99.5)
    set_bar(m, 1, 4, c=100.4)
    t = cells.c2(m)
    assert len(t) == 1
    risk = 100.4 - 99.5
    assert t[0].entry_bar == 5 and t[0].weight == pytest.approx(min(0.01 * 100.4 / risk, 4.0))
    m = minutes(DAYS)                              # a doji does not trade, in either mode
    assert cells.c2(m) == [] and cells.c2(m, long_short=True) == []


def test_c2_take_profit_at_ten_r():
    m = minutes(DAYS)
    set_bar(m, 1, 0, o=100.0, l=99.9)
    set_bar(m, 1, 4, c=100.1)                      # R = 0.2, TP = 102.1
    m.open[1, 5:] = 100.1; m.low[1, 5:] = 100.1; m.close[1, 5:] = 100.1; m.high[1, 5:] = 100.1
    set_bar(m, 1, 50, h=102.2)
    t = cells.c2(m)[0]
    assert (t.kind, t.exit) == ("tp", pytest.approx(102.1))


def _c3_setup() -> Minutes:
    days = pd.bdate_range("2020-01-02", periods=16).strftime("%Y-%m-%d").tolist()
    m = minutes(days)
    for i in range(15):                            # past moves of 0.1% / 0.2% from the open
        m.close[i, :] = 100.0 * (1 + 0.001 * (1 + i % 2))
    return refresh_prev(m)


def test_c3_noise_area_reads_only_the_past():
    m = _c3_setup()
    sig, vol = cells.noise_area(m)
    assert np.isnan(sig[13]).all() and np.isfinite(sig[14, 0])
    m.close[15, :] = 150.0                          # today's own bars cannot move today's sigma
    sig2, vol2 = cells.noise_area(m)
    assert np.allclose(sig[15], sig2[15]) and vol[15] == vol2[15]


def test_c3_enters_above_the_boundary_and_stops_at_max_ub_vwap():
    m = _c3_setup()
    i = 15
    m.close[i, 20:] = 100.8; m.open[i, 20:] = 100.8; m.high[i, 20:] = 100.8; m.low[i, 20:] = 100.8
    m.vwap[i, 20:] = 100.8
    set_bar(m, i, 40, l=100.0)                      # falls through the stop after entry
    sig, vol = cells.noise_area(m)
    ub = max(m.open[i, 0], m.prev_close[i]) * (1 + sig[i])
    vw = cells.session_vwap(m, i)
    t = cells.c3(m)
    assert len(t) >= 1
    first = t[0]
    assert first.entry_bar == 30 and first.kind == "sl" and first.exit_bar == 40
    assert first.exit == pytest.approx(min(max(ub[29], vw[29]), m.open[i, 40]))
    assert first.weight == pytest.approx(min(4.0, 0.02 / vol[i]))


def test_c3_needs_price_above_vwap():
    m = _c3_setup()
    i = 15
    m.close[i, 20:] = 100.8; m.open[i, 20:] = 100.8; m.high[i, 20:] = 100.8; m.low[i, 20:] = 100.8
    m.vwap[i, :20] = 200.0                          # session VWAP far above the price
    assert [t for t in cells.c3(m) if t.date == m.dates[i]] == []


def test_daily_sums_trades():
    d = pd.DatetimeIndex(pd.to_datetime(DAYS))
    ts = [cells.Trade(d[1], "X", 1, 2.0, 100.0, 101.0, "time", 1, 5),
          cells.Trade(d[1], "X", 1, 1.0, 100.0, 99.0, "sl", 6, 9)]
    g = cells.daily(ts, d, None)
    assert g.iloc[0] == 0.0 and g.iloc[1] == pytest.approx(2 * 0.01 - 0.01)


# ------------------------------------------------------------------ benchmark
def test_compare_flat_ok_for_intraday():
    from research.penny.eval import benchmark as bm
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    s = pd.DataFrame({"0": np.where(rng.random(300) < 0.5, 0.0, rng.normal(0, 0.01, 300))}, index=idx)
    b = pd.DataFrame({"SPY": rng.normal(0, 0.01, 300), "QQQ": rng.normal(0, 0.01, 300)}, index=idx)
    rf = pd.Series(0.0, index=idx)
    with pytest.raises(ValueError):
        bm.compare(s, b, rf, n_boot=50)
    out = bm.compare(s, b, rf, n_boot=50, flat_ok=True)
    assert out.n_days == 299


def test_infinite_stop_is_a_market_exit_at_the_open():
    o, h, l, c = bars()
    o[6] = 100.3
    stop = np.full(10, np.nan)
    stop[6] = np.inf
    r = simulate(o, h, l, c, Order(2, 100.1, 9, stop=stop))
    assert (r.kind, r.exit_bar, r.exit) == ("sl", 6, 100.3)
    r = simulate(o, h, l, c, Order(2, 99.9, 9, stop=-stop), side=-1)
    assert (r.kind, r.exit_bar, r.exit) == ("sl", 6, 100.3)


def test_c3_checktime_stops_exit_at_the_next_open():
    m = _c3_setup()
    i = 15
    m.close[i, 20:] = 100.8; m.open[i, 20:] = 100.8; m.high[i, 20:] = 100.8; m.low[i, 20:] = 100.8
    m.vwap[i, 20:] = 100.8
    m.close[i, 59] = 100.0                         # below the stop at the 10:30 check only
    t = [x for x in cells.c3(m, resting_stop=False) if x.date == m.dates[i]][0]
    assert (t.kind, t.exit_bar, t.exit) == ("sl", 60, 100.8)
