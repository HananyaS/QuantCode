from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from research.intraday.contracts.codec import to_record
from research.intraday.contracts.events import Bar, Event, OrderIntent, Signal
from research.intraday.contracts.fixtures import T_DEV, make_bar, make_quote, make_signal
from research.intraday.risk.agent import RiskAgent
from research.intraday.risk.config import RiskConfig
from research.intraday.risk.session import ET


def test_last_entry_must_precede_flatten() -> None:
    with pytest.raises(ValueError, match="last_entry_et"):
        RiskConfig(last_entry_et=time(15, 55), flatten_et=time(15, 55))
    with pytest.raises(ValueError, match="last_entry_et"):
        RiskConfig(last_entry_et=time(15, 56), flatten_et=time(15, 55))
    RiskConfig(last_entry_et=time(15, 54), flatten_et=time(15, 55))  # ok


D = date(2024, 3, 4)
def et(h: int, m: int, d: date = D) -> datetime:
    return datetime.combine(d, time(h, m), ET).astimezone(timezone.utc)
assert et(9, 31) == T_DEV
def bar(sym: str, ts: datetime, close: float = 1.02, volume: int = 10_000,
        high: float | None = None, low: float | None = None, open_: float | None = None) -> Bar:
    o = close if open_ is None else open_
    return make_bar(symbol=sym, ts=ts, bar_start=ts - timedelta(minutes=1), open=o,
                    high=max(o, close) if high is None else high, low=min(o, close) if low is None else low,
                    close=close, vwap=close, volume=volume)
def sig(sym: str, ts: datetime, score: float = 0.7, horizon: int = 15) -> Signal:
    return make_signal(symbol=sym, ts=ts, score=score, horizon_min=horizon)
def run(agent: RiskAgent, *events: Event) -> list[Event]:
    return [e for ev in events for e in agent.on_event(ev)]

def test_only_buy_intents_with_signal_ts() -> None:
    out = run(RiskAgent(D), bar("AAA", T_DEV), sig("AAA", T_DEV))
    assert len(out) == 1 and isinstance(out[0], OrderIntent)
    oi = out[0]
    assert (oi.ts, oi.side, oi.qty, oi.limit_price, oi.take_profit, oi.stop_loss) == (T_DEV, "buy", 200, None, None, None)
    assert oi.time_stop == et(9, 46)
    assert oi.reason == "entry score=0.7 thr=0.5 cap=bar"

def test_intent_ids_unique_and_deterministic() -> None:
    evs = (bar("AAA", T_DEV), bar("BBB", T_DEV), sig("AAA", T_DEV), sig("BBB", T_DEV))
    a, b = run(RiskAgent(D), *evs), run(RiskAgent(D), *evs)
    assert [o.intent_id for o in a] == ["20240304-AAA-1", "20240304-BBB-2"]
    assert [to_record(o) for o in a] == [to_record(o) for o in b]

def test_below_threshold_signal_skipped() -> None:
    agent = RiskAgent(D)
    assert run(agent, bar("AAA", T_DEV), sig("AAA", T_DEV, score=0.49)) == []
    assert agent.skip_counts == {"below_threshold": 1}
    assert len(run(agent, sig("AAA", T_DEV, score=0.5))) == 1     # >= threshold enters

def test_time_stop_capped_at_flatten_time() -> None:
    cfg = RiskConfig(last_entry_et=time(15, 50), flatten_et=time(15, 55))
    (oi,) = run(RiskAgent(D, cfg), bar("AAA", et(15, 45)), sig("AAA", et(15, 45), horizon=15))
    assert oi.time_stop == et(15, 55)
    (oi,) = run(RiskAgent(D, RiskConfig(hold_min=5)), bar("AAA", T_DEV), sig("AAA", T_DEV, horizon=15))
    assert oi.time_stop == et(9, 36)

def test_no_entry_at_or_after_last_entry_time() -> None:
    agent = RiskAgent(D)
    out = run(agent, bar("AAA", et(15, 29)), sig("AAA", et(15, 29)),
              bar("BBB", et(15, 30)), sig("BBB", et(15, 30)))
    assert [o.symbol for o in out] == ["AAA"] and out[0].time_stop == et(15, 44)
    assert agent.skip_counts == {"after_last_entry": 1}

def test_last_entry_boundary_in_summer_session() -> None:
    s = date(2024, 7, 1)
    agent = RiskAgent(s)
    out = run(agent, bar("AAA", et(15, 29, s)), sig("AAA", et(15, 29, s)),
              bar("BBB", et(15, 30, s)), sig("BBB", et(15, 30, s)))
    assert [o.symbol for o in out] == ["AAA"]
    assert et(15, 30, s) == datetime(2024, 7, 1, 19, 30, tzinfo=timezone.utc)

def test_signal_without_bar_skipped() -> None:
    agent = RiskAgent(D)
    assert run(agent, sig("AAA", T_DEV)) == [] and agent.skip_counts == {"no_bar": 1}

def test_signal_from_other_session_raises() -> None:
    with pytest.raises(ValueError):
        run(RiskAgent(D), sig("AAA", T_DEV + timedelta(days=1)))

def test_out_of_order_bar_raises() -> None:
    agent = RiskAgent(D); run(agent, bar("AAA", et(9, 32)))
    with pytest.raises(ValueError):
        run(agent, bar("AAA", et(9, 32)))
    run(agent, bar("BBB", et(9, 31)))   # another symbol's clock is independent

def test_size_capped_by_notional() -> None:
    (oi,) = run(RiskAgent(D), bar("AAA", T_DEV, volume=1_000_000), sig("AAA", T_DEV))
    assert oi.qty == 1960 and oi.qty * 1.02 <= 2_000.0 and oi.reason.endswith("cap=notional")

def test_size_capped_by_shares_at_best_ask() -> None:
    quote = make_quote(ts=et(9, 30), symbol="AAA", ask_size=150)
    (oi,) = run(RiskAgent(D), quote, bar("AAA", T_DEV, volume=1_000_000), sig("AAA", T_DEV))
    assert oi.qty == 150 and oi.reason.endswith("cap=quote")

def test_stale_quote_falls_back_to_bar_volume_cap() -> None:
    stale = make_quote(ts=et(9, 25), symbol="AAA", ask_size=150)
    (oi,) = run(RiskAgent(D), stale, bar("AAA", T_DEV), sig("AAA", T_DEV))
    assert oi.qty == 200 and oi.reason.endswith("cap=bar")

def test_zero_qty_after_cap_skipped() -> None:
    agent = RiskAgent(D)
    out = run(agent, make_quote(ts=et(9, 30), symbol="AAA", ask_size=0), bar("AAA", T_DEV), sig("AAA", T_DEV),
              bar("BBB", T_DEV, volume=0), sig("BBB", T_DEV))
    assert out == [] and agent.skip_counts == {"zero_qty": 2}

def test_fixed_pct_bracket_levels() -> None:
    cfg = RiskConfig(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=0.03)
    (oi,) = run(RiskAgent(D, cfg), bar("AAA", T_DEV), sig("AAA", T_DEV))
    assert oi.take_profit == pytest.approx(1.071) and oi.stop_loss == pytest.approx(0.9894)
    assert oi.time_stop == et(9, 46)

def test_atr_bracket_needs_atr_period_bars() -> None:
    cfg = RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0, atr_period=3)
    agent = RiskAgent(D, cfg)
    b1 = bar("AAA", et(9, 31), close=1.00, high=1.02, low=0.98)
    b2 = bar("AAA", et(9, 32), close=1.01, high=1.03, low=1.00)
    b3 = bar("AAA", et(9, 33), close=1.05, high=1.06, low=1.02)
    assert run(agent, b1, b2, sig("AAA", et(9, 32))) == []
    assert agent.skip_counts == {"atr_warmup": 1}
    (oi,) = run(agent, b3, sig("AAA", et(9, 33)))          # ATR = (0.04 + 0.03 + 0.05) / 3
    assert oi.take_profit == pytest.approx(1.13) and oi.stop_loss == pytest.approx(1.01)

def test_flat_bars_atr_bracket_skipped_not_raised() -> None:
    cfg = RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0, atr_period=2)
    agent = RiskAgent(D, cfg)
    out = run(agent, bar("AAA", et(9, 31)), bar("AAA", et(9, 32)), sig("AAA", et(9, 32)))
    assert out == [] and agent.skip_counts == {"bad_bracket": 1}
