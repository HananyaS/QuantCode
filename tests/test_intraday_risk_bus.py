from __future__ import annotations

from datetime import date, timedelta

from research.intraday.contracts.codec import canonical_json, to_record
from research.intraday.contracts.events import Bar, Event, OrderIntent
from research.intraday.contracts.fixtures import T_DEV, make_bar, make_quote, make_signal
from research.intraday.engine.bus import EventBus
from research.intraday.risk import RiskAgent

D = date(2024, 3, 4)
def _bar(sym: str) -> Bar:
    return make_bar(symbol=sym, ts=T_DEV, bar_start=T_DEV - timedelta(minutes=1), open=1.02,
                    high=1.02, low=1.02, close=1.02, vwap=1.02, volume=1_000_000)
EVENTS: list[Event] = [
    make_quote(ts=T_DEV - timedelta(minutes=1), symbol="AAA", ask_size=150),
    _bar("AAA"), _bar("BBB"), _bar("CCC"),
    make_signal(ts=T_DEV, symbol="AAA", score=0.7), make_signal(ts=T_DEV, symbol="BBB", score=0.7),
    make_signal(ts=T_DEV, symbol="CCC", score=0.1),            # below threshold: no intent
]

def _log(events: list[Event]) -> list[str]:
    bus = EventBus([RiskAgent(D)])
    return [canonical_json(dict(to_record(env.event), source=env.source, seq=env.seq, cause=env.cause_seq))
            for ev in events for env in bus.publish(ev)]

def test_on_bus_risk_emits_only_order_intents_with_cause_ts() -> None:
    bus = EventBus([RiskAgent(D)])
    envs = [env for ev in EVENTS for env in bus.publish(ev)]
    mine = [e for e in envs if e.source == "risk"]
    assert [(type(e.event), e.event.symbol, e.event.qty) for e in mine] == [(OrderIntent, "AAA", 150), (OrderIntent, "BBB", 1960)]
    by_seq = {e.seq: e for e in envs}
    assert all(e.event.ts == by_seq[e.cause_seq].event.ts for e in mine)

def test_two_runs_give_identical_logs() -> None:
    assert _log(EVENTS) == _log(EVENTS)
