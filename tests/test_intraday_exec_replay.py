"""ExecutionAgent on the bus: cascade order, replay determinism, invariant sweep (plan Task 8)."""
from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pytest

from research.intraday.contracts.events import (
    Bar, ClockTick, Event, Fill, Halt, Kill, OrderIntent, OrderUpdate, Quote,
)
from research.intraday.contracts.fixtures import make_clock_tick, make_order_intent
from research.intraday.contracts.protocols import Envelope
from research.intraday.engine.bus import EventBus
from research.intraday.engine.replay import clock_ticks, run_replay
from research.intraday.exec import ExecConfig, ExecutionAgent, SimBroker, StaticCostInputs
from research.intraday.ops.recorder import read_log
from tests.test_intraday_exec import INPUTS, SESSION, at, bar, make_exec

SYMBOLS = ("AAA", "BBB", "TEST")


@dataclass
class ScriptedAgent:
    name: str
    subscribes: tuple[type[Event], ...]
    script: dict[tuple[str, datetime], tuple[Event, ...]] = field(default_factory=dict)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        return self.script.get((type(event).__name__, event.ts), ())


def new_broker() -> SimBroker:
    return SimBroker(ExecConfig(), StaticCostInputs({(s, SESSION): INPUTS for s in SYMBOLS}))


def _bar(rng: random.Random, symbol: str, k: int, price: float) -> Bar:
    open_ = price
    close = open_ * (1.0 + rng.uniform(-0.02, 0.02))
    high = max(open_, close) * (1.0 + rng.uniform(0.0, 0.01))
    low = min(open_, close) * (1.0 - rng.uniform(0.0, 0.01))
    return Bar(ts=at(k + 1), symbol=symbol, bar_start=at(k), open=open_, high=high, low=low,
               close=close, volume=rng.randint(0, 20_000), vwap=(high + low) / 2.0,
               trade_count=rng.randint(0, 200), feed="sip")


def _quote(rng: random.Random, symbol: str, k: int, price: float) -> Quote:
    return Quote(ts=at(k + 1), symbol=symbol, bid=price * (1.0 - rng.uniform(0.0, 0.01)),
                 ask=price * (1.0 + rng.uniform(0.0, 0.01)),
                 bid_size=rng.randint(0, 500), ask_size=rng.randint(0, 500))


def _intent(rng: random.Random, symbol: str, k: int, price: float, n: int,
            ids: list[str]) -> OrderIntent:
    ts = at(k + 1)
    iid = rng.choice(ids) if ids and rng.random() < 0.1 else f"{symbol}-{n}"
    if rng.random() < 0.1:
        return make_order_intent(ts=ts, intent_id=iid, symbol=symbol, side="sell",
                                 qty=rng.randint(1, 400))
    kind = rng.choice(("none", "fixed", "time"))
    bracket: dict[str, object] = {}
    if kind == "fixed":
        bracket = dict(take_profit=price * 1.05, stop_loss=price * 0.95)
    elif kind == "time":
        bracket = dict(time_stop=ts + timedelta(minutes=rng.randint(2, 30)))
    return make_order_intent(ts=ts, intent_id=iid, symbol=symbol, qty=rng.randint(1, 400),
                             **bracket)


def _market(seed: int) -> list[Event]:
    rng = random.Random(seed)
    price = {s: 1.00 for s in SYMBOLS}
    halt_at = rng.randint(30, 300)
    ids: list[str] = []
    out: list[Event] = []
    n = 0
    for k in range(390):
        for s in SYMBOLS:
            if rng.random() >= 0.1:
                b = _bar(rng, s, k, price[s])
                price[s] = b.close
                out.append(b)
            if rng.random() < 0.2:
                out.append(_quote(rng, s, k, price[s]))
        if k == halt_at:
            out.append(Halt(ts=at(k + 1), symbol="BBB", halted=True, reason="LULD"))
        if k == halt_at + 6:
            out.append(Halt(ts=at(k + 1), symbol="BBB", halted=False, reason=""))
        for s in SYMBOLS:
            if at(k + 1) < at(359) and rng.random() < 0.05:
                n += 1
                it = _intent(rng, s, k, price[s], n, ids)
                ids.append(it.intent_id)
                out.append(it)
        if k == 199 and seed % 3 == 0:
            out.append(Kill(ts=at(200), reason="test"))
    return out


def session_sources(seed: int) -> list[Iterable[Event]]:
    return [clock_ticks(at(-1), at(391)), _market(seed)]


# ---- invariant checks over a recorded log -----------------------------------
def _first_intents(envs: list[Envelope]) -> dict[str, OrderIntent]:
    out: dict[str, OrderIntent] = {}
    for e in envs:
        if isinstance(e.event, OrderIntent) and e.event.intent_id not in out:
            out[e.event.intent_id] = e.event
    return out


def _fills(envs: list[Envelope]) -> list[Envelope]:
    return [e for e in envs if isinstance(e.event, Fill)]


def fills_on_bars_at_or_after_decision(envs: list[Envelope]) -> None:
    by_seq = {e.seq: e for e in envs}
    intents = _first_intents(envs)
    for e in _fills(envs):
        cause = by_seq[e.cause_seq].event
        if e.event.side == "buy":
            ok = isinstance(cause, Bar) and cause.bar_start >= intents[e.event.intent_id].ts
        else:
            ok = isinstance(cause, (Bar, ClockTick))
        if not ok:
            raise AssertionError(f"fill seq {e.seq} caused by {type(cause).__name__}")


def no_fill_while_halted(envs: list[Envelope]) -> None:
    halted: set[str] = set()
    for e in envs:
        if isinstance(e.event, Halt):
            (halted.add if e.event.halted else halted.discard)(e.event.symbol)
        elif isinstance(e.event, Fill) and e.event.symbol in halted:
            raise AssertionError(f"fill seq {e.seq} in halted {e.event.symbol}")


def buy_qty_within_intent(envs: list[Envelope]) -> None:
    intents = _first_intents(envs)
    total: dict[str, int] = {}
    for e in _fills(envs):
        if e.event.side == "buy":
            total[e.event.intent_id] = total.get(e.event.intent_id, 0) + e.event.qty
            if total[e.event.intent_id] > intents[e.event.intent_id].qty:
                raise AssertionError(f"fill seq {e.seq} overfills {e.event.intent_id}")


def net_position_never_negative(envs: list[Envelope]) -> None:
    pos: dict[str, int] = {}
    for e in _fills(envs):
        f = e.event
        pos[f.symbol] = pos.get(f.symbol, 0) + (f.qty if f.side == "buy" else -f.qty)
        if pos[f.symbol] < 0:
            raise AssertionError(f"fill seq {e.seq} takes {f.symbol} short")


def one_ack_per_intent_and_no_fill_for_duplicates(envs: list[Envelope]) -> None:
    acks: dict[int, list[str]] = {}
    for e in envs:
        if isinstance(e.event, OrderUpdate) and e.event.status in ("accepted", "rejected"):
            acks.setdefault(e.cause_seq, []).append(e.event.status)
    first_status: dict[str, str] = {}
    for e in envs:
        if isinstance(e.event, OrderIntent):
            got = acks.get(e.seq, [])
            if len(got) != 1:
                raise AssertionError(f"intent seq {e.seq} got acks {got}")
            first_status.setdefault(e.event.intent_id, got[0])
    for e in _fills(envs):
        if first_status.get(e.event.intent_id) != "accepted":
            raise AssertionError(f"fill seq {e.seq} for never-accepted {e.event.intent_id}")


def no_buy_after_kill(envs: list[Envelope]) -> None:
    killed = False
    for e in envs:
        if isinstance(e.event, Kill):
            killed = True
        elif killed and isinstance(e.event, Fill) and e.event.side == "buy":
            raise AssertionError(f"buy fill seq {e.seq} after kill")


def partial_buy_followed_by_cancel(envs: list[Envelope]) -> None:
    intents = _first_intents(envs)
    for i, e in enumerate(envs):
        f = e.event
        if isinstance(f, Fill) and f.side == "buy" and f.qty < intents[f.intent_id].qty:
            nxt = envs[i + 1] if i + 1 < len(envs) else None
            if not (nxt is not None and isinstance(nxt.event, OrderUpdate)
                    and nxt.event.status == "cancelled" and nxt.event.intent_id == f.intent_id
                    and nxt.cause_seq == e.cause_seq):
                raise AssertionError(f"partial fill seq {e.seq} not followed by its cancel")


def exit_prices_are_opens_or_last_close(envs: list[Envelope]) -> None:
    by_seq = {e.seq: e for e in envs}
    last_close: dict[str, float] = {}
    for e in envs:
        ev = e.event
        if isinstance(ev, Bar):
            last_close[ev.symbol] = ev.close
        elif isinstance(ev, Fill) and ev.side == "sell":
            cause = by_seq[e.cause_seq].event
            want = cause.open if isinstance(cause, Bar) else last_close[ev.symbol]
            if ev.price != want:
                raise AssertionError(f"sell fill seq {e.seq} price {ev.price} != {want}")


CHECKS = (fills_on_bars_at_or_after_decision, no_fill_while_halted, buy_qty_within_intent,
          net_position_never_negative, one_ack_per_intent_and_no_fill_for_duplicates,
          no_buy_after_kill, partial_buy_followed_by_cancel, exit_prices_are_opens_or_last_close)


# ---- tests --------------------------------------------------------------------
def test_bus_partial_fill_then_cancel_in_one_cascade() -> None:
    feed = ScriptedAgent("feed", (ClockTick,), {("ClockTick", at(1)): (bar(0),), ("ClockTick", at(2)): (bar(1),)})
    risk = ScriptedAgent("risk", (Bar,), {("Bar", at(1)): (make_order_intent(ts=at(1), qty=300),)})
    ex, _ = make_exec()
    bus = EventBus([feed, risk, ex])
    first = bus.publish(make_clock_tick(ts=at(1)))
    second = bus.publish(make_clock_tick(ts=at(2)))
    assert [e.event.status for e in first if e.source == "execution"] == ["accepted"]   # no fill on bar(0)
    out = [e for e in second if e.source == "execution"]
    assert [type(e.event).__name__ for e in out] == ["Fill", "OrderUpdate"]
    assert out[0].event.qty == 200 and out[1].event.status == "cancelled"
    assert out[0].cause_seq == out[1].cause_seq          # same cause: the bar(1) envelope


def test_same_session_twice_identical_hash(tmp_path: Path) -> None:
    h = [run_replay([ExecutionAgent(new_broker())], session_sources(7), tmp_path / f"{i}.jsonl") for i in (0, 1)]
    assert len(h[0]) == 64 and h[0] == h[1]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHILD = (
    "import sys\n"
    "from pathlib import Path\n"
    "from tests.test_intraday_exec_replay import session_sources, new_broker\n"
    "from research.intraday.exec import ExecutionAgent\n"
    "from research.intraday.engine.replay import run_replay\n"
    "print(run_replay([ExecutionAgent(new_broker())], session_sources(7), Path(sys.argv[1])))\n"
)


def _hash_in_subprocess(log_path: Path, hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.run([sys.executable, "-c", _CHILD, str(log_path)], cwd=_REPO_ROOT,
                         env=env, capture_output=True, text=True, timeout=120, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_exec_hash_identical_across_hash_seeds(tmp_path: Path) -> None:
    # the tests/test_intraday_replay.py::_replay_in_subprocess pattern; the child imports
    # session_sources and new_broker from tests.test_intraday_exec_replay
    assert _hash_in_subprocess(tmp_path / "a.jsonl", "0") == _hash_in_subprocess(tmp_path / "b.jsonl", "12345")


@pytest.mark.parametrize("seed", range(20))
def test_invariants_hold(seed: int, tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    run_replay([ExecutionAgent(new_broker())], session_sources(seed), path)
    envs = list(read_log(path))
    for check in CHECKS:
        check(envs)                                       # each raises AssertionError with the offending seq


def test_sweep_exercises_fills_and_exits(tmp_path: Path) -> None:
    """Guard against a vacuous sweep: the synthetic sessions do fill, exit and partial-fill."""
    kinds: set[str] = set()
    for seed in (0, 1, 2):
        path = tmp_path / f"{seed}.jsonl"
        run_replay([ExecutionAgent(new_broker())], session_sources(seed), path)
        for e in read_log(path):
            if isinstance(e.event, Fill):
                kinds.add(f"fill:{e.event.side}")
            elif isinstance(e.event, OrderUpdate):
                kinds.add(f"update:{e.event.status}")
    assert {"fill:buy", "fill:sell", "update:accepted", "update:rejected",
            "update:cancelled", "update:expired"} <= kinds
