from __future__ import annotations

from datetime import timedelta

import pytest

from research.intraday.contracts import Bar, Envelope, Event, Kill, Signal
from research.intraday.contracts.fixtures import T0, make_bar, make_signal
from research.intraday.engine.bus import EventBus


class Echo:
    def __init__(self, name: str = "echo") -> None:
        self.name = name
        self.subscribes: tuple[type[Event], ...] = (Bar,)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        return (make_signal(ts=event.ts),)


class Counter:
    def __init__(self, name: str = "counter",
                 subscribes: tuple[type[Event], ...] = (Bar, Signal)) -> None:
        self.name = name
        self.subscribes = subscribes
        self.seen: list[Event] = []

    def on_event(self, event: Event) -> tuple[Event, ...]:
        self.seen.append(event)
        return ()


class Late:
    name = "late"
    subscribes: tuple[type[Event], ...] = (Bar,)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        return (make_signal(ts=event.ts + timedelta(minutes=1)),)


class SignalToKill:
    name = "a"
    subscribes: tuple[type[Event], ...] = (Signal,)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        return (Kill(ts=event.ts, reason="x"),)


class KillToSignal:
    name = "b"
    subscribes: tuple[type[Event], ...] = (Kill,)

    def on_event(self, event: Event) -> tuple[Event, ...]:
        return (make_signal(ts=event.ts),)


class SelfEcho:
    name = "self"
    subscribes: tuple[type[Event], ...] = (Bar, Signal)

    def __init__(self) -> None:
        self.seen: list[Event] = []

    def on_event(self, event: Event) -> tuple[Event, ...]:
        self.seen.append(event)
        return (make_signal(ts=event.ts),) if isinstance(event, Bar) else ()


def _shape(envs: list[Envelope]) -> list[tuple[int, str, str]]:
    return [(e.seq, e.source, type(e.event).__name__) for e in envs]


def test_dispatch_is_breadth_first_in_registration_order() -> None:
    counter = Counter()
    bus = EventBus([Echo("e1"), Echo("e2"), counter])
    out = bus.publish(make_bar())
    assert _shape(out) == [
        (0, "external", "Bar"), (1, "e1", "Signal"), (2, "e2", "Signal")]
    assert [e.cause_seq for e in out] == [None, 0, 0]
    assert [type(e).__name__ for e in counter.seen] == ["Bar", "Signal", "Signal"]


def test_agent_receives_only_subscribed_types() -> None:
    sig_only = Counter("s", (Signal,))
    none = Counter("n", ())
    bus = EventBus([Echo(), sig_only, none])
    bus.publish(make_bar())
    assert [type(e).__name__ for e in sig_only.seen] == ["Signal"]
    assert none.seen == []


def test_agent_never_receives_its_own_emissions() -> None:
    agent = SelfEcho()
    out = EventBus([agent]).publish(make_bar())
    assert [type(e).__name__ for e in agent.seen] == ["Bar"]
    assert _shape(out) == [(0, "external", "Bar"), (1, "self", "Signal")]


def test_emitted_event_must_carry_cause_ts() -> None:
    bus = EventBus([Late()])
    with pytest.raises(ValueError, match="late") as exc:
        bus.publish(make_bar())
    msg = str(exc.value)
    assert "Signal" in msg
    assert str(T0) in msg
    assert str(T0 + timedelta(minutes=1)) in msg


def test_runaway_cascade_raises() -> None:
    bus = EventBus([SignalToKill(), KillToSignal()], max_cascade=50)
    with pytest.raises(RuntimeError, match="50"):
        bus.publish(make_signal())


def test_cascade_counter_resets_per_publish() -> None:
    bus = EventBus([Echo()], max_cascade=2)
    assert len(bus.publish(make_bar())) == 2
    assert len(bus.publish(make_bar())) == 2


def test_non_monotone_external_publish_raises() -> None:
    bus = EventBus([])
    bus.publish(make_bar(ts=T0))
    with pytest.raises(ValueError, match="non-monotone source") as exc:
        bus.publish(make_signal(ts=T0 - timedelta(seconds=1)))
    assert str(T0) in str(exc.value)


def test_equal_ts_external_publish_is_allowed() -> None:
    bus = EventBus([])
    bus.publish(make_bar(ts=T0))
    bus.publish(make_signal(ts=T0))


def test_sink_sees_every_envelope_with_increasing_seq() -> None:
    seen: list[Envelope] = []
    bus = EventBus([Echo()], sink=seen.append)
    first = bus.publish(make_bar())
    second = bus.publish(make_bar())
    assert seen == first + second
    assert [e.seq for e in seen] == [0, 1, 2, 3]


def test_duplicate_agent_names_rejected() -> None:
    with pytest.raises(ValueError, match="dup"):
        EventBus([Counter("dup"), Counter("dup")])


def test_empty_agent_name_rejected() -> None:
    with pytest.raises(ValueError):
        EventBus([Counter("")])


def test_agents_frozen_at_construction() -> None:
    agents = [Counter("c")]
    bus = EventBus(agents)
    agents.append(Echo())
    assert len(bus.publish(make_bar())) == 1


def test_sink_exception_propagates() -> None:
    def boom(env: Envelope) -> None:
        raise OSError("disk")

    with pytest.raises(OSError):
        EventBus([], sink=boom).publish(make_bar())


# ---- hardening (final-review fix A5) ----------------------------------------
from research.intraday.engine.bus import EXTERNAL  # noqa: E402


def test_agent_named_external_rejected() -> None:
    with pytest.raises(ValueError, match=EXTERNAL):
        EventBus([Counter(EXTERNAL)])


def test_bus_is_poisoned_after_a_failed_cascade() -> None:
    bus = EventBus([Late()])
    with pytest.raises(ValueError, match="late"):
        bus.publish(make_bar())
    with pytest.raises(RuntimeError, match="failed"):
        bus.publish(make_bar())
    with pytest.raises(RuntimeError, match="failed"):
        bus.publish(make_signal())


def test_bus_is_poisoned_after_sink_failure() -> None:
    calls: list[int] = []

    def boom(env: Envelope) -> None:
        calls.append(env.seq)
        raise OSError("disk")

    bus = EventBus([], sink=boom)
    with pytest.raises(OSError):
        bus.publish(make_bar())
    with pytest.raises(RuntimeError, match="failed"):
        bus.publish(make_bar())
    assert calls == [0]


def test_bus_is_poisoned_after_runaway_cascade() -> None:
    bus = EventBus([SignalToKill(), KillToSignal()], max_cascade=5)
    with pytest.raises(RuntimeError, match="5"):
        bus.publish(make_signal())
    with pytest.raises(RuntimeError, match="failed"):
        bus.publish(make_signal())


def test_non_monotone_rejection_does_not_poison() -> None:
    bus = EventBus([])
    bus.publish(make_bar(ts=T0))
    with pytest.raises(ValueError, match="non-monotone"):
        bus.publish(make_signal(ts=T0 - timedelta(seconds=1)))
    assert len(bus.publish(make_bar(ts=T0))) == 1
