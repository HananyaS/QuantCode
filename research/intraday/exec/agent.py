"""ExecutionAgent: idempotency, the kill flag and the oversized-sell check; the rest is the Broker's."""
from __future__ import annotations

from research.intraday.contracts.events import (
    Bar, ClockTick, Event, Halt, Kill, OrderIntent, OrderUpdate, Quote,
)
from research.intraday.exec.broker import Broker


class ExecutionAgent:
    subscribes: tuple[type[Event], ...] = (OrderIntent, Bar, Quote, ClockTick, Halt, Kill)

    def __init__(self, broker: Broker, name: str = "execution") -> None:
        if not name:
            raise ValueError("ExecutionAgent name must be non-empty")
        self.name = name
        self._broker = broker
        self._seen: set[str] = set()
        self._killed = False

    def on_event(self, event: Event) -> tuple[Event, ...]:
        if isinstance(event, OrderIntent):
            return (self._on_intent(event),)
        if isinstance(event, Bar):
            return self._broker.on_bar(event)
        if isinstance(event, Quote):
            self._broker.on_quote(event)
            return ()
        if isinstance(event, ClockTick):
            return self._broker.on_tick(event)
        if isinstance(event, Halt):
            return self._broker.on_halt(event)
        if isinstance(event, Kill):
            self._killed = True
            return self._broker.on_kill(event)
        return ()

    def _on_intent(self, intent: OrderIntent) -> OrderUpdate:
        def update(status: str, detail: str) -> OrderUpdate:
            return OrderUpdate(ts=intent.ts, intent_id=intent.intent_id, status=status, detail=detail)

        if intent.intent_id in self._seen:
            return update("rejected", "duplicate intent_id")
        self._seen.add(intent.intent_id)
        if self._killed:
            return update("rejected", "after kill")
        if intent.side == "sell" and intent.qty > self._broker.sellable_qty(intent.symbol):
            return update("rejected", "sell exceeds position")
        reason = self._broker.refusal(intent)
        if reason is not None:
            return update("rejected", f"broker refusal: {reason}")
        self._broker.submit(intent)
        return update("accepted", "")
