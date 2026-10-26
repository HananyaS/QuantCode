"""The broker seam: ExecutionAgent talks to a Broker; SimBroker now, AlpacaPaperBroker in Phase 5."""
from __future__ import annotations

from typing import Protocol

from research.intraday.contracts.events import Bar, ClockTick, Fill, Halt, Kill, OrderIntent, OrderUpdate, Quote

ExecEvent = Fill | OrderUpdate


class Broker(Protocol):
    def refusal(self, intent: OrderIntent) -> str | None:
        """Why the broker would refuse this intent now, or None."""
        ...

    def submit(self, intent: OrderIntent) -> None: ...

    def sellable_qty(self, symbol: str) -> int:
        """Open qty not already committed to a pending exit or a working sell."""
        ...

    def on_bar(self, bar: Bar) -> tuple[ExecEvent, ...]: ...

    def on_quote(self, quote: Quote) -> None: ...

    def on_tick(self, tick: ClockTick) -> tuple[ExecEvent, ...]: ...

    def on_halt(self, halt: Halt) -> tuple[ExecEvent, ...]: ...

    def on_kill(self, kill: Kill) -> tuple[ExecEvent, ...]: ...
