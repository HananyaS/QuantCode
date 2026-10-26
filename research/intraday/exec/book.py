"""SimBroker's mutable internal book: working orders, position lots, pending exits."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from research.intraday.contracts.events import OrderIntent

ExitReason = Literal["stop", "target", "time", "kill"]


@dataclass(slots=True)
class WorkingOrder:
    intent: OrderIntent
    client_order_id: str
    remaining: int
    not_before: datetime
    deadline: datetime | None  # intent.ts + entry_max_wait_min for a buy; None for a sell
    seq: int

    @property
    def symbol(self) -> str:
        return self.intent.symbol


@dataclass(slots=True)
class PendingExit:
    reason: ExitReason
    not_before: datetime
    client_order_id: str


@dataclass(slots=True)
class Lot:
    entry_id: str
    symbol: str
    open_qty: int
    take_profit: float | None
    stop_loss: float | None
    time_stop: datetime | None
    seq: int
    exit: PendingExit | None = None
    stranded: bool = False
