"""Position, working-intent and cash books, kept exactly from Fills and OrderUpdates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from research.intraday.contracts.events import Fill

_BPS = 10_000.0


@dataclass(slots=True)
class _Position:
    qty: int
    cost_basis: float


@dataclass(slots=True)
class _Working:
    symbol: str
    remaining: int
    ref_price: float


class Books:
    """One session's account. Cost basis includes the buy leg's cost."""

    def __init__(self, start_equity_usd: float) -> None:
        assert start_equity_usd > 0
        self.cash: float = start_equity_usd
        self.realised_pnl: float = 0.0
        self._symbol_of: dict[str, str] = {}
        self._working: dict[str, _Working] = {}
        self._positions: dict[str, _Position] = {}

    def open_intent(self, intent_id: str, symbol: str, qty: int, ref_price: float) -> None:
        if intent_id in self._symbol_of:
            raise ValueError(f"intent_id {intent_id!r} already opened")
        assert qty >= 1 and ref_price > 0
        self._symbol_of[intent_id] = symbol
        self._working[intent_id] = _Working(symbol, qty, ref_price)

    def require_known(self, intent_id: str) -> None:
        if intent_id not in self._symbol_of:
            raise ValueError(f"unknown intent_id {intent_id!r}")

    def apply_fill(self, fill: Fill) -> None:
        self.require_known(fill.intent_id)
        if fill.symbol != self._symbol_of[fill.intent_id]:
            raise ValueError(
                f"fill symbol {fill.symbol!r} does not match intent {fill.intent_id!r} "
                f"({self._symbol_of[fill.intent_id]!r})")
        if fill.side == "buy":
            self._apply_buy(fill)
        else:
            self._apply_sell(fill)

    def _apply_buy(self, fill: Fill) -> None:
        w = self._working.get(fill.intent_id)
        if w is None:
            raise ValueError(f"buy fill for intent {fill.intent_id!r} that is not working")
        if fill.qty > w.remaining:
            raise ValueError(f"overfill on {fill.intent_id!r}: qty {fill.qty} > remaining {w.remaining}")
        paid = fill.price * fill.qty * (1 + fill.cost_bps / _BPS)
        self.cash -= paid
        pos = self._positions.setdefault(fill.symbol, _Position(0, 0.0))
        pos.qty += fill.qty
        pos.cost_basis += paid
        w.remaining -= fill.qty
        if w.remaining == 0:
            del self._working[fill.intent_id]

    def _apply_sell(self, fill: Fill) -> None:
        pos = self._positions.get(fill.symbol)
        held = 0 if pos is None else pos.qty
        if pos is None or fill.qty > held:
            raise ValueError(f"sell fill of {fill.qty} {fill.symbol!r} exceeds position {held}")
        proceeds = fill.price * fill.qty * (1 - fill.cost_bps / _BPS)
        released = pos.cost_basis * fill.qty / pos.qty
        self.cash += proceeds
        self.realised_pnl += proceeds - released
        pos.cost_basis -= released
        pos.qty -= fill.qty
        if pos.qty == 0:
            del self._positions[fill.symbol]

    def close_working(self, intent_id: str) -> None:
        """Drop a working remainder (its filled part stays a position); no-op if not working."""
        self.require_known(intent_id)
        self._working.pop(intent_id, None)

    def position_qty(self, symbol: str) -> int:
        pos = self._positions.get(symbol)
        return 0 if pos is None else pos.qty

    def _busy_symbols(self) -> set[str]:
        return set(self._positions) | {w.symbol for w in self._working.values()}

    def is_busy(self, symbol: str) -> bool:
        return symbol in self._busy_symbols()

    def occupied(self) -> int:
        return len(self._busy_symbols())

    def reserved_cash(self) -> float:
        return sum((w.remaining * w.ref_price for w in self._working.values()), 0.0)

    def marked_equity(self, closes: Mapping[str, float]) -> float:
        missing = sorted(s for s in self._positions if s not in closes)
        if missing:
            raise ValueError(f"no close to mark positions {missing}")
        return self.cash + sum(p.qty * closes[s] for s, p in sorted(self._positions.items()))

    def positions(self) -> dict[str, tuple[int, float]]:
        return {s: (p.qty, p.cost_basis) for s, p in sorted(self._positions.items())}
