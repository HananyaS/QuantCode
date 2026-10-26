"""Order sizing: notional (and uncommitted cash), capped by quote or bar-volume liquidity."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Literal

from research.intraday.contracts.events import Quote
from research.intraday.risk.config import RiskConfig

CapName = Literal["notional", "quote", "bar"]

_FLOOR_TOL = 1e-9


def floor_shares(x: float) -> int:
    """Floor a liquidity cap, tolerant of float error just under an integer."""
    assert x >= 0, f"floor_shares needs x >= 0, got {x}"
    return math.floor(x + _FLOOR_TOL)


def quote_is_fresh(quote: Quote | None, signal_ts: datetime, max_age_min: int) -> bool:
    """A quote no older than `max_age_min` (inclusive) and not after the signal."""
    if quote is None:
        return False
    age = signal_ts - quote.ts
    return timedelta(0) <= age <= timedelta(minutes=max_age_min)


def _liquidity_cap(signal_ts: datetime, bar_volume: int, quote: Quote | None,
                   cfg: RiskConfig) -> tuple[int, CapName]:
    if quote is not None and quote_is_fresh(quote, signal_ts, cfg.quote_max_age_min):
        return floor_shares(cfg.quote_size_frac * quote.ask_size), "quote"
    return floor_shares(cfg.bar_participation_cap * bar_volume), "bar"


def size_order(*, ref_price: float, available_cash: float, signal_ts: datetime, bar_volume: int,
               quote: Quote | None, cfg: RiskConfig) -> tuple[int, CapName]:
    """(qty, binding cap). A tie between liquidity and notional reads `notional`."""
    assert ref_price > 0, f"ref_price must be > 0, got {ref_price}"
    budget = max(0.0, min(cfg.position_notional_usd, available_cash))
    notional_qty = math.floor(budget / ref_price)
    liq_qty, liq_name = _liquidity_cap(signal_ts, bar_volume, quote, cfg)
    if liq_qty < notional_qty:
        return liq_qty, liq_name
    return notional_qty, "notional"
