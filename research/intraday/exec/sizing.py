"""Fill size cap and quote freshness (execution.md, Size cap; same expression as risk.md)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Literal

from research.intraday.contracts.events import Bar, Quote
from research.intraday.exec.config import ExecConfig

CapSource = Literal["quote", "bar"]


def fresh_quote(quote: Quote | None, fill_instant: datetime, config: ExecConfig) -> Quote | None:
    """The quote if ``0 <= fill_instant - quote.ts <= quote_max_age_min``; a later quote raises."""
    if quote is None:
        return None
    if quote.ts > fill_instant:
        raise AssertionError(
            f"quote for {quote.symbol} stamped {quote.ts.isoformat()} is after the fill "
            f"instant {fill_instant.isoformat()}: look-ahead into the fill bar")
    if fill_instant - quote.ts > timedelta(minutes=config.quote_max_age_min):
        return None
    return quote


def fill_cap(side: Literal["buy", "sell"], fill_instant: datetime, bar: Bar,
             quote: Quote | None, config: ExecConfig) -> tuple[int, CapSource]:
    """Shares at the best quote of the side (fresh quote), else a share of the bar's volume."""
    q = fresh_quote(quote, fill_instant, config)
    if q is not None:
        size = q.ask_size if side == "buy" else q.bid_size
        cap, src = math.floor(config.quote_size_frac * size), "quote"
    else:
        cap, src = math.floor(config.bar_participation_cap * bar.volume), "bar"
    if type(cap) is not int or cap < 0:
        raise AssertionError(f"fill cap must be a non-negative int, got {cap!r}")
    return cap, src
