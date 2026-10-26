"""One leg of the refit cost model (execution.md, Cost; penny etl/quotes.py::leg_cost_bps).

The scalar wrapper feeds one-element Series into the penny frame function, so the
two can never drift (pinned by a parity test).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Mapping, Protocol

import numpy as np
import pandas as pd

from research.intraday.contracts.events import Quote
from research.intraday.exec.config import SUMMARY_READY_ET, ExecConfig, to_et
from research.penny.etl.quotes import leg_cost_bps

Side = Literal["buy", "sell"]
CostSource = Literal["quote", "summary", "default"]


def _finite_float(name: str, x: object) -> float:
    if not isinstance(x, float) or not math.isfinite(x):
        raise ValueError(f"CostInputs.{name} must be a finite float, got {x!r}")
    return x


@dataclass(frozen=True, slots=True, kw_only=True)
class CostInputs:
    """Per symbol-day inputs that are not events (unadjusted share basis)."""

    sigma_daily: float
    adv_shares: float
    half_spread_bps: float | None = None
    ask_notional: float | None = None
    bid_notional: float | None = None

    def __post_init__(self) -> None:
        if _finite_float("sigma_daily", self.sigma_daily) < 0:
            raise ValueError(f"CostInputs.sigma_daily must be >= 0, got {self.sigma_daily}")
        if _finite_float("adv_shares", self.adv_shares) <= 0:
            raise ValueError(f"CostInputs.adv_shares must be > 0, got {self.adv_shares}")
        for name in ("half_spread_bps", "ask_notional", "bid_notional"):
            v = getattr(self, name)
            if v is not None and _finite_float(name, v) <= 0:
                raise ValueError(f"CostInputs.{name} must be None or > 0, got {v}")


class CostInputsSource(Protocol):
    def get(self, symbol: str, session_date: date) -> CostInputs | None: ...


class StaticCostInputs:
    """A fixed (symbol, session_date) -> CostInputs table."""

    def __init__(self, table: Mapping[tuple[str, date], CostInputs]) -> None:
        self._table: dict[tuple[str, date], CostInputs] = dict(table)

    def get(self, symbol: str, session_date: date) -> CostInputs | None:
        return self._table.get((symbol, session_date))

    def __len__(self) -> int:
        return len(self._table)


@dataclass(frozen=True, slots=True)
class CostBook:
    half_spread_bps: float
    side_notional: float | None
    source: CostSource


def assert_summary_causal(fill_instant: datetime) -> None:
    """The 09:34 summary is causal only for fills at or after 09:35 ET."""
    if to_et(fill_instant).time() < SUMMARY_READY_ET:
        raise AssertionError(
            f"09:34 quote summary used for a fill at {to_et(fill_instant).isoformat()} "
            f"(before {SUMMARY_READY_ET} ET): look-ahead")


def quote_half_spread_bps(quote: Quote) -> float:
    mid = (quote.bid + quote.ask) / 2.0
    return float((quote.ask - quote.bid) / (2.0 * mid) * 10_000)


def select_cost_book(side: Side, fill_instant: datetime, quote: Quote | None,
                     inputs: CostInputs, config: ExecConfig) -> CostBook:
    """Fresh quote, else the 09:34 summary (from 09:35 ET), else the default spread."""
    if quote is not None:
        notional = quote.ask * quote.ask_size if side == "buy" else quote.bid * quote.bid_size
        return CostBook(quote_half_spread_bps(quote), float(notional), "quote")
    if inputs.half_spread_bps is not None and to_et(fill_instant).time() >= SUMMARY_READY_ET:
        assert_summary_causal(fill_instant)
        notional = inputs.ask_notional if side == "buy" else inputs.bid_notional
        return CostBook(inputs.half_spread_bps, notional, "summary")
    return CostBook(config.default_half_spread_bps, None, "default")


def leg_cost_bps_scalar(order_notional: float, side_notional: float | None,
                        half_spread_bps: float, sigma_daily: float, participation: float,
                        impact_coefficient: float, levels_penalty: float) -> float:
    """Scalar wrapper over the penny frame function; ``None`` depth -> no shortfall charge."""
    out = leg_cost_bps(
        pd.Series([float(order_notional)]),
        pd.Series([np.nan if side_notional is None else float(side_notional)]),
        pd.Series([float(half_spread_bps)]), pd.Series([float(sigma_daily)]),
        pd.Series([float(participation)]), impact_coefficient, levels_penalty)
    return float(out.iloc[0])


def fill_cost_bps(side: Side, qty: int, price: float, fill_instant: datetime,
                  quote: Quote | None, inputs: CostInputs,
                  config: ExecConfig) -> tuple[float, CostSource]:
    if type(qty) is not int or qty <= 0:
        raise AssertionError(f"fill qty must be a positive int, got {qty!r}")
    book = select_cost_book(side, fill_instant, quote, inputs, config)
    cost = leg_cost_bps_scalar(qty * price, book.side_notional, book.half_spread_bps,
                               inputs.sigma_daily, qty / inputs.adv_shares,
                               config.impact_coefficient, config.levels_penalty)
    if not (math.isfinite(cost) and cost >= 0):
        raise AssertionError(f"leg cost must be finite and >= 0, got {cost}")
    return cost, book.source
