"""Per-side trading costs for SPY/QQQ on Alpaca ($0 commission), as fractions of notional.

PREREGISTRATION_L1 §2.5. Prices here are dividend-adjusted and lower than the
raw prices, so per-share amounts divided by them overstate the cost (conservative).

- Marketable entry, stop exit, end-of-day exit pay the half-spread ($0.005/share).
- A stop exit pays one more tick ($0.01/share) of slippage.
- A take-profit is a resting limit and pays no spread.
- Sells pay the SEC fee (2.78e-5 x notional) and FINRA TAF ($0.000195/share,
  cap ignored: conservative). Both sides pay CAT ($0.000003/share).
Constants: research/intraday/docs/research/2026-11-alpaca-order-facts.md.
"""
from __future__ import annotations

from dataclasses import dataclass

HALF_SPREAD = 0.005
STOP_SLIP = 0.01
SEC_RATE = 2.78e-5
TAF_PER_SHARE = 0.000195
CAT_PER_SHARE = 0.000003

EXIT_KINDS = ("tp", "sl", "time")


@dataclass(frozen=True)
class CostSpec:
    """`scale` multiplies spread and slippage; commission/slip are extra $/share each side."""
    scale: float = 1.0
    commission: float = 0.0
    extra_slip: float = 0.0
    spread: bool = True

    def entry_frac(self, price: float) -> float:
        per_share = CAT_PER_SHARE + self.commission + self.extra_slip
        if self.spread:
            per_share += self.scale * HALF_SPREAD
        return per_share / price

    def exit_frac(self, price: float, kind: str) -> float:
        assert kind in EXIT_KINDS, kind
        per_share = CAT_PER_SHARE + TAF_PER_SHARE + self.commission
        if kind != "tp":
            per_share += self.extra_slip
            if self.spread:
                per_share += self.scale * HALF_SPREAD
        if kind == "sl" and self.spread:
            per_share += self.scale * STOP_SLIP
        return per_share / price + SEC_RATE


BASE = CostSpec()
DOUBLE = CostSpec(scale=2.0)
# the papers' own assumptions (diagnostic only)
PAPER_ORB = CostSpec(commission=0.0005, spread=False)            # Zarattini & Aziz 2023, Table 1
PAPER_BTM = CostSpec(commission=0.0035, extra_slip=0.001, spread=False)   # Zarattini et al. 2024 §4.6
