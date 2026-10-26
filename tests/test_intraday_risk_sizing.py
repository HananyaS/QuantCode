from __future__ import annotations

from datetime import timedelta

from research.intraday.contracts.events import Quote
from research.intraday.contracts.fixtures import T_DEV, make_quote
from research.intraday.risk.config import RiskConfig
from research.intraday.risk.sizing import size_order

TS = T_DEV
def q(age_min: int, ask_size: int) -> Quote:
    return make_quote(ts=TS - timedelta(minutes=age_min), symbol="AAA", ask_size=ask_size)
def size(**kw):
    base = dict(ref_price=1.02, available_cash=25_000.0, signal_ts=TS, bar_volume=10_000, quote=None, cfg=RiskConfig())
    base.update(kw); return size_order(**base)

def test_notional_binds() -> None:
    qty, cap = size(bar_volume=1_000_000)
    assert (qty, cap) == (1960, "notional") and qty * 1.02 <= 2_000.0

def test_bar_cap_binds_without_quote() -> None:
    assert size() == (200, "bar")

def test_fresh_quote_cap_binds() -> None:
    assert size(bar_volume=1_000_000, quote=q(1, 150)) == (150, "quote")

def test_quote_age_boundary() -> None:
    assert size(quote=q(5, 150)) == (150, "quote")   # exactly max age: fresh
    assert size(quote=q(6, 150)) == (200, "bar")     # stale: bar fallback

def test_zero_liquidity_gives_zero() -> None:
    assert size(quote=q(0, 0))[0] == 0
    assert size(bar_volume=0)[0] == 0

def test_liquidity_cap_floor_is_robust_to_float_error() -> None:
    assert size(bar_volume=1_000_000, quote=q(1, 100), cfg=RiskConfig(quote_size_frac=0.29)) == (29, "quote")

def test_size_uses_available_cash() -> None:
    assert size(bar_volume=1_000_000, available_cash=1_000.0) == (980, "notional")
    assert size(bar_volume=1_000_000, available_cash=-5.0)[0] == 0
