from __future__ import annotations

from datetime import date, datetime, time, timezone

import pytest

from research.intraday.contracts.fixtures import T_DEV
from research.intraday.risk.config import RiskConfig
from research.intraday.risk.session import et_date, session_bounds


def test_defaults_match_design_doc() -> None:
    c = RiskConfig()
    assert (c.start_equity_usd, c.position_notional_usd, c.max_open_positions) == (25_000.0, 2_000.0, 5)
    assert (c.exit_mode, c.atr_period, c.quote_size_frac, c.quote_max_age_min) == ("time_stop", 14, 1.0, 5)
    assert (c.bar_participation_cap, c.daily_loss_limit_pct) == (0.02, 0.02)
    assert (c.last_entry_et, c.flatten_et, c.hold_min) == (time(15, 30), time(15, 55), None)

@pytest.mark.parametrize("kw", [
    dict(exit_mode="time_stop", tp_pct=0.05),
    dict(exit_mode="fixed_pct", tp_pct=0.05),                 # sl_pct missing
    dict(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=1.0),     # stop at zero
    dict(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=0.03, tp_atr=2.0),
    dict(exit_mode="atr", tp_atr=2.0),
    dict(exit_mode="atr", tp_atr=0.0, sl_atr=1.0),
    dict(exit_mode="trailing"),
])
def test_exit_params_must_match_mode(kw) -> None:
    with pytest.raises(ValueError):
        RiskConfig(**kw)

def test_each_exit_mode_constructs() -> None:
    RiskConfig(exit_mode="time_stop")
    RiskConfig(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=0.03)
    RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0)

@pytest.mark.parametrize("kw", [dict(max_open_positions=0), dict(quote_size_frac=1.5),
    dict(bar_participation_cap=0.0), dict(daily_loss_limit_pct=1.0), dict(hold_min=0),
    dict(atr_period=0), dict(position_notional_usd=0.0)])
def test_bad_scalars_rejected(kw) -> None:
    with pytest.raises(ValueError):
        RiskConfig(**kw)

def test_session_bounds_are_utc_and_match_t_dev() -> None:
    last, flat = session_bounds(date(2024, 3, 4), RiskConfig())
    assert last == datetime(2024, 3, 4, 20, 30, tzinfo=timezone.utc)
    assert flat == datetime(2024, 3, 4, 20, 55, tzinfo=timezone.utc)
    assert last.tzinfo is timezone.utc and flat.tzinfo is timezone.utc
    assert et_date(T_DEV) == date(2024, 3, 4)

def test_session_bounds_follow_dst() -> None:
    last, flat = session_bounds(date(2024, 7, 1), RiskConfig())   # EDT, UTC-4
    assert last == datetime(2024, 7, 1, 19, 30, tzinfo=timezone.utc)
    assert flat == datetime(2024, 7, 1, 19, 55, tzinfo=timezone.utc)

@pytest.mark.parametrize("kw", [dict(atr_period=14.0), dict(max_open_positions=True),
    dict(max_entries_per_symbol=1.0), dict(quote_max_age_min=5.0), dict(hold_min=5.0), dict(hold_min=True)])
def test_integer_fields_must_be_int(kw) -> None:
    with pytest.raises(ValueError):
        RiskConfig(**kw)

def test_atr_floor_defaults_to_none_and_accepts_a_positive_float() -> None:
    assert RiskConfig().atr_floor is None
    assert RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0, atr_floor=0.01).atr_floor == 0.01

@pytest.mark.parametrize("floor", [0.0, -0.01, float("nan"), float("inf"), 1, True])
def test_bad_atr_floor_rejected(floor) -> None:
    with pytest.raises(ValueError, match="atr_floor"):
        RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0, atr_floor=floor)
