from __future__ import annotations

from datetime import timedelta

import pytest

from research.intraday.contracts.fixtures import T_DEV, make_bar
from research.intraday.risk.brackets import (
    BRACKETS, Bracket, make_bracket, mean_true_range, true_range,
)
from research.intraday.risk.config import EXIT_MODES, RiskConfig

FIX = RiskConfig(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=0.03)
ATR = RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0, atr_period=3)

def test_dispatch_table_covers_exactly_the_exit_modes() -> None:
    assert set(BRACKETS) == set(EXIT_MODES)

def test_time_stop_has_no_levels() -> None:
    assert make_bracket(1.02, None, RiskConfig()) == Bracket(None, None)

def test_fixed_pct_levels() -> None:
    b = make_bracket(1.02, None, FIX)
    assert b.take_profit == pytest.approx(1.071) and b.stop_loss == pytest.approx(0.9894)

def test_true_range_first_bar_and_gap() -> None:
    b = make_bar(ts=T_DEV, bar_start=T_DEV - timedelta(minutes=1), open=1.05, high=1.06, low=1.02, close=1.05, vwap=1.05)
    assert true_range(b, None) == pytest.approx(0.04)
    assert true_range(b, 1.01) == pytest.approx(0.05)   # prior close below the bar's low
    assert true_range(b, 1.10) == pytest.approx(0.08)   # prior close above the bar's high

def test_mean_true_range_needs_period() -> None:
    assert mean_true_range([0.04, 0.03], 3) is None
    assert mean_true_range([0.04, 0.03, 0.05], 3) == pytest.approx(0.04)

def test_atr_levels_and_warmup() -> None:
    assert make_bracket(1.05, None, ATR) == "atr_warmup"
    b = make_bracket(1.05, 0.04, ATR)
    assert b.take_profit == pytest.approx(1.13) and b.stop_loss == pytest.approx(1.01)

def test_zero_atr_bracket_is_bad() -> None:
    assert make_bracket(1.05, 0.0, ATR) == "bad_bracket"

def test_atr_stop_at_or_below_zero_is_bad() -> None:
    wide = RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=30.0, atr_period=3)
    assert make_bracket(1.05, 0.04, wide) == "bad_bracket"
