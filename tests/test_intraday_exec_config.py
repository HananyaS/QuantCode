"""ExecConfig defaults/validation, ET conversion and client order ids (plan Task 1)."""
from __future__ import annotations

from datetime import date, time

import pytest

from research.intraday.contracts.fixtures import T_DEV
from research.intraday.exec.config import ExecConfig, to_et
from research.intraday.exec.ids import client_order_id


def test_defaults_match_design_table() -> None:
    c = ExecConfig()
    assert (c.quote_size_frac, c.quote_max_age_min, c.bar_participation_cap) == (1.0, 5, 0.02)
    assert (c.entry_max_wait_min, c.eod_et, c.default_half_spread_bps) == (2, time(16, 1), 100.0)
    assert (c.impact_coefficient, c.levels_penalty) == (1.0, 1.0)


@pytest.mark.parametrize("kw", [dict(quote_size_frac=1), dict(quote_size_frac=0.0),
    dict(bar_participation_cap=1.5), dict(quote_max_age_min=5.0), dict(entry_max_wait_min=0),
    dict(default_half_spread_bps=0.0), dict(levels_penalty=-1.0), dict(quote_max_age_min=True)])
def test_invalid_config_rejected(kw) -> None:
    with pytest.raises(ValueError):
        ExecConfig(**kw)


def test_to_et_converts_t_dev() -> None:
    assert to_et(T_DEV).time() == time(9, 31) and to_et(T_DEV).date() == date(2024, 3, 4)


def test_client_order_id_deterministic_and_distinct_per_leg() -> None:
    assert client_order_id("20240304-TEST-1", "entry") == "20240304-TEST-1"
    assert client_order_id("20240304-TEST-1", "exit") == "20240304-TEST-1-exit"
    assert client_order_id("x", "exit") == client_order_id("x", "exit")
    with pytest.raises(ValueError):
        client_order_id("a" * 124, "exit")
