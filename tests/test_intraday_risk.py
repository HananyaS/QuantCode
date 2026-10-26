from __future__ import annotations

from datetime import time

import pytest

from research.intraday.risk.config import RiskConfig


def test_last_entry_must_precede_flatten() -> None:
    with pytest.raises(ValueError, match="last_entry_et"):
        RiskConfig(last_entry_et=time(15, 55), flatten_et=time(15, 55))
    with pytest.raises(ValueError, match="last_entry_et"):
        RiskConfig(last_entry_et=time(15, 56), flatten_et=time(15, 55))
    RiskConfig(last_entry_et=time(15, 54), flatten_et=time(15, 55))  # ok
