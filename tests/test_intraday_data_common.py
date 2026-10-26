"""Shared data-slice pieces: ET clock, holdout guard, paths, traded basis, cache readers."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from research.intraday.contracts.fixtures import T_DEV
from research.intraday.data.clock import et_date, et_hhmm, et_to_utc, minute_index_to_utc
from research.intraday.data.errors import SealedHoldoutError
from research.intraday.data.holdout import check_holdout
from research.intraday.data.paths import (
    daily_file, minute_file, premarket_file, requested_file, volume_daily_file,
)


def test_et_to_utc_handles_dst() -> None:
    assert et_to_utc(date(2024, 3, 4), "09:25") == datetime(2024, 3, 4, 14, 25, tzinfo=timezone.utc)
    assert et_to_utc(date(2024, 7, 1), "09:25") == datetime(2024, 7, 1, 13, 25, tzinfo=timezone.utc)
    assert et_to_utc(date(2024, 3, 4), "09:25").tzinfo is timezone.utc


def test_et_date_and_hhmm_round_trip() -> None:
    assert et_date(T_DEV) == date(2024, 3, 4) and et_hhmm(T_DEV) == "09:31"
    assert et_date(datetime(2024, 3, 5, 3, 0, tzinfo=timezone.utc)) == date(2024, 3, 4)


def test_minute_index_to_utc_maps_each_distinct_minute() -> None:
    m = minute_index_to_utc(date(2024, 3, 4), ["09:30", "09:31", "09:30"])
    assert list(m) == ["09:30", "09:31"] and m["09:31"] == datetime(2024, 3, 4, 14, 31, tzinfo=timezone.utc)


def test_check_holdout_boundary() -> None:
    check_holdout(date(2025, 8, 6))
    with pytest.raises(SealedHoldoutError):
        check_holdout(date(2025, 8, 7))
    check_holdout(date(2025, 8, 7), unseal=True)


def test_paths_layout() -> None:
    root = Path("R")
    assert minute_file(root, date(2024, 3, 4)) == Path("R/minute/dt=2024-03-04/minute.parquet")
    assert requested_file(root, date(2024, 3, 4)) == Path("R/minute/dt=2024-03-04/requested.json")
    assert premarket_file(root, date(2024, 3, 4)) == Path("R/premarket/dt=2024-03-04/premarket.parquet")
    assert daily_file(root, 2024) == Path("R/daily/bars_2024.parquet")
    assert volume_daily_file(Path("V"), 2024) == Path("V/bars_2024.parquet")


# ---- Task 2: traded-basis arithmetic ----------------------------------------
import math  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.intraday.data.basis import (  # noqa: E402
    factor_consistent, price_factor, split_flags, to_traded_basis, volume_factor,
)


def test_price_factor_is_raw_over_adjusted() -> None:
    f = price_factor(pd.Series({"BRSP": 6.60, "Z": 1.0}), pd.Series({"BRSP": 4.907, "Z": 0.0}))
    assert f["BRSP"] == pytest.approx(1.345, rel=1e-3) and np.isnan(f["Z"])


def test_dividend_factor_leaves_share_volume_unchanged() -> None:
    bars = pd.DataFrame({"symbol": ["BRSP"], "open": [5.0], "high": [5.1], "low": [4.9],
                         "close": [5.0], "vwap": [5.0], "volume": [1_034_853.0]})
    s = volume_factor(pd.Series({"BRSP": 1_034_853.0}), pd.Series({"BRSP": 1_034_853.0}))
    out = to_traded_basis(bars, pd.Series({"BRSP": 1.345}), s)
    assert out["volume"].iloc[0] == 1_034_853.0
    assert out["close"].iloc[0] == pytest.approx(5.0 * 1.345)


def test_reverse_split_scales_volume_by_split_factor() -> None:
    s = volume_factor(pd.Series({"ENVB": 7798.0}), pd.Series({"ENVB": 1_403_699.0}))
    bars = pd.DataFrame({"symbol": ["ENVB"], "open": [279.0], "high": [279.0], "low": [279.0],
                         "close": [279.0], "vwap": [279.0], "volume": [7798.0]})
    out = to_traded_basis(bars, pd.Series({"ENVB": 1.55 / 279.0}), s)
    assert out["volume"].iloc[0] == pytest.approx(1_403_699.0)
    assert out["close"].iloc[0] == pytest.approx(1.55)


def test_split_flags_threshold_and_missing_session_row() -> None:
    f_prev = pd.Series({"SPLIT": 0.1, "DIV": 0.97, "NOROW": 0.1, "FWD": 1.0})
    f_d = pd.Series({"SPLIT": 1.0, "DIV": 1.0, "FWD": 0.5})
    flags = split_flags(f_prev, f_d, math.log(1.25))
    assert flags.to_dict() == {"SPLIT": True, "DIV": False, "NOROW": False, "FWD": True}


def test_factor_consistent_nan_and_missing_are_false() -> None:
    ok = factor_consistent(pd.Series({"A": 1.01, "B": np.nan, "C": 1.0}), pd.Series({"A": 1.0, "B": 1.0}), 0.02)
    assert ok.to_dict() == {"A": True, "B": False, "C": False}


def test_to_traded_basis_missing_factor_raises() -> None:
    bars = pd.DataFrame({"symbol": ["X"], "open": [1.0], "high": [1.0], "low": [1.0],
                         "close": [1.0], "vwap": [1.0], "volume": [1.0]})
    with pytest.raises(ValueError, match="X"):
        to_traded_basis(bars, pd.Series(dtype=float), pd.Series({"X": 1.0}))
