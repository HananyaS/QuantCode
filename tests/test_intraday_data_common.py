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
