"""Tests for scripts/validate_data_sources.py — cross-provider price divergence check.

The CLI's `main()` is a thin wrapper (like main_multi.py / run_live.py) and
isn't unit tested directly. `compare_sources` is the testable core — it
accepts an injectable `sources` dict of fetch functions, so no live network
calls happen in tests.
"""
import pandas as pd
import pytest

from scripts.validate_data_sources import compare_sources


def _df(dates, closes):
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1_000_000.0},
        index=pd.DatetimeIndex(dates),
    )


def test_identical_sources_zero_divergence():
    dates = pd.bdate_range("2024-01-01", periods=5)
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    sources = {
        "a": lambda t, s, e: {"AAPL": _df(dates, closes)},
        "b": lambda t, s, e: {"AAPL": _df(dates, closes)},
    }
    report = compare_sources(["AAPL"], "2024-01-01", "2024-01-05", sources=sources)
    row = report.iloc[0]
    assert row["max_rel_diff"] == 0.0
    assert bool(row["flagged"]) is False


def test_divergent_sources_flagged():
    dates = pd.bdate_range("2024-01-01", periods=5)
    closes_a = [100.0] * 5
    closes_b = [90.0] * 5  # 10% off — well above the 1% threshold
    sources = {
        "a": lambda t, s, e: {"AAPL": _df(dates, closes_a)},
        "b": lambda t, s, e: {"AAPL": _df(dates, closes_b)},
    }
    report = compare_sources(["AAPL"], "2024-01-01", "2024-01-05", sources=sources)
    row = report.iloc[0]
    assert bool(row["flagged"]) is True
    assert row["max_rel_diff"] == pytest.approx(0.10, abs=1e-6)


def test_small_divergence_not_flagged():
    dates = pd.bdate_range("2024-01-01", periods=5)
    closes_a = [100.0] * 5
    closes_b = [100.05] * 5  # 0.05% off — below threshold
    sources = {
        "a": lambda t, s, e: {"AAPL": _df(dates, closes_a)},
        "b": lambda t, s, e: {"AAPL": _df(dates, closes_b)},
    }
    report = compare_sources(["AAPL"], "2024-01-01", "2024-01-05", sources=sources)
    assert bool(report.iloc[0]["flagged"]) is False


def test_missing_source_data_skipped_not_raised():
    dates = pd.bdate_range("2024-01-01", periods=5)
    sources = {
        "a": lambda t, s, e: {"AAPL": _df(dates, [100.0] * 5)},
        "b": lambda t, s, e: {},  # no data for AAPL at all
    }
    report = compare_sources(["AAPL"], "2024-01-01", "2024-01-05", sources=sources)
    assert len(report) == 0


def test_no_overlapping_dates_skipped():
    dates_a = pd.bdate_range("2024-01-01", periods=5)
    dates_b = pd.bdate_range("2025-01-01", periods=5)
    sources = {
        "a": lambda t, s, e: {"AAPL": _df(dates_a, [100.0] * 5)},
        "b": lambda t, s, e: {"AAPL": _df(dates_b, [100.0] * 5)},
    }
    report = compare_sources(["AAPL"], "2024-01-01", "2024-01-05", sources=sources)
    assert len(report) == 0


def test_report_columns():
    dates = pd.bdate_range("2024-01-01", periods=5)
    sources = {
        "a": lambda t, s, e: {"AAPL": _df(dates, [100.0] * 5)},
        "b": lambda t, s, e: {"AAPL": _df(dates, [100.0] * 5)},
    }
    report = compare_sources(["AAPL"], "2024-01-01", "2024-01-05", sources=sources)
    for col in ("ticker", "source_a", "source_b", "n_overlap_days", "max_rel_diff", "mean_rel_diff", "flagged"):
        assert col in report.columns


def test_multiple_sources_compares_every_pair():
    dates = pd.bdate_range("2024-01-01", periods=5)
    sources = {
        "a": lambda t, s, e: {"AAPL": _df(dates, [100.0] * 5)},
        "b": lambda t, s, e: {"AAPL": _df(dates, [100.0] * 5)},
        "c": lambda t, s, e: {"AAPL": _df(dates, [100.0] * 5)},
    }
    report = compare_sources(["AAPL"], "2024-01-01", "2024-01-05", sources=sources)
    assert len(report) == 3  # (a,b), (a,c), (b,c)
