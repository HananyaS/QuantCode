"""Tests for utils/data_cache.py — per-ticker parquet cache with source
provenance tagging and within-range gap detection.

The first block (existing behavior) characterizes utils/data_cache.py's
pre-existing, previously-untested contract before it's extended — a safety
net against regressions from the provenance/gap-detection additions below.
"""
import numpy as np
import pandas as pd
import pytest

from utils.data_cache import find_gaps, load_or_fetch


def _make_ohlcv(dates) -> pd.DataFrame:
    n = len(dates)
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1_000_000.0},
        index=pd.DatetimeIndex(dates),
    )


# ---------------------------------------------------------------------------
# Existing behavior (characterization — pre-existing, previously untested code)
# ---------------------------------------------------------------------------

def test_fetches_when_cache_empty(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=10)
    calls = []

    def fetch_fn(tickers, start, end):
        calls.append(tickers)
        return {"AAPL": _make_ohlcv(dates)}

    result = load_or_fetch(["AAPL"], "2024-01-01", "2024-01-10", fetch_fn, cache_dir=tmp_path)
    assert "AAPL" in result
    assert len(calls) == 1


def test_second_call_uses_cache_no_fetch(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=10)
    calls = []

    def fetch_fn(tickers, start, end):
        calls.append(tickers)
        return {"AAPL": _make_ohlcv(dates)}

    load_or_fetch(["AAPL"], "2024-01-01", "2024-01-10", fetch_fn, cache_dir=tmp_path)
    load_or_fetch(["AAPL"], "2024-01-01", "2024-01-10", fetch_fn, cache_dir=tmp_path)
    assert len(calls) == 1, "second call should be served entirely from cache"


def test_first_fetch_with_duplicate_dates_is_deduplicated_before_caching(tmp_path):
    """A raw fetch can return duplicate index entries (seen in practice from
    yfinance) even on the very first fetch for a ticker, with no existing
    cache to merge against. That path must still dedupe before writing —
    otherwise the cache is corrupted and the next read chokes on a
    non-unique index."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    dup_dates = list(dates) + [dates[2]]  # one duplicated date
    df = _make_ohlcv(dup_dates)

    def fetch_fn(tickers, start, end):
        return {"QQQ": df}

    result = load_or_fetch(["QQQ"], "2024-01-01", "2024-01-05", fetch_fn, cache_dir=tmp_path)
    assert result["QQQ"].index.is_unique
    assert len(result["QQQ"]) == 5

    # The cache file itself must also be clean — re-reading it must not raise.
    reread = load_or_fetch(["QQQ"], "2024-01-01", "2024-01-05", fetch_fn, cache_dir=tmp_path)
    assert reread["QQQ"].index.is_unique


def test_merges_new_data_with_existing_cache(tmp_path):
    dates1 = pd.bdate_range("2024-01-01", periods=5)
    dates2 = pd.bdate_range("2024-01-08", periods=5)

    def fetch_fn_1(tickers, start, end):
        return {"AAPL": _make_ohlcv(dates1)}

    def fetch_fn_2(tickers, start, end):
        return {"AAPL": _make_ohlcv(dates2)}

    load_or_fetch(["AAPL"], "2024-01-01", "2024-01-05", fetch_fn_1, cache_dir=tmp_path)
    result = load_or_fetch(["AAPL"], "2024-01-01", "2024-01-12", fetch_fn_2, cache_dir=tmp_path)
    assert len(result["AAPL"]) == 10


# ---------------------------------------------------------------------------
# Source provenance tagging (new)
# ---------------------------------------------------------------------------

def test_source_name_tags_fetched_rows(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=5)

    def fetch_fn(tickers, start, end):
        return {"AAPL": _make_ohlcv(dates)}

    result = load_or_fetch(
        ["AAPL"], "2024-01-01", "2024-01-05", fetch_fn, cache_dir=tmp_path, source_name="tiingo",
    )
    assert (result["AAPL"]["source"] == "tiingo").all()


def test_no_source_name_omits_source_column(tmp_path):
    """Backward compatibility: omitting source_name must not add a column."""
    dates = pd.bdate_range("2024-01-01", periods=5)

    def fetch_fn(tickers, start, end):
        return {"AAPL": _make_ohlcv(dates)}

    result = load_or_fetch(["AAPL"], "2024-01-01", "2024-01-05", fetch_fn, cache_dir=tmp_path)
    assert "source" not in result["AAPL"].columns


def test_merging_different_sources_preserves_each_rows_source(tmp_path):
    dates1 = pd.bdate_range("2024-01-01", periods=5)
    dates2 = pd.bdate_range("2024-01-08", periods=5)

    def fetch_fn_1(tickers, start, end):
        return {"AAPL": _make_ohlcv(dates1)}

    def fetch_fn_2(tickers, start, end):
        return {"AAPL": _make_ohlcv(dates2)}

    load_or_fetch(["AAPL"], "2024-01-01", "2024-01-05", fetch_fn_1, cache_dir=tmp_path, source_name="tiingo")
    result = load_or_fetch(
        ["AAPL"], "2024-01-01", "2024-01-12", fetch_fn_2, cache_dir=tmp_path, source_name="alpaca",
    )
    df = result["AAPL"]
    assert (df.loc[dates1, "source"] == "tiingo").all()
    assert (df.loc[dates2, "source"] == "alpaca").all()


# ---------------------------------------------------------------------------
# Gap detection (new)
# ---------------------------------------------------------------------------

def test_find_gaps_returns_empty_for_complete_range():
    dates = pd.bdate_range("2024-01-01", periods=10)
    df = _make_ohlcv(dates)
    gaps = find_gaps(df, "2024-01-01", str(dates[-1].date()))
    assert gaps == []


def test_find_gaps_detects_missing_business_days():
    dates = pd.bdate_range("2024-01-01", periods=10)
    df = _make_ohlcv(dates)
    df_with_hole = df.drop(dates[4])  # remove one business day from the middle
    gaps = find_gaps(df_with_hole, "2024-01-01", str(dates[-1].date()))
    assert dates[4] in gaps


def test_find_gaps_empty_dataframe_reports_all_business_days():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    empty.index = pd.DatetimeIndex([])
    gaps = find_gaps(empty, "2024-01-01", "2024-01-05")
    expected = list(pd.bdate_range("2024-01-01", "2024-01-05"))
    assert gaps == expected
