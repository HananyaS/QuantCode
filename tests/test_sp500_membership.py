"""Tests for utils/sp500_membership.py — point-in-time S&P 500 constituent lookup.

Uses a small hand-written fixture CSV matching the real fja05680/sp500
schema (columns: date, tickers — a quoted comma-delimited string, where
symbols may carry a synthetic "-YYYYMM" stint-disambiguation suffix, e.g.
"AAL-199702", which is not part of the real tradeable ticker and must be
stripped). No live network calls in tests.
"""
import pandas as pd
import pytest

from utils.sp500_membership import (
    download_membership_csv,
    get_all_historical_tickers,
    get_constituents_on,
)

_FIXTURE_ROWS = [
    # date, tickers (raw dataset format, some with -YYYYMM suffix)
    ("1996-01-02", "AAPL,AAL-199702,MSFT"),
    ("1997-03-01", "AAPL,MSFT,GOOG"),  # AAL dropped out of the index here
    ("2000-06-15", "AAPL,MSFT,GOOG,TMC-200006"),
]


@pytest.fixture
def fixture_csv(tmp_path):
    path = tmp_path / "sp500_membership.csv"
    df = pd.DataFrame(_FIXTURE_ROWS, columns=["date", "tickers"])
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Ticker suffix stripping
# ---------------------------------------------------------------------------

def test_suffix_stripped_from_returned_tickers(fixture_csv):
    tickers = get_constituents_on("1996-01-02", path=fixture_csv)
    assert "AAL" in tickers
    assert "AAL-199702" not in tickers


def test_dotted_share_class_symbols_unaffected(tmp_path):
    path = tmp_path / "sp500_membership.csv"
    df = pd.DataFrame([("2020-01-01", "BF.B,RDS.A,AZA.A-200106")], columns=["date", "tickers"])
    df.to_csv(path, index=False)
    tickers = get_constituents_on("2020-01-01", path=path)
    assert "BF.B" in tickers
    assert "RDS.A" in tickers
    assert "AZA.A" in tickers


# ---------------------------------------------------------------------------
# get_constituents_on — point-in-time snapshot
# ---------------------------------------------------------------------------

def test_exact_snapshot_date(fixture_csv):
    tickers = get_constituents_on("1997-03-01", path=fixture_csv)
    assert set(tickers) == {"AAPL", "MSFT", "GOOG"}


def test_date_between_snapshots_uses_most_recent_prior(fixture_csv):
    """A date with no exact row should use the last known membership carried forward."""
    tickers = get_constituents_on("1998-01-01", path=fixture_csv)
    assert set(tickers) == {"AAPL", "MSFT", "GOOG"}


def test_date_before_first_snapshot_raises(fixture_csv):
    with pytest.raises(AssertionError, match="No membership data"):
        get_constituents_on("1990-01-01", path=fixture_csv)


def test_missing_cache_file_raises(tmp_path):
    with pytest.raises(AssertionError, match="not found"):
        get_constituents_on("2000-01-01", path=tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# get_all_historical_tickers — download universe for a window
# ---------------------------------------------------------------------------

def test_union_includes_delisted_ticker_within_window(fixture_csv):
    tickers = get_all_historical_tickers("1996-01-01", "1997-06-01", path=fixture_csv)
    assert "AAL" in tickers, "AAL was a constituent within the window and must be included"


def test_union_excludes_ticker_outside_window(fixture_csv):
    tickers = get_all_historical_tickers("1998-01-01", "1999-01-01", path=fixture_csv)
    assert "AAL" not in tickers


def test_union_includes_snapshot_immediately_before_window_start(fixture_csv):
    """Membership as of just before the window start should carry into the window."""
    tickers = get_all_historical_tickers("1996-06-01", "1996-12-01", path=fixture_csv)
    assert "AAL" in tickers  # still a member per the 1996-01-02 snapshot, no change row until 1997-03-01


def test_union_sorted_and_deduped(fixture_csv):
    tickers = get_all_historical_tickers("1996-01-01", "2001-01-01", path=fixture_csv)
    assert tickers == sorted(set(tickers))


# ---------------------------------------------------------------------------
# download_membership_csv
# ---------------------------------------------------------------------------

def test_download_writes_response_content_to_dest(tmp_path):
    dest = tmp_path / "sub" / "sp500_membership.csv"

    class FakeResponse:
        content = b"date,tickers\n2020-01-01,AAPL,MSFT\n"
        def raise_for_status(self):
            pass

    result_path = download_membership_csv(dest=dest, url="http://example.com/x.csv",
                                            http_get=lambda url, timeout=None: FakeResponse())
    assert result_path == dest
    assert dest.exists()
    assert dest.read_bytes() == FakeResponse.content
