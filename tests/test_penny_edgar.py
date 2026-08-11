"""Tests for EDGAR point-in-time filing features.

The failure mode these guard against is invisible in results: a filing accepted
at 18:00 informing a 09:35 decision earlier the same day would improve every
metric and look like signal.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.etl.edgar import (
    EdgarConfig,
    assert_no_future_filings,
    build_pit_features,
    fetch_ticker_cik_map,
    parse_filings,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _submissions(forms, items, dates, acceptances):
    return {
        "filings": {
            "recent": {
                "form": forms, "items": items,
                "filingDate": dates, "acceptanceDateTime": acceptances,
            }
        }
    }


# ---- Ticker map ----


def test_ticker_map_parses_and_uppercases():
    payload = {
        "0": {"ticker": "aapl", "cik_str": 320193, "title": "Apple Inc."},
        "1": {"ticker": "PENY", "cik_str": 999, "title": "Tiny Corp"},
    }
    df = fetch_ticker_cik_map(http_get=lambda *a, **k: FakeResponse(payload))
    assert set(df["symbol"]) == {"AAPL", "PENY"}
    assert df.set_index("symbol").loc["AAPL", "cik"] == 320193


# ---- Filing parsing ----


def test_acceptance_datetime_is_converted_to_eastern():
    payload = _submissions(
        ["8-K"], ["3.01"], ["2024-03-05"], ["2024-03-05T18:30:00.000Z"],
    )
    df = parse_filings(payload, "PENY", 999)
    assert len(df) == 1
    assert str(df.iloc[0]["accepted_et"].tz) == "America/New_York"


def test_missing_acceptance_falls_back_to_end_of_filing_day():
    """The conservative direction: delay the fact, never advance it."""
    payload = _submissions(["S-3"], [""], ["2024-03-05"], [None])
    df = parse_filings(payload, "PENY", 999)
    accepted = df.iloc[0]["accepted_et"]
    assert accepted.date() == pd.Timestamp("2024-03-05").date()
    assert accepted.hour == 23


def test_empty_submissions_yields_empty_frame():
    assert parse_filings({"filings": {"recent": {}}}, "X", 1).empty


# ---- The point-in-time boundary ----


def _candidates(dates, symbol="PENY"):
    return pd.DataFrame({"symbol": symbol, "date": [pd.Timestamp(d) for d in dates]})


def test_filing_accepted_after_the_decision_is_invisible():
    """The core guarantee.

    An 8-K accepted at 13:00 ET must not inform a decision made at 09:35
    that morning -- it had not happened yet.
    """
    filings = parse_filings(
        _submissions(["8-K"], ["3.01"], ["2024-03-05"], ["2024-03-05T18:00:00.000Z"]),
        "PENY", 999,
    )
    same_day = build_pit_features(_candidates(["2024-03-05"]), filings)
    next_day = build_pit_features(_candidates(["2024-03-06"]), filings)

    assert same_day.iloc[0]["edgar_deficiency_180d"] == False, "an evening filing leaked backwards"
    assert next_day.iloc[0]["edgar_deficiency_180d"] == True


def test_filing_accepted_before_the_open_is_visible_same_day():
    filings = parse_filings(
        _submissions(["8-K"], ["3.01"], ["2024-03-05"], ["2024-03-05T12:00:00.000Z"]),
        "PENY", 999,
    )  # 12:00 UTC = 07:00 ET, before the 09:35 decision
    out = build_pit_features(_candidates(["2024-03-05"]), filings)
    assert out.iloc[0]["edgar_deficiency_180d"] == True


def test_days_since_shelf_is_measured_from_acceptance():
    filings = parse_filings(
        _submissions(["S-3"], [""], ["2024-01-05"], ["2024-01-05T14:00:00.000Z"]),
        "PENY", 999,
    )
    out = build_pit_features(_candidates(["2024-03-05"]), filings)
    assert out.iloc[0]["edgar_days_since_shelf"] == pytest.approx(60, abs=1)


def test_shelf_and_takedown_are_tracked_separately():
    """A shelf registers stock; a 424B takedown actually sells it."""
    filings = parse_filings(
        _submissions(
            ["S-3", "424B5"], ["", ""],
            ["2024-01-05", "2024-02-20"],
            ["2024-01-05T14:00:00.000Z", "2024-02-20T14:00:00.000Z"],
        ),
        "PENY", 999,
    )
    out = build_pit_features(_candidates(["2024-03-05"]), filings).iloc[0]
    assert out["edgar_days_since_shelf"] > out["edgar_days_since_takedown"]


def test_lookback_window_is_respected():
    filings = parse_filings(
        _submissions(["8-K"], ["3.01"], ["2023-01-05"], ["2023-01-05T14:00:00.000Z"]),
        "PENY", 999,
    )
    out = build_pit_features(
        _candidates(["2024-03-05"]), filings, EdgarConfig(lookback_days=180)
    )
    assert out.iloc[0]["edgar_deficiency_180d"] == False, "a filing 14 months old is not recent"


def test_item_matching_handles_comma_separated_lists():
    filings = parse_filings(
        _submissions(["8-K"], ["1.01,3.02,9.01"], ["2024-03-01"], ["2024-03-01T14:00:00.000Z"]),
        "PENY", 999,
    )
    out = build_pit_features(_candidates(["2024-03-05"]), filings).iloc[0]
    assert out["edgar_dilutive_8k_180d"] == True


def test_item_matching_does_not_match_substrings():
    """Item 13.01 must not register as 3.01."""
    filings = parse_filings(
        _submissions(["8-K"], ["13.01"], ["2024-03-01"], ["2024-03-01T14:00:00.000Z"]),
        "PENY", 999,
    )
    out = build_pit_features(_candidates(["2024-03-05"]), filings).iloc[0]
    assert out["edgar_deficiency_180d"] == False


# ---- Coverage honesty ----


def test_symbols_without_filings_are_flagged_not_silently_false():
    """A delisted symbol with no CIK must be distinguishable from a clean one."""
    filings = parse_filings(
        _submissions(["S-3"], [""], ["2024-01-05"], ["2024-01-05T14:00:00.000Z"]),
        "PENY", 999,
    )
    cands = pd.DataFrame(
        {"symbol": ["PENY", "GONE"], "date": [pd.Timestamp("2024-03-05")] * 2}
    )
    out = build_pit_features(cands, filings).set_index("symbol")
    assert out.loc["PENY", "edgar_has_coverage"] == True
    assert out.loc["GONE", "edgar_has_coverage"] == False
    assert np.isnan(out.loc["GONE", "edgar_days_since_shelf"])


def test_no_filings_at_all_yields_null_features_with_coverage_false():
    out = build_pit_features(_candidates(["2024-03-05"]), pd.DataFrame())
    assert out.iloc[0]["edgar_has_coverage"] == False


# ---- Tripwire ----


def test_tripwire_catches_a_future_filing():
    filings = parse_filings(
        _submissions(["8-K"], ["3.01"], ["2024-03-05"], ["2024-03-05T18:00:00.000Z"]),
        "PENY", 999,
    )
    decision = pd.Timestamp("2024-03-05 09:35", tz="America/New_York")
    with pytest.raises(AssertionError, match="accepted at or after"):
        assert_no_future_filings(filings, decision)


def test_tripwire_passes_on_past_filings():
    filings = parse_filings(
        _submissions(["8-K"], ["3.01"], ["2024-03-01"], ["2024-03-01T14:00:00.000Z"]),
        "PENY", 999,
    )
    decision = pd.Timestamp("2024-03-05 09:35", tz="America/New_York")
    assert_no_future_filings(filings, decision)
