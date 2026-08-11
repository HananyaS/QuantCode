"""Tests for the XBRL share-count / float proxy.

The trap: an XBRL frame is stamped with the period end, but the number is not
public until the periodic report is filed weeks later. Keying on the period end
would let an April decision use a share count the market first saw in May.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.etl.xbrl import (
    FALLBACK_LAG_DAYS,
    assert_no_future_share_counts,
    attach_known_from,
    build_pit_features,
    fetch_frame,
    quarter_frames,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _frame_payload(rows):
    return {"data": [
        {"cik": cik, "entityName": "X", "val": val, "end": end, "accn": "a"}
        for cik, val, end in rows
    ]}


TICKERS = pd.DataFrame({"symbol": ["PENY"], "cik": [999]})


def _filings(acceptances, form="10-Q", symbol="PENY"):
    return pd.DataFrame({
        "symbol": symbol, "form": form,
        "accepted_et": pd.to_datetime(acceptances).tz_localize("America/New_York"),
    })


# ---- Frames ----


def test_quarter_frames_are_instantaneous_labels():
    fr = quarter_frames(2024, 2024)
    assert fr == ["CY2024Q1I", "CY2024Q2I", "CY2024Q3I", "CY2024Q4I"]


def test_frame_parses_values_and_period_end():
    payload = _frame_payload([(999, 5_000_000, "2024-03-31")])
    df = fetch_frame("CY2024Q1I", http_get=lambda *a, **k: FakeResponse(payload))
    assert df.iloc[0]["val"] == 5_000_000
    assert df.iloc[0]["period_end"] == pd.Timestamp("2024-03-31")


def test_missing_frame_returns_empty():
    assert fetch_frame("CY1999Q1I", http_get=lambda *a, **k: FakeResponse({}, 404)).empty


# ---- The point-in-time dating ----


def test_value_is_dated_by_the_filing_that_disclosed_it():
    """Not by the period end -- the market cannot see it then."""
    shares = pd.DataFrame({
        "cik": [999], "val": [5e6],
        "period_end": [pd.Timestamp("2024-03-31")], "period": ["CY2024Q1I"],
    })
    filings = _filings(["2024-05-14 16:30"])
    out = attach_known_from(shares, filings, TICKERS)
    assert out.iloc[0]["known_from"] == pd.Timestamp("2024-05-14 16:30", tz="America/New_York")


def test_fallback_lag_is_used_when_no_filing_is_on_record():
    shares = pd.DataFrame({
        "cik": [999], "val": [5e6],
        "period_end": [pd.Timestamp("2024-03-31")], "period": ["CY2024Q1I"],
    })
    out = attach_known_from(shares, _filings([]).iloc[0:0], TICKERS)
    expected = pd.Timestamp("2024-03-31", tz="America/New_York") + pd.Timedelta(days=FALLBACK_LAG_DAYS)
    assert out.iloc[0]["known_from"] == expected


def test_fallback_is_late_rather_than_early():
    """The conservative direction: delay the fact, never advance it."""
    shares = pd.DataFrame({
        "cik": [999], "val": [5e6],
        "period_end": [pd.Timestamp("2024-03-31")], "period": ["CY2024Q1I"],
    })
    out = attach_known_from(shares, _filings([]).iloc[0:0], TICKERS)
    assert out.iloc[0]["known_from"] > pd.Timestamp("2024-03-31", tz="America/New_York")


def test_only_filings_after_the_period_end_can_disclose_it():
    """A 10-Q filed BEFORE the quarter ended cannot contain that quarter's count."""
    shares = pd.DataFrame({
        "cik": [999], "val": [5e6],
        "period_end": [pd.Timestamp("2024-03-31")], "period": ["CY2024Q1I"],
    })
    filings = _filings(["2024-02-10 16:30", "2024-05-14 16:30"])
    out = attach_known_from(shares, filings, TICKERS)
    assert out.iloc[0]["known_from"] == pd.Timestamp("2024-05-14 16:30", tz="America/New_York")


# ---- Feature construction ----


def _shares(known_from, vals, symbol="PENY"):
    return pd.DataFrame({
        "symbol": symbol, "val": vals,
        "known_from": pd.to_datetime(known_from).tz_localize("America/New_York"),
        "period_end": pd.to_datetime(known_from),
    })


def _cands(dates, symbol="PENY", price=2.0, gap=0.3):
    return pd.DataFrame({
        "symbol": symbol, "date": [pd.Timestamp(d) for d in dates],
        "close_unadj_prev": price, "gap": gap,
    })


def test_a_count_disclosed_after_the_decision_is_invisible():
    sh = _shares(["2024-05-14 16:30"], [5e6])
    same_day = build_pit_features(_cands(["2024-05-14"]), sh)
    next_day = build_pit_features(_cands(["2024-05-15"]), sh)
    assert np.isnan(same_day.iloc[0]["shares_outstanding"]), "an evening filing leaked backwards"
    assert next_day.iloc[0]["shares_outstanding"] == pytest.approx(5e6)


def test_the_most_recent_disclosed_count_is_used():
    sh = _shares(["2024-02-14 16:30", "2024-05-14 16:30"], [4e6, 6e6])
    out = build_pit_features(_cands(["2024-06-03"]), sh)
    assert out.iloc[0]["shares_outstanding"] == pytest.approx(6e6)


def test_share_growth_detects_dilution():
    """Serial dilution is the mechanism this feature exists to catch."""
    sh = _shares(["2024-02-14 16:30", "2024-05-14 16:30"], [4e6, 8e6])
    out = build_pit_features(_cands(["2024-06-03"]), sh)
    assert out.iloc[0]["shares_growth_qoq"] == pytest.approx(1.0)   # doubled


def test_float_dollar_uses_the_previous_unadjusted_close():
    sh = _shares(["2024-05-14 16:30"], [5e6])
    out = build_pit_features(_cands(["2024-06-03"], price=3.0), sh)
    assert out.iloc[0]["float_dollar"] == pytest.approx(15e6)


def test_gap_to_float_is_larger_for_a_smaller_share_count():
    """The same gap on a small float is a stronger signal."""
    small = build_pit_features(_cands(["2024-06-03"]), _shares(["2024-05-14"], [1e6]))
    large = build_pit_features(_cands(["2024-06-03"]), _shares(["2024-05-14"], [1e9]))
    assert small.iloc[0]["gap_to_float"] > large.iloc[0]["gap_to_float"]


def test_uncovered_symbols_are_flagged_not_silently_zero():
    sh = _shares(["2024-05-14"], [5e6], symbol="PENY")
    cands = pd.DataFrame({
        "symbol": ["PENY", "GONE"], "date": [pd.Timestamp("2024-06-03")] * 2,
        "close_unadj_prev": 2.0, "gap": 0.3,
    })
    out = build_pit_features(cands, sh).set_index("symbol")
    assert out.loc["PENY", "shares_has_coverage"] == True
    assert out.loc["GONE", "shares_has_coverage"] == False
    assert np.isnan(out.loc["GONE", "shares_outstanding"])


def test_no_data_at_all_yields_nulls_with_coverage_false():
    out = build_pit_features(_cands(["2024-06-03"]), pd.DataFrame())
    assert np.isnan(out.iloc[0]["shares_outstanding"])
    assert out.iloc[0]["shares_has_coverage"] == False


def test_row_count_is_preserved():
    sh = _shares(["2024-05-14"], [5e6])
    cands = _cands(["2024-06-03", "2024-06-04", "2024-06-05"])
    assert len(build_pit_features(cands, sh)) == 3


# ---- Tripwire ----


def test_tripwire_catches_a_future_share_count():
    sh = _shares(["2024-06-10 16:30"], [5e6])
    with pytest.raises(AssertionError, match="became public at or after"):
        assert_no_future_share_counts(sh, pd.Timestamp("2024-06-03 09:35", tz="America/New_York"))


def test_tripwire_passes_on_disclosed_counts():
    sh = _shares(["2024-05-14 16:30"], [5e6])
    assert_no_future_share_counts(sh, pd.Timestamp("2024-06-03 09:35", tz="America/New_York"))
