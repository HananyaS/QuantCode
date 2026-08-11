"""Tests for FINRA short-volume features.

The failure guarded against here is subtle and flattering: FINRA publishes
day-t short volume after day t closes, and same-day short volume correlates
with same-day returns. A feature that leaked it would look like an excellent
predictor rather than like a bug.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from research.penny.etl.finra import (
    HISTORY_START,
    assert_short_volume_is_lagged,
    build_lagged_features,
    fetch_day,
    fetch_range,
)


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


_BODY = (
    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    "20240305|PENY|400|0|1000|Q\n"
    "20240305|BIGCO|100|0|1000|Q\n"
)


# ---- Fetching ----


def test_day_parses_and_renames():
    df = fetch_day(date(2024, 3, 5), http_get=lambda *a, **k: FakeResponse(_BODY))
    assert set(df.columns) >= {"date", "symbol", "short_volume", "total_volume"}
    assert len(df) == 2


def test_dates_before_history_start_are_not_requested():
    """Consolidated files begin 2018-08-01; earlier dates must not be fetched."""
    called = []

    def spy(*a, **k):
        called.append(1)
        return FakeResponse(_BODY)

    assert fetch_day(date(2017, 5, 1), http_get=spy) is None
    assert not called, "requested a date before the file series begins"


def test_missing_file_returns_none():
    assert fetch_day(date(2024, 3, 5), http_get=lambda *a, **k: FakeResponse("", 404)) is None


def test_range_skips_weekends_and_filters_symbols():
    out = fetch_range(
        date(2024, 3, 4), date(2024, 3, 8), symbols=["PENY"],
        http_get=lambda *a, **k: FakeResponse(_BODY),
    )
    assert set(out["symbol"]) == {"PENY"}, "symbol filter not applied"
    assert len(out) == 5, "expected one row per weekday"


# ---- The lag: the core guarantee ----


def _raw(dates, ratios, symbol="PENY"):
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d in dates],
            "symbol": symbol,
            "short_volume": [r * 1000 for r in ratios],
            "short_exempt_volume": 0,
            "total_volume": 1000,
        }
    )


def test_feature_uses_the_previous_session_not_todays():
    """The whole point. Day t's file does not exist at 09:35 on day t."""
    raw = _raw(["2024-03-04", "2024-03-05"], [0.20, 0.90])
    cands = pd.DataFrame({"symbol": ["PENY"], "date": [pd.Timestamp("2024-03-05")]})
    out = build_lagged_features(cands, raw)
    assert out.iloc[0]["short_volume_ratio"] == pytest.approx(0.20), (
        "feature took today's ratio (0.90) instead of yesterday's (0.20)"
    )


def test_first_observed_session_has_no_prior_ratio():
    raw = _raw(["2024-03-05"], [0.40])
    cands = pd.DataFrame({"symbol": ["PENY"], "date": [pd.Timestamp("2024-03-05")]})
    assert np.isnan(build_lagged_features(cands, raw).iloc[0]["short_volume_ratio"])


def test_lag_tripwire_catches_a_same_day_leak():
    raw = _raw(["2024-03-04", "2024-03-05"], [0.20, 0.90])
    leaked = pd.DataFrame(
        {"symbol": ["PENY"], "date": [pd.Timestamp("2024-03-05")], "short_volume_ratio": [0.90]}
    )
    with pytest.raises(AssertionError, match="publishes after the close"):
        assert_short_volume_is_lagged(leaked, raw, "PENY", pd.Timestamp("2024-03-05"))


def test_lag_tripwire_passes_on_correctly_lagged_features():
    raw = _raw(["2024-03-04", "2024-03-05"], [0.20, 0.90])
    cands = pd.DataFrame({"symbol": ["PENY"], "date": [pd.Timestamp("2024-03-05")]})
    out = build_lagged_features(cands, raw)
    assert_short_volume_is_lagged(out, raw, "PENY", pd.Timestamp("2024-03-05"))


def test_symbols_do_not_bleed_into_each_other():
    raw = pd.concat([
        _raw(["2024-03-04", "2024-03-05"], [0.10, 0.10], symbol="AAA"),
        _raw(["2024-03-04", "2024-03-05"], [0.90, 0.90], symbol="BBB"),
    ])
    cands = pd.DataFrame(
        {"symbol": ["AAA", "BBB"], "date": [pd.Timestamp("2024-03-05")] * 2}
    )
    out = build_lagged_features(cands, raw).set_index("symbol")
    assert out.loc["AAA", "short_volume_ratio"] == pytest.approx(0.10)
    assert out.loc["BBB", "short_volume_ratio"] == pytest.approx(0.90)


# ---- Derived features ----


def test_change_feature_compares_yesterday_to_its_own_trailing_mean():
    dates = [f"2024-03-{d:02d}" for d in (1, 4, 5, 6, 7, 8, 11)]
    raw = _raw(dates, [0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2])
    cands = pd.DataFrame({"symbol": ["PENY"], "date": [pd.Timestamp("2024-03-11")]})
    out = build_lagged_features(cands, raw, windows=(5,))
    # Yesterday (03-08) spiked to 0.8 against a trailing mean near 0.2.
    assert out.iloc[0]["short_volume_ratio"] == pytest.approx(0.8)
    assert out.iloc[0]["short_ratio_chg_5d"] > 0.4


def test_exempt_volume_is_included_in_the_ratio():
    raw = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")],
            "symbol": "PENY", "short_volume": [300, 300],
            "short_exempt_volume": [200, 200], "total_volume": [1000, 1000],
        }
    )
    cands = pd.DataFrame({"symbol": ["PENY"], "date": [pd.Timestamp("2024-03-05")]})
    assert build_lagged_features(cands, raw).iloc[0]["short_volume_ratio"] == pytest.approx(0.5)


def test_zero_total_volume_yields_nan_not_infinity():
    raw = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")],
            "symbol": "PENY", "short_volume": [0, 0],
            "short_exempt_volume": [0, 0], "total_volume": [0, 1000],
        }
    )
    cands = pd.DataFrame({"symbol": ["PENY"], "date": [pd.Timestamp("2024-03-05")]})
    assert np.isnan(build_lagged_features(cands, raw).iloc[0]["short_volume_ratio"])


# ---- Coverage honesty ----


def test_uncovered_symbols_are_flagged_rather_than_silently_null():
    raw = _raw(["2024-03-04", "2024-03-05"], [0.2, 0.2], symbol="PENY")
    cands = pd.DataFrame(
        {"symbol": ["PENY", "GONE"], "date": [pd.Timestamp("2024-03-05")] * 2}
    )
    out = build_lagged_features(cands, raw).set_index("symbol")
    assert out.loc["PENY", "short_has_coverage"] == True
    assert out.loc["GONE", "short_has_coverage"] == False


def test_no_data_at_all_yields_nulls_with_coverage_false():
    cands = pd.DataFrame({"symbol": ["PENY"], "date": [pd.Timestamp("2024-03-05")]})
    out = build_lagged_features(cands, pd.DataFrame())
    assert np.isnan(out.iloc[0]["short_volume_ratio"])
    assert out.iloc[0]["short_has_coverage"] == False


def test_row_count_is_preserved():
    """The merge must not duplicate or drop candidate rows."""
    raw = _raw(["2024-03-04", "2024-03-05"], [0.2, 0.3])
    cands = pd.DataFrame(
        {"symbol": ["PENY"] * 3, "date": pd.to_datetime(["2024-03-05"] * 3)}
    )
    assert len(build_lagged_features(cands, raw)) == 3
