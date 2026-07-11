"""Tests for utils/tiingo_loader.py — deep historical OHLCV backfill via Tiingo.

All tests inject a fake `http_get` — no live network calls, matching the
project's existing convention (see tests for utils/alpaca_loader-adjacent code).
"""
import pandas as pd
import pytest

from utils.tiingo_loader import fetch_bars


class FakeResponse:
    def __init__(self, json_data, status_ok=True):
        self._json = json_data
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise ConnectionError("simulated HTTP error")

    def json(self):
        return self._json


_SAMPLE_RECORD = {
    "date": "2024-01-02T00:00:00.000Z",
    "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1_000_000,
    "adjOpen": 98.0, "adjHigh": 103.0, "adjLow": 97.0, "adjClose": 101.0, "adjVolume": 990_000,
    "divCash": 0.0, "splitFactor": 1.0,
}


def _make_http_get(per_ticker: dict):
    """per_ticker: Dict[ticker, list-of-records or Exception]."""
    calls = []

    def http_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        ticker = url.rsplit("/", 2)[-2]  # .../daily/{ticker}/prices
        payload = per_ticker.get(ticker)
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload if payload is not None else [])

    http_get.calls = calls
    return http_get


# ---------------------------------------------------------------------------
# Schema / adjustment convention
# ---------------------------------------------------------------------------

def test_returns_dict_with_adjusted_ohlcv_schema():
    http_get = _make_http_get({"aapl": [_SAMPLE_RECORD]})
    result = fetch_bars(["AAPL"], "2024-01-01", "2024-01-05", api_key="tok", pace=False, http_get=http_get)
    df = result["AAPL"]
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_uses_adjusted_fields_not_raw():
    """The raw close (103.0) must NOT appear; the adjusted close (101.0) must."""
    http_get = _make_http_get({"aapl": [_SAMPLE_RECORD]})
    result = fetch_bars(["AAPL"], "2024-01-01", "2024-01-05", api_key="tok", pace=False, http_get=http_get)
    assert result["AAPL"]["Close"].iloc[0] == 101.0


def test_index_is_tz_naive():
    http_get = _make_http_get({"aapl": [_SAMPLE_RECORD]})
    result = fetch_bars(["AAPL"], "2024-01-01", "2024-01-05", api_key="tok", pace=False, http_get=http_get)
    assert result["AAPL"].index.tz is None


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def test_request_includes_token_and_date_range():
    http_get = _make_http_get({"aapl": [_SAMPLE_RECORD]})
    fetch_bars(["AAPL"], "2024-01-01", "2024-01-05", api_key="secret-tok", pace=False, http_get=http_get)
    params = http_get.calls[0]["params"]
    assert params["token"] == "secret-tok"
    assert params["startDate"] == "2024-01-01"
    assert params["endDate"] == "2024-01-05"


def test_ticker_lowercased_in_url():
    http_get = _make_http_get({"aapl": [_SAMPLE_RECORD]})
    fetch_bars(["AAPL"], "2024-01-01", "2024-01-05", api_key="tok", pace=False, http_get=http_get)
    assert "/aapl/" in http_get.calls[0]["url"]


# ---------------------------------------------------------------------------
# Error handling — never raises, absent tickers are logged and skipped
# ---------------------------------------------------------------------------

def test_empty_response_ticker_excluded_from_result():
    http_get = _make_http_get({"aapl": []})
    result = fetch_bars(["AAPL"], "2024-01-01", "2024-01-05", api_key="tok", pace=False, http_get=http_get)
    assert "AAPL" not in result


def test_failed_ticker_does_not_raise_and_is_excluded():
    http_get = _make_http_get({"aapl": ConnectionError("network down")})
    result = fetch_bars(["AAPL"], "2024-01-01", "2024-01-05", api_key="tok", pace=False, http_get=http_get)
    assert result == {}


def test_partial_failure_other_tickers_still_returned():
    http_get = _make_http_get({"aapl": ConnectionError("boom"), "msft": [_SAMPLE_RECORD]})
    result = fetch_bars(["AAPL", "MSFT"], "2024-01-01", "2024-01-05", api_key="tok", pace=False, http_get=http_get)
    assert "AAPL" not in result
    assert "MSFT" in result


# ---------------------------------------------------------------------------
# Rate-limit pacing
# ---------------------------------------------------------------------------

def test_pace_true_sleeps_between_requests(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("utils.tiingo_loader.time.sleep", lambda s: sleep_calls.append(s))
    http_get = _make_http_get({"aapl": [_SAMPLE_RECORD], "msft": [_SAMPLE_RECORD], "goog": [_SAMPLE_RECORD]})
    fetch_bars(["AAPL", "MSFT", "GOOG"], "2024-01-01", "2024-01-05", api_key="tok", pace=True, http_get=http_get)
    assert len(sleep_calls) == 2  # paced between requests, not before the first


def test_pace_false_does_not_sleep(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("utils.tiingo_loader.time.sleep", lambda s: sleep_calls.append(s))
    http_get = _make_http_get({"aapl": [_SAMPLE_RECORD], "msft": [_SAMPLE_RECORD]})
    fetch_bars(["AAPL", "MSFT"], "2024-01-01", "2024-01-05", api_key="tok", pace=False, http_get=http_get)
    assert sleep_calls == []
