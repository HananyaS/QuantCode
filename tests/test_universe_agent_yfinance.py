"""Tests for UniverseAgent._download_yfinance.

A real bug was found in production use: the original implementation spawned
a ThreadPoolExecutor thread per ticker, each independently calling
yf.download(single_ticker). yfinance's internal session/cache state is not
safely shared across threads this way — a real 5-ticker concurrent fetch
returned IDENTICAL (cross-contaminated) data for every ticker. The fix uses
yfinance's own multi-ticker batch API (a single yf.download(tickers=[...])
call) instead of manual threading — these tests mock that call to verify
per-ticker data is correctly split with no cross-contamination, without
hitting the network.
"""
import pandas as pd
import pytest

from agents.universe_agent import UniverseAgent


def _agent():
    return UniverseAgent(
        tickers=["QQQ"], start_date="2020-01-01", end_date="2020-01-10",
        data_source="yfinance",
    )


def _multiticker_frame(tickers_and_closes: dict, n_days: int = 3) -> pd.DataFrame:
    """Mimic yfinance's group_by='ticker' multi-ticker response shape:
    columns are a 2-level MultiIndex (ticker, field)."""
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    columns = pd.MultiIndex.from_product(
        [list(tickers_and_closes.keys()), ["Open", "High", "Low", "Close", "Volume"]]
    )
    data = {}
    for ticker, close in tickers_and_closes.items():
        data[(ticker, "Open")] = [close] * n_days
        data[(ticker, "High")] = [close + 1] * n_days
        data[(ticker, "Low")] = [close - 1] * n_days
        data[(ticker, "Close")] = [close] * n_days
        data[(ticker, "Volume")] = [1_000_000] * n_days
    return pd.DataFrame(data, index=dates, columns=columns)


def _single_ticker_frame(close: float, n_days: int = 3) -> pd.DataFrame:
    """Mimic yfinance's response shape for a single-ticker call: flat columns."""
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {"Open": [close] * n_days, "High": [close + 1] * n_days,
         "Low": [close - 1] * n_days, "Close": [close] * n_days,
         "Volume": [1_000_000] * n_days},
        index=dates,
    )


def test_multi_ticker_batch_no_cross_contamination(monkeypatch):
    frame = _multiticker_frame({"QQQ": 400.0, "TQQQ": 60.0, "SQQQ": 10.0})

    def fake_download(tickers, start, end, progress, auto_adjust, group_by, threads):
        assert set(tickers) == {"QQQ", "TQQQ", "SQQQ"}
        return frame

    monkeypatch.setattr("yfinance.download", fake_download)
    result = _agent()._download_yfinance(["QQQ", "TQQQ", "SQQQ"], "2020-01-01", "2020-01-10")

    assert result["QQQ"]["Close"].iloc[0] == 400.0
    assert result["TQQQ"]["Close"].iloc[0] == 60.0
    assert result["SQQQ"]["Close"].iloc[0] == 10.0
    # The core regression check: no ticker's data equals another's.
    assert result["QQQ"]["Close"].iloc[0] != result["TQQQ"]["Close"].iloc[0]
    assert result["TQQQ"]["Close"].iloc[0] != result["SQQQ"]["Close"].iloc[0]


def test_single_ticker_flat_columns_handled(monkeypatch):
    frame = _single_ticker_frame(123.45)

    def fake_download(tickers, start, end, progress, auto_adjust, group_by, threads):
        assert tickers == ["QQQ"]
        return frame

    monkeypatch.setattr("yfinance.download", fake_download)
    result = _agent()._download_yfinance(["QQQ"], "2020-01-01", "2020-01-10")

    assert result["QQQ"]["Close"].iloc[0] == 123.45


def test_ticker_missing_from_batch_result_is_dropped_not_crashed(monkeypatch):
    frame = _multiticker_frame({"QQQ": 400.0})  # TQQQ requested but absent from response

    def fake_download(tickers, start, end, progress, auto_adjust, group_by, threads):
        return frame

    monkeypatch.setattr("yfinance.download", fake_download)
    result = _agent()._download_yfinance(["QQQ", "TQQQ"], "2020-01-01", "2020-01-10")

    assert "QQQ" in result
    assert "TQQQ" not in result


def test_empty_response_returns_empty_dict(monkeypatch):
    def fake_download(tickers, start, end, progress, auto_adjust, group_by, threads):
        return pd.DataFrame()

    monkeypatch.setattr("yfinance.download", fake_download)
    result = _agent()._download_yfinance(["QQQ"], "2020-01-01", "2020-01-10")

    assert result == {}


def test_batch_failure_does_not_raise(monkeypatch):
    def failing_download(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr("yfinance.download", failing_download)
    result = _agent()._download_yfinance(["QQQ"], "2020-01-01", "2020-01-10")

    assert result == {}
