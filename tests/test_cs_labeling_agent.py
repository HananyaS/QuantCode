"""Tests for CrossSectionalLabelingAgent."""
import numpy as np
import pandas as pd
import pytest

from agents.cs_labeling_agent import CrossSectionalLabelingAgent


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_output_is_multiindex_series(universe_data):
    agent = CrossSectionalLabelingAgent(forward_period=5)
    ctx = agent.run({"universe_data": universe_data})
    labels = ctx["cs_labels"]
    assert isinstance(labels, pd.Series)
    assert isinstance(labels.index, pd.MultiIndex)
    assert labels.index.names == ["date", "ticker"]
    assert labels.name == "label"


def test_all_tickers_present_in_labels(universe_data):
    agent = CrossSectionalLabelingAgent(forward_period=5)
    ctx = agent.run({"universe_data": universe_data})
    tickers = set(ctx["cs_labels"].index.get_level_values("ticker").unique())
    assert tickers == set(universe_data.keys())


def test_no_nan_labels(universe_data):
    agent = CrossSectionalLabelingAgent(forward_period=5)
    ctx = agent.run({"universe_data": universe_data})
    assert ctx["cs_labels"].notna().all()


def test_label_count_is_n_minus_forward_period(universe_data):
    N = 5
    agent = CrossSectionalLabelingAgent(forward_period=N)
    ctx = agent.run({"universe_data": universe_data})
    labels = ctx["cs_labels"]
    n_total_rows = sum(len(df) for df in universe_data.values())
    n_expected = n_total_rows - N * len(universe_data)
    assert len(labels) == n_expected


# ---------------------------------------------------------------------------
# No leakage: last N dates must be absent
# ---------------------------------------------------------------------------

def test_last_forward_period_dates_absent(universe_data):
    N = 5
    agent = CrossSectionalLabelingAgent(forward_period=N)
    ctx = agent.run({"universe_data": universe_data})
    labels = ctx["cs_labels"]

    for ticker, df in universe_data.items():
        last_n = df.index[-N:]
        ticker_dates = labels.xs(ticker, level="ticker").index
        overlap = ticker_dates.intersection(last_n)
        assert len(overlap) == 0, (
            f"Leakage: ticker {ticker} has labels on {overlap.tolist()}"
        )


def test_label_value_matches_forward_return(universe_data):
    """Spot-check: label[t] == (close[t+N] - close[t]) / close[t]."""
    N = 5
    ticker = list(universe_data.keys())[0]
    df = universe_data[ticker]
    close = df["Close"]

    agent = CrossSectionalLabelingAgent(forward_period=N)
    ctx = agent.run({"universe_data": universe_data})
    labels = ctx["cs_labels"]

    ticker_labels = labels.xs(ticker, level="ticker")
    # Check first 10 dates
    for date in ticker_labels.index[:10]:
        loc = close.index.get_loc(date)
        expected = close.iloc[loc + N] / close.iloc[loc] - 1
        assert abs(ticker_labels.loc[date] - expected) < 1e-10


def test_configurable_forward_period(universe_data):
    for N in [1, 3, 10]:
        agent = CrossSectionalLabelingAgent(forward_period=N)
        ctx = agent.run({"universe_data": universe_data})
        labels = ctx["cs_labels"]
        for ticker, df in universe_data.items():
            last_n = df.index[-N:]
            ticker_dates = labels.xs(ticker, level="ticker").index
            assert len(ticker_dates.intersection(last_n)) == 0


# ---------------------------------------------------------------------------
# Edge cases / errors
# ---------------------------------------------------------------------------

def test_raises_if_universe_data_missing():
    agent = CrossSectionalLabelingAgent()
    with pytest.raises(AssertionError, match="universe_data"):
        agent.run({})


def test_forward_period_zero_raises():
    with pytest.raises(AssertionError, match="forward_period"):
        CrossSectionalLabelingAgent(forward_period=0)
