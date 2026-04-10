"""Tests for UniverseAgent (synthetic data, no network calls)."""
import numpy as np
import pandas as pd
import pytest

from agents.universe_agent import UniverseAgent


def _make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n)))
    spread = np.abs(rng.normal(0, 0.005, size=n))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, size=n)),
            "High": close * (1 + spread),
            "Low": close * (1 - spread),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, size=n).astype(float),
        },
        index=dates,
    )


def _make_universe(n_tickers=4, n_days=200):
    return {f"T{i:02d}": _make_ohlcv(n_days, seed=i) for i in range(n_tickers)}


def _inject_universe(agent: UniverseAgent, universe: dict, benchmark: pd.DataFrame) -> dict:
    """Bypass network: directly write into context as UniverseAgent would."""
    context = {
        "universe": None,
        "universe_data": None,
        "benchmark_data": None,
    }
    # Simulate what UniverseAgent.run writes, for agents downstream
    aligned = agent._align(universe)
    context["universe"] = list(aligned.keys())
    context["universe_data"] = aligned
    context["benchmark_data"] = benchmark
    return context


# ---------------------------------------------------------------------------
# _align
# ---------------------------------------------------------------------------

def test_align_keeps_common_dates():
    dates_a = pd.bdate_range("2020-01-01", periods=100)
    dates_b = pd.bdate_range("2020-01-05", periods=100)  # starts 3 days later
    universe = {
        "A": pd.DataFrame({"Open": 1, "High": 1, "Low": 1, "Close": 1.0, "Volume": 1.0}, index=dates_a),
        "B": pd.DataFrame({"Open": 1, "High": 1, "Low": 1, "Close": 1.0, "Volume": 1.0}, index=dates_b),
    }
    agent = UniverseAgent(["A", "B"], "2020-01-01", "2021-01-01")
    aligned = agent._align(universe)
    assert aligned["A"].index.equals(aligned["B"].index)
    assert len(aligned["A"]) < 100  # fewer dates after inner join


def test_align_raises_on_too_few_common_dates():
    dates_a = pd.bdate_range("2020-01-01", periods=50)
    dates_b = pd.bdate_range("2022-01-01", periods=50)  # no overlap
    universe = {
        "A": pd.DataFrame({"Open": 1, "High": 1, "Low": 1, "Close": 1.0, "Volume": 1.0}, index=dates_a),
        "B": pd.DataFrame({"Open": 1, "High": 1, "Low": 1, "Close": 1.0, "Volume": 1.0}, index=dates_b),
    }
    agent = UniverseAgent(["A", "B"], "2020-01-01", "2023-01-01")
    with pytest.raises(AssertionError):
        agent._align(universe)


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------

def test_validate_passes_for_valid_df():
    df = _make_ohlcv(50)
    UniverseAgent._validate(df, "TEST")  # no exception


def test_validate_rejects_empty():
    with pytest.raises(AssertionError, match="empty"):
        UniverseAgent._validate(pd.DataFrame(), "EMPTY")


def test_validate_rejects_missing_column():
    df = _make_ohlcv(50).drop(columns=["Volume"])
    with pytest.raises(AssertionError, match="missing columns"):
        UniverseAgent._validate(df, "X")


def test_validate_rejects_nan_close():
    df = _make_ohlcv(50)
    df.iloc[5, df.columns.get_loc("Close")] = np.nan
    with pytest.raises(AssertionError, match="NaN in Close"):
        UniverseAgent._validate(df, "X")


def test_validate_rejects_non_positive_close():
    df = _make_ohlcv(50)
    df.iloc[3, df.columns.get_loc("Close")] = 0.0
    with pytest.raises(AssertionError, match="non-positive Close"):
        UniverseAgent._validate(df, "X")


def test_validate_rejects_too_few_rows():
    df = _make_ohlcv(10)
    with pytest.raises(AssertionError, match="rows"):
        UniverseAgent._validate(df, "X")


# ---------------------------------------------------------------------------
# Context keys
# ---------------------------------------------------------------------------

def test_context_keys_written(universe_data):
    agent = UniverseAgent(list(universe_data.keys()), "2020-01-01", "2021-01-01")
    bm = _make_ohlcv(200, seed=99)
    ctx = _inject_universe(agent, universe_data, bm)

    assert ctx["universe"] is not None
    assert ctx["universe_data"] is not None
    assert ctx["benchmark_data"] is not None
    assert set(ctx["universe"]) == set(universe_data.keys())


def test_all_assets_have_same_dates(universe_data):
    agent = UniverseAgent(list(universe_data.keys()), "2020-01-01", "2021-01-01")
    aligned = agent._align(universe_data)
    first_index = next(iter(aligned.values())).index
    for df in aligned.values():
        assert df.index.equals(first_index)
