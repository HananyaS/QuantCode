"""Shared pytest fixtures — all use synthetic data; no network calls."""
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# OHLCV fixture helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 500, seed: int = 42, constant_close: bool = False) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with a business-day DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)

    if constant_close:
        close = np.full(n, 100.0)
    else:
        log_returns = rng.normal(0.0005, 0.01, size=n)
        close = 100.0 * np.exp(np.cumsum(log_returns))

    spread = np.abs(rng.normal(0, 0.005, size=n))
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = close * (1 + rng.normal(0, 0.003, size=n))
    volume = rng.integers(1_000_000, 10_000_000, size=n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Public fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_ohlcv() -> pd.DataFrame:
    """Standard 500-row OHLCV DataFrame."""
    return _make_ohlcv(n=500)


@pytest.fixture(scope="session")
def small_ohlcv() -> pd.DataFrame:
    """Minimal 60-row OHLCV DataFrame (above 30-row validation threshold)."""
    return _make_ohlcv(n=60, seed=7)


@pytest.fixture(scope="session")
def constant_ohlcv() -> pd.DataFrame:
    """OHLCV DataFrame with constant Close prices."""
    return _make_ohlcv(n=100, constant_close=True)


@pytest.fixture(scope="session")
def nan_ohlcv(sample_ohlcv) -> pd.DataFrame:
    """OHLCV DataFrame with NaN values injected into Close."""
    df = sample_ohlcv.copy()
    df.iloc[10, df.columns.get_loc("Close")] = np.nan
    return df


@pytest.fixture(scope="session")
def features_and_labels(sample_ohlcv):
    """Pre-computed features and labels for the standard dataset."""
    from agents.feature_agent import FeatureAgent
    from agents.labeling_agent import LabelingAgent

    ctx: dict = {"data": sample_ohlcv}
    ctx = FeatureAgent().run(ctx)
    ctx = LabelingAgent().run(ctx)
    return ctx["features"], ctx["labels"]
