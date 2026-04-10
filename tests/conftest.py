"""Shared pytest fixtures — all use synthetic data; no network calls."""
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Multi-asset fixtures
# ---------------------------------------------------------------------------

def _make_universe_data(
    n_tickers: int = 5,
    n_days: int = 300,
    seed: int = 0,
) -> dict:
    """Return a Dict[ticker, OHLCV DataFrame] with aligned business-day dates."""
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    universe = {}
    for i, ticker in enumerate(tickers):
        log_ret = rng.normal(0.0005, 0.01, size=n_days)
        close = 100.0 * np.exp(np.cumsum(log_ret))
        spread = np.abs(rng.normal(0, 0.005, size=n_days))
        universe[ticker] = pd.DataFrame(
            {
                "Open": close * (1 + rng.normal(0, 0.003, size=n_days)),
                "High": close * (1 + spread),
                "Low": close * (1 - spread),
                "Close": close,
                "Volume": rng.integers(1_000_000, 5_000_000, size=n_days).astype(float),
            },
            index=dates,
        )
    return universe


@pytest.fixture(scope="session")
def universe_data() -> dict:
    """5-ticker, 300-day synthetic universe."""
    return _make_universe_data(n_tickers=5, n_days=300)


@pytest.fixture(scope="session")
def cs_features_and_labels(universe_data):
    """Pre-computed cs_features and cs_labels from the synthetic universe."""
    from agents.cs_feature_agent import CrossSectionalFeatureAgent
    from agents.cs_labeling_agent import CrossSectionalLabelingAgent

    ctx = {"universe_data": universe_data}
    ctx = CrossSectionalFeatureAgent(cross_sectional=False).run(ctx)
    ctx = CrossSectionalLabelingAgent(forward_period=5).run(ctx)
    return ctx["cs_features"], ctx["cs_labels"]
