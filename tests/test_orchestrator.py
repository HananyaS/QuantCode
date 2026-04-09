"""Tests for Orchestrator — context initialisation and end-to-end pipeline."""
import numpy as np
import pandas as pd
import pytest

from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.labeling_agent import LabelingAgent
from agents.model_agent import ModelAgent
from agents.backtest_agent import BacktestAgent
from agents.evaluation_agent import EvaluationAgent
from agents.critic_agent import CriticAgent
from agents.orchestrator import Orchestrator


_DEFAULT_CONFIG = {
    "data": {
        "ticker": "AAPL",
        "benchmark": "SPY",
        "start_date": "2018-01-01",
        "end_date": "2023-12-31",
    },
    "features": {"sma_windows": [5, 10, 20], "rsi_period": 14, "bb_window": 20, "momentum_period": 5},
    "labeling": {"forward_period": 1, "threshold": 0.0},
    "model": {"n_estimators": 50, "max_depth": 4, "random_state": 42, "test_size": 0.2},
    "backtest": {"initial_capital": 100_000, "transaction_cost": 0.001},
}


# ---------------------------------------------------------------------------
# Context structure
# ---------------------------------------------------------------------------

def test_make_context_has_all_keys():
    orch = Orchestrator(_DEFAULT_CONFIG)
    ctx = orch._make_context()
    expected = {"data", "benchmark_data", "features", "labels", "model", "predictions", "backtest", "metrics"}
    assert expected == set(ctx.keys())


def test_make_context_all_none():
    orch = Orchestrator(_DEFAULT_CONFIG)
    ctx = orch._make_context()
    for v in ctx.values():
        assert v is None


# ---------------------------------------------------------------------------
# End-to-end pipeline with synthetic data (no network)
# ---------------------------------------------------------------------------

def _build_synthetic_pipeline(ohlcv: pd.DataFrame) -> list:
    """Build a full pipeline that injects synthetic data for both ticker and benchmark."""
    return [
        DataAgent("AAPL", "2020-01-01", "2023-01-01", context_key="data", data_override=ohlcv),
        DataAgent("SPY", "2020-01-01", "2023-01-01", context_key="benchmark_data", data_override=ohlcv),
        FeatureAgent(),
        LabelingAgent(),
        ModelAgent(n_estimators=20, random_state=42, test_size=0.2),
        BacktestAgent(initial_capital=100_000, transaction_cost=0.001),
        EvaluationAgent(),
        CriticAgent(),
    ]


def test_full_pipeline_runs_without_error(sample_ohlcv):
    orch = Orchestrator(_DEFAULT_CONFIG)
    pipeline = _build_synthetic_pipeline(sample_ohlcv)
    ctx = orch.run(agents=pipeline)
    assert ctx["metrics"] is not None


def test_full_pipeline_populates_all_keys(sample_ohlcv):
    orch = Orchestrator(_DEFAULT_CONFIG)
    ctx = orch.run(agents=_build_synthetic_pipeline(sample_ohlcv))
    for key in ("data", "benchmark_data", "features", "labels", "model", "predictions", "backtest", "metrics"):
        assert ctx[key] is not None, f"context['{key}'] is None after pipeline run"


def test_full_pipeline_metrics_are_finite(sample_ohlcv):
    import math
    orch = Orchestrator(_DEFAULT_CONFIG)
    ctx = orch.run(agents=_build_synthetic_pipeline(sample_ohlcv))
    metrics = ctx["metrics"]
    for key in ("sharpe", "max_drawdown", "annualized_return"):
        val = metrics[key]
        assert math.isfinite(val), f"metrics['{key}'] = {val} is not finite"


def test_full_pipeline_no_temporal_leakage(sample_ohlcv):
    """CriticAgent would raise DataLeakageError if there were leakage — so if we reach
    this assertion, the pipeline is clean."""
    orch = Orchestrator(_DEFAULT_CONFIG)
    ctx = orch.run(agents=_build_synthetic_pipeline(sample_ohlcv))
    model_info = ctx["model"]
    train_end = pd.Timestamp(model_info["train_end"])
    test_start = pd.Timestamp(model_info["test_start"])
    assert train_end < test_start


def test_full_pipeline_deterministic(sample_ohlcv):
    """Two runs with the same config must produce identical predictions."""
    orch = Orchestrator(_DEFAULT_CONFIG)
    ctx1 = orch.run(agents=_build_synthetic_pipeline(sample_ohlcv))
    ctx2 = orch.run(agents=_build_synthetic_pipeline(sample_ohlcv))
    pd.testing.assert_series_equal(
        ctx1["predictions"]["values"],
        ctx2["predictions"]["values"],
    )


def test_pipeline_with_custom_agents_override(sample_ohlcv):
    """Orchestrator.run(agents=...) must use the provided agents, not the default ones."""
    orch = Orchestrator(_DEFAULT_CONFIG)
    pipeline = _build_synthetic_pipeline(sample_ohlcv)
    ctx = orch.run(agents=pipeline)
    # Verify data came from our synthetic override, not a real download
    assert ctx["data"].equals(sample_ohlcv)
