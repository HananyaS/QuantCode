"""Tests for agents/live_orchestrator.py — paper-trading pipeline orchestrator."""
import pandas as pd
import pytest

from agents.base_agent import BaseAgent
from agents.execution_agent import ExecutionAgent
from agents.live_orchestrator import LiveOrchestrator
from agents.portfolio_agent import PortfolioAgent
from utils.state_store import StateStore


@pytest.fixture
def store(tmp_path):
    """Injected in place of the default StateStore so tests never touch
    the real data/live_state.db — constructing ExecutionAgent (even without
    calling .run()) initializes the ledger's schema on disk immediately."""
    return StateStore(db_path=str(tmp_path / "live_state.db"))


class _RecordingAgent(BaseAgent):
    """Test double: appends its class name to context['calls'] and returns context."""

    def __init__(self, name, write_key=None, write_value=None):
        self.name = name
        self.write_key = write_key
        self.write_value = write_value

    def run(self, context: dict) -> dict:
        context.setdefault("calls", []).append(self.name)
        if self.write_key:
            context[self.write_key] = self.write_value
        return context


_MINIMAL_CONFIG = {
    "universe": {"tickers": ["AAPL"], "start_date": "2020-01-01", "end_date": "2020-12-31", "benchmark": "SPY"},
    "features": {"returns": [1, 5], "volatility_window": 20, "rsi_period": 14, "sma_windows": [10, 20], "cross_sectional": True},
    "labeling": {"forward_period": 5},
    "model": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "random_state": 42, "test_size": 0.2, "type": "lightgbm"},
    "portfolio": {"max_positions": 5, "entry_rank": 5, "exit_rank": 10},
    "data_source": {"type": "alpaca", "api_key": "k", "secret_key": "s"},
}


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def test_build_pipeline_ends_with_execution_agent(store):
    orch = LiveOrchestrator(_MINIMAL_CONFIG, api_key="k", secret_key="s", state_store=store)
    pipeline = orch._build_pipeline()
    assert isinstance(pipeline[-1], ExecutionAgent)


def test_build_pipeline_includes_portfolio_agent_before_execution(store):
    orch = LiveOrchestrator(_MINIMAL_CONFIG, api_key="k", secret_key="s", state_store=store)
    pipeline = orch._build_pipeline()
    classes = [type(a) for a in pipeline]
    assert PortfolioAgent in classes
    assert classes.index(PortfolioAgent) < classes.index(ExecutionAgent)


def test_build_pipeline_excludes_backtest_and_evaluation_agents(store):
    orch = LiveOrchestrator(_MINIMAL_CONFIG, api_key="k", secret_key="s", state_store=store)
    pipeline = orch._build_pipeline()
    names = [type(a).__name__ for a in pipeline]
    assert "MultiAssetBacktestAgent" not in names
    assert "MultiAssetEvaluationAgent" not in names


# ---------------------------------------------------------------------------
# run() threads context through injected agents in order
# ---------------------------------------------------------------------------

def test_run_executes_agents_in_order_with_injection(store):
    agents = [_RecordingAgent("first"), _RecordingAgent("second"), _RecordingAgent("third")]
    orch = LiveOrchestrator(_MINIMAL_CONFIG, api_key="k", secret_key="s", state_store=store)
    context = orch.run(agents=agents)
    assert context["calls"] == ["first", "second", "third"]


def test_run_returns_context_with_initial_keys_present(store):
    agents = [_RecordingAgent("noop")]
    orch = LiveOrchestrator(_MINIMAL_CONFIG, api_key="k", secret_key="s", state_store=store)
    context = orch.run(agents=agents)
    for key in ("universe_data", "cs_features", "cs_predictions", "portfolio_weights",
                "execution_orders", "execution_account"):
        assert key in context
