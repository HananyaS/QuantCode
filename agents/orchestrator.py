"""Orchestrator: wires agents into a sequential pipeline and manages the context."""
from typing import Dict, List, Optional

from agents.backtest_agent import BacktestAgent
from agents.base_agent import BaseAgent
from agents.critic_agent import CriticAgent
from agents.data_agent import DataAgent
from agents.evaluation_agent import EvaluationAgent
from agents.feature_agent import FeatureAgent
from agents.labeling_agent import LabelingAgent
from agents.model_agent import ModelAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Builds and executes the full data → features → labels → model → backtest → eval pipeline.

    Args:
        config: Nested config dict (matches configs/experiment.yaml structure).
    """

    def __init__(self, config: Dict) -> None:
        self.config = config

    def run(self, agents: Optional[List[BaseAgent]] = None) -> dict:
        """Execute the pipeline.

        Args:
            agents: Optional override list of agents (useful for testing with synthetic data).
                    If None, the default pipeline is built from self.config.

        Returns:
            Final context dict with all pipeline outputs populated.
        """
        context = self._make_context()
        pipeline = agents if agents is not None else self._build_pipeline()
        for agent in pipeline:
            logger.info("Orchestrator → running %s", agent.__class__.__name__)
            context = agent.run(context)
        logger.info("Orchestrator: pipeline complete")
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_context(self) -> dict:
        """Return an empty pipeline context with all expected keys."""
        return {
            "data": None,
            "benchmark_data": None,
            "features": None,
            "labels": None,
            "model": None,
            "predictions": None,
            "backtest": None,
            "metrics": None,
        }

    def _build_pipeline(self) -> List[BaseAgent]:
        """Instantiate all agents from config in pipeline order."""
        cfg = self.config

        data_agent = DataAgent(
            ticker=cfg["data"]["ticker"],
            start_date=cfg["data"]["start_date"],
            end_date=cfg["data"]["end_date"],
            context_key="data",
        )
        benchmark_agent = DataAgent(
            ticker=cfg["data"]["benchmark"],
            start_date=cfg["data"]["start_date"],
            end_date=cfg["data"]["end_date"],
            context_key="benchmark_data",
        )
        feature_agent = FeatureAgent(
            sma_windows=cfg["features"]["sma_windows"],
            rsi_period=cfg["features"]["rsi_period"],
            bb_window=cfg["features"]["bb_window"],
            momentum_period=cfg["features"]["momentum_period"],
        )
        labeling_agent = LabelingAgent(
            forward_period=cfg["labeling"]["forward_period"],
            threshold=cfg["labeling"]["threshold"],
        )
        model_agent = ModelAgent(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            random_state=cfg["model"]["random_state"],
            test_size=cfg["model"]["test_size"],
        )
        backtest_agent = BacktestAgent(
            initial_capital=cfg["backtest"]["initial_capital"],
            transaction_cost=cfg["backtest"]["transaction_cost"],
        )
        return [
            data_agent,
            benchmark_agent,
            feature_agent,
            labeling_agent,
            model_agent,
            backtest_agent,
            EvaluationAgent(),
            CriticAgent(),
        ]
