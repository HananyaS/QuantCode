"""LiveOrchestrator: runs the paper-trading pipeline.

Shares the exact same research pipeline as MultiAssetOrchestrator
(UniverseAgent -> ... -> PortfolioAgent, via
MultiAssetOrchestrator._build_research_pipeline) so live and backtest
decisions are made identically. Where MultiAssetOrchestrator feeds the
resulting weights into a historical simulation (MultiAssetBacktestAgent),
LiveOrchestrator feeds them into ExecutionAgent to submit real (paper)
orders for today.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.execution_agent import ExecutionAgent
from agents.multi_orchestrator import MultiAssetOrchestrator
from utils.logger import get_logger
from utils.state_store import StateStore

logger = get_logger(__name__)


class LiveOrchestrator:
    """Builds and runs the daily paper-trading pipeline.

    Pipeline order
    --------------
    UniverseAgent -> CrossSectionalFeatureAgent -> CrossSectionalLabelingAgent
      -> RankingModelAgent -> PortfolioAgent -> ExecutionAgent

    Args:
        config: Parsed universe.yaml config dict (same shape as
            MultiAssetOrchestrator's config).
        api_key, secret_key: Alpaca paper-trading credentials.
        state_store: Optional StateStore override (defaults to the
            standard data/live_state.db ledger).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: str,
        secret_key: str,
        state_store: Optional[StateStore] = None,
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.secret_key = secret_key
        self.state_store = state_store

    def run(self, agents: Optional[List[BaseAgent]] = None) -> dict:
        """Execute the pipeline and return the final context.

        Args:
            agents: Optional override list of agents (used in tests with
                    synthetic data to bypass network/broker calls).
        """
        context = self._make_context()
        pipeline = agents if agents is not None else self._build_pipeline()
        for agent in pipeline:
            logger.info("LiveOrchestrator -> running %s", agent.__class__.__name__)
            context = agent.run(context)
        logger.info("LiveOrchestrator: pipeline complete")
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_context(self) -> dict:
        return {
            "universe": None,
            "universe_data": None,
            "benchmark_data": None,
            "cs_features": None,
            "cs_labels": None,
            "cs_model": None,
            "cs_predictions": None,
            "portfolio_weights": None,
            "execution_orders": None,
            "execution_account": None,
        }

    def _build_pipeline(self) -> List[BaseAgent]:
        research = MultiAssetOrchestrator(self.config)._build_research_pipeline()
        execution = ExecutionAgent(
            api_key=self.api_key,
            secret_key=self.secret_key,
            state_store=self.state_store,
        )
        return research + [execution]
