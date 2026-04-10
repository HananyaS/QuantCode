"""MultiAssetOrchestrator: orchestrates the cross-sectional multi-asset pipeline.

Completely independent of the single-asset Orchestrator — shares no state,
no context keys conflict (new keys: universe, universe_data, cs_*, portfolio_*,
multi_backtest, multi_metrics).  The original single-asset pipeline is untouched.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.cs_feature_agent import CrossSectionalFeatureAgent
from agents.cs_labeling_agent import CrossSectionalLabelingAgent
from agents.multi_backtest_agent import MultiAssetBacktestAgent
from agents.multi_evaluation_agent import MultiAssetEvaluationAgent
from agents.portfolio_agent import PortfolioAgent
from agents.ranking_model_agent import RankingModelAgent
from agents.universe_agent import UniverseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiAssetOrchestrator:
    """Builds and runs the multi-asset scanning pipeline.

    Pipeline order
    --------------
    UniverseAgent
      -> CrossSectionalFeatureAgent
      -> CrossSectionalLabelingAgent
      -> RankingModelAgent
      -> PortfolioAgent
      -> MultiAssetBacktestAgent
      -> MultiAssetEvaluationAgent

    Args:
        config: Parsed universe.yaml config dict.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def run(self, agents: Optional[List[BaseAgent]] = None) -> dict:
        """Execute the pipeline and return the final context.

        Args:
            agents: Optional override list of agents (used in tests with
                    synthetic data to bypass network calls).
        """
        context = self._make_context()
        pipeline = agents if agents is not None else self._build_pipeline()
        for agent in pipeline:
            logger.info("MultiAssetOrchestrator -> running %s", agent.__class__.__name__)
            context = agent.run(context)
        logger.info("MultiAssetOrchestrator: pipeline complete")
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
            "multi_backtest": None,
            "multi_metrics": None,
        }

    def _build_pipeline(self) -> List[BaseAgent]:
        cfg = self.config
        u = cfg["universe"]
        f = cfg["features"]
        lb = cfg["labeling"]
        m = cfg["model"]
        p = cfg["portfolio"]
        bt = cfg["backtest"]

        ds = cfg.get("data_source", {})
        data_source = ds.get("type", "alpaca")

        return [
            UniverseAgent(
                tickers=u["tickers"],
                start_date=u["start_date"],
                end_date=u["end_date"],
                benchmark=u["benchmark"],
                min_history_days=u.get("min_history_days", 1260),
                data_source=data_source,
                alpaca_key=ds.get("api_key"),
                alpaca_secret=ds.get("secret_key"),
                alpaca_feed=ds.get("feed"),
            ),
            CrossSectionalFeatureAgent(
                returns_windows=f["returns"],
                vol_window=f["volatility_window"],
                rsi_period=f["rsi_period"],
                sma_windows=f["sma_windows"],
                cross_sectional=f["cross_sectional"],
            ),
            CrossSectionalLabelingAgent(
                forward_period=lb["forward_period"],
            ),
            RankingModelAgent(
                n_estimators=m["n_estimators"],
                max_depth=m["max_depth"],
                learning_rate=m["learning_rate"],
                random_state=m["random_state"],
                test_size=m["test_size"],
                model_type=m["type"],
            ),
            PortfolioAgent(
                max_positions=p["max_positions"],
                entry_rank=p["entry_rank"],
                exit_rank=p["exit_rank"],
                min_score=p.get("min_score", 0.0),
                trailing_stop_atr_mult=p.get("trailing_stop_atr_mult", 1.5),
                atr_period=p.get("atr_period", 14),
            ),
            MultiAssetBacktestAgent(
                initial_capital=bt["initial_capital"],
                transaction_cost=bt["transaction_cost"],
            ),
            MultiAssetEvaluationAgent(),
        ]
