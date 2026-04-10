"""QuantCode — agent package."""
from agents.backtest_agent import BacktestAgent
from agents.base_agent import BaseAgent
from agents.critic_agent import CriticAgent, DataLeakageError
from agents.cs_feature_agent import CrossSectionalFeatureAgent
from agents.cs_labeling_agent import CrossSectionalLabelingAgent
from agents.data_agent import DataAgent
from agents.evaluation_agent import EvaluationAgent
from agents.feature_agent import FeatureAgent
from agents.labeling_agent import LabelingAgent
from agents.model_agent import ModelAgent
from agents.multi_backtest_agent import MultiAssetBacktestAgent
from agents.multi_evaluation_agent import MultiAssetEvaluationAgent
from agents.multi_orchestrator import MultiAssetOrchestrator
from agents.orchestrator import Orchestrator
from agents.portfolio_agent import PortfolioAgent
from agents.ranking_model_agent import RankingModelAgent
from agents.universe_agent import UniverseAgent

__all__ = [
    # Single-asset pipeline
    "BaseAgent",
    "DataAgent",
    "FeatureAgent",
    "LabelingAgent",
    "ModelAgent",
    "BacktestAgent",
    "EvaluationAgent",
    "CriticAgent",
    "DataLeakageError",
    "Orchestrator",
    # Multi-asset pipeline
    "UniverseAgent",
    "CrossSectionalFeatureAgent",
    "CrossSectionalLabelingAgent",
    "RankingModelAgent",
    "PortfolioAgent",
    "MultiAssetBacktestAgent",
    "MultiAssetEvaluationAgent",
    "MultiAssetOrchestrator",
]
