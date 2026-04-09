"""CriticAgent: validates the full pipeline context for data leakage and correctness."""
import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLeakageError(Exception):
    """Raised when data leakage is detected anywhere in the pipeline."""


class CriticAgent(BaseAgent):
    """Validates the pipeline context after all agents have run.

    Checks performed:
    1. Train/test split is strictly temporal (no overlap).
    2. The feature matrix does not contain the label column.
    3. All predictions fall strictly after the training window ends.
    4. All required metric keys are present.
    """

    REQUIRED_METRICS = ("sharpe", "max_drawdown", "annualized_return")

    def run(self, context: dict) -> dict:
        """Run all validation checks; raises DataLeakageError or AssertionError on failure."""
        self._check_temporal_split(context)
        self._check_label_not_in_features(context)
        self._check_predictions_after_train(context)
        self._check_metrics_complete(context)
        logger.info("CriticAgent: all validation checks passed")
        return context

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_temporal_split(self, context: dict) -> None:
        """Ensure training window ends before test window begins."""
        model_info = context.get("model")
        if model_info is None:
            return
        train_end = pd.Timestamp(model_info["train_end"])
        test_start = pd.Timestamp(model_info["test_start"])
        if train_end >= test_start:
            raise DataLeakageError(
                f"Temporal leakage detected: train_end={train_end.date()} "
                f">= test_start={test_start.date()}"
            )

    def _check_label_not_in_features(self, context: dict) -> None:
        """Ensure the target variable was not accidentally included as a feature."""
        features = context.get("features")
        if features is None:
            return
        if "label" in features.columns:
            raise DataLeakageError(
                "Label column found inside the feature matrix — direct leakage"
            )

    def _check_predictions_after_train(self, context: dict) -> None:
        """Ensure every prediction date is strictly after the training end date."""
        predictions = context.get("predictions")
        model_info = context.get("model")
        if predictions is None or model_info is None:
            return
        pred_values: pd.Series = predictions["values"]
        train_end = pd.Timestamp(model_info["train_end"])
        earliest_pred = pred_values.index.min()
        if earliest_pred <= train_end:
            raise DataLeakageError(
                f"Prediction dates overlap with training window: "
                f"earliest prediction {earliest_pred.date()} <= train_end {train_end.date()}"
            )

    def _check_metrics_complete(self, context: dict) -> None:
        """Assert that all required metric keys are present."""
        metrics = context.get("metrics")
        if metrics is None:
            return
        missing = [k for k in self.REQUIRED_METRICS if k not in metrics]
        assert not missing, f"Missing required metrics: {missing}"
