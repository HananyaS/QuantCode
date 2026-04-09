"""LabelingAgent: generates forward-return binary labels without data leakage."""
import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class LabelingAgent(BaseAgent):
    """Creates binary labels: 1 if the forward return exceeds the threshold, else 0.

    Label at day t = sign( close[t + forward_period] / close[t] - 1 - threshold ).

    The final `forward_period` rows are dropped because no future price is available.
    No information about day t+1 onwards is used to build features for day t.

    Args:
        forward_period: Number of trading days ahead to measure the return.
        threshold: Minimum return (exclusive) required for a positive label.
    """

    def __init__(self, forward_period: int = 1, threshold: float = 0.0) -> None:
        assert forward_period >= 1, "forward_period must be >= 1"
        self.forward_period = forward_period
        self.threshold = threshold

    def run(self, context: dict) -> dict:
        """Compute labels and store in context['labels']."""
        assert context.get("data") is not None, (
            "LabelingAgent requires context['data']"
        )
        close = context["data"]["Close"]
        labels = self._compute_labels(close)
        self._validate(labels)
        context["labels"] = labels
        logger.info(
            "LabelingAgent: %d labels | pos=%.1f%% neg=%.1f%%",
            len(labels),
            100 * labels.mean(),
            100 * (1 - labels.mean()),
        )
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_labels(self, close: pd.Series) -> pd.Series:
        """Shift prices forward to obtain future returns, then binarise."""
        # forward_return[t] = close[t + k] / close[t] - 1
        # Uses shift(-k) which is a look-ahead — but this defines the TARGET, not a feature.
        # Features at day t must NOT include any shifted(-k) values.
        forward_return = close.shift(-self.forward_period) / close - 1
        labels = (forward_return > self.threshold).astype(int)
        # Drop the last `forward_period` rows where no future price exists
        labels = labels.iloc[: -self.forward_period]
        return labels.rename("label")

    def _validate(self, labels: pd.Series) -> None:
        """Assert labels are binary and NaN-free."""
        assert len(labels) > 0, "Label series is empty"
        assert labels.notna().all(), "Labels contain NaN values"
        assert set(labels.unique()).issubset({0, 1}), (
            f"Labels must be binary (0/1), got: {set(labels.unique())}"
        )
