"""ModelAgent: trains a RandomForest classifier with a strict temporal train/test split."""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelAgent(BaseAgent):
    """Fits a RandomForest on the training window and predicts over the test window.

    Walk-forward split: the first (1 - test_size) fraction of aligned rows is used
    for training; the remainder is used for testing.  The split is purely temporal —
    no shuffling, no future data bleeds into the training set.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth (controls overfitting).
        random_state: Fixed seed for reproducibility.
        test_size: Fraction of data reserved for the test window (0 < test_size < 1).
    """

    MIN_TEST_ROWS = 20

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> None:
        assert 0 < test_size < 1, "test_size must be in (0, 1)"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.test_size = test_size

    def run(self, context: dict) -> dict:
        """Train model and store results in context['model'] and context['predictions']."""
        assert context.get("features") is not None, (
            "ModelAgent requires context['features']"
        )
        assert context.get("labels") is not None, (
            "ModelAgent requires context['labels']"
        )

        features: pd.DataFrame = context["features"]
        labels: pd.Series = context["labels"]

        X, y = self._align(features, labels)
        X_train, X_test, y_train, y_test = self._temporal_split(X, y)

        model = self._fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        context["model"] = self._build_model_record(model, X, X_train, X_test)
        context["predictions"] = self._build_predictions_record(
            model, X_test, test_pred, y_train, train_pred, y_test
        )

        logger.info(
            "ModelAgent: train_acc=%.3f  test_acc=%.3f  train=%s->%s  test=%s->%s",
            accuracy_score(y_train, train_pred),
            accuracy_score(y_test, test_pred),
            X_train.index[0].date(),
            X_train.index[-1].date(),
            X_test.index[0].date(),
            X_test.index[-1].date(),
        )
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _align(self, features: pd.DataFrame, labels: pd.Series) -> tuple:
        """Inner-join features and labels on date index, drop any residual NaN."""
        combined = features.join(labels, how="inner").dropna()
        assert len(combined) > 0, "No rows remain after aligning features and labels"
        X = combined.drop(columns=["label"])
        y = combined["label"]
        return X, y

    def _temporal_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple:
        """Chronological split — no shuffling."""
        split = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        assert len(X_test) >= self.MIN_TEST_ROWS, (
            f"Test set too small ({len(X_test)} rows); need at least {self.MIN_TEST_ROWS}"
        )
        assert X_train.index.max() < X_test.index.min(), (
            "Temporal leakage: training window overlaps with test window"
        )
        return X_train, X_test, y_train, y_test

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Fit a deterministic RandomForest."""
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    def _build_model_record(
        self,
        model: RandomForestClassifier,
        X: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Dict:
        return {
            "estimator": model,
            "features": list(X.columns),
            "train_start": str(X_train.index[0].date()),
            "train_end": str(X_train.index[-1].date()),
            "test_start": str(X_test.index[0].date()),
            "test_end": str(X_test.index[-1].date()),
        }

    def _build_predictions_record(
        self,
        model: RandomForestClassifier,
        X_test: pd.DataFrame,
        test_pred: np.ndarray,
        y_train: pd.Series,
        train_pred: np.ndarray,
        y_test: pd.Series,
    ) -> Dict:
        importances: List = list(model.feature_importances_)
        return {
            "values": pd.Series(test_pred, index=X_test.index, name="prediction"),
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "feature_importances": dict(
                zip(model.feature_names_in_, importances)
            ),
        }
