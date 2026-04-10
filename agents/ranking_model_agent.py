"""RankingModelAgent: trains a LightGBM regression model on (date, ticker) rows.

Temporal split
--------------
Dates are sorted chronologically.  The first (1 - test_size) fraction of
*unique dates* goes to training; the remainder to testing.  No shuffling
of any kind occurs.  The explicit assertion ``train_end < test_start``
guards against any accidental overlap.

Leakage check
-------------
All feature rows at a given date use only historical data (enforced by
CrossSectionalFeatureAgent).  The inner join with labels automatically
drops dates where labels are unavailable (last N dates per forward period).

Sharpe sanity
-------------
If the out-of-sample IC > 0.15 or an external caller reports Sharpe > 2,
a WARNING is emitted — such values usually indicate a bug or leakage.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # pragma: no cover
    _HAS_LGB = False

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)

_HIGH_IC_THRESHOLD = 0.15


class RankingModelAgent(BaseAgent):
    """Fits a gradient-boosted regression model to predict cross-sectional returns.

    Writes
    ------
    context['cs_model']       : Dict — estimator + metadata
    context['cs_predictions'] : pd.Series(MultiIndex(date, ticker)) — predicted scores
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
        test_size: float = 0.2,
        model_type: str = "lightgbm",
    ) -> None:
        assert 0 < test_size < 1, "test_size must be in (0, 1)"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.test_size = test_size
        self.model_type = model_type

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        assert context.get("cs_features") is not None, (
            "RankingModelAgent requires context['cs_features']"
        )
        assert context.get("cs_labels") is not None, (
            "RankingModelAgent requires context['cs_labels']"
        )

        features: pd.DataFrame = context["cs_features"]
        labels: pd.Series = context["cs_labels"]

        # Align features and labels on (date, ticker)
        combined = features.join(labels, how="inner").dropna()
        assert len(combined) > 0, "No rows remain after aligning features and labels"

        X = combined.drop(columns=["label"])
        y = combined["label"]

        # Temporal split on unique dates
        X_train, X_test, y_train, y_test = self._temporal_split(X, y)

        train_start = X_train.index.get_level_values("date").min()
        train_end   = X_train.index.get_level_values("date").max()
        test_start  = X_test.index.get_level_values("date").min()
        test_end    = X_test.index.get_level_values("date").max()

        assert train_end < test_start, (
            f"Temporal leakage: train_end={train_end} >= test_start={test_start}"
        )

        model = self._fit(X_train, y_train)

        test_scores = pd.Series(
            model.predict(X_test.values),
            index=X_test.index,
            name="score",
        )

        ic = self._mean_ic(test_scores, y_test)
        if abs(ic) > _HIGH_IC_THRESHOLD:
            logger.warning(
                "RankingModelAgent: IC=%.4f exceeds threshold %.2f — "
                "verify for potential leakage before trusting results",
                ic,
                _HIGH_IC_THRESHOLD,
            )

        logger.info(
            "RankingModelAgent: IC=%.4f  train=%s->%s  test=%s->%s",
            ic,
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
        )

        context["cs_model"] = {
            "estimator": model,
            "features": list(X.columns),
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "ic": ic,
            "model_type": self.model_type if _HAS_LGB else "gradient_boosting",
        }
        context["cs_predictions"] = test_scores
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _temporal_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple:
        dates = X.index.get_level_values("date").unique().sort_values()
        split_idx = int(len(dates) * (1 - self.test_size))
        assert split_idx > 0 and split_idx < len(dates), (
            f"test_size={self.test_size} produces an empty train or test set"
        )
        train_dates = set(dates[:split_idx])
        test_dates = set(dates[split_idx:])

        mask_train = X.index.get_level_values("date").isin(train_dates)
        mask_test = X.index.get_level_values("date").isin(test_dates)

        return X[mask_train], X[mask_test], y[mask_train], y[mask_test]

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        if _HAS_LGB and self.model_type == "lightgbm":
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
        model.fit(X_train.values, y_train.values)
        return model

    @staticmethod
    def _mean_ic(scores: pd.Series, actual: pd.Series) -> float:
        """Average Spearman rank IC across test dates."""
        ics = []
        for date in scores.index.get_level_values("date").unique():
            try:
                s = scores.xs(date, level="date")
                a = actual.xs(date, level="date")
            except KeyError:
                continue
            aligned = pd.concat([s, a], axis=1).dropna()
            if len(aligned) >= 2:
                corr, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                if not np.isnan(corr):
                    ics.append(corr)
        return float(np.mean(ics)) if ics else 0.0
