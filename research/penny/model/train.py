"""train.py — meta-labelling model, trained strictly inside each fold.

Structure
---------
Meta-labelling (Lopez de Prado 2018, ch. 3) splits the decision in two:

* the **primary** rule is the candidate screen -- gap >= 10%, priced 0.50-10,
  liquid enough. It has high recall and no precision, which is exactly the
  failure round 1 measured.
* the **secondary** model, here, predicts whether a given signal will clear
  costs. Its job is precision, not forecasting.

The target is therefore binary -- "would this trade have cleared the cost bar"
-- not the return. That framing matters: a regressor trained on returns spends
its capacity fitting the enormous right tail (1.5% of candidates double
intraday), which is the part least likely to repeat.

Deliberately fixed hyperparameters
----------------------------------
No hyperparameter search. With ~10k training rows and a search already running
over feature sets and thresholds, tuning depth and leaf count inside each fold
would add a second layer of selection whose cost the multiple-testing
correction does not capture. Fixed, conservative, heavily-regularised settings
are the honest choice; the model either finds signal with them or the signal is
too fragile to trust.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.penny.eval.folds import Fold, split_frame

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Fixed, conservative settings. See the module docstring for why."""

    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 4
    min_child_samples: int = 80
    subsample: float = 0.7
    subsample_freq: int = 1
    colsample_bytree: float = 0.7
    reg_lambda: float = 10.0
    random_state: int = 42

    def to_lgb(self) -> Dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }


def train_one_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    config: Optional[ModelConfig] = None,
) -> Tuple[np.ndarray, Optional[pd.Series]]:
    """Fit on train, return P(clears cost) for test rows and feature importances."""
    from lightgbm import LGBMClassifier

    config = config or ModelConfig()

    x_train = train[list(feature_cols)].astype(float)
    y_train = train[label_col].astype(int)
    x_test = test[list(feature_cols)].astype(float)

    # A degenerate fold cannot be fitted; return the base rate rather than
    # crashing the whole walk-forward.
    if y_train.nunique() < 2 or len(train) < 200:
        logger.warning(
            "fold has %d rows and %d classes -- returning base rate",
            len(train), y_train.nunique(),
        )
        return np.full(len(test), float(y_train.mean() if len(y_train) else 0.0)), None

    model = LGBMClassifier(**config.to_lgb())
    model.fit(x_train, y_train)

    scores = model.predict_proba(x_test)[:, 1]
    importance = pd.Series(model.feature_importances_, index=list(feature_cols))
    return scores, importance


def walk_forward_scores(
    df: pd.DataFrame,
    folds: List[Fold],
    feature_cols: Sequence[str],
    label_col: str,
    config: Optional[ModelConfig] = None,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every test fold with a model trained only on its own train fold.

    Returns (scored out-of-sample rows, per-fold feature importances). The
    scored frame is the ONLY thing downstream may evaluate -- anything fitted
    and scored on the same rows is in-sample by construction.
    """
    scored: List[pd.DataFrame] = []
    importances: List[pd.Series] = []

    for fold in folds:
        train, test = split_frame(df, fold, date_col=date_col)
        if train.empty or test.empty:
            continue

        train = train.dropna(subset=[label_col])
        if train.empty:
            continue

        scores, importance = train_one_fold(
            train, test, feature_cols, label_col, config
        )
        out = test.copy()
        out["model_score"] = scores
        out["fold"] = fold.index
        scored.append(out)

        if importance is not None:
            importances.append(importance.rename(f"fold_{fold.index}"))

        logger.info(
            "fold %d: trained on %s rows (%.1f%% positive), scored %s",
            fold.index, f"{len(train):,}",
            100 * train[label_col].mean(), f"{len(test):,}",
        )

    if not scored:
        return pd.DataFrame(), pd.DataFrame()

    imp = pd.concat(importances, axis=1) if importances else pd.DataFrame()
    return pd.concat(scored, ignore_index=True), imp


def select_top_fraction(
    scored: pd.DataFrame, fraction: float, score_col: str = "model_score"
) -> pd.DataFrame:
    """Keep the highest-scoring ``fraction`` of rows, ranked within each fold.

    Ranking within fold rather than globally matters: score distributions drift
    between folds because each is produced by a different model, so a global
    threshold would silently allocate most of the selection to whichever fold
    happened to score high.
    """
    if scored.empty or not 0 < fraction <= 1:
        return scored.iloc[0:0]

    # Rank descending inside each fold and keep the head. Expressed as a
    # transform rather than groupby.apply, which pandas is deprecating and
    # which silently changes behaviour around the grouping columns.
    rank = scored.groupby("fold")[score_col].rank(method="first", ascending=False)
    keep_n = scored.groupby("fold")[score_col].transform("size").mul(fraction).round()
    keep_n = keep_n.clip(lower=1)
    return scored[rank <= keep_n].reset_index(drop=True)


def score_decile_table(
    scored: pd.DataFrame,
    return_col: str = "ret_c2c",
    score_col: str = "model_score",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Mean outcome by model-score decile.

    The single most informative diagnostic: a model with real signal produces a
    monotone gradient here. A model that merely fits noise produces a jagged
    one with a good top bin, which is what an unwary reader mistakes for edge.
    """
    df = scored.dropna(subset=[score_col, return_col]).copy()
    if df.empty:
        return pd.DataFrame()

    df["score_bin"] = pd.qcut(
        df[score_col].rank(method="first"), n_bins, labels=False, duplicates="drop"
    )
    table = (
        df.groupby("score_bin")
        .agg(
            n=(return_col, "size"),
            mean_gross=(return_col, "mean"),
            median_gross=(return_col, "median"),
            hit_rate=(return_col, lambda s: float((s > 0).mean())),
            mean_score=(score_col, "mean"),
        )
        .reset_index()
    )
    return table


def spearman_monotonicity(table: pd.DataFrame) -> float:
    """Rank correlation between score bin and mean outcome.

    Near +1 means the score orders outcomes; near 0 means the top bin is luck.
    """
    from scipy.stats import spearmanr

    if len(table) < 3:
        return float("nan")
    rho, _ = spearmanr(table["score_bin"], table["mean_gross"])
    return float(rho)
