"""cascade.py — staged selection: pool first, exact opportunities last.

Why a cascade rather than one model
-----------------------------------
Rounds 1-4 used a single model over a single hand-fixed screen. That structure
has two defects the results kept running into.

**One model must serve two incompatible jobs.** Deciding *which names are worth
looking at* and deciding *which of those to buy today* are different questions
with different features and different natural populations. A single fit trades
one off against the other and does neither well.

**Expensive features are wasted on names that will never be traded.** Premarket
sequences and 09:34 microstructure cost a fetch per symbol-day. Computing them
across the whole pool to use 1% of it is backwards; a cascade spends them only
on survivors.

So: each stage has its own feature set, fits on the population its predecessor
would have handed it, and passes a shrinking survivor set downward.

The leakage hazard specific to a cascade
----------------------------------------
Fitting stage 2 on the survivors of a stage-1 model that has already seen the
test fold contaminates every later stage, and the contamination compounds
silently. So within each fold the chain is:

    fit stage k on TRAIN survivors -> score TEST survivors -> select both
    independently, by the same rule

Stage k's training population is derived only from training rows. Nothing about
the test fold reaches any stage. ``tests/test_penny_cascade.py`` pins it.

Objectives
----------
Three, because round 4 showed the binary "clears a cost bar" target is a poor
fit for this problem:

``binary``   P(return exceeds a FIXED cost). Never a per-trade threshold: round
             4 labelled ``ret > cost_hat`` and the model learned to minimise the
             hurdle instead of maximising the return.
``rank``     LambdaRank with **the session as the group**. This is the question a
             daily strategy actually asks -- "which of today's candidates are
             best" -- and it is invariant to the level shifts that make a
             classifier's scores incomparable across regimes.
``quantile`` Regression on the 0.9 quantile of the return. Predicts the right
             tail directly, which is where this population's payoff lives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.penny.eval.folds import Fold, split_frame

logger = logging.getLogger(__name__)

OBJECTIVES = ("binary", "rank", "quantile")


@dataclass
class Stage:
    """One rung of the cascade."""

    name: str
    features: Sequence[str]
    keep_fraction: float = 0.5
    objective: str = "binary"
    quantile_alpha: float = 0.9
    min_train_rows: int = 500

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}; use {OBJECTIVES}")
        if not 0 < self.keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")


_COMMON = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4,
               min_child_samples=80, subsample=0.7, subsample_freq=1,
               colsample_bytree=0.7, reg_lambda=10.0, random_state=42,
               n_jobs=-1, verbose=-1)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stage: Stage,
    label_col: str,
    return_col: str,
    date_col: str,
) -> np.ndarray:
    """Fit one stage on ``train`` and score ``test``. Higher is better, always."""
    cols = [c for c in stage.features if c in train.columns]
    if not cols:
        raise ValueError(f"stage {stage.name}: no features present")

    x_train, x_test = train[cols].astype(float), test[cols].astype(float)

    if stage.objective == "binary":
        from lightgbm import LGBMClassifier

        y = train[label_col].astype(int)
        if y.nunique() < 2:
            return np.full(len(test), float(y.mean() if len(y) else 0.0))
        model = LGBMClassifier(**_COMMON)
        model.fit(x_train, y)
        return model.predict_proba(x_test)[:, 1]

    if stage.objective == "quantile":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(objective="quantile", alpha=stage.quantile_alpha,
                              **_COMMON)
        model.fit(x_train, train[return_col].astype(float))
        return model.predict(x_test)

    # rank: LambdaRank, grouped by session.
    from lightgbm import LGBMRanker

    tr = train.sort_values(date_col)
    x_tr = tr[cols].astype(float)
    groups = tr.groupby(date_col, sort=False).size().to_numpy()

    # LambdaRank needs small non-negative integer relevance. Within-day quartile
    # of the return: ordinal, bounded, and computed only from training rows.
    rel = (
        tr.groupby(date_col)[return_col]
        .transform(lambda s: pd.qcut(s.rank(method="first"), min(4, max(1, s.nunique())),
                                     labels=False, duplicates="drop"))
        .fillna(0).astype(int)
    )
    if rel.nunique() < 2:
        return np.zeros(len(test))

    model = LGBMRanker(objective="lambdarank", **_COMMON)
    model.fit(x_tr, rel, group=groups)
    return model.predict(x_test)


def _select(scored: pd.DataFrame, fraction: float, score_col: str,
            fold_col: Optional[str] = None) -> np.ndarray:
    """Boolean mask keeping the top ``fraction`` by score.

    Ranked within fold when a fold column is present: score distributions drift
    between folds because each comes from a different model, so a pooled
    threshold would hand most of the selection to whichever fold scored high.
    """
    if scored.empty:
        return np.zeros(0, dtype=bool)
    if fraction >= 1.0:
        return np.ones(len(scored), dtype=bool)

    if fold_col and fold_col in scored.columns:
        rank = scored.groupby(fold_col)[score_col].rank(method="first", ascending=False)
        keep_n = scored.groupby(fold_col)[score_col].transform("size").mul(fraction).round()
    else:
        rank = scored[score_col].rank(method="first", ascending=False)
        keep_n = pd.Series(round(len(scored) * fraction), index=scored.index)
    return (rank <= keep_n.clip(lower=1)).to_numpy()


def run_cascade(
    df: pd.DataFrame,
    folds: List[Fold],
    stages: Sequence[Stage],
    label_col: str = "meta_label",
    return_col: str = "ret_c2c",
    date_col: str = "date",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Run the staged selection walk-forward.

    Returns the surviving test-fold rows -- carrying one ``score_<stage>`` column
    per stage -- and a per-stage trace of how the population narrowed.
    """
    # Validate configuration before touching data. A stage naming columns that
    # do not exist is a configuration error, and it must not be reported as a
    # thin-fold warning -- which is what happens if the row-count guard fires
    # first and quietly stops the fold.
    for stage in stages:
        present = [c for c in stage.features if c in df.columns]
        if not present:
            raise ValueError(
                f"stage {stage.name}: no features present in the frame "
                f"(asked for {list(stage.features)[:5]})"
            )
        missing = set(stage.features) - set(df.columns)
        if missing:
            logger.warning("stage %s: %d features absent, using %d: %s",
                           stage.name, len(missing), len(present),
                           sorted(missing)[:5])

    survivors: List[pd.DataFrame] = []
    trace: List[Dict] = []

    for fold in folds:
        train, test = split_frame(df, fold, date_col=date_col)
        train = train.dropna(subset=[label_col, return_col])
        if train.empty or test.empty:
            continue

        for stage in stages:
            if len(train) < stage.min_train_rows or test.empty:
                logger.warning("fold %d stage %s: %d train / %d test rows -- stopping",
                               fold.index, stage.name, len(train), len(test))
                test = test.iloc[0:0]
                break

            # Score the training population with an in-fold model purely to
            # decide who advances. This is in-sample by construction and is used
            # ONLY to narrow the next stage's training set, never to evaluate.
            train_scores = _fit_predict(train, train, stage, label_col,
                                        return_col, date_col)
            test_scores = _fit_predict(train, test, stage, label_col,
                                       return_col, date_col)

            train = train.assign(**{f"score_{stage.name}": train_scores})
            test = test.assign(**{f"score_{stage.name}": test_scores})

            keep_tr = _select(train, stage.keep_fraction, f"score_{stage.name}")
            keep_te = _select(test, stage.keep_fraction, f"score_{stage.name}")

            trace.append({
                "fold": fold.index, "stage": stage.name,
                "objective": stage.objective,
                "train_in": int(len(train)), "train_out": int(keep_tr.sum()),
                "test_in": int(len(test)), "test_out": int(keep_te.sum()),
                "keep_fraction": stage.keep_fraction,
                "test_gross_in": float(test[return_col].mean()),
                "test_gross_out": float(test.loc[keep_te, return_col].mean())
                if keep_te.any() else float("nan"),
            })

            train, test = train[keep_tr], test[keep_te]

        if not test.empty:
            survivors.append(test.assign(fold=fold.index))

    if not survivors:
        return pd.DataFrame(), trace
    return pd.concat(survivors, ignore_index=True), trace


def trace_table(trace: List[Dict]) -> pd.DataFrame:
    """Per-stage funnel, averaged across folds — did each stage add anything?"""
    if not trace:
        return pd.DataFrame()
    t = pd.DataFrame(trace)
    out = t.groupby("stage", sort=False).agg(
        objective=("objective", "first"),
        folds=("fold", "nunique"),
        test_in=("test_in", "sum"),
        test_out=("test_out", "sum"),
        gross_in=("test_gross_in", "mean"),
        gross_out=("test_gross_out", "mean"),
    ).reset_index()
    out["lift"] = out["gross_out"] - out["gross_in"]
    out["kept"] = out["test_out"] / out["test_in"].replace(0, np.nan)
    return out
