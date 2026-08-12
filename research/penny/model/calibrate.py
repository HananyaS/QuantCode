"""calibrate.py — turn a model score into a decision.

Why a rank cut is the wrong decision rule
-----------------------------------------
Rounds 2 and 3 selected the top 10% of scores. That number was arbitrary, and
worse, it is *unconditional*: it takes the best tenth of whatever is in front of
it, whether or not any of them clear their own cost. A rank cut cannot abstain.

The decision the strategy actually faces is per-trade:

    take it  <=>  E[return] > cost of this particular trade

which needs two things a raw LightGBM score does not provide:

1. **A calibrated probability.** ``predict_proba`` is monotone in the right
   direction but is not a probability -- gradient boosting on an imbalanced
   target is systematically overconfident. Comparing an uncalibrated score to a
   cost in return units is a category error.
2. **A payoff model.** ``p`` alone is not enough; a 30% chance of +40% and a 30%
   chance of +2% are different trades.

So: isotonic calibration for ``p``, conditional moments for the payoff, and an
expected-value comparison against that trade's own ``cost_hat``.

The leakage hazard, and how it is handled
-----------------------------------------
Fitting the calibrator on the same rows the GBDT was fitted on is useless: those
scores are in-sample and wildly overconfident, so the isotonic map learned from
them does not describe test-fold scores at all. It would also be a subtle
*optimistic* bug rather than a crash.

Each training window is therefore split chronologically: the GBDT is fitted on
the earlier part, and the calibrator plus the payoff moments are fitted on the
later part, which is out-of-sample for the GBDT and closest in time to the test
fold. This mirrors the inner validation split already used by
``model.sequence.train_sequence_model``.

Nothing here ever sees a test-fold row. ``tests/test_penny_calibrate.py`` pins
that as an executable claim rather than a comment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.penny.eval.folds import Fold, split_frame
from research.penny.model.train import ModelConfig, train_one_fold

logger = logging.getLogger(__name__)

# Share of each training window reserved for calibration. Large enough that the
# isotonic fit is not itself noise, small enough to leave the GBDT real data.
CALIB_FRACTION = 0.30

# Below this many calibration rows, an isotonic fit is noise. Fall back to the
# identity map and say so, rather than producing a confident-looking artefact.
MIN_CALIB_ROWS = 200


@dataclass
class PayoffModel:
    """What a win and a loss are worth, estimated in-fold.

    ``loss_mean`` is stored as a positive magnitude; the expected-value formula
    subtracts it. Keeping the sign convention explicit here avoids the classic
    double-negative that silently turns a cost into a bonus.
    """

    win_mean: float
    loss_mean: float
    base_rate: float

    def expected_value(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return p * self.win_mean - (1.0 - p) * self.loss_mean


def fit_payoff(returns: pd.Series, labels: pd.Series) -> PayoffModel:
    """Conditional moments of the outcome, from training rows only."""
    r = pd.Series(returns).astype(float)
    y = pd.Series(labels).astype(float)
    ok = r.notna() & y.notna()
    r, y = r[ok], y[ok]

    if r.empty:
        return PayoffModel(0.0, 0.0, 0.0)

    wins = r[y > 0.5]
    losses = r[y <= 0.5]
    return PayoffModel(
        win_mean=float(wins.mean()) if len(wins) else 0.0,
        loss_mean=float(losses.abs().mean()) if len(losses) else 0.0,
        base_rate=float(y.mean()),
    )


def fit_calibrator(scores: np.ndarray, labels: np.ndarray):
    """Isotonic map from raw score to probability.

    Isotonic rather than Platt: the distortion in a boosted-tree score is not
    sigmoid-shaped, and with a few thousand rows the extra flexibility is
    affordable. ``out_of_bounds="clip"`` keeps test scores outside the training
    range mapped to the nearest fitted value instead of NaN.
    """
    from sklearn.isotonic import IsotonicRegression

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]

    if len(s) < MIN_CALIB_ROWS or len(np.unique(y)) < 2:
        logger.warning(
            "calibration set has %d rows and %d classes -- using identity map",
            len(s), len(np.unique(y)),
        )
        return None

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s, y)
    return iso


def apply_calibrator(calibrator, scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if calibrator is None:
        return np.clip(s, 0.0, 1.0)
    return np.clip(calibrator.predict(s), 0.0, 1.0)


def chronological_split(train: pd.DataFrame, date_col: str = "date",
                        calib_fraction: float = CALIB_FRACTION
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Earlier rows fit the model; later rows fit the calibrator.

    Split on whole *days*, not rows: a day straddling the boundary would put
    correlated observations on both sides, and same-day correlation is the
    dominant dependence in this panel.
    """
    ordered = train.sort_values(date_col)
    days = ordered[date_col].drop_duplicates().to_numpy()
    if len(days) < 4:
        return ordered, ordered.iloc[0:0]

    cut = int(round(len(days) * (1.0 - calib_fraction)))
    cut = max(1, min(cut, len(days) - 1))
    boundary = days[cut]
    return ordered[ordered[date_col] < boundary], ordered[ordered[date_col] >= boundary]


def walk_forward_ev(
    df: pd.DataFrame,
    folds: List[Fold],
    feature_cols: Sequence[str],
    label_col: str,
    return_col: str = "ret_c2c",
    config: Optional[ModelConfig] = None,
    date_col: str = "date",
    calib_fraction: float = CALIB_FRACTION,
) -> pd.DataFrame:
    """Walk-forward scores plus a calibrated probability and expected value.

    Adds ``model_score`` (raw), ``p_calib`` (calibrated) and ``ev`` (expected
    return under the in-fold payoff model). Selection against ``cost_hat``
    happens downstream in :func:`select_by_expected_value` so that the cost
    column, and therefore the order size, can vary without refitting anything.
    """
    scored: List[pd.DataFrame] = []

    for fold in folds:
        train, test = split_frame(df, fold, date_col=date_col)
        train = train.dropna(subset=[label_col])
        if train.empty or test.empty:
            continue

        fit_part, calib_part = chronological_split(train, date_col, calib_fraction)
        if fit_part.empty:
            continue

        # Score the calibration slice with a model that has not seen it.
        if not calib_part.empty:
            calib_scores, _ = train_one_fold(
                fit_part, calib_part, feature_cols, label_col, config
            )
            calibrator = fit_calibrator(calib_scores, calib_part[label_col].to_numpy())
            payoff = fit_payoff(calib_part[return_col], calib_part[label_col])
        else:
            calibrator, payoff = None, fit_payoff(train[return_col], train[label_col])

        # The deployed model uses the whole training window.
        test_scores, _ = train_one_fold(train, test, feature_cols, label_col, config)

        out = test.copy()
        out["model_score"] = test_scores
        out["p_calib"] = apply_calibrator(calibrator, test_scores)
        out["ev"] = payoff.expected_value(out["p_calib"].to_numpy())
        out["fold"] = fold.index
        out["payoff_win_mean"] = payoff.win_mean
        out["payoff_loss_mean"] = payoff.loss_mean
        scored.append(out)

        logger.info(
            "fold %d: fit %s rows, calibrated on %s, scored %s | "
            "E[r|win]=%+.3f E[|r||lose]=%.3f base=%.3f",
            fold.index, f"{len(fit_part):,}", f"{len(calib_part):,}",
            f"{len(test):,}", payoff.win_mean, payoff.loss_mean, payoff.base_rate,
        )

    if not scored:
        return pd.DataFrame()
    return pd.concat(scored, ignore_index=True)


def select_by_expected_value(
    scored: pd.DataFrame,
    cost_col: str,
    ev_col: str = "ev",
    margin: float = 0.0,
) -> pd.Series:
    """Take a trade when its expected value clears its own cost.

    Unlike a rank cut this can select nothing at all, which is the point: on a
    day where no candidate clears its cost, the correct number of trades is
    zero. Recall is an output here, not a parameter.
    """
    if scored.empty:
        return pd.Series(dtype=bool)
    ev = scored[ev_col].astype(float)
    cost = scored[cost_col].astype(float)
    return (ev > cost + margin).fillna(False)


def calibration_table(scored: pd.DataFrame, label_col: str,
                      prob_col: str = "p_calib", n_bins: int = 10) -> pd.DataFrame:
    """Predicted probability against realised frequency, by bin.

    The diagnostic that says whether ``p_calib`` means anything. A calibrated
    model puts ``predicted`` and ``realised`` on the diagonal; a rank cut never
    had to make that claim, and this rule does.
    """
    df = scored.dropna(subset=[prob_col, label_col]).copy()
    if df.empty:
        return pd.DataFrame()

    df["bin"] = pd.qcut(
        df[prob_col].rank(method="first"), n_bins, labels=False, duplicates="drop"
    )
    return (
        df.groupby("bin")
        .agg(
            n=(prob_col, "size"),
            predicted=(prob_col, "mean"),
            realised=(label_col, "mean"),
        )
        .reset_index()
    )
