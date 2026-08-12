"""Tests for calibrated probabilities and the expected-value decision rule.

Two failures are guarded here, and both would flatter the result rather than
crash:

* **Calibrating on the model's own training rows.** Those scores are in-sample
  and overconfident, so the isotonic map fitted to them does not describe test
  scores. Nothing raises; the numbers just come out better than they are.
* **Sign confusion in the payoff model.** ``loss_mean`` is stored as a positive
  magnitude and subtracted. Store it negative and subtract it and every losing
  trade becomes a reason to trade.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.eval.folds import Fold
from research.penny.model.calibrate import (
    MIN_CALIB_ROWS,
    PayoffModel,
    apply_calibrator,
    calibration_table,
    chronological_split,
    fit_calibrator,
    fit_payoff,
    select_by_expected_value,
    walk_forward_ev,
)


def _panel(n_days=200, per_day=12, seed=0):
    """A panel with genuine signal: feature x0 raises P(win)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for d in days:
        for _ in range(per_day):
            x0 = rng.normal()
            p = 1.0 / (1.0 + np.exp(-1.2 * x0))
            win = rng.random() < p
            ret = rng.normal(0.08, 0.05) if win else rng.normal(-0.03, 0.04)
            rows.append({"date": d, "symbol": f"S{rng.integers(0, 40)}",
                         "x0": x0, "x1": rng.normal(), "ret_c2c": ret,
                         "meta_label": int(win)})
    return pd.DataFrame(rows)


def _folds(df, n=2, embargo=3):
    days = np.sort(df["date"].unique())
    out, block = [], len(days) // (n + 1)
    for i in range(n):
        tr_end = block * (i + 1)
        te = days[tr_end + embargo: tr_end + embargo + block]
        if not len(te):
            continue
        out.append(Fold(
            index=i,
            train_dates=pd.DatetimeIndex(days[:tr_end]),
            test_dates=pd.DatetimeIndex(te),
            embargoed_dates=pd.DatetimeIndex(days[tr_end: tr_end + embargo]),
        ))
    return out


# ---- Payoff model ----


def test_payoff_stores_loss_as_a_positive_magnitude():
    """The sign convention that a double negative would invert."""
    r = pd.Series([0.10, 0.12, -0.05, -0.03])
    y = pd.Series([1, 1, 0, 0])
    p = fit_payoff(r, y)
    assert p.win_mean == pytest.approx(0.11)
    assert p.loss_mean == pytest.approx(0.04), "losses must be a positive magnitude"
    assert p.base_rate == pytest.approx(0.5)


def test_expected_value_is_negative_when_losing_is_likely():
    p = PayoffModel(win_mean=0.10, loss_mean=0.05, base_rate=0.2)
    assert p.expected_value(np.array([0.1]))[0] < 0


def test_expected_value_is_positive_when_winning_is_likely():
    p = PayoffModel(win_mean=0.10, loss_mean=0.05, base_rate=0.2)
    assert p.expected_value(np.array([0.9]))[0] > 0


def test_expected_value_crosses_zero_at_the_breakeven_probability():
    p = PayoffModel(win_mean=0.10, loss_mean=0.05, base_rate=0.2)
    breakeven = 0.05 / (0.10 + 0.05)
    assert p.expected_value(np.array([breakeven]))[0] == pytest.approx(0.0, abs=1e-12)


def test_payoff_survives_a_class_with_no_members():
    p = fit_payoff(pd.Series([0.1, 0.2]), pd.Series([1, 1]))
    assert p.loss_mean == 0.0 and p.win_mean > 0


def test_payoff_ignores_nan_returns():
    p = fit_payoff(pd.Series([0.1, np.nan, -0.2]), pd.Series([1, 1, 0]))
    assert p.win_mean == pytest.approx(0.1)


# ---- Isotonic calibration ----


def test_calibrator_maps_scores_toward_realised_frequency():
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 1, 4_000)
    # True probability is half the score: a systematically overconfident model.
    y = (rng.random(4_000) < s / 2).astype(int)
    iso = fit_calibrator(s, y)
    out = apply_calibrator(iso, np.array([0.8]))
    assert 0.25 < out[0] < 0.55, f"calibration did not correct overconfidence: {out[0]}"


def test_calibrator_output_is_bounded_to_the_unit_interval():
    rng = np.random.default_rng(2)
    s = rng.uniform(0.2, 0.8, 1_000)
    y = (rng.random(1_000) < s).astype(int)
    iso = fit_calibrator(s, y)
    out = apply_calibrator(iso, np.array([-5.0, 0.5, 5.0]))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_too_few_rows_falls_back_to_identity_rather_than_fitting_noise():
    rng = np.random.default_rng(3)
    n = MIN_CALIB_ROWS - 1
    assert fit_calibrator(rng.uniform(0, 1, n), rng.integers(0, 2, n)) is None


def test_single_class_falls_back_to_identity():
    rng = np.random.default_rng(4)
    assert fit_calibrator(rng.uniform(0, 1, 1_000), np.ones(1_000)) is None


def test_identity_fallback_still_clips():
    out = apply_calibrator(None, np.array([-1.0, 0.3, 2.0]))
    assert list(out) == [0.0, 0.3, 1.0]


# ---- The chronological split ----


def test_split_puts_earlier_days_in_the_fitting_part():
    df = _panel(n_days=40, per_day=3)
    fit, calib = chronological_split(df)
    assert fit["date"].max() < calib["date"].min(), "the split is not chronological"


def test_split_never_puts_one_day_on_both_sides():
    """Same-day correlation is the dominant dependence in this panel."""
    df = _panel(n_days=40, per_day=6)
    fit, calib = chronological_split(df)
    assert not (set(fit["date"]) & set(calib["date"]))


def test_split_reserves_roughly_the_requested_fraction():
    df = _panel(n_days=100, per_day=4)
    fit, calib = chronological_split(df, calib_fraction=0.30)
    share = calib["date"].nunique() / df["date"].nunique()
    assert 0.25 <= share <= 0.35


def test_split_degrades_gracefully_on_a_tiny_window():
    df = _panel(n_days=3, per_day=2)
    fit, calib = chronological_split(df)
    assert len(fit) == len(df) and calib.empty


# ---- Walk-forward: the leakage guarantee ----


def test_no_test_fold_row_is_ever_used_to_fit_that_folds_model():
    """The load-bearing claim of the module, as an executable assertion.

    Scoped **per fold**, deliberately. Walk-forward is expanding-window, so
    fold 1 trains on days that were fold 0's test period -- that is the design,
    not a leak: by the time fold 1 is fitted, those days are in the past. The
    violation that matters is a fold being fitted on its *own* test days.
    """
    df = _panel()
    folds = _folds(df)
    fitted_on: list = []

    import research.penny.model.calibrate as C

    real = C.train_one_fold

    def spy(train, test, *a, **k):
        fitted_on.append(set(pd.to_datetime(train["date"]).unique()))
        return real(train, test, *a, **k)

    C.train_one_fold = spy
    try:
        walk_forward_ev(df, folds, ["x0", "x1"], "meta_label")
    finally:
        C.train_one_fold = real

    # Two fits per fold: the inner calibration fit, then the deployed model.
    assert len(fitted_on) == 2 * len(folds)
    for i, fold in enumerate(folds):
        own_test = set(pd.to_datetime(fold.test_dates))
        for used in fitted_on[2 * i: 2 * i + 2]:
            assert not (used & own_test), f"fold {fold.index} was fitted on its own test days"


def test_the_calibrator_is_never_fitted_on_the_models_own_training_rows():
    """The subtle, flattering failure: in-sample scores are overconfident.

    The inner fit must be strictly earlier than the calibration slice, so the
    scores the isotonic map learns from are genuinely out-of-sample.
    """
    df = _panel()
    folds = _folds(df)
    pairs: list = []

    import research.penny.model.calibrate as C

    real = C.train_one_fold

    def spy(train, test, *a, **k):
        pairs.append((set(pd.to_datetime(train["date"]).unique()),
                      set(pd.to_datetime(test["date"]).unique())))
        return real(train, test, *a, **k)

    C.train_one_fold = spy
    try:
        walk_forward_ev(df, folds, ["x0", "x1"], "meta_label")
    finally:
        C.train_one_fold = real

    for i in range(0, len(pairs), 2):
        inner_train, calib_rows = pairs[i]
        assert not (inner_train & calib_rows), (
            "the calibration slice was scored by a model fitted on it"
        )
        assert max(inner_train) < min(calib_rows), "the inner split is not chronological"


def test_walk_forward_emits_calibrated_probability_and_ev():
    df = _panel()
    out = walk_forward_ev(df, _folds(df), ["x0", "x1"], "meta_label")
    assert {"model_score", "p_calib", "ev", "fold"} <= set(out.columns)
    assert out["p_calib"].between(0, 1).all()


def test_ev_is_monotone_in_the_calibrated_probability_within_a_fold():
    """Monotone *within* a fold only.

    The payoff moments are estimated per fold, so E[r|win] and E[|r||lose]
    differ between folds and a pooled sort is not monotone. That is correct
    behaviour -- each fold prices its own trades using only its own past -- and
    asserting it pooled would be asserting the folds share a payoff model they
    deliberately do not.
    """
    df = _panel()
    out = walk_forward_ev(df, _folds(df), ["x0", "x1"], "meta_label")
    assert out["fold"].nunique() >= 2, "test needs multiple folds to be meaningful"
    for _, part in out.groupby("fold"):
        o = part.sort_values("p_calib")
        assert o["ev"].is_monotonic_increasing


def test_scores_only_cover_test_folds():
    df = _panel()
    folds = _folds(df)
    out = walk_forward_ev(df, folds, ["x0", "x1"], "meta_label")
    expected = set()
    for f in folds:
        expected.update(f.test_dates)
    assert set(out["date"].unique()) <= expected


def test_payoff_moments_are_recorded_per_fold_for_audit():
    df = _panel()
    out = walk_forward_ev(df, _folds(df), ["x0", "x1"], "meta_label")
    assert out.groupby("fold")["payoff_win_mean"].nunique().max() == 1
    assert (out["payoff_loss_mean"] >= 0).all()


# ---- The decision rule ----


def test_rule_takes_only_trades_whose_ev_clears_their_own_cost():
    scored = pd.DataFrame({"ev": [0.05, 0.01, 0.03], "cost_hat": [0.02, 0.02, 0.04]})
    assert list(select_by_expected_value(scored, "cost_hat")) == [True, False, False]


def test_rule_can_select_nothing_at_all():
    """The property a rank cut structurally cannot have."""
    scored = pd.DataFrame({"ev": [-0.01, -0.02], "cost_hat": [0.02, 0.02]})
    assert select_by_expected_value(scored, "cost_hat").sum() == 0


def test_a_higher_cost_selects_strictly_fewer_trades():
    scored = pd.DataFrame({"ev": [0.05, 0.03, 0.01], "cost_hat": [0.02] * 3})
    cheap = select_by_expected_value(scored, "cost_hat").sum()
    scored["cost_hat"] = 0.04
    assert select_by_expected_value(scored, "cost_hat").sum() < cheap


def test_margin_tightens_the_rule():
    scored = pd.DataFrame({"ev": [0.03], "cost_hat": [0.02]})
    assert bool(select_by_expected_value(scored, "cost_hat").iloc[0])
    assert not bool(select_by_expected_value(scored, "cost_hat", margin=0.02).iloc[0])


def test_missing_cost_is_not_traded():
    scored = pd.DataFrame({"ev": [0.05], "cost_hat": [np.nan]})
    assert select_by_expected_value(scored, "cost_hat").sum() == 0


def test_empty_input_returns_empty_mask():
    assert select_by_expected_value(pd.DataFrame(), "cost_hat").empty


# ---- Calibration diagnostic ----


def test_calibration_table_puts_a_calibrated_model_near_the_diagonal():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, 6_000)
    scored = pd.DataFrame({"p_calib": p, "meta_label": (rng.random(6_000) < p).astype(int)})
    tab = calibration_table(scored, "meta_label")
    assert (tab["predicted"] - tab["realised"]).abs().max() < 0.10


def test_calibration_table_exposes_a_miscalibrated_model():
    rng = np.random.default_rng(6)
    p = rng.uniform(0.05, 0.95, 6_000)
    scored = pd.DataFrame({"p_calib": p, "meta_label": (rng.random(6_000) < p / 3).astype(int)})
    tab = calibration_table(scored, "meta_label")
    assert (tab["predicted"] - tab["realised"]).abs().max() > 0.20
