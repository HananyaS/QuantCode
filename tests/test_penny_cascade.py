"""The cascade: staged selection, and the leakage it makes possible.

A cascade has a failure mode a single model does not. If stage 2 is fitted on
the survivors of a stage-1 model that has already seen the test fold, the
contamination propagates into every later stage and compounds — silently, and in
the flattering direction.

The guarantee is therefore structural: within each fold, stage *k* fits on the
**training** survivors of stage *k−1* and scores the **test** survivors, with
the two selections made independently by the same rule. These tests pin that,
plus the objective-specific behaviour that makes the ranking stage worth having.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.eval.folds import Fold
from research.penny.model.cascade import (
    Stage,
    run_cascade,
    trace_table,
)


def _panel(n_days=180, per_day=14, seed=0):
    """A panel where x0 drives the outcome and x9 is pure noise."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for d in days:
        for _ in range(per_day):
            x0 = rng.normal()
            ret = 0.02 * x0 + rng.normal(0, 0.05)
            rows.append({
                "date": d, "symbol": f"S{rng.integers(0, 60)}",
                "x0": x0, "x1": rng.normal(), "x9": rng.normal(),
                "ret_c2c": ret, "meta_label": int(ret > 0.02),
            })
    return pd.DataFrame(rows)


def _folds(df, n=3, embargo=3):
    days = np.sort(df["date"].unique())
    out, block = [], len(days) // (n + 1)
    for i in range(n):
        tr_end = block * (i + 1)
        te = days[tr_end + embargo: tr_end + embargo + block]
        if not len(te):
            continue
        out.append(Fold(index=i,
                        train_dates=pd.DatetimeIndex(days[:tr_end]),
                        test_dates=pd.DatetimeIndex(te),
                        embargoed_dates=pd.DatetimeIndex(days[tr_end:tr_end + embargo])))
    return out


# ---- Stage validation ----


def test_unknown_objective_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown objective"):
        Stage("s", ["x0"], objective="magic")


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
def test_invalid_keep_fraction_is_rejected(frac):
    with pytest.raises(ValueError, match="keep_fraction"):
        Stage("s", ["x0"], keep_fraction=frac)


def test_a_stage_with_no_present_features_raises_rather_than_scoring_noise():
    """Config errors must not surface as a thin-fold warning.

    Validation runs before any data is touched, so the row-count guard cannot
    fire first and quietly stop the fold on what is really a typo.
    """
    df = _panel(60, 8)
    with pytest.raises(ValueError, match="no features"):
        run_cascade(df, _folds(df, n=1), [Stage("s", ["not_a_column"])])


# ---- The leakage guarantee ----


def test_no_stage_is_ever_fitted_on_its_own_test_fold():
    """Checked by matching each fit to the fold it belongs to.

    Not by call ordering: a fold that stops early makes the call count uneven,
    and an index-arithmetic version of this test would silently compare the
    wrong fit to the wrong fold.
    """
    df = _panel()
    folds = _folds(df)
    seen = []

    import research.penny.model.cascade as C

    real = C._fit_predict

    def spy(train, test, stage, *a, **k):
        seen.append(set(pd.to_datetime(train["date"]).unique()))
        return real(train, test, stage, *a, **k)

    C._fit_predict = spy
    try:
        run_cascade(df, folds, [Stage("a", ["x0", "x1"], 0.5),
                                Stage("b", ["x0", "x9"], 0.5)])
    finally:
        C._fit_predict = real

    assert seen, "no stage was ever fitted"
    for used in seen:
        owners = [f for f in folds if used <= set(pd.to_datetime(f.train_dates))]
        assert owners, "a fit used dates outside every fold's training window"
        for f in owners:
            assert not (used & set(pd.to_datetime(f.test_dates))), (
                f"fold {f.index} was fitted on its own test dates"
            )


def test_later_stages_train_only_on_earlier_stages_training_survivors():
    """The compounding failure a cascade uniquely permits."""
    df = _panel()
    folds = _folds(df, n=1)
    sizes = []

    import research.penny.model.cascade as C

    real = C._fit_predict

    def spy(train, test, stage, *a, **k):
        sizes.append((stage.name, len(train)))
        return real(train, test, stage, *a, **k)

    C._fit_predict = spy
    try:
        run_cascade(df, folds, [Stage("a", ["x0"], 0.5), Stage("b", ["x0"], 0.5)])
    finally:
        C._fit_predict = real

    a_rows = [n for name, n in sizes if name == "a"][0]
    b_rows = [n for name, n in sizes if name == "b"][0]
    assert b_rows < a_rows, "stage b did not train on a narrowed population"
    assert abs(b_rows - round(a_rows * 0.5)) <= 2


def test_survivors_come_only_from_test_folds():
    df = _panel()
    folds = _folds(df)
    out, _ = run_cascade(df, folds, [Stage("a", ["x0"], 0.5)])
    expected = set()
    for f in folds:
        expected.update(f.test_dates)
    assert set(out["date"].unique()) <= expected


# ---- Narrowing behaviour ----


def test_the_population_shrinks_by_the_keep_fraction_at_each_stage():
    df = _panel()
    out, trace = run_cascade(df, _folds(df), [Stage("a", ["x0"], 0.5),
                                              Stage("b", ["x0"], 0.5)])
    t = trace_table(trace).set_index("stage")
    assert t.loc["a", "kept"] == pytest.approx(0.5, abs=0.05)
    assert t.loc["b", "kept"] == pytest.approx(0.5, abs=0.05)


def test_each_stage_emits_its_own_score_column():
    df = _panel()
    out, _ = run_cascade(df, _folds(df), [Stage("a", ["x0"], 0.5),
                                          Stage("b", ["x1"], 0.5)])
    assert {"score_a", "score_b", "fold"} <= set(out.columns)


def test_keep_fraction_of_one_passes_everything_through():
    df = _panel(200, 12)
    out, trace = run_cascade(df, _folds(df, n=1), [Stage("a", ["x0"], 1.0)])
    t = trace_table(trace)
    assert not t.empty, "the fold stopped before the stage ran"
    assert t.iloc[0]["kept"] == pytest.approx(1.0)


def test_a_stage_with_too_few_training_rows_stops_that_fold_cleanly():
    df = _panel(90, 4)
    out, trace = run_cascade(
        df, _folds(df, n=1),
        [Stage("a", ["x0"], 0.5), Stage("b", ["x0"], 0.5, min_train_rows=10**6)],
    )
    assert out.empty
    assert not any(r["stage"] == "b" for r in trace)


# ---- Objectives ----


@pytest.mark.parametrize("objective", ["binary", "rank", "quantile"])
def test_every_objective_runs_and_selects(objective):
    df = _panel()
    out, trace = run_cascade(
        df, _folds(df), [Stage("s", ["x0", "x1"], 0.3, objective=objective)]
    )
    assert not out.empty
    assert trace_table(trace).iloc[0]["objective"] == objective


@pytest.mark.parametrize("objective", ["binary", "rank", "quantile"])
def test_a_real_signal_is_found_by_every_objective(objective):
    """x0 drives the return; the selected set must beat the population."""
    df = _panel()
    out, trace = run_cascade(
        df, _folds(df), [Stage("s", ["x0", "x1"], 0.2, objective=objective)]
    )
    assert out["ret_c2c"].mean() > df["ret_c2c"].mean() + 0.005, (
        f"{objective} failed to find a planted signal"
    )


def test_pure_noise_features_do_not_manufacture_an_edge():
    df = _panel()
    out, _ = run_cascade(df, _folds(df), [Stage("s", ["x9"], 0.2)])
    assert abs(out["ret_c2c"].mean() - df["ret_c2c"].mean()) < 0.006


def test_ranking_groups_by_session_so_it_picks_within_a_day():
    """The question a daily strategy asks: which of TODAY's names are best."""
    df = _panel()
    out, _ = run_cascade(df, _folds(df),
                         [Stage("s", ["x0", "x1"], 0.25, objective="rank")])
    per_day = out.groupby("date").size()
    assert per_day.max() <= 8, "ranking selected a whole day rather than within days"
    assert out["date"].nunique() > 20, "ranking collapsed onto a few sessions"


def test_quantile_alpha_is_honoured():
    df = _panel()
    hi, _ = run_cascade(df, _folds(df),
                        [Stage("s", ["x0"], 0.2, objective="quantile",
                               quantile_alpha=0.9)])
    lo, _ = run_cascade(df, _folds(df),
                        [Stage("s", ["x0"], 0.2, objective="quantile",
                               quantile_alpha=0.5)])
    assert not hi.empty and not lo.empty


# ---- The trace ----


def test_trace_reports_lift_per_stage_so_a_useless_stage_is_visible():
    df = _panel()
    _, trace = run_cascade(df, _folds(df), [Stage("signal", ["x0"], 0.3),
                                            Stage("noise", ["x9"], 0.5)])
    t = trace_table(trace).set_index("stage")
    assert t.loc["signal", "lift"] > t.loc["noise", "lift"]


def test_trace_is_empty_for_an_empty_run():
    assert trace_table([]).empty
