"""Tests for the conditional-expectation surface and its null tests.

The three guards being verified here are the ones standing between a veto
search and a fabricated result: in-fold selection, a selectivity-matched
baseline, and a search-size-aware correction.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.eval import conditional as cx
from research.penny.eval import nulls
from research.penny.eval.folds import walk_forward_folds


def _trades(n_days=300, per_day=12, seed=0, edge_when_flag=False):
    """Synthetic trade table with two veto columns.

    ``veto_bad`` marks a genuinely toxic subset when ``edge_when_flag`` is set;
    ``veto_noise`` is pure noise and must never be selected.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for d in dates:
        for j in range(per_day):
            bad = rng.random() < 0.30
            noise = rng.random() < 0.30
            if edge_when_flag:
                mean = -0.08 if bad else 0.02
            else:
                mean = 0.0
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{j}",
                    "net_return": rng.normal(mean, 0.05),
                    "outcome": "target" if rng.random() < 0.3 else "stop",
                    "holding_minutes": 60,
                    "veto_bad": bad,
                    "veto_noise": noise,
                }
            )
    return pd.DataFrame(rows)


VETOES = ["veto_bad", "veto_noise"]


# ---- Veto mechanics ----


def test_apply_vetoes_keeps_only_unflagged_trades():
    df = pd.DataFrame({"veto_a": [True, False, False], "veto_b": [False, True, False]})
    mask = cx.apply_vetoes(df, ["veto_a", "veto_b"])
    assert mask.tolist() == [False, False, True]


def test_no_vetoes_keeps_everything():
    df = pd.DataFrame({"veto_a": [True, False]})
    assert cx.apply_vetoes(df, []).all()


def test_vetoes_are_a_union_not_an_intersection():
    """Any single veto firing rejects the trade."""
    df = pd.DataFrame({"veto_a": [True], "veto_b": [False]})
    assert not cx.apply_vetoes(df, ["veto_a", "veto_b"]).iloc[0]


def test_enumerate_veto_sets_includes_empty_and_respects_max_size():
    sets = cx.enumerate_veto_sets(["a", "b", "c"], max_size=2)
    assert () in sets
    assert ("a", "b") in sets
    assert all(len(s) <= 2 for s in sets)
    assert len(sets) == 1 + 3 + 3


# ---- The surface ----


def test_a_real_veto_is_found_and_improves_the_mean():
    trades = _trades(edge_when_flag=True, seed=1)
    table = cx.search_veto_sets(trades, VETOES, max_size=2, min_trades=100, n_boot=300)
    assert not table.empty
    best = table.iloc[0]
    assert "veto_bad" in best["vetoes"]
    baseline = table[table["name"] == "(no veto)"].iloc[0]
    assert best["mean_net"] > baseline["mean_net"]


def test_pure_noise_yields_no_survivors_after_correction():
    trades = _trades(edge_when_flag=False, seed=2)
    table = cx.search_veto_sets(trades, VETOES, max_size=2, min_trades=100, n_boot=300)
    assert table["q_pass"].sum() == 0, "a null panel must produce no FDR survivors"


def test_search_reports_the_number_of_hypotheses_tested():
    """An FDR quoted without the search size is meaningless."""
    trades = _trades(seed=3)
    table = cx.search_veto_sets(trades, VETOES, max_size=2, min_trades=100, n_boot=200)
    assert (table["n_hypotheses"] == len(table)).all()
    assert table["n_hypotheses"].iloc[0] > 1


def test_tiny_populations_are_excluded_from_the_search():
    """A filter keeping eleven trades can post a spectacular mean."""
    trades = _trades(seed=4)
    trades["veto_almost_all"] = np.arange(len(trades)) % 100 != 0
    table = cx.search_veto_sets(
        trades, VETOES + ["veto_almost_all"], max_size=1,
        min_trades=500, min_retained=0.10, n_boot=200,
    )
    assert not any("veto_almost_all" in v for v in table["vetoes"])


def test_retained_fraction_is_relative_to_the_baseline():
    trades = _trades(seed=5)
    table = cx.search_veto_sets(trades, VETOES, max_size=1, min_trades=100, n_boot=200)
    baseline = table[table["name"] == "(no veto)"].iloc[0]
    assert baseline["retained_frac"] == pytest.approx(1.0)
    assert (table["retained_frac"] <= 1.0 + 1e-9).all()


# ---- The fair baseline ----


def test_random_placebo_centres_on_the_unfiltered_mean():
    """A filter that merely trades less is not a filter that trades better."""
    trades = _trades(edge_when_flag=False, seed=6)
    placebo = cx.random_veto_placebo(trades, retained_frac=0.5, n_draws=100)
    assert placebo["placebo_p05"] < trades["net_return"].mean() < placebo["placebo_p95"]


def test_placebo_interval_widens_as_selectivity_increases():
    trades = _trades(seed=7)
    loose = cx.random_veto_placebo(trades, retained_frac=0.8, n_draws=150)
    tight = cx.random_veto_placebo(trades, retained_frac=0.1, n_draws=150)
    assert tight["placebo_std"] > loose["placebo_std"], (
        "keeping fewer days must widen the placebo distribution"
    )


# ---- Baseline-relative comparison ----


def test_baseline_comparison_detects_a_helpful_veto():
    trades = _trades(edge_when_flag=True, seed=20)
    res = cx.compare_to_baseline(trades, ["veto_bad"], n_boot=800)
    assert res["delta"] > 0
    assert res["helps"] is True
    assert res["lo"] > 0


def test_baseline_comparison_rejects_a_useless_veto():
    trades = _trades(edge_when_flag=True, seed=21)
    res = cx.compare_to_baseline(trades, ["veto_noise"], n_boot=800)
    assert res["helps"] is False, "a noise veto must not read as helpful"


def test_baseline_comparison_of_no_veto_is_exactly_zero():
    trades = _trades(seed=22)
    res = cx.compare_to_baseline(trades, [], n_boot=200)
    assert res["delta"] == pytest.approx(0.0, abs=1e-12)


def test_profitable_and_helpful_are_distinct_questions():
    """A veto can help a lot and still leave a losing population.

    In a universe whose baseline is strongly negative, "beats no-veto" and
    "clears zero" come apart -- which is why both are reported.
    """
    rng = np.random.default_rng(31)
    dates = pd.bdate_range("2022-01-03", periods=200)
    rows = []
    for d in dates:
        for j in range(10):
            bad = rng.random() < 0.5
            # Even the good half loses money, just less.
            rows.append(
                {
                    "date": d, "symbol": f"S{j}",
                    "net_return": rng.normal(-0.20 if bad else -0.02, 0.03),
                    "outcome": "stop", "holding_minutes": 60,
                    "veto_bad": bad, "veto_noise": rng.random() < 0.3,
                }
            )
    trades = pd.DataFrame(rows)

    helps = cx.compare_to_baseline(trades, ["veto_bad"], n_boot=800)
    profitable = cx.evaluate_population(
        trades[cx.apply_vetoes(trades, ["veto_bad"])], n_boot=800
    )
    assert helps["helps"] is True, "the veto clearly helps"
    assert profitable.ci_hi < 0, "yet the surviving population still loses money"


# ---- Out-of-sample selection ----


def test_walk_forward_selects_on_train_and_scores_on_test():
    trades = _trades(n_days=400, edge_when_flag=True, seed=8)
    folds = walk_forward_folds(trades["date"].unique(), n_splits=3,
                               embargo_days=5, min_train_days=150)
    summary, oos = cx.walk_forward_veto_selection(
        trades, VETOES, folds, max_size=2, min_trades=50, min_retained=0.05
    )
    assert len(summary) == len(folds)
    assert not oos.empty
    # The real veto should be chosen in every fold.
    assert summary["chosen_vetoes"].str.contains("veto_bad").all()


def test_out_of_sample_trades_come_only_from_test_windows():
    trades = _trades(n_days=400, edge_when_flag=True, seed=9)
    folds = walk_forward_folds(trades["date"].unique(), n_splits=3,
                               embargo_days=5, min_train_days=150)
    _, oos = cx.walk_forward_veto_selection(
        trades, VETOES, folds, max_size=2, min_trades=50, min_retained=0.05
    )
    test_days = set()
    for f in folds:
        test_days |= set(f.test_dates)
    assert set(oos["date"]).issubset(test_days), "OOS contains non-test dates"


def test_walk_forward_on_noise_degrades_from_train_to_test():
    """In-sample selection on noise must not survive out of sample."""
    trades = _trades(n_days=400, edge_when_flag=False, seed=10)
    folds = walk_forward_folds(trades["date"].unique(), n_splits=3,
                               embargo_days=5, min_train_days=150)
    summary, _ = cx.walk_forward_veto_selection(
        trades, VETOES, folds, max_size=2, min_trades=50, min_retained=0.05
    )
    assert (summary["train_mean"] > summary["test_mean"]).mean() >= 0.5, (
        "selected-on-noise vetoes should degrade out of sample"
    )


# ---- Null tests ----


def test_constant_return_null_passes_on_sane_plumbing():
    result = nulls.constant_return_test(_trades(seed=12), VETOES)
    assert result.passed, result.detail


def test_shuffled_labels_null_passes_on_a_real_edge_panel():
    """Shuffling must destroy even a genuine edge."""
    trades = _trades(n_days=200, edge_when_flag=True, seed=13)
    result = nulls.shuffled_label_test(trades, VETOES, n_repeats=5, max_size=1)
    assert result.passed, result.detail


def test_feature_timing_tripwire_catches_a_post_decision_bar():
    bad = pd.DataFrame({"minute_et": ["09:30", "09:36"]})
    result = nulls.feature_timing_tripwire(bad)
    assert not result.passed
    assert "1 feature rows" in result.detail


def test_feature_timing_tripwire_passes_on_clean_observation_window():
    good = pd.DataFrame({"minute_et": ["09:30", "09:31", "09:34"]})
    assert nulls.feature_timing_tripwire(good).passed


def test_minute_string_comparison_is_safe_across_the_hour_boundary():
    """Zero-padded HH:MM compares correctly lexicographically.

    "10:00" > "09:35" must hold as strings; an unpadded format would break the
    barrier scan silently and only for part of the session.
    """
    assert "10:00" > "09:35"
    assert "09:34" < "09:35"
    assert "15:55" > "09:35"
    result = nulls.feature_timing_tripwire(pd.DataFrame({"minute_et": ["10:00"]}))
    assert not result.passed, "a 10:00 bar must be flagged as post-decision"
