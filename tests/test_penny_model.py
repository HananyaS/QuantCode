"""Tests for the meta-labelling model layer.

The highest-value tests here are the ones that stop an outcome-derived column
from reaching the feature list. That mistake does not produce an error; it
produces an excellent result.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.eval.folds import walk_forward_folds
from research.penny.model.features import (
    add_cross_sectional_ranks,
    add_regime_features,
    build_matrix,
    feature_columns,
)
from research.penny.model.train import (
    ModelConfig,
    score_decile_table,
    select_top_fraction,
    spearman_monotonicity,
    walk_forward_scores,
)


def _trades(n_days=300, per_day=8, seed=0, signal=False):
    """Synthetic trade table shaped like the real one."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for d in dates:
        for j in range(per_day):
            obs_range = rng.uniform(0.01, 0.40)
            gap = rng.uniform(0.10, 1.20)
            # When `signal`, wide opening ranges genuinely predict losses.
            base = -0.30 * obs_range if signal else 0.0
            rows.append(
                {
                    "date": d, "symbol": f"S{j}",
                    "gap": gap, "rvol_prev": rng.uniform(0.5, 8),
                    "adv_5": rng.uniform(1e5, 1e7), "adv_20": rng.uniform(1e5, 1e7),
                    "adv_60": rng.uniform(1e5, 1e7),
                    "max_ret_20d": rng.uniform(0, 1.5), "vol_20d": rng.uniform(0.02, 0.3),
                    "hl_range_20d": rng.uniform(0.02, 0.4), "consec_up": rng.integers(0, 5),
                    "ret_1d_prev": rng.normal(0, 0.1), "ret_5d_prev": rng.normal(0, 0.2),
                    "close_unadj_prev": rng.uniform(0.5, 10),
                    "split_factor_180d": 1.0,
                    "obs_return": rng.normal(0, 0.05), "obs_range": obs_range,
                    "obs_close_position": rng.uniform(0, 1),
                    "obs_dollar_volume": rng.uniform(1e4, 5e6),
                    "minutes_obs": rng.integers(1, 6), "trades_obs": rng.integers(5, 500),
                    "ret_c2c": rng.normal(base, 0.15),
                    # outcome-derived, must never become a feature
                    "net_return": rng.normal(-0.04, 0.1), "outcome": "stop",
                    "mfe_full": rng.uniform(0, 1), "coverage": rng.uniform(0, 1),
                    "fut_close": rng.uniform(1, 10),
                    "diag_days_to_last_bar": rng.integers(1, 900),
                }
            )
    df = pd.DataFrame(rows)
    df["meta_label"] = (df["ret_c2c"] > 0.0437).astype(int)
    return df


# ---- The leakage boundary ----


def test_outcome_columns_never_enter_the_feature_list():
    """An allowlist, not a dtype sweep. This is the whole guard."""
    cols = feature_columns(build_matrix(_trades(n_days=60)), include_edgar=False)
    for banned in ["net_return", "ret_c2c", "mfe_full", "outcome", "coverage",
                   "fut_close", "diag_days_to_last_bar", "meta_label"]:
        assert banned not in cols, f"{banned} reached the feature list"


def test_guard_fires_if_an_outcome_column_is_smuggled_in():
    df = build_matrix(_trades(n_days=60))
    df["xs_rank_net_return"] = 0.5   # looks like a legitimate rank feature
    from research.penny.model import features as F
    original = F._forbidden_columns
    F._forbidden_columns = lambda d: original(d) | {"xs_rank_net_return"}
    try:
        with pytest.raises(AssertionError, match="outcome-derived columns"):
            feature_columns(df, include_edgar=False)
    finally:
        F._forbidden_columns = original


def test_feature_list_contains_the_real_signals():
    cols = feature_columns(build_matrix(_trades(n_days=60)), include_edgar=False)
    for expected in ["obs_range", "gap", "hl_range_20d", "xs_rank_gap", "xs_n_candidates"]:
        assert expected in cols


# ---- Cross-sectional ranks ----


def test_ranks_are_computed_within_a_session():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 3 + [pd.Timestamp("2024-01-03")] * 3,
            "symbol": list("ABCDEF"),
            "gap": [0.1, 0.2, 0.3, 0.9, 0.5, 0.1],
        }
    )
    out = add_cross_sectional_ranks(df, ["gap"])
    day1 = out[out["date"] == pd.Timestamp("2024-01-02")]
    day2 = out[out["date"] == pd.Timestamp("2024-01-03")]
    # 0.3 is top of day 1; 0.9 is top of day 2 -- both rank 1.0
    assert day1.loc[day1["gap"] == 0.3, "xs_rank_gap"].iloc[0] == pytest.approx(1.0)
    assert day2.loc[day2["gap"] == 0.9, "xs_rank_gap"].iloc[0] == pytest.approx(1.0)
    # A 0.1 gap ranks bottom on both days despite identical absolute value
    assert day1.loc[day1["gap"] == 0.1, "xs_rank_gap"].iloc[0] < 0.5


def test_candidate_count_is_per_session():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 3 + [pd.Timestamp("2024-01-03")] * 5,
            "symbol": list("ABCDEFGH"), "gap": np.arange(8) / 10,
        }
    )
    out = add_cross_sectional_ranks(df, ["gap"])
    assert out[out["date"] == pd.Timestamp("2024-01-02")]["xs_n_candidates"].iloc[0] == 3
    assert out[out["date"] == pd.Timestamp("2024-01-03")]["xs_n_candidates"].iloc[0] == 5


# ---- Regime features must be trailing ----


def test_regime_features_exclude_the_current_session():
    """Otherwise the model knows today's activity before deciding on today."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    rows = []
    for i, d in enumerate(dates):
        # A sharp jump in activity on the final day.
        n = 2 if i < 59 else 50
        for j in range(n):
            rows.append({"date": d, "symbol": f"S{j}", "gap": 0.2})
    df = add_regime_features(pd.DataFrame(rows), windows=(20,))
    last = df[df["date"] == dates[-1]]
    # The trailing mean must reflect the quiet history, not the busy day itself.
    assert last["regime_n_cand_20d"].iloc[0] == pytest.approx(2.0, abs=0.01)


def test_first_sessions_have_no_regime_history():
    dates = pd.bdate_range("2024-01-02", periods=10)
    rows = [{"date": d, "symbol": "A", "gap": 0.2} for d in dates]
    df = add_regime_features(pd.DataFrame(rows), windows=(20,))
    assert df["regime_n_cand_20d"].isna().iloc[0]


# ---- Walk-forward scoring ----


def test_scores_come_only_from_test_folds():
    df = build_matrix(_trades(n_days=320, signal=True))
    folds = walk_forward_folds(df["date"].unique(), n_splits=3, embargo_days=5,
                               min_train_days=150)
    scored, _ = walk_forward_scores(df, folds, feature_columns(df, False), "meta_label")
    test_days = set()
    for f in folds:
        test_days |= set(f.test_dates)
    assert not scored.empty
    assert set(scored["date"]).issubset(test_days)


def test_every_scored_row_gets_a_probability():
    df = build_matrix(_trades(n_days=320))
    folds = walk_forward_folds(df["date"].unique(), n_splits=3, embargo_days=5,
                               min_train_days=150)
    scored, _ = walk_forward_scores(df, folds, feature_columns(df, False), "meta_label")
    assert scored["model_score"].between(0, 1).all()
    assert scored["model_score"].notna().all()


def test_model_recovers_a_planted_signal():
    """Sanity: with obs_range genuinely predictive, the score must order outcomes."""
    df = build_matrix(_trades(n_days=400, per_day=10, signal=True, seed=3))
    folds = walk_forward_folds(df["date"].unique(), n_splits=3, embargo_days=5,
                               min_train_days=180)
    scored, _ = walk_forward_scores(df, folds, feature_columns(df, False), "meta_label")
    table = score_decile_table(scored)
    assert spearman_monotonicity(table) > 0.5, "planted signal was not recovered"


def test_model_finds_nothing_in_pure_noise():
    df = build_matrix(_trades(n_days=400, per_day=10, signal=False, seed=4))
    folds = walk_forward_folds(df["date"].unique(), n_splits=3, embargo_days=5,
                               min_train_days=180)
    scored, _ = walk_forward_scores(df, folds, feature_columns(df, False), "meta_label")
    table = score_decile_table(scored)
    assert abs(spearman_monotonicity(table)) < 0.75, "found structure in noise"


def test_degenerate_fold_returns_base_rate_rather_than_crashing():
    df = build_matrix(_trades(n_days=320))
    df["meta_label"] = 0                      # single class
    folds = walk_forward_folds(df["date"].unique(), n_splits=3, embargo_days=5,
                               min_train_days=150)
    scored, _ = walk_forward_scores(df, folds, feature_columns(df, False), "meta_label")
    assert (scored["model_score"] == 0.0).all()


# ---- Selection ----


def test_top_fraction_is_taken_within_each_fold():
    """Score distributions drift between folds; a global cut would skew selection."""
    scored = pd.DataFrame(
        {
            "fold": [0] * 100 + [1] * 100,
            "model_score": np.concatenate([np.linspace(0, 0.2, 100),
                                           np.linspace(0.8, 1.0, 100)]),
            "ret_c2c": 0.0,
        }
    )
    top = select_top_fraction(scored, 0.10)
    assert len(top) == 20
    assert top["fold"].value_counts().tolist() == [10, 10]


def test_top_fraction_edge_cases():
    scored = pd.DataFrame({"fold": [0] * 10, "model_score": np.arange(10), "ret_c2c": 0.0})
    assert len(select_top_fraction(scored, 1.0)) == 10
    assert len(select_top_fraction(scored, 0.01)) == 1
    assert select_top_fraction(pd.DataFrame(), 0.1).empty


# ---- Diagnostics ----


def test_decile_table_is_ordered_and_complete():
    rng = np.random.default_rng(1)
    scored = pd.DataFrame(
        {"model_score": rng.uniform(size=1000), "ret_c2c": rng.normal(size=1000)}
    )
    table = score_decile_table(scored)
    assert len(table) == 10
    assert table["n"].sum() == 1000


def test_monotonicity_is_high_for_a_perfect_score():
    table = pd.DataFrame({"score_bin": range(10), "mean_gross": np.linspace(-0.1, 0.1, 10)})
    assert spearman_monotonicity(table) == pytest.approx(1.0)


# ---- The selection-lift test ----


def test_selection_lift_detects_a_genuinely_better_subset():
    """The test that replaces decile monotonicity as evidence."""
    from research.penny.eval.conditional import compare_selection_to_baseline

    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2022-01-03", periods=250)
    rows = []
    for d in dates:
        for j in range(12):
            good = j < 3
            rows.append({"date": d, "symbol": f"S{j}",
                         "ret_c2c": rng.normal(0.04 if good else -0.02, 0.05),
                         "good": good})
    df = pd.DataFrame(rows)
    res = compare_selection_to_baseline(df, df["good"].to_numpy(), return_col="ret_c2c")
    assert res["delta"] > 0
    assert res["helps"] is True
    assert res["p_value"] < 0.05


def test_selection_lift_finds_nothing_for_a_random_subset():
    from research.penny.eval.conditional import compare_selection_to_baseline

    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2022-01-03", periods=250)
    rows = []
    for d in dates:
        for j in range(12):
            rows.append({"date": d, "symbol": f"S{j}", "ret_c2c": rng.normal(-0.01, 0.05)})
    df = pd.DataFrame(rows)
    mask = rng.random(len(df)) < 0.25
    res = compare_selection_to_baseline(df, mask, return_col="ret_c2c")
    assert res["helps"] is False


def test_selecting_everything_gives_exactly_zero_lift():
    from research.penny.eval.conditional import compare_selection_to_baseline

    df = pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=50).repeat(4),
        "symbol": list("ABCD") * 50,
        "ret_c2c": np.random.default_rng(1).normal(0, 0.05, 200),
    })
    res = compare_selection_to_baseline(df, np.ones(len(df), dtype=bool),
                                        return_col="ret_c2c", n_boot=500)
    assert res["delta"] == pytest.approx(0.0, abs=1e-12)


def test_decile_monotonicity_is_documented_as_unreliable():
    """Ten bins give Spearman rho a standard deviation near 0.33.

    Pins the reason the study does not treat rho as evidence: pure noise
    produces large |rho| routinely.
    """
    rng = np.random.default_rng(3)
    rhos = []
    for _ in range(40):
        table = pd.DataFrame({"score_bin": range(10), "mean_gross": rng.normal(size=10)})
        rhos.append(abs(spearman_monotonicity(table)))
    assert np.mean(rhos) > 0.2, "noise should produce substantial |rho| by chance"
    assert max(rhos) > 0.6, "noise should sometimes produce |rho| above 0.6"
