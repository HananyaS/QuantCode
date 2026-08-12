"""Which cost columns a model may see, and why.

Rounds 1-3 banned every quote-derived column. That was hygiene, not a leakage
argument -- ``etl/quotes.py`` says plainly that the NBBO is "directly observable
from the decision instant" -- and it had a real cost: the model could not learn
to avoid expensive names, while measured cost varies 6.3x across the candidate
universe.

Round 4 draws the line where it actually falls, on the **window** rather than
the quantity:

* the 09:34 book is knowable before entering -> ``cost_hat``, admissible;
* the 09:35 entry book and 15:55 exit book are not -> ``cost_realized``, charged
  in evaluation and never an input.

These tests pin that distinction so it cannot drift back into ambiguity.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.etl.quotes import attach_costs
from research.penny.model.features import (
    COST_FEATURES,
    _forbidden_columns,
    feature_columns,
)


def _trades(n=4):
    return pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)],
        "date": pd.to_datetime(["2024-03-05"] * n),
        "adv_20": [1e7] * n,
        "vol_20d": [0.20] * n,
        "gap": [0.10] * n,
    })


def _windows(n=4, decision_spread=40.0, entry_spread=45.0, exit_spread=30.0,
             decision_ask=50_000.0, entry_ask=50_000.0, exit_bid=50_000.0):
    return pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)],
        "date": pd.to_datetime(["2024-03-05"] * n),
        "half_spread_bps_decision": [decision_spread] * n,
        "ask_notional_decision": [decision_ask] * n,
        "bid_notional_decision": [decision_ask] * n,
        "half_spread_bps_entry": [entry_spread] * n,
        "ask_notional_entry": [entry_ask] * n,
        "half_spread_bps_exit": [exit_spread] * n,
        "bid_notional_exit": [exit_bid] * n,
    })


# ---- The allowlist distinction ----


def test_decision_window_cost_features_are_allowed():
    df = _trades().assign(**{c: 1.0 for c in COST_FEATURES})
    cols = feature_columns(df, include_cost=True)
    assert set(COST_FEATURES) <= set(cols)


def test_cost_features_are_off_by_default():
    """Opt-in, so the round-3 ablation ladder keeps its exact meaning."""
    df = _trades().assign(**{c: 1.0 for c in COST_FEATURES})
    assert not (set(COST_FEATURES) & set(feature_columns(df)))


@pytest.mark.parametrize("col", [
    "half_spread_bps_entry", "ask_notional_entry",
    "half_spread_bps_exit", "bid_notional_exit",
    "entry_leg_bps", "exit_leg_bps", "round_trip_bps",
    "cost_realized", "cost_realized_bps",
])
def test_realised_cost_columns_stay_forbidden(col):
    """These describe how the day's liquidity actually turned out."""
    df = _trades().assign(**{col: 1.0})
    assert col in _forbidden_columns(df)


@pytest.mark.parametrize("col", ["half_spread_bps", "ask_notional", "bid_notional"])
def test_round_three_unwindowed_quote_columns_stay_forbidden(col):
    """Round 3 measured these over [09:35:00, 09:36:00) -- the entry minute."""
    df = _trades().assign(**{col: 1.0})
    assert col in _forbidden_columns(df)


@pytest.mark.parametrize("col", ["p_calib", "ev", "payoff_win_mean", "payoff_loss_mean"])
def test_decision_rule_outputs_cannot_feed_back_as_features(col):
    df = _trades().assign(**{col: 1.0})
    assert col in _forbidden_columns(df)


def test_enabling_cost_features_never_admits_a_realised_column():
    """The assertion inside feature_columns is the real guard; prove it fires."""
    df = _trades()
    for c in COST_FEATURES:
        df[c] = 1.0
    df["cost_realized_bps"] = 500.0
    df["half_spread_bps_entry"] = 45.0
    cols = feature_columns(df, include_cost=True)
    assert "cost_realized_bps" not in cols
    assert "half_spread_bps_entry" not in cols


# ---- attach_costs ----


def test_hat_and_realised_are_both_produced_and_differ():
    out = attach_costs(_trades(), _windows(), order_notional=2_000.0)
    assert {"cost_hat", "cost_hat_bps", "cost_realized", "cost_realized_bps"} <= set(out.columns)
    # decision spread 40 on both legs vs entry 45 + exit 30 -> different totals
    assert out["cost_hat_bps"].iloc[0] != out["cost_realized_bps"].iloc[0]


def test_hat_uses_only_the_decision_book():
    """Changing the entry/exit books must not move the estimate."""
    base = attach_costs(_trades(), _windows(), 2_000.0)["cost_hat_bps"].iloc[0]
    moved = attach_costs(
        _trades(), _windows(entry_spread=900.0, exit_spread=900.0, exit_bid=10.0), 2_000.0
    )["cost_hat_bps"].iloc[0]
    assert base == pytest.approx(moved), "cost_hat leaked post-decision information"


def test_realised_responds_to_the_exit_book():
    """The round-3 defect, as a positive assertion."""
    deep = attach_costs(_trades(), _windows(exit_bid=1e9), 10_000.0)
    thin = attach_costs(_trades(), _windows(exit_bid=200.0), 10_000.0)
    assert thin["cost_realized_bps"].iloc[0] > deep["cost_realized_bps"].iloc[0] * 5


def test_realised_splits_into_two_auditable_legs():
    out = attach_costs(_trades(), _windows(), 2_000.0)
    assert out["cost_realized_bps"].iloc[0] == pytest.approx(
        out["entry_leg_bps"].iloc[0] + out["exit_leg_bps"].iloc[0]
    )


def test_depth_ratio_is_inside_book_over_order_size():
    out = attach_costs(_trades(), _windows(decision_ask=20_000.0), order_notional=2_000.0)
    assert out["depth_ratio_decision"].iloc[0] == pytest.approx(10.0)


def test_depth_ratio_is_capped_so_one_huge_book_cannot_dominate():
    out = attach_costs(_trades(), _windows(decision_ask=1e12), order_notional=2_000.0)
    assert out["depth_ratio_decision"].iloc[0] == pytest.approx(100.0)


def test_log_inside_notional_is_finite_on_a_zero_book():
    out = attach_costs(_trades(), _windows(decision_ask=0.0), 2_000.0)
    assert np.isfinite(out["log_inside_notional_decision"].iloc[0])


def test_row_count_is_preserved_and_unmatched_days_stay_nan():
    trades = _trades(4)
    win = _windows(2)  # only S0, S1 have quotes
    out = attach_costs(trades, win, 2_000.0)
    assert len(out) == 4
    assert out["cost_hat_bps"].isna().sum() == 2, "unmatched trades must not inherit a cost"


def test_a_bigger_order_costs_more():
    small = attach_costs(_trades(), _windows(decision_ask=5_000.0), 2_000.0)
    large = attach_costs(_trades(), _windows(decision_ask=5_000.0), 20_000.0)
    assert large["cost_hat_bps"].iloc[0] > small["cost_hat_bps"].iloc[0]
