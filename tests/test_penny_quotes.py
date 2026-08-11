"""Tests for measured execution costs.

The cost bar is the single most load-bearing number in the project: every
verdict across three rounds has been "gross beats/loses to costs". Rounds 1-2
used an assumed 437 bps. These tests guard the measurement that replaces it.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.etl.quotes import (
    CostModel,
    bucket_summary,
    build_cost_table,
    depth_shortfall_bps,
    impact_bps,
    summarise_quotes,
)


def _quotes(rows):
    """(symbol, bid, ask, bid_size, ask_size) -> a quote frame."""
    df = pd.DataFrame(rows, columns=["symbol", "bid_price", "ask_price", "bid_size", "ask_size"])
    df["date"] = pd.Timestamp("2026-03-05")
    return df


# ---- Half-spread measurement ----


def test_half_spread_is_half_the_relative_quoted_spread():
    q = _quotes([("A", 1.00, 1.02, 10, 10)])
    out = summarise_quotes(q)
    # spread 0.02 on a 1.01 mid = 198 bps; half is ~99
    assert out.iloc[0]["half_spread_bps"] == pytest.approx(99.0, abs=1.0)


def test_median_not_first_quote_is_used():
    """Opening quotes are noisy and the first is usually the widest."""
    q = _quotes([
        ("A", 1.00, 1.20, 10, 10),   # wide outlier first
        ("A", 1.00, 1.02, 10, 10),
        ("A", 1.00, 1.02, 10, 10),
        ("A", 1.00, 1.02, 10, 10),
    ])
    out = summarise_quotes(q)
    assert out.iloc[0]["half_spread_bps"] < 110, "the wide first quote dominated the estimate"


def test_crossed_and_locked_books_are_discarded():
    """A crossed market is an artefact, not a tradable price."""
    q = _quotes([
        ("A", 1.05, 1.00, 10, 10),   # crossed
        ("A", 1.02, 1.02, 10, 10),   # locked
        ("A", 1.00, 1.02, 10, 10),   # the only real market
    ])
    out = summarise_quotes(q)
    assert len(out) == 1
    assert out.iloc[0]["n_quotes"] == 1


def test_zero_size_quotes_are_discarded():
    q = _quotes([("A", 1.00, 1.02, 0, 10), ("A", 1.00, 1.02, 10, 10)])
    assert summarise_quotes(q).iloc[0]["n_quotes"] == 1


def test_all_invalid_quotes_yield_empty_rather_than_a_fake_number():
    q = _quotes([("A", 1.05, 1.00, 10, 10)])
    assert summarise_quotes(q).empty


def test_symbols_are_summarised_independently():
    q = _quotes([("A", 1.00, 1.02, 10, 10), ("B", 1.00, 1.50, 10, 10)])
    out = summarise_quotes(q).set_index("symbol")
    assert out.loc["A", "half_spread_bps"] < out.loc["B", "half_spread_bps"]


# ---- Impact ----


def test_impact_scales_with_volatility():
    """The property the old bare constant lacked entirely."""
    quiet = impact_bps(pd.Series([0.02]), pd.Series([0.01]))
    wild = impact_bps(pd.Series([0.20]), pd.Series([0.01]))
    assert wild.iloc[0] > 5 * quiet.iloc[0]


def test_impact_is_concave_in_participation():
    """Square-root, not linear: doubling size less than doubles the cost."""
    small = impact_bps(pd.Series([0.10]), pd.Series([0.01])).iloc[0]
    big = impact_bps(pd.Series([0.10]), pd.Series([0.04])).iloc[0]
    assert big == pytest.approx(2 * small, rel=1e-6)


def test_zero_participation_costs_nothing():
    assert impact_bps(pd.Series([0.10]), pd.Series([0.0])).iloc[0] == pytest.approx(0.0)


# ---- Depth ----


def test_no_shortfall_when_the_inside_can_absorb_the_order():
    out = depth_shortfall_bps(pd.Series([5_000]), pd.Series([20_000]), pd.Series([25.0]))
    assert out.iloc[0] == pytest.approx(0.0)


def test_shortfall_grows_when_the_order_exceeds_the_inside():
    """A 400-share inside quote is why this term exists at all."""
    out = depth_shortfall_bps(pd.Series([40_000]), pd.Series([10_000]), pd.Series([25.0]))
    assert out.iloc[0] == pytest.approx(3 * 25.0)


def test_missing_depth_is_treated_as_no_shortfall_not_infinite():
    out = depth_shortfall_bps(pd.Series([10_000]), pd.Series([np.nan]), pd.Series([25.0]))
    assert out.iloc[0] == pytest.approx(0.0)


# ---- Assembly ----


def _trades():
    return pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": [pd.Timestamp("2026-03-05")] * 2,
            "adv_20": [1e6, 1e6],
            "vol_20d": [0.10, 0.10],
            "close_unadj_prev": [2.0, 8.0],
        }
    )


def test_cost_table_combines_all_three_components():
    summary = pd.DataFrame(
        {
            "symbol": ["A", "B"], "date": [pd.Timestamp("2026-03-05")] * 2,
            "half_spread_bps": [25.0, 25.0], "ask_notional": [1e6, 1_000.0],
            "mid": [2.0, 8.0], "n_quotes": [10, 10],
        }
    )
    out = build_cost_table(summary, _trades())
    # A has deep inside quotes; B is thin and must pay a depth shortfall.
    a, b = out.set_index("symbol").loc["A"], out.set_index("symbol").loc["B"]
    assert a["depth_shortfall_bps"] == pytest.approx(0.0)
    assert b["depth_shortfall_bps"] > 0
    assert b["measured_round_trip_bps"] > a["measured_round_trip_bps"]


def test_round_trip_charges_the_spread_twice():
    summary = pd.DataFrame(
        {
            "symbol": ["A"], "date": [pd.Timestamp("2026-03-05")],
            "half_spread_bps": [30.0], "ask_notional": [1e9], "mid": [2.0], "n_quotes": [5],
        }
    )
    out = build_cost_table(summary, _trades().head(1))
    expected = 2 * (30.0 + out.iloc[0]["impact_bps"])
    assert out.iloc[0]["measured_round_trip_bps"] == pytest.approx(expected)


def test_unmatched_trades_keep_nan_rather_than_a_default_cost():
    """A trade with no quote data must not silently inherit a made-up cost."""
    empty = pd.DataFrame(columns=["symbol", "date", "half_spread_bps", "ask_notional",
                                  "mid", "n_quotes"])
    out = build_cost_table(empty, _trades())
    assert out["half_spread_bps"].isna().all()
    assert out["measured_round_trip_bps"].isna().all()


def test_bucket_summary_splits_by_the_named_column():
    summary = pd.DataFrame(
        {
            "symbol": ["A", "B"], "date": [pd.Timestamp("2026-03-05")] * 2,
            "half_spread_bps": [100.0, 20.0], "ask_notional": [1e9, 1e9],
            "mid": [1.0, 9.0], "n_quotes": [5, 5],
        }
    )
    table = build_cost_table(summary, _trades())
    out = bucket_summary(table, "close_unadj_prev", [0, 3, 10])
    assert len(out) == 2
    # Cheaper stocks quote wider; the buckets must show it.
    assert out.iloc[0]["half_spread_bps"] > out.iloc[1]["half_spread_bps"]


# ---- The model object ----


def test_round_trip_is_two_crossings_plus_depth():
    m = CostModel(half_spread_bps=25.0, impact_bps=10.0, depth_shortfall_bps=5.0)
    assert m.round_trip_bps == pytest.approx(2 * 35.0 + 5.0)
    assert m.round_trip == pytest.approx(0.0075)


def test_assumed_bar_is_reproducible_from_the_old_constants():
    """Sanity-check the number rounds 1-2 actually charged."""
    old = CostModel(half_spread_bps=118.5, impact_bps=100.0)
    assert old.round_trip_bps == pytest.approx(437.0, abs=1.0)
