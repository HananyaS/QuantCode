"""Round-4 cost accounting: two measured legs, and a decision/charge split.

Round 3's cost model had three defects, each of which flattered the result:

1. ``depth_shortfall`` was added once for a round trip that crosses the book
   twice -- and depth was 398 of the measured 728 bps, so the largest term was
   half-counted.
2. Only the ask book was ever measured, though a long exit sells into the bid.
3. The half-spread was measured over ``[09:35:00, 09:36:00)`` while entry is the
   09:35 bar *open*. Harmless while cost was evaluation-only; look-ahead the
   moment cost informs the decision, which is the whole of round 4.

These tests pin the fixes. See PREREGISTRATION_R4.md section 4.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.etl.quotes import (
    DECISION_MINUTE,
    ENTRY_MINUTE,
    EXIT_MINUTE,
    assert_decision_quotes_are_pre_entry,
    bucket_summary,
    leg_cost_bps,
    round_trip_cost_bps,
    summarise_quotes,
    tradeable_mask,
    trim_to_before,
)


def _quotes(rows, day="2026-03-05"):
    df = pd.DataFrame(rows, columns=["symbol", "bid_price", "ask_price", "bid_size", "ask_size"])
    df["date"] = pd.Timestamp(day)
    return df


def _series(v, n=1):
    return pd.Series([v] * n, dtype=float)


# ---- The bid book is now measured ----


def test_bid_notional_is_reported_alongside_ask():
    """A long round trip lifts the ask and then hits the bid."""
    q = _quotes([("A", 1.00, 1.02, 5, 20)])
    out = summarise_quotes(q).iloc[0]
    assert out["ask_notional"] == pytest.approx(1.02 * 20 * 100)
    assert out["bid_notional"] == pytest.approx(1.00 * 5 * 100)


def test_the_two_sides_can_differ_by_an_order_of_magnitude():
    """The reason a single depth number cannot describe a round trip."""
    q = _quotes([("A", 1.00, 1.02, 2, 200)])
    out = summarise_quotes(q).iloc[0]
    assert out["ask_notional"] / out["bid_notional"] > 50


# ---- Depth is charged on both legs ----


def test_round_trip_charges_each_leg_against_its_own_book():
    """The round-3 defect: exit depth was never charged at all."""
    order = _series(10_000.0)
    entry = pd.DataFrame({"half_spread_bps": [50.0], "ask_notional": [10_000.0]})
    # Exit book is a tenth of the order -> the exit must be the expensive leg.
    exit_ = pd.DataFrame({"half_spread_bps": [50.0], "bid_notional": [1_000.0]})

    out = round_trip_cost_bps(order, entry, exit_, _series(0.0), _series(0.0)).iloc[0]

    assert out["entry_leg_bps"] == pytest.approx(50.0), "entry book covered the order"
    # 10k into a 1k book = 9 extra multiples * 50 bps, plus the spread itself.
    assert out["exit_leg_bps"] == pytest.approx(50.0 + 9 * 50.0)
    assert out["round_trip_bps"] == pytest.approx(out["entry_leg_bps"] + out["exit_leg_bps"])


def test_a_thin_exit_book_is_not_free():
    """Regression guard for the exact round-3 behaviour, stated as a failure."""
    order = _series(10_000.0)
    entry = pd.DataFrame({"half_spread_bps": [20.0], "ask_notional": [1e9]})
    exit_ = pd.DataFrame({"half_spread_bps": [20.0], "bid_notional": [500.0]})

    rt = round_trip_cost_bps(order, entry, exit_, _series(0.0), _series(0.0)).iloc[0]
    round3_equivalent = 2 * 20.0  # spread twice, depth once, exit book ignored
    assert rt["round_trip_bps"] > round3_equivalent * 5, (
        "exit-side depth is being ignored, which is the round-3 bug"
    )


def test_deep_books_on_both_sides_cost_only_the_spread_twice():
    order = _series(1_000.0)
    entry = pd.DataFrame({"half_spread_bps": [30.0], "ask_notional": [1e6]})
    exit_ = pd.DataFrame({"half_spread_bps": [30.0], "bid_notional": [1e6]})
    rt = round_trip_cost_bps(order, entry, exit_, _series(0.0), _series(0.0)).iloc[0]
    assert rt["round_trip_bps"] == pytest.approx(60.0)


def test_legs_are_reported_separately_so_the_split_stays_auditable():
    order = _series(5_000.0)
    entry = pd.DataFrame({"half_spread_bps": [10.0], "ask_notional": [1e6]})
    exit_ = pd.DataFrame({"half_spread_bps": [90.0], "bid_notional": [1e6]})
    rt = round_trip_cost_bps(order, entry, exit_, _series(0.0), _series(0.0)).iloc[0]
    assert rt["entry_leg_bps"] < rt["exit_leg_bps"], "leg asymmetry was collapsed"


def test_impact_scales_with_volatility_on_each_leg():
    order = _series(1_000.0)
    book = pd.DataFrame({"half_spread_bps": [10.0], "ask_notional": [1e6]})
    exit_ = pd.DataFrame({"half_spread_bps": [10.0], "bid_notional": [1e6]})
    quiet = round_trip_cost_bps(order, book, exit_, _series(0.01), _series(0.01))
    wild = round_trip_cost_bps(order, book, exit_, _series(0.50), _series(0.01))
    assert wild.iloc[0]["round_trip_bps"] > quiet.iloc[0]["round_trip_bps"]


def test_levels_penalty_is_honoured_so_sensitivity_can_be_reported():
    """The depth model is crude; a conclusion must survive a harsher setting."""
    order = _series(10_000.0)
    entry = pd.DataFrame({"half_spread_bps": [40.0], "ask_notional": [1_000.0]})
    exit_ = pd.DataFrame({"half_spread_bps": [40.0], "bid_notional": [1_000.0]})
    gentle = round_trip_cost_bps(order, entry, exit_, _series(0.0), _series(0.0), levels_penalty=1.0)
    harsh = round_trip_cost_bps(order, entry, exit_, _series(0.0), _series(0.0), levels_penalty=2.0)
    assert harsh.iloc[0]["round_trip_bps"] > gentle.iloc[0]["round_trip_bps"] * 1.5


def test_leg_cost_is_never_negative():
    order = _series(100.0)
    out = leg_cost_bps(order, _series(1e9), _series(5.0), _series(0.0), _series(0.0))
    assert float(out.iloc[0]) >= 0.0


# ---- cost_hat may not see the entry minute ----


def _ts_quotes(times, day="2026-03-05"):
    ts = pd.to_datetime([f"{day} {t}" for t in times]).tz_localize("America/New_York")
    return pd.DataFrame({"symbol": "A", "timestamp": ts.tz_convert("UTC")})


def test_decision_window_constants_are_ordered():
    assert DECISION_MINUTE < ENTRY_MINUTE < EXIT_MINUTE


def test_tripwire_passes_on_strictly_pre_entry_quotes():
    assert_decision_quotes_are_pre_entry(_ts_quotes(["09:34:00", "09:34:59"]))


def test_tripwire_catches_a_quote_at_the_entry_instant():
    """09:35:00 is the entry, not before it. The boundary is the whole point."""
    with pytest.raises(AssertionError, match="strictly before"):
        assert_decision_quotes_are_pre_entry(_ts_quotes(["09:34:30", "09:35:00"]))


def test_tripwire_catches_the_round_three_window():
    """Round 3 measured the median over [09:35:00, 09:36:00)."""
    with pytest.raises(AssertionError, match="strictly before"):
        assert_decision_quotes_are_pre_entry(_ts_quotes(["09:35:30"]))


def test_tripwire_reports_how_many_and_how_late():
    with pytest.raises(AssertionError, match=r"2 quote\(s\) at or after entry"):
        assert_decision_quotes_are_pre_entry(_ts_quotes(["09:34:00", "09:35:01", "09:40:00"]))


def test_tripwire_is_per_day_not_per_panel():
    """A late quote on one day must not be excused by an early one on another."""
    early = _ts_quotes(["09:34:00"], day="2026-03-05")
    late = _ts_quotes(["09:36:00"], day="2026-03-06")
    with pytest.raises(AssertionError):
        assert_decision_quotes_are_pre_entry(pd.concat([early, late], ignore_index=True))


def test_tripwire_is_a_noop_on_empty_input():
    assert_decision_quotes_are_pre_entry(pd.DataFrame())


# ---- Trimming the inclusive endpoint ----


def test_trim_drops_the_quote_stamped_exactly_at_entry():
    """Observed live on ~3% of sessions: Alpaca's `end` is inclusive.

    The same trap that made rounds 1-2 drop each target day's own bar. Asking
    for [09:34:00, 09:35:00) returns the 09:35:00.000 quote when one exists.
    """
    q = _ts_quotes(["09:34:00", "09:34:59", "09:35:00"])
    out = trim_to_before(q, ENTRY_MINUTE)
    assert len(out) == 2
    assert_decision_quotes_are_pre_entry(out)


def test_trim_keeps_everything_strictly_before():
    q = _ts_quotes(["09:34:00", "09:34:30", "09:34:59"])
    assert len(trim_to_before(q, ENTRY_MINUTE)) == 3


def test_trim_is_per_session_across_a_multi_day_frame():
    a = _ts_quotes(["09:34:30", "09:35:00"], day="2026-03-05")
    b = _ts_quotes(["09:34:30", "09:36:00"], day="2026-03-06")
    out = trim_to_before(pd.concat([a, b], ignore_index=True), ENTRY_MINUTE)
    assert len(out) == 2
    assert_decision_quotes_are_pre_entry(out)


def test_trim_is_a_noop_on_empty_input():
    assert trim_to_before(pd.DataFrame()).empty


# ---- Tradeability filter ----


def _summary(half_spread, ask_notional):
    return pd.DataFrame(
        {"half_spread_bps": half_spread, "ask_notional": ask_notional}
    )


def test_untradeable_names_are_filtered_not_silently_dropped():
    q = _summary([30.0, 30.0, 30.0, np.nan], [10_000.0, 500.0, 10_000.0, 10_000.0])
    mask = tradeable_mask(q, order_notional=2_000.0, max_spread_bps=200.0)
    assert list(mask) == [True, False, True, False]


def test_a_book_smaller_than_the_order_is_untradeable():
    q = _summary([10.0], [1_999.0])
    assert not bool(tradeable_mask(q, 2_000.0, 200.0).iloc[0])


def test_a_ruinous_spread_is_untradeable():
    q = _summary([201.0], [1e6])
    assert not bool(tradeable_mask(q, 2_000.0, 200.0).iloc[0])


def test_the_spread_ceiling_is_inclusive_at_the_boundary():
    q = _summary([200.0], [1e6])
    assert bool(tradeable_mask(q, 2_000.0, 200.0).iloc[0])


def test_tradeability_tightens_as_the_order_grows():
    q = _summary([20.0] * 3, [1_000.0, 5_000.0, 50_000.0])
    small = tradeable_mask(q, 2_000.0, 200.0).sum()
    large = tradeable_mask(q, 10_000.0, 200.0).sum()
    assert small == 2 and large == 1


def test_mask_never_returns_na():
    q = _summary([np.nan, 20.0], [np.nan, 1e6])
    assert tradeable_mask(q, 2_000.0, 200.0).isna().sum() == 0


# ---- Cost and edge in one table ----


def _cost_table(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "half_spread_bps": rng.uniform(10, 100, n),
            "impact_bps": rng.uniform(1, 20, n),
            "depth_shortfall_bps": rng.uniform(0, 300, n),
            "measured_round_trip_bps": rng.uniform(50, 700, n),
            "adv_20": rng.uniform(1e5, 5e7, n),
            "ret_c2c": rng.normal(-0.01, 0.1, n),
        }
    )


def test_bucket_summary_can_report_return_beside_cost():
    """Round 3 reported these separately and so never crossed them."""
    out = bucket_summary(_cost_table(), "adv_20", [0, 1e6, 5e6, np.inf], return_col="ret_c2c")
    assert {"gross_mean", "gross_median", "round_trip_bps"} <= set(out.columns)
    assert out["n"].sum() > 0


def test_bucket_summary_without_a_return_column_is_unchanged():
    """The round-3 call signature must keep working."""
    out = bucket_summary(_cost_table(), "adv_20", [0, 1e6, 5e6, np.inf])
    assert "gross_mean" not in out.columns
    assert {"n", "round_trip_bps"} <= set(out.columns)


def test_a_missing_return_column_is_ignored_rather_than_raising():
    out = bucket_summary(_cost_table(), "adv_20", [0, 1e6, np.inf], return_col="not_a_column")
    assert "gross_mean" not in out.columns
