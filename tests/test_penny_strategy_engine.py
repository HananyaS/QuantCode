"""Tests for the strategy engine and the anti-bias screens.

Each test here corresponds to a specific failure that has already happened in
this project. They exist so those failures cannot recur silently:

* §72 — a selection filter on adjusted prices (survivorship built from the future)
* §74 — time-series operations on a PRE-FILTERED frame, so shifts walked over
  surviving rows rather than sessions
* §57 / §65 — corporate actions inside the holding window, and delistings
  dropped by `notna()` rather than treated as an outcome
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.penny.etl.screens import (  # noqa: E402
    AdjustedPriceInScreenError, adjustment_tripwire, assert_unadjusted, price_band,
)
from research.penny.strategy.engine import (  # noqa: E402
    Spec, build_trades, forward_paths, max_lookback_span,
)


# ---------------------------------------------------------------- screens --
def test_adjusted_price_level_in_a_filter_raises():
    with pytest.raises(AdjustedPriceInScreenError, match="adjusted price level"):
        assert_unadjusted(["close"])


def test_unadjusted_and_ratio_columns_are_allowed():
    assert_unadjusted(["close_unadj", "adv20", "ret_5d_unadj", "c2c_unadj", "volume"])


def test_price_band_refuses_an_adjusted_column():
    df = pd.DataFrame({"close": [1.0], "close_unadj": [1.0]})
    with pytest.raises(AdjustedPriceInScreenError):
        price_band(df, 0.5, 2.0, col="close")


def test_price_band_is_half_open():
    df = pd.DataFrame({"close_unadj": [0.49, 0.50, 1.99, 2.00]})
    assert price_band(df, 0.50, 2.00).tolist() == [False, True, True, False]


def test_tripwire_flags_a_population_full_of_future_splits():
    df = pd.DataFrame({"close": [10.0, 10.0, 1.0], "close_unadj": [1.0, 1.0, 1.0]})
    out = adjustment_tripwire(df, np.array([True, True, True]))
    assert out["share_gt_1_5"] == pytest.approx(2 / 3)
    assert out["max_ratio"] == pytest.approx(10.0)


# ------------------------------------------------------------ forward path --
def _panel(rows):
    """rows: (symbol, date, o, h, l, c)"""
    df = pd.DataFrame(rows, columns=["symbol", "date", "open_unadj", "high_unadj",
                                     "low_unadj", "close_unadj"])
    df["date"] = pd.to_datetime(df["date"])
    df["any_split"] = False
    df["is_suspect_fwd_split"] = False
    return df


def test_forward_path_stops_at_the_symbol_boundary():
    u = _panel([("A", "2020-01-02", 1, 1, 1, 1), ("A", "2020-01-03", 2, 2, 2, 2),
                ("B", "2020-01-02", 9, 9, 9, 9), ("B", "2020-01-03", 8, 8, 8, 8)])
    p = forward_paths(u, np.array([1]), 2)     # last row of A
    assert p["valid"][0].tolist() == [False, False], "path leaked into symbol B"


def test_forward_path_reads_consecutive_rows():
    u = _panel([("A", "2020-01-02", 1, 1, 1, 1), ("A", "2020-01-03", 2, 3, 2, 2),
                ("A", "2020-01-06", 4, 5, 4, 4)])
    p = forward_paths(u, np.array([0]), 2)
    assert p["valid"][0].tolist() == [True, True]
    assert p["high"][0].tolist() == [3.0, 5.0]


# ------------------------------------------------------------ the §74 bug --
def test_signal_rejects_a_lookback_that_spans_a_data_hole():
    """The §74 bug: a '5-day crash' measured across a three-month gap.

    Two identical signals; only `span_5d` differs. The one whose window really
    spans 97 calendar days must not be selected.
    """
    n = 2
    u = pd.DataFrame({
        "symbol": ["A", "B"], "date": pd.to_datetime(["2020-06-01", "2020-06-01"]),
        "open_unadj": [1.0, 1.0], "high_unadj": [1.0, 1.0], "low_unadj": [1.0, 1.0],
        "close_unadj": [1.0, 1.0], "prev_close_unadj": [2.0, 2.0],
        "prev_high_unadj": [2.0, 2.0], "prev_low_unadj": [1.0, 1.0],
        "volume": [1e6] * n, "prev_volume": [1e5] * n,
        "adv20": [1e6] * n, "cost": [0.015] * n,
        "c2c_unadj": [0.10, 0.10], "c2c_prev_unadj": [0.0, 0.0],
        "ret_5d_unadj": [-0.60, -0.60],
        "span_5d": [7, 97],                       # <-- the only difference
        "close_5b_unadj": [2.5, 2.5], "low_5d_unadj": [0.9, 0.9],
        "any_split": [False, False], "is_suspect_fwd_split": [False, False],
        "in_universe": [True, True],
    })
    from research.penny.strategy.engine import signal_mask
    spec = Spec(name="t", drop_pct=0.50, drop_window=5, confirm="green_day")
    assert signal_mask(u, spec).tolist() == [True, False]


def test_max_lookback_span_scales_with_the_window():
    assert max_lookback_span(5) == pytest.approx(18.0)
    assert max_lookback_span(20) == pytest.approx(42.0)
    assert max_lookback_span(20) > max_lookback_span(5)


def test_signal_requires_in_universe():
    """Eligibility is a column now; a row outside the band must not be selected
    even though every other condition holds."""
    from research.penny.strategy.engine import signal_mask
    base = dict(symbol=["A"], date=pd.to_datetime(["2020-06-01"]),
                open_unadj=[1.0], high_unadj=[1.0], low_unadj=[1.0],
                close_unadj=[1.0], prev_close_unadj=[2.0], prev_high_unadj=[2.0],
                prev_low_unadj=[1.0], volume=[1e6], prev_volume=[1e5],
                adv20=[1e6], cost=[0.015], c2c_unadj=[0.10], c2c_prev_unadj=[0.0],
                ret_5d_unadj=[-0.60], span_5d=[7], close_5b_unadj=[2.5],
                low_5d_unadj=[0.9], any_split=[False], is_suspect_fwd_split=[False])
    spec = Spec(name="t", drop_pct=0.50, drop_window=5, confirm="green_day")
    assert signal_mask(pd.DataFrame({**base, "in_universe": [True]}), spec).tolist() == [True]
    assert signal_mask(pd.DataFrame({**base, "in_universe": [False]}), spec).tolist() == [False]


# ------------------------------------------------------------ spec guards --
def test_spec_validate_rejects_nonsense():
    for bad in (Spec(name="a", confirm="wishful"),
                Spec(name="b", stop_mode="vibes"),
                Spec(name="c", drop_pct=1.5),
                Spec(name="d", max_hold=0),
                Spec(name="e", price_lo=5.0, price_hi=1.0)):
        with pytest.raises(AssertionError):
            bad.validate()


def test_spec_roundtrips_through_json(tmp_path):
    s = Spec(name="rt", drop_pct=0.42, max_hold=17, confirm="two_green")
    p = tmp_path / "s.json"
    p.write_text(s.to_json(), encoding="utf-8")
    back = Spec.load(p)
    assert back.drop_pct == pytest.approx(0.42)
    assert back.max_hold == 17 and back.confirm == "two_green"


def test_spec_ignores_unknown_keys():
    s = Spec.from_dict({"name": "x", "drop_pct": 0.3, "not_a_field": 99})
    assert s.name == "x" and s.drop_pct == pytest.approx(0.3)
