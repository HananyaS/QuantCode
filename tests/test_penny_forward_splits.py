"""Tests for forward-split detection.

A forward split prints as a large overnight *decline*, which is exactly the
shape a buy-the-crash rule selects on. The reverse-split detector cannot see it,
so these tests pin the behaviour that keeps a unit change out of a long screen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.penny.etl.corporate_actions import (  # noqa: E402
    flag_forward_splits,
    flag_reverse_splits,
)


def _frame(rows):
    """rows: [prev_close, open, high, low]"""
    return pd.DataFrame(rows, columns=["prev_close_unadj", "open", "high", "low"])


def test_clean_ten_for_one_forward_split_is_flagged():
    """$50 -> $5 overnight with an inert session is a 1:10 forward split."""
    df = _frame([[50.0, 5.0, 5.05, 4.98]])
    out = flag_forward_splits(df)
    assert bool(out["is_suspect_fwd_split"].iloc[0])
    assert out["fwd_split_divisor"].iloc[0] == pytest.approx(10.0)


def test_real_crash_with_a_live_session_is_not_flagged():
    """The TENX case, with its real prices.

    -89.8% overnight, divisor 9.74 (within tolerance of 10), and an
    open-to-high of only 2.2% -- which would read as inert. The session range
    is 39.7%, which is what actually distinguishes it from a unit change.
    """
    df = _frame([[13.44, 1.79, 1.83, 1.31]])
    out = flag_forward_splits(df)
    assert out["fwd_split_divisor"].iloc[0] == pytest.approx(7.508, rel=1e-3)
    assert not bool(out["is_suspect_fwd_split"].iloc[0])


def test_inertness_uses_range_not_open_to_high():
    """A name that opens and collapses must not read as inert.

    Same round divisor and the same tiny open-to-high in both rows; only the
    low differs. The regression this pins: measuring inertness on open-to-high
    flags the second row, because a stock that never rallies looks quiet.
    """
    df = _frame([[50.0, 5.0, 5.05, 4.98],    # genuinely inert
                 [50.0, 5.0, 5.05, 3.20]])   # opened, then collapsed 36%
    out = flag_forward_splits(df)
    assert out["is_suspect_fwd_split"].tolist() == [True, False]


def test_ordinary_decline_is_not_flagged():
    df = _frame([[10.0, 8.5, 8.6, 8.4]])
    out = flag_forward_splits(df)
    assert not bool(out["is_suspect_fwd_split"].iloc[0])
    assert not bool(out["is_possible_fwd_split"].iloc[0])


def test_off_ratio_decline_is_not_flagged():
    """A 9x divisor falls in the gap between the 8 and 10 ratios.

    `nearest_ratio` breaks the tie toward 8, giving a 12.5% error against a 12%
    tolerance. The ratio table is dense, so genuine gaps are narrow -- this
    documents where one is.
    """
    df = _frame([[90.0, 10.0, 10.1, 9.9]])
    out = flag_forward_splits(df)
    assert out["fwd_split_divisor"].iloc[0] == pytest.approx(9.0)
    assert not bool(out["is_suspect_fwd_split"].iloc[0])


def test_reverse_and_forward_detectors_do_not_overlap():
    """The two detectors partition the space; nothing is both."""
    df = _frame([[1.0, 10.0, 10.1, 9.9],     # 1:10 reverse
                 [10.0, 1.0, 1.01, 0.99],    # 1:10 forward
                 [10.0, 10.2, 10.3, 10.1]])  # neither
    df["o2h"] = df["high"] / df["open"] - 1.0
    rev = flag_reverse_splits(df)
    fwd = flag_forward_splits(df)
    assert rev["is_suspect_split"].tolist() == [True, False, False]
    assert fwd["is_suspect_fwd_split"].tolist() == [False, True, False]
    assert not (rev["is_suspect_split"] & fwd["is_suspect_fwd_split"]).any()


def test_missing_and_zero_prices_do_not_raise():
    df = _frame([[np.nan, 5.0, 5.1, 4.9], [0.0, 5.0, 5.1, 4.9],
                 [50.0, np.nan, np.nan, np.nan]])
    out = flag_forward_splits(df)
    assert not out["is_suspect_fwd_split"].any()
    assert len(out) == 3


def test_flag_is_boolean_not_object():
    """Downstream code does `~mask`, which silently misbehaves on object dtype."""
    df = _frame([[50.0, 5.0, 5.05, 4.98], [10.0, 9.0, 9.1, 8.9]])
    out = flag_forward_splits(df)
    assert out["is_suspect_fwd_split"].dtype == bool
    assert out["is_possible_fwd_split"].dtype == bool
