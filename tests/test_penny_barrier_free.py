"""Tests for barrier-free path statistics.

These exist because round 1 reported truncated excursions as if they described
the opportunity rather than the trade, and the error was invisible in the
output -- P(MFE >= 20%) = 0.004 looks like a fact, not an artifact.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.labels.barrier_free import (
    assert_dominates_truncated,
    meta_label,
    path_stats_day,
    path_stats_one,
)
from research.penny.labels.triple_barrier import BarrierSpec, label_one

DAY = pd.Timestamp("2026-04-07")


def _path(rows, symbol="TEST"):
    df = pd.DataFrame(rows, columns=["minute_et", "open", "high", "low", "close", "volume"])
    df["symbol"] = symbol
    df["trade_count"] = 10
    return df


def _minutes(start=36, n=60):
    for i in range(n):
        m = 9 * 60 + start + i
        yield f"{m // 60:02d}:{m % 60:02d}"


# ---- Core measurement ----


def test_mfe_is_measured_over_the_whole_session():
    """The defining property: no barrier truncates it."""
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100)]
    mins = list(_minutes(n=10))
    rows.append((mins[0], 1.0, 1.60, 1.0, 1.50, 500))     # +60% early
    for m in mins[1:]:
        rows.append((m, 1.50, 1.50, 1.45, 1.48, 200))
    stats = path_stats_one("T", DAY, _path(rows))
    assert stats.mfe_full == pytest.approx(0.60)


def test_barrier_free_mfe_exceeds_the_truncated_one():
    """The exact round-1 bug, pinned.

    With a +10% target the trade exits at +10%, so the truncated MFE can never
    exceed it -- but the price kept going to +60%.
    """
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100)]
    mins = list(_minutes(n=12))
    rows.append((mins[0], 1.0, 1.12, 1.0, 1.11, 500))     # crosses +10%
    rows.append((mins[1], 1.11, 1.60, 1.10, 1.55, 900))   # then runs to +60%
    for m in mins[2:]:
        rows.append((m, 1.55, 1.56, 1.50, 1.52, 200))
    path = _path(rows)

    free = path_stats_one("T", DAY, path)
    trade = label_one("T", DAY, path, BarrierSpec(
        profit_target=0.10, stop_loss=0.05,
        half_spread_floor_bps=0.0, impact_coefficient=0.0, stop_slippage_bps=0.0,
    ))

    assert trade.outcome == "target"
    assert trade.max_favourable < 0.15, "truncated MFE is capped near the target"
    assert free.mfe_full == pytest.approx(0.60)
    assert free.mfe_full > trade.max_favourable


def test_mae_is_measured_over_the_whole_session():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100)]
    mins = list(_minutes(n=8))
    rows.append((mins[0], 1.0, 1.0, 0.40, 0.45, 900))     # -60%
    for m in mins[1:]:
        rows.append((m, 0.45, 0.50, 0.45, 0.48, 200))
    stats = path_stats_one("T", DAY, _path(rows))
    assert stats.mae_full == pytest.approx(-0.60)


def test_hold_to_close_uses_the_final_bar():
    rows = [("09:35", 2.0, 2.0, 2.0, 2.0, 100)]
    mins = list(_minutes(n=5))
    for m in mins[:-1]:
        rows.append((m, 2.0, 3.0, 1.5, 2.5, 200))
    rows.append(("15:55", 2.5, 2.6, 2.4, 2.40, 300))
    stats = path_stats_one("T", DAY, _path(rows))
    assert stats.ret_c2c == pytest.approx(0.20)


def test_ordering_of_extremes_is_recorded():
    """Whether the high came before the low decides what a stop would have done."""
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100)]
    mins = list(_minutes(n=6))
    rows.append((mins[0], 1.0, 1.50, 1.0, 1.40, 500))     # high first
    rows.append((mins[1], 1.40, 1.40, 0.60, 0.70, 500))   # low second
    for m in mins[2:]:
        rows.append((m, 0.70, 0.75, 0.65, 0.70, 100))
    stats = path_stats_one("T", DAY, _path(rows))
    assert stats.mfe_before_mae is True
    assert stats.minutes_to_mfe < stats.minutes_to_mae


def test_low_first_is_recorded_as_such():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100)]
    mins = list(_minutes(n=6))
    rows.append((mins[0], 1.0, 1.0, 0.60, 0.70, 500))
    rows.append((mins[1], 0.70, 1.50, 0.70, 1.40, 500))
    for m in mins[2:]:
        rows.append((m, 1.40, 1.45, 1.35, 1.40, 100))
    stats = path_stats_one("T", DAY, _path(rows))
    assert stats.mfe_before_mae is False


# ---- Unenterable cases ----


def test_missing_decision_bar_yields_nothing():
    rows = [("09:30", 1.0, 1.0, 1.0, 1.0, 100), ("09:40", 1.0, 1.0, 1.0, 1.0, 100)]
    assert path_stats_one("T", DAY, _path(rows)) is None


def test_empty_scan_window_yields_nothing():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100)]
    assert path_stats_one("T", DAY, _path(rows)) is None


def test_zero_entry_price_yields_nothing():
    rows = [("09:35", 0.0, 0.0, 0.0, 0.0, 100), ("09:36", 1.0, 1.0, 1.0, 1.0, 100)]
    assert path_stats_one("T", DAY, _path(rows)) is None


# ---- Multi-symbol ----


def test_day_level_handles_symbols_independently():
    up = _path(
        [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 2.0, 1.0, 1.9, 900)],
        symbol="UP",
    )
    down = _path(
        [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.0, 0.5, 0.6, 900)],
        symbol="DOWN",
    )
    out = path_stats_day(pd.concat([up, down]), DAY).set_index("symbol")
    assert out.loc["UP", "mfe_full"] == pytest.approx(1.0)
    assert out.loc["DOWN", "mae_full"] == pytest.approx(-0.5)


# ---- The dominance guard ----


def test_dominance_guard_accepts_a_valid_pair():
    assert_dominates_truncated(pd.Series([0.6, 0.3]), pd.Series([0.1, 0.3]))


def test_dominance_guard_catches_a_mismatched_join():
    """If full < truncated, the two are not the same path."""
    with pytest.raises(AssertionError, match="not measured over the same path"):
        assert_dominates_truncated(pd.Series([0.05]), pd.Series([0.40]))


# ---- Meta-label ----


def test_meta_label_threshold_is_the_cost_not_zero():
    """A +20bps gross trade is a loser once 437bps is paid."""
    ret = pd.Series([0.002, 0.05, -0.01])
    labels = meta_label(ret, cost=0.0437)
    assert labels.tolist() == [0, 1, 0]


def test_meta_label_rejects_short_direction():
    with pytest.raises(ValueError, match="only long"):
        meta_label(pd.Series([0.1]), cost=0.01, direction="short")


def test_meta_label_is_monotone_in_cost():
    ret = pd.Series([0.01, 0.03, 0.05])
    cheap = meta_label(ret, cost=0.005).sum()
    dear = meta_label(ret, cost=0.045).sum()
    assert cheap > dear, "a higher cost bar can only reduce the positive class"
