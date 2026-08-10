"""Tests for triple-barrier labelling.

The label is the dependent variable, so an optimistic bug here becomes a fake
edge everywhere downstream. Each of the five stated pessimistic assumptions is
asserted, not merely documented.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.labels.triple_barrier import (
    BarrierSpec,
    label_one,
    label_day,
    observation_features,
)


def _path(rows, symbol="TEST"):
    """Build a minute path from (minute, o, h, l, c, v) tuples."""
    df = pd.DataFrame(rows, columns=["minute_et", "open", "high", "low", "close", "volume"])
    df["symbol"] = symbol
    df["trade_count"] = 10
    return df


def _flat_path(start_minute=36, n=100, price=1.0):
    rows = []
    for i in range(n):
        m = 9 * 60 + start_minute + i
        rows.append((f"{m // 60:02d}:{m % 60:02d}", price, price, price, price, 1000))
    return rows


DAY = pd.Timestamp("2026-03-02")
# Zero frictions isolates barrier logic from cost logic in most tests.
FRICTIONLESS = BarrierSpec(
    half_spread_floor_bps=0.0, stop_slippage_bps=0.0, impact_coefficient=0.0
)


# ---- Entry ----


def test_missing_decision_bar_is_not_fillable():
    """No print at 09:35 means the trade could not be entered."""
    path = _path([("09:30", 1, 1, 1, 1, 100), ("09:40", 1, 1, 1, 1, 100)])
    res = label_one("T", DAY, path, FRICTIONLESS)
    assert res.outcome == "no_fill"
    assert not np.isfinite(res.net_return)


def test_entry_uses_the_open_of_the_decision_bar():
    path = _path([("09:35", 2.0, 2.5, 1.9, 2.1, 500)] + _flat_path(price=2.1))
    res = label_one("T", DAY, path, FRICTIONLESS)
    assert res.entry_price == pytest.approx(2.0)


# ---- Barriers ----


def test_target_hit_exits_at_the_target():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.40, 1.0, 1.35, 900)]
    rows += _flat_path(start_minute=37, n=20, price=1.35)
    res = label_one("T", DAY, _path(rows), FRICTIONLESS)
    assert res.outcome == "target"
    assert res.exit_price == pytest.approx(1.30)      # +30% barrier
    assert res.net_return == pytest.approx(0.30, abs=1e-9)


def test_stop_hit_exits_at_the_stop():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.0, 0.85, 0.86, 900)]
    rows += _flat_path(start_minute=37, n=20, price=0.86)
    res = label_one("T", DAY, _path(rows), FRICTIONLESS)
    assert res.outcome == "stop"
    assert res.exit_price == pytest.approx(0.90)      # -10% barrier
    assert res.net_return == pytest.approx(-0.10, abs=1e-9)


def test_vertical_barrier_exits_at_the_final_close():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100)] + _flat_path(price=1.05)
    rows.append(("15:55", 1.05, 1.06, 1.04, 1.07, 500))
    res = label_one("T", DAY, _path(rows), FRICTIONLESS)
    assert res.outcome == "vertical"
    assert res.exit_price == pytest.approx(1.07)


def test_earlier_barrier_wins_when_both_are_eventually_touched():
    rows = [
        ("09:35", 1.0, 1.0, 1.0, 1.0, 100),
        ("09:36", 1.0, 1.0, 0.85, 0.88, 900),   # stop first
        ("09:37", 0.88, 1.50, 0.88, 1.40, 900), # target later
    ]
    res = label_one("T", DAY, _path(rows), FRICTIONLESS)
    assert res.outcome == "stop", "the first barrier touched must win"


# ---- Assumption 1: ambiguity resolves against the trade ----


def test_bar_touching_both_barriers_resolves_to_the_stop():
    """A single bar cannot say which barrier came first.

    Resolving optimistically would manufacture edge out of bar granularity
    alone -- and microcap minute bars are wide enough for this to be common.
    """
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.45, 0.80, 1.20, 900)]
    rows += _flat_path(start_minute=37, n=10, price=1.20)
    res = label_one("T", DAY, _path(rows), FRICTIONLESS)
    assert res.outcome == "stop"
    assert res.ambiguous_bar is True
    assert res.net_return < 0


# ---- Assumption 3: stops slip, targets do not ----


def test_stop_slippage_makes_the_loss_worse_than_the_barrier():
    spec = BarrierSpec(half_spread_floor_bps=0.0, impact_coefficient=0.0, stop_slippage_bps=50.0)
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.0, 0.85, 0.86, 900)]
    rows += _flat_path(start_minute=37, n=10, price=0.86)
    res = label_one("T", DAY, _path(rows), spec)
    assert res.net_return < -0.10, "a triggered stop must cost more than the barrier"


def test_target_does_not_slip():
    spec = BarrierSpec(half_spread_floor_bps=0.0, impact_coefficient=0.0, stop_slippage_bps=50.0)
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.40, 1.0, 1.35, 900)]
    rows += _flat_path(start_minute=37, n=10, price=1.35)
    res = label_one("T", DAY, _path(rows), spec)
    assert res.net_return == pytest.approx(0.30, abs=1e-9)


# ---- Assumption 4: entry pays the spread ----


def test_costs_make_net_worse_than_gross_on_a_winner():
    spec = BarrierSpec(half_spread_floor_bps=25.0, impact_coefficient=0.10)
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.60, 1.0, 1.55, 900)]
    rows += _flat_path(start_minute=37, n=10, price=1.55)
    res = label_one("T", DAY, _path(rows), spec)
    assert res.net_return < res.gross_return
    assert res.cost_bps > 0
    assert res.entry_price_filled > res.entry_price, "entry must pay the spread"


def test_wider_bars_imply_a_wider_estimated_spread():
    """The high-low range proxies the spread when quotes are unavailable."""
    tight = [("09:35", 1.0, 1.001, 0.999, 1.0, 100)] + _flat_path(n=30, price=1.0)
    wide = [("09:35", 1.0, 1.10, 0.90, 1.0, 100)]
    wide += [(m, 1.0, 1.10, 0.90, 1.0, 100) for m, *_ in _flat_path(n=30, price=1.0)]
    spec = BarrierSpec(half_spread_floor_bps=1.0, impact_coefficient=0.0)
    r_tight = label_one("T", DAY, _path(tight), spec)
    r_wide = label_one("T", DAY, _path(wide), spec)
    assert r_wide.cost_bps > r_tight.cost_bps


# ---- Assumption 2 & 5: coverage and halt suspicion ----


def test_coverage_reflects_sparse_prints():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("10:00", 1.0, 1.0, 1.0, 1.0, 100)]
    res = label_one("T", DAY, _path(rows), FRICTIONLESS)
    assert res.coverage < 0.05


def test_full_session_has_high_coverage():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100)]
    for i in range(1, 381):
        m = 9 * 60 + 35 + i
        if m > 15 * 60 + 55:
            break
        rows.append((f"{m // 60:02d}:{m % 60:02d}", 1.0, 1.0, 1.0, 1.0, 100))
    res = label_one("T", DAY, _path(rows), FRICTIONLESS)
    assert res.coverage > 0.95


def test_long_data_gap_is_flagged_as_halt_suspect():
    """A gap cannot be distinguished from a halt without paid halt history."""
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.0, 1.0, 1.0, 100)]
    rows += [("11:00", 1.0, 1.0, 1.0, 1.0, 100), ("11:01", 1.0, 1.0, 1.0, 1.0, 100)]
    res = label_one("T", DAY, _path(rows), FRICTIONLESS)
    assert res.halt_suspect is True


def test_contiguous_path_is_not_halt_suspect():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100)] + _flat_path(n=60, price=1.0)
    res = label_one("T", DAY, _path(rows), FRICTIONLESS)
    assert res.halt_suspect is False


# ---- The leakage boundary ----


def test_observation_features_never_touch_the_decision_bar_or_later():
    """This function is the intraday leakage boundary."""
    rows = [
        ("09:30", 1.00, 1.05, 0.98, 1.02, 500),
        ("09:31", 1.02, 1.08, 1.01, 1.06, 400),
        ("09:34", 1.06, 1.10, 1.04, 1.09, 300),
        ("09:35", 1.09, 9.99, 1.09, 9.99, 900),   # decision bar: must be invisible
        ("10:00", 9.99, 99.0, 9.99, 99.0, 900),   # future: must be invisible
    ]
    feats = observation_features(_path(rows))
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["minutes_obs"] == 3
    assert row["high_obs"] == pytest.approx(1.10), "a post-decision high leaked in"
    assert row["close_obs"] == pytest.approx(1.09)
    assert row["volume_obs"] == 1200


def test_observation_close_position_locates_price_in_the_opening_range():
    rows = [
        ("09:30", 1.00, 2.00, 1.00, 1.50, 500),
        ("09:34", 1.50, 2.00, 1.00, 2.00, 500),
    ]
    feats = observation_features(_path(rows))
    assert feats.iloc[0]["obs_close_position"] == pytest.approx(1.0)


def test_observation_window_empty_returns_empty():
    rows = [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:40", 1.0, 1.0, 1.0, 1.0, 100)]
    assert observation_features(_path(rows)).empty


# ---- Multi-symbol ----


def test_label_day_handles_multiple_symbols_independently():
    winner = _path(
        [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.40, 1.0, 1.35, 900)]
        + _flat_path(start_minute=37, n=10, price=1.35),
        symbol="WIN",
    )
    loser = _path(
        [("09:35", 1.0, 1.0, 1.0, 1.0, 100), ("09:36", 1.0, 1.0, 0.80, 0.82, 900)]
        + _flat_path(start_minute=37, n=10, price=0.82),
        symbol="LOSE",
    )
    out = label_day(pd.concat([winner, loser]), DAY, FRICTIONLESS)
    assert set(out["symbol"]) == {"WIN", "LOSE"}
    assert out.set_index("symbol").loc["WIN", "outcome"] == "target"
    assert out.set_index("symbol").loc["LOSE", "outcome"] == "stop"
