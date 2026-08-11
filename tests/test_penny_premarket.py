"""Tests for premarket features.

The 09:35 boundary is the one that would invalidate every result in the study,
so it is guarded twice: once when the cache is written, once here. Both are
tested.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.features.premarket import (
    add_relative_features,
    assert_no_post_decision_bars,
    premarket_features,
)


def _panel(rows, symbol="TEST"):
    df = pd.DataFrame(rows, columns=["minute_et", "open", "high", "low", "close", "volume"])
    df["symbol"] = symbol
    df["date"] = pd.Timestamp("2026-04-07")
    df["trade_count"] = 5
    return df


def _flat(start_hhmm, n, price=2.0, vol=1000, step=1):
    h, m = map(int, start_hhmm.split(":"))
    base = h * 60 + m
    out = []
    for i in range(n):
        t = base + i * step
        out.append((f"{t // 60:02d}:{t % 60:02d}", price, price, price, price, vol))
    return out


# ---- The leakage boundary ----


def test_guard_rejects_a_bar_at_the_decision_minute():
    panel = _panel(_flat("09:30", 5) + [("09:35", 2, 9, 2, 9, 100)])
    with pytest.raises(AssertionError, match="at or after"):
        assert_no_post_decision_bars(panel)


def test_guard_rejects_bars_after_the_decision():
    panel = _panel(_flat("09:30", 3) + [("10:00", 2, 9, 2, 9, 100)])
    with pytest.raises(AssertionError, match="at or after"):
        assert_no_post_decision_bars(panel)


def test_guard_passes_on_a_clean_premarket_panel():
    assert_no_post_decision_bars(_panel(_flat("04:00", 100, step=3)))


def test_feature_builder_refuses_a_contaminated_panel():
    """The builder must not silently compute features from a leaked bar."""
    panel = _panel(_flat("09:30", 4) + [("09:35", 2, 99, 2, 99, 100)])
    with pytest.raises(AssertionError, match="entry bar"):
        premarket_features(panel)


# ---- Core measurements ----


def test_dollar_volume_and_vwap():
    panel = _panel([
        ("09:00", 2.0, 2.0, 2.0, 2.0, 1000),
        ("09:01", 4.0, 4.0, 4.0, 4.0, 1000),
    ])
    out = premarket_features(panel).iloc[0]
    assert out["pm_dollar_volume"] == pytest.approx(6000.0)
    assert out["pm_vwap"] == pytest.approx(3.0)


def test_close_position_locates_price_in_the_premarket_range():
    """Near the high is accumulation into the open; near the low is not."""
    panel = _panel([
        ("09:00", 1.0, 2.0, 1.0, 1.5, 100),
        ("09:30", 1.5, 2.0, 1.0, 2.0, 100),
    ])
    assert premarket_features(panel).iloc[0]["pm_close_position"] == pytest.approx(1.0)

    panel_low = _panel([
        ("09:00", 2.0, 2.0, 1.0, 1.5, 100),
        ("09:30", 1.5, 2.0, 1.0, 1.0, 100),
    ])
    assert premarket_features(panel_low).iloc[0]["pm_close_position"] == pytest.approx(0.0)


def test_distance_from_vwap_signs_correctly():
    panel = _panel([
        ("09:00", 1.0, 1.0, 1.0, 1.0, 1000),
        ("09:30", 3.0, 3.0, 3.0, 3.0, 1000),
    ])
    out = premarket_features(panel).iloc[0]
    assert out["pm_vwap"] == pytest.approx(2.0)
    assert out["pm_dist_from_vwap"] == pytest.approx(0.5)   # closed 50% above VWAP


def test_volume_centroid_distinguishes_early_from_late_interest():
    """Overnight-news volume that fades differs from interest building into the bell."""
    # 09:20 + 14 bars ends at 09:34; one more would cross the decision minute
    # and the guard would (correctly) reject the fixture.
    early = _panel(_flat("04:00", 20) + _flat("09:20", 2, vol=10))
    late = _panel(_flat("04:00", 2, vol=10) + _flat("09:20", 14))
    assert (premarket_features(early).iloc[0]["pm_vol_centroid"]
            < premarket_features(late).iloc[0]["pm_vol_centroid"])


def test_late_dv_share_measures_concentration_into_the_open():
    late_heavy = _panel(_flat("04:00", 5, vol=10) + _flat("09:10", 20, vol=1000))
    out = premarket_features(late_heavy).iloc[0]
    assert out["pm_late_dv_share"] > 0.9
    assert out["pm_early_dv_share"] < 0.1


def test_late_drift_captures_a_roll_over_into_the_bell():
    rows = _flat("04:00", 10, price=2.0)
    rows += [(f"09:{m:02d}", 3.0, 3.0, 3.0, 3.0 - 0.05 * i, 500)
             for i, m in enumerate(range(10, 34))]
    out = premarket_features(_panel(rows)).iloc[0]
    assert out["pm_late_drift"] < 0, "a fading tape must show negative late drift"


def test_bars_and_minutes_traded_reflect_sparseness():
    sparse = _panel([("04:00", 2, 2, 2, 2, 10), ("09:30", 2, 2, 2, 2, 10)])
    out = premarket_features(sparse).iloc[0]
    assert out["pm_bars"] == 2
    assert out["pm_minutes_traded"] == 2


# ---- Gap features ----


def test_premarket_gap_uses_the_previous_close():
    panel = _panel(_flat("09:00", 5, price=3.0))
    prev = pd.Series({"TEST": 2.0})
    out = premarket_features(panel, prev_close=prev).iloc[0]
    assert out["pm_gap"] == pytest.approx(0.5)


def test_fade_from_high_is_negative_when_price_gives_back_gains():
    rows = [("09:00", 2.0, 4.0, 2.0, 4.0, 100), ("09:30", 4.0, 4.0, 3.0, 3.0, 100)]
    prev = pd.Series({"TEST": 2.0})
    out = premarket_features(_panel(rows), prev_close=prev).iloc[0]
    assert out["pm_high_gap"] == pytest.approx(1.0)      # ran to +100%
    assert out["pm_fade_from_high"] == pytest.approx(-0.25)   # gave back a quarter


# ---- Relative scaling ----


def test_dollar_volume_is_scaled_by_trailing_adv():
    """$500k pre-open means different things at $300k ADV and $20m ADV."""
    feats = pd.DataFrame({
        "symbol": ["THIN", "THICK"], "date": [pd.Timestamp("2026-04-07")] * 2,
        "pm_dollar_volume": [500_000.0, 500_000.0],
    })
    daily = pd.DataFrame({
        "symbol": ["THIN", "THICK"], "date": [pd.Timestamp("2026-04-07")] * 2,
        "adv_20": [300_000.0, 20_000_000.0], "prev_close": [2.0, 2.0],
    })
    out = add_relative_features(feats, daily).set_index("symbol")
    assert out.loc["THIN", "pm_dv_to_adv"] > 1.0
    assert out.loc["THICK", "pm_dv_to_adv"] < 0.1


def test_zero_adv_yields_nan_not_infinity():
    feats = pd.DataFrame({
        "symbol": ["A"], "date": [pd.Timestamp("2026-04-07")], "pm_dollar_volume": [1000.0],
    })
    daily = pd.DataFrame({
        "symbol": ["A"], "date": [pd.Timestamp("2026-04-07")],
        "adv_20": [0.0], "prev_close": [1.0],
    })
    assert np.isnan(add_relative_features(feats, daily).iloc[0]["pm_dv_to_adv"])


# ---- Multi-symbol and edge cases ----


def test_symbols_are_computed_independently():
    a = _panel(_flat("09:00", 5, price=2.0, vol=100), symbol="A")
    b = _panel(_flat("09:00", 5, price=5.0, vol=900), symbol="B")
    out = premarket_features(pd.concat([a, b])).set_index("symbol")
    assert out.loc["B", "pm_dollar_volume"] > out.loc["A", "pm_dollar_volume"]


def test_empty_panel_returns_empty():
    assert premarket_features(pd.DataFrame()).empty


def test_single_bar_does_not_crash_and_yields_nan_where_undefined():
    out = premarket_features(_panel([("09:00", 2, 2, 2, 2, 100)])).iloc[0]
    assert out["pm_bars"] == 1
    assert np.isnan(out["pm_close_position"])   # zero range -> undefined
    assert np.isnan(out["pm_late_drift"])       # too few bars for a trend


def test_no_boolean_features_are_produced():
    """Round 2's lesson: binary flags measured -0.64% against +1.05% continuous."""
    out = premarket_features(_panel(_flat("04:00", 50, step=5)))
    assert not any(out[c].dtype == bool for c in out.columns), "a boolean feature slipped in"
