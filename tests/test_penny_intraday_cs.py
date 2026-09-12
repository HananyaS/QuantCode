"""Tests for research/penny/features/intraday_cs.py.

The failure this module exists to prevent
------------------------------------------
Minute bars are the last untapped source in the project, and the two ways to
ruin them are both silent. The first is a leak: a feature that reads a bar after
the decision point still scores beautifully and is impossible to spot in an IC
table. The second is misalignment: minute partitions are per-session files whose
symbol coverage was set by whichever round last fetched them, so a feature frame
can come back the right shape with the wrong rows in it.

Every feature here is therefore built from a synthetic session whose answer is
known by construction, and the window boundaries are asserted rather than
assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.penny.features import intraday_cs as ic


def _session(symbol: str = "AAA", n: int = 390, day: str = "2024-03-05",
             close=None, volume=None, high=None, low=None,
             start_min: int = 0) -> pd.DataFrame:
    """One synthetic regular-hours session, 09:30 onward."""
    mins = np.arange(start_min, start_min + n)
    et = [f"{9 + (30 + m) // 60:02d}:{(30 + m) % 60:02d}" for m in mins]
    c = np.full(n, 10.0) if close is None else np.asarray(close, dtype=float)
    v = np.full(n, 100.0) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame({
        "symbol": symbol, "date": pd.Timestamp(day), "minute_et": et,
        "open": c, "close": c,
        "high": c if high is None else np.asarray(high, dtype=float),
        "low": c if low is None else np.asarray(low, dtype=float),
        "volume": v, "trade_count": np.full(n, 10.0), "vwap": c,
    })


def _prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["mi"] = ic._minute_index(d["minute_et"])
    return d.sort_values(["symbol", "mi"])


# ---- the time index ------------------------------------------------------
def test_the_minute_index_is_measured_from_the_open():
    s = pd.Series(["09:30", "09:31", "10:00", "15:59", "16:00"])
    assert list(ic._minute_index(s)) == [0, 1, 30, 389, 390]


def test_regular_hours_are_the_only_bars_that_survive_loading():
    """Premarket and post-close prints must not reach a signal-day feature."""
    d = pd.concat([
        _session(n=5, start_min=-5),      # 09:25-09:29, premarket
        _session(n=5, start_min=0),       # 09:30-09:34
    ], ignore_index=True)
    kept = d[(d["minute_et"] >= ic.OPEN_ET) & (d["minute_et"] <= ic.CLOSE_ET)]
    assert len(kept) == 5
    assert kept["minute_et"].min() == "09:30"


# ---- the features --------------------------------------------------------
def test_a_flat_session_has_zero_intraday_volatility():
    f = ic.session_features(_prep(_session()))
    assert f["M_intraday_vol"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_recovery_is_one_when_the_close_is_the_high_and_zero_at_the_low():
    n = 100
    up = _session(n=n, close=np.linspace(8, 12, n),
                  high=np.linspace(8, 12, n), low=np.linspace(8, 12, n))
    assert ic.session_features(_prep(up))["M_recovery"].iloc[0] == pytest.approx(1.0)
    down = _session(n=n, close=np.linspace(12, 8, n),
                    high=np.linspace(12, 8, n), low=np.linspace(12, 8, n))
    assert ic.session_features(_prep(down))["M_recovery"].iloc[0] == pytest.approx(0.0)


def test_time_of_low_locates_the_session_low_as_a_fraction_of_the_day():
    n = 390
    c = np.full(n, 10.0)
    c[97] = 5.0                                  # low a quarter of the way in
    f = ic.session_features(_prep(_session(n=n, close=c, low=c, high=c)))
    assert f["M_time_of_low"].iloc[0] == pytest.approx(97 / 389, abs=1e-6)


def test_open_and_close_participation_sum_the_right_windows():
    n = 390
    v = np.zeros(n)
    v[:15] = 1.0          # all volume in the first 15 minutes
    f = ic.session_features(_prep(_session(n=n, volume=v)))
    assert f["M_open_participation"].iloc[0] == pytest.approx(1.0)
    assert f["M_close_participation"].iloc[0] == pytest.approx(0.0)

    v2 = np.zeros(n)
    v2[375:] = 1.0        # all volume in the last 15
    f2 = ic.session_features(_prep(_session(n=n, volume=v2)))
    assert f2["M_close_participation"].iloc[0] == pytest.approx(1.0)


def test_vwap_position_is_zero_when_the_close_equals_the_session_vwap():
    f = ic.session_features(_prep(_session()))
    assert f["M_vwap_position"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_vwap_position_is_positive_when_the_close_beats_the_average_print():
    n = 100
    c = np.concatenate([np.full(n - 1, 10.0), [12.0]])
    f = ic.session_features(_prep(_session(n=n, close=c, high=c, low=c)))
    assert f["M_vwap_position"].iloc[0] > 0


def test_volume_spikiness_is_one_when_a_single_minute_carries_the_day():
    n = 100
    v = np.zeros(n); v[42] = 1000.0
    f = ic.session_features(_prep(_session(n=n, volume=v)))
    assert f["M_volume_spikiness"].iloc[0] == pytest.approx(1.0)


def test_minutes_traded_is_a_fraction_of_a_full_session():
    f = ic.session_features(_prep(_session(n=195)))
    assert f["M_minutes_traded"].iloc[0] == pytest.approx(0.5)


def test_intraday_drawdown_is_never_positive():
    n = 100
    lo = np.full(n, 10.0); lo[50] = 7.0
    f = ic.session_features(_prep(_session(n=n, low=lo)))
    assert f["M_intraday_drawdown"].iloc[0] == pytest.approx(-0.3)


def test_features_are_computed_per_symbol_not_across_the_session():
    """Two symbols in one partition must not contaminate each other."""
    a = _session("AAA", n=100, close=np.full(100, 10.0))
    b = _session("BBB", n=100, close=np.linspace(5, 15, 100),
                 high=np.linspace(5, 15, 100), low=np.linspace(5, 15, 100))
    f = ic.session_features(_prep(pd.concat([a, b], ignore_index=True)))
    f = f.set_index("symbol")
    assert f.loc["AAA", "M_intraday_vol"] == pytest.approx(0.0, abs=1e-12)
    assert f.loc["BBB", "M_intraday_vol"] > 0
    assert f.loc["AAA", "M_open_to_close"] == pytest.approx(0.0)
    assert f.loc["BBB", "M_open_to_close"] == pytest.approx(2.0)


def test_no_emitted_feature_is_a_price_level():
    """The minute cache is ADJUSTED; a price level in it encodes the future.

    `alpaca_fetch` requests `adjustment="all"`, so a name that later
    reverse-splits carries an inflated adjusted level today. `M_mean_trade_usd`
    was exactly that and scored IC -0.0528 before it was caught. Every feature
    must be invariant to a constant rescaling of the session's prices.
    """
    n = 120
    c = np.linspace(8.0, 12.0, n)
    plain = ic.session_features(_prep(_session(n=n, close=c, high=c, low=c)))
    # A split adjustment moves PRICE and VOLUME in OPPOSITE directions, so a
    # guard that rescales price alone passes a volume level straight through.
    # That is exactly how M_shares_per_trade survived the first version of this
    # test and reached the holdout as a look-ahead.
    scaled = ic.session_features(_prep(_session(
        n=n, close=c * 750.0, high=c * 750.0, low=c * 750.0,
        volume=np.full(n, 100.0) / 750.0)))
    cols = [x for x in plain.columns if x.startswith("M_")]
    for col in cols:
        a, b = plain[col].iloc[0], scaled[col].iloc[0]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == pytest.approx(b, rel=1e-6), (
            f"{col} moved when prices were rescaled 750x -- it is a level, and "
            f"an adjusted level is a look-ahead")


def test_no_trade_size_feature_is_emitted_at_all():
    """Both attempts at a trade-size feature were look-aheads.

    The dollar version is a price level; the shares version is a volume level,
    and volume is adjusted too -- in the opposite direction. There is no way to
    express "retail-sized prints" from this cache without a level, so it is not
    expressed.
    """
    f = ic.session_features(_prep(_session(n=100)))
    for dead in ("M_mean_trade_usd", "M_shares_per_trade"):
        assert dead not in f.columns, f"{dead} is a look-ahead and must stay out"


def test_an_empty_session_returns_an_empty_frame_rather_than_raising():
    assert ic.session_features(pd.DataFrame()).empty


def test_every_emitted_column_is_prefixed_so_the_ledger_can_count_it():
    f = ic.session_features(_prep(_session()))
    extra = {"symbol", "date"}
    assert all(c.startswith("M_") for c in f.columns if c not in extra), (
        "intraday features must be distinguishable from the daily B* blocks")


def test_a_missing_partition_loads_as_empty_rather_than_raising():
    assert ic.load_session(pd.Timestamp("1990-01-02")).empty
