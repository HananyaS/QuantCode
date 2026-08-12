"""Market regime and idiosyncrasy features.

No index symbol exists in the daily cache — the universe fetch was
microcap-focused — so the market factor is built from the studied cross-section.
That is the more relevant benchmark for this segment anyway, but it makes the
leakage surface larger: the factor is computed from the same rows the model
scores, so a missing ``shift(1)`` would hand the model the market's own return
for the day it is trying to trade.

That failure is invisible in any single number and enormously flattering, which
is what these tests exist for.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.features.market_regime import (
    REGIME_FEATURES,
    assert_regime_is_causal,
    attach_regime,
    build_market_factor,
)


def _panel(n_days=40, n_names=50, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for i, d in enumerate(days):
        shock = drift * np.sin(i / 3.0)
        for k in range(n_names):
            rows.append({
                "symbol": f"S{k}", "date": d,
                "daily_ret": float(rng.normal(shock, 0.03)),
                "gap": float(rng.normal(shock, 0.02)),
                "ret_5d_prev": float(rng.normal(0, 0.06)),
            })
    return pd.DataFrame(rows)


# ---- The one-day shift ----


def test_trailing_market_return_is_the_previous_session():
    p = _panel(n_days=6, n_names=40)
    mkt = build_market_factor(p, min_names=5).set_index("date")
    realised = p.groupby("date")["daily_ret"].median()
    days = list(realised.index)
    for prev, cur in zip(days, days[1:]):
        assert mkt.loc[cur, "mkt_ret_1d_prev"] == pytest.approx(realised.loc[prev])


def test_the_first_session_has_no_trailing_state():
    p = _panel(n_days=5, n_names=40)
    mkt = build_market_factor(p, min_names=5).sort_values("date")
    assert np.isnan(mkt.iloc[0]["mkt_ret_1d_prev"])


def test_causality_tripwire_passes_on_a_correct_factor():
    p = _panel(n_days=30, n_names=40)
    assert_regime_is_causal(build_market_factor(p, min_names=5), p)


def test_causality_tripwire_catches_a_missing_shift():
    """The exact bug: same-session market return handed to the model."""
    p = _panel(n_days=30, n_names=40)
    mkt = build_market_factor(p, min_names=5)
    same = p.groupby("date")["daily_ret"].median().rename("r").reset_index()
    broken = mkt.drop(columns=["mkt_ret_1d_prev"]).merge(same, on="date")
    broken = broken.rename(columns={"r": "mkt_ret_1d_prev"})
    with pytest.raises(AssertionError, match="one-day shift is missing"):
        assert_regime_is_causal(broken, p)


# ---- Same-day open state is allowed, later state is not ----


def test_market_gap_today_is_unshifted_because_it_is_known_at_0930():
    p = _panel(n_days=6, n_names=40)
    mkt = build_market_factor(p, min_names=5).set_index("date")
    realised_gap = p.groupby("date")["gap"].median()
    for d in realised_gap.index:
        assert mkt.loc[d, "mkt_gap_today"] == pytest.approx(realised_gap.loc[d])


def test_market_factor_never_reads_a_close_column():
    """Same-day fields must derive from opening prints only."""
    p = _panel(n_days=6, n_names=40)
    p["close"] = 999.0
    a = build_market_factor(p, min_names=5)
    p2 = p.copy()
    p2["close"] = -999.0
    b = build_market_factor(p2, min_names=5)
    pd.testing.assert_frame_equal(a, b)


# ---- Idiosyncrasy: the mechanism feature ----


def test_gap_vs_market_isolates_the_idiosyncratic_part():
    cands = pd.DataFrame({
        "symbol": ["A", "B"], "date": [pd.Timestamp("2024-03-05")] * 2,
        "gap": [-0.06, -0.06], "ret_5d_prev": [0.0, 0.0],
    })
    mkt = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-05")],
        "mkt_gap_today": [-0.05], "mkt_ret_5d_prev": [0.0],
        "mkt_ret_1d_prev": [0.0],
    })
    out = attach_regime(cands, mkt)
    # Down 6% on a day the complex is down 5% is a 1% idiosyncratic move.
    assert out["gap_vs_market"].iloc[0] == pytest.approx(-0.01)


def test_a_systematic_selloff_and_an_idiosyncratic_one_are_distinguished():
    cands = pd.DataFrame({
        "symbol": ["A", "B"],
        "date": [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-06")],
        "gap": [-0.06, -0.06], "ret_5d_prev": [0.0, 0.0],
    })
    mkt = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-06")],
        "mkt_gap_today": [-0.055, 0.0],
        "mkt_ret_5d_prev": [0.0, 0.0], "mkt_ret_1d_prev": [0.0, 0.0],
    })
    out = attach_regime(cands, mkt).set_index("symbol")
    assert out.loc["A", "gap_vs_market"] > out.loc["B", "gap_vs_market"]


def test_relative_strength_subtracts_the_market():
    cands = pd.DataFrame({
        "symbol": ["A"], "date": [pd.Timestamp("2024-03-05")],
        "gap": [-0.05], "ret_5d_prev": [0.10],
    })
    mkt = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-05")], "mkt_gap_today": [0.0],
        "mkt_ret_5d_prev": [0.04], "mkt_ret_1d_prev": [0.0],
    })
    assert attach_regime(cands, mkt)["rel_strength_5d"].iloc[0] == pytest.approx(0.06)


# ---- Robustness ----


def test_thin_sessions_are_blanked_not_interpolated():
    """A handful of names is not a market."""
    full = _panel(n_days=3, n_names=40)
    thin = _panel(n_days=1, n_names=3, seed=9)
    # Push the thin session onto its own date; overlapping it would just make
    # a full day fuller and test nothing.
    thin["date"] = full["date"].max() + pd.Timedelta(days=1)
    mkt = build_market_factor(pd.concat([full, thin], ignore_index=True),
                              min_names=10).sort_values("date")

    assert mkt["n_names"].min() == 3, "fixture did not create a thin session"
    assert np.isnan(mkt.iloc[-1]["mkt_gap_today"]), "thin session was not blanked"
    assert mkt.iloc[0]["mkt_gap_today"] == mkt.iloc[0]["mkt_gap_today"], (
        "a full session was blanked too"
    )


def test_breadth_is_a_share_between_zero_and_one():
    mkt = build_market_factor(_panel(n_days=10, n_names=40), min_names=5)
    b = mkt["mkt_breadth_prev"].dropna()
    assert b.between(0, 1).all()


def test_drawdown_is_never_positive():
    mkt = build_market_factor(_panel(n_days=40, n_names=30, drift=0.01), min_names=5)
    dd = mkt["mkt_drawdown_20d_prev"].dropna()
    assert (dd <= 1e-9).all()


def test_attach_preserves_the_row_count():
    p = _panel(n_days=10, n_names=20)
    mkt = build_market_factor(p, min_names=5)
    cands = p[["symbol", "date", "gap", "ret_5d_prev"]].head(37)
    assert len(attach_regime(cands, mkt)) == 37


def test_missing_required_columns_raise_rather_than_return_nonsense():
    with pytest.raises(KeyError):
        build_market_factor(pd.DataFrame({"date": [pd.Timestamp("2024-01-02")]}))


def test_every_declared_feature_is_produced():
    p = _panel(n_days=30, n_names=30)
    mkt = build_market_factor(p, min_names=5)
    out = attach_regime(p[["symbol", "date", "gap", "ret_5d_prev"]], mkt)
    assert set(REGIME_FEATURES) <= set(out.columns)
