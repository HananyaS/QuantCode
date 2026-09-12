"""Tests for research/penny/strategy/book.py.

The failure this module exists to prevent
------------------------------------------
Three slot simulators disagreed about how much a trade costs. `R` arrives net of
one round trip and the `R_t*` target columns arrive gross, but every simulator
subtracted the same `0.5 * cost` from whichever it was handed -- so the time-stop
arm paid 1.5 round trips and the profit-target arm paid 0.5, a threefold
difference between two arms of one sweep. The sharpest statement of the fix is
`test_flat_cost_on_the_time_stop_path_reproduces_R_exactly`: undoing the embedded
charge and applying one flat round trip must land back on `R` itself.

Two further regressions are pinned here: exits scheduled on the market calendar
rather than on candidate days, and a fill rule that is genuinely independent of
the veto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.penny.strategy.book import (
    BookConfig, BookResult, gross_return, run_book,
)

MIN_CELL = 5


def _calendar(n: int, start: str = "2021-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _frame(calendar, n_per_day: int = 8, seed: int = 0,
           r: float | None = None) -> pd.DataFrame:
    """A candidate frame with the columns `run_book` reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in calendar:
        for k in range(n_per_day):
            cost = 0.0150
            ret = r if r is not None else float(rng.normal(0.02, 0.30))
            rows.append({
                "symbol": f"S{k:02d}",
                "date": d,
                "R": ret,                      # NET of one round trip
                "cost": cost,
                "B7_realised_vol": 0.05 + 0.01 * k,
                "B3_log_adv": 6.0,             # log10 -> $1M ADV
                "COMPOSITE": float(k),
                "sector": f"SEC{k % 2}",       # two sectors, alternating
            })
    return pd.DataFrame(rows)


def _cfg(**kw) -> BookConfig:
    base = dict(n_slots=5, hold=5, vol_cap=1.0, veto_frac=0.0, target=0.0,
                cost_mode="flat", account_usd=50_000.0)
    base.update(kw)
    return BookConfig(**base)


# ---- the cost is charged exactly once ------------------------------------
def test_gross_return_undoes_the_embedded_cost_on_the_time_stop_path():
    df = _frame(_calendar(1), n_per_day=1, r=0.10)
    g = gross_return(df, target=0.0)
    assert g[0] == pytest.approx(0.10 + 0.0150), (
        "R is net of one round trip; gross must add it back, or the trade is "
        "charged twice downstream")


def test_gross_return_leaves_the_target_column_alone():
    df = _frame(_calendar(1), n_per_day=1, r=0.10)
    df["R_t100"] = 1.00                       # already gross
    g = gross_return(df, target=1.00)
    assert g[0] == pytest.approx(1.00), (
        "R_t* columns are gross already; subtracting cost here would undercharge")


def test_a_missing_target_column_raises_rather_than_silently_falling_back():
    df = _frame(_calendar(1), n_per_day=1)
    with pytest.raises(KeyError, match="R_t50"):
        gross_return(df, target=0.50)


def test_flat_cost_on_the_time_stop_path_reproduces_R_exactly():
    """gross = R + cost, then one flat round trip, must land back on R.

    This is the whole bug in one assertion. The old loops produced R - 0.5*cost.
    """
    cal = _calendar(40)
    df = _frame(cal, n_per_day=8, r=0.10)
    res = run_book(df, cal, _cfg(), seed=0)
    assert res is not None
    assert res.trades["net"].to_numpy() == pytest.approx(0.10), (
        "one flat round trip applied to the undone gross must return R itself")
    assert res.trades["gross"].to_numpy() == pytest.approx(0.115)


def test_cost_mode_none_charges_nothing():
    cal = _calendar(40)
    df = _frame(cal, n_per_day=8, r=0.10)
    res = run_book(df, cal, _cfg(cost_mode="none"), seed=0)
    assert res is not None
    assert res.trades["net"].to_numpy() == pytest.approx(0.115), (
        "with cost switched off the net return is the gross return")


# ---- exits run on the market calendar, not on candidate days -------------
def test_days_without_candidates_still_advance_the_clock():
    """A 5-session hold must be 5 sessions even if no candidate prints between.

    The older loops indexed a candidate-day array, so a hold spanned 5 *days
    with candidates* -- which can be weeks of calendar. Returns were right;
    their timing was not.
    """
    cal = _calendar(30)
    sparse = _frame(cal[::6], n_per_day=8, r=0.10)      # candidates every 6th day
    res = run_book(sparse, cal, _cfg(), seed=0, min_entries=1)
    assert res is not None
    assert res.n_sessions == len(cal), (
        "the return series must span the market calendar, not the candidate days")
    first = res.trades.iloc[0]
    held = list(cal).index(first["exit_date"]) - list(cal).index(first["date"])
    assert held == 5, f"hold must be 5 market sessions, got {held}"


def test_the_equity_series_is_date_indexed_and_compounds():
    cal = _calendar(40)
    df = _frame(cal, n_per_day=8, r=0.05)
    res = run_book(df, cal, _cfg(), seed=0)
    assert res is not None
    assert isinstance(res.equity.index, pd.DatetimeIndex), (
        "an integer index forces callers to degrade a clustered bootstrap to IID")
    assert res.daily.index.equals(res.equity.index)
    assert res.equity.iloc[0] == pytest.approx(1.0)
    assert res.equity.iloc[-1] > 1.0, "a book of winners must compound upwards"


# ---- the control is a real control ---------------------------------------
def test_the_fill_rule_is_independent_of_the_veto():
    """`fill="top"` must pick top-composite names even with the veto switched off.

    slots.py's "noveto" arm skipped the veto trim but still filled with
    nlargest(..., COMPOSITE), so it changed two things at once and was never a
    control for selection.
    """
    cal = _calendar(40)
    df = _frame(cal, n_per_day=8, r=0.10)
    top = run_book(df, cal, _cfg(fill="top", veto_frac=0.0), seed=0, min_entries=1)
    rnd = run_book(df, cal, _cfg(fill="random", veto_frac=0.0), seed=0, min_entries=1)
    assert top is not None and rnd is not None
    # a symbol cannot be re-entered while held, so a top fill walks down the
    # ranking (S07, S06, ...) rather than repeating one name -- but it must
    # never reach the bottom of the book, and a random fill must.
    def rank(res):
        return np.mean([int(x[1:]) for x in res.trades["symbol"]])

    assert set(top.trades["symbol"]).issubset({f"S{k:02d}" for k in range(3, 8)}), (
        "a top fill with no veto must still take the highest composites")
    assert rank(top) > rank(rnd) + 1.0, (
        "the fill rule must move the book's average composite on its own, with "
        "the veto switched off")


def test_the_veto_trims_the_pool_without_changing_the_fill():
    cal = _calendar(40)
    df = _frame(cal, n_per_day=8, r=0.10)
    res = run_book(df, cal, _cfg(fill="random", veto_frac=0.25), seed=0, min_entries=1)
    assert res is not None
    # 8 names, drop worst 25% -> keep 6 -> composites 2..7
    assert set(res.trades["symbol"]).issubset({f"S{k:02d}" for k in range(2, 8)}), (
        "the veto must remove the bottom quarter by composite")


# ---- construction --------------------------------------------------------
def test_stagger_caps_entries_at_ceil_slots_over_hold():
    cal = _calendar(90)
    df = _frame(cal, n_per_day=60, r=0.01)
    res = run_book(df, cal, _cfg(n_slots=60, hold=60), seed=0)
    assert res is not None
    per_day = res.trades.groupby("date").size()
    assert per_day.max() == 1, (
        "ceil(60/60) = 1 entry per session -- this is the round-14 orders/day column")


def test_disabling_stagger_fills_the_book_greedily():
    cal = _calendar(60)
    df = _frame(cal, n_per_day=8, r=0.01)
    res = run_book(df, cal, _cfg(n_slots=6, hold=30, stagger=False), seed=0,
                   min_entries=1)
    assert res is not None
    assert res.trades.groupby("date").size().max() > 1, (
        "without the stagger the whole book fills as one synchronised cohort")


def test_the_vol_cap_keeps_only_the_low_volatility_half():
    cal = _calendar(40)
    df = _frame(cal, n_per_day=8, r=0.10)
    res = run_book(df, cal, _cfg(fill="top", vol_cap=0.50), seed=0, min_entries=1)
    assert res is not None
    assert res.trades["realised_vol"].max() <= 0.085 + 1e-9, (
        "with 8 names at vols 0.05..0.12 the median cap admits at most 0.085")


def test_a_symbol_is_never_held_twice_at_once():
    cal = _calendar(60)
    df = _frame(cal, n_per_day=8, r=0.02)
    res = run_book(df, cal, _cfg(n_slots=6, hold=20, fill="top"), seed=0,
                   min_entries=1)
    assert res is not None
    for _, t in res.trades.iterrows():
        overlap = res.trades[(res.trades["symbol"] == t["symbol"])
                             & (res.trades["date"] > t["date"])
                             & (res.trades["date"] < t["exit_date"])]
        assert overlap.empty, f"{t['symbol']} was re-entered while already held"


def test_inverse_vol_actually_tilts_when_only_one_name_is_filled_a_session():
    """The stagger fills one name a session; the tilt must survive that.

    Normalising `1/sigma` over the SAMPLED subset makes `iv / iv.mean()`
    identically 1 whenever that subset holds one name -- so "inverse vol"
    silently becomes equal weight at every N where `ceil(N/H) == 1`, which is
    every retail configuration. The normaliser has to come from the day's
    eligible set instead. `risk_shaped.py:120` still has the broken form.
    """
    cal = _calendar(60)
    df = _frame(cal, n_per_day=8, r=0.01)      # vols 0.05 .. 0.12
    eq = run_book(df, cal, _cfg(n_slots=60, hold=60, weight="equal"),
                  seed=0, min_entries=1)
    iv = run_book(df, cal, _cfg(n_slots=60, hold=60, weight="invvol"),
                  seed=0, min_entries=1)
    assert eq is not None and iv is not None
    assert eq.trades.groupby("date").size().max() == 1, "fixture must stagger"
    assert iv.trades["weight"].std() > 0, (
        "inverse-vol weights must vary across names")
    assert not np.allclose(eq.trades["weight"].to_numpy(),
                           iv.trades["weight"].to_numpy()), (
        "inverse-vol must differ from equal weight even at one fill a session")
    # a low-vol name must get MORE than its equal-weight share
    lo = iv.trades.nsmallest(5, "realised_vol")["weight"].mean()
    hi = iv.trades.nlargest(5, "realised_vol")["weight"].mean()
    assert lo > hi, "the low-volatility names must carry the larger weight"


def test_inverse_vol_weighting_keeps_average_weight_at_one_over_n():
    cal = _calendar(60)
    df = _frame(cal, n_per_day=8, r=0.01)
    res = run_book(df, cal, _cfg(n_slots=6, hold=30, weight="invvol"), seed=0,
                   min_entries=1)
    assert res is not None
    assert res.trades["weight"].mean() == pytest.approx(1 / 6, rel=0.35), (
        "invvol redistributes the book; it must not change its gross exposure")


# ---- concentration control -----------------------------------------------
def test_the_sector_cap_limits_concurrent_holdings_per_sector():
    """Ten slots filled from one theme is one position held ten times.

    The 2026-08-28 live screen fired ten names of which six were China-domiciled
    microcaps unwinding together. This caps CONCURRENT holdings per sector; it
    does not rank, so it cannot smuggle in a selection effect.
    """
    cal = _calendar(60)
    df = _frame(cal, n_per_day=8, r=0.01)      # sectors SEC0 / SEC1
    res = run_book(df, cal, _cfg(n_slots=6, hold=30, stagger=False,
                                 sector_cap=2), seed=0, min_entries=1)
    assert res is not None
    live = {}
    for _, t in res.trades.iterrows():
        held = res.trades[(res.trades["date"] <= t["date"])
                          & (res.trades["exit_date"] > t["date"])]
        for sec, n in held["sector"].value_counts().items():
            live[sec] = max(live.get(sec, 0), n)
    assert live and max(live.values()) <= 2, (
        f"sector cap of 2 was exceeded: {live}")


def test_no_sector_cap_lets_one_sector_fill_the_book():
    cal = _calendar(60)
    df = _frame(cal, n_per_day=8, r=0.01)
    capped = run_book(df, cal, _cfg(n_slots=6, hold=30, stagger=False,
                                    sector_cap=1), seed=0, min_entries=1)
    free = run_book(df, cal, _cfg(n_slots=6, hold=30, stagger=False),
                    seed=0, min_entries=1)
    assert capped is not None and free is not None
    assert capped.n_entered < free.n_entered, (
        "a cap of one per sector must refuse fills the uncapped book takes")


def test_the_sector_cap_refuses_to_run_without_a_sector_column():
    cal = _calendar(40)
    df = _frame(cal, n_per_day=8, r=0.01).drop(columns=["sector"])
    with pytest.raises(KeyError, match="sector"):
        run_book(df, cal, _cfg(sector_cap=2), seed=0)


# ---- degenerate input ----------------------------------------------------
def test_too_few_entries_returns_none_rather_than_a_misleading_result():
    cal = _calendar(10)
    df = _frame(cal[:1], n_per_day=8, r=0.10)
    assert run_book(df, cal, _cfg(), seed=0, min_entries=40) is None


def test_an_empty_frame_returns_none():
    cal = _calendar(30)
    empty = _frame(cal, n_per_day=1).iloc[0:0]
    assert run_book(empty, cal, _cfg(), seed=0) is None


def test_a_day_below_min_cell_is_never_traded():
    cal = _calendar(40)
    thin = _frame(cal, n_per_day=MIN_CELL - 1, r=0.10)
    assert run_book(thin, cal, _cfg(), seed=0, min_entries=1) is None, (
        "a cell too small to rank must not be traded at all")


def test_config_validation_rejects_impossible_settings():
    with pytest.raises(AssertionError, match="n_slots"):
        _cfg(n_slots=0).validate()
    with pytest.raises(AssertionError, match="veto_frac"):
        _cfg(veto_frac=1.0).validate()
    with pytest.raises(AssertionError, match="vol_cap"):
        _cfg(vol_cap=0.0).validate()


def test_result_to_dict_drops_the_series_fields():
    cal = _calendar(40)
    df = _frame(cal, n_per_day=8, r=0.05)
    res = run_book(df, cal, _cfg(), seed=0)
    assert res is not None
    d = res.to_dict()
    assert "daily" not in d and "equity" not in d and "trades" not in d
    assert d["n_entered"] == len(res.trades)
