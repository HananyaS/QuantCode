"""Tests for research/penny/strategy/causal.py and book.py's mark-to-market mode.

What PREREGISTRATION_R17 promises, pinned on synthetic data:

* a trade whose holding window contains a reverse split is KEPT, and the split
  cannot fake the +100% target -- whether the vendor's adjusted series absorbed
  it or only a corporate-action record says it happened;
* the opening-auction buy limit fills only at or under prior close x 1.10, and
  only if the stock trades that session;
* a stop gapped through fills at the open; a bar touching both barriers is a
  stop; a delisting exits at the last print; a halt carries the mark; nothing
  after the cutoff is ever known;
* the frame for date D does not depend on any row after D;
* the MTM book's equity is cash plus size x (mark - cost), unfilled orders take
  no slot, and the slot frees at the entry session + hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.penny.dates import SEAL, assert_dev
from research.penny.strategy.book import BookConfig, run_book
from research.penny.strategy.causal import (
    LIMIT_DELTA, Paths, causal_frame, fees_frac, holding_paths, split_corrections,
)

DAYS = pd.bdate_range("2021-01-04", periods=40)


def _bars(sym: str, closes, days=DAYS, opens=None, highs=None, lows=None,
          adj_factor=None) -> pd.DataFrame:
    """Unadjusted bars; `adj_factor` multiplies them into the vendor-adjusted series."""
    c = np.asarray(closes, float)
    o = c.copy() if opens is None else np.asarray(opens, float)
    h = np.maximum(o, c) if highs is None else np.asarray(highs, float)
    lo = np.minimum(o, c) if lows is None else np.asarray(lows, float)
    f = np.ones(len(c)) if adj_factor is None else np.asarray(adj_factor, float)
    return pd.DataFrame({"symbol": sym, "date": days[:len(c)],
                         "open_unadj": o, "high_unadj": h, "low_unadj": lo, "close_unadj": c,
                         "open": o * f, "high": h * f, "low": lo * f, "close": c * f})


def _panel(*frames: pd.DataFrame) -> pd.DataFrame:
    # a filler symbol trades every day so the market calendar is complete
    filler = _bars("ZZZZ", np.full(len(DAYS), 5.0))
    u = pd.concat([*frames, filler], ignore_index=True)
    return u.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def _row(u: pd.DataFrame, sym: str, day: int) -> int:
    return int(np.flatnonzero((u["symbol"] == sym).to_numpy() & (u["date"] == DAYS[day]).to_numpy())[0])


def _paths(u, sym="AAA", day=0, hold=10, target=1.0, stop=None, corr=None):
    corr = np.ones(len(u)) if corr is None else corr
    return holding_paths(u, np.array([_row(u, sym, day)]), hold, target, stop, corr)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def test_absorbed_reverse_split_is_kept_and_cannot_fake_the_target():
    # 1-for-10 on day 5: unadjusted price x10, vendor-adjusted series flat
    c = np.r_[np.full(5, 1.0), np.full(35, 10.0)]
    f = np.r_[np.full(5, 1.0), np.full(35, 0.1)]
    u = _panel(_bars("AAA", c, adj_factor=f))
    p = _paths(u)
    assert p.filled[0] and p.kind[0] == "time"
    assert p.exit_value[0] == pytest.approx(1.0)
    assert np.nanmax(p.marks[0, :11]) == pytest.approx(1.0)


def test_missed_reverse_split_is_corrected_from_the_vendor_record():
    # the vendor-adjusted series carries the x10 jump (adj factor stays 1)
    c = np.r_[np.full(5, 1.0), np.full(35, 10.0)]
    u = _panel(_bars("AAA", c))
    splits = pd.DataFrame({"symbol": ["AAA"], "ex_date": [DAYS[5]], "factor": [0.1]})
    corr = split_corrections(u, splits)
    assert corr[_row(u, "AAA", 5)] == pytest.approx(0.1)
    assert (corr != 1).sum() == 1
    p = _paths(u, corr=corr)
    assert p.kind[0] == "time" and p.exit_value[0] == pytest.approx(1.0)
    assert p.n_corr[0] == 1


def test_vendor_record_on_a_neighbouring_day_still_matches():
    c = np.r_[np.full(6, 1.0), np.full(34, 10.0)]            # jump lands a day after ex_date
    u = _panel(_bars("AAA", c))
    splits = pd.DataFrame({"symbol": ["AAA"], "ex_date": [DAYS[5]], "factor": [0.1]})
    corr = split_corrections(u, splits)
    assert corr[_row(u, "AAA", 6)] == pytest.approx(0.1)


def test_absorbed_split_gets_no_correction():
    c = np.r_[np.full(5, 1.0), np.full(35, 10.0)]
    f = np.r_[np.full(5, 1.0), np.full(35, 0.1)]
    u = _panel(_bars("AAA", c, adj_factor=f))
    splits = pd.DataFrame({"symbol": ["AAA"], "ex_date": [DAYS[5]], "factor": [0.1]})
    assert (split_corrections(u, splits) == 1).all()


def test_jump_with_no_vendor_record_is_a_genuine_move():
    # a real +150% gap: no record, so it is a target hit, not an artifact
    c = np.r_[np.full(5, 1.0), np.full(35, 2.5)]
    u = _panel(_bars("AAA", c))
    p = _paths(u)
    assert p.kind[0] == "tp" and p.exit_value[0] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------
@pytest.mark.parametrize("open_px, filled", [(1.0 * (1 + LIMIT_DELTA), True),
                                              (1.0 * (1 + LIMIT_DELTA) + 0.01, False),
                                              (0.80, True)])
def test_opening_auction_limit(open_px, filled):
    c = np.full(40, 1.0)
    o = c.copy()
    o[1] = open_px
    u = _panel(_bars("AAA", c, opens=o))
    p = _paths(u)
    assert bool(p.filled[0]) is filled
    if filled:
        assert p.entry_unadj[0] == pytest.approx(open_px)
    else:
        assert p.kind[0] == "unfilled" and np.isnan(p.marks[0]).all()


def test_no_bar_on_the_entry_session_is_unfilled():
    df = _bars("AAA", np.full(40, 1.0))
    df = df[df["date"] != DAYS[1]]
    u = _panel(df)
    assert not _paths(u).filled[0]


# --------------------------------------------------------------------------
# exits
# --------------------------------------------------------------------------
def test_stop_gapped_through_fills_at_the_open():
    c = np.full(40, 1.0)
    o, lo = c.copy(), c.copy()
    o[4], lo[4], c[4] = 0.30, 0.25, 0.35
    u = _panel(_bars("AAA", c, opens=o, lows=lo))
    p = _paths(u, stop=0.5)
    assert p.kind[0] == "sl" and p.exit_value[0] == pytest.approx(0.30)
    assert p.exit_off[0] == 3                               # day 4 is offset 3 from entry day 1
    assert np.all(p.marks[0, 3:11] == pytest.approx(0.30))


def test_stop_without_gap_fills_at_the_stop_and_ties_go_to_the_stop():
    c = np.full(40, 1.0)
    h, lo = c.copy(), c.copy()
    h[3], lo[3] = 2.5, 0.4                                  # both barriers on one bar
    u = _panel(_bars("AAA", c, highs=h, lows=lo))
    p = _paths(u, stop=0.5)
    assert p.kind[0] == "sl" and p.exit_value[0] == pytest.approx(0.5)


def test_target_fills_at_the_limit_even_through_a_gap():
    c = np.full(40, 1.0)
    o, h = c.copy(), c.copy()
    o[3], h[3], c[3] = 3.0, 3.2, 3.1
    u = _panel(_bars("AAA", c, opens=o, highs=h))
    p = _paths(u)
    assert p.kind[0] == "tp" and p.exit_value[0] == pytest.approx(2.0)


def test_barriers_are_live_on_the_entry_bar():
    c = np.full(40, 1.0)
    h = c.copy()
    h[1] = 2.05
    u = _panel(_bars("AAA", c, highs=h))
    p = _paths(u)
    assert p.kind[0] == "tp" and p.exit_off[0] == 0


def test_time_exit_at_the_close_of_session_e_plus_hold():
    c = np.linspace(1.0, 1.5, 40)
    u = _panel(_bars("AAA", c))
    p = _paths(u, hold=10)
    assert p.kind[0] == "time" and p.exit_off[0] == 10
    assert p.exit_value[0] == pytest.approx(c[11] / c[1])


def test_delisting_exits_at_the_last_print():
    u = _panel(_bars("AAA", np.r_[np.full(6, 1.0), 0.6]))   # last bar day 6
    p = _paths(u, hold=10)
    assert p.kind[0] == "delist" and p.exit_off[0] == 5
    assert p.exit_value[0] == pytest.approx(0.6)
    assert np.all(p.marks[0, 5:11] == pytest.approx(0.6))


def test_halt_carries_the_mark():
    df = _bars("AAA", np.r_[np.full(4, 1.0), 1.2, np.full(35, 1.0)])
    df = df[~df["date"].isin(DAYS[5:8])]                    # halted days 5-7
    u = _panel(df)
    p = _paths(u, hold=10)
    assert np.all(p.marks[0, 3:7] == pytest.approx(1.2))    # day 4 mark carried through day 7


def test_late_time_exit_when_halted_on_the_exit_session():
    df = _bars("AAA", np.full(40, 1.0))
    df = df[df["date"] != DAYS[11]]                         # no bar on e + 10
    u = _panel(df)
    p = _paths(u, hold=10)
    assert p.kind[0] == "time" and p.exit_off[0] == 11


def test_nothing_after_the_cutoff_is_known():
    u = _panel(_bars("AAA", np.full(40, 1.0)))
    u = u[u["date"] <= DAYS[8]].reset_index(drop=True)      # what load_panel does
    p = _paths(u, hold=10)
    assert p.kind[0] == "open" and p.exit_off[0] == -1
    assert np.isfinite(p.marks[0, :8]).all() and np.isnan(p.marks[0, 8:]).all()


# --------------------------------------------------------------------------
# the frame reads nothing after its signal date
# --------------------------------------------------------------------------
def _frame_panel(n_days=70, n_sym=12, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    rows = []
    for s in range(n_sym):
        r = rng.normal(-0.01, 0.06, n_days)
        c = 3.0 * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, 0.02, n_days))
        rows.append(pd.DataFrame({
            "symbol": f"S{s:02d}", "date": days, "close_unadj": c, "open_unadj": o,
            "high_unadj": np.maximum(o, c) * 1.03, "low_unadj": np.minimum(o, c) * 0.97,
            "volume": rng.integers(2e5, 2e6, n_days).astype(float)}))
    u = pd.concat(rows, ignore_index=True)
    for k in ("open", "high", "low", "close"):
        u[k] = u[f"{k}_unadj"]
    g = u.groupby("symbol")
    u["ret_5d_unadj"] = g["close_unadj"].transform(lambda s: s / s.shift(5) - 1)
    u["span_5d"] = g["date"].transform(lambda s: (s - s.shift(5)).dt.days)
    u["adv20"] = g["close_unadj"].transform(lambda s: s.shift(1).rolling(5).mean()) * 1e6
    u["in_universe"] = u["adv20"].notna()
    u["any_split"] = False
    u["is_suspect_fwd_split"] = False
    u["cost"] = 0.01
    u["sector"] = np.where(u["symbol"].str[-1].astype(int) % 2 == 0, "A", "B")
    return u.sort_values(["symbol", "date"]).reset_index(drop=True)


def test_causal_frame_ignores_rows_after_the_signal_date():
    signs = {"B7_realised_vol": -1, "B3_log_adv": 1}
    u = _frame_panel()
    cut = u["date"].unique()[50]
    full = causal_frame(u, u["date"].min(), cut, drop_pct=0.0, signs=signs)
    short_u = u[u["date"] < cut].reset_index(drop=True)
    short = causal_frame(short_u, u["date"].min(), cut, drop_pct=0.0, signs=signs)
    assert len(full) > 20
    cols = ["symbol", "date", "B7_realised_vol", "COMPOSITE"]
    pd.testing.assert_frame_equal(full[cols].reset_index(drop=True),
                                  short[cols].reset_index(drop=True))


# --------------------------------------------------------------------------
# the marked book
# --------------------------------------------------------------------------
def _book_inputs(filled, marks, exit_off, hold=3):
    """Five candidates per day on days 0 and 1; each carries its own path row."""
    days = pd.bdate_range("2021-01-04", periods=12)
    rows = []
    for d in (0, 1):
        for k in range(5):
            rows.append({"symbol": f"S{d}{k}", "date": days[d], "B7_realised_vol": 0.1,
                         "COMPOSITE": 0.0, "sector": "A", "cost": 0.0, "adv20": 1e6,
                         "_path": d * 5 + k})
    df = pd.DataFrame(rows)
    m = len(df)
    width = marks.shape[1]
    paths = Paths(filled=np.asarray(filled, bool), entry_unadj=np.full(m, 1.0), marks=marks,
                  exit_off=np.asarray(exit_off), kind=np.full(m, "time", dtype=object),
                  exit_value=marks[:, -1].copy(), n_corr=np.zeros(m, int), hold=hold)
    assert width >= hold + 1
    return df, days, paths


def test_mtm_equity_is_cash_plus_marked_positions():
    hold = 3
    marks = np.tile(np.array([1.1, 1.2, 1.3, 1.4, 1.4]), (10, 1))
    df, days, paths = _book_inputs(np.ones(10, bool), marks, np.full(10, hold), hold)
    cfg = BookConfig(n_slots=3, hold=hold, vol_cap=1.0, veto_frac=0.0, target=1.0,
                     cost_mode="none", mark="mtm", account_usd=3_000.0)
    r = run_book(df, days, cfg, seed=0, min_entries=1, paths=paths)
    c = fees_frac(1_000.0, 1.0)
    size = 1.0 / 3
    eq = r.equity.to_numpy()
    assert eq[0] == pytest.approx(1.0)                      # decided on day 0, cash reserved
    # day 1: first position marked at 1.1; second order decided, still cash
    assert eq[1] == pytest.approx(1 - size + size * (1.1 - c))
    # day 4: first position released at e + hold = 1 + 3 with mark 1.4
    first = size * (1.4 - c)
    assert r.trades.loc[0, "release_date"] == days[4]
    assert r.trades.loc[0, "net"] == pytest.approx(0.4 - c)
    assert (r.daily.abs() > 0).sum() >= 3


def test_unfilled_order_takes_no_slot_and_spends_the_session():
    hold = 3
    marks = np.tile(np.array([1.0, 1.0, 1.0, 1.0, 1.0]), (10, 1))
    df, days, paths = _book_inputs(np.zeros(10, bool), marks, np.full(10, hold), hold)
    cfg = BookConfig(n_slots=3, hold=hold, vol_cap=1.0, veto_frac=0.0, target=1.0,
                     cost_mode="none", mark="mtm")
    r = run_book(df, days, cfg, seed=0, min_entries=1, paths=paths)
    assert r is None or r.n_entered == 0
    # with every order unfilled the account never leaves cash
    filled = np.zeros(10, bool)
    filled[5:] = True                                       # only day-1 candidates fill
    df, days, paths = _book_inputs(filled, marks, np.full(10, hold), hold)
    r = run_book(df, days, cfg, seed=0, min_entries=1, paths=paths)
    assert r.n_entered == 1 and r.orders_per_session == pytest.approx(2 / len(days))


def test_mtm_refuses_missing_paths():
    df, days, paths = _book_inputs(np.ones(10, bool), np.ones((10, 5)), np.full(10, 3))
    cfg = BookConfig(n_slots=3, hold=3, vol_cap=1.0, veto_frac=0.0, mark="mtm")
    with pytest.raises(ValueError):
        run_book(df, days, cfg, min_entries=1)
    with pytest.raises(ValueError):
        run_book(df, days, BookConfig(n_slots=3, hold=4, vol_cap=1.0, veto_frac=0.0, mark="mtm"),
                 min_entries=1, paths=paths)


def test_fees_are_small_and_capped():
    assert fees_frac(1_000.0, 1.0) == pytest.approx(2.78e-5 + 0.195 / 1000 + 0.006 / 1000)
    big = fees_frac(1e6, 0.5)                               # TAF cap binds at 2M shares
    assert big == pytest.approx(2.78e-5 + 9.79 / 1e6 + 2 * 0.000003 * 2e6 / 1e6)


def test_assert_dev_refuses_sealed_dates():
    assert_dev(pd.Series([pd.Timestamp("2023-12-29")]))
    with pytest.raises(ValueError):
        assert_dev(pd.Series([SEAL]))
