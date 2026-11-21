"""causal.py -- R15's frame and holding paths using only what is known when (Round 17).

Why this exists
---------------
R15's frame (`veto_holdout.frame`) keeps a candidate only if its next 61 bars
contain no flagged split and a finite forward return (`cross_section.py:89-90`).
That reads the future, and it does so *before* the day's ranking, so it changes
which names are selected, not only which trades are counted. Its equity curve
is realised-only (`book.py:329`), and its dev windows run into the sealed
period. PREREGISTRATION_R17 fixes the replacement; this module is §2 of it.

Three guarantees, each enforced rather than promised
----------------------------------------------------
1. **Nothing after the cutoff is read.** `load_panel` truncates the panel at
   the last allowed session before anything else touches it.
2. **No candidate is removed for what happens after its signal date.** The
   frame is `candidate_mask & period` and the >=5-name check counts candidates.
   An entry that cannot fill is a *skipped trade*, decided by the book, not a
   missing row.
3. **Paths are split-adjusted from vendor records.** The adjusted series
   absorbs most splits; Alpaca's corporate-action records correct the ones it
   missed. A panel flag with no vendor record is a genuine price move (84% of
   in-window flags are, §2.3 of the pre-registration).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.penny.etl import bars
from research.penny.strategy.cross_section import MIN_CELL, build_features, candidate_mask

logger = logging.getLogger(__name__)

LIMIT_DELTA = 0.10          # opening-auction buy limit = prior close x (1 + delta)
LATE = 20                   # sessions a time exit may slip past H on a halt
MATCH_TOL = np.log(1.25)    # a vendor split "not absorbed" if the adjusted jump is within 25%
SEC_FEE = 2.78e-5           # per $ of sell notional
TAF_PER_SHARE = 0.000195    # sells, capped per trade
TAF_CAP = 9.79
CAT_PER_SHARE = 0.000003    # both sides
SPLIT_TYPES = ("reverse_splits", "forward_splits", "unit_splits")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_panel(cutoff: pd.Timestamp, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The strategy panel, sorted by symbol and date, with no row after `cutoff`."""
    u = pd.read_parquet(bars.DATA_ROOT / "panel_strategy.parquet", columns=columns)
    u = u[u["date"] <= pd.Timestamp(cutoff)]
    u = u.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    assert u["date"].max() <= pd.Timestamp(cutoff)
    return u


def load_vendor_splits(cutoff: pd.Timestamp) -> pd.DataFrame:
    """Alpaca split records with ex-date <= cutoff, as `symbol, ex_date, factor`.

    `factor = new_rate / old_rate` converts a post-split price into entry-share
    units: 1-for-10 reverse (old 10, new 1) -> 0.1; 3-for-1 forward -> 3.
    """
    files = sorted((bars.DATA_ROOT / "corporate_actions").glob("ca_*.parquet"))
    if not files:
        raise FileNotFoundError("no corporate-action records; run scripts/fetch_corporate_actions.py")
    ca = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ca = ca[ca["type"].isin(SPLIT_TYPES) & (ca["ex_date"] <= pd.Timestamp(cutoff))].copy()
    ca["factor"] = ca["new_rate"].astype(float) / ca["old_rate"].astype(float)
    ca = ca[np.isfinite(ca["factor"]) & (ca["factor"] > 0)]
    return ca[["symbol", "ex_date", "factor"]].drop_duplicates().reset_index(drop=True)


def split_corrections(u: pd.DataFrame, splits: pd.DataFrame) -> np.ndarray:
    """Per-row multiplier that undoes a vendor split the adjusted series missed.

    For each record, look at the adjusted overnight jump on ex-date -1, 0 and +1.
    If the best of the three is within 25% of the split's price multiple, the
    adjusted series still carries the split: put `factor` on that row. A split
    whose multiple is itself within 25% of 1 cannot be told from noise and is
    left alone.
    """
    corr = np.ones(len(u))
    if splits.empty:
        return corr
    codes = pd.factorize(u["symbol"], sort=False)[0]
    adj_o, adj_c = u["open"].to_numpy(float), u["close"].to_numpy(float)
    prev_same = np.r_[False, codes[1:] == codes[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = np.where(prev_same, adj_o / np.r_[np.nan, adj_c[:-1]], np.nan)
    row_of = pd.Series(np.arange(len(u)), index=pd.MultiIndex.from_arrays([u["symbol"], u["date"]]))
    for sym, ex, f in splits.itertuples(index=False):
        mult = 1.0 / f
        if abs(np.log(mult)) < MATCH_TOL:
            continue
        r = row_of.get((sym, pd.Timestamp(ex)))
        if r is None:
            continue
        best, best_err = None, MATCH_TOL
        for j in (r - 1, r, r + 1):
            if 0 <= j < len(u) and codes[j] == codes[r] and np.isfinite(jump[j]) and jump[j] > 0:
                err = abs(np.log(jump[j]) - np.log(mult))
                if err < best_err:
                    best, best_err = j, err
        if best is not None:
            corr[best] *= f
    return corr


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------
def causal_frame(u: pd.DataFrame, start: pd.Timestamp, end_excl: pd.Timestamp,
                 drop_pct: float, signs: Dict[str, float]) -> pd.DataFrame:
    """Candidates on [start, end_excl) with R15's features, composite and sector.

    Mirrors `veto_holdout.frame` + `reproduce_round14.build_frame` +
    `r15_concurrency.attach_sector` with the forward filter removed. `u` must
    come from `load_panel` (sorted; truncated) and carry `sector` and `sec_med`.
    """
    from research.penny.scripts.run_composite import add_composite
    per = ((u["date"] >= pd.Timestamp(start)) & (u["date"] < pd.Timestamp(end_excl))).to_numpy()
    pos = np.flatnonzero(candidate_mask(u, drop_pct=drop_pct) & per)
    f = build_features(u, pos).reset_index(drop=True)
    df = pd.concat([pd.DataFrame({
        "symbol": u["symbol"].to_numpy()[pos], "date": u["date"].to_numpy()[pos],
        "cost": u["cost"].to_numpy()[pos], "adv20": u["adv20"].to_numpy()[pos],
        "close_unadj": u["close_unadj"].to_numpy()[pos],
        "sector": u["sector"].to_numpy()[pos], "_pos": pos}), f], axis=1)
    df = df[df.groupby("date")["symbol"].transform("size") >= MIN_CELL].copy()
    df = add_composite(df, signs, "COMPOSITE")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# holding paths
# --------------------------------------------------------------------------
@dataclass
class Paths:
    """One row per candidate: the trade that would follow if it were selected.

    `marks[i, k]` is the gross value (price / entry) at the close of market
    session e+k, frozen at the exit price from the exit session on, NaN where
    the session lies after the cutoff. `exit_off` is the exit session's offset
    from e, or -1 while the trade is still open at the cutoff.
    """

    filled: np.ndarray
    entry_unadj: np.ndarray
    marks: np.ndarray
    exit_off: np.ndarray
    kind: np.ndarray
    exit_value: np.ndarray
    n_corr: np.ndarray
    hold: int

    def summary(self, idx: Optional[np.ndarray] = None) -> Dict[str, float]:
        k = self.kind if idx is None else self.kind[idx]
        out = {"n": int(len(k))}
        for name in ("unfilled", "tp", "sl", "time", "delist", "halt_cap", "open"):
            out[name] = int((k == name).sum())
        return out


def holding_paths(u: pd.DataFrame, pos: np.ndarray, hold: int, target: float,
                  stop: Optional[float], corr: np.ndarray,
                  delta: float = LIMIT_DELTA) -> Paths:
    """Pre-registration §2.2-§2.4 for every candidate row `pos` of `u`.

    `u` must be the truncated, symbol-sorted panel from `load_panel`; its last
    date is the cutoff and its distinct dates are the market calendar.
    """
    assert target > 0
    assert stop is None or 0 < stop < 1
    cal = pd.DatetimeIndex(np.sort(u["date"].unique()))
    cal_idx = cal.get_indexer(u["date"]).astype(np.int64)
    last_cal = len(cal) - 1
    codes = pd.factorize(u["symbol"], sort=False)[0]
    sym_end = pd.Series(np.arange(len(u))).groupby(codes).transform("max").to_numpy()
    ao, ah, al, ac = (u[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    uo, uc = u["open_unadj"].to_numpy(float), u["close_unadj"].to_numpy(float)
    width = hold + LATE + 1
    m = len(pos)
    filled = np.zeros(m, bool)
    entry = np.full(m, np.nan)
    marks = np.full((m, width), np.nan)
    exit_off = np.full(m, -1, np.int64)
    kind = np.full(m, "unfilled", dtype=object)
    exit_value = np.full(m, np.nan)
    n_corr = np.zeros(m, np.int64)
    up, down = 1.0 + target, (None if stop is None else 1.0 - stop)

    for i, p in enumerate(pos):
        e = cal_idx[p] + 1
        if e > last_cal:
            continue                                   # no next session in the data
        hi = min(sym_end[p], p + width)
        rows = np.arange(p + 1, hi + 1)
        if not len(rows):
            continue
        ks = cal_idx[rows] - e
        rows, ks = rows[ks < width], ks[ks < width]
        if not len(rows) or ks[0] != 0:
            continue                                   # no bar on e: the auction order cannot fill
        r0 = rows[0]
        px = uo[r0]
        if not (np.isfinite(px) and px > 0 and np.isfinite(uc[p])
                and px <= uc[p] * (1.0 + delta)):
            continue
        base = ao[r0]
        if not (np.isfinite(base) and base > 0):
            continue
        filled[i], entry[i] = True, px
        cc = np.cumprod(np.r_[1.0, corr[rows[1:]]])
        o, h, l, c = ao[rows] * cc / base, ah[rows] * cc / base, al[rows] * cc / base, ac[rows] * cc / base
        o[0] = 1.0                                     # the fill is the entry, by definition
        kx, vx, kd, jx = -1, np.nan, None, len(rows) - 1
        last_val, last_k = 1.0, 0
        for j, k in enumerate(ks):
            if not np.isfinite(c[j]):
                continue
            marks[i, last_k:k] = last_val              # carry through sessions with no bar
            if down is not None and np.isfinite(l[j]) and l[j] <= down:
                kx, vx, kd = k, min(down, o[j]) if np.isfinite(o[j]) else down, "sl"
            elif np.isfinite(h[j]) and h[j] >= up:
                kx, vx, kd = k, up, "tp"
            elif k >= hold:
                kx, vx, kd = k, c[j], "time"
            if kd is not None:
                jx = j
                break
            last_val, last_k = c[j], k
            marks[i, k] = c[j]
        if kd is None:
            # no exit inside the window: delisted, still open, or halted past LATE
            last_row = rows[-1]
            if sym_end[p] == last_row and cal_idx[last_row] < last_cal:
                kd, kx, vx = "delist", last_k, last_val
            elif cal_idx[last_row] == last_cal or e + hold > last_cal:
                kd = "open"
            else:
                kd, kx, vx = "halt_cap", width - 1, last_val
        if kd == "open":
            stop_at = min(width, last_cal - e + 1)
            marks[i, last_k:stop_at] = last_val
        else:
            marks[i, last_k:kx] = last_val
            marks[i, kx:] = vx
            exit_off[i], exit_value[i] = kx, vx
        # sessions after the cutoff are unknown, whatever happened
        beyond = max(0, last_cal - e + 1)
        if beyond < width:
            marks[i, beyond:] = np.nan
        kind[i] = kd
        n_corr[i] = int(np.sum(corr[rows[1:jx + 1]] != 1.0))
    return Paths(filled, entry, marks, exit_off, kind, exit_value, n_corr, hold)


def fees_frac(notional: float, entry_px: float) -> float:
    """Regulatory fees on one round trip, as a fraction of the entry notional (§2.6)."""
    if not (np.isfinite(entry_px) and entry_px > 0 and notional > 0):
        return 0.0
    shares = notional / entry_px
    return float(SEC_FEE + min(TAF_PER_SHARE * shares, TAF_CAP) / notional
                 + 2.0 * CAT_PER_SHARE * shares / notional)


def frozen_signs() -> Dict[str, float]:
    spec = json.loads(Path("research/penny/specs/VETO.json").read_text(encoding="utf-8"))
    return {k: float(v) for k, v in spec["composite_signs"].items()}
