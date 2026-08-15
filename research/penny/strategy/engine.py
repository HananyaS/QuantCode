"""engine.py — one declarative strategy spec in, one honest number out.

Why this exists
---------------
Eight rounds of this project each hand-wrote their own selection, barrier and
portfolio code. That is how §65 shipped a price band on adjusted prices while
every other module used unadjusted ones: there was no single place where the
rules lived, so there was no single place to check them.

Here a strategy is **data** -- a `Spec` -- and there is exactly one
implementation of signal selection, barrier exits and portfolio simulation.
Changing a strategy means changing a dict, not writing code, which is what makes
an automated search over strategies trustworthy rather than a fresh chance to
reintroduce a bug per iteration.

The three rules that killed earlier rounds, enforced here once
--------------------------------------------------------------
1. **Filters touch only unadjusted prices** (`etl/screens.py` raises otherwise).
   §72: an adjusted price band is a survivorship screen built from the future.
2. **Corporate actions are excluded over the lookback AND the holding window**,
   both directions. §57 (reverse splits on the exit day), §72's forward-split
   cousin (a forward split in the lookback fakes the crash).
3. **A truncated forward window is a delisting, not missing data.** §65 TEST 7:
   `notna()` silently drops precisely the worst outcomes for a long.

And two conventions that keep the numbers conservative:

* **Ties go to the stop.** Daily bars cannot order intraday events; assuming the
  favourable order manufactures edge out of missing information.
* **Gaps through the stop fill at the open**, not at the stop level. These names
  gap; a stop that always fills at its level is fiction.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.penny.etl.screens import (
    adjustment_tripwire,
    assert_unadjusted,
    price_band,
)

logger = logging.getLogger(__name__)

# Measured round-trip cost by liquidity band (round 4, §37-38).
COST_BPS = [(0, 1e5, 400.0), (1e5, 2.5e5, 250.0), (2.5e5, 1e6, 150.0),
            (1e6, 5e6, 90.0), (5e6, 2e7, 65.0), (2e7, np.inf, 45.0)]

CONFIRMATIONS = ("none", "green_day", "two_green", "above_prior_high",
                 "up_3pct", "up_5pct", "up_10pct", "green_on_higher_volume",
                 "close_above_open", "inside_up")

STOP_MODES = ("none", "fixed_pct", "signal_low", "swing_low")
TARGET_MODES = ("fixed_pct", "retrace_frac")


# --------------------------------------------------------------------------
# the spec
# --------------------------------------------------------------------------
@dataclass
class Spec:
    """A complete, executable strategy definition."""
    name: str = "unnamed"
    # universe
    price_lo: float = 0.50
    price_hi: float = 10.00
    adv_floor: float = 2.5e5
    # entry signal
    drop_pct: float = 0.50          # fell at least this much ...
    drop_window: int = 5            # ... over this many sessions
    confirm: str = "green_day"
    min_dollar_vol: float = 0.0     # signal-day dollar volume floor (0 = off)
    max_price_ratio: float = 0.0    # 0 = off; else require close/low_window <= x
    # timing
    entry_at: str = "next_open"     # next_open | close
    # exits
    target_mode: str = "retrace_frac"
    target: float = 0.50            # pct if fixed_pct, else fraction of the gap
    stop_mode: str = "signal_low"
    stop: float = 0.15              # used only when stop_mode == fixed_pct
    max_hold: int = 10
    # portfolio
    slots: int = 30

    def validate(self) -> None:
        assert self.confirm in CONFIRMATIONS, f"confirm must be one of {CONFIRMATIONS}"
        assert self.stop_mode in STOP_MODES, f"stop_mode must be one of {STOP_MODES}"
        assert self.target_mode in TARGET_MODES, f"target_mode must be one of {TARGET_MODES}"
        assert self.entry_at in ("next_open", "close")
        assert 0 < self.drop_pct < 1
        assert 1 <= self.drop_window <= 60
        assert 1 <= self.max_hold <= 120
        assert self.slots >= 1
        assert self.price_lo >= 0 and self.price_hi > self.price_lo

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "Spec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "Spec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def cost_for(adv: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=adv.index)
    for lo, hi, bps in COST_BPS:
        out = out.mask((adv >= lo) & (adv < hi), bps / 1e4)
    return out.fillna(400.0 / 1e4)


# --------------------------------------------------------------------------
# forward paths
# --------------------------------------------------------------------------
def forward_paths(u: pd.DataFrame, pos: np.ndarray, hold: int) -> Dict[str, np.ndarray]:
    """Bars 1..hold ahead of each position, respecting symbol boundaries.

    Positional rather than groupby-shift: shifting materialises `hold` full
    columns for six fields across the whole panel, which on 3.4M rows is
    gigabytes for a few thousand candidates.
    """
    codes = pd.factorize(u["symbol"], sort=False)[0]
    n = len(u)
    arrs = {k: u[k].to_numpy() for k in ("high_unadj", "low_unadj", "close_unadj",
                                         "open_unadj")}
    dt = u["date"].to_numpy()
    sp = (u["any_split"].to_numpy(bool) | u["is_suspect_fwd_split"].to_numpy(bool))

    m = len(pos)
    out = {k: np.full((m, hold), np.nan) for k in
           ("high", "low", "close", "open")}
    out["date"] = np.full((m, hold), np.datetime64("NaT"), dtype="datetime64[ns]")
    out["split"] = np.zeros((m, hold), dtype=bool)
    out["valid"] = np.zeros((m, hold), dtype=bool)
    src = {"high": "high_unadj", "low": "low_unadj",
           "close": "close_unadj", "open": "open_unadj"}

    for k in range(1, hold + 1):
        nxt = pos + k
        ok = nxt < n
        safe = np.where(ok, nxt, 0)
        ok = ok & (codes[safe] == codes[pos])
        j = k - 1
        out["valid"][:, j] = ok
        for key, col in src.items():
            out[key][ok, j] = arrs[col][safe[ok]]
        out["date"][ok, j] = dt[safe[ok]]
        out["split"][ok, j] = sp[safe[ok]]
    return out


# --------------------------------------------------------------------------
# signal
# --------------------------------------------------------------------------
def _confirmation_mask(u: pd.DataFrame, kind: str) -> np.ndarray:
    c2c = u["c2c_unadj"].to_numpy()
    if kind == "none":
        return np.ones(len(u), dtype=bool)
    if kind == "green_day":
        return np.nan_to_num(c2c, nan=-1.0) > 0
    if kind == "two_green":
        return (np.nan_to_num(c2c, nan=-1.0) > 0) & (
            np.nan_to_num(u["c2c_prev_unadj"].to_numpy(), nan=-1.0) > 0)
    if kind == "above_prior_high":
        return (u["close_unadj"] > u["prev_high_unadj"]).fillna(False).to_numpy()
    if kind == "up_3pct":
        return np.nan_to_num(c2c, nan=-1.0) >= 0.03
    if kind == "up_5pct":
        return np.nan_to_num(c2c, nan=-1.0) >= 0.05
    if kind == "up_10pct":
        return np.nan_to_num(c2c, nan=-1.0) >= 0.10
    if kind == "green_on_higher_volume":
        return ((np.nan_to_num(c2c, nan=-1.0) > 0)
                & (u["volume"] > u["prev_volume"]).fillna(False).to_numpy())
    if kind == "close_above_open":
        return (u["close_unadj"] > u["open_unadj"]).fillna(False).to_numpy()
    if kind == "inside_up":
        return ((np.nan_to_num(c2c, nan=-1.0) > 0)
                & (u["low_unadj"] >= u["prev_low_unadj"]).fillna(False).to_numpy())
    raise ValueError(f"unknown confirmation {kind!r}")


def signal_mask(u: pd.DataFrame, spec: Spec) -> np.ndarray:
    """Every filter here is checked against `screens.assert_unadjusted`."""
    ret_col = f"ret_{spec.drop_window}d_unadj"
    assert_unadjusted([ret_col, "close_unadj", "adv20"])

    m = (
        (u["adv20"] >= spec.adv_floor).to_numpy()
        & (~u["any_split"]).to_numpy()
        & (~u["is_suspect_fwd_split"]).to_numpy()
        & (u[ret_col] <= -spec.drop_pct).fillna(False).to_numpy()
        & price_band(u, spec.price_lo, spec.price_hi)
        & _confirmation_mask(u, spec.confirm)
    )
    if spec.min_dollar_vol > 0:
        m &= (u["close_unadj"] * u["volume"] >= spec.min_dollar_vol).fillna(False).to_numpy()
    if spec.max_price_ratio > 0:
        m &= (u["close_unadj"] / u[f"low_{spec.drop_window}d_unadj"]
              <= spec.max_price_ratio).fillna(False).to_numpy()
    return m


# --------------------------------------------------------------------------
# barrier engine
# --------------------------------------------------------------------------
def build_trades(u: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    spec.validate()
    sig = signal_mask(u, spec)
    pos = np.flatnonzero(sig)
    if len(pos) == 0:
        return pd.DataFrame(columns=["symbol", "date", "exit_date", "net",
                                     "took_target", "took_stop", "adv20"])

    off = 1 if spec.entry_at == "next_open" else 0
    p = forward_paths(u, pos, spec.max_hold + off)

    if spec.entry_at == "next_open":
        entry = p["open"][:, 0]
        hi, lo_, cl = p["high"][:, 1:], p["low"][:, 1:], p["close"][:, 1:]
        dts, val = p["date"][:, 1:], p["valid"][:, 1:]
    else:
        entry = u["close_unadj"].to_numpy()[pos]
        hi, lo_, cl = p["high"], p["low"], p["close"]
        dts, val = p["date"], p["valid"]

    split_in_window = p["split"].any(axis=1)

    # ---- per-row barriers, both known at signal time ----------------------
    pre = u[f"close_{spec.drop_window}b_unadj"].to_numpy()[pos]
    if spec.target_mode == "fixed_pct":
        tgt = entry * (1.0 + spec.target)
    else:                                   # retrace a fraction of the decline
        tgt = entry + spec.target * (pre - entry)

    if spec.stop_mode == "none":
        stp = np.full(len(pos), -np.inf)
    elif spec.stop_mode == "fixed_pct":
        stp = entry * (1.0 - spec.stop)
    elif spec.stop_mode == "signal_low":
        stp = u["low_unadj"].to_numpy()[pos]
    else:                                   # swing_low over the decline
        stp = u[f"low_{spec.drop_window}d_unadj"].to_numpy()[pos]

    with np.errstate(invalid="ignore"):
        hit_t = val & (hi >= tgt[:, None])
        hit_s = val & (lo_ <= stp[:, None])

    any_t, any_s = hit_t.any(axis=1), hit_s.any(axis=1)
    first_t = np.where(any_t, hit_t.argmax(axis=1), 10**6)
    first_s = np.where(any_s, hit_s.argmax(axis=1), 10**6)

    # Ties go to the STOP -- daily bars cannot order intraday events.
    took_stop = any_s & (first_s <= first_t)
    took_tgt = any_t & ~took_stop

    last_valid = np.where(val.any(axis=1),
                          val.shape[1] - 1 - val[:, ::-1].argmax(axis=1), -1)
    rows = np.arange(len(pos))
    k = np.where(took_tgt, np.minimum(first_t, val.shape[1] - 1),
                 np.where(took_stop, np.minimum(first_s, val.shape[1] - 1),
                          last_valid))

    # A gap THROUGH the stop fills at the open, which is worse than the stop.
    gap_fill = np.where(took_stop, np.minimum(p["open"][rows, np.maximum(k, 0) + off]
                                              if spec.entry_at == "next_open"
                                              else p["open"][rows, np.maximum(k, 0)],
                                              stp), np.nan)
    exit_px = np.where(took_tgt, tgt,
                       np.where(took_stop, gap_fill, cl[rows, np.maximum(k, 0)]))

    with np.errstate(invalid="ignore", divide="ignore"):
        gross = exit_px / entry - 1.0

    keep = (np.isfinite(entry) & (entry > 0) & (last_valid >= 0)
            & ~split_in_window & np.isfinite(gross))
    t = pd.DataFrame({
        "symbol": u["symbol"].to_numpy()[pos][keep],
        "date": u["date"].to_numpy()[pos][keep],
        "exit_date": dts[rows, np.maximum(k, 0)][keep],
        "entry": entry[keep],
        "net": gross[keep] - u["cost"].to_numpy()[pos][keep],
        "took_target": took_tgt[keep],
        "took_stop": took_stop[keep],
        "adv20": u["adv20"].to_numpy()[pos][keep],
    }).sort_values(["date", "symbol"])

    # one open position per symbol at a time (§66: 63% of §65's entries were
    # re-entries a day-clustered bootstrap cannot see)
    busy: Dict[str, np.datetime64] = {}
    idx: List[int] = []
    for i, sym, d, xd in zip(t.index, t["symbol"], t["date"], t["exit_date"]):
        if sym in busy and d <= busy[sym]:
            continue
        busy[sym] = xd
        idx.append(i)
    return t.loc[idx]
