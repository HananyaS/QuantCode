"""cross_section.py — the within-cell estimator from FORMULATION.md.

Implements equations (2)-(7). Where this file and FORMULATION.md disagree, this
file is wrong.

The point of the design
-----------------------
Rounds 1-10 estimated the unconditional mean of a screened set. §76 showed that
object is dominated by a day effect: the sector arm put 286 trades on 48 days
and the paired control returned -1.06% with a CI 22 points wide. "No effect" and
"an effect too small to see" were indistinguishable.

Demeaning inside a day removes the day effect **algebraically** -- no control
group, no matching, no assumption about the market -- and takes the design from
48 clusters to 1,837.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_CELL = 5                      # a cell smaller than this cannot be ranked
N_BOOT = 3000


# --------------------------------------------------------------------------
# population and outcome
# --------------------------------------------------------------------------
def candidate_mask(u: pd.DataFrame, drop_pct: float = 0.15,
                   price_lo: float = 0.50, price_hi: float = 10.0,
                   adv_floor: float = 2.5e5, window: int = 5) -> np.ndarray:
    """The population fixed in Amendment 1. Unadjusted prices only (§72)."""
    return (
        u["in_universe"].to_numpy()
        & (u["adv20"] >= adv_floor).to_numpy()
        & (~u["any_split"]).to_numpy()
        & (~u["is_suspect_fwd_split"]).to_numpy()
        & (u[f"ret_{window}d_unadj"] <= -drop_pct).fillna(False).to_numpy()
        & (u[f"span_{window}d"] <= 1.6 * window + 10).fillna(False).to_numpy()
        & (u["close_unadj"] >= price_lo).to_numpy()
        & (u["close_unadj"] < price_hi).to_numpy()
    )


def _paths(u: pd.DataFrame, pos: np.ndarray, k: int, back: bool = False) -> Dict[str, np.ndarray]:
    """Bars k ahead (or behind) each position, respecting symbol boundaries."""
    codes = pd.factorize(u["symbol"], sort=False)[0]
    n = len(u)
    cols = ("open_unadj", "high_unadj", "low_unadj", "close_unadj", "volume")
    arrs = {c: u[c].to_numpy() for c in cols}
    sp = u["any_split"].to_numpy(bool) | u["is_suspect_fwd_split"].to_numpy(bool)
    m = len(pos)
    out = {c: np.full((m, k), np.nan) for c in cols}
    out["valid"] = np.zeros((m, k), dtype=bool)
    out["split"] = np.zeros((m, k), dtype=bool)
    for j in range(k):
        step = -(j + 1) if back else (j + 1)
        nxt = pos + step
        ok = (nxt >= 0) & (nxt < n)
        safe = np.where(ok, nxt, 0)
        ok = ok & (codes[safe] == codes[pos])
        out["valid"][:, j] = ok
        for c in cols:
            out[c][ok, j] = arrs[c][safe[ok]]
        out["split"][ok, j] = sp[safe[ok]]
    return out


def forward_return(u: pd.DataFrame, pos: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Plain h-session return from the NEXT open, net of cost.

    No barriers: exit engineering is a separate question and mixing it in
    confounds signal quality with it (Amendment 1). A truncated window is a
    delisting and exits at the last print rather than being dropped (§65).
    """
    p = _paths(u, pos, h + 1)
    entry = p["open_unadj"][:, 0]
    cl, val = p["close_unadj"][:, 1:], p["valid"][:, 1:]
    lastv = np.where(val.any(axis=1), val.shape[1] - 1 - val[:, ::-1].argmax(axis=1), -1)
    rows = np.arange(len(pos))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cl[rows, np.maximum(lastv, 0)] / entry - 1.0 - u["cost"].to_numpy()[pos]
    ok = (np.isfinite(entry) & (entry > 0) & (lastv >= 0)
          & ~p["split"].any(axis=1) & np.isfinite(r))
    return r, ok


# --------------------------------------------------------------------------
# equations (2)-(7)
# --------------------------------------------------------------------------
def demean_within_cell(df: pd.DataFrame, value: str = "R",
                       cell: str = "date") -> pd.Series:
    """Equation (2). Removes mu_t algebraically."""
    return df[value] - df.groupby(cell)[value].transform("mean")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class ICResult:
    feature: str
    n_cells: int
    n_obs: int
    ic: float
    lo: float
    hi: float
    t_stat: float
    ls_net: float
    ls_lo: float
    ls_hi: float

    @property
    def significant(self) -> bool:
        return self.lo > 0 or self.hi < 0


def score_feature(df: pd.DataFrame, feature: str, cell: str = "date",
                  min_cell: int = MIN_CELL, n_boot: int = N_BOOT,
                  seed: int = 0) -> Optional[ICResult]:
    """Equations (3)-(7): rank IC and the cell-neutral net return.

    The bootstrap resamples **cells**, because observations inside one day share
    everything the demeaning did not remove.
    """
    d = df[[cell, feature, "R", "Rt", "cost"]].dropna()
    sizes = d.groupby(cell)[feature].transform("size")
    d = d[sizes >= min_cell]
    if d[cell].nunique() < 30:
        return None

    per_cell: List[Tuple[float, float]] = []
    keys: List = []
    for k, g in d.groupby(cell, sort=False):
        ic = _spearman(g[feature].to_numpy(), g["Rt"].to_numpy())
        # equation (5): rank weights, dollar-neutral, gross exposure 1
        rk = g[feature].rank().to_numpy()
        w = rk - rk.mean()
        denom = np.abs(w).sum()
        if denom < 1e-12 or not np.isfinite(ic):
            continue
        w = w / denom
        ls = float(np.dot(w, g["R"].to_numpy())
                   - np.dot(np.abs(w), g["cost"].to_numpy()))   # equation (7)
        per_cell.append((ic, ls))
        keys.append(k)
    if len(per_cell) < 30:
        return None

    arr = np.array(per_cell)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)
    ic_mean, ls_mean = arr.mean(axis=0)
    ic_lo, ic_hi = np.percentile(boot[:, 0], [2.5, 97.5])
    ls_lo, ls_hi = np.percentile(boot[:, 1], [2.5, 97.5])
    sd = arr[:, 0].std(ddof=1)
    t = float(ic_mean / (sd / np.sqrt(len(arr)))) if sd > 1e-12 else 0.0
    return ICResult(feature, len(arr), int(len(d)), float(ic_mean),
                    float(ic_lo), float(ic_hi), t,
                    float(ls_mean), float(ls_lo), float(ls_hi))


# --------------------------------------------------------------------------
# improvement blocks (FORMULATION.md §3)
# --------------------------------------------------------------------------
def build_features(u: pd.DataFrame, pos: np.ndarray, window: int = 5,
                   seed: int = 0) -> pd.DataFrame:
    """Every feature is computed from bars at or before the signal day.

    B0 is the calibration block: a pure random feature must score zero, and the
    crash depth itself is included because if *it* scores, the estimator is
    picking up something mechanical rather than a signal.
    """
    rng = np.random.default_rng(seed)
    back = _paths(u, pos, 21, back=True)          # bars t-1 .. t-21
    o = u["open_unadj"].to_numpy()[pos]
    h = u["high_unadj"].to_numpy()[pos]
    lo = u["low_unadj"].to_numpy()[pos]
    c = u["close_unadj"].to_numpy()[pos]
    v = u["volume"].to_numpy()[pos]
    adv = u["adv20"].to_numpy()[pos]
    ret5 = u[f"ret_{window}d_unadj"].to_numpy()[pos]

    # --- daily closes over the crash window, oldest last -------------------
    bc = back["close_unadj"][:, :window]           # t-1 .. t-window
    bo = back["open_unadj"][:, :window]
    bh = back["high_unadj"][:, :window]
    bl = back["low_unadj"][:, :window]
    closes = np.column_stack([c, bc])              # t, t-1, ..., t-window
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = closes[:, :-1] / closes[:, 1:] - 1.0            # window daily returns
        worst = np.nanmin(daily, axis=1)
        # B1: how much of the fall happened in its single worst session
        worst_share = np.clip(worst / np.where(ret5 < -1e-9, ret5, np.nan), 0, 1)
        # B1: overnight vs intraday split of the decline
        opens = np.column_stack([o, bo])
        overnight = opens[:, :-1] / closes[:, 1:] - 1.0
        on_sum = np.nansum(overnight, axis=1)
        intraday = closes[:, :-1] / opens[:, :-1] - 1.0
        id_sum = np.nansum(intraday, axis=1)
        on_share = on_sum / np.where(np.abs(on_sum + id_sum) > 1e-9, on_sum + id_sum, np.nan)
        # B1: range expansion during the crash vs the prior month
        rng_w = np.nanmean(np.column_stack([h, bh])[:, :-1]
                           / np.column_stack([lo, bl])[:, :-1] - 1.0, axis=1)
        rng_prior = np.nanmean(back["high_unadj"][:, window:21]
                               / back["low_unadj"][:, window:21] - 1.0, axis=1)
        rng_exp = rng_w / np.where(rng_prior > 1e-9, rng_prior, np.nan)

        # --- B3 liquidity / attention -------------------------------------
        dv = c * v
        dv_shock = dv / np.where(adv > 0, adv, np.nan)
        log_adv = np.log10(np.where(adv > 0, adv, np.nan))

        # --- B7 prior behaviour -------------------------------------------
        pc = back["close_unadj"][:, :21]
        pdaily = pc[:, :-1] / pc[:, 1:] - 1.0
        rvol = np.nanstd(pdaily, axis=1)
        prior_spikes = np.nansum(np.abs(pdaily) > 0.50, axis=1)

        # --- B8 reversal state --------------------------------------------
        low20 = np.nanmin(back["low_unadj"][:, :21], axis=1)
        low20 = np.fmin(low20, lo)
        dist_low = c / np.where(low20 > 0, low20, np.nan) - 1.0
        high20 = np.nanmax(back["high_unadj"][:, :21], axis=1)
        pct_range = (c - low20) / np.where(high20 - low20 > 1e-9, high20 - low20, np.nan)
        down = (daily < 0)
        consec = np.zeros(len(pos))
        run = np.ones(len(pos), dtype=bool)
        for j in range(daily.shape[1]):
            run = run & down[:, j]
            consec += run.astype(float)

    f = pd.DataFrame({
        # B0 -- calibration
        "B0_random": rng.standard_normal(len(pos)),
        "B0_crash_depth": -ret5,
        # B1 -- crash shape
        "B1_worst_day_share": worst_share,
        "B1_overnight_share": on_share,
        "B1_range_expansion": rng_exp,
        # B3 -- liquidity / attention
        "B3_log_adv": log_adv,
        "B3_dv_shock": dv_shock,
        # B7 -- prior behaviour
        "B7_realised_vol": rvol,
        "B7_prior_spikes": prior_spikes,
        # B8 -- reversal state
        "B8_dist_from_low": dist_low,
        "B8_pct_of_range": pct_range,
        "B8_consec_down": consec,
    })
    # B2 -- idiosyncratic depth relative to the sector (Amendment 1)
    if "sec_med" in u.columns:
        f["B2_idio_depth"] = -(ret5 - u["sec_med"].to_numpy()[pos])
        f["B2_sector_move"] = u["sec_med"].to_numpy()[pos]
    return f
