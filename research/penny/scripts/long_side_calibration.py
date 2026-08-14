"""long_side_calibration.py — the four supplied hypotheses, calibrated LONG.

What this adds over `hypothesis_grid.py`
----------------------------------------
The 441-cell grid tested the supplied hypotheses as *mean* effects: "does
condition X imply a higher average return over the next H sessions". That is not
quite what was asked. The hypotheses were phrased as **thresholds** --

    "stock down by A -> raise by B in the next C days"

-- and "rise by B" is a question about whether the price *touches* +B%, not about
where it happens to close on day C. Those are different questions with different
answers, and the threshold version is the tradable one: a limit order at +B% and
a time stop at C days captures a touch, whereas a mean-reversion cell captures
only the terminal close.

So every family here is evaluated as a **long triple barrier**:

    enter at the close of day t
    exit at +B% if the high touches it on any of days t+1 .. t+C   (limit fill)
    otherwise exit at the close of day t+C                          (time stop)
    optionally exit at -S% if the low touches it first              (stop loss)

and the headline number is a net return per trade, not a hit rate, because a 70%
hit rate at +10% with a 30% tail of -40% is a losing strategy.

Ordering within a day is unknowable from daily bars. When both the +B% high and
the -S% low occur on the same session, the **stop is assumed to fill first**.
That is the conservative assignment and it matters: the optimistic one
manufactures edge out of ambiguity.

Families
--------
The four supplied, each with its parameters swept rather than chosen:

    H1  down by A over C_look days      -> long, target B, horizon C
    H2  up by A over C_look days        -> long, target B, horizon C   (momentum)
    H3  reverse split                   -> long, target B, horizon C
    H4  premarket / overnight gap +/-A  -> long, target B, horizon C

plus enrichments that the grid never tested, all long:

    H5  capitulation      -- down A% AND volume shock AND at the 20-day low
    H6  confirmation      -- down A% over C days THEN one green day (no knife)
    H7  quiet decline     -- down A% on BELOW-average volume (no panic)
    H8  post-split bounce -- a split that already fell X% on the day
    H9  dip in an uptrend -- up over 20d, down over 3d
    H10 overnight only    -- buy the close, sell the next open
    H11 price band        -- does any of the above depend on the price level

Discipline
----------
Same as the grid, and for the same reasons: disjoint in-sample / out-of-sample
windows, Benjamini-Hochberg over **every** cell tested rather than the
survivors, measured costs by liquidity band charged before judging, and a
day-clustered bootstrap because same-day trades are not independent.

An unconditional control is included in every table. Round 5 recorded the reason:
a +1.08% result at p<0.0001 over 650,865 rows turned out to be equity drift with
no selection at all. Without a control on the same page, drift reads as alpha.

    python research/penny/scripts/long_side_calibration.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.hypothesis_grid import (  # noqa: E402
    OOS_START, build_panel, cost_for,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("long")

MIN_TRADES = 200
MIN_DAYS = 60
FDR_Q = 0.05
ADV_FLOOR = 2.5e5          # below this the measured round trip exceeds any edge
HORIZONS = [1, 2, 3, 5, 10]
TARGETS = [0.05, 0.10, 0.20, 0.50]      # "raise by B"
STOPS = [None, 0.10, 0.20]


# --------------------------------------------------------------------------
# barrier machinery
# --------------------------------------------------------------------------
def attach_forward_paths(u: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Running max-high / min-low over the next h sessions, per symbol.

    Built from shifts rather than a reversed rolling window so that the symbol
    boundary is respected -- a rolling window would happily read the next
    ticker's prices across the join.
    """
    g = u.groupby("symbol", sort=False)
    hi_shifts, lo_shifts = [], []
    for k in range(1, max(horizons) + 1):
        hi_shifts.append(g["high"].shift(-k))
        lo_shifts.append(g["low"].shift(-k))
        if k in horizons:
            u[f"fwd_maxhigh_{k}"] = pd.concat(hi_shifts, axis=1).max(axis=1)
            u[f"fwd_minlow_{k}"] = pd.concat(lo_shifts, axis=1).min(axis=1)
    return u


def barrier_return(sub: pd.DataFrame, h: int, target: float,
                   stop: Optional[float]) -> pd.Series:
    """Net return to a long triple barrier entered at the close of day t.

    Ties -- both barriers touched within the window -- resolve to the stop.
    Daily bars cannot order intraday events, and assuming the favourable
    ordering would invent edge from missing information.
    """
    entry = sub["close"]
    hit_target = sub[f"fwd_maxhigh_{h}"] >= entry * (1.0 + target)
    hit_stop = (sub[f"fwd_minlow_{h}"] <= entry * (1.0 - stop)) if stop else pd.Series(
        False, index=sub.index)

    gross = sub[f"fwd_close_{h}"].copy()            # time stop, the default
    gross = gross.where(~hit_target.fillna(False), target)
    if stop is not None:
        gross = gross.where(~hit_stop.fillna(False), -stop)   # stop wins ties
    return gross - sub["cost"]


@dataclass
class Cell:
    family: str
    label: str
    mask: np.ndarray
    horizon: int
    target: float
    stop: Optional[float]


def evaluate(u: pd.DataFrame, cell: Cell) -> Optional[Dict]:
    m = cell.mask & u[f"fwd_close_{cell.horizon}"].notna().to_numpy()
    if m.sum() < MIN_TRADES:
        return None
    sub = u[m]
    if sub["date"].nunique() < MIN_DAYS:
        return None
    r = barrier_return(sub, cell.horizon, cell.target, cell.stop)
    r = r.dropna()
    if len(r) < MIN_TRADES:
        return None
    d = sub.loc[r.index, "date"]
    est = pstats.cluster_bootstrap_mean(r, d, n_boot=1_500)
    return {"family": cell.family, "label": cell.label, "h": cell.horizon,
            "target": cell.target, "stop": cell.stop, "n": int(len(r)),
            "n_days": int(est.n_days), "mean": float(est.mean),
            "lo": float(est.lo), "hi": float(est.hi),
            "p": float(est.p_value_greater), "median": float(r.median()),
            "hit": float((r > 0).mean()),
            # the target is reached net of cost, so compare against the
            # per-row net figure -- comparing to the gross target reports 0%
            "touch": float((r >= cell.target - sub.loc[r.index, "cost"] - 1e-9).mean())}


# --------------------------------------------------------------------------
# the families
# --------------------------------------------------------------------------
def build_cells(u: pd.DataFrame) -> List[Cell]:
    liquid = (u["adv20"] >= ADV_FLOOR).to_numpy()
    clean = liquid & (~u["any_split"]).to_numpy()
    split = liquid & u["any_split"].to_numpy()
    cells: List[Cell] = []

    def add(family: str, label: str, mask: np.ndarray,
            horizons=None, targets=None, stops=None) -> None:
        for h in (horizons or HORIZONS):
            for b in (targets or TARGETS):
                for s in (stops or STOPS):
                    cells.append(Cell(family, label, mask, h, b, s))

    # H1 -- "stock down by A -> raise by B in the next C days"
    for look in (3, 5, 10, 20):
        for a in (0.15, 0.25, 0.40, 0.60):
            mask = clean & (u[f"ret_{look}d"] <= -a).fillna(False).to_numpy()
            add("H1_reversion", f"down>={100*a:.0f}% over {look}d", mask)

    # H2 -- "up by A -> continues to raise by B"
    for look in (3, 5, 10, 20):
        for a in (0.20, 0.50, 1.00):
            mask = clean & (u[f"ret_{look}d"] >= a).fillna(False).to_numpy()
            add("H2_momentum", f"up>={100*a:.0f}% over {look}d", mask)

    # H3 -- "reverse stock split -> raise by A"
    add("H3_split", "reverse split, long", split)

    # H4 -- "large gap up/down -> raise by B"
    for a in (0.05, 0.10, 0.20):
        add("H4_gap_up", f"overnight gap >= +{100*a:.0f}%",
            clean & (u["overnight"] >= a).fillna(False).to_numpy())
        add("H4_gap_down", f"overnight gap <= -{100*a:.0f}%",
            clean & (u["overnight"] <= -a).fillna(False).to_numpy())

    # H5 -- capitulation: the decline, the volume, and the low together
    for a in (0.25, 0.40):
        for v in (3.0, 5.0):
            mask = (clean & (u["ret_5d"] <= -a).fillna(False).to_numpy()
                    & (u["vol_shock"] >= v).fillna(False).to_numpy()
                    & u["at_20d_low"].to_numpy())
            add("H5_capitulation", f"down>={100*a:.0f}%/5d, vol>={v:.0f}x, at 20d low", mask)

    # H6 -- confirmation: wait for one green day rather than catching the knife
    for look in (5, 10, 20):
        for a in (0.25, 0.40):
            mask = (clean
                    & (u[f"ret_{look}d"] <= -a).fillna(False).to_numpy()
                    & (u["c2c"] > 0).fillna(False).to_numpy())
            add("H6_confirmation", f"down>={100*a:.0f}%/{look}d then a green day", mask)

    # H7 -- quiet decline: a drift down on no volume, not a panic
    for a in (0.20, 0.35):
        mask = (clean & (u["ret_10d"] <= -a).fillna(False).to_numpy()
                & (u["vol_shock"] <= 0.8).fillna(False).to_numpy())
        add("H7_quiet", f"down>={100*a:.0f}%/10d on below-avg volume", mask)

    # H8 -- post-split bounce, the long reading of H3 conditioned on the drop
    for x in (0.10, 0.25):
        mask = split & (u["o2c"] <= -x).fillna(False).to_numpy()
        add("H8_split_bounce", f"split that fell >={100*x:.0f}% on the day", mask)

    # H9 -- pullback inside an uptrend
    for a in (0.20, 0.50):
        mask = (clean & (u["ret_20d"] >= a).fillna(False).to_numpy()
                & (u["ret_3d"] <= -0.05).fillna(False).to_numpy())
        add("H9_dip_in_trend", f"up>={100*a:.0f}%/20d, down>=5%/3d", mask)

    # H11 -- does the best reversion cell depend on the price level
    for lo, hi in ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0)):
        mask = (clean & (u["ret_5d"] <= -0.25).fillna(False).to_numpy()
                & (u["close"] >= lo) & (u["close"] < hi))
        add("H11_price_band", f"down>=25%/5d, price ${lo:g}-${hi:g}", mask)

    # CONTROL -- no selection at all. Mandatory; see the docstring.
    add("CONTROL", "every liquid day, no condition", clean)
    return cells


def overnight_family(u: pd.DataFrame) -> List[Dict]:
    """H10 -- buy the close, sell the next open. No target, no stop."""
    liquid = (u["adv20"] >= ADV_FLOOR).to_numpy()
    clean = liquid & (~u["any_split"]).to_numpy()
    g = u.groupby("symbol", sort=False)
    on = (g["open"].shift(-1) / u["close"] - 1.0) - u["cost"]
    rows: List[Dict] = []
    specs = [("every liquid day", clean),
             ("down>=25% over 5d", clean & (u["ret_5d"] <= -0.25).fillna(False).to_numpy()),
             ("down>=40% over 10d", clean & (u["ret_10d"] <= -0.40).fillna(False).to_numpy()),
             ("up>=50% over 5d", clean & (u["ret_5d"] >= 0.50).fillna(False).to_numpy()),
             ("at the 20d low", clean & u["at_20d_low"].to_numpy()),
             ("vol shock >=5x", clean & (u["vol_shock"] >= 5).fillna(False).to_numpy())]
    for label, mask in specs:
        m = mask & on.notna().to_numpy()
        if m.sum() < MIN_TRADES:
            continue
        r, d = on[m], u.loc[m, "date"]
        est = pstats.cluster_bootstrap_mean(r, d, n_boot=1_500)
        rows.append({"family": "H10_overnight", "label": label, "h": 0,
                     "target": 0.0, "stop": None, "n": int(m.sum()),
                     "n_days": int(est.n_days), "mean": float(est.mean),
                     "lo": float(est.lo), "hi": float(est.hi),
                     "p": float(est.p_value_greater), "median": float(r.median()),
                     "hit": float((r > 0).mean()), "touch": float("nan")})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Long-side calibration of the four hypotheses.")
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "long_side_calibration.json"))
    args = ap.parse_args(argv)

    u = build_panel()
    u = attach_forward_paths(u, HORIZONS)
    logger.info("panel: %s symbol-days | %d sessions | %s tickers",
                f"{len(u):,}", u["date"].nunique(), f"{u['symbol'].nunique():,}")

    cells = build_cells(u)
    logger.info("cells to evaluate: %s", f"{len(cells):,}")

    is_mask = (u["date"] < OOS_START).to_numpy()
    oos_mask = ~is_mask

    rows: List[Dict] = []
    for i, c in enumerate(cells):
        if i and i % 200 == 0:
            logger.info("  ... %d/%d", i, len(cells))
        r = evaluate(u, c)
        if r is None:
            continue
        # The period re-measurement is only informative for a cell that is
        # positive overall; running it on the 90% that are not would triple the
        # runtime to confirm a negative result twice.
        r["is_mean"] = r["oos_mean"] = r["is_lo"] = r["oos_lo"] = None
        r["is_n"] = r["oos_n"] = None
        if r["mean"] <= 0:
            rows.append(r)
            continue
        for tag, per in (("is", is_mask), ("oos", oos_mask)):
            sub = evaluate(u, Cell(c.family, c.label, c.mask & per, c.horizon,
                                   c.target, c.stop))
            r[f"{tag}_mean"] = sub["mean"] if sub else None
            r[f"{tag}_lo"] = sub["lo"] if sub else None
            r[f"{tag}_n"] = sub["n"] if sub else None
        rows.append(r)

    rows.extend(overnight_family(u))
    logger.info("cells with enough trades: %s", f"{len(rows):,}")

    # --- BH over EVERY cell tested, not the survivors ----------------------
    pvals = [r["p"] for r in rows]
    passed = pstats.benjamini_hochberg(pvals, q=FDR_Q)
    for r, ok in zip(rows, passed):
        r["bh_pass"] = bool(ok)

    logger.info("=" * 128)
    logger.info("BEST CELL PER FAMILY (net of measured cost, long, day-clustered CI)")
    logger.info("-" * 128)
    logger.info("%-18s %-42s %3s %5s %5s %7s %20s %7s %6s",
                "family", "condition", "h", "tgt", "stop", "net", "95% CI", "median", "n")
    for fam in sorted({r["family"] for r in rows}):
        fr = [r for r in rows if r["family"] == fam]
        best = max(fr, key=lambda r: r["mean"])
        star = "  <<< CI>0" if best["lo"] > 0 else ""
        logger.info("%-18s %-42s %3d %5s %5s %+6.2f%% [%+7.2f%%, %+7.2f%%] %+6.2f%% %6s%s",
                    fam, best["label"][:42], best["h"],
                    f"{100*best['target']:.0f}%" if best["target"] else "-",
                    f"{100*best['stop']:.0f}%" if best["stop"] else "-",
                    100 * best["mean"], 100 * best["lo"], 100 * best["hi"],
                    100 * best["median"], f"{best['n']:,}", star)

    survivors = [r for r in rows if r["bh_pass"] and r["mean"] > 0]
    logger.info("=" * 128)
    if not survivors:
        logger.info("NO long cell survives BH-FDR at q=%.2f over %d cells.",
                    FDR_Q, len(rows))
    else:
        logger.info("%d cell(s) survive BH-FDR. Confirming on disjoint periods:", len(survivors))
        for s in sorted(survivors, key=lambda r: -r["mean"]):
            ok = (s["is_mean"] is not None and s["oos_mean"] is not None
                  and s["is_mean"] > 0 and s["oos_mean"] > 0)
            logger.info("  %-18s %-40s h=%-2d tgt=%-4s net %+6.2f%%  IS %+6.2f%%  OOS %+6.2f%%  %s",
                        s["family"], s["label"][:40], s["h"],
                        f"{100*s['target']:.0f}%", 100 * s["mean"],
                        100 * (s["is_mean"] or 0), 100 * (s["oos_mean"] or 0),
                        "BOTH PERIODS POSITIVE" if ok else "fails period confirmation")

    ctrl = [r for r in rows if r["family"] == "CONTROL"]
    if ctrl:
        best_c = max(ctrl, key=lambda r: r["mean"])
        logger.info("-" * 128)
        logger.info("CONTROL, no selection: %+.2f%% CI [%+.2f%%, %+.2f%%] over %s trades "
                    "-- anything not clearly above this is drift, not skill.",
                    100 * best_c["mean"], 100 * best_c["lo"], 100 * best_c["hi"],
                    f"{best_c['n']:,}")
    logger.info("=" * 128)

    Path(args.out).write_text(json.dumps(
        {"n_cells": len(rows), "fdr_q": FDR_Q, "adv_floor": ADV_FLOOR,
         "min_trades": MIN_TRADES, "results": rows}, indent=2, default=str),
        encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
