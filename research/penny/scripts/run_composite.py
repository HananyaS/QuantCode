"""run_composite.py — four ways to get the measured signal past the cost floor.

`run_blocks.py` established that the cross-section carries real structure: rank
ICs to |0.085| at t=-12, with signs matching the lottery/reversal literature.
It also established why that is not yet money: with gross exposure 1 the random
control returns **-1.06%**, which is the round-trip cost, and the best single
feature's gross spread is about +0.95%.

Four levers, each tested separately so its contribution is attributable:

    L1  COMBINE      several weak signals into one composite -- IC adds
    L2  HORIZON      cost is per round trip, so a longer hold amortises it
    L3  LIQUIDITY    the measured schedule falls 150 -> 65 -> 45 bps with ADV,
                     and log-ADV already has a POSITIVE IC, so the restriction
                     costs nothing in signal
    L4  LONG-ONLY    §67 established 84% of this universe cannot be borrowed,
                     so the short leg is fiction anyway; a long-only top-quantile
                     pays one side of the spread instead of two

Signs and weights for the composite are fitted on the FIRST HALF of development
and every number reported is from the SECOND HALF. The holdout stays sealed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.strategy.cross_section import (  # noqa: E402
    MIN_CELL, N_BOOT, score_feature,
)
from research.penny.scripts.run_blocks import build_frame  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SPLIT = pd.Timestamp("2020-01-01")     # first half of development ends here

CANDIDATES = ["B7_realised_vol", "B7_prior_spikes", "B8_dist_from_low",
              "B3_dv_shock", "B3_log_adv", "B1_range_expansion",
              "B1_worst_day_share", "B8_pct_of_range", "B2_idio_depth"]


def zrank(g: pd.Series) -> pd.Series:
    r = g.rank()
    m, s = r.mean(), r.std()
    return (r - m) / s if s > 1e-12 else r * 0.0


def add_composite(df: pd.DataFrame, signs: Dict[str, float], name: str) -> pd.DataFrame:
    parts = []
    for f, sg in signs.items():
        z = df.groupby("date")[f].transform(zrank) * sg
        parts.append(z.fillna(0.0))
    df[name] = np.sum(parts, axis=0)
    return df


def quantile_long(df: pd.DataFrame, feat: str, q: float) -> Dict:
    """L4: long-only top quantile inside each cell, equal weight, one-sided cost."""
    rows = []
    for d, g in df.groupby("date", sort=False):
        if len(g) < MIN_CELL:
            continue
        k = max(1, int(np.ceil(q * len(g))))
        top = g.nlargest(k, feat)
        rows.append((d, float(top["R"].mean() - 0.5 * top["cost"].mean())))
    if len(rows) < 30:
        return {}
    arr = np.array([r[1] for r in rows])
    rng = np.random.default_rng(0)
    boot = arr[rng.integers(0, len(arr), size=(N_BOOT, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_cells": len(rows), "mean": float(arr.mean()),
            "lo": float(lo), "hi": float(hi)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "composite.json"))
    args = ap.parse_args(argv)
    out: Dict[str, object] = {}

    # ---- L1 + L2: composite, swept over horizon ---------------------------
    print("=" * 116)
    print("L1 COMBINE + L2 HORIZON  (signs fitted on dev<2020, reported on dev>=2020)")
    print("=" * 116)
    print(f"  {'h':>4}{'cells':>7}{'IC(single best)':>17}{'IC(composite)':>15}"
          f"{'LS net':>10}{'95% CI':>19}{'cost floor':>12}")
    best_signs: Dict[str, float] = {}
    for h in (5, 10, 20, 40, 60):
        df = build_frame(h, 0.15)
        tr, te = df[df["date"] < SPLIT], df[df["date"] >= SPLIT].copy()
        signs = {}
        for f in CANDIDATES:
            r = score_feature(tr, f)
            if r and abs(r.t_stat) > 2.0:
                signs[f] = float(np.sign(r.ic))
        if not signs:
            print(f"  {h:>4}   (no feature clears |t|>2 in the fitting half)")
            continue
        te = add_composite(te, signs, "COMPOSITE")
        rc = score_feature(te, "COMPOSITE")
        singles = [score_feature(te, f) for f in signs]
        best = max((s for s in singles if s), key=lambda s: abs(s.ic), default=None)
        ctrl = score_feature(te, "B0_random")
        if rc is None:
            continue
        best_signs = signs
        print(f"  {h:>4}{rc.n_cells:>7,}{(best.ic if best else 0):>+17.4f}{rc.ic:>+15.4f}"
              f"{100*rc.ls_net:>+9.2f}%  [{100*rc.ls_lo:>+6.2f}%,{100*rc.ls_hi:>+6.2f}%]"
              f"{100*(ctrl.ls_net if ctrl else 0):>+11.2f}%")
        out[f"h{h}"] = {"ic": rc.ic, "ls_net": rc.ls_net, "lo": rc.ls_lo,
                        "hi": rc.ls_hi, "signs": signs,
                        "cost_floor": ctrl.ls_net if ctrl else None}

    print(f"\n  composite signs fitted on dev<2020: "
          f"{ {k: int(v) for k, v in best_signs.items()} }")

    # ---- L3 + L4 at the best horizon --------------------------------------
    print("\n" + "=" * 116)
    print("L3 LIQUIDITY + L4 LONG-ONLY   (h=60, composite, dev>=2020)")
    print("=" * 116)
    df = build_frame(60, 0.15)
    te = df[df["date"] >= SPLIT].copy()
    te = add_composite(te, best_signs, "COMPOSITE")
    print(f"  {'ADV floor':>12}{'cells':>7}{'obs':>8}{'IC':>9}{'LS net':>10}"
          f"{'cost floor':>12}   long-only top decile / quintile")
    for floor, lab in ((0, "none"), (1e6, "$1M"), (5e6, "$5M"), (2e7, "$20M")):
        s = te[te["cost"] <= (0.0150 if floor == 0 else
                             0.0090 if floor == 1e6 else
                             0.0065 if floor == 5e6 else 0.0045) + 1e-9]
        if s["date"].nunique() < 40:
            print(f"  {lab:>12}   (too few cells)")
            continue
        r = score_feature(s, "COMPOSITE")
        c = score_feature(s, "B0_random")
        q10 = quantile_long(s, "COMPOSITE", 0.10)
        q20 = quantile_long(s, "COMPOSITE", 0.20)
        if not r:
            continue
        f10 = f"{100*q10['mean']:+.2f}% [{100*q10['lo']:+.2f}%,{100*q10['hi']:+.2f}%]" if q10 else "-"
        f20 = f"{100*q20['mean']:+.2f}%" if q20 else "-"
        print(f"  {lab:>12}{r.n_cells:>7,}{r.n_obs:>8,}{r.ic:>+9.4f}{100*r.ls_net:>+9.2f}%"
              f"{100*(c.ls_net if c else 0):>+11.2f}%   d1 {f10}  q1 {f20}")
        out[f"liq_{lab}"] = {"ic": r.ic, "ls_net": r.ls_net,
                             "long_d1": q10, "long_q1": q20}
    print("=" * 116)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
