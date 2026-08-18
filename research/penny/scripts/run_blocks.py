"""run_blocks.py — score every improvement block with the within-cell estimator.

Follows FORMULATION.md: population from Amendment 1, outcome = plain h-session
forward return from the next open net of cost, cells = days, estimator =
equations (2)-(7).

B0 runs first and is a **kill rule**: a pure random feature must score zero. If
it does not, the estimator is broken and nothing else in the output means
anything.

    python research/penny/scripts/run_blocks.py --h 20
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
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.sector_split import (  # noqa: E402
    attach_sector, attach_sector_relative,
)
from research.penny.strategy.cross_section import (  # noqa: E402
    MIN_CELL, build_features, candidate_mask, demean_within_cell,
    forward_return, score_feature,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SEAL = pd.Timestamp("2024-01-01")


def build_frame(h: int, drop_pct: float) -> pd.DataFrame:
    u = pd.read_parquet(bars.DATA_ROOT / "panel_strategy.parquet")
    u = attach_sector(u)
    u = attach_sector_relative(u, 5)
    m = candidate_mask(u, drop_pct=drop_pct) & (u["date"] < SEAL).to_numpy()
    pos = np.flatnonzero(m)
    R, ok = forward_return(u, pos, h)
    pos, R = pos[ok], R[ok]
    f = build_features(u, pos)
    f = f.iloc[np.arange(len(pos))].reset_index(drop=True)
    df = pd.DataFrame({
        "symbol": u["symbol"].to_numpy()[pos],
        "date": u["date"].to_numpy()[pos],
        "sector": u["sector"].to_numpy()[pos],
        "R": R,
        "cost": u["cost"].to_numpy()[pos],
    })
    df = pd.concat([df, f], axis=1)
    # keep only cells that can be ranked, then demean (equation 2)
    df["cell_n"] = df.groupby("date")["R"].transform("size")
    df = df[df["cell_n"] >= MIN_CELL].copy()
    df["Rt"] = demean_within_cell(df, "R", "date")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "blocks_ic.json"))
    args = ap.parse_args(argv)

    df = build_frame(args.h, args.drop)
    feats = [c for c in df.columns if c.startswith("B")]
    print("=" * 118)
    print(f"WITHIN-CELL BLOCK SCORES   h={args.h} sessions   crash>={100*args.drop:.0f}%   "
          f"cells=days   dev only")
    print(f"  candidates {len(df):,} | cells {df['date'].nunique():,} | "
          f"median cell {int(df.groupby('date').size().median())} | "
          f"raw mean R {100*df['R'].mean():+.2f}%  (this is mu_t, and it is NOT the estimand)")
    print("=" * 118)
    print(f"  {'feature':<24}{'cells':>7}{'obs':>8}{'IC':>9}{'95% CI':>20}{'t':>7}"
          f"{'LS net':>10}{'95% CI':>20}")

    rows: List[Dict] = []
    for feat in sorted(feats):
        r = score_feature(df, feat)
        if r is None:
            print(f"  {feat:<24}  (insufficient cells)")
            continue
        mark = ""
        if feat.startswith("B0_random"):
            mark = "   <-- MUST be ~0" if not r.significant else "   <-- ESTIMATOR BROKEN"
        elif r.significant and r.ls_lo > 0:
            mark = "   <<< signal"
        elif r.significant:
            mark = "   <<  IC only"
        print(f"  {feat:<24}{r.n_cells:>7,}{r.n_obs:>8,}{r.ic:>+9.4f}"
              f"  [{r.lo:>+7.4f},{r.hi:>+7.4f}]{r.t_stat:>7.2f}"
              f"{100*r.ls_net:>+9.2f}%  [{100*r.ls_lo:>+6.2f}%,{100*r.ls_hi:>+6.2f}%]{mark}")
        rows.append(r.__dict__)

    ctrl = [r for r in rows if r["feature"] == "B0_random"]
    print("=" * 118)
    if ctrl and (ctrl[0]["lo"] > 0 or ctrl[0]["hi"] < 0):
        print("KILL RULE TRIPPED: the random control is significant. The estimator is "
              "broken; ignore every row above.")
    else:
        print("Control passes: the random feature is indistinguishable from zero.")
    Path(args.out).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
