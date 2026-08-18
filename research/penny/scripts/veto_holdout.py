"""veto_holdout.py — the one read of the sealed holdout.

The composite signs come from FORMULATION.md's fitting half (development before
2020-01-01) and are NOT refitted here. The population, the horizon and the veto
fraction were fixed in specs/VETO.json and committed before this ran.
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.penny.etl import bars                                    # noqa: E402
from research.penny.scripts.sector_split import (                      # noqa: E402
    attach_sector, attach_sector_relative)
from research.penny.scripts.run_composite import add_composite         # noqa: E402
from research.penny.strategy.cross_section import (                    # noqa: E402
    MIN_CELL, build_features, candidate_mask, demean_within_cell, forward_return)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SEAL = pd.Timestamp("2024-01-01")


def frame(h: int, drop: float, period: str) -> pd.DataFrame:
    u = pd.read_parquet(bars.DATA_ROOT / "panel_strategy.parquet")
    u = attach_sector(u); u = attach_sector_relative(u, 5)
    per = (u["date"] >= SEAL) if period == "holdout" else (u["date"] < SEAL)
    m = candidate_mask(u, drop_pct=drop) & per.to_numpy()
    pos = np.flatnonzero(m)
    R, ok = forward_return(u, pos, h)
    pos, R = pos[ok], R[ok]
    f = build_features(u, pos).reset_index(drop=True)
    df = pd.concat([pd.DataFrame({
        "symbol": u["symbol"].to_numpy()[pos], "date": u["date"].to_numpy()[pos],
        "R": R, "cost": u["cost"].to_numpy()[pos]}), f], axis=1)
    df = df[df.groupby("date")["R"].transform("size") >= MIN_CELL].copy()
    df["Rt"] = demean_within_cell(df, "R", "date")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="holdout", choices=("holdout", "dev"))
    args = ap.parse_args(argv)
    spec = json.loads(Path("research/penny/specs/VETO.json").read_text(encoding="utf-8"))
    signs = {k: float(v) for k, v in spec["composite_signs"].items()}
    h = spec["horizon_sessions"]

    if args.period == "holdout":
        print("=" * 100)
        print("!! READING THE SEALED HOLDOUT -- signs fitted on dev<2020, never refitted !!")
        print("=" * 100)
    df = frame(h, spec["population"]["crash_pct"], args.period)
    df = add_composite(df, signs, "COMPOSITE")
    rng = np.random.default_rng(7)
    df["RAND"] = rng.standard_normal(len(df))

    print(f"period {args.period}: cells {df['date'].nunique():,} | obs {len(df):,} | "
          f"span {str(df['date'].min())[:10]} .. {str(df['date'].max())[:10]}")
    print(f"\n  {'keep':>16}{'feature':>11}{'portfolio':>11}{'cell avg':>10}"
          f"{'EXCESS':>10}{'95% CI':>21}")
    out = {}
    for keep, klab in ((0.90, "drop worst 10%"), (0.75, "drop worst 25%")):
        for feat in ("COMPOSITE", "RAND"):
            rows = []
            for d, g in df.groupby("date", sort=False):
                if len(g) < MIN_CELL:
                    continue
                k = max(1, int(np.ceil(keep * len(g))))
                sel = g.nlargest(k, feat)
                rows.append((float(sel["R"].mean() - 0.5 * sel["cost"].mean()),
                             float(g["R"].mean() - 0.5 * g["cost"].mean())))
            a = np.array(rows); d_ = a[:, 0] - a[:, 1]
            idx = rng.integers(0, len(a), size=(5000, len(a)))
            b = d_[idx].mean(axis=1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            flag = ("  <<< HOLDOUT CONFIRMS" if lo > 0 and feat == "COMPOSITE"
                    else ("   (control)" if feat == "RAND" else ""))
            print(f"  {klab:>16}{feat:>11}{100*a[:,0].mean():>+10.2f}%"
                  f"{100*a[:,1].mean():>+9.2f}%{100*d_.mean():>+9.2f}%"
                  f"  [{100*lo:>+6.2f}%,{100*hi:>+6.2f}%]{flag}")
            if feat == "COMPOSITE":
                out[klab] = {"n_cells": len(a), "portfolio": float(a[:, 0].mean()),
                             "cell_avg": float(a[:, 1].mean()), "excess": float(d_.mean()),
                             "lo": float(lo), "hi": float(hi),
                             "pct_cells_positive": float((d_ > 0).mean())}
    print("=" * 100)
    Path(str(bars.DATA_ROOT / f"veto_{args.period}.json")).write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
