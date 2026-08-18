"""breadth_curve.py — at what position count does the veto actually pay?

The tension, stated as arithmetic
---------------------------------
The veto's edge is an expectation over a population: +2.43% per 60 sessions.
A portfolio of N names estimates that expectation with standard error

    SE = sigma / sqrt(N),      sigma ~ 70% over 60 sessions for these names

so N=30 gives SE ~ 13%, five times the signal. `slots.py` confirmed it
empirically: at 20-60 slots the no-veto control matched or beat the veto, and
max drawdowns ran -40% to -63%.

Two corrections this module makes to that sweep
-----------------------------------------------
1. **Fill slots at random among survivors, not best-score-first.** Section 78
   established the signal is monotone but LEFT-tailed: the top decile does not
   beat the cell average. Filling "best first" is therefore a selection the data
   says does not work, and it was costing 5-20 points of annual return against
   random fills in the same slots.
2. **Average over seeds.** A single random fill is one draw from a very wide
   distribution; comparing one veto draw against one no-veto draw measures
   sampling noise. Every cell below is the mean over `--seeds` independent runs,
   with the spread reported.

The output is a breadth curve: veto minus no-veto Sharpe as a function of N.
Where it crosses and stays above zero is the smallest honest portfolio.
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
from research.penny.scripts.run_composite import add_composite  # noqa: E402
from research.penny.scripts.slots import SlotConfig, run_slots  # noqa: E402
from research.penny.scripts.veto_holdout import frame  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SPEC = json.loads(Path("research/penny/specs/VETO.json").read_text(encoding="utf-8"))
SIGNS = {k: float(v) for k, v in SPEC["composite_signs"].items()}
DEV_START = pd.Timestamp("2020-01-01")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "breadth_curve.json"))
    args = ap.parse_args(argv)

    rows: List[Dict] = []
    for H in (60, 90):
        df = add_composite(frame(H, SPEC["population"]["crash_pct"], args.period),
                           SIGNS, "COMPOSITE")
        if args.period == "dev":
            df = df[df["date"] >= DEV_START]
        for n in (30, 60, 100, 150, 250, 400):
            acc: Dict[str, List[Dict]] = {"random": [], "noveto": []}
            for s in range(args.seeds):
                for rule in ("random", "noveto"):
                    r = run_slots(df, SlotConfig(n, H, rule), seed=s)
                    if r:
                        acc[rule].append(r)
            if not acc["random"] or not acc["noveto"]:
                continue
            v = pd.DataFrame(acc["random"])
            b = pd.DataFrame(acc["noveto"])
            rows.append({
                "H": H, "n_slots": n, "weight_pct": 100.0 / n,
                "orders_per_session": float(v["orders_per_session"].mean()),
                "veto_ann": float(v["ann"].mean()), "veto_sharpe": float(v["sharpe"].mean()),
                "veto_sharpe_sd": float(v["sharpe"].std()),
                "veto_dd": float(v["maxdd"].mean()),
                "base_ann": float(b["ann"].mean()), "base_sharpe": float(b["sharpe"].mean()),
                "base_dd": float(b["maxdd"].mean()),
                "d_sharpe": float(v["sharpe"].mean() - b["sharpe"].mean()),
                "d_ann": float(v["ann"].mean() - b["ann"].mean()),
                # paired across seeds: same fills, veto on vs off
                "d_sharpe_se": float((v["sharpe"].reset_index(drop=True)
                                      - b["sharpe"].reset_index(drop=True)).std()
                                     / np.sqrt(len(v))),
            })
        print(f"  ... H={H} done", flush=True)

    res = pd.DataFrame(rows)
    print("=" * 124)
    print(f"BREADTH CURVE ({args.period}, {args.seeds} seeds per cell, random fills "
          f"among survivors, signs frozen)")
    print("=" * 124)
    print(f"  {'H':>3}{'slots':>7}{'wt%':>7}{'ord/day':>9}{'base ann':>10}{'VETO ann':>10}"
          f"{'base Sh':>9}{'VETO Sh':>9}{'d Sharpe':>10}{'+/- se':>8}{'veto DD':>9}")
    for _, r in res.iterrows():
        sig = "  <<<" if r.d_sharpe > 2 * r.d_sharpe_se and r.d_sharpe > 0 else ""
        print(f"  {int(r.H):>3}{int(r.n_slots):>7}{r.weight_pct:>7.2f}"
              f"{r.orders_per_session:>9.2f}{r.base_ann:>+9.1f}%{r.veto_ann:>+9.1f}%"
              f"{r.base_sharpe:>9.2f}{r.veto_sharpe:>9.2f}{r.d_sharpe:>+10.2f}"
              f"{r.d_sharpe_se:>8.2f}{r.veto_dd:>8.1f}%{sig}")
    print("=" * 124)
    res.to_json(args.out, orient="records", indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
