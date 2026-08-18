"""risk_shaped.py — attack sigma instead of N.

Where the breadth argument leaves us
-------------------------------------
The veto's edge is +1.07% per 60 sessions on development against a single-name
sigma near 70%. A portfolio of N names estimates that with SE = sigma/sqrt(N),
so the edge is only visible once N is in the low hundreds -- and the measured
breadth curve agrees exactly: the veto's Sharpe contribution is negative or nil
below 150 slots and turns positive only at 250-400.

Cutting N therefore cannot work by itself. But Sharpe is mu/sigma, and nothing
so far has touched the denominator. Two ways to:

**A. Inverse-volatility weights.** Weight name i by 1/sigma_i rather than 1/N.
For a portfolio of independent names this minimises variance for a given mean
when means are equal -- which is exactly the situation here, since section 78
established the survivors are flat in the score. Realised volatility is already
computed as feature `B7_realised_vol`, so this needs no new data.

**B. A hard volatility ceiling.** Keep only names below a quantile of realised
vol. This shrinks sigma AND the candidate count at once, which is the only lever
that helps both problems in the same direction. It is a *risk* filter, not a
selection on the composite -- section 78 showed selection on the composite does
not work, and this is a different claim that must be tested on its own.

Everything else is held fixed: composite signs frozen at the pre-2020 fit,
random fills among survivors, development only.
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
from research.penny.scripts.run_composite import add_composite  # noqa: E402
from research.penny.scripts.veto_holdout import frame  # noqa: E402
from research.penny.strategy.cross_section import MIN_CELL  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SPEC = json.loads(Path("research/penny/specs/VETO.json").read_text(encoding="utf-8"))
SIGNS = {k: float(v) for k, v in SPEC["composite_signs"].items()}
VETO_FRAC = SPEC["veto_fraction"]
DEV_START = pd.Timestamp("2020-01-01")
SLOTS = [30, 60, 100]


@dataclass
class RiskConfig:
    n_slots: int
    H: int
    vol_cap: float       # keep names below this quantile of realised vol (1.0 = off)
    weight: str          # "equal" | "invvol"
    veto: bool


def run(df: pd.DataFrame, cfg: RiskConfig, seed: int = 0) -> Optional[Dict]:
    rng = np.random.default_rng(seed)
    sess = np.sort(df["date"].unique())
    by_day = dict(tuple(df.groupby("date")))

    open_slots: List[Optional[Tuple[int, str, float, float]]] = [None] * cfg.n_slots
    cash = 1.0
    equity = np.ones(len(sess))
    n_entered = 0
    vols: List[float] = []

    for i, d in enumerate(sess):
        for s in range(cfg.n_slots):
            p = open_slots[s]
            if p and p[0] <= i:
                cash += p[3] * (1.0 + p[2])
                open_slots[s] = None

        free = [s for s in range(cfg.n_slots) if open_slots[s] is None]
        g = by_day.get(d)
        if free and g is not None and len(g) >= MIN_CELL:
            pool = g
            if cfg.vol_cap < 1.0:
                thr = pool["B7_realised_vol"].quantile(cfg.vol_cap)
                pool = pool[pool["B7_realised_vol"] <= thr]
            if cfg.veto and len(pool) >= MIN_CELL:
                keep = max(1, int(np.ceil((1.0 - VETO_FRAC) * len(pool))))
                pool = pool.nlargest(keep, "COMPOSITE")
            held = {p[1] for p in open_slots if p}
            pool = pool[~pool["symbol"].isin(held)]
            pool = pool[pool["B7_realised_vol"] > 0]
            if len(pool):
                pool = pool.sample(min(len(pool), len(free)),
                                   random_state=int(rng.integers(1 << 31)))
                deployed = sum(p[3] for p in open_slots if p)
                eq_now = cash + deployed
                if cfg.weight == "invvol":
                    iv = 1.0 / pool["B7_realised_vol"].to_numpy()
                    # normalise so the AVERAGE weight equals 1/n_slots: the book
                    # keeps the same gross exposure, only its distribution moves
                    w = iv / iv.mean() / cfg.n_slots
                else:
                    w = np.full(len(pool), 1.0 / cfg.n_slots)
                for s, (_, row), wi in zip(free, pool.iterrows(), w):
                    size = min(eq_now * float(wi), cash)
                    if size <= 1e-12:
                        break
                    net = float(row["R"] - 0.5 * row["cost"])
                    cash -= size
                    open_slots[s] = (min(i + cfg.H, len(sess) - 1),
                                     row["symbol"], net, size)
                    n_entered += 1
                    vols.append(float(row["B7_realised_vol"]))
        equity[i] = cash + sum(p[3] for p in open_slots if p)

    if n_entered < 40:
        return None
    eq = pd.Series(equity, index=sess)
    r = eq.pct_change().fillna(0.0)
    years = len(sess) / 252.0
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    dd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return {"n_slots": cfg.n_slots, "H": cfg.H, "vol_cap": cfg.vol_cap,
            "weight": cfg.weight, "veto": cfg.veto, "entered": n_entered,
            "orders_per_session": n_entered / len(sess),
            "ann": 100 * ann, "sharpe": sharpe, "maxdd": 100 * dd,
            "median_vol": float(np.median(vols))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--slots", default="30,60,100")
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "risk_shaped.json"))
    args = ap.parse_args(argv)

    H = 60
    global SLOTS
    SLOTS = [int(x) for x in args.slots.split(",")]
    df = add_composite(frame(H, SPEC["population"]["crash_pct"], args.period),
                       SIGNS, "COMPOSITE")
    if args.period == "dev":
        df = df[df["date"] >= DEV_START]

    rows: List[Dict] = []
    for n in SLOTS:
        for cap in (1.0, 0.50, 0.30):
            for wt in ("equal", "invvol"):
                for veto in (True, False):
                    runs = [run(df, RiskConfig(n, H, cap, wt, veto), seed=s)
                            for s in range(args.seeds)]
                    runs = [x for x in runs if x]
                    if not runs:
                        continue
                    r = pd.DataFrame(runs)
                    rows.append({"n_slots": n, "vol_cap": cap, "weight": wt,
                                 "veto": veto,
                                 "ann": float(r["ann"].mean()),
                                 "sharpe": float(r["sharpe"].mean()),
                                 "sharpe_sd": float(r["sharpe"].std()),
                                 "maxdd": float(r["maxdd"].mean()),
                                 "orders": float(r["orders_per_session"].mean()),
                                 "med_vol": float(r["median_vol"].mean())})
        print(f"  ... {n} slots done", flush=True)

    res = pd.DataFrame(rows)
    print("=" * 118)
    print(f"RISK-SHAPED PORTFOLIO  H=60, {args.period}, {args.seeds} seeds, "
          f"random fills, signs frozen")
    print("vol_cap = keep names below this quantile of realised vol; 1.00 = no cap")
    print("=" * 118)
    print(f"  {'slots':>6}{'volcap':>8}{'weight':>9}{'veto':>6}{'ord/day':>9}"
          f"{'med vol':>9}{'annual':>10}{'Sharpe':>8}{'+/-':>6}{'maxDD':>9}")
    for _, r in res.sort_values(["n_slots", "vol_cap", "weight"]).iterrows():
        star = "  <<<" if r.sharpe > 1.2 and r.maxdd > -35 else ""
        print(f"  {int(r.n_slots):>6}{r.vol_cap:>8.2f}{r['weight']:>9}"
              f"{str(bool(r.veto)):>6}{r.orders:>9.2f}{r.med_vol:>9.3f}"
              f"{r['ann']:>+9.1f}%{r.sharpe:>8.2f}{r.sharpe_sd:>6.2f}{r.maxdd:>8.1f}%{star}")
    print("=" * 118)
    res.to_json(args.out, orient="records", indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
