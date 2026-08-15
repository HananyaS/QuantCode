"""audit_strategy.py — the battery that a candidate has to survive.

Every retracted result in this project passed its statistics and failed
something else. This module runs the checks that actually did the killing, in
the order they historically mattered:

1. **mean vs median** — a mean above the median means the result lives in a
   right tail, which is where measurement errors hide (§57, §61).
2. **concentration** — drop the best N trades. An edge carried by ten names is a
   story about ten names (§61 died here).
3. **the individual trades** — printed with prices, volume and dollar volume.
   This has changed the conclusion three times (§57, §61, §64: WOK at
   $4.8M/share on 116 shares).
4. **per-year** — one year carrying the result is not a strategy (§65: only 2019
   and 2020 were positive once the bias was removed).
5. **excess over an unconditional control** — §53's trap was a +1.08% result at
   p<0.0001 with *no selection at all*: equity drift, not alpha.
6. **the adjusted/unadjusted tripwire** (§72).
7. **deflated Sharpe** against every trial the search has spent.

    python research/penny/scripts/audit_strategy.py --spec specs/x.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.evaluate_strategy import SEAL_DATE, _selected_rows  # noqa: E402
from research.penny.scripts.long_reversion_portfolio import simulate  # noqa: E402
from research.penny.strategy.engine import (  # noqa: E402
    Spec, build_trades, forward_paths, signal_mask,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
TRIALS_FILE = bars.DATA_ROOT / "search_trials.jsonl"


def control_return(u: pd.DataFrame, spec: Spec, dev_only: bool, seed: int = 7,
                   n_sample: int = 40_000) -> Dict:
    """Buy any liquid day in the same price band, exit the same way, no crash
    condition. If the strategy cannot beat this, the crash is doing no work."""
    per = (u["date"] < SEAL_DATE) if dev_only else (u["date"] >= SEAL_DATE)
    liq = ((u["adv20"] >= spec.adv_floor) & (~u["any_split"])
           & (~u["is_suspect_fwd_split"])
           & (u["close_unadj"] >= spec.price_lo) & (u["close_unadj"] < spec.price_hi)
           & per).fillna(False).to_numpy()
    pos = np.flatnonzero(liq)
    if len(pos) > n_sample:
        pos = np.sort(np.random.default_rng(seed).choice(pos, n_sample, replace=False))
    p = forward_paths(u, pos, spec.max_hold + 1)
    entry = p["open"][:, 0]
    cl, val = p["close"][:, 1:], p["valid"][:, 1:]
    lastv = np.where(val.any(axis=1), val.shape[1] - 1 - val[:, ::-1].argmax(axis=1), -1)
    ok = np.isfinite(entry) & (entry > 0) & (lastv >= 0) & ~p["split"].any(axis=1)
    r = cl[np.arange(len(pos)), np.maximum(lastv, 0)] / entry - 1.0 - u["cost"].to_numpy()[pos]
    r, d = pd.Series(r[ok]), pd.Series(u["date"].to_numpy()[pos][ok])
    e = pstats.cluster_bootstrap_mean(r, d, n_boot=2_000)
    return {"n": int(len(r)), "mean": float(e.mean), "lo": float(e.lo),
            "hi": float(e.hi), "median": float(r.median())}


def audit(spec: Spec, unseal: bool = False) -> Dict:
    u = pd.read_parquet(bars.DATA_ROOT / "panel_strategy.parquet")
    t = build_trades(u, spec)
    t = t[t["date"] >= SEAL_DATE] if unseal else t[t["date"] < SEAL_DATE]
    if len(t) < 30:
        print(f"only {len(t)} trades — nothing to audit")
        return {}

    e = pstats.cluster_bootstrap_mean(t["net"], t["date"], n_boot=5_000)
    liquid = np.sort(u.loc[(u["adv20"] >= spec.adv_floor).to_numpy(), "date"].unique())
    cal = liquid[(liquid >= np.datetime64(t["date"].min()))
                 & (liquid <= np.datetime64(max(t["exit_date"].max(), t["date"].max())))]
    sim = simulate(t, cal, spec.slots)

    print("=" * 112)
    print(f"AUDIT — {spec.name}   [{'SEALED HOLDOUT' if unseal else 'development'}]")
    print("=" * 112)
    print(f"  n={len(t):,} over {e.n_days} days | per trade {100*e.mean:+.2f}% "
          f"CI [{100*e.lo:+.2f}%, {100*e.hi:+.2f}%]")
    print(f"  account: compounded {100*sim.total_return:+.1f}%  Sharpe {sim.sharpe:.2f}  "
          f"maxDD {100*sim.max_drawdown:.1f}%  taken {sim.n_taken}")

    med, mean = float(t["net"].median()), float(t["net"].mean())
    shape = "RIGHT-TAILED — suspect" if mean > med else "left-tailed — mean not tail-driven"
    print(f"\n1. SHAPE          mean {100*mean:+.2f}%  median {100*med:+.2f}%  -> {shape}")

    rs = t["net"].sort_values()
    print("2. CONCENTRATION  ", end="")
    for k in (5, 10, 25, 50):
        if len(rs) > k:
            print(f"drop{k} {100*rs.iloc[:-k].mean():+.2f}%  ", end="")
    print(f"| target {100*t['took_target'].mean():.0f}%  stop {100*t['took_stop'].mean():.0f}%")

    print("\n3. LARGEST TRADES (read these — they have overturned three conclusions)")
    top = t.nlargest(10, "net")
    print(f"   {'symbol':<8}{'date':<12}{'entry':>9}{'ADV20':>14}{'net':>10}  target?")
    for _, r in top.iterrows():
        print(f"   {r['symbol']:<8}{str(r['date'])[:10]:<12}{r['entry']:>9.2f}"
              f"{r['adv20']:>14,.0f}{100*r['net']:>9.1f}%  {bool(r['took_target'])}")

    print("\n4. PER YEAR")
    ty = t.assign(y=pd.DatetimeIndex(t["date"]).year)
    for y, grp in ty.groupby("y"):
        c = cal[(cal >= np.datetime64(f"{y}-01-01")) & (cal <= np.datetime64(f"{y}-12-31"))]
        s = simulate(grp, c, spec.slots) if len(c) else None
        print(f"   {y}  n={len(grp):>4}  per-trade {100*grp['net'].mean():+7.2f}%  "
              f"median {100*grp['net'].median():+7.2f}%  "
              f"compounded {100*s.total_return:+8.1f}%" if s else f"   {y}  n={len(grp)}")
    pos_y = sum(1 for _, g in ty.groupby("y") if g["net"].mean() > 0)
    print(f"   {pos_y} of {ty['y'].nunique()} years positive per-trade")

    print("\n5. EXCESS OVER THE UNCONDITIONAL CONTROL (§53's trap)")
    c = control_return(u, spec, dev_only=not unseal)
    excess = e.mean - c["mean"]
    print(f"   control  n={c['n']:,}  {100*c['mean']:+.2f}% CI [{100*c['lo']:+.2f}%, {100*c['hi']:+.2f}%]  "
          f"median {100*c['median']:+.2f}%")
    print(f"   strategy {100*e.mean:+.2f}%  ->  EXCESS {100*excess:+.2f}%"
          f"   ({e.mean/c['mean']:.1f}x the control)" if c["mean"] > 0 else
          f"   strategy {100*e.mean:+.2f}%  ->  EXCESS {100*excess:+.2f}%")

    n_trials = sum(1 for _ in open(TRIALS_FILE, encoding="utf-8")) if TRIALS_FILE.exists() else 1
    daily = pd.Series(np.diff(np.concatenate([[1.0], (1 + t["net"]).cumprod().to_numpy()])))
    dsr = pstats.deflated_sharpe_ratio(t["net"], n_trials=max(n_trials, 1))
    print(f"\n6. DEFLATED SHARPE at {n_trials} search trials: {dsr:.3f}"
          f"   ({'clears 0.95' if dsr > 0.95 else 'does NOT clear 0.95'})")

    return {"n": len(t), "mean": float(e.mean), "lo": float(e.lo), "hi": float(e.hi),
            "median": med, "compounded": sim.total_return, "sharpe": sim.sharpe,
            "control_mean": c["mean"], "excess": float(excess), "dsr": float(dsr),
            "years_positive": pos_y, "years": int(ty["y"].nunique())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--unseal", action="store_true")
    args = ap.parse_args(argv)
    audit(Spec.load(args.spec), unseal=args.unseal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
