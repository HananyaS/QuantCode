"""feasibility.py — make the veto implementable without breaking it.

Round 11's veto is validated and unimplementable: 787 concurrent positions at
0.040% of capital each, $40 a name on a $100k account. Amendment 2 fixes the
constraint and the identity

    N_pos = (H / tau) * K * (1 - delta)

with tau the entry cadence in sessions, H the holding period, K the per-cohort
size and delta the duplicate rate. This module sweeps (tau, K, H, rule) on
DEVELOPMENT ONLY, with the composite signs frozen at their pre-2020 fit.

Two selection rules are tested inside the surviving 75%:

* **random-K** -- unbiased for the cohort mean if the survivors are flat in the
  score, which section 78 measured but which must hold again here;
* **top-K by composite** -- if the survivors are NOT flat, this beats random and
  section 78's "the top decile does nothing" was population-specific.

Reporting both is the point: they disagree only if that central claim is wrong.

Entry days are a fixed calendar grid (every tau-th session from the first),
never chosen by which days look good. Nothing is refitted.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
from research.penny.dates import DEV_START                             # noqa: E402


@dataclass
class Config:
    tau: int          # sessions between cohorts
    K: int            # names per cohort (0 = all survivors)
    H: int            # holding period, sessions
    rule: str         # "random" | "top" | "all"


def simulate(df: pd.DataFrame, cfg: Config, seed: int = 0) -> Optional[Dict]:
    """Fixed calendar grid of entry days; equal weight inside a cohort.

    The base arm is the SAME cohort without the veto and at the same K, so the
    excess isolates the veto rather than the position count.
    """
    rng = np.random.default_rng(seed)
    sess = np.sort(df["date"].unique())
    entry_days = set(pd.Series(sess[::cfg.tau]).tolist())
    si = {d: i for i, d in enumerate(sess)}

    cohorts: List[Dict] = []
    live: Dict[int, set] = {}
    for d, g in df.groupby("date", sort=False):
        if d not in entry_days or len(g) < MIN_CELL:
            continue
        keep = max(1, int(np.ceil((1.0 - VETO_FRAC) * len(g))))
        surv = g.nlargest(keep, "COMPOSITE")
        if cfg.K and cfg.K < len(surv):
            sel = (surv.nlargest(cfg.K, "COMPOSITE") if cfg.rule == "top"
                   else surv.sample(cfg.K, random_state=int(rng.integers(1 << 31))))
            bsel = g.sample(min(cfg.K, len(g)),
                            random_state=int(rng.integers(1 << 31)))
        else:
            sel, bsel = surv, g
        cohorts.append({
            "date": d, "n": len(sel),
            "veto": float(sel["R"].mean() - 0.5 * sel["cost"].mean()),
            "base": float(bsel["R"].mean() - 0.5 * bsel["cost"].mean()),
        })
        i = si[d]
        for j in range(i, min(i + cfg.H, len(sess))):
            live.setdefault(j, set()).update(sel["symbol"])

    if len(cohorts) < 25:
        return None
    c = pd.DataFrame(cohorts)
    held = pd.Series({i: len(s) for i, s in live.items() if s})

    def ann(x: pd.Series):
        """Overlapping holds: 1/H of capital enters each session, so the daily
        return is the mean daily-equivalent of the live cohorts."""
        dly = (1 + x) ** (1 / cfg.H) - 1
        a = (1 + dly.mean()) ** 252 - 1
        sh = dly.mean() / dly.std() * np.sqrt(252) if dly.std() > 0 else 0.0
        return 100 * a, sh

    ex = c["veto"] - c["base"]
    b = ex.to_numpy()[rng.integers(0, len(ex), size=(4000, len(ex)))].mean(axis=1)
    a_v, s_v = ann(c["veto"])
    a_b, s_b = ann(c["base"])
    n90 = float(held.quantile(0.90)) if len(held) else 0.0
    return {"tau": cfg.tau, "K": cfg.K, "H": cfg.H, "rule": cfg.rule,
            "cohorts": len(c), "names_per_cohort": float(c["n"].median()),
            "N_med": float(held.median()) if len(held) else 0.0, "N_p90": n90,
            "weight_pct": 100.0 / n90 if n90 else float("nan"),
            "veto_ann": a_v, "veto_sharpe": s_v,
            "base_ann": a_b, "base_sharpe": s_b,
            "excess_60d": 100 * ex.mean(),
            "ex_lo": 100 * np.percentile(b, 2.5),
            "ex_hi": 100 * np.percentile(b, 97.5)}


def admissible(r: Dict) -> bool:
    """Amendment 2's constraint, applied verbatim."""
    return bool(r["N_p90"] <= 60 and r["weight_pct"] >= 100.0 / 60
                and r["names_per_cohort"] <= 25 and r["ex_lo"] > 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "feasibility.json"))
    args = ap.parse_args(argv)

    rows: List[Dict] = []
    for H in (40, 60, 90):
        df = add_composite(frame(H, SPEC["population"]["crash_pct"], "dev"),
                           SIGNS, "COMPOSITE")
        df = df[df["date"] >= DEV_START]
        for tau in (1, 5, 10, 21):
            for K in (0, 5, 10, 20):
                for rule in (("random", "top") if K else ("all",)):
                    r = simulate(df, Config(tau, K, H, rule))
                    if r:
                        rows.append(r)
        print(f"  ... H={H} done", flush=True)

    res = pd.DataFrame(rows)
    res["ok"] = res.apply(admissible, axis=1)

    print("=" * 128)
    print("FEASIBILITY SWEEP (development >=2020, composite signs frozen at the pre-2020 fit)")
    print("Admissible: N_p90 <= 60 positions, weight >= 1.67%, <=25 orders per entry day, "
          "excess CI excluding zero")
    print("=" * 128)
    print(f"  {'H':>3}{'tau':>5}{'K':>4}{'rule':>8}{'coh':>6}{'N_med':>7}{'N_p90':>7}"
          f"{'wt%':>7}{'base ann':>10}{'VETO ann':>10}{'Sharpe':>8}{'excess':>9}{'95% CI':>17}")
    for _, r in res.sort_values(["H", "tau", "K", "rule"]).iterrows():
        mark = "  <<< ADMISSIBLE" if r["ok"] else ""
        print(f"  {int(r.H):>3}{int(r.tau):>5}{int(r.K):>4}{r['rule']:>8}{int(r.cohorts):>6}"
              f"{r.N_med:>7.0f}{r.N_p90:>7.0f}{r.weight_pct:>7.2f}{r.base_ann:>+9.2f}%"
              f"{r.veto_ann:>+9.2f}%{r.veto_sharpe:>8.2f}{r.excess_60d:>+8.2f}%"
              f"  [{r.ex_lo:>+5.2f},{r.ex_hi:>+5.2f}]{mark}")
    print("=" * 128)
    ok = res[res["ok"]]
    print(f"{len(ok)} of {len(res)} configurations are admissible")
    if len(ok):
        best = ok.sort_values("veto_sharpe", ascending=False).iloc[0]
        print(f"best by Sharpe: H={int(best.H)} tau={int(best.tau)} K={int(best.K)} "
              f"rule={best['rule']}  ->  {best.veto_ann:+.2f}% annualised, "
              f"Sharpe {best.veto_sharpe:.2f}, {best.N_p90:.0f} positions at "
              f"{best.weight_pct:.2f}% each")
    res.to_json(args.out, orient="records", indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
