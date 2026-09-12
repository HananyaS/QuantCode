"""compare_strategies.py — the comparison table that is not just a Sharpe.

What this answers that the old tables could not
------------------------------------------------
Every prior round ranked configurations on a single Sharpe with no interval, no
deflation for the size of the search behind it, and no test of the variance
assumption the breadth argument rests on. This runs the full battery
(`eval/battery.py`) on the corrected book (`strategy/book.py`) across the
position-count sweep and the two filters, and reports:

* **Sharpe with a standard error** (Mertens/Lo), a stationary-bootstrap
  interval, and the seed-to-seed dispersion side by side. Three different kinds
  of uncertainty, all of which were previously invisible.
* **Deflated Sharpe** against the project's real cumulative trial count, which
  `stats.deflated_sharpe_ratio` has been able to compute since round 4 and has
  never been given.
* **The tail index** of per-trade returns. §80's "breadth is irreducible"
  argument is `SE = sigma/sqrt(N)`; if `alpha < 2` that arithmetic does not
  apply and the required breadth is a different number.
* **Average uniqueness**, converting the nominal trade count into the number of
  independent observations behind it. At a 60-session hold this is the
  difference between n=973 and something far smaller.
* **PBO** over the whole sweep -- whether picking the best cell of this table
  tells you anything about which cell will be best next.

The four arms are a real factorial, which `breadth_curve.py` never was: its
"noveto" control still filled with `nlargest(COMPOSITE)`, changing the veto and
the fill rule together.

    python research/penny/scripts/compare_strategies.py --period dev
    python research/penny/scripts/compare_strategies.py --period holdout --unseal
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

from research.penny.etl import bars                                    # noqa: E402
from research.penny.eval import battery as bt                          # noqa: E402
from research.penny.scripts.reproduce_round14 import (                 # noqa: E402
    HOLD, TARGET, build_frame, market_calendar)
from research.penny.strategy.book import BookConfig, run_book          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("compare")

SLOTS = [15, 30, 60, 150, 400]

# vol_cap, veto_frac, label -- a genuine 2x2, with the fill rule held at
# "random" in every arm so the veto is the only thing that changes.
ARMS = [
    (0.50, 0.25, "volcap+veto (V3)"),
    (0.50, 0.00, "volcap only"),
    (1.00, 0.25, "veto only"),
    (1.00, 0.00, "base (no filter)"),
]

# The project's cumulative trial ledger. `search_trials.jsonl` is the automated
# round-9 search; PRIOR_TRIALS=72 is fixed by PREREGISTRATION_R4.md:70. Rounds
# 11-14 add the block scores, the slot sweeps and the target sweep. Counting
# generously is the conservative direction for a deflated Sharpe.
PRIOR_TRIALS = 72


def cumulative_trials(this_round: int) -> int:
    f = bars.DATA_ROOT / "search_trials.jsonl"
    searched = sum(1 for _ in f.open(encoding="utf-8")) if f.exists() else 0
    return searched + PRIOR_TRIALS + this_round


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--account", type=float, default=50_000.0)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.period == "holdout" and not args.unseal:
        raise SystemExit("refusing to read the sealed holdout without --unseal")

    cal = market_calendar(args.period)
    df = build_frame(args.period)
    logger.info("calendar %d sessions | frame %s rows | %d cells",
                len(cal), f"{len(df):,}", df["date"].nunique())

    n_cells = len(SLOTS) * len(ARMS)
    n_trials = cumulative_trials(n_cells)
    logger.info("cumulative trial count for the deflated Sharpe: %d "
                "(%d searched + %d prior + %d this round)",
                n_trials, n_trials - PRIOR_TRIALS - n_cells, PRIOR_TRIALS, n_cells)

    rows: List[Dict] = []
    curves: Dict[str, pd.Series] = {}
    seed_sharpes: List[float] = []

    for cap, veto, label in ARMS:
        for n in SLOTS:
            sheets, per_seed = [], []
            for s in range(args.seeds):
                cfg = BookConfig(n_slots=n, hold=HOLD, vol_cap=cap,
                                 veto_frac=veto, target=TARGET, weight="equal",
                                 fill="random", stagger=True,
                                 account_usd=args.account,
                                 cost_mode="size_aware")
                res = run_book(df, cal, cfg, seed=s)
                if res is None:
                    continue
                sheet = bt.tearsheet(res.daily, res.equity, res.trades, cal,
                                     n_trials=n_trials, bootstrap=(s == 0))
                sheet["orders_per_session"] = res.orders_per_session
                sheet["cost_bps"] = res.mean_cost_bps
                sheet["n_entered"] = res.n_entered
                sheets.append(sheet)
                per_seed.append(sheet["sharpe"])
                if s == 0:
                    curves[f"{label} N={n}"] = res.daily
            if not sheets:
                continue
            agg = pd.DataFrame(sheets).mean(numeric_only=True).to_dict()
            agg.update({"arm": label, "n_slots": n,
                        "seed_sharpe_sd": float(np.std(per_seed, ddof=1))
                        if len(per_seed) > 1 else float("nan"),
                        "n_seeds": len(sheets)})
            rows.append(agg)
            seed_sharpes.extend(per_seed)
            logger.info("  %-20s N=%-4d Sharpe %.2f", label, n, agg["sharpe"])

    res = pd.DataFrame(rows)

    # --- deflate against the DISPERSION OF THIS SEARCH ---------------------
    trial_sd = float(np.std(res["sharpe"], ddof=1))
    res["expected_max_sharpe"] = bt.expected_max_sharpe(n_trials, trial_sd)
    res["sharpe_minus_expected_max"] = res["sharpe"] - res["expected_max_sharpe"]

    # --- PBO across the whole sweep ---------------------------------------
    mat = pd.DataFrame(curves).dropna()
    pbo = bt.pbo_cscv(mat, n_splits=8) if mat.shape[1] >= 2 else {"pbo": np.nan}

    # --- report ------------------------------------------------------------
    print("=" * 132)
    print(f"STRATEGY COMPARISON — {args.period}, ${args.account:,.0f}, "
          f"{args.seeds} seeds, H={HOLD}, target +{100*TARGET:.0f}%, "
          f"one size-aware round trip")
    print(f"deflating against {n_trials} cumulative trials, "
          f"sd(Sharpe) across this sweep = {trial_sd:.2f}  ->  "
          f"expected max from noise = {res['expected_max_sharpe'].iloc[0]:.2f}")
    print("=" * 132)
    hdr = (f"  {'arm':<20}{'N':>5}{'Sharpe':>8}{'+/-SE':>7}{'seed sd':>9}"
           f"{'boot 95%':>18}{'vs E[max]':>10}{'Sortino':>9}{'Calmar':>8}"
           f"{'CVaR95':>9}{'uniq':>7}{'eff n':>8}")
    print(hdr)
    for _, r in res.sort_values(["arm", "n_slots"]).iterrows():
        boot = (f"[{r.get('sharpe_boot_lo', np.nan):+.2f},"
                f"{r.get('sharpe_boot_hi', np.nan):+.2f}]")
        print(f"  {r['arm']:<20}{int(r.n_slots):>5}{r['sharpe']:>8.2f}"
              f"{r['sharpe_se']:>7.2f}{r['seed_sharpe_sd']:>9.2f}{boot:>18}"
              f"{r['sharpe_minus_expected_max']:>+10.2f}{r['sortino']:>9.2f}"
              f"{r['calmar']:>8.2f}{100*r['cvar_95']:>8.2f}%"
              f"{r['avg_uniqueness']:>7.3f}{r['effective_n_trades']:>8.0f}")
    print("-" * 132)
    print(f"  {'arm':<20}{'N':>5}{'ord/day':>9}{'cost bps':>10}{'trades':>9}"
          f"{'@barrier':>10}{'tail a-':>9}{'maxDD':>9}{'E[maxDD]':>10}"
          f"{'DD ratio':>10}{'under H2O':>11}{'MinTRL yr':>11}")
    for _, r in res.sort_values(["arm", "n_slots"]).iterrows():
        trl = r["min_track_record_days"] / 252.0
        print(f"  {r['arm']:<20}{int(r.n_slots):>5}{r['orders_per_session']:>9.2f}"
              f"{r['cost_bps']:>10.0f}{r['n_entered']:>9,.0f}"
              f"{100*r['barrier_share']:>9.1f}%{r['tail_alpha_losses']:>9.2f}"
              f"{100*r['max_drawdown']:>8.1f}%{100*r['expected_max_drawdown']:>9.1f}%"
              f"{r['mdd_vs_expected']:>10.2f}{r['pct_underwater']:>11.2f}"
              f"{trl:>11.1f}")
    print("-" * 132)
    print(f"  PBO over {pbo.get('n_configs', 0)} configurations, "
          f"{pbo.get('n_combinations', 0)} splits: {pbo.get('pbo', float('nan')):.3f}")
    print("    (0.5 = selecting the best cell of this sweep carries no "
          "information about which will be best next)")
    print("=" * 132)

    out = args.out or str(bars.DATA_ROOT / f"battery_{args.period}.json")
    payload = {"period": args.period, "account": args.account,
               "seeds": args.seeds, "n_trials": n_trials,
               "trial_sharpe_sd": trial_sd, "pbo": pbo,
               "rows": json.loads(res.to_json(orient="records"))}
    Path(out).write_text(json.dumps(payload, indent=1, default=float),
                         encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
