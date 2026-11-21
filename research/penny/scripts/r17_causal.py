"""r17_causal.py -- PREREGISTRATION_R17: R15 measured the way it would be traded (dev only).

Two cells, fixed before this ran:
  A  causal frame, opening-auction buy limit at prior close x 1.10, split-adjusted
     path, +100% target, time exit at session e+60, marked to market daily
  B  as A, plus a stop at -50% of the split-adjusted entry

Everything else is R15 (60 slots, sector cap 3, vol cap 0.5, composite veto 0.25,
random fill, 15 seeds, $50k, size-aware cost) plus regulatory fees. The panel is
truncated at the last dev session before anything reads it, so no sealed-period
price can enter. The decision rules of §5 are applied and printed; the holdout is
not touched here.

    python research/penny/scripts/r17_causal.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.dates import DEV_START, SEAL, assert_dev                # noqa: E402
from research.penny.etl import bars                                         # noqa: E402
from research.penny.eval import benchmark as bm                              # noqa: E402
from research.penny.eval.stats import deflated_sharpe_ratio                  # noqa: E402
from research.penny.scripts.fetch_benchmarks import load as load_bench      # noqa: E402
from research.penny.scripts.reproduce_round14 import (                      # noqa: E402
    HOLD, TARGET, VETO_FRAC, VOL_CAP, market_calendar)
from research.penny.scripts.sector_split import (                            # noqa: E402
    attach_sector, attach_sector_relative)
from research.penny.strategy import causal                                   # noqa: E402
from research.penny.strategy.book import BookConfig, run_book                # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("r17")

SEEDS = 15
N_TRIALS = 317
CELLS = {"A": None, "B": 0.50}
SLOTS, SECTOR_CAP, ACCOUNT = 60, 3, 50_000.0
OUT_DIR = bars.DATA_ROOT / "r17"


def live_positions(trades: pd.DataFrame, cal: pd.DatetimeIndex) -> float:
    idx = {d: i for i, d in enumerate(cal)}
    live = np.zeros(len(cal))
    for d, rel in zip(trades["date"], trades["release_date"]):
        a = idx[pd.Timestamp(d)] + 1
        b = idx[pd.Timestamp(rel)] if pd.notna(rel) else len(cal) - 1
        live[a:b + 1] += 1
    live = live[live > 0]
    return float(live.mean()) if len(live) else float("nan")


def run_cell(df: pd.DataFrame, cal: pd.DatetimeIndex, paths: causal.Paths) -> Dict:
    cfg = BookConfig(n_slots=SLOTS, hold=HOLD, vol_cap=VOL_CAP, veto_frac=VETO_FRAC,
                     target=TARGET, sector_cap=SECTOR_CAP, account_usd=ACCOUNT,
                     cost_mode="size_aware", mark="mtm")
    daily, rows, trades = {}, [], []
    for s in range(SEEDS):
        r = run_book(df, cal, cfg, seed=s, paths=paths)
        if r is None:
            raise RuntimeError(f"seed {s}: too few entries")
        daily[s] = r.daily
        t = r.trades.assign(seed=s)
        trades.append(t)
        rows.append({"seed": s, "sharpe_book": r.sharpe, "max_dd_book": r.max_drawdown,
                     "ann_book": r.ann, "n_entered": r.n_entered,
                     "orders_per_session": r.orders_per_session,
                     "live": live_positions(t, cal), "mean_cost_bps": r.mean_cost_bps})
    tr = pd.concat(trades, ignore_index=True)
    per_seed = pd.DataFrame(rows)
    orders = per_seed["orders_per_session"].mean() * len(cal)
    diag = {
        "n_entered_mean": float(per_seed["n_entered"].mean()),
        "limit_fill_rate": float(per_seed["n_entered"].mean() / orders) if orders else float("nan"),
        "live_positions_mean": float(per_seed["live"].mean()),
        "mean_cost_bps": float(per_seed["mean_cost_bps"].mean()),
        "exit_mix": tr["kind"].value_counts(normalize=True).round(4).to_dict(),
        "trades_with_split_correction": int((tr["n_corr"] > 0).sum()),
        "net_per_trade_closed_mean": float(tr["net"].dropna().mean()),
        "net_per_trade_closed_median": float(tr["net"].dropna().median()),
        "dsr_mean_over_seeds": float(np.mean([deflated_sharpe_ratio(daily[s].to_numpy()[1:], N_TRIALS)
                                              for s in daily])),
    }
    return {"daily": pd.DataFrame(daily), "per_seed": per_seed, "trades": tr, "diag": diag}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DIR / "r17_dev.json"))
    args = ap.parse_args(argv)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cal = market_calendar("dev")
    assert cal[0] >= DEV_START and cal[-1] < SEAL
    cutoff = cal[-1]
    u = causal.load_panel(cutoff)
    assert_dev(u["date"])
    u = attach_sector_relative(attach_sector(u), 5)
    assert u["symbol"].is_monotonic_increasing                     # merges kept the sort
    df = causal.causal_frame(u, DEV_START, SEAL, 0.15, causal.frozen_signs())
    assert_dev(df["date"])
    df["_path"] = np.arange(len(df))
    logger.info("calendar %d sessions %s..%s | frame %s rows, %d days | %.0fs",
                len(cal), cal[0].date(), cutoff.date(), f"{len(df):,}", df["date"].nunique(),
                time.time() - t0)

    splits = causal.load_vendor_splits(cutoff)
    corr = causal.split_corrections(u, splits)
    logger.info("vendor splits %d | rows corrected %d", len(splits), int((corr != 1).sum()))

    bench, rf = load_bench(cal[0], cutoff)
    out: Dict = {"cutoff": str(cutoff.date()), "sessions": len(cal), "frame_rows": int(len(df)),
                 "rows_corrected": int((corr != 1).sum()), "cells": {}}
    results = {}
    for name, stop in CELLS.items():
        paths = causal.holding_paths(u, df["_pos"].to_numpy(), HOLD, TARGET, stop, corr)
        logger.info("cell %s paths: %s | %.0fs", name, paths.summary(), time.time() - t0)
        res = run_cell(df, cal, paths)
        comp = bm.compare(res["daily"], bench[["SPY", "QQQ", "IWM"]], rf)
        results[name] = (res, comp)
        out["cells"][name] = {"stop": stop, "comparison": comp.to_dict(), "diag": res["diag"],
                              "per_seed": res["per_seed"].to_dict("records"),
                              "path_summary": paths.summary()}
        res["daily"].rename(columns=str).to_parquet(OUT_DIR / f"r17_dev_daily_{name}.parquet")
        logger.info("cell %s done | %.0fs", name, time.time() - t0)

    # ---------------- decision rules, §5 --------------------------------------
    sA = results["A"][1].table.loc["strategy"]
    sB = results["B"][1].table.loc["strategy"]
    adopt = "B" if (sB["sharpe"] >= sA["sharpe"] - 0.05 and sB["max_dd"] > sA["max_dd"]) else "A"
    comp = results[adopt][1]
    ci = comp.tests["strategy_sharpe_ci"]
    spy_sh = float(comp.table.loc["SPY", "sharpe"])
    kill_ci = ci[0] <= 0.0
    kill_spy = float(comp.table.loc["strategy", "sharpe"]) < spy_sh
    verdict = "KILL" if (kill_ci or kill_spy) else "PROCEED (holdout read #8 needs the user's go)"
    out["decision"] = {"adopted_cell": adopt, "sharpe_ci": ci, "spy_sharpe": spy_sh,
                       "kill_ci_includes_zero": bool(kill_ci), "kill_below_spy": bool(kill_spy),
                       "verdict": verdict}

    # ---------------- print ----------------------------------------------------
    print("=" * 112)
    print(f"ROUND 17 -- R15 causal, marked to market, dev {cal[0].date()}..{cutoff.date()} "
          f"({len(cal)} sessions), {SEEDS} seeds, N trials {N_TRIALS}")
    print("=" * 112)
    print(f"  {'series':<12}{'Sharpe':>8}{'seed sd':>9}{'CAGR':>9}{'maxDD':>9}{'total':>9}")
    for name in CELLS:
        t = results[name][1].table
        r = t.loc["strategy"]
        print(f"  {'cell ' + name:<12}{r.sharpe:>8.2f}{r.sharpe_seed_sd:>9.2f}{100*r.cagr:>8.1f}%"
              f"{100*r.max_dd:>8.1f}%{100*r.total_return:>8.1f}%")
    t = results["A"][1].table
    for b in ("SPY", "QQQ", "IWM"):
        r = t.loc[b]
        print(f"  {b:<12}{r.sharpe:>8.2f}{'':>9}{100*r.cagr:>8.1f}%{100*r.max_dd:>8.1f}%{100*r.total_return:>8.1f}%")
    for name in CELLS:
        te = results[name][1].tests
        a = te["alpha_vs_SPY"]
        print(f"\n  cell {name}: Sharpe CI {np.round(te['strategy_sharpe_ci'], 2).tolist()} | "
              f"vs SPY {te['sharpe_minus_SPY']['point']:+.2f} CI {np.round(te['sharpe_minus_SPY']['boot_ci'], 2).tolist()} "
              f"LW p {te['sharpe_minus_SPY']['ledoit_wolf_seed_avg']['p']:.3f} | "
              f"vs QQQ {te['sharpe_minus_QQQ']['point']:+.2f} CI {np.round(te['sharpe_minus_QQQ']['boot_ci'], 2).tolist()}")
        print(f"          alpha {100*a['alpha_ann']:+.1f}%/yr NW [{100*a['alpha_lo']:+.1f}, {100*a['alpha_hi']:+.1f}] "
              f"boot {np.round(100*np.array(a['alpha_boot_ci']), 1).tolist()} | beta SPY {a['beta']:.2f}, "
              f"IWM {te['beta_vs_IWM']['beta']:.2f}")
        print(f"          diag {json.dumps(out['cells'][name]['diag'], default=float)}")
    print("-" * 112)
    print(f"  adopted cell {adopt} | Sharpe CI {np.round(ci, 2).tolist()} | SPY dev Sharpe {spy_sh:.2f} "
          f"-> {verdict}")
    print("=" * 112)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    logger.info("wrote %s | %.0fs", args.out, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
