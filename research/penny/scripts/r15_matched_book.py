"""r15_matched_book.py — the H2 comparison that controls for book size.

The confound H2's first pass could not avoid
---------------------------------------------
There are 17 sectors, so a cap of one name per sector makes the book physically
incapable of holding more than 17 positions. "N=30 with cap 1" is therefore not
a 30-name book at all, and comparing its Sharpe to an uncapped 30-name book
compares two different-sized books.

The comparison that means something is **matched on what actually gets held**:
measure the realised concurrent position count and the realised order rate for
every configuration, then read the Sharpe of books that hold the same number of
names. If a sector-diversified 17-name book beats an unconstrained 17-name book,
`sigma_common` is a lever that the three-term identity `N = (H/tau) * K * (1-d)`
never contemplated — because it changes the Sharpe achievable at a given N
rather than changing N.

    python research/penny/scripts/r15_matched_book.py --period dev
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
from research.penny.scripts.r15_concurrency import attach_sector       # noqa: E402
from research.penny.scripts.reproduce_round14 import (                 # noqa: E402
    HOLD, TARGET, VETO_FRAC, VOL_CAP, build_frame, market_calendar)
from research.penny.strategy.book import BookConfig, run_book          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("matched")

SEEDS = 15


def concurrency(trades: pd.DataFrame, cal) -> Dict[str, float]:
    """Realised concurrent holdings — what the operator actually tracks."""
    idx = {d: i for i, d in enumerate(pd.DatetimeIndex(sorted(pd.Series(cal).unique())))}
    live = np.zeros(len(idx))
    s = pd.to_datetime(trades["date"]).map(idx).to_numpy()
    e = pd.to_datetime(trades["exit_date"]).map(idx).to_numpy()
    for a, b in zip(s, e):
        if np.isfinite(a) and np.isfinite(b):
            live[int(a):int(b) + 1] += 1
    live = live[live > 0]
    return {"n_live_mean": float(live.mean()) if len(live) else np.nan,
            "n_live_p90": float(np.percentile(live, 90)) if len(live) else np.nan,
            "n_live_max": float(live.max()) if len(live) else np.nan}


def run(df, cal, **kw) -> Dict[str, float]:
    sh, ann, dd, ordr, live, sects = [], [], [], [], [], []
    for s in range(SEEDS):
        r = run_book(df, cal, BookConfig(**kw), seed=s)
        if r is None:
            continue
        sh.append(r.sharpe); ann.append(r.ann); dd.append(r.max_drawdown)
        ordr.append(r.orders_per_session)
        c = concurrency(r.trades, cal)
        live.append(c["n_live_mean"])
        if "sector" in r.trades:
            sects.append(r.trades["sector"].nunique())
    if not sh:
        return {}
    return {"sharpe": float(np.mean(sh)), "sharpe_sd": float(np.std(sh, ddof=1)),
            "ann": float(np.mean(ann)), "maxdd": float(np.mean(dd)),
            "orders": float(np.mean(ordr)), "n_live": float(np.mean(live)),
            "sectors": float(np.mean(sects)) if sects else np.nan}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.period == "holdout" and not args.unseal:
        raise SystemExit("refusing to read the sealed holdout without --unseal")

    cal = market_calendar(args.period)
    df = attach_sector(build_frame(args.period))
    logger.info("frame %s rows | %d sectors", f"{len(df):,}", df["sector"].nunique())

    base = dict(hold=HOLD, vol_cap=VOL_CAP, veto_frac=VETO_FRAC, target=TARGET,
                cost_mode="size_aware")
    rows: List[Dict] = []

    print("=" * 120)
    print(f"MATCHED-BOOK COMPARISON ({args.period}) — Sharpe read against the "
          f"book size actually held")
    print("=" * 120)
    print(f"  {'config':<26}{'slots':>7}{'cap':>5}{'live':>7}{'ord/day':>9}"
          f"{'sectors':>9}{'Sharpe':>9}{'+/-':>7}{'annual':>9}{'maxDD':>9}")

    grid = [("unconstrained", n, 0) for n in (15, 20, 30, 60, 100)]
    grid += [("sector cap 1", n, 1) for n in (30, 60, 100)]
    grid += [("sector cap 2", n, 2) for n in (30, 60, 100)]
    grid += [("sector cap 3", n, 3) for n in (60, 100)]

    for label, n, cap in grid:
        r = run(df, cal, n_slots=n, sector_cap=cap, **base)
        if not r:
            continue
        rows.append({"config": label, "n_slots": n, "sector_cap": cap, **r})
        print(f"  {label:<26}{n:>7}{(cap or '-'):>5}{r['n_live']:>7.1f}"
              f"{r['orders']:>9.2f}{r['sectors']:>9.1f}{r['sharpe']:>9.2f}"
              f"{r['sharpe_sd']:>7.2f}{r['ann']:>+8.1f}%{r['maxdd']:>8.1f}%")

    res = pd.DataFrame(rows)
    print("-" * 120)
    print("  MATCHED PAIRS — books of the same realised size, one diversified:")
    print(f"  {'live positions':>16}{'unconstrained':>16}{'sector-capped':>16}"
          f"{'delta':>9}{'maxDD unc':>11}{'maxDD cap':>11}")
    unc = res[res["sector_cap"] == 0].sort_values("n_live")
    cap = res[res["sector_cap"] > 0].sort_values("n_live")
    for _, c in cap.iterrows():
        near = unc.iloc[(unc["n_live"] - c["n_live"]).abs().argmin()]
        if abs(near["n_live"] - c["n_live"]) > 0.25 * c["n_live"]:
            continue
        print(f"  {c['n_live']:>16.1f}{near['sharpe']:>16.2f}{c['sharpe']:>16.2f}"
              f"{c['sharpe'] - near['sharpe']:>+9.2f}"
              f"{near['maxdd']:>10.1f}%{c['maxdd']:>10.1f}%")
    print("=" * 120)

    out = args.out or str(bars.DATA_ROOT / f"r15_matched_{args.period}.json")
    Path(out).write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
