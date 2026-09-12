"""r15_selector.py — does H3's precision lift actually buy a smaller book?

The question this settles
--------------------------
H3 found two intraday features whose top decile beats the barrier-touch base
rate: `M_intraday_vol` (17.1% -> 22.9%, bottom decile 6.4%) and
`M_shares_per_trade` (17.1% -> 21.8%). A lift in `P(touch)` is only interesting
if it survives being turned into a fill rule, because that is the only form in
which it can reduce the position count.

The bar is arithmetic and was fixed in advance. `N = H * lambda`, so cutting the
book cuts the bet rate one-for-one, and `IR = IC * sqrt(BR)` prices it:

    to run 30 names instead of 60 the selector must add  +41% IC per bet
    to run 15 names instead of 60                        +100%

So the test is not "does the selector help at fixed N" -- it is "does a
selector-filled book of N names beat a randomly-filled book of MORE names".

Every arm holds the population, filters, horizon, target, stagger and cost
fixed. Only the rule choosing WHICH survivor fills a free slot changes, so a
comparison against the random arm is a clean read on the selector alone --
which `slots.py`'s confounded "noveto" control could never provide.

    python research/penny/scripts/r15_selector.py --period dev
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
from research.penny.features.intraday_cs import build_intraday_features  # noqa: E402
from research.penny.scripts.reproduce_round14 import (                 # noqa: E402
    HOLD, TARGET, VETO_FRAC, VOL_CAP, build_frame, market_calendar)
from research.penny.strategy.book import BookConfig, run_book          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sel")

SEEDS = 15
SLOTS = [15, 30, 60]
# the two H3 survivors, plus the random-feature control that must score nothing
SELECTORS = ["M_intraday_vol", "M_shares_per_trade", "B0_random"]


def run(df, cal, **kw) -> Dict[str, float]:
    sh, ann, dd, ordr, live = [], [], [], [], []
    for s in range(SEEDS):
        r = run_book(df, cal, BookConfig(**kw), seed=s)
        if r is None:
            continue
        sh.append(r.sharpe); ann.append(r.ann); dd.append(r.max_drawdown)
        ordr.append(r.orders_per_session); live.append(r.n_entered)
    if not sh:
        return {}
    return {"sharpe": float(np.mean(sh)), "sharpe_sd": float(np.std(sh, ddof=1)),
            "ann": float(np.mean(ann)), "maxdd": float(np.mean(dd)),
            "orders": float(np.mean(ordr)), "trades": float(np.mean(live))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.period == "holdout" and not args.unseal:
        raise SystemExit("refusing to read the sealed holdout without --unseal")

    cal = market_calendar(args.period)
    df = build_frame(args.period)
    fp = bars.DATA_ROOT / f"intraday_features_{args.period}.parquet"
    mf = (pd.read_parquet(fp) if fp.exists()
          else build_intraday_features(df[["symbol", "date"]]))
    if not fp.exists():
        mf.to_parquet(fp, index=False)
    df = df.merge(mf, on=["symbol", "date"], how="left")
    cov = df["M_intraday_vol"].notna().mean()
    logger.info("frame %s rows | minute coverage %.1f%%", f"{len(df):,}", 100 * cov)

    rows: List[Dict] = []
    base = dict(hold=HOLD, vol_cap=VOL_CAP, veto_frac=VETO_FRAC, target=TARGET,
                cost_mode="size_aware")

    print("=" * 108)
    print(f"H3 AS A FILL RULE ({args.period}) — same population, same filters, "
          f"same stagger; only the choice of WHICH survivor fills a slot changes")
    print("=" * 108)
    print(f"  {'N':>5}{'fill rule':<22}{'ord/day':>9}{'trades':>9}{'Sharpe':>9}"
          f"{'+/-':>7}{'annual':>9}{'maxDD':>9}{'vs random':>11}")
    for n in SLOTS:
        r0 = run(df, cal, n_slots=n, fill="random", **base)
        if not r0:
            continue
        rows.append({"n_slots": n, "fill": "random", **r0, "delta": 0.0})
        print(f"  {n:>5}{'random':<22}{r0['orders']:>9.2f}{r0['trades']:>9,.0f}"
              f"{r0['sharpe']:>9.2f}{r0['sharpe_sd']:>7.2f}{r0['ann']:>+8.1f}%"
              f"{r0['maxdd']:>8.1f}%{'—':>11}")
        for col in SELECTORS:
            if col not in df.columns:
                continue
            r = run(df, cal, n_slots=n, fill="top", fill_col=col, **base)
            if not r:
                continue
            d = r["sharpe"] - r0["sharpe"]
            rows.append({"n_slots": n, "fill": col, **r, "delta": d})
            tag = "  <<<" if d > 0 and col != "B0_random" else ""
            print(f"  {n:>5}{('top ' + col):<22}{r['orders']:>9.2f}"
                  f"{r['trades']:>9,.0f}{r['sharpe']:>9.2f}{r['sharpe_sd']:>7.2f}"
                  f"{r['ann']:>+8.1f}%{r['maxdd']:>8.1f}%{d:>+11.2f}{tag}")
        print()

    # --- the question that matters: does a selector-filled SMALL book beat a
    #     randomly-filled LARGE one?
    res = pd.DataFrame(rows)
    ref = res[(res["n_slots"] == 60) & (res["fill"] == "random")]
    if len(ref):
        r60 = float(ref["sharpe"].iloc[0])
        print("-" * 108)
        print(f"  Against the standing recommendation — 60 names, random fill, "
              f"Sharpe {r60:.2f}:")
        print(f"  {'N':>5}{'fill rule':<22}{'Sharpe':>9}{'gap to N=60':>14}"
              f"{'positions saved':>18}")
        for _, r in res[res["n_slots"] < 60].iterrows():
            print(f"  {int(r.n_slots):>5}{r['fill']:<22}{r['sharpe']:>9.2f}"
                  f"{r['sharpe'] - r60:>+14.2f}{60 - int(r.n_slots):>18}")
        print("=" * 108)

    out = args.out or str(bars.DATA_ROOT / f"r15_selector_{args.period}.json")
    Path(out).write_text(json.dumps(rows, indent=1, default=float),
                         encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
