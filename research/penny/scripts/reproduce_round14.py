"""reproduce_round14.py — rebuild the N=15..400 table, then re-cost it honestly.

Why this script exists
-----------------------
The round-14 table is the basis of the current recommendation (N=60, one order a
session) and **it is not reproducible from committed code**. `bf2ccd4` touched
only `LOG.md`, `specs/FINAL_RETAIL.json` and `strategy/cost_model.py`; `fdc7d26`
touched `LOG.md` alone. `cost_model.py` is imported by nothing. The numbers came
from an uncommitted in-session variant of `scripts/risk_shaped.py:run`.

So this does two runs of the same configuration:

**LEGACY** reproduces what the old loop actually charged -- `R_t100` is a gross
column and the loop subtracted `0.5 * cost`, so the profit-target arm paid
**half** a round trip of the flat schedule.

**HONEST** charges exactly one size-aware round trip from `cost_model`, at the
position size the account actually trades.

The `orders/day` column is the reconstruction check: at H=60 the stagger admits
`ceil(N/60)` entries per session, which must print 1.0/1.0/1.0/3.0/7.0 across
N=15/30/60/150/400 to match the published table.

    python research/penny/scripts/reproduce_round14.py --period dev
    python research/penny/scripts/reproduce_round14.py --period holdout --unseal
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
from research.penny.scripts.risk_shaped import attach_target_returns   # noqa: E402
from research.penny.scripts.run_composite import add_composite         # noqa: E402
from research.penny.scripts.veto_holdout import SEAL, frame            # noqa: E402
from research.penny.strategy.book import BookConfig, run_book          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("r14")

SPEC = json.loads(Path("research/penny/specs/FINAL_RETAIL.json").read_text(encoding="utf-8"))
VETO_SPEC = json.loads(Path("research/penny/specs/VETO.json").read_text(encoding="utf-8"))
SIGNS = {k: float(v) for k, v in VETO_SPEC["composite_signs"].items()}
from research.penny.dates import DEV_START                             # noqa: E402

HOLD = SPEC["portfolio"]["horizon_sessions"]
VOL_CAP = SPEC["filter"]["vol_cap_quantile"]
VETO_FRAC = SPEC["filter"]["composite_veto_fraction"]
TARGET = SPEC["portfolio"]["profit_target"]
SLOTS = [15, 30, 60, 150, 400]


def market_calendar(period: str) -> pd.DatetimeIndex:
    """Every session in the panel, not only the days that had candidates.

    The old loops iterated `df["date"].unique()` -- candidate days only -- so a
    60-session hold was 60 *candidate* days and the annualisation divided by the
    wrong number of sessions.
    """
    d = pd.read_parquet(bars.DATA_ROOT / "panel_strategy.parquet", columns=["date"])
    cal = pd.DatetimeIndex(np.sort(d["date"].unique()))
    cal = cal[cal >= SEAL] if period == "holdout" else cal[cal < SEAL]
    if period == "dev":
        cal = cal[cal >= DEV_START]
    return cal


def build_frame(period: str) -> pd.DataFrame:
    df = add_composite(frame(HOLD, VETO_SPEC["population"]["crash_pct"], period),
                       SIGNS, "COMPOSITE")
    if period == "dev":
        df = df[df["date"] >= DEV_START]
    return attach_target_returns(df, HOLD, [TARGET])


def sweep(df: pd.DataFrame, cal: pd.DatetimeIndex, account: float,
          seeds: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for n in SLOTS:
        for mode, scale, label in (("flat", 0.5, "LEGACY 0.5x flat"),
                                   ("size_aware", 1.0, "HONEST 1x size-aware")):
            runs = []
            for s in range(seeds):
                cfg = BookConfig(n_slots=n, hold=HOLD, vol_cap=VOL_CAP,
                                 veto_frac=VETO_FRAC, target=TARGET,
                                 weight="equal", fill="random", stagger=True,
                                 account_usd=account, cost_mode=mode,
                                 cost_scale=scale)
                # LEGACY reproduces the old charge exactly: the target column is
                # gross and the loop took HALF a flat round trip off it.
                r = run_book(df, cal, cfg, seed=s)
                if r is not None:
                    runs.append(r.to_dict())
            if not runs:
                continue
            t = pd.DataFrame(runs)
            rows.append({
                "n_slots": n, "cost_mode": label,
                "usd_per_name": account / n,
                "cost_bps": float(t["mean_cost_bps"].mean()),
                "orders_per_session": float(t["orders_per_session"].mean()),
                "sharpe": float(t["sharpe"].mean()),
                "sharpe_sd": float(t["sharpe"].std()),
                "ann": float(t["ann"].mean()),
                "maxdd": float(t["max_drawdown"].mean()),
                "entered": float(t["n_entered"].mean()),
            })
        logger.info("  N=%d done", n)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--unseal", action="store_true",
                    help="required to read the holdout")
    ap.add_argument("--account", type=float, default=50_000.0)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.period == "holdout" and not args.unseal:
        raise SystemExit("refusing to read the sealed holdout without --unseal")

    cal = market_calendar(args.period)
    logger.info("calendar: %d sessions, %s .. %s", len(cal),
                str(cal[0])[:10], str(cal[-1])[:10])
    df = build_frame(args.period)
    logger.info("frame: %s rows | %d cells | %s .. %s", f"{len(df):,}",
                df["date"].nunique(), str(df["date"].min())[:10],
                str(df["date"].max())[:10])

    res = sweep(df, cal, args.account, args.seeds)

    print("=" * 112)
    print(f"ROUND 14 REBUILT — {args.period}, ${args.account:,.0f} account, "
          f"{args.seeds} seeds, H={HOLD}, vol cap {VOL_CAP}, veto {VETO_FRAC}, "
          f"target +{100*TARGET:.0f}%")
    print("LEGACY = what the old loop charged (gross target column minus HALF a "
          "flat round trip)")
    print("HONEST = exactly one size-aware round trip at the traded position size")
    print("=" * 112)
    print(f"  {'N':>5}{'$/name':>9}{'cost bps':>10}{'ord/day':>9}{'Sharpe':>8}"
          f"{'+/-':>7}{'annual':>10}{'maxDD':>9}{'trades':>9}  mode")
    for _, r in res.iterrows():
        print(f"  {int(r.n_slots):>5}{r.usd_per_name:>9,.0f}{r.cost_bps:>10.0f}"
              f"{r.orders_per_session:>9.2f}{r.sharpe:>8.2f}{r.sharpe_sd:>7.2f}"
              f"{r['ann']:>+9.1f}%{r.maxdd:>8.1f}%{r.entered:>9,.0f}  {r.cost_mode}")
    print("=" * 112)

    out = args.out or str(bars.DATA_ROOT / f"round14_rebuilt_{args.period}.json")
    res.to_json(out, orient="records", indent=1)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
