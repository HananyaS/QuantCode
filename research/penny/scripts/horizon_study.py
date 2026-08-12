"""horizon_study.py — is the edge bigger if you hold longer than a day?

Why this, and why now
---------------------
Five rounds have held **one intraday session**: enter 09:35, flat by 15:55. That
horizon was fixed in round 1 and, like the gap screen, never varied.

Round 5 measured the best pool the project has found — liquid low-priced names
gapping down 3-10% — at **+0.31% gross over 51,361 trades**, against a round
trip measured at **66.6 bps**. The edge is about half its cost. No model fixes a
factor of two; a *bigger edge* would.

Cost is paid once per round trip regardless of holding period, so a multi-day
hold amortises it. If 1-day reversal is 31 bps and 5-day reversal is 150 bps,
the second is tradable on the same cost base and the first is not.

What this measures
------------------
Close-to-close returns from the daily cache over horizons of 1, 2, 3, 5, 10 and
20 sessions, for the same pool, plus the gap-up population rounds 1-4 traded, as
a control. Entry is the **09:35 decision proxied by the daily open** and exit at
the close of session *t+h*.

Overnight risk is real and is reported: a multi-day hold is exposed to gaps,
which an intraday strategy is not, so the same mean return is worth less. Vol
and worst-case are reported alongside the mean rather than left implicit.

**Exploratory.** This is a new horizon, not the pre-registered one; it counts as
a trial and any follow-up needs its own pre-registration.

    python research/penny/scripts/horizon_study.py
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
from research.penny.eval import stats as pstats  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("horizon")

HORIZONS = [1, 2, 3, 5, 10, 20]

# Measured in round 4 at ADV >= $5M, both books, both legs, 99.9% coverage.
# The pools below reach down to $1M, so this is a floor, not an estimate.
COST_ROUND_TRIP = 0.00666


def load_universe(start=2016, end=2026) -> pd.DataFrame:
    frames = []
    for y in range(start, end + 1):
        frames.append(pd.read_parquet(
            bars.DATA_ROOT / "daily" / f"bars_{y}.parquet",
            columns=["symbol", "date", "open", "close", "volume", "close_unadj"]))
    p = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])

    g = p.groupby("symbol", sort=False)
    p["prev_close"] = g["close"].shift(1)
    p["prev_close_unadj"] = g["close_unadj"].shift(1)
    p["dv"] = p["close"] * p["volume"]
    p["adv20"] = g["dv"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    p["gap"] = p["open"] / p["prev_close"] - 1.0

    # Forward closes. shift(-h) inside the symbol group: the guard against
    # bleeding one symbol's future into another's row.
    for h in HORIZONS:
        p[f"fwd_{h}"] = g["close"].shift(-h) / p["open"] - 1.0
    return p


def measure(pool: pd.DataFrame, label: str) -> List[Dict]:
    rows: List[Dict] = []
    for h in HORIZONS:
        col = f"fwd_{h}"
        s = pool[col].replace([np.inf, -np.inf], np.nan)
        ok = s.notna()
        # The last h sessions of the sample have no forward close and must drop
        # out, not be filled -- a truncated tail would bias the longest horizons.
        if ok.sum() < 500:
            continue
        est = pstats.cluster_bootstrap_mean(s[ok], pool.loc[ok, "date"], n_boot=3_000)
        rows.append({
            "pool": label, "horizon": h, "n": int(est.n), "n_days": int(est.n_days),
            "mean": float(est.mean), "lo": float(est.lo), "hi": float(est.hi),
            "p_greater": float(est.p_value_greater),
            "hit": float((s[ok] > 0).mean()),
            "sd": float(s[ok].std()),
            "net_at_cost": float(est.mean - COST_ROUND_TRIP),
            # Cost is paid once whatever the horizon, so per-day return is what
            # compounds -- and what makes horizons comparable.
            "net_per_day": float((est.mean - COST_ROUND_TRIP) / h),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Return by holding horizon.")
    parser.add_argument("--out", default=str(bars.DATA_ROOT / "horizon_study.json"))
    args = parser.parse_args(argv)

    p = load_universe()
    base = (p["prev_close_unadj"].between(0.50, 10.0) & p["adv20"].notna()
            & np.isfinite(p["gap"]))
    logger.info("universe %s rows", f"{len(p):,}")

    pools = {
        "reversal (gap -10..-3%, ADV>=$1M)":
            base & (p["adv20"] >= 1e6) & p["gap"].between(-0.10, -0.03),
        "reversal deep (gap -20..-5%, ADV>=$1M)":
            base & (p["adv20"] >= 1e6) & p["gap"].between(-0.20, -0.05),
        "reversal liquid (gap -10..-3%, ADV>=$5M)":
            base & (p["adv20"] >= 5e6) & p["gap"].between(-0.10, -0.03),
        "CONTROL gap-up (gap>=+3%, ADV>=$5M)":
            base & (p["adv20"] >= 5e6) & (p["gap"] >= 0.03),
        "CONTROL no screen (ADV>=$5M)":
            base & (p["adv20"] >= 5e6),
    }

    rows: List[Dict] = []
    for label, mask in pools.items():
        pool = p[mask]
        logger.info("%-42s %s symbol-days", label, f"{len(pool):,}")
        rows += measure(pool, label)

    table = pd.DataFrame(rows)
    logger.info("=" * 100)
    logger.info("RETURN BY HOLDING HORIZON (gross; net charges one %.0f bp round trip)\n%s",
                1e4 * COST_ROUND_TRIP,
                table.to_string(index=False, float_format="%.4f"))
    logger.info("-" * 100)

    tradable = table[(table["lo"] > 0) & (table["net_at_cost"] > 0)]
    if tradable.empty:
        logger.info("NO horizon on any pool clears cost with a CI excluding zero.")
    else:
        logger.info("Horizons clearing cost with a CI excluding zero:\n%s",
                    tradable.to_string(index=False, float_format="%.4f"))

    Path(args.out).write_text(json.dumps(
        {"cost_round_trip": COST_ROUND_TRIP, "horizons": HORIZONS,
         "rows": rows, "exploratory": True}, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
