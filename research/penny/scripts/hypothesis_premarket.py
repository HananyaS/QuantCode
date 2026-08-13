"""hypothesis_premarket.py — does premarket volume predict the day, at the size
where explosions actually happen?

The hypothesis
--------------
Across the thirty largest genuine gainers of the last two months, the most
frequent shared precursor is a **premarket dollar volume enormous relative to the
stock's own average daily volume** -- 70% traded at least 1x their ADV before the
bell, median 13x.

Round 4 already tested that signal and found it *negative*: on 41,892 candidates
with ADV >= $5M, premarket >= 1x ADV returned **-1.97%** from 09:35 to 15:55.

But the movers are not $5M-ADV names. Their **median ADV is $548,901**, and 60%
sit below $1M. The round-4 test was run on a population that structurally
excludes them, so it does not falsify the hypothesis at the size that matters.

    H: within ADV $100k-$5M, premarket dollar volume >= 1x ADV predicts a
       positive open-to-close return.

Design
------
* Population fixed **before** looking: every non-split symbol-day with
  ``adv20`` in [$100k, $5M), 2025-02 to 2026-08 (420,446 symbol-days,
  379 sessions, 2,441 tickers, containing 13 of the 30 movers).
* Signal from the 04:00-09:34 window only. The premarket cache asserts on write
  that no bar at or after 09:35 is ever stored.
* Outcome is **open to close** from the daily bar. Entry at the open is the
  earliest a premarket signal could be acted on; using 09:35 would need minute
  bars for 420k symbol-days, and the open is the more conservative choice
  anyway since it is the worst fill of the morning.
* Reverse splits excluded -- 16.1% of +50% "gainers" are unit changes, and they
  fade ~15% from the open, which would bias any gap-related test.
* Day-clustered bootstrap throughout; same-day trades are not independent.

**Exploratory.** The pool and threshold were chosen after reading the movers, so
this is a hypothesis generated from data and tested on the same period. A
positive result would need walk-forward confirmation before it meant anything.

    python research/penny/scripts/hypothesis_premarket.py
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
from research.penny.scripts.fetch_premarket import session_path  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("hyp")

ADV_MIN, ADV_MAX = 1e5, 5e6


def build_premarket_panel(pool: pd.DataFrame) -> pd.DataFrame:
    """Premarket dollar volume per symbol-day, from the 04:00-09:34 cache."""
    wanted = pool.groupby("date")["symbol"].apply(set)
    rows: List[pd.DataFrame] = []
    missing = 0
    for day in sorted(pool["date"].unique()):
        p = session_path(pd.Timestamp(day))
        if not p.exists():
            missing += 1
            continue
        try:
            m = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            missing += 1
            continue
        if m.empty or "symbol" not in m.columns:
            continue
        m = m[m["symbol"].isin(wanted.loc[day])]
        if m.empty:
            continue
        m = m.assign(dv=m["close"] * m["volume"])
        agg = m.groupby("symbol").agg(
            pm_dv=("dv", "sum"), pm_bars=("dv", "size"),
            pm_last=("close", "last"), pm_high=("high", "max"),
        ).reset_index()
        agg["date"] = pd.Timestamp(day)
        rows.append(agg)
    logger.info("premarket sessions read: %d (%d missing)", len(rows), missing)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def report(df: pd.DataFrame, mask: np.ndarray, label: str,
           col: str = "o2c") -> Dict:
    n = int(mask.sum())
    if n < 100:
        logger.info("  %-52s n=%-6d (too few)", label, n)
        return {}
    sub = df[mask]
    est = pstats.cluster_bootstrap_mean(sub[col], sub["date"], n_boot=3_000)
    flag = "  <<<" if est.lo > 0 else ""
    logger.info("  %-52s n=%-6s %+7.3f%%  CI [%+.3f%%, %+.3f%%]  hit %.1f%%%s",
                label, f"{n:,}", 100 * est.mean, 100 * est.lo, 100 * est.hi,
                100 * float((sub[col] > 0).mean()), flag)
    return {"label": label, "n": n, "mean": float(est.mean),
            "lo": float(est.lo), "hi": float(est.hi),
            "p_greater": float(est.p_value_greater),
            "hit_rate": float((sub[col] > 0).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Premarket-volume hypothesis test.")
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "hypothesis_premarket.json"))
    args = ap.parse_args(argv)

    u = pd.read_parquet(bars.DATA_ROOT / "forensic_universe.parquet")
    u = u[~u["any_split"]]
    pool = u[(u["adv20"] >= ADV_MIN) & (u["adv20"] < ADV_MAX)].copy()
    logger.info("pool: %s symbol-days | %d sessions | %s tickers",
                f"{len(pool):,}", pool["date"].nunique(),
                f"{pool['symbol'].nunique():,}")

    pm = build_premarket_panel(pool[["symbol", "date"]])
    if pm.empty:
        raise SystemExit("no premarket data")

    d = pool.merge(pm, on=["symbol", "date"], how="left")
    d["pm_dv"] = d["pm_dv"].fillna(0.0)
    d["pm_bars"] = d["pm_bars"].fillna(0).astype(int)
    d["pm_dv_to_adv"] = d["pm_dv"] / d["adv20"].replace(0, np.nan)
    d = d.dropna(subset=["o2c", "pm_dv_to_adv"])
    logger.info("joined: %s symbol-days with premarket + outcome", f"{len(d):,}")

    results: List[Dict] = []
    logger.info("=" * 100)
    base = report(d, np.ones(len(d), dtype=bool), "WHOLE POOL (no condition)")
    results.append(base)

    logger.info("-" * 100)
    logger.info("open->close by premarket dollar volume / ADV")
    for lo, hi, lab in [(0, 0.01, "pm/adv < 0.01 (no pre-open trade)"),
                        (0.01, 0.1, "pm/adv 0.01 - 0.1"),
                        (0.1, 1.0, "pm/adv 0.1 - 1"),
                        (1.0, 5.0, "pm/adv 1 - 5"),
                        (5.0, 20.0, "pm/adv 5 - 20"),
                        (20.0, np.inf, "pm/adv >= 20")]:
        m = ((d["pm_dv_to_adv"] >= lo) & (d["pm_dv_to_adv"] < hi)).to_numpy()
        r = report(d, m, lab)
        if r:
            results.append(r)

    logger.info("-" * 100)
    logger.info("the hypothesis, and the prior-decline interaction")
    hi_pm = (d["pm_dv_to_adv"] >= 1.0).to_numpy()
    results.append(report(d, hi_pm, "H: pm/adv >= 1x ADV"))
    if "ret_20d" not in d.columns:
        g = u.sort_values(["symbol", "date"]).groupby("symbol", sort=False)
        tmp = u.assign(ret_20d=g["close"].shift(1) / g["close"].shift(21) - 1.0)
        d = d.merge(tmp[["symbol", "date", "ret_20d"]], on=["symbol", "date"], how="left")
    fell = (d["ret_20d"] < -0.25).fillna(False).to_numpy()
    results.append(report(d, hi_pm & fell, "H+: pm >= 1x ADV AND fell >25% over 20d"))
    results.append(report(d, hi_pm & ~fell, "   pm >= 1x ADV AND did not fall"))
    results.append(report(d, ~hi_pm & fell, "   pm <  1x ADV AND fell >25%"))

    logger.info("=" * 100)
    winners = [r for r in results if r and r.get("lo", -1) > 0]
    if not winners:
        logger.info("NO condition produces a positive open-to-close return with a "
                    "CI excluding zero.")
    else:
        for w in winners:
            logger.info("POSITIVE: %s -> %+.3f%% CI [%+.3f%%, %+.3f%%] n=%s",
                        w["label"], 100 * w["mean"], 100 * w["lo"], 100 * w["hi"],
                        f"{w['n']:,}")

    Path(args.out).write_text(json.dumps(
        {"pool_adv_min": ADV_MIN, "pool_adv_max": ADV_MAX,
         "n_pool": int(len(d)), "results": [r for r in results if r],
         "exploratory": True}, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
