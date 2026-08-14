"""short_side.py — the profitable side of every reliable effect found.

Why this exists
---------------
Six rounds and a 441-cell sweep found no profitable long. But they did find
several effects that are *reliably negative*, with tight intervals and large
samples:

    reverse splits (open -> close)          -15.22%   [-17.64%, -12.82%]
    premarket dollar volume >= 20x ADV       -2.93%   [ -4.30%,  -1.51%]
    gap up >= +5%, buy the open              -2.06%   gross, 10,108 obs
    >=3 shelf registrations in 180 days      -0.60%   [ -0.82%,  -0.38%]
    SEC filing accepted the same morning     -0.40%   [ -0.60%,  -0.19%]

Every one of those is a *short* signal that the study has only ever used as a
veto, because the long-only constraint was fixed early and never revisited. It
should have been: the strongest evidence in the whole project points one way and
the search was pointed the other.

What shorting actually costs here, and why it may still not work
----------------------------------------------------------------
The gross numbers above are not the trade. A short in this universe pays:

* the same round-trip spread as a long (measured, ~60-150 bps by band);
* a **borrow fee**, which for hard-to-borrow microcaps is not the 0.3% of a
  large cap. Rates of 20-100% annualised are routine and 300%+ occurs;
* **locate risk** -- much of this universe is simply unborrowable, so a
  backtest that assumes every name can be shorted is measuring a portfolio
  nobody could hold;
* **buy-in risk** -- the lender can recall at the worst moment, which is exactly
  when a squeeze is running.

The first two are modelled below. The third and fourth cannot be, because IBKR's
borrow feed is unreachable from this network (§probe, round 0) and no free
substitute exists. That is stated as a hard limitation rather than assumed away:
**a positive result here is an upper bound, not a strategy.**

    python research/penny/scripts/short_side.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.hypothesis_grid import OOS_START, build_panel  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("short")

# Annualised borrow, charged per calendar day held. Microcap hard-to-borrow
# rates are the rule, not the exception; 50% is a moderate central assumption
# and the sweep below reports 20% / 50% / 150% so the reader can pick.
BORROW_ANNUAL = [0.20, 0.50, 1.50]
DAYS_PER_YEAR = 252.0


def short_return(gross_long: pd.Series, cost: pd.Series, days: int,
                 borrow_annual: float) -> pd.Series:
    """Return to a short: the negated move, less spread, less borrow."""
    return -gross_long - cost - borrow_annual * days / DAYS_PER_YEAR


def report(u: pd.DataFrame, mask: np.ndarray, label: str, col: str,
           days: int, borrow: float, extra: Optional[Dict] = None) -> Dict:
    m = mask & u[col].notna().to_numpy()
    n = int(m.sum())
    if n < 150:
        logger.info("  %-46s n=%-6d (too few)", label, n)
        return {}
    s = u[m]
    r = short_return(s[col], s["cost"], days, borrow)
    est = pstats.cluster_bootstrap_mean(r, s["date"], n_boot=2_000)
    star = "  <<<" if est.lo > 0 else ""
    logger.info("  %-46s n=%-6s short %+7.2f%%  CI [%+.2f%%, %+.2f%%]  hit %.0f%%  "
                "median %+6.2f%%%s",
                label, f"{n:,}", 100 * est.mean, 100 * est.lo, 100 * est.hi,
                100 * float((r > 0).mean()), 100 * float(r.median()), star)
    out = {"label": label, "n": n, "mean": float(est.mean), "lo": float(est.lo),
           "hi": float(est.hi), "hit": float((r > 0).mean()),
           "median": float(r.median()), "n_days": int(est.n_days),
           "borrow": borrow}
    if extra:
        out.update(extra)
    return out


def concentration(u: pd.DataFrame, mask: np.ndarray, col: str, days: int,
                  borrow: float) -> Dict:
    """The check that killed both long candidates: is it a handful of trades?"""
    s = u[mask & u[col].notna().to_numpy()]
    r = short_return(s[col], s["cost"], days, borrow).sort_values()
    if len(r) < 20:
        return {}
    return {"mean": float(r.mean()), "median": float(r.median()),
            "drop_top3": float(r.iloc[:-3].mean()),
            "drop_top5": float(r.iloc[:-5].mean()),
            "drop_top10": float(r.iloc[:-10].mean()),
            "worst": float(r.iloc[0]), "n": int(len(r))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Short-side tests.")
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "short_side.json"))
    args = ap.parse_args(argv)

    u = build_panel()
    split = u.get("any_split", u["is_suspect_split"]).to_numpy()
    clean = ~split
    liquid = (u["adv20"] >= 2.5e5).to_numpy()
    results: Dict[str, List[Dict]] = {}

    logger.info("=" * 122)
    logger.info("SHORT SIDE -- enter at the open, cover at the close (same session)")
    logger.info("negative numbers below are LOSSES to the short")

    for borrow in BORROW_ANNUAL:
        logger.info("-" * 122)
        logger.info("borrow %.0f%% annualised", 100 * borrow)
        rows: List[Dict] = []
        rows.append(report(u, split, "short a suspected reverse split", "o2c", 1, borrow))
        rows.append(report(u, split & liquid, "  ... and ADV >= $250k", "o2c", 1, borrow))
        for a in (0.05, 0.10, 0.20, 0.50):
            rows.append(report(u, clean & liquid & (u["gap"] >= a).to_numpy(),
                               f"short gap up >= +{100*a:.0f}%", "o2c", 1, borrow))
        rows.append(report(u, clean & liquid, "CONTROL: short every liquid day",
                           "o2c", 1, borrow))
        results[f"borrow_{borrow}"] = [r for r in rows if r]

    # The headline cell, examined the way the long candidates were.
    logger.info("=" * 122)
    logger.info("FORENSICS on the strongest cell (short reverse splits, borrow 50%)")
    c = concentration(u, split, "o2c", 1, 0.50)
    if c:
        logger.info("  n=%d  mean %+.2f%%  median %+.2f%%", c["n"],
                    100 * c["mean"], 100 * c["median"])
        logger.info("  drop best 3 trades : %+.2f%%", 100 * c["drop_top3"])
        logger.info("  drop best 5 trades : %+.2f%%", 100 * c["drop_top5"])
        logger.info("  drop best 10 trades: %+.2f%%", 100 * c["drop_top10"])
        logger.info("  worst single trade : %+.2f%%", 100 * c["worst"])
    results["concentration"] = [c] if c else []

    logger.info("-" * 122)
    logger.info("period split (short reverse splits, borrow 50%)")
    rows = []
    for lab, per in (("in-sample", (u["date"] < OOS_START).to_numpy()),
                     ("out-of-sample", (u["date"] >= OOS_START).to_numpy())):
        rows.append(report(u, split & per, lab, "o2c", 1, 0.50))
    results["period"] = [r for r in rows if r]

    logger.info("-" * 122)
    logger.info("holding longer (short reverse splits, borrow 50%)")
    rows = []
    for h in (1, 2, 3, 5):
        col = f"fwd_open_{h}"
        rows.append(report(u, split, f"cover after {h} session(s)", col, h, 0.50))
    results["horizon"] = [r for r in rows if r]

    logger.info("=" * 122)
    logger.info("LIMITATION: borrow availability is NOT modelled. IBKR's feed is")
    logger.info("unreachable from this network and no free substitute exists, so every")
    logger.info("number above assumes a locate that may not exist. Treat as an upper bound.")
    logger.info("=" * 122)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
