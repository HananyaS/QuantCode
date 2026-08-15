"""long_reversion_stops.py — should the long strategy carry a stop loss?

The calibrated rule (§65) has **no stop**. Its only exits are the +50% limit and
the close of the tenth session, so a position's downside is bounded by nothing
except the stock reaching zero. That was never a decision -- `build_trades`
takes `stop=None` by default and the portfolio sweep varied drop, hold, target
and slot count while leaving the stop untouched. This file closes that gap.

Why a stop is not obviously good here
--------------------------------------
The strategy is a convex bet: 27% of trades reach +50% and the rest expire near
flat. A stop truncates the left tail, which sounds strictly good, but these
names routinely fall 30% and then double. **Every stop that fires on a name that
would have recovered converts a winner into a realised loss**, and the winners
are the entire edge -- dropping the best 50 of 421 trades already turns the
result negative.

So this is a genuine trade-off, not a free improvement, and the answer has to
come from the data.

Ordering within a day is unknowable from daily bars, so when the +50% high and
the stop's low both occur in one session the **stop is assumed to fill first**.
That is the pessimistic assignment, and it biases this test *against* the stop's
apparent value only in the sense of assigning it more fills -- which is the
conservative direction for deciding whether to add one.

    python research/penny/scripts/long_reversion_stops.py
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
from research.penny.scripts.hypothesis_grid import OOS_START, build_panel  # noqa: E402
from research.penny.scripts.long_reversion_portfolio import (  # noqa: E402
    build_trades, simulate,
)
from research.penny.scripts.long_reversion_verify import select  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("stops")

HOLD, TARGET, SLOTS = 10, 0.50, 30
STOPS = [None, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "long_reversion_stops.json"))
    args = ap.parse_args(argv)

    u = build_panel()
    calendar = np.sort(u.loc[(u["adv20"] >= 2.5e5).to_numpy(), "date"].unique())
    rows: List[Dict] = []

    logger.info("=" * 124)
    logger.info("STOP LOSS SWEEP -- drop>=40%%, hold %d, target +%.0f%%, entry next open, "
                "%d slots", HOLD, 100 * TARGET, SLOTS)
    logger.info("-" * 124)
    logger.info("  %-10s %6s %8s %22s %8s %7s %7s %9s %8s", "stop", "n",
                "per trade", "95% CI", "median", "target", "stopped", "compound", "maxDD")

    for stop in STOPS:
        tr = build_trades(u, HOLD, TARGET, stop=stop, entry_at="next_open")
        sel = select(u, tr)
        if len(sel) < 60:
            continue
        e = pstats.cluster_bootstrap_mean(sel["net"], sel["date"], n_boot=3_000)
        sm = simulate(sel, calendar, SLOTS)
        stopped = float((sel["net"] <= -(stop or 9.9) + 1e-9).mean()) if stop else 0.0
        label = "none" if stop is None else f"-{100*stop:.0f}%"
        mark = "  <-- as tested" if stop is None else ""
        logger.info("  %-10s %6d %+8.2f%% [%+8.2f%%, %+8.2f%%] %+7.2f%% %6.0f%% %7.0f%% "
                    "%+8.1f%% %7.1f%%%s",
                    label, len(sel), 100 * e.mean, 100 * e.lo, 100 * e.hi,
                    100 * sel["net"].median(), 100 * sel["took_target"].mean(),
                    100 * stopped, 100 * sm.total_return, 100 * sm.max_drawdown, mark)
        rows.append({"stop": stop, "n": len(sel), "mean": float(e.mean),
                     "lo": float(e.lo), "hi": float(e.hi),
                     "median": float(sel["net"].median()),
                     "target_rate": float(sel["took_target"].mean()),
                     "stop_rate": stopped, "sharpe": sm.sharpe,
                     "total_return": sm.total_return,
                     "max_drawdown": sm.max_drawdown,
                     "worst_trade": float(sel["net"].min())})

    logger.info("-" * 124)
    logger.info("WORST SINGLE TRADE, and what a stop would have capped it at")
    for r in rows:
        lab = "none" if r["stop"] is None else f"-{100*r['stop']:.0f}%"
        logger.info("  stop %-6s worst trade %+7.1f%%   Sharpe %5.2f", lab,
                    100 * r["worst_trade"], r["sharpe"])

    logger.info("-" * 124)
    logger.info("PERIOD CHECK on the best stop vs no stop")
    best = max(rows, key=lambda r: r["total_return"])
    for stop in ({None, best["stop"]}):
        tr = build_trades(u, HOLD, TARGET, stop=stop, entry_at="next_open")
        sel = select(u, tr)
        lab = "none" if stop is None else f"-{100*stop:.0f}%"
        for pl, per in (("IS ", sel["date"] < OOS_START), ("OOS", sel["date"] >= OOS_START)):
            sub = sel[per.to_numpy()]
            if len(sub) < 50:
                continue
            e = pstats.cluster_bootstrap_mean(sub["net"], sub["date"], n_boot=2_000)
            logger.info("  stop %-6s %s n=%-5d %+6.2f%%  CI [%+.2f%%, %+.2f%%]",
                        lab, pl, len(sub), 100 * e.mean, 100 * e.lo, 100 * e.hi)

    logger.info("=" * 124)
    Path(args.out).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
