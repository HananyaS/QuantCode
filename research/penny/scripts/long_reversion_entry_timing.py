"""long_reversion_entry_timing.py — buy the signal day's close, or the next open?

The question
------------
The signal is day *t*'s close, and §65 entered at day *t+1*'s **open**. That was
chosen for defensibility, not for return: entering at the same close the signal
is computed from looks like look-ahead. But it is only *mild* look-ahead --
a market-on-close order placed at ~15:55 acts on a price a few minutes old, not
on a price from the future -- and skipping the overnight session forfeits
whatever that session pays.

So: what is the overnight move actually worth on these names, and does capturing
it survive the cost of capturing it?

What decides it
---------------
Three things, in order:

1. **The mean overnight gap** on signal days. If it is negative there is nothing
   to discuss and the open is strictly better.
2. **Its distribution.** A positive mean built from a handful of large gap-ups
   is not a reason to take overnight risk on every position -- the same
   right-tail problem that killed two earlier candidates.
3. **Execution slippage.** A market-on-close order does not fill at the printed
   close. If the advantage is 33 bps and MOC slippage is 50, the conservative
   choice was right for a reason that has nothing to do with look-ahead.

    python research/penny/scripts/long_reversion_entry_timing.py
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
logger = logging.getLogger("entry")

HOLD, TARGET, SLOTS = 10, 0.50, 30
SLIPPAGE_BPS = [0, 10, 25, 50, 75, 100]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "long_entry_timing.json"))
    args = ap.parse_args(argv)

    u = build_panel()
    calendar = np.sort(u.loc[(u["adv20"] >= 2.5e5).to_numpy(), "date"].unique())
    res: Dict[str, object] = {}

    tr_close = build_trades(u, HOLD, TARGET, entry_at="close")
    tr_open = build_trades(u, HOLD, TARGET, entry_at="next_open")
    sel_close, sel_open = select(u, tr_close), select(u, tr_open)

    # ---- 1. what the overnight session actually pays -----------------------
    g = u.groupby("symbol", sort=False)
    u["overnight_next"] = g["open"].shift(-1) / u["close"] - 1.0
    on = u.loc[sel_close.index, "overnight_next"].dropna()

    logger.info("=" * 118)
    logger.info("1. THE OVERNIGHT MOVE on signal days (close -> next open), n=%d", len(on))
    e = pstats.cluster_bootstrap_mean(on, u.loc[on.index, "date"], n_boot=3_000)
    logger.info("   mean %+.2f%%  CI [%+.2f%%, %+.2f%%]   median %+.2f%%   sd %.2f%%",
                100 * e.mean, 100 * e.lo, 100 * e.hi, 100 * on.median(), 100 * on.std())
    logger.info("   gaps UP %.0f%% of the time", 100 * (on > 0).mean())
    qs = [1, 5, 25, 50, 75, 95, 99]
    logger.info("   percentiles: %s",
                "  ".join(f"p{q}={100*on.quantile(q/100):+.1f}%" for q in qs))
    logger.info("   drop the best 10 gaps: mean %+.2f%%", 100 * on.sort_values().iloc[:-10].mean())
    res["overnight"] = {"n": len(on), "mean": float(e.mean), "lo": float(e.lo),
                        "hi": float(e.hi), "median": float(on.median()),
                        "pct_up": float((on > 0).mean()),
                        "drop_best_10": float(on.sort_values().iloc[:-10].mean())}

    # ---- 2. the two entries, head to head ---------------------------------
    logger.info("-" * 118)
    logger.info("2. HEAD TO HEAD")
    for lab, sel in (("close  (MOC)", sel_close), ("next open (MOO)", sel_open)):
        e2 = pstats.cluster_bootstrap_mean(sel["net"], sel["date"], n_boot=3_000)
        sm = simulate(sel, calendar, SLOTS)
        logger.info("   %-16s n=%-5d %+6.2f%%  CI [%+.2f%%, %+.2f%%]  target %2.0f%%  "
                    "Sharpe %5.2f  compounded %+7.1f%%  maxDD %6.1f%%",
                    lab, len(sel), 100 * e2.mean, 100 * e2.lo, 100 * e2.hi,
                    100 * sel["took_target"].mean(), sm.sharpe,
                    100 * sm.total_return, 100 * sm.max_drawdown)
        res[f"entry_{lab.split()[0]}"] = {"n": len(sel), "mean": float(e2.mean),
                                          "sharpe": sm.sharpe,
                                          "total_return": sm.total_return}

    logger.info("   period split:")
    for lab, sel in (("close", sel_close), ("open ", sel_open)):
        for pl, per in (("IS ", sel["date"] < OOS_START), ("OOS", sel["date"] >= OOS_START)):
            sub = sel[per.to_numpy()]
            e3 = pstats.cluster_bootstrap_mean(sub["net"], sub["date"], n_boot=2_000)
            logger.info("     %s %s n=%-5d %+6.2f%%  CI [%+.2f%%, %+.2f%%]",
                        lab, pl, len(sub), 100 * e3.mean, 100 * e3.lo, 100 * e3.hi)

    # ---- 3. how much slippage erases the advantage ------------------------
    logger.info("-" * 118)
    logger.info("3. MOC SLIPPAGE -- a market-on-close order does not fill at the print")
    base_open = simulate(sel_open, calendar, SLOTS)
    open_mean = float(sel_open["net"].mean())
    logger.info("   next-open benchmark: per trade %+.2f%%, compounded %+.1f%%",
                100 * open_mean, 100 * base_open.total_return)
    rows: List[Dict] = []
    for bps in SLIPPAGE_BPS:
        s = sel_close.copy()
        s["net"] = s["net"] - bps / 1e4
        sm = simulate(s, calendar, SLOTS)
        better = s["net"].mean() > open_mean
        logger.info("   close entry +%3d bps slippage: per trade %+6.2f%%  compounded %+7.1f%%  %s",
                    bps, 100 * s["net"].mean(), 100 * sm.total_return,
                    "still better than MOO" if better else "<-- MOO now wins")
        rows.append({"slippage_bps": bps, "mean": float(s["net"].mean()),
                     "total_return": sm.total_return, "beats_open": bool(better)})
    res["slippage"] = rows
    breakeven = (sel_close["net"].mean() - open_mean) * 1e4
    logger.info("   BREAK-EVEN: the close entry stays ahead up to %.0f bps of slippage", breakeven)
    res["breakeven_bps"] = float(breakeven)

    logger.info("=" * 118)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
