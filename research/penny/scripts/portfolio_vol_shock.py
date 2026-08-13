"""portfolio_vol_shock.py — would an account have made money on this signal?

Round 6 established the trap this file exists to avoid: a rule with a *positive
arithmetic mean per trade* compounded to **-15%** with a 98% drawdown, because
daily volatility was ~50x the edge and sigma^2/2 exceeded mu. Per-trade
expectancy is not what an account earns.

So the accumulation signal -- volume >= Nx its 20-day average while the close
moves less than 5%, hold one session -- is re-evaluated as a portfolio: equal
weight across whatever fires each day, flat on days nothing fires, compounded.

The signal's per-trade profile makes this test essential rather than routine:
it wins only **39%** of the time, so the mean lives in a right tail, and a
right-tailed distribution is exactly where the arithmetic and geometric means
diverge most.

Sat-out days count as flat, never dropped -- dropping them would report the
strategy's return only on days it chose to act.

    python research/penny/scripts/portfolio_vol_shock.py
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
from research.penny.eval import portfolio as PF, stats as pstats  # noqa: E402
from research.penny.scripts.hypothesis_grid import OOS_START, build_panel  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("pf")


def run(u: pd.DataFrame, vol: float, move: float, advmin: float,
        caps: List, label: str) -> List[Dict]:
    m = ((~u["is_suspect_split"]).to_numpy()
         & (u["adv20"] >= advmin).to_numpy()
         & (u["vol_shock"] >= vol).fillna(False).to_numpy()
         & (u["c2c"].abs() <= move).fillna(False).to_numpy()
         & u["fwd_close_1"].notna().to_numpy())
    sel = u[m].copy()
    sel["net_return"] = sel["fwd_close_1"] - sel["cost"]
    calendar = np.sort(u.loc[(~u["is_suspect_split"]).to_numpy(), "date"].unique())

    rows: List[Dict] = []
    for cap in caps:
        res = PF.evaluate_portfolio(sel, "net_return", max_positions=cap,
                                    all_sessions=calendar)
        if res is None:
            continue
        d = res.to_dict()
        d.update({"variant": label, "cap": cap if cap else 0})
        rows.append(d)
        logger.info("  %-30s cap=%-4s days=%-5d trades=%-6d avg pos %.1f  "
                    "mean/day %+.4f%%  Sharpe %5.2f  compounded %+9.1f%%  maxDD %6.1f%%  "
                    "days active %.0f%%",
                    label, cap if cap else "none", res.n_days, res.n_trades,
                    res.avg_positions, 100 * res.mean_daily, res.sharpe,
                    100 * res.total_return, 100 * res.max_drawdown,
                    100 * res.pct_days_active)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "portfolio_vol_shock.json"))
    args = ap.parse_args(argv)

    u = build_panel()
    out: Dict[str, List[Dict]] = {}
    caps = [3, 5, 10, None]

    logger.info("=" * 126)
    logger.info("PORTFOLIO VIEW -- equal weight per day, flat when nothing fires, compounded")
    logger.info("-" * 126)
    out["v5"] = run(u, 5.0, 0.05, 2.5e5, caps, "vol>=5x, ADV>=250k")
    logger.info("-" * 126)
    out["v7"] = run(u, 7.0, 0.05, 2.5e5, caps, "vol>=7x, ADV>=250k")
    logger.info("-" * 126)
    out["v10"] = run(u, 10.0, 0.05, 2.5e5, caps, "vol>=10x, ADV>=250k")
    logger.info("-" * 126)
    out["v5_liquid"] = run(u, 5.0, 0.05, 1e6, caps, "vol>=5x, ADV>=$1M")
    logger.info("-" * 126)
    out["v4"] = run(u, 4.0, 0.05, 2.5e5, caps, "vol>=4x, ADV>=250k")

    # Period split on the best-behaved variant.
    logger.info("-" * 126)
    logger.info("PERIOD SPLIT -- vol>=5x, ADV>=250k, cap 10")
    for lab, per in (("in-sample", u["date"] < OOS_START),
                     ("out-of-sample", u["date"] >= OOS_START)):
        sub = u[per.to_numpy()]
        rows = run(sub, 5.0, 0.05, 2.5e5, [10], f"{lab}")
        out[f"split_{lab}"] = rows

    logger.info("=" * 126)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
