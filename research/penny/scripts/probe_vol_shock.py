"""probe_vol_shock.py — stress the one cell that survived both periods.

The grid sweep produced exactly one hypothesis positive in both the calibration
and the confirmation window with intervals excluding zero:

    volume >= 5x its 20-day average WHILE the close moves less than 5%
    -> buy the close, sell the next close

    in-sample   n=836  gross +3.71%  net +2.51%  CI [+0.54%, ...]
    out-sample  n=527              net +2.96%  CI [+0.42%, ...]

That is interesting and almost certainly not yet real. It is one cell of 422; it
did not survive Benjamini-Hochberg over the grid; and a two-window agreement is
weak evidence when the windows are adjacent in time and share market regimes.

This file tries to break it. A finding worth trading should be insensitive to
arbitrary choices, so each probe perturbs one:

* **thresholds** -- if the edge exists only at exactly 5x and 5%, it is a fit
* **liquidity** -- does it survive in bands where costs are actually payable
* **cost** -- at what multiple of the assumed schedule does it die
* **concentration** -- is it a handful of days, or is it distributed
* **direction of the quiet day** -- up-quiet, down-quiet, or both
* **placebo** -- the same liquidity with no condition at all

    python research/penny/scripts/probe_vol_shock.py
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("probe")


def measure(u: pd.DataFrame, mask: np.ndarray, label: str, h: int = 1,
            cost_mult: float = 1.0) -> Dict:
    m = mask & u[f"fwd_close_{h}"].notna().to_numpy()
    n = int(m.sum())
    if n < 100:
        logger.info("  %-46s n=%-6d (too few)", label, n)
        return {}
    ret = u.loc[m, f"fwd_close_{h}"]
    net = ret - u.loc[m, "cost"] * cost_mult
    est = pstats.cluster_bootstrap_mean(net, u.loc[m, "date"], n_boot=2_000)
    star = "  <<<" if est.lo > 0 else ""
    logger.info("  %-46s n=%-6s gross %+7.2f%%  net %+7.2f%%  CI [%+.2f%%, %+.2f%%]  hit %.0f%%%s",
                label, f"{n:,}", 100 * ret.mean(), 100 * est.mean,
                100 * est.lo, 100 * est.hi, 100 * float((net > 0).mean()), star)
    return {"label": label, "n": n, "gross": float(ret.mean()),
            "net": float(est.mean), "lo": float(est.lo), "hi": float(est.hi),
            "hit": float((net > 0).mean()), "n_days": int(est.n_days)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "probe_vol_shock.json"))
    args = ap.parse_args(argv)

    u = build_panel()
    clean = (~u["is_suspect_split"]).to_numpy()
    liquid = (u["adv20"] >= 2.5e5).to_numpy()
    out: Dict[str, List[Dict]] = {}

    def base(vol=5.0, move=0.05, advmin=2.5e5):
        return (clean & (u["adv20"] >= advmin).to_numpy()
                & (u["vol_shock"] >= vol).fillna(False).to_numpy()
                & (u["c2c"].abs() <= move).fillna(False).to_numpy())

    logger.info("=" * 118)
    logger.info("BASELINE (full period, vol>=5x, |move|<=5%, ADV>=$250k, H=1)")
    out["baseline"] = [measure(u, base(), "baseline")]

    logger.info("-" * 118)
    logger.info("PROBE 1 -- threshold sensitivity (is 5x / 5% a fit?)")
    r = []
    for vol in (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0):
        r.append(measure(u, base(vol=vol), f"vol >= {vol:g}x   (|move|<=5%)"))
    for mv in (0.02, 0.03, 0.05, 0.08, 0.12):
        r.append(measure(u, base(move=mv), f"|move| <= {100*mv:g}%   (vol>=5x)"))
    out["thresholds"] = [x for x in r if x]

    logger.info("-" * 118)
    logger.info("PROBE 2 -- liquidity bands")
    r = []
    for lo, hi, lab in ((2.5e5, 1e6, "$250k-$1M"), (1e6, 5e6, "$1M-$5M"),
                        (5e6, 2e7, "$5M-$20M"), (2e7, np.inf, ">$20M")):
        m = (clean & (u["adv20"] >= lo).to_numpy() & (u["adv20"] < hi).to_numpy()
             & (u["vol_shock"] >= 5.0).fillna(False).to_numpy()
             & (u["c2c"].abs() <= 0.05).fillna(False).to_numpy())
        r.append(measure(u, m, f"ADV {lab}"))
    out["liquidity"] = [x for x in r if x]

    logger.info("-" * 118)
    logger.info("PROBE 3 -- cost sensitivity")
    r = [measure(u, base(), f"cost x{cm:g}", cost_mult=cm) for cm in (1.0, 1.5, 2.0, 3.0)]
    out["cost"] = [x for x in r if x]

    logger.info("-" * 118)
    logger.info("PROBE 4 -- holding horizon")
    r = [measure(u, base(), f"hold {h} session(s)", h=h) for h in (1, 2, 3, 5, 10)]
    out["horizon"] = [x for x in r if x]

    logger.info("-" * 118)
    logger.info("PROBE 5 -- direction of the quiet day")
    m = base()
    r = [measure(u, m & (u["c2c"] > 0).to_numpy(), "quiet day closed UP"),
         measure(u, m & (u["c2c"] <= 0).to_numpy(), "quiet day closed DOWN")]
    out["direction"] = [x for x in r if x]

    logger.info("-" * 118)
    logger.info("PROBE 6 -- concentration")
    sub = u[base() & u["fwd_close_1"].notna().to_numpy()]
    by_day = sub.groupby("date")["fwd_close_1"].mean().sort_values()
    tot = float(sub["fwd_close_1"].mean())
    trimmed = float(by_day.iloc[:-5].mean()) if len(by_day) > 20 else float("nan")
    logger.info("  sessions with a trade           : %d", len(by_day))
    logger.info("  mean gross                      : %+.2f%%", 100 * tot)
    logger.info("  after dropping the best 5 days  : %+.2f%%", 100 * trimmed)
    logger.info("  median session mean             : %+.2f%%", 100 * float(by_day.median()))
    logger.info("  share of sessions positive      : %.0f%%", 100 * float((by_day > 0).mean()))
    out["concentration"] = [{"n_days": len(by_day), "mean": tot, "trimmed5": trimmed,
                             "median_day": float(by_day.median()),
                             "pct_days_pos": float((by_day > 0).mean())}]

    logger.info("-" * 118)
    logger.info("PROBE 7 -- placebo: same liquidity, no condition")
    out["placebo"] = [measure(u, clean & liquid, "all liquid clean days")]

    logger.info("-" * 118)
    logger.info("PROBE 8 -- period stability")
    r = [measure(u, base() & (u["date"] < OOS_START).to_numpy(), "in-sample (< 2026-02)"),
         measure(u, base() & (u["date"] >= OOS_START).to_numpy(), "out-of-sample (>= 2026-02)"),
         measure(u, base() & (u["date"].dt.year == 2025).to_numpy(), "year 2025"),
         measure(u, base() & (u["date"].dt.year == 2026).to_numpy(), "year 2026")]
    out["stability"] = [x for x in r if x]
    logger.info("=" * 118)

    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
