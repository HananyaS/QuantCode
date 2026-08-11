"""build_barrier_free.py — whole-session path statistics for every candidate.

Produces the model's dependent variable. Separate from ``build_dataset`` because
these quantities do not depend on barrier geometry, so they are computed once
and reused across every configuration -- and because conflating them with the
trade-level statistics is exactly the error round 1 made.

    python research/penny/scripts/build_barrier_free.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.labels import barrier_free as bf  # noqa: E402
from research.penny.scripts.fetch_intraday import session_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("barrier_free")

OUT_PATH = bars.DATA_ROOT / "barrier_free.parquet"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compute barrier-free path stats.")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)

    cand_path = bars.DATA_ROOT / "candidates.parquet"
    if not cand_path.exists():
        raise SystemExit("candidates.parquet missing")

    candidates = pd.read_parquet(cand_path, columns=["symbol", "date"])
    wanted = candidates.groupby("date")["symbol"].apply(set)
    days = sorted(candidates["date"].unique())

    frames: List[pd.DataFrame] = []
    missing = 0

    for i, day in enumerate(days, start=1):
        day_ts = pd.Timestamp(day)
        path = session_path(day_ts)
        if not path.exists():
            missing += 1
            continue

        minute = pd.read_parquet(path)
        minute = minute[minute["symbol"].isin(wanted.loc[day])]
        if minute.empty:
            continue

        stats = bf.path_stats_day(minute, day_ts)
        if not stats.empty:
            frames.append(stats)

        if i % 400 == 0:
            logger.info("%d/%d sessions (%s rows)", i, len(days),
                        f"{sum(len(f) for f in frames):,}")

    if missing:
        logger.warning("%d sessions had no cached minute data", missing)
    if not frames:
        raise SystemExit("no path statistics produced")

    out = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, engine="pyarrow", index=False, compression="snappy")

    summary = {
        "rows": int(len(out)),
        "symbols": int(out["symbol"].nunique()),
        "sessions": int(out["date"].nunique()),
        "mean_ret_c2c": float(out["ret_c2c"].mean()),
        "median_ret_c2c": float(out["ret_c2c"].median()),
        "p_mfe_ge": {
            f"{x:.0%}": float((out["mfe_full"] >= x).mean())
            for x in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
        },
        "p_mae_le": {
            f"-{x:.0%}": float((out["mae_full"] <= -x).mean())
            for x in (0.05, 0.10, 0.20, 0.30, 0.50)
        },
        "p_high_before_low": float(out["mfe_before_mae"].mean()),
    }
    (bars.DATA_ROOT / "barrier_free_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info("wrote %s (%s rows)", args.out, f"{len(out):,}")
    logger.info("mean hold-to-close (gross) %.4f", summary["mean_ret_c2c"])
    logger.info("P(MFE >= 20%%) = %.4f   P(MFE >= 100%%) = %.4f",
                summary["p_mfe_ge"]["20%"], summary["p_mfe_ge"]["100%"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
