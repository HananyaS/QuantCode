"""build_premarket_features.py — aggregate 04:00-09:34 bars into features.

Separate from the fetch so features can be recomputed without re-downloading,
and separate from the model so the leakage boundary is enforced once, in one
place, on the way in.

    python research/penny/scripts/build_premarket_features.py
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
from research.penny.features import premarket as PM  # noqa: E402
from research.penny.scripts.fetch_premarket import session_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pm_features")

OUT_PATH = bars.DATA_ROOT / "premarket_features.parquet"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build premarket features.")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--candidates", default=str(bars.DATA_ROOT / "candidates.parquet"))
    args = parser.parse_args(argv)

    candidates = pd.read_parquet(args.candidates)
    needed = ["symbol", "date", "prev_close", "adv_20"]
    daily = candidates[[c for c in needed if c in candidates.columns]].copy()

    wanted = candidates.groupby("date")["symbol"].apply(set)
    days = sorted(candidates["date"].unique())

    frames: List[pd.DataFrame] = []
    missing = empty = 0

    for i, day in enumerate(days, start=1):
        day_ts = pd.Timestamp(day)
        path = session_path(day_ts)
        if not path.exists():
            missing += 1
            continue

        panel = pd.read_parquet(path)
        if panel.empty:
            empty += 1
            continue

        panel = panel[panel["symbol"].isin(wanted.loc[day])]
        if panel.empty:
            empty += 1
            continue

        prev = daily[daily["date"] == day_ts].set_index("symbol")["prev_close"]
        feats = PM.premarket_features(panel, prev_close=prev)
        if feats.empty:
            continue
        feats["date"] = day_ts
        frames.append(feats)

        if i % 400 == 0:
            logger.info("%d/%d sessions | %s rows", i, len(days),
                        f"{sum(len(f) for f in frames):,}")

    if missing:
        logger.warning("%d sessions had no premarket file", missing)
    if empty:
        logger.info("%d sessions had a file but no candidate bars (quiet pre-open)", empty)
    if not frames:
        raise SystemExit("no premarket features produced")

    out = pd.concat(frames, ignore_index=True)
    out = PM.add_relative_features(out, daily)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, engine="pyarrow", index=False, compression="snappy")

    coverage = len(out) / max(len(candidates), 1)
    summary = {
        "rows": int(len(out)),
        "candidate_rows": int(len(candidates)),
        "coverage": float(coverage),
        "sessions": int(out["date"].nunique()),
        "median_pm_bars": float(out["pm_bars"].median()),
        "median_pm_dollar_volume": float(out["pm_dollar_volume"].median()),
        "median_pm_dv_to_adv": float(out["pm_dv_to_adv"].median()),
        "pct_with_meaningful_premarket": float((out["pm_dollar_volume"] > 50_000).mean()),
        "sessions_missing": missing,
        "sessions_empty": empty,
    }
    (bars.DATA_ROOT / "premarket_features_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info("coverage %.1f%% of candidate rows", 100 * coverage)
    logger.info("median pre-open bars %.0f | median pre-open $volume $%,.0f".replace(",", ""),
                summary["median_pm_bars"], summary["median_pm_dollar_volume"])
    logger.info("pre-open dollar volume exceeds $50k on %.1f%% of candidates",
                100 * summary["pct_with_meaningful_premarket"])
    logger.info("wrote %s (%s rows)", args.out, f"{len(out):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
