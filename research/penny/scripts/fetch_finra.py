"""fetch_finra.py — consolidated NMS daily short-sale volume.

Filtered to candidate symbols during the fetch: the raw files carry ~12,000
symbols a day and this study needs ~3,000, so filtering early keeps the cache
small enough to commit alongside the other derived artifacts.

Coverage note recorded in the summary: the consolidated series begins
2018-08-01, while the study starts 2016-01. Roughly a fifth of candidate
sessions therefore have no short-volume data at all, and those rows must read
as "unknown" rather than "zero".

    python research/penny/scripts/fetch_finra.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars, finra  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finra")

RAW_PATH = bars.DATA_ROOT / "finra_short_volume.parquet"
FEATURES_PATH = bars.DATA_ROOT / "finra_features.parquet"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch FINRA short volume.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--candidates", default=str(bars.DATA_ROOT / "candidates.parquet"))
    parser.add_argument("--features-out", default=str(FEATURES_PATH))
    args = parser.parse_args(argv)

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        raise SystemExit(f"{cand_path} missing")

    candidates = pd.read_parquet(cand_path, columns=["symbol", "date"])
    symbols = sorted(candidates["symbol"].unique())
    start = candidates["date"].min().date()
    end = candidates["date"].max().date()
    logger.info("candidates: %s rows, %d symbols, %s..%s",
                f"{len(candidates):,}", len(symbols), start, end)

    # Incremental by symbol. The cache was filtered to whichever screen built
    # it, so a rebuilt candidate set leaves symbols uncovered -- and a
    # file-keyed early return would report "cached" while giving them nothing.
    # The daily files contain every symbol, so this costs the same downloads as
    # a full refetch; it is the parse and the merge that are saved.
    cached = pd.DataFrame()
    missing = sorted({s.upper() for s in symbols})
    if RAW_PATH.exists() and not args.force:
        cached = pd.read_parquet(RAW_PATH)
        missing = sorted({s.upper() for s in symbols} - set(cached["symbol"].unique()))
        logger.info("short volume: %s rows cached, %d symbols missing",
                    f"{len(cached):,}", len(missing))

    if missing:
        t0 = time.time()
        fresh = finra.fetch_range(start, end, symbols=missing)
        if fresh.empty and cached.empty:
            raise SystemExit("no short-volume data retrieved")
        # Plain concat, deliberately no drop_duplicates. `missing` is the
        # cache's symbol complement, so the two frames are disjoint and there is
        # nothing to dedupe -- while FINRA's consolidated files legitimately
        # carry several rows per (symbol, date), one per market, which
        # build_lagged_features sums. A content-based dedupe here would silently
        # destroy those and understate short volume.
        if not cached.empty and not fresh.empty:
            overlap = set(cached["symbol"].unique()) & set(fresh["symbol"].unique())
            assert not overlap, f"cached and fresh symbol sets overlap: {sorted(overlap)[:5]}"
        raw = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
        raw.to_parquet(RAW_PATH, index=False)
        logger.info("wrote %s (%s rows, +%s new, %.0fs)", RAW_PATH,
                    f"{len(raw):,}", f"{len(fresh):,}", time.time() - t0)
    else:
        raw = cached

    features = finra.build_lagged_features(candidates, raw)
    features.to_parquet(args.features_out, index=False)

    # Spot-check the lag on a real symbol-day rather than trusting the unit
    # tests alone: a leak here would look like an excellent feature.
    checked = 0
    for _, row in features.dropna(subset=["short_volume_ratio"]).head(200).iterrows():
        finra.assert_short_volume_is_lagged(features, raw, row["symbol"], row["date"])
        checked += 1
    logger.info("lag tripwire passed on %d real symbol-days", checked)

    covered = features["short_volume_ratio"].notna()
    in_window = features["date"] >= pd.Timestamp(finra.HISTORY_START)
    summary = {
        "raw_rows": int(len(raw)),
        "sessions": int(raw["date"].nunique()),
        "symbols": int(raw["symbol"].nunique()),
        "candidate_rows": int(len(features)),
        "with_short_ratio": int(covered.sum()),
        "coverage_overall": float(covered.mean()),
        "coverage_since_2018_08": float(covered[in_window].mean()),
        "rows_before_history_start": int((~in_window).sum()),
        "median_short_ratio": float(features["short_volume_ratio"].median()),
    }
    (bars.DATA_ROOT / "finra_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("coverage overall %.1f%% | since 2018-08 %.1f%% | median ratio %.3f",
                100 * summary["coverage_overall"],
                100 * summary["coverage_since_2018_08"],
                summary["median_short_ratio"])
    logger.info("wrote %s", args.features_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
