"""build_candidates.py — screen the daily panel down to the studied population.

Produces ``data/research_penny/candidates.parquet``, the denominator for every
result in the study, plus a coverage report describing what the screen kept and
what it discarded.

    python research/penny/scripts/build_candidates.py --min-gap 0.10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars, candidates as cand  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("candidates")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the candidate population.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--min-gap", type=float, default=0.10)
    parser.add_argument("--min-price", type=float, default=0.50)
    parser.add_argument("--max-price", type=float, default=10.00)
    parser.add_argument("--min-adv", type=float, default=100_000.0)
    parser.add_argument("--max-per-day", type=int, default=None)
    # Round 4 rebuilds the funnel at a different screen. Writing to a tagged
    # path keeps round 3's artefacts byte-identical, so its published numbers
    # stay reproducible from the same repo state.
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args(argv)

    years = list(range(args.start_year, args.end_year + 1))
    logger.info("loading daily panel for %s ...", years)
    panel = bars.load_years(years)
    if panel.empty:
        raise SystemExit("no daily bars cached; run fetch_daily_history first")

    logger.info(
        "panel: %s rows, %d symbols, %s .. %s",
        f"{len(panel):,}", panel["symbol"].nunique(),
        panel["date"].min().date(), panel["date"].max().date(),
    )

    missing_unadj = panel["close_unadj"].isna().mean()
    logger.info("unadjusted close missing on %.2f%% of rows", 100 * missing_unadj)

    logger.info("computing cross-day features ...")
    panel = bars.add_cross_day_features(panel)

    spec = cand.CandidateSpec(
        min_gap=args.min_gap,
        min_price=args.min_price,
        max_price=args.max_price,
        min_adv_20=args.min_adv,
        max_per_day=args.max_per_day,
    )
    selected = cand.build_candidates(panel, spec)
    selected = cand.add_delisting_distance(selected, panel)

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = bars.DATA_ROOT / f"candidates{suffix}.parquet"
    selected.to_parquet(out_path, engine="pyarrow", index=False, compression="snappy")

    per_year = (
        selected.assign(year=selected["date"].dt.year)
        .groupby("year")
        .agg(candidates=("symbol", "size"), sessions=("date", "nunique"),
             symbols=("symbol", "nunique"))
    )
    logger.info("candidates by year:\n%s", per_year.to_string())

    report = {
        "spec": vars(spec),
        "panel_rows": int(len(panel)),
        "panel_symbols": int(panel["symbol"].nunique()),
        "candidates": int(len(selected)),
        "sessions": int(selected["date"].nunique()),
        "unique_symbols": int(selected["symbol"].nunique()),
        "per_year": per_year.to_dict(orient="index"),
        "pct_unadj_missing": float(missing_unadj),
        "date_min": str(selected["date"].min().date()),
        "date_max": str(selected["date"].max().date()),
    }
    (bars.DATA_ROOT / f"candidates_report{suffix}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    logger.info("wrote %s (%s rows)", out_path, f"{len(selected):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
