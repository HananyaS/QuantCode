"""fetch_xbrl.py — shares outstanding via the XBRL frames API.

One request per quarter across all filers, rather than one large payload per
company: ~44 requests instead of ~2,600.

Each value is dated by the acceptance timestamp of the periodic report that
actually disclosed it, taken from the EDGAR filing index already on disk. That
is what makes the resulting feature point-in-time.

    python research/penny/scripts/fetch_xbrl.py
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

from research.penny.etl import bars, xbrl  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("xbrl")

RAW_PATH = bars.DATA_ROOT / "xbrl_shares.parquet"
FEATURES_PATH = bars.DATA_ROOT / "xbrl_features.parquet"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch shares outstanding.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--candidates", default=str(bars.DATA_ROOT / "candidates.parquet"))
    parser.add_argument("--features-out", default=str(FEATURES_PATH))
    args = parser.parse_args(argv)

    candidates = pd.read_parquet(args.candidates)
    filings = pd.read_parquet(bars.DATA_ROOT / "edgar_filings.parquet")
    ticker_map = pd.read_parquet(bars.DATA_ROOT / "edgar_ticker_map.parquet")

    # Rebuild when the cache does not cover the candidate symbols. The frames
    # API is one request per quarter across all filers -- ~44 requests -- so a
    # full rebuild is cheap and there is no reason to merge partial state. The
    # cache is keyed on nothing but its own existence, which was fine until
    # round 4 rebuilt the candidate screen.
    wanted = {s.upper() for s in candidates["symbol"].unique()}
    rebuild = args.force or not RAW_PATH.exists()
    if not rebuild:
        dated = pd.read_parquet(RAW_PATH)
        missing = wanted - set(dated["symbol"].unique())
        logger.info("share counts: %s rows cached, %d candidate symbols missing",
                    f"{len(dated):,}", len(missing))
        # Many microcaps genuinely never report the DEI tag, so a nonzero gap is
        # expected; refetch only when it is large enough to be a cache artefact.
        rebuild = len(missing) > 0.05 * len(wanted)
        if rebuild:
            logger.info("gap exceeds 5%% of candidates -- refetching frames")

    if rebuild:
        periods = xbrl.quarter_frames(args.start_year, args.end_year)
        logger.info("fetching %d quarterly frames ...", len(periods))
        t0 = time.time()
        raw = xbrl.fetch_all_frames(periods)
        if raw.empty:
            raise SystemExit("no XBRL frames retrieved")
        logger.info("raw frames: %s rows in %.0fs", f"{len(raw):,}", time.time() - t0)

        dated = xbrl.attach_known_from(raw, filings, ticker_map)
        if dated.empty:
            raise SystemExit("no share counts matched a candidate symbol")
        dated.to_parquet(RAW_PATH, index=False)
        logger.info("wrote %s (%s rows, %d symbols)",
                    RAW_PATH, f"{len(dated):,}", dated["symbol"].nunique())

    features = xbrl.build_pit_features(candidates, dated)
    features.to_parquet(args.features_out, index=False)

    covered = features["shares_outstanding"].notna()
    summary = {
        "share_count_rows": int(len(dated)),
        "symbols_with_counts": int(dated["symbol"].nunique()),
        "candidate_rows": int(len(features)),
        "with_shares": int(covered.sum()),
        "coverage": float(covered.mean()),
        "median_shares": float(features["shares_outstanding"].median()),
        "median_float_dollar": float(features["float_dollar"].median()),
        "median_growth_qoq": float(features["shares_growth_qoq"].median()),
        "pct_diluting_qoq": float((features["shares_growth_qoq"] > 0.05).mean()),
    }
    (bars.DATA_ROOT / "xbrl_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info("coverage %.1f%% | median shares %.0f | median float $%.0f",
                100 * summary["coverage"], summary["median_shares"],
                summary["median_float_dollar"])
    logger.info("share count grew >5%% QoQ on %.1f%% of candidate days",
                100 * summary["pct_diluting_qoq"])
    logger.info("wrote %s", args.features_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
