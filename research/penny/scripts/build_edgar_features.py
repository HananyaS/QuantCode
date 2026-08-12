"""build_edgar_features.py — point-in-time filing features for a candidate set.

Round 3 produced ``edgar_features.parquet`` ad hoc, with no script in the repo
that regenerates it. That is a reproducibility hole: the file feeding the
largest claimed lift of round 2 could not be rebuilt from the codebase. This
closes it, and lets round 4 build the same features against its own funnel.

The point-in-time rule lives in ``etl.edgar.build_pit_features``: a filing
counts only if its ``acceptanceDateTime`` is strictly before the 09:35 decision
on the candidate's own date. Filing *date* is not sufficient -- an 18:00
acceptance carries the same filing date as the morning that preceded it.

    python research/penny/scripts/build_edgar_features.py \
        --candidates data/research_penny/candidates_r4.parquet \
        --out data/research_penny/edgar_features_r4.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars, edgar  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edgar_features")

FILINGS_PATH = bars.DATA_ROOT / "edgar_filings.parquet"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build point-in-time EDGAR features.")
    parser.add_argument("--candidates", default=str(bars.DATA_ROOT / "candidates.parquet"))
    parser.add_argument("--out", default=str(bars.DATA_ROOT / "edgar_features.parquet"))
    parser.add_argument("--filings", default=str(FILINGS_PATH))
    args = parser.parse_args(argv)

    cand_path, filings_path = Path(args.candidates), Path(args.filings)
    if not cand_path.exists():
        raise SystemExit(f"{cand_path} missing; run build_candidates first")
    if not filings_path.exists():
        raise SystemExit(f"{filings_path} missing; run fetch_edgar first")

    candidates = pd.read_parquet(cand_path)
    filings = pd.read_parquet(filings_path)
    logger.info("candidates %s | filings %s (%d symbols)",
                f"{len(candidates):,}", f"{len(filings):,}",
                filings["symbol"].nunique())

    features = edgar.build_pit_features(candidates, filings)
    assert len(features) == len(candidates), "feature build changed the row count"

    # Spot-check causality on real data rather than trusting the unit tests
    # alone: a leak here would look like an excellent feature, which is exactly
    # what round 2 mistook for a 10x improvement.
    edgar.assert_no_future_filings(candidates, filings)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)

    covered = features["edgar_has_coverage"].fillna(False)
    summary = {
        "candidates": int(len(candidates)),
        "with_coverage": int(covered.sum()),
        "coverage": float(covered.mean()),
        "pct_deficiency_180d": float(features["edgar_deficiency_180d"].fillna(False).mean()),
        "pct_dilutive_8k_180d": float(features["edgar_dilutive_8k_180d"].fillna(False).mean()),
        "median_days_since_shelf": float(features["edgar_days_since_shelf"].median()),
    }
    Path(args.out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("coverage %.1f%% | deficiency %.1f%% | dilutive 8-K %.1f%%",
                100 * summary["coverage"], 100 * summary["pct_deficiency_180d"],
                100 * summary["pct_dilutive_8k_180d"])
    logger.info("wrote %s (%s rows)", args.out, f"{len(features):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
