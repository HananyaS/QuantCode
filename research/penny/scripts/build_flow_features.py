"""build_flow_features.py — insider and institutional flow for a candidate set.

Uses the EDGAR forms that rounds 1-4 filtered out: 568,896 Form 4, 73,880
SC 13G, 22,580 SC 13D, 56,299 Form 3. See ``etl.filings_flow`` for why they were
excluded (a supply question) and why they are worth including now (a demand one).

    python research/penny/scripts/build_flow_features.py \
        --candidates data/research_penny/candidates_r4.parquet \
        --out data/research_penny/flow_features_r4.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars, filings_flow as FF  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("flow")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build insider/institutional flow features.")
    parser.add_argument("--candidates", default=str(bars.DATA_ROOT / "candidates_r4.parquet"))
    parser.add_argument("--out", default=str(bars.DATA_ROOT / "flow_features_r4.parquet"))
    parser.add_argument("--filings", default=str(bars.DATA_ROOT / "edgar_filings.parquet"))
    args = parser.parse_args(argv)

    candidates = pd.read_parquet(args.candidates, columns=["symbol", "date"])
    filings = pd.read_parquet(args.filings)
    logger.info("candidates %s | filings %s", f"{len(candidates):,}", f"{len(filings):,}")

    out = FF.build_flow_features(candidates, filings)

    # Spot-check causality on real data, on a symbol that actually has filings.
    checked = 0
    for _, row in out.dropna(subset=["days_since_insider_txn"]).head(300).iterrows():
        FF.assert_no_future_flow(out, filings, row["symbol"], row["date"])
        checked += 1
    logger.info("causality spot check: %d rows, 0 offenders", checked)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    cov = out["flow_has_coverage"].mean()
    summary = {
        "rows": int(len(out)),
        "coverage": float(cov),
        "pct_insider_txn_30d": float((out["insider_txn_30d"].fillna(0) > 0).mean()),
        "pct_activist_365d": float((out["activist_365d"].fillna(0) > 0).mean()),
        "pct_passive_365d": float((out["passive_365d"].fillna(0) > 0).mean()),
        "median_days_since_insider_txn": float(out["days_since_insider_txn"].median()),
        "pct_delist_notice": float(out["days_since_delist_notice"].notna().mean()),
    }
    Path(args.out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("coverage %.1f%% | insider txn in 30d on %.1f%% of rows | "
                "activist 365d %.1f%% | passive 365d %.1f%%",
                100 * cov, 100 * summary["pct_insider_txn_30d"],
                100 * summary["pct_activist_365d"], 100 * summary["pct_passive_365d"])
    logger.info("wrote %s (%s rows)", args.out, f"{len(out):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
