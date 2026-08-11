"""fetch_edgar.py — download filing indexes for every candidate symbol.

One request per symbol against ``data.sec.gov/submissions``, paced under the
SEC's 10 req/s ceiling. Resumable: an existing cache is reused unless --force.

    python research/penny/scripts/fetch_edgar.py
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

from research.penny.etl import bars, edgar  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edgar")

FILINGS_PATH = bars.DATA_ROOT / "edgar_filings.parquet"
MAP_PATH = bars.DATA_ROOT / "edgar_ticker_map.parquet"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch EDGAR filing indexes.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cand_path = bars.DATA_ROOT / "candidates.parquet"
    if not cand_path.exists():
        raise SystemExit("candidates.parquet missing; run build_candidates first")

    symbols = sorted(pd.read_parquet(cand_path, columns=["symbol"])["symbol"].unique())
    logger.info("candidate symbols: %d", len(symbols))

    if MAP_PATH.exists() and not args.force:
        ticker_map = pd.read_parquet(MAP_PATH)
        logger.info("ticker map: %d rows (cached)", len(ticker_map))
    else:
        ticker_map = edgar.fetch_ticker_cik_map()
        ticker_map.to_parquet(MAP_PATH, index=False)

    matched = set(ticker_map["symbol"]) & {s.upper() for s in symbols}
    logger.info(
        "%d/%d candidate symbols map to a CIK (%.1f%%) -- the rest are delisted "
        "or renamed and will carry no filing features",
        len(matched), len(symbols), 100 * len(matched) / max(len(symbols), 1),
    )

    if FILINGS_PATH.exists() and not args.force:
        filings = pd.read_parquet(FILINGS_PATH)
        logger.info("filings: %s rows (cached)", f"{len(filings):,}")
        return 0

    t0 = time.time()
    filings = edgar.fetch_all_filings(symbols, ticker_map)
    if filings.empty:
        raise SystemExit("no filings retrieved")

    FILINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    filings.to_parquet(FILINGS_PATH, index=False)

    by_form = filings["form"].value_counts().head(12).to_dict()
    logger.info(
        "wrote %s (%s filings, %d symbols) in %.0fs",
        FILINGS_PATH, f"{len(filings):,}", filings["symbol"].nunique(), time.time() - t0,
    )
    logger.info("most common forms: %s", by_form)

    (bars.DATA_ROOT / "edgar_summary.json").write_text(
        json.dumps(
            {
                "filings": int(len(filings)),
                "symbols_with_filings": int(filings["symbol"].nunique()),
                "candidate_symbols": len(symbols),
                "cik_match_rate": len(matched) / max(len(symbols), 1),
                "by_form": {str(k): int(v) for k, v in by_form.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
