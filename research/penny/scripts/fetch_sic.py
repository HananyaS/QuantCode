"""fetch_sic.py — SIC codes for every CIK, and a sector mapping built from them.

Why
---
Nine rounds have treated a 50% crash as one population. It is at least two:

* a stock down 50% while its **sector is flat** fell for its own reasons -- a
  failed trial, a guidance cut, a fraud;
* a stock down 50% while its **sector is down 20%** fell because a factor moved
  -- an oil price, a rate, a policy.

Those should not bounce the same way, and nothing in this project distinguishes
them. Round 9's ten largest trades were seven oil and gas names on two
consecutive days: the systematic case, reported as though it were the general
one.

Sector is not in the Alpaca registry, but EDGAR publishes an SIC code in the
same `submissions` payload `etl/edgar.py` already fetches, so this costs one
pass over the CIKs already known and no new data source.

SIC is assigned by the registrant and is coarse and occasionally stale. It is
adequate here because the question is only "did this stock's peers fall too",
which needs a usable grouping rather than a precise taxonomy.

    python research/penny/scripts/fetch_sic.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars, edgar  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sic")

OUT = bars.DATA_ROOT / "sic_sectors.parquet"

# SIC ranges -> a grouping coarse enough to leave usable daily samples in a
# microcap universe. Order matters: the first matching range wins.
SECTOR_RANGES: List[tuple] = [
    ((2833, 2836), "biotech_pharma"),
    ((8731, 8731), "biotech_pharma"),
    ((3826, 3829), "biotech_pharma"),
    ((3841, 3851), "medical_devices"),
    ((8000, 8099), "healthcare_services"),
    ((7370, 7379), "software_it"),
    ((7371, 7372), "software_it"),
    ((3570, 3579), "hardware_semis"),
    ((3670, 3679), "hardware_semis"),
    ((3660, 3669), "hardware_semis"),
    ((4810, 4899), "telecom_media"),
    ((2700, 2799), "telecom_media"),
    ((7800, 7999), "telecom_media"),
    ((1311, 1311), "energy"),
    ((1300, 1399), "energy"),
    ((2911, 2999), "energy"),
    ((4922, 4924), "energy"),
    ((1000, 1099), "mining_metals"),
    ((1040, 1049), "mining_metals"),
    ((3300, 3399), "mining_metals"),
    ((6000, 6199), "financials"),
    ((6200, 6299), "financials"),
    ((6300, 6499), "financials"),
    ((6500, 6599), "real_estate"),
    ((6798, 6798), "real_estate"),
    ((5200, 5999), "retail_consumer"),
    ((2000, 2199), "retail_consumer"),
    ((2300, 2399), "retail_consumer"),
    ((4400, 4799), "transport_shipping"),
    ((3700, 3799), "transport_shipping"),
    ((1500, 1799), "industrials"),
    ((3400, 3569), "industrials"),
    ((2800, 2832), "chemicals_materials"),
    ((2837, 2899), "chemicals_materials"),
    ((3080, 3089), "chemicals_materials"),
    ((6770, 6770), "shell_blank_check"),
]


def sic_to_sector(sic: Optional[float]) -> str:
    if sic is None or not (isinstance(sic, (int, float)) and sic == sic):
        return "unknown"
    s = int(sic)
    for (lo, hi), name in SECTOR_RANGES:
        if lo <= s <= hi:
            return name
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    filings = pd.read_parquet(bars.DATA_ROOT / "edgar_filings.parquet",
                              columns=["symbol", "cik"])
    pairs = filings.drop_duplicates(["symbol", "cik"])
    if args.limit:
        pairs = pairs.head(args.limit)
    logger.info("CIKs to query: %s", f"{len(pairs):,}")

    done: Dict[int, dict] = {}
    if Path(args.out).exists():
        prev = pd.read_parquet(args.out)
        done = {int(r.cik): {"sic": r.sic, "sic_desc": r.sic_desc}
                for r in prev.itertuples()}
        logger.info("resuming: %s CIKs already cached", f"{len(done):,}")

    rows: List[dict] = []
    miss = 0
    for i, (sym, cik) in enumerate(zip(pairs["symbol"], pairs["cik"]), 1):
        cik = int(cik)
        if cik in done:
            rows.append({"symbol": sym, "cik": cik, **done[cik]})
            continue
        time.sleep(edgar.REQUEST_PAUSE)
        try:
            payload = edgar.fetch_submissions(cik)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cik %d failed: %s", cik, str(exc)[:70])
            payload = None
        if not payload:
            miss += 1
            rows.append({"symbol": sym, "cik": cik, "sic": None, "sic_desc": None})
            continue
        sic = payload.get("sic")
        rec = {"sic": float(sic) if sic not in (None, "") else None,
               "sic_desc": payload.get("sicDescription")}
        done[cik] = rec
        rows.append({"symbol": sym, "cik": cik, **rec})
        if i % 250 == 0:
            logger.info("  %d/%d (%d without a payload)", i, len(pairs), miss)

    df = pd.DataFrame(rows)
    df["sector"] = df["sic"].map(sic_to_sector)
    df.to_parquet(args.out, index=False)

    have = df["sic"].notna()
    logger.info("wrote %s | %s symbols | SIC present for %s (%.1f%%)", args.out,
                f"{len(df):,}", f"{int(have.sum()):,}", 100 * have.mean())
    logger.info("sector composition:\n%s",
                df["sector"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
