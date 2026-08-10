"""fetch_daily_history.py — download the daily bar panel for the study.

Fetches raw and adjusted daily bars for every plausible exchange-listed common
stock Alpaca knows about, **including delisted names**, one calendar year per
parquet file. Resumable: a year already on disk is skipped unless --force.

    python research/penny/scripts/fetch_daily_history.py --start-year 2021
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import alpaca_fetch, bars, universe  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_CREDENTIAL_PAIRS = [
    ("ALPACA_API_KEY_RESEARCH", "ALPACA_SECRET_KEY_RESEARCH"),
    ("ALPACA_API_KEY_KELLY", "ALPACA_SECRET_KEY_KELLY"),
    ("ALPACA_API_KEY_LINEAR", "ALPACA_SECRET_KEY_LINEAR"),
    ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
]


def resolve_credentials():
    """First credential pair that authenticates.

    The plain ALPACA_API_KEY pair in .env is revoked and returns 401 on every
    feed, which reads exactly like a missing SIP entitlement (403) if not
    checked. Order matters: try the known-good pairs first.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    for key_name, secret_name in _CREDENTIAL_PAIRS:
        key, secret = os.environ.get(key_name), os.environ.get(secret_name)
        if not key or not secret:
            continue
        try:
            client = alpaca_fetch.make_data_client(key, secret)
            df = client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=["AAPL"], timeframe=TimeFrame.Day,
                    start="2026-07-01", end="2026-07-10", feed="sip",
                )
            ).df
            if not df.empty:
                logger.info("credentials: using %s (SIP verified)", key_name)
                return key, secret
        except Exception as exc:  # noqa: BLE001
            logger.warning("credentials: %s rejected (%s)", key_name, str(exc)[:80])
    raise SystemExit("no working Alpaca credentials found")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the daily bar panel.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--force", action="store_true", help="Refetch years already cached.")
    args = parser.parse_args(argv)

    load_dotenv(_PROJECT_ROOT / ".env")
    key, secret = resolve_credentials()
    client = alpaca_fetch.make_data_client(key, secret)

    bars.DATA_ROOT.mkdir(parents=True, exist_ok=True)

    registry_path = bars.DATA_ROOT / "asset_registry.parquet"
    if registry_path.exists() and not args.force:
        registry = pd.read_parquet(registry_path)
        logger.info("registry: loaded %d rows from cache", len(registry))
    else:
        registry = universe.fetch_asset_registry(key, secret)
        registry.to_parquet(registry_path, index=False)

    listed = universe.select_listed_common(registry)
    symbols: List[str] = sorted(listed["symbol"].unique())
    logger.info(
        "universe: %d symbols (%d active, %d delisted) -- delisted names are "
        "included deliberately to avoid survivorship bias",
        len(symbols),
        (listed["status"] == "active").sum(),
        (listed["status"] == "inactive").sum(),
    )

    today = date.today()
    summary = {}
    for year in range(args.start_year, args.end_year + 1):
        path = bars._year_path(year)
        if path.exists() and not args.force:
            n = len(pd.read_parquet(path, columns=["symbol"]))
            logger.info("%d: cached (%d rows), skipping", year, n)
            summary[year] = {"rows": n, "cached": True}
            continue

        end_cap = None
        if year == today.year:
            # Stay clear of the free tier's 15-minute SIP delay.
            end_cap = (pd.Timestamp(today) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        t0 = time.time()
        logger.info("%d: fetching %d symbols ...", year, len(symbols))
        df, rejects = bars.fetch_year(client, symbols, year, end_cap=end_cap)
        if df.empty:
            logger.warning("%d: no data returned", year)
            continue

        written = bars.write_year(df, year)
        elapsed = time.time() - t0
        n_sym = df["symbol"].nunique()
        logger.info(
            "%d: %d rows, %d symbols, %d rejected -> %s (%.0fs)",
            year, len(df), n_sym, len(set(rejects.get("all", []))), written, elapsed,
        )
        summary[year] = {
            "rows": int(len(df)),
            "symbols": int(n_sym),
            "rejected": sorted(set(rejects.get("all", [])))[:50],
            "seconds": round(elapsed, 1),
            "cached": False,
        }

    (bars.DATA_ROOT / "fetch_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("done. summary -> %s", bars.DATA_ROOT / "fetch_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
