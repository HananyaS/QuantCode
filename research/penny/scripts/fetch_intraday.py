"""fetch_intraday.py — minute bars for candidate symbol-days.

Grouping requests by *session* rather than by symbol is what makes intraday
labelling affordable: one request covers every candidate that day, so the whole
study costs roughly one request per trading day instead of one per symbol-day.

Resumable -- a session already cached is skipped.

    python research/penny/scripts/fetch_intraday.py
    python research/penny/scripts/fetch_intraday.py --limit-days 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import alpaca_fetch, bars  # noqa: E402
from research.penny.scripts.fetch_daily_history import resolve_credentials  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("intraday")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MINUTE_DIR = bars.DATA_ROOT / "minute"

_KEEP = [
    "symbol", "date", "minute_et", "open", "high", "low", "close",
    "volume", "trade_count", "vwap",
]


def session_path(day: pd.Timestamp) -> Path:
    return MINUTE_DIR / f"dt={day:%Y-%m-%d}" / "minute.parquet"


def requested_path(day: pd.Timestamp) -> Path:
    """Sidecar recording which symbols were *asked for* on this session.

    Existence of ``minute.parquet`` is not sufficient to skip a session. Round 4
    rebuilt the candidate screen, so a session cached under round 3's screen is
    present on disk but missing the symbols round 4 needs -- and a skip keyed on
    the file alone would silently return round-3 coverage while reporting a
    clean run.

    The sidecar records the requested set rather than the returned set, because
    a symbol can legitimately have no prints that day (halted, no quotes); keying
    on what came back would refetch those forever.
    """
    return MINUTE_DIR / f"dt={day:%Y-%m-%d}" / "requested.json"


def cached_symbols(day: pd.Timestamp) -> set:
    """Symbols already covered for this session, from sidecar or the data."""
    side = requested_path(day)
    if side.exists():
        try:
            return set(json.loads(side.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            pass
    path = session_path(day)
    if path.exists():
        try:
            return set(pd.read_parquet(path, columns=["symbol"])["symbol"].unique())
        except Exception:  # noqa: BLE001
            return set()
    return set()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch minute bars for candidates.")
    parser.add_argument("--candidates", default=str(bars.DATA_ROOT / "candidates.parquet"))
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv(_PROJECT_ROOT / ".env")

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        raise SystemExit(f"candidates not found: {cand_path}. Run build_candidates first.")

    candidates = pd.read_parquet(cand_path, columns=["symbol", "date"])
    days = sorted(candidates["date"].unique())
    if args.limit_days:
        days = days[-args.limit_days :]

    logger.info(
        "%d candidate symbol-days across %d sessions", len(candidates), len(days)
    )

    key, secret = resolve_credentials()
    client = alpaca_fetch.make_data_client(key, secret)

    by_day = candidates.groupby("date")["symbol"].apply(lambda s: sorted(set(s)))

    fetched = skipped = failed = 0
    total_rows = 0
    t_start = time.time()

    for i, day in enumerate(days, start=1):
        day_ts = pd.Timestamp(day)
        path = session_path(day_ts)

        needed = set(by_day.loc[day])
        have = set() if args.force else cached_symbols(day_ts)
        missing = sorted(needed - have)
        if not missing:
            skipped += 1
            continue

        symbols: List[str] = missing
        try:
            df, rejected = alpaca_fetch.fetch_minute_for_day(
                client, symbols, day_ts.strftime("%Y-%m-%d")
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: FAILED %s", day_ts.date(), str(exc)[:160])
            failed += 1
            continue

        if df.empty:
            logger.warning("%s: no minute data for %d symbols", day_ts.date(), len(symbols))
            failed += 1
            continue

        df = df[[c for c in _KEEP if c in df.columns]]
        path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with whatever this session already held rather than overwriting:
        # the cached rows are valid data for symbols still in the screen.
        if path.exists() and not args.force:
            try:
                prior = pd.read_parquet(path)
                df = (
                    pd.concat([prior, df], ignore_index=True)
                    .drop_duplicates(subset=["symbol", "date", "minute_et"], keep="last")
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: could not merge cache (%s); rewriting",
                               day_ts.date(), str(exc)[:80])

        df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
        requested_path(day_ts).write_text(
            json.dumps(sorted(have | needed)), encoding="utf-8"
        )

        fetched += 1
        total_rows += len(df)
        if fetched % 20 == 0 or i == len(days):
            rate = fetched / max(time.time() - t_start, 1e-9)
            remaining = (len(days) - i) / max(rate, 1e-9) / 60
            logger.info(
                "%s: %d symbols, %d rows | %d fetched / %d skipped / %d failed | "
                "~%.0f min left",
                day_ts.date(), df["symbol"].nunique(), len(df),
                fetched, skipped, failed, remaining,
            )

    logger.info(
        "done: %d sessions fetched, %d skipped, %d failed, %s total minute rows",
        fetched, skipped, failed, f"{total_rows:,}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
