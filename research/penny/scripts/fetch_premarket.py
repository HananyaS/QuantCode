"""fetch_premarket.py — extended-hours bars, 04:00-09:34 ET.

The single largest missing input. The existing minute cache starts at 09:25,
so the study has five minutes of pre-open information; Alpaca serves from
04:00, which is 330 minutes. For a gap-up microcap the entire setup forms
before the bell -- overnight news, the volume that built the gap, whether the
move is being accumulated or distributed into the open.

Written to a **separate** directory rather than extending the existing cache.
Rewriting 173 MB of working data in place, mid-study, to add a feature that may
not help is a bad trade: if this fetch is interrupted or wrong, the labelled
dataset that produced every result so far is untouched.

The window deliberately ends at 09:34, one minute before the decision. Nothing
in this file can see the entry bar.

    python research/penny/scripts/fetch_premarket.py
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
logger = logging.getLogger("premarket")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREMARKET_DIR = bars.DATA_ROOT / "premarket"

# Ends one minute before the 09:35 decision. This bound is the leakage
# guarantee for every feature derived from these bars.
START_ET = "04:00"
END_ET = "09:34"

_KEEP = ["symbol", "date", "minute_et", "open", "high", "low", "close",
         "volume", "trade_count", "vwap"]


def session_path(day: pd.Timestamp) -> Path:
    return PREMARKET_DIR / f"dt={day:%Y-%m-%d}" / "premarket.parquet"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch 04:00-09:34 bars for candidates.")
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv(_PROJECT_ROOT / ".env")

    cand_path = bars.DATA_ROOT / "candidates.parquet"
    if not cand_path.exists():
        raise SystemExit("candidates.parquet missing")

    candidates = pd.read_parquet(cand_path, columns=["symbol", "date"])
    days = sorted(candidates["date"].unique())
    if args.limit_days:
        days = days[-args.limit_days:]

    by_day = candidates.groupby("date")["symbol"].apply(lambda s: sorted(set(s)))
    logger.info(
        "%s candidate symbol-days across %d sessions; window %s-%s ET",
        f"{len(candidates):,}", len(days), START_ET, END_ET,
    )

    key, secret = resolve_credentials()
    client = alpaca_fetch.make_data_client(key, secret)

    fetched = skipped = failed = empty = 0
    total_rows = 0
    t0 = time.time()

    for i, day in enumerate(days, start=1):
        day_ts = pd.Timestamp(day)
        path = session_path(day_ts)
        if path.exists() and not args.force:
            skipped += 1
            continue

        try:
            df, _ = alpaca_fetch.fetch_minute_for_day(
                client, list(by_day.loc[day]), day_ts.strftime("%Y-%m-%d"),
                start_et=START_ET, end_et=END_ET,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: FAILED %s", day_ts.date(), str(exc)[:160])
            failed += 1
            continue

        if df.empty:
            # Normal on quiet sessions: thin names simply do not trade
            # pre-open. Recorded so the absence is evidence, not a gap.
            empty += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=_KEEP).to_parquet(path, index=False)
            continue

        # Hard guarantee: nothing at or after the decision minute may be stored.
        late = df[df["minute_et"] >= "09:35"]
        assert late.empty, (
            f"{day_ts.date()}: {len(late)} bars at/after 09:35 in a premarket file"
        )

        df = df[[c for c in _KEEP if c in df.columns]]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")

        fetched += 1
        total_rows += len(df)
        if fetched % 100 == 0:
            rate = fetched / max(time.time() - t0, 1e-9)
            logger.info(
                "%s | %d fetched / %d skipped / %d empty / %d failed | "
                "%s rows | ~%.0f min left",
                day_ts.date(), fetched, skipped, empty, failed,
                f"{total_rows:,}", (len(days) - i) / max(rate, 1e-9) / 60,
            )

    summary = {
        "sessions_fetched": fetched,
        "sessions_skipped": skipped,
        "sessions_empty": empty,
        "sessions_failed": failed,
        "total_rows": total_rows,
        "window": f"{START_ET}-{END_ET}",
    }
    (bars.DATA_ROOT / "premarket_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("done: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
