"""fetch_corporate_actions.py -- the vendor's corporate-action records, as ground truth.

Why this exists
---------------
Round 17 keeps trades whose holding window contains a split (the old frame
dropped them, which reads the future). Keeping them needs the split itself.
The price panel cannot supply it: inside dev holding windows, 79% of the days
the panel flags as splits (`any_split | is_suspect_fwd_split`) still jump in the
vendor-adjusted series, so the adjusted series has not absorbed them, and a
price-only detector cannot tell a missed 1-for-2 reverse split from a genuine
+100% gap. Alpaca's corporate-actions endpoint records the event, its ex-date
and its ratio, and is what the live account will act on.

Writes one parquet per run under `data/research_penny/corporate_actions/`.
Reads no prices. Refuses end dates on or after SEAL without --unseal, so dev
work never looks at holdout-period records.

    python research/penny/scripts/fetch_corporate_actions.py --start 2019-10-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.dates import SEAL                                   # noqa: E402
from research.penny.etl import bars                                     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("ca")

OUT_DIR = bars.DATA_ROOT / "corporate_actions"
ENV = Path(__file__).resolve().parents[3] / ".env"
KEY_PAIRS = (("ALPACA_API_KEY_KELLY", "ALPACA_SECRET_KEY_KELLY"),
             ("ALPACA_API_KEY_LINEAR", "ALPACA_SECRET_KEY_LINEAR"))
FIELDS = ("symbol", "new_symbol", "old_symbol", "acquirer_symbol",
          "acquiree_symbol", "ex_date", "process_date", "effective_date",
          "old_rate", "new_rate", "rate", "cash_rate")


def _client():
    from dotenv import dotenv_values
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    # A worktree has no .env of its own; the checkout's sits further up.
    found = next((p / ".env" for p in Path(__file__).resolve().parents
                  if (p / ".env").is_file()), ENV)
    env = {**dotenv_values(found)}
    for k, s in KEY_PAIRS:
        if env.get(k) and env.get(s):
            return CorporateActionsClient(env[k], env[s])
    raise SystemExit("no Alpaca key pair found in .env")


def _months(start: date, end: date):
    a = start
    while a <= end:
        b = min((a.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1), end)
        yield a, b
        a = b + timedelta(days=1)


def fetch(start: date, end: date) -> pd.DataFrame:
    from alpaca.data.enums import CorporateActionsType as T
    from alpaca.data.requests import CorporateActionsRequest
    cli = _client()
    types = [T.REVERSE_SPLIT, T.FORWARD_SPLIT, T.UNIT_SPLIT, T.STOCK_DIVIDEND,
             T.NAME_CHANGE, T.WORTHLESS_REMOVAL, T.CASH_MERGER, T.STOCK_MERGER,
             T.STOCK_AND_CASH_MERGER, T.SPIN_OFF]
    rows: List[Dict] = []
    for a, b in _months(start, end):
        for attempt in range(4):
            try:
                res = cli.get_corporate_actions(
                    CorporateActionsRequest(types=types, start=a, end=b))
                break
            except Exception as exc:                    # network / rate limit
                logger.warning("%s..%s attempt %d: %s", a, b, attempt + 1, exc)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed {a}..{b}")
        data = res.data if hasattr(res, "data") else res
        n = 0
        for kind, items in data.items():
            for x in items:
                d = {"type": kind}
                for f in FIELDS:
                    d[f] = getattr(x, f, None)
                rows.append(d)
                n += 1
        logger.info("%s..%s  %d records", a, b, n)
    df = pd.DataFrame(rows)
    for c in ("ex_date", "process_date", "effective_date"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--unseal", action="store_true")
    args = ap.parse_args(argv)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if pd.Timestamp(end) >= SEAL and not args.unseal:
        raise SystemExit(f"end {end} reaches the sealed period ({SEAL.date()}); pass --unseal")
    df = fetch(start, end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"ca_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    df.to_parquet(out, index=False)
    logger.info("wrote %s  (%d rows)\n%s", out, len(df), df["type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
