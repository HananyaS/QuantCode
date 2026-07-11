"""sp500_membership.py — point-in-time S&P 500 constituent membership.

Why this exists
-----------------
utils/sp500.py returns only *today's* constituents — fine for the live
pipeline (agents/universe_agent.py's live path trades only currently-listed
names) but a source of survivorship bias for backtesting: historical dates
get evaluated against a universe that excludes every ticker later removed
from the index. This module answers "who was actually in the S&P 500 on
date X" from a free, static, point-in-time dataset — not a live API — so
it's a one-time download, cached locally, refreshed manually/periodically.

Source: https://github.com/fja05680/sp500
  "S&P 500 Historical Components & Changes.csv" — one row per date the
  membership changed, columns `date, tickers` where `tickers` is a quoted,
  comma-delimited list of every symbol in the index as of that date.

Ticker suffix caveat
----------------------
The dataset appends a synthetic "-YYYYMM" suffix to some symbols (e.g.
"AAL-199702") to disambiguate a ticker being reused by a different company
at a different time — it is NOT part of the real tradeable symbol and is
stripped by this module before returning tickers. The dataset's own README
only says ticker names are "correct on any day you pick" and that a symbol
disappearing from later rows means it left the index — the exact suffix
semantics aren't authoritatively documented, so treat this as a best-effort
free approximation (per teddykoker's write-up on the same dataset: "not
100% historically accurate but a much better start than current
constituents"). Cross-check with `scripts/validate_data_sources.py` before
trusting results derived from it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Union

import pandas as pd
import requests

_SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
)
_DEFAULT_PATH = Path("data/reference/sp500_membership.csv")
_SUFFIX_RE = re.compile(r"-\d{6}$")

PathLike = Union[str, Path]


def download_membership_csv(
    dest: PathLike = _DEFAULT_PATH,
    url: str = _SOURCE_URL,
    http_get=requests.get,
) -> Path:
    """Download the point-in-time membership CSV to `dest` (overwrites).

    A one-time (or infrequent, e.g. quarterly) operation — index membership
    changes rarely, so there's no need to re-fetch on every run.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = http_get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def _strip_suffix(raw_symbol: str) -> str:
    """Strip the dataset's '-YYYYMM' stint-disambiguation suffix."""
    return _SUFFIX_RE.sub("", raw_symbol)


def _parse_tickers(raw: str) -> List[str]:
    return [_strip_suffix(t.strip()) for t in str(raw).split(",") if t.strip()]


def _load_membership_table(path: PathLike = _DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    assert path.exists(), (
        f"{path} not found — call download_membership_csv() first to fetch "
        "the free point-in-time dataset from fja05680/sp500."
    )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_constituents_on(date, path: PathLike = _DEFAULT_PATH) -> List[str]:
    """Return the S&P 500 ticker list as of `date` (point-in-time).

    Uses the most recent membership snapshot at or before `date` — the
    dataset only has a row when membership actually changed, so the last
    known snapshot is carried forward, exactly like a real point-in-time
    query.
    """
    date = pd.Timestamp(date)
    table = _load_membership_table(path)
    eligible = table[table["date"] <= date]
    assert len(eligible) > 0, (
        f"No membership data at or before {date.date()} — dataset starts "
        f"{table['date'].min().date()}"
    )
    row = eligible.iloc[-1]
    return _parse_tickers(row["tickers"])


def get_all_historical_tickers(start, end, path: PathLike = _DEFAULT_PATH) -> List[str]:
    """Return the union of every ticker that was a constituent at any point
    in [start, end] — the download universe needed to backfill delisted names.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    table = _load_membership_table(path)

    window = table[(table["date"] >= start) & (table["date"] <= end)]
    # Membership as of the start of the window may predate any change row
    # inside it — carry the last snapshot before `start` forward too.
    before = table[table["date"] < start].tail(1)
    combined = pd.concat([before, window])
    assert len(combined) > 0, (
        f"No membership data available for [{start.date()}, {end.date()}]"
    )

    all_tickers = set()
    for raw in combined["tickers"]:
        all_tickers.update(_parse_tickers(raw))
    return sorted(all_tickers)
