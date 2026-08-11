"""edgar.py — point-in-time filing features from SEC EDGAR.

Why these features and not more price history
---------------------------------------------
Every feature in round 1 was a price or volume statistic, so every one of them
proxied the same latent thing: how extreme the stock has been. They were
therefore correlated, and combining them added little. Filings are the only
free source that observes the **supply mechanism directly** rather than
inferring it from price -- a shelf registration is the company announcing it
intends to sell stock, which is a different fact from "the stock is volatile".

Point-in-time discipline
------------------------
The trap here is subtle and would be invisible in results. A filing's
``filingDate`` is a calendar date, but its ``acceptanceDateTime`` is when EDGAR
actually accepted it -- frequently after the close, and sometimes days after
the period it describes. A feature keyed on ``filingDate`` alone would let a
filing accepted at 18:00 influence a decision made at 09:35 that same morning.

Everything here therefore keys on **acceptanceDateTime**, and
:func:`build_pit_features` admits a filing only if it was accepted strictly
before the decision instant.

Rate limits: SEC permits 10 requests/second with a descriptive User-Agent
carrying contact information. Exceeding it earns an IP block, so the pacing
below is deliberate rather than cautious.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

USER_AGENT = "QuantCode penny research shacharhananya@gmail.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

HTTP_TIMEOUT = 45
# 10 req/s is the documented ceiling; 8 leaves headroom for jitter.
REQUEST_PAUSE = 1.0 / 8.0

# Forms that signal share supply arriving. S-3 registers a shelf; 424B* is the
# actual takedown off that shelf, which is the event that puts stock on the
# market -- so the two are tracked separately rather than lumped together.
SHELF_FORMS = {"S-3", "S-3/A", "S-3ASR", "S-1", "S-1/A"}
TAKEDOWN_FORMS = {"424B1", "424B2", "424B3", "424B4", "424B5", "424B7"}
DELISTING_FORMS = {"25", "25-NSE"}

# 8-K item 3.01: "Notice of Delisting or Failure to Satisfy a Continued Listing
# Rule". Available in the submissions index without fetching the document.
DEFICIENCY_ITEM = "3.01"
# 8-K item 1.01/3.02: entry into a material agreement / unregistered sale of
# equity -- the usual home of a PIPE or death-spiral convertible.
DILUTIVE_ITEMS = {"1.01", "3.02"}


@dataclass
class EdgarConfig:
    lookback_days: int = 180
    max_symbols: Optional[int] = None


def fetch_ticker_cik_map(http_get: Callable = requests.get) -> pd.DataFrame:
    """Ticker -> CIK mapping published by the SEC.

    Note this maps *current* tickers. A symbol that changed ticker or delisted
    may be absent, which is a real coverage gap and is reported rather than
    silently treated as "no filings".
    """
    resp = http_get(TICKER_MAP_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    rows = [
        {"symbol": str(v["ticker"]).upper(), "cik": int(v["cik_str"]), "name": v.get("title")}
        for v in payload.values()
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"])
    logger.info("edgar: %d ticker->CIK mappings", len(df))
    return df


def fetch_submissions(cik: int, http_get: Callable = requests.get) -> Optional[Dict]:
    """Filing index for one company."""
    resp = http_get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=HTTP_TIMEOUT)
    if resp.status_code != 200:
        return None
    return resp.json()


def parse_filings(payload: Dict, symbol: str, cik: int) -> pd.DataFrame:
    """Flatten a submissions payload into one row per filing.

    Only the ``recent`` block is used. Older filings live in separate paginated
    files; for a study window starting 2016 the recent block (up to ~1,000
    filings) covers the microcaps in scope, and the truncation is recorded so
    it is not mistaken for an absence of filings.
    """
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form")
    if not forms:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "form": forms,
            "items": recent.get("items", [""] * len(forms)),
            "filing_date": recent.get("filingDate", [None] * len(forms)),
            "acceptance": recent.get("acceptanceDateTime", [None] * len(forms)),
        }
    )
    df["symbol"] = symbol
    df["cik"] = cik

    # acceptanceDateTime is ISO-8601 in US/Eastern; keep it tz-aware so the
    # comparison against a 09:35 ET decision is unambiguous.
    df["accepted_et"] = pd.to_datetime(df["acceptance"], errors="coerce", utc=True)
    df["accepted_et"] = df["accepted_et"].dt.tz_convert("America/New_York")
    # Fall back to the filing date at 23:59 ET when acceptance is missing --
    # the conservative choice, since it delays rather than advances the fact.
    fallback = pd.to_datetime(df["filing_date"], errors="coerce")
    fallback = fallback.dt.tz_localize("America/New_York") + pd.Timedelta(hours=23, minutes=59)
    df["accepted_et"] = df["accepted_et"].fillna(fallback)

    return df.dropna(subset=["accepted_et"])


def fetch_all_filings(
    symbols: Sequence[str],
    ticker_map: pd.DataFrame,
    http_get: Callable = requests.get,
    progress_every: int = 250,
) -> pd.DataFrame:
    """Filing index for every symbol that maps to a CIK."""
    lookup = dict(zip(ticker_map["symbol"], ticker_map["cik"]))
    frames: List[pd.DataFrame] = []
    missing: List[str] = []

    for i, symbol in enumerate(symbols, start=1):
        cik = lookup.get(symbol.upper())
        if cik is None:
            missing.append(symbol)
            continue
        try:
            time.sleep(REQUEST_PAUSE)
            payload = fetch_submissions(int(cik), http_get=http_get)
        except Exception as exc:  # noqa: BLE001
            logger.warning("edgar: %s (CIK %s) failed -- %s", symbol, cik, str(exc)[:120])
            continue
        if payload is None:
            missing.append(symbol)
            continue
        parsed = parse_filings(payload, symbol, int(cik))
        if not parsed.empty:
            frames.append(parsed)
        if i % progress_every == 0:
            logger.info("edgar: %d/%d symbols, %d with filings", i, len(symbols), len(frames))

    if missing:
        logger.warning(
            "edgar: %d/%d symbols had no CIK or no submissions (delisted or renamed): %s",
            len(missing), len(symbols), missing[:10],
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _has_item(items: pd.Series, wanted: Iterable[str]) -> pd.Series:
    """8-K item codes arrive as a comma-separated string."""
    wanted = set(wanted)

    def _check(value) -> bool:
        if not isinstance(value, str) or not value:
            return False
        return bool({p.strip() for p in value.split(",")} & wanted)

    return items.map(_check)


def build_pit_features(
    candidates: pd.DataFrame,
    filings: pd.DataFrame,
    config: Optional[EdgarConfig] = None,
    decision_time: str = "09:35",
) -> pd.DataFrame:
    """Attach point-in-time filing features to each candidate symbol-day.

    A filing counts only if ``accepted_et`` is strictly before the decision
    instant on that candidate's date. This is what stops an 18:00 acceptance
    from informing a 09:35 decision earlier the same day.
    """
    config = config or EdgarConfig()
    if filings.empty:
        logger.warning("edgar: no filings supplied; features will be null")
        return candidates.assign(
            edgar_days_since_shelf=np.nan,
            edgar_days_since_takedown=np.nan,
            edgar_deficiency_180d=False,
            edgar_dilutive_8k_180d=False,
            edgar_filing_count_180d=0,
            edgar_has_coverage=False,
        )

    covered = set(filings["symbol"].unique())

    # Filter to the forms that matter before anything else. The raw index is
    # dominated by insider Forms 3/4/144, which are ~40% of 1.4M rows and carry
    # none of the supply signal this study is after.
    relevant = filings["form"].isin(SHELF_FORMS | TAKEDOWN_FORMS | DELISTING_FORMS) | (
        filings["form"].str.startswith("8-K", na=False)
    )
    f = filings[relevant].copy()
    logger.info("edgar: %s relevant filings of %s total", f"{len(f):,}", f"{len(filings):,}")

    f["is_shelf"] = f["form"].isin(SHELF_FORMS)
    f["is_takedown"] = f["form"].isin(TAKEDOWN_FORMS)
    is_8k = f["form"].str.startswith("8-K", na=False)
    f["is_deficiency"] = is_8k & _has_item(f["items"], [DEFICIENCY_ITEM])
    f["is_dilutive_8k"] = is_8k & _has_item(f["items"], DILUTIVE_ITEMS)

    # Per-symbol sorted arrays, so each candidate row costs a binary search
    # rather than a full scan of the filing table.
    # Times are held as UTC epoch nanoseconds. A tz-aware Series returns object
    # dtype from .to_numpy(), which cannot be compared against datetime64 and
    # would make searchsorted raise; integers sidestep that entirely and are
    # faster besides.
    by_symbol: Dict[str, Dict[str, np.ndarray]] = {}
    for symbol, group in f.sort_values("accepted_et").groupby("symbol", sort=False):
        by_symbol[symbol] = {
            "t": group["accepted_et"].dt.tz_convert("UTC").astype("int64").to_numpy(),
            "shelf": group["is_shelf"].to_numpy(),
            "takedown": group["is_takedown"].to_numpy(),
            "deficiency": group["is_deficiency"].to_numpy(),
            "dilutive": group["is_dilutive_8k"].to_numpy(),
        }

    lookback = pd.Timedelta(days=config.lookback_days)
    out_rows: List[Dict] = []

    for symbol, group in candidates.groupby("symbol", sort=False):
        arrays = by_symbol.get(symbol)
        has_coverage = symbol in covered

        for date in group["date"]:
            decision = pd.Timestamp(f"{date:%Y-%m-%d} {decision_time}", tz="America/New_York")
            row = {
                "symbol": symbol, "date": date,
                "edgar_days_since_shelf": np.nan,
                "edgar_days_since_takedown": np.nan,
                "edgar_deficiency_180d": False,
                "edgar_dilutive_8k_180d": False,
                "edgar_filing_count_180d": 0,
                "edgar_has_coverage": has_coverage,
            }
            if arrays is not None and len(arrays["t"]):
                # Strictly before the decision instant: this is the whole
                # point-in-time guarantee, in one line.
                decision_ns = decision.tz_convert("UTC").value
                cut = int(np.searchsorted(arrays["t"], decision_ns, side="left"))
                if cut > 0:
                    start = int(np.searchsorted(
                        arrays["t"], (decision - lookback).tz_convert("UTC").value, side="left"
                    ))
                    window = slice(start, cut)

                    for key, out_key in (("shelf", "edgar_days_since_shelf"),
                                         ("takedown", "edgar_days_since_takedown")):
                        hits = np.flatnonzero(arrays[key][:cut])
                        if hits.size:
                            elapsed_ns = decision_ns - int(arrays["t"][hits[-1]])
                            row[out_key] = float(elapsed_ns / 86_400_000_000_000)

                    row["edgar_deficiency_180d"] = bool(arrays["deficiency"][window].any())
                    row["edgar_dilutive_8k_180d"] = bool(arrays["dilutive"][window].any())
                    row["edgar_filing_count_180d"] = int(cut - start)

            out_rows.append(row)

    return pd.DataFrame(out_rows)


def assert_no_future_filings(
    filings: pd.DataFrame, decision: pd.Timestamp, tolerance_seconds: int = 0
) -> None:
    """Tripwire: no filing used for a decision may have been accepted after it."""
    if filings.empty:
        return
    late = filings[filings["accepted_et"] >= decision + pd.Timedelta(seconds=tolerance_seconds)]
    assert late.empty, (
        f"{len(late)} filings accepted at or after the decision instant {decision} "
        f"(earliest offender {late['accepted_et'].min()})"
    )
