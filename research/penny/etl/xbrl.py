"""xbrl.py — shares outstanding, as a point-in-time float proxy.

Float is the classic microcap squeeze variable and the study has had no proxy
for it at all. A 30% gap on 4 million shares is a different animal from the
same gap on 400 million.

Why the frames API rather than companyfacts
-------------------------------------------
``/api/xbrl/companyfacts/CIK*.json`` returns every fact a company ever filed --
megabytes per issuer, and ~2,600 issuers here. ``/api/xbrl/frames/...`` returns
one concept for one period across **all** filers in a single response, so the
whole panel costs ~40 requests instead of ~2,600 large ones.

The point-in-time problem, and how it is solved
-----------------------------------------------
A frame is stamped with the *period end* (e.g. 2024Q1), but that number does
not become public until the 10-Q is filed -- typically 45 days later for a
non-accelerated filer, and 90 for an annual report. Keying a feature on the
period end would let a decision in April use a share count the market did not
see until mid-May.

So each value is dated by the acceptance timestamp of the **first periodic
report that symbol actually filed after the period end**, taken from the EDGAR
filing index already collected. Where no such filing is on record, the value
falls back to ``period_end + 90 days`` -- deliberately late rather than early,
matching the convention used everywhere else in this study.
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

FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json"
HTTP_TIMEOUT = 60
REQUEST_PAUSE = 1.0 / 8.0

# The cover-page share count, which is the closest thing XBRL has to "shares
# outstanding as of the filing". Preferred over the us-gaap balance-sheet
# concepts, which are period-end and more often restated.
PRIMARY_CONCEPT = ("dei", "EntityCommonStockSharesOutstanding", "shares")
FALLBACK_CONCEPTS = [
    ("us-gaap", "CommonStockSharesOutstanding", "shares"),
    ("us-gaap", "CommonStockSharesIssued", "shares"),
]

# Periodic reports whose acceptance makes a cover-page share count public.
PERIODIC_FORMS = ("10-Q", "10-K", "20-F", "40-F", "10-KT", "10-QT")

# Used only when a symbol has no periodic filing on record after the period end.
FALLBACK_LAG_DAYS = 90


def quarter_frames(start_year: int, end_year: int) -> List[str]:
    """Instantaneous frame labels, e.g. ``CY2024Q1I``."""
    return [
        f"CY{year}Q{q}I"
        for year in range(start_year, end_year + 1)
        for q in (1, 2, 3, 4)
    ]


def fetch_frame(
    period: str,
    taxonomy: str = PRIMARY_CONCEPT[0],
    concept: str = PRIMARY_CONCEPT[1],
    unit: str = PRIMARY_CONCEPT[2],
    http_get: Callable = requests.get,
) -> pd.DataFrame:
    """One concept, one period, all filers."""
    url = FRAMES_URL.format(taxonomy=taxonomy, concept=concept, unit=unit, period=period)
    time.sleep(REQUEST_PAUSE)
    resp = http_get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
    if resp.status_code != 200:
        return pd.DataFrame()

    payload = resp.json()
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    keep = [c for c in ("cik", "entityName", "val", "end", "accn") if c in df.columns]
    df = df[keep].copy()
    df["period"] = period
    df["period_end"] = pd.to_datetime(df["end"], errors="coerce")
    return df.dropna(subset=["period_end"])


def fetch_all_frames(
    periods: Sequence[str], http_get: Callable = requests.get
) -> pd.DataFrame:
    """Every requested period, with fallback concepts when the primary is empty."""
    frames: List[pd.DataFrame] = []
    for i, period in enumerate(periods, start=1):
        df = fetch_frame(period, http_get=http_get)
        if df.empty:
            for tax, concept, unit in FALLBACK_CONCEPTS:
                df = fetch_frame(period, tax, concept, unit, http_get=http_get)
                if not df.empty:
                    break
        if not df.empty:
            frames.append(df)
        if i % 8 == 0:
            logger.info("xbrl: %d/%d frames, %s rows",
                        i, len(periods), f"{sum(len(f) for f in frames):,}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def attach_known_from(
    shares: pd.DataFrame,
    filings: pd.DataFrame,
    ticker_map: pd.DataFrame,
) -> pd.DataFrame:
    """Date each share count by when it actually became public.

    Uses the acceptance timestamp of the first periodic report filed after the
    period end. This is what makes the feature point-in-time; the period end
    alone would be weeks early.
    """
    df = shares.merge(ticker_map[["symbol", "cik"]], on="cik", how="inner")
    if df.empty:
        return df

    periodic = filings[filings["form"].isin(PERIODIC_FORMS)][["symbol", "accepted_et"]]
    periodic = periodic.sort_values("accepted_et")

    out_parts: List[pd.DataFrame] = []
    by_symbol = {s: g["accepted_et"].to_numpy() for s, g in periodic.groupby("symbol")}

    for symbol, group in df.groupby("symbol", sort=False):
        g = group.sort_values("period_end").copy()
        accepted = by_symbol.get(symbol)

        fallback = (
            g["period_end"].dt.tz_localize("America/New_York")
            + pd.Timedelta(days=FALLBACK_LAG_DAYS)
        )
        if accepted is None or len(accepted) == 0:
            g["known_from"] = fallback
        else:
            ends = g["period_end"].dt.tz_localize("America/New_York").to_numpy()
            # First periodic filing strictly after each period end.
            idx = np.searchsorted(accepted, ends, side="right")
            picked = np.where(idx < len(accepted), accepted[np.clip(idx, 0, len(accepted) - 1)],
                              np.datetime64("NaT"))
            known = pd.Series(picked, index=g.index)
            g["known_from"] = known.fillna(fallback)
        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def build_pit_features(
    candidates: pd.DataFrame,
    shares: pd.DataFrame,
    decision_time: str = "09:35",
) -> pd.DataFrame:
    """Point-in-time share-count features per candidate symbol-day."""
    base_cols = {
        "shares_outstanding": np.nan,
        "shares_growth_qoq": np.nan,
        "shares_growth_yoy": np.nan,
        "float_dollar": np.nan,
        "gap_to_float": np.nan,
        "shares_has_coverage": False,
    }

    if shares.empty or "known_from" not in shares.columns:
        logger.warning("xbrl: no dated share counts; features will be null")
        out = candidates[["symbol", "date"]].copy()
        for k, v in base_cols.items():
            out[k] = v
        return out

    sh = shares.dropna(subset=["known_from"]).sort_values(["symbol", "known_from"]).copy()
    sh["known_ns"] = sh["known_from"].dt.tz_convert("UTC").astype("int64")

    by_symbol: Dict[str, Dict[str, np.ndarray]] = {
        s: {"t": g["known_ns"].to_numpy(), "val": g["val"].to_numpy(dtype=float)}
        for s, g in sh.groupby("symbol", sort=False)
    }
    covered = set(by_symbol)

    has_price = "close_unadj_prev" in candidates.columns
    has_gap = "gap" in candidates.columns

    rows: List[Dict] = []
    for symbol, group in candidates.groupby("symbol", sort=False):
        arrays = by_symbol.get(symbol)
        for _, cand in group.iterrows():
            row = {"symbol": symbol, "date": cand["date"], **base_cols}
            row["shares_has_coverage"] = symbol in covered

            if arrays is not None and len(arrays["t"]):
                decision = pd.Timestamp(
                    f"{cand['date']:%Y-%m-%d} {decision_time}", tz="America/New_York"
                ).tz_convert("UTC").value
                cut = int(np.searchsorted(arrays["t"], decision, side="left"))
                if cut > 0:
                    latest = float(arrays["val"][cut - 1])
                    row["shares_outstanding"] = latest
                    if cut >= 2 and arrays["val"][cut - 2] > 0:
                        row["shares_growth_qoq"] = latest / float(arrays["val"][cut - 2]) - 1.0
                    if cut >= 5 and arrays["val"][cut - 5] > 0:
                        row["shares_growth_yoy"] = latest / float(arrays["val"][cut - 5]) - 1.0
                    if has_price and pd.notna(cand.get("close_unadj_prev")):
                        row["float_dollar"] = latest * float(cand["close_unadj_prev"])
                        if has_gap and pd.notna(cand.get("gap")) and latest > 0:
                            # Gap size relative to the share count: the same
                            # move on a small float is a far stronger signal.
                            row["gap_to_float"] = float(cand["gap"]) / np.log1p(latest)
            rows.append(row)

    return pd.DataFrame(rows)


def assert_no_future_share_counts(
    shares: pd.DataFrame, decision: pd.Timestamp
) -> None:
    """Tripwire: no share count used for a decision may post-date it."""
    if shares.empty or "known_from" not in shares.columns:
        return
    late = shares[shares["known_from"] >= decision]
    assert late.empty, (
        f"{len(late)} share counts became public at or after {decision} "
        f"(earliest offender {late['known_from'].min()})"
    )
