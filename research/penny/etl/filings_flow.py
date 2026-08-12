"""filings_flow.py — insider and institutional flow from the EDGAR index.

What this adds that ``etl.edgar`` deliberately does not
------------------------------------------------------
``edgar.build_pit_features`` filters the index down to shelf, takedown,
delisting and 8-K forms, with the comment that Forms 3/4/144 "carry none of the
supply signal this study is after". That is correct as far as it goes: rounds
1-4 were looking for *dilution*, and insider paperwork says nothing about it.

But it means **568,896 Form 4 filings across 2,467 symbols sat unused for four
rounds**, along with 73,880 SC 13G and 22,580 SC 13D filings. Those are not
supply signals; they are *demand* signals, and a different question:

* **Form 4** — an officer, director or 10% holder transacted. Insider purchases
  in small caps are among the better-documented anomalies (Lakonishok & Lee
  2001; Cohen, Malloy & Pomorski 2012, "Decoding Inside Information").
* **SC 13D** — someone crossed 5% *with intent to influence*. An activist stake.
* **SC 13G** — someone crossed 5% passively. Institutional accumulation.
* **Form 3** — a new insider was appointed; governance changed.
* **25-NSE** — the exchange filed to delist. Terminal.

Counts are normalised against each symbol's own trailing rate
--------------------------------------------------------------
A raw count is confounded by company size and filer habit: a firm with twelve
insiders files twelve Form 4s for one board decision. What carries information
is the *departure* from that symbol's own baseline, so every count is also
reported as a ratio to its trailing 365-day rate.

Point-in-time discipline is identical to ``etl.edgar``: a filing counts only if
its ``accepted_et`` is strictly before the decision instant. Filing *date* is
not sufficient — an 18:00 acceptance carries the same filing date as the
morning that preceded it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Form families. Amendments count: a 13G/A raising a stake is news.
INSIDER_TXN_FORMS = {"4", "4/A"}
INSIDER_NEW_FORMS = {"3", "3/A"}
ACTIVIST_FORMS = {"SC 13D", "SC 13D/A"}
PASSIVE_FORMS = {"SC 13G", "SC 13G/A"}
DELIST_FORMS = {"25-NSE", "25"}

NS_PER_DAY = 86_400_000_000_000


@dataclass
class FlowConfig:
    """Windows over which flow is counted, in days."""

    short_days: int = 30
    long_days: int = 90
    baseline_days: int = 365


COUNT_FEATURES: List[str] = [
    "insider_txn_30d", "insider_txn_90d", "insider_new_180d",
    "activist_365d", "passive_365d",
]

# Standardised deviations from the symbol's own trailing rate, not raw ratios.
# A ratio with a smoothing constant in the denominator damps small-count
# symbols hardest, which is exactly where a burst is most informative. A
# Poisson-style z -- (observed - expected) / sqrt(expected + 1) -- is scale-free
# and stays finite when the baseline is zero.
ABNORMAL_FEATURES: List[str] = [
    "insider_txn_abnormal_30d", "passive_abnormal_365d", "filing_burst_30d",
]

RECENCY_FEATURES: List[str] = [
    "days_since_insider_txn", "days_since_activist",
    "days_since_passive", "days_since_delist_notice",
]

FLOW_FEATURES: List[str] = COUNT_FEATURES + ABNORMAL_FEATURES + RECENCY_FEATURES


def _abnormal(observed: float, baseline_count: float, window_days: float,
              baseline_days: float) -> float:
    """Poisson-style standardised excess over the symbol's own trailing rate."""
    expected = baseline_count * (window_days / baseline_days)
    return float((observed - expected) / np.sqrt(expected + 1.0))


def _epoch_ns(s: pd.Series) -> np.ndarray:
    """Tz-aware timestamps to UTC epoch nanoseconds.

    Integers rather than datetime64 on purpose: a tz-aware Series returns object
    dtype from ``.to_numpy()``, which ``searchsorted`` cannot compare against
    and which would raise rather than silently mis-sort.
    """
    return s.dt.tz_convert("UTC").astype("int64").to_numpy()


def build_flow_features(
    candidates: pd.DataFrame,
    filings: pd.DataFrame,
    config: Optional[FlowConfig] = None,
    decision_time: str = "09:35",
) -> pd.DataFrame:
    """Point-in-time insider and institutional flow for each candidate day."""
    config = config or FlowConfig()

    base = candidates[["symbol", "date"]].copy()
    if filings.empty:
        logger.warning("filings_flow: no filings supplied; features will be null")
        for col in FLOW_FEATURES:
            base[col] = np.nan
        base["flow_has_coverage"] = False
        return base

    f = filings.dropna(subset=["accepted_et"]).copy()
    covered = set(f["symbol"].unique())

    f["is_txn"] = f["form"].isin(INSIDER_TXN_FORMS)
    f["is_new"] = f["form"].isin(INSIDER_NEW_FORMS)
    f["is_activist"] = f["form"].isin(ACTIVIST_FORMS)
    f["is_passive"] = f["form"].isin(PASSIVE_FORMS)
    f["is_delist"] = f["form"].isin(DELIST_FORMS)

    keep = f[["is_txn", "is_new", "is_activist", "is_passive", "is_delist"]].any(axis=1)
    f = f[keep]
    logger.info("filings_flow: %s flow filings of %s indexed",
                f"{len(f):,}", f"{len(filings):,}")

    by_symbol: Dict[str, Dict[str, np.ndarray]] = {}
    for symbol, group in f.sort_values("accepted_et").groupby("symbol", sort=False):
        by_symbol[symbol] = {
            "t": _epoch_ns(group["accepted_et"]),
            "txn": group["is_txn"].to_numpy(),
            "new": group["is_new"].to_numpy(),
            "activist": group["is_activist"].to_numpy(),
            "passive": group["is_passive"].to_numpy(),
            "delist": group["is_delist"].to_numpy(),
        }

    short_ns = config.short_days * NS_PER_DAY
    long_ns = config.long_days * NS_PER_DAY
    base_ns = config.baseline_days * NS_PER_DAY
    half_year_ns = 180 * NS_PER_DAY

    rows: List[Dict] = []
    for symbol, group in candidates.groupby("symbol", sort=False):
        arrays = by_symbol.get(symbol)
        has_coverage = symbol in covered

        for date in group["date"]:
            decision = pd.Timestamp(f"{date:%Y-%m-%d} {decision_time}",
                                    tz="America/New_York")
            row: Dict = {"symbol": symbol, "date": date,
                         "flow_has_coverage": has_coverage}
            # Counts default to zero, recencies to NaN. The distinction is real:
            # for a symbol EDGAR covers, "no insider filed in 30 days" is an
            # observation worth zero, whereas "days since the last one" is
            # genuinely unknown when there has never been one. Collapsing both
            # to NaN would throw away the observation.
            for col in COUNT_FEATURES + ABNORMAL_FEATURES:
                row[col] = 0.0 if has_coverage else np.nan
            for col in RECENCY_FEATURES:
                row[col] = np.nan

            if arrays is None or not len(arrays["t"]):
                rows.append(row)
                continue

            now = decision.tz_convert("UTC").value
            t = arrays["t"]
            cut = int(np.searchsorted(t, now, side="left"))
            if cut == 0:
                rows.append(row)
                continue

            def window_start(span_ns: int) -> int:
                return int(np.searchsorted(t, now - span_ns, side="left"))

            s30, s90, s180, s365 = (window_start(short_ns), window_start(long_ns),
                                    window_start(half_year_ns), window_start(base_ns))

            txn30 = int(arrays["txn"][s30:cut].sum())
            txn90 = int(arrays["txn"][s90:cut].sum())
            txn365 = int(arrays["txn"][s365:cut].sum())
            row["insider_txn_30d"] = float(txn30)
            row["insider_txn_90d"] = float(txn90)
            row["insider_txn_abnormal_30d"] = _abnormal(
                txn30, txn365, config.short_days, config.baseline_days
            )
            row["insider_new_180d"] = float(arrays["new"][s180:cut].sum())
            row["activist_365d"] = float(arrays["activist"][s365:cut].sum())

            passive365 = int(arrays["passive"][s365:cut].sum())
            row["passive_365d"] = float(passive365)
            row["passive_abnormal_365d"] = _abnormal(
                passive365, int(arrays["passive"][:cut].sum()),
                config.baseline_days,
                max((now - int(t[0])) / NS_PER_DAY, config.baseline_days),
            )
            row["filing_burst_30d"] = _abnormal(
                cut - s30, cut - s365, config.short_days, config.baseline_days
            )

            for key, out_key in (("txn", "days_since_insider_txn"),
                                 ("activist", "days_since_activist"),
                                 ("passive", "days_since_passive"),
                                 ("delist", "days_since_delist_notice")):
                hits = np.flatnonzero(arrays[key][:cut])
                if hits.size:
                    row[out_key] = float((now - int(t[hits[-1]])) / NS_PER_DAY)

            rows.append(row)

    out = pd.DataFrame(rows)
    assert len(out) == len(candidates), "flow build changed the row count"
    return out


def assert_no_future_flow(
    features: pd.DataFrame,
    filings: pd.DataFrame,
    symbol: str,
    date: pd.Timestamp,
    decision_time: str = "09:35",
) -> None:
    """Tripwire: the most recent insider filing used must precede the decision.

    Checked against the raw index rather than recomputed from the feature, so
    the assertion can actually disagree with the thing it is checking.
    """
    row = features[(features["symbol"] == symbol) & (features["date"] == date)]
    if row.empty or pd.isna(row.iloc[0].get("days_since_insider_txn")):
        return

    decision = pd.Timestamp(f"{date:%Y-%m-%d} {decision_time}", tz="America/New_York")
    mine = filings[(filings["symbol"] == symbol)
                   & filings["form"].isin(INSIDER_TXN_FORMS)].dropna(subset=["accepted_et"])
    prior = mine[mine["accepted_et"] < decision]
    assert not prior.empty, (
        f"{symbol} {date:%Y-%m-%d}: feature reports an insider filing but none "
        f"was accepted before {decision}"
    )

    implied = decision - pd.Timedelta(days=float(row.iloc[0]["days_since_insider_txn"]))
    latest = prior["accepted_et"].max()
    assert abs((implied - latest).total_seconds()) < 90, (
        f"{symbol} {date:%Y-%m-%d}: days_since_insider_txn points at {implied}, "
        f"but the latest prior acceptance is {latest}"
    )
