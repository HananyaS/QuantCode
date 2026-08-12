"""measure_costs_r4.py — the cost bar, measured honestly.

Round 3 measured a cost bar and concluded it was the binding constraint. Three
things about that measurement flattered the result, and all three are fixed here.

1. **Depth was charged once for a round trip that crosses the book twice.**
   ``2 * (half_spread + impact) + depth`` assumes the 15:55 exit fills entirely
   at the inside. Depth was 398 of the measured 728 bps, so the largest term was
   half-counted. Here each leg is charged against its own measured book.

2. **Only the ask was ever measured.** A long exit sells into the *bid*, and in
   this universe the two sides are routinely an order of magnitude apart.

3. **The estimate peeked at the entry minute.** Round 3 took the median quote
   over ``[09:35:00, 09:36:00)`` while entry is the 09:35 bar open. Harmless
   while cost was evaluation-only; look-ahead the moment cost informs the
   decision, which is exactly what round 4 does. So three windows are fetched:

       09:34  cost_hat        -- decision input, strictly before entry
       09:35  entry leg       -- what was actually paid to get in
       15:55  exit leg        -- what was actually paid to get out

Round 3 also sampled 400 of 2,522 sessions and dropped the unmatched 44% of
trades from the bootstrap rather than defaulting them. This runs every session,
and anything still unquoted is *filtered as untradeable* rather than silently
excluded -- you cannot trade what you cannot quote, so those symbol-days are
outside the strategy by construction, not a gap to apologise for.

Resumable per session: round 3's barrier sweep died to a laptop sleep and lost
hours of work, so nothing here holds progress in memory.

    python research/penny/scripts/measure_costs_r4.py
    python research/penny/scripts/measure_costs_r4.py --limit-days 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import alpaca_fetch, bars, quotes as Q  # noqa: E402
from research.penny.scripts.fetch_daily_history import resolve_credentials  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("costs4")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUOTE_DIR = bars.DATA_ROOT / "quotes_r4"
SUMMARY_PATH = bars.DATA_ROOT / "quote_summary_r4.parquet"

# Window label -> clock minute. Order matters only for logging.
WINDOWS = {
    "decision": Q.DECISION_MINUTE,
    "entry": Q.ENTRY_MINUTE,
    "exit": Q.EXIT_MINUTE,
}


def session_path(day: pd.Timestamp) -> Path:
    return QUOTE_DIR / f"dt={day:%Y-%m-%d}" / "quotes.parquet"


def fetch_session(client, symbols: List[str], day: pd.Timestamp) -> pd.DataFrame:
    """All three windows for one session, tagged by window."""
    frames = []
    for label, minute in WINDOWS.items():
        raw = Q.fetch_quotes_for_minute(
            client, symbols, day.strftime("%Y-%m-%d"), minute=minute
        )
        if raw.empty:
            continue
        # The tripwire only makes sense for the decision window; the other two
        # are meant to sit at/after entry.
        #
        # Trim first, then assert. Alpaca's `end` is an inclusive instant, so a
        # request for [09:34:00, 09:35:00) returns the 09:35:00.000 quote when
        # one exists -- observed on ~3% of sessions in the live run. Without the
        # trim the assertion aborts the session and silently costs coverage.
        if label == "decision":
            raw = Q.trim_to_before(raw, Q.ENTRY_MINUTE)
            if raw.empty:
                continue
            Q.assert_decision_quotes_are_pre_entry(raw)
        part = Q.summarise_quotes(raw)
        if part.empty:
            continue
        part["window"] = label
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure round-4 execution costs.")
    parser.add_argument("--candidates", default=str(bars.DATA_ROOT / "candidates_r4.parquet"))
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-summary", action="store_true",
                        help="skip fetching; just re-aggregate what is cached")
    args = parser.parse_args(argv)

    load_dotenv(_PROJECT_ROOT / ".env")

    cand = pd.read_parquet(args.candidates, columns=["symbol", "date"])
    days = sorted(cand["date"].unique())
    if args.limit_days:
        days = days[-args.limit_days:]
    by_day = cand.groupby("date")["symbol"].apply(lambda s: sorted(set(s)))
    logger.info("%s candidate symbol-days across %d sessions",
                f"{len(cand):,}", len(days))

    if not args.rebuild_summary:
        key, secret = resolve_credentials()
        client = alpaca_fetch.make_data_client(key, secret)

        fetched = skipped = failed = 0
        t0 = time.time()
        for i, day in enumerate(days, start=1):
            day_ts = pd.Timestamp(day)
            path = session_path(day_ts)
            if path.exists() and not args.force:
                skipped += 1
                continue
            try:
                part = fetch_session(client, list(by_day.loc[day]), day_ts)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if failed <= 5:
                    logger.warning("%s: %s", day_ts.date(), str(exc)[:160])
                continue
            if part.empty:
                failed += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(path, index=False, compression="snappy")
            fetched += 1
            if fetched % 25 == 0:
                rate = fetched / max(time.time() - t0, 1e-9)
                logger.info(
                    "%d/%d sessions | %d fetched / %d skipped / %d failed | ~%.0f min left",
                    i, len(days), fetched, skipped, failed,
                    (len(days) - i) / max(rate, 1e-9) / 60,
                )
        logger.info("fetch done: %d fetched, %d skipped, %d failed",
                    fetched, skipped, failed)

    # --- aggregate -----------------------------------------------------------
    parts = sorted(QUOTE_DIR.glob("dt=*/quotes.parquet"))
    if not parts:
        raise SystemExit("no quote sessions cached")
    summary = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    summary.to_parquet(SUMMARY_PATH, index=False, compression="snappy")
    logger.info("wrote %s (%s rows, %d sessions)",
                SUMMARY_PATH, f"{len(summary):,}", len(parts))

    wide = summary.pivot_table(
        index=["symbol", "date"], columns="window",
        values=["half_spread_bps", "ask_notional", "bid_notional"], aggfunc="first",
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    coverage = {
        label: float(wide[f"half_spread_bps_{label}"].notna().mean())
        for label in WINDOWS
        if f"half_spread_bps_{label}" in wide.columns
    }
    logger.info("=" * 66)
    logger.info("WINDOW COVERAGE (share of measured symbol-days)")
    for label, frac in coverage.items():
        logger.info("  %-9s %5.1f%%", label, 100 * frac)

    stats = {}
    for label in WINDOWS:
        col = f"half_spread_bps_{label}"
        if col not in wide.columns:
            continue
        s = wide[col].dropna()
        stats[label] = {
            "n": int(len(s)),
            "half_spread_bps_median": float(s.median()),
            "half_spread_bps_p90": float(s.quantile(0.90)),
        }
        logger.info("  %-9s half-spread median %.1f bps  p90 %.1f  (n=%s)",
                    label, s.median(), s.quantile(0.90), f"{len(s):,}")

    # The decision-vs-entry gap is the size of the look-ahead round 3 carried.
    if {"half_spread_bps_decision", "half_spread_bps_entry"} <= set(wide.columns):
        both = wide.dropna(subset=["half_spread_bps_decision", "half_spread_bps_entry"])
        if not both.empty:
            drift = both["half_spread_bps_entry"] - both["half_spread_bps_decision"]
            logger.info(
                "  entry minus decision half-spread: median %+.1f bps (n=%s) "
                "-- the look-ahead round 3 was carrying",
                drift.median(), f"{len(both):,}",
            )
            stats["decision_to_entry_drift_bps_median"] = float(drift.median())

    # Depth: the term that decides everything.
    for side, label in (("ask_notional_entry", "entry ask"), ("bid_notional_exit", "exit bid")):
        if side in wide.columns:
            s = wide[side].dropna()
            if not s.empty:
                logger.info("  %-9s inside notional: median $%s  p25 $%s",
                            label, f"{s.median():,.0f}", f"{s.quantile(0.25):,.0f}")
                stats[side] = {"median": float(s.median()), "p25": float(s.quantile(0.25))}
    logger.info("=" * 66)

    wide.to_parquet(bars.DATA_ROOT / "quote_windows_r4.parquet", index=False,
                    compression="snappy")
    (bars.DATA_ROOT / "cost_summary_r4.json").write_text(
        json.dumps({"coverage": coverage, "windows": stats,
                    "sessions": len(parts), "symbol_days": int(len(wide))},
                   indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("wrote quote_windows_r4.parquet (%s symbol-days)", f"{len(wide):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
