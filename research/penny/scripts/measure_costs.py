"""measure_costs.py — Gate A. Measure the execution cost bar.

Every conclusion in rounds 1 and 2 was evaluated against an assumed 437 bps
round-trip cost that nobody measured. Round 2's best model reached +1.51%
gross, so if the true bar is materially lower, the cost model rather than the
model is what has been failing the hypothesis.

Samples candidate symbol-days, pulls the NBBO inside the 09:35 minute, and
reports the real distribution -- by price and liquidity bucket, because a
single average is meaningless across a population spanning $0.50 names and $9
ones.

    python research/penny/scripts/measure_costs.py --sample-days 400
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
logger = logging.getLogger("costs")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUOTES_PATH = bars.DATA_ROOT / "quote_summary.parquet"
COST_PATH = bars.DATA_ROOT / "cost_table.parquet"

# The assumed bar every previous result was judged against.
ASSUMED_ROUND_TRIP_BPS = 437.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure execution costs from NBBO quotes.")
    parser.add_argument("--sample-days", type=int, default=400)
    parser.add_argument("--participation", type=float, default=0.01)
    parser.add_argument("--impact-coefficient", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv(_PROJECT_ROOT / ".env")

    trades = pd.read_parquet(bars.DATA_ROOT / "trades.parquet")
    logger.info("trades: %s over %d sessions", f"{len(trades):,}", trades["date"].nunique())

    if QUOTES_PATH.exists() and not args.force:
        summary = pd.read_parquet(QUOTES_PATH)
        logger.info("quote summary: %s rows (cached)", f"{len(summary):,}")
    else:
        key, secret = resolve_credentials()
        client = alpaca_fetch.make_data_client(key, secret)

        # Sample sessions rather than take the most recent: spreads have
        # narrowed structurally over a decade, so the tail of the sample would
        # understate the cost across the study window.
        rng = np.random.default_rng(17)
        all_days = np.sort(trades["date"].unique())
        chosen = np.sort(rng.choice(all_days, size=min(args.sample_days, len(all_days)),
                                    replace=False))
        by_day = trades.groupby("date")["symbol"].apply(lambda s: sorted(set(s)))

        frames: List[pd.DataFrame] = []
        failed = 0
        t0 = time.time()
        for i, day in enumerate(chosen, start=1):
            day_ts = pd.Timestamp(day)
            try:
                raw = Q.fetch_quotes_for_minute(
                    client, by_day.loc[day], day_ts.strftime("%Y-%m-%d")
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if failed <= 3:
                    logger.warning("%s: %s", day_ts.date(), str(exc)[:140])
                continue
            part = Q.summarise_quotes(raw)
            if not part.empty:
                frames.append(part)
            if i % 50 == 0:
                rate = i / max(time.time() - t0, 1e-9)
                logger.info(
                    "%d/%d sessions | %s symbol-days | ~%.0f min left",
                    i, len(chosen), f"{sum(len(f) for f in frames):,}",
                    (len(chosen) - i) / max(rate, 1e-9) / 60,
                )

        if not frames:
            raise SystemExit("no quotes retrieved")
        summary = pd.concat(frames, ignore_index=True)
        summary.to_parquet(QUOTES_PATH, index=False)
        logger.info("wrote %s (%s symbol-days, %d sessions failed)",
                    QUOTES_PATH, f"{len(summary):,}", failed)

    # --- the measurement ---------------------------------------------------
    cost = Q.build_cost_table(
        summary, trades, participation_cap=args.participation,
        impact_coefficient=args.impact_coefficient,
    )
    matched = cost.dropna(subset=["half_spread_bps"])
    cost.to_parquet(COST_PATH, index=False)

    hs = matched["half_spread_bps"]
    rt = matched["measured_round_trip_bps"]

    logger.info("=" * 66)
    logger.info("MEASURED vs ASSUMED (n = %s matched trades)", f"{len(matched):,}")
    logger.info("  half-spread   assumed 118.0 bps | measured median %.1f bps", hs.median())
    logger.info("                measured p25 %.1f  p75 %.1f  p90 %.1f",
                hs.quantile(0.25), hs.quantile(0.75), hs.quantile(0.90))
    logger.info("  impact        assumed 100.0 bps | measured median %.1f bps",
                matched["impact_bps"].median())
    logger.info("  depth cost    assumed   0.0 bps | measured median %.1f bps",
                matched["depth_shortfall_bps"].median())
    logger.info("  ROUND TRIP    assumed %.0f bps | measured median %.1f bps",
                ASSUMED_ROUND_TRIP_BPS, rt.median())
    logger.info("                measured mean %.1f  p75 %.1f  p90 %.1f",
                rt.mean(), rt.quantile(0.75), rt.quantile(0.90))
    logger.info("=" * 66)

    price_buckets = Q.bucket_summary(matched, "close_unadj_prev", [0, 1, 2, 3, 5, 10])
    adv_buckets = Q.bucket_summary(matched, "adv_20", [0, 3e5, 1e6, 5e6, 2e7, 1e12])
    logger.info("by price bucket:\n%s", price_buckets.to_string(index=False))
    logger.info("by trailing-ADV bucket:\n%s", adv_buckets.to_string(index=False))

    # Gate A: does the measured bar leave round 2's best gross viable?
    best_gross_round2 = 0.0151
    verdict = {
        "n_matched": int(len(matched)),
        "assumed_round_trip_bps": ASSUMED_ROUND_TRIP_BPS,
        "measured_round_trip_bps_median": float(rt.median()),
        "measured_round_trip_bps_mean": float(rt.mean()),
        "measured_half_spread_bps_median": float(hs.median()),
        "measured_impact_bps_median": float(matched["impact_bps"].median()),
        "measured_depth_bps_median": float(matched["depth_shortfall_bps"].median()),
        "round2_best_gross": best_gross_round2,
        "gate_a_pass": bool(rt.median() / 10_000 < best_gross_round2),
        "price_buckets": price_buckets.to_dict(orient="records"),
        "adv_buckets": adv_buckets.to_dict(orient="records"),
        "participation": args.participation,
        "impact_coefficient": args.impact_coefficient,
    }
    (bars.DATA_ROOT / "cost_summary.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8"
    )

    logger.info(
        "GATE A: measured median round-trip %.1f bps vs round-2 best gross %.0f bps -> %s",
        rt.median(), 10_000 * best_gross_round2,
        "PASS (cost is beatable)" if verdict["gate_a_pass"] else "FAIL (cost exceeds gross)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
