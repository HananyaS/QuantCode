"""forensic_sample.py — pick the ten most recent 50%+ movers, reproducibly.

Six rounds averaged over tens of thousands of candidates and found the mean is
~0. Averaging is exactly what hides a precursor present in 2% of cases, so this
inverts the method: read ten individual explosions closely, catalogue what was
visible beforehand, and hand each candidate precursor back to the statistical
machinery to be tested against its own base rate.

Selection is deliberately mechanical -- most recent N events meeting a fixed
condition -- so the sample cannot drift toward whichever names make the nicest
story.

**These names are selected because they moved.** Every pattern found in them is
conditioned on success, which is why ``forensic_precursors.py`` exists and why
nothing here is a result on its own.

    python research/penny/scripts/forensic_sample.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("forensic")

PRICE_MIN, PRICE_MAX = 0.50, 10.00
MOVE_THRESHOLD = 0.50

# A gap-driven mover has already made its move before the bell; an intraday
# runner makes it after. They are different species and may have different
# precursors, so the split is recorded rather than averaged over.
INTRADAY_RUNNER_MIN_O2H = 0.50


def build_universe(start_year: int, end_year: int) -> pd.DataFrame:
    """Every penny-universe symbol-day with the outcome columns attached."""
    frames = []
    for year in range(start_year, end_year + 1):
        path = bars.DATA_ROOT / "daily" / f"bars_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(
                path, columns=["symbol", "date", "open", "high", "low", "close",
                               "volume", "close_unadj"]))
    d = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])

    g = d.groupby("symbol", sort=False)
    d["prev_close"] = g["close"].shift(1)
    d["prev_close_unadj"] = g["close_unadj"].shift(1)
    d["dv"] = d["close"] * d["volume"]
    d["adv20"] = g["dv"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    d["prev_volume"] = g["volume"].shift(1)

    d["gap"] = d["open"] / d["prev_close"] - 1.0
    d["c2c"] = d["close"] / d["prev_close"] - 1.0
    d["o2h"] = d["high"] / d["open"] - 1.0
    d["l2h"] = d["high"] / d["low"] - 1.0
    d["o2c"] = d["close"] / d["open"] - 1.0

    universe = d[
        d["prev_close_unadj"].between(PRICE_MIN, PRICE_MAX)
        & d["adv20"].notna()
        & np.isfinite(d["gap"])
        & np.isfinite(d["c2c"])
    ].copy()
    logger.info("penny universe: %s symbol-days, %d symbols, %s .. %s",
                f"{len(universe):,}", universe["symbol"].nunique(),
                universe["date"].min().date(), universe["date"].max().date())
    return universe


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Select the forensic sample.")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--out", default=str(bars.DATA_ROOT / "forensic_sample.parquet"))
    args = parser.parse_args(argv)

    universe = build_universe(args.start_year, args.end_year)
    universe.to_parquet(bars.DATA_ROOT / "forensic_universe.parquet", index=False)

    movers = universe[universe["c2c"] >= MOVE_THRESHOLD]
    base_rate = len(movers) / max(len(universe), 1)
    logger.info("+%.0f%% close-to-close events: %s of %s  (base rate %.3f%%)",
                100 * MOVE_THRESHOLD, f"{len(movers):,}", f"{len(universe):,}",
                100 * base_rate)

    sample = movers.sort_values("date", ascending=False).head(args.n).copy()
    sample["species"] = np.where(sample["o2h"] >= INTRADAY_RUNNER_MIN_O2H,
                                 "intraday_runner", "gap_driven")
    sample = sample.sort_values("date", ascending=False).reset_index(drop=True)
    sample.to_parquet(args.out, index=False)

    cols = ["symbol", "date", "species", "prev_close_unadj", "open", "close",
            "c2c", "gap", "o2h", "l2h", "adv20", "volume"]
    logger.info("=" * 108)
    logger.info("FORENSIC SAMPLE -- the %d most recent +%.0f%% movers\n%s",
                args.n, 100 * MOVE_THRESHOLD,
                sample[cols].assign(date=sample["date"].dt.date)
                .to_string(index=False, float_format="%.3f"))
    logger.info("-" * 108)
    logger.info("species: %s", sample["species"].value_counts().to_dict())
    logger.info("ADV range: $%s .. $%s (median $%s)",
                f"{sample['adv20'].min():,.0f}", f"{sample['adv20'].max():,.0f}",
                f"{sample['adv20'].median():,.0f}")

    (bars.DATA_ROOT / "forensic_sample.json").write_text(json.dumps({
        "n": int(len(sample)),
        "threshold": MOVE_THRESHOLD,
        "universe_rows": int(len(universe)),
        "n_events": int(len(movers)),
        "base_rate": float(base_rate),
        "symbols": sample["symbol"].tolist(),
        "dates": [str(d.date()) for d in sample["date"]],
        "species": sample["species"].tolist(),
    }, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
