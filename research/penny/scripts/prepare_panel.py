"""prepare_panel.py — a CONTINUOUS panel: every session, filtered only at signal time.

The bug this exists to fix
--------------------------
`forensic_universe.parquet` is pre-filtered: it keeps only rows whose previous
close sat in $0.50-$10.00 with a computable ADV. Computing `shift(1)`,
`shift(5)` or a forward path on that frame walks over **surviving rows, not
sessions**. A stock that falls below the $0.50 floor simply vanishes from the
frame and reappears weeks later at a different price, and every time-series
operation silently steps across the hole.

Measured on the first attempt: 13,279 rows had a "previous row" more than ten
calendar days old, 3,815 more than ninety. **886 of 4,670 signals had a
five-day window spanning more than ten calendar days.** MI showed 2026-03-17 at
$0.48 and the next panel row at $10.01 on 2026-07-02, with 73 of 91 intervening
sessions invisible -- so a 90-bar hold skipped exactly the months the stock
spent at its lows and booked the recovery. That produced a +36% holdout result
which was an artefact.

The rule
--------
**Filter at signal time; never filter the series a shift walks over.** This
module keeps every session for every symbol that ever qualifies, computes all
derived columns on the continuous series, and marks eligibility as a boolean
column (`in_universe`) for the selection step to use.

`session_gap_days` is retained so a signal whose own lookback spans a data hole
can be rejected rather than trusted.

    python research/penny/scripts/prepare_panel.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.dates import SEAL  # noqa: E402
from research.penny.etl import bars  # noqa: E402
from research.penny.etl.corporate_actions import (  # noqa: E402
    flag_forward_splits, flag_reverse_splits,
)
from research.penny.strategy.engine import cost_for  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("panel")

WINDOWS = (3, 5, 10, 20)
PRICE_MIN, PRICE_MAX = 0.50, 10.00
READ = ["symbol", "date", "open", "high", "low", "close", "volume",
        "open_unadj", "high_unadj", "low_unadj", "close_unadj"]


def _derive(d: pd.DataFrame) -> pd.DataFrame:
    """All of this runs on CONSECUTIVE sessions, by construction."""
    d = d.sort_values(["symbol", "date"])
    g = d.groupby("symbol", sort=False)

    d["session_gap_days"] = (d["date"] - g["date"].shift(1)).dt.days
    d["prev_close_unadj"] = g["close_unadj"].shift(1)
    d["prev_high_unadj"] = g["high_unadj"].shift(1)
    d["prev_low_unadj"] = g["low_unadj"].shift(1)
    d["prev_volume"] = g["volume"].shift(1)
    d["c2c_unadj"] = d["close_unadj"] / d["prev_close_unadj"] - 1.0
    d["c2c_prev_unadj"] = g["c2c_unadj"].shift(1)

    d["dv"] = d["close_unadj"] * d["volume"]
    d["adv20"] = g["dv"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean())

    d["o2h"] = d["high_unadj"] / d["open_unadj"] - 1.0

    for n in WINDOWS:
        d[f"close_{n}b_unadj"] = g["close_unadj"].shift(n)
        d[f"ret_{n}d_unadj"] = d["close_unadj"] / d[f"close_{n}b_unadj"] - 1.0
        d[f"low_{n}d_unadj"] = g["low_unadj"].transform(
            lambda s, w=n: s.rolling(w + 1, min_periods=2).min())
        # calendar span of the lookback -- a window straddling a data hole is
        # not a five-day crash and must not be treated as one
        d[f"span_{n}d"] = (d["date"] - g["date"].shift(n)).dt.days

    d = flag_reverse_splits(d, prev_close_col="prev_close_unadj",
                            open_col="open_unadj", o2h_col="o2h")
    d = flag_forward_splits(d, prev_close_col="prev_close_unadj",
                            open_col="open_unadj", high_col="high_unadj",
                            low_col="low_unadj")
    d["any_split"] = d["is_suspect_split"] | d["is_possible_split"]

    # eligibility is a COLUMN, not a filter
    d["in_universe"] = (
        d["prev_close_unadj"].between(PRICE_MIN, PRICE_MAX)
        & d["adv20"].notna()
        & np.isfinite(d["c2c_unadj"])
    ).fillna(False)
    d["cost"] = cost_for(d["adv20"])
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--batch-size", type=int, default=700)
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "panel_strategy.parquet"))
    args = ap.parse_args(argv)

    years = [y for y in range(args.start_year, args.end_year + 1)
             if (bars.DATA_ROOT / "daily" / f"bars_{y}.parquet").exists()]
    logger.info("years: %s", years)

    # Only symbols that ever qualify -- but for those, EVERY session is kept.
    uni = pd.read_parquet(bars.DATA_ROOT / "forensic_universe.parquet",
                          columns=["symbol"])
    symbols = np.array(sorted(uni["symbol"].unique()))
    logger.info("symbols that ever qualify: %s", f"{len(symbols):,}")

    frames: List[pd.DataFrame] = []
    batches = [symbols[i:i + args.batch_size]
               for i in range(0, len(symbols), args.batch_size)]
    for i, batch in enumerate(batches, 1):
        want = set(batch.tolist())
        parts = []
        for y in years:
            df = pd.read_parquet(bars.DATA_ROOT / "daily" / f"bars_{y}.parquet",
                                 columns=READ)
            parts.append(df[df["symbol"].isin(want)])
        d = pd.concat(parts, ignore_index=True)
        if d.empty:
            continue
        d = _derive(d)
        frames.append(d)
        logger.info("  batch %d/%d: %s rows (running %s, eligible %s)", i, len(batches),
                    f"{len(d):,}", f"{sum(len(f) for f in frames):,}",
                    f"{int(sum(f['in_universe'].sum() for f in frames)):,}")

    u = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    u = u.reset_index(drop=True)

    floats = [c for c in u.columns if u[c].dtype == np.float64 and c != "adv20"]
    u[floats] = u[floats].astype(np.float32)

    gaps = u["session_gap_days"]
    logger.info("panel: %s rows | %s symbols | %s .. %s", f"{len(u):,}",
                f"{u['symbol'].nunique():,}", str(u["date"].min())[:10],
                str(u["date"].max())[:10])
    logger.info("eligible (in_universe) rows: %s (%.1f%%)",
                f"{int(u['in_universe'].sum()):,}", 100 * u["in_universe"].mean())
    logger.info("consecutive-row gaps >10 days: %s (%.3f%%) | >90 days: %s",
                f"{int((gaps > 10).sum()):,}", 100 * (gaps > 10).mean(),
                f"{int((gaps > 90).sum()):,}")

    u.to_parquet(args.out, index=False)
    logger.info("wrote %s", args.out)
    dev = int((u["date"] < SEAL).sum())
    logger.info("signal-date split: DEV %s | SEALED %s", f"{dev:,}", f"{len(u)-dev:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
