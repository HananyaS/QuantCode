"""top30_sample.py — the 30 largest close-to-close gainers of the last two months.

Selection is mechanical: rank distinct tickers by their largest single-session
close-to-close gain inside a fixed window, keep the top 30. One event per ticker,
because the deliverable is a study of thirty companies rather than thirty rows.

Repeat offenders are recorded rather than deduplicated away -- a ticker that
explodes twice is itself a finding.

Everything downstream is computed strictly before 09:35 on the event date; this
file only fixes *which* events are studied.

    python research/penny/scripts/top30_sample.py --months 2 --n 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("top30")

MOVE_THRESHOLD = 0.50
INTRADAY_RUNNER_MIN_O2H = 0.50


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Top-N gainers sample.")
    p.add_argument("--months", type=int, default=2)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out", default=str(bars.DATA_ROOT / "top30_sample.parquet"))
    args = p.parse_args(argv)

    u = pd.read_parquet(bars.DATA_ROOT / "forensic_universe.parquet")
    end = u["date"].max()
    start = end - pd.Timedelta(days=30 * args.months)
    window = u[(u["date"] >= start) & (u["date"] <= end)]

    events = window[window["c2c"] >= MOVE_THRESHOLD].copy()
    logger.info("window %s .. %s | %s symbol-days | %d events >=+%.0f%% across %d tickers",
                start.date(), end.date(), f"{len(window):,}", len(events),
                100 * MOVE_THRESHOLD, events["symbol"].nunique())

    # Largest event per ticker, then the top N tickers by that magnitude.
    best = events.sort_values("c2c", ascending=False).drop_duplicates("symbol")
    sample = best.head(args.n).copy()

    # How often did each name do this inside the window?
    counts = events.groupby("symbol").size().rename("events_in_window")
    sample = sample.merge(counts, on="symbol", how="left")

    sample["species"] = np.where(sample["o2h"] >= INTRADAY_RUNNER_MIN_O2H,
                                 "intraday_runner", "gap_driven")
    sample = sample.sort_values("c2c", ascending=False).reset_index(drop=True)
    sample.to_parquet(args.out, index=False)

    cols = ["symbol", "date", "species", "prev_close_unadj", "open", "close",
            "c2c", "gap", "o2h", "o2c", "adv20", "volume", "events_in_window"]
    logger.info("=" * 120)
    logger.info("TOP %d GAINERS, %s .. %s\n%s", args.n, start.date(), end.date(),
                sample[cols].assign(date=sample["date"].dt.date)
                .to_string(index=False, float_format="%.3f"))
    logger.info("-" * 120)
    logger.info("species: %s", sample["species"].value_counts().to_dict())
    logger.info("repeat offenders (>1 event in window): %d",
                int((sample["events_in_window"] > 1).sum()))
    logger.info("ADV: min $%s  median $%s  max $%s",
                f"{sample['adv20'].min():,.0f}", f"{sample['adv20'].median():,.0f}",
                f"{sample['adv20'].max():,.0f}")

    (bars.DATA_ROOT / "top30_sample.json").write_text(json.dumps({
        "window_start": str(start.date()), "window_end": str(end.date()),
        "window_rows": int(len(window)),
        "events_in_window": int(len(events)),
        "tickers_in_window": int(events["symbol"].nunique()),
        "n_selected": int(len(sample)),
        "symbols": sample["symbol"].tolist(),
        "dates": [str(d.date()) for d in sample["date"]],
    }, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
