"""prepare_panel.py — one enriched, unadjusted panel for the strategy search.

Every derived column here is computed from the **unadjusted** series, because
these columns feed *filters* and an adjusted level encodes future corporate
actions (LOG §72). Returns and intraday ratios would be adjustment-invariant
either way; computing them unadjusted as well removes the chance of picking the
wrong one at the call site.

One panel, not two
------------------
The search seals 2024-01-01 onward. The seal is applied to **signal dates**, not
to the data itself: a position opened in December 2023 must be allowed to exit
in January 2024, exactly as it would in real trading. Splitting the parquet
would silently truncate every trade near the boundary and quietly bias the
development period toward short holds.

    python research/penny/scripts/prepare_panel.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.strategy.engine import cost_for  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("panel")

WINDOWS = (3, 5, 10, 20)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "panel_strategy.parquet"))
    args = ap.parse_args(argv)

    u = pd.read_parquet(
        bars.DATA_ROOT / "forensic_universe.parquet",
        columns=["symbol", "date", "open_unadj", "high_unadj", "low_unadj",
                 "close_unadj", "close", "volume", "prev_volume", "adv20",
                 "any_split", "is_suspect_split", "is_suspect_fwd_split"],
    ).sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info("universe: %s rows | %s symbols | %s .. %s", f"{len(u):,}",
                f"{u['symbol'].nunique():,}", str(u["date"].min())[:10],
                str(u["date"].max())[:10])

    g = u.groupby("symbol", sort=False)
    u["prev_close_unadj"] = g["close_unadj"].shift(1)
    u["prev_high_unadj"] = g["high_unadj"].shift(1)
    u["prev_low_unadj"] = g["low_unadj"].shift(1)
    u["c2c_unadj"] = u["close_unadj"] / u["prev_close_unadj"] - 1.0
    u["c2c_prev_unadj"] = g["c2c_unadj"].shift(1)

    for n in WINDOWS:
        u[f"close_{n}b_unadj"] = g["close_unadj"].shift(n)
        u[f"ret_{n}d_unadj"] = u["close_unadj"] / u[f"close_{n}b_unadj"] - 1.0
        # lowest low across the decline INCLUDING the signal day -- this is the
        # swing low a structural stop would sit under, and it is fully known.
        u[f"low_{n}d_unadj"] = g["low_unadj"].transform(
            lambda s, w=n: s.rolling(w + 1, min_periods=2).min())

    u["cost"] = cost_for(u["adv20"])

    floats = [c for c in u.columns if u[c].dtype == np.float64 and c != "adv20"]
    u[floats] = u[floats].astype(np.float32)

    out = Path(args.out)
    u.to_parquet(out, index=False)
    logger.info("wrote %s (%s rows, %d cols)", out, f"{len(u):,}", u.shape[1])

    dev = (u["date"] < pd.Timestamp("2024-01-01")).sum()
    logger.info("signal-date split: DEV (<2024) %s rows | SEALED (>=2024) %s rows",
                f"{dev:,}", f"{len(u) - dev:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
