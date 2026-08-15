"""build_universe.py — the canonical penny universe, over any year range.

Why this exists
---------------
`forensic_universe.parquet` is the input to every study in this project, and
**no committed script could reproduce it.** `forensic_sample.py:build_universe`
writes the file but produces only 17 of its 24 columns -- `open_unadj`, the
three reverse-split columns, and the two split flags were added by an ad-hoc run
during the forensics work and never checked in. Re-running the committed builder
would therefore have silently dropped seven columns, and every split filter
downstream would have degraded to a no-op rather than an error.

That matters now because the strategy in §65 was fitted and measured on **18
months** while **eleven years of daily bars sit in `daily/bars_2016..2026`**
unused. Extending the test is the cheapest disconfirming evidence available, and
it needs a builder that produces the whole column set for an arbitrary span.

Two discrepancies fixed here
-----------------------------
1. **ADV is strictly trailing.** `adv20` excludes the signal day
   (`shift(1).rolling(20, min_periods=20)`), matching what the backtests use. A
   crashed name trades enormous volume on its signal day, so including it would
   inflate the liquidity measure by exactly the amount the crash caused.
2. **Both split detectors run.** Reverse splits print as an overnight jump,
   forward splits as an overnight collapse; a buy-the-crash screen is blind to
   the second and would rank it as its most extreme signal.

Memory
------
Eleven years of raw bars is ~16M rows before filtering, so symbols are processed
in batches and each batch is filtered down to the penny universe before being
kept. There are no cross-symbol dependencies, so batching changes nothing about
the result.

    python research/penny/scripts/build_universe.py --start-year 2016 --end-year 2026
    python research/penny/scripts/build_universe.py --start-year 2025 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.etl.corporate_actions import (  # noqa: E402
    flag_forward_splits,
    flag_reverse_splits,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("universe")

PRICE_MIN, PRICE_MAX = 0.50, 10.00
ADV_WINDOW = 20

# The column set the existing 1.5-year file carries. A rebuild that does not
# produce all of these breaks a downstream filter silently, so it is asserted.
REQUIRED_COLUMNS = [
    "adv20", "any_split", "c2c", "close", "close_unadj", "date", "dv", "gap",
    "high", "is_possible_split", "is_suspect_split", "low", "o2c", "o2h",
    "open", "open_unadj", "prev_close", "prev_close_unadj", "prev_volume",
    "split_multiple", "split_nearest_ratio", "split_ratio_error", "symbol",
    "volume",
]

_READ_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume",
                 "open_unadj", "high_unadj", "low_unadj", "close_unadj"]


def _available_years(start_year: int, end_year: int) -> List[int]:
    years = []
    for y in range(start_year, end_year + 1):
        if (bars.DATA_ROOT / "daily" / f"bars_{y}.parquet").exists():
            years.append(y)
        else:
            logger.warning("no bars for %d -- skipping", y)
    if not years:
        raise SystemExit("no daily bar files found in the requested range")
    return years


def _symbols_in(years: List[int]) -> np.ndarray:
    seen: set = set()
    for y in years:
        s = pd.read_parquet(bars.DATA_ROOT / "daily" / f"bars_{y}.parquet",
                            columns=["symbol"])["symbol"].unique()
        seen.update(s.tolist())
    return np.array(sorted(seen))


def _derive(d: pd.DataFrame) -> pd.DataFrame:
    """Trailing state, intraday ratios, and both split detectors."""
    d = d.sort_values(["symbol", "date"])
    g = d.groupby("symbol", sort=False)

    d["prev_close"] = g["close"].shift(1)
    d["prev_close_unadj"] = g["close_unadj"].shift(1)
    d["prev_volume"] = g["volume"].shift(1)
    d["dv"] = d["close"] * d["volume"]
    # Strictly trailing: the signal day's own volume is excluded, because a
    # crash day's volume is caused by the very event being screened on.
    d["adv20"] = g["dv"].transform(
        lambda s: s.shift(1).rolling(ADV_WINDOW, min_periods=ADV_WINDOW).mean())

    # Intraday ratios are adjustment-invariant (the factor cancels within a
    # session), so these are identical on either series.
    d["gap"] = d["open"] / d["prev_close"] - 1.0
    d["c2c"] = d["close"] / d["prev_close"] - 1.0
    d["o2h"] = d["high"] / d["open"] - 1.0
    d["l2h"] = d["high"] / d["low"] - 1.0
    d["o2c"] = d["close"] / d["open"] - 1.0

    # Both detectors run on the UNADJUSTED series. Mixing adjusted with
    # unadjusted here is what produced WOK at $4.8M/share in §64.
    d = flag_reverse_splits(d, prev_close_col="prev_close_unadj",
                            open_col="open_unadj", o2h_col="o2h")
    d = flag_forward_splits(d, prev_close_col="prev_close_unadj",
                            open_col="open_unadj", high_col="high_unadj",
                            low_col="low_unadj")
    d["any_split"] = d["is_suspect_split"] | d["is_possible_split"]
    return d


def build_universe(start_year: int, end_year: int,
                   batch_size: int = 900) -> pd.DataFrame:
    years = _available_years(start_year, end_year)
    symbols = _symbols_in(years)
    logger.info("years %d..%d | %s symbols", years[0], years[-1], f"{len(symbols):,}")

    frames: List[pd.DataFrame] = []
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    for i, batch in enumerate(batches, 1):
        want = set(batch.tolist())
        parts = []
        for y in years:
            df = pd.read_parquet(bars.DATA_ROOT / "daily" / f"bars_{y}.parquet",
                                 columns=_READ_COLUMNS)
            parts.append(df[df["symbol"].isin(want)])
        d = pd.concat(parts, ignore_index=True)
        if d.empty:
            continue
        d = _derive(d)
        keep = d[
            d["prev_close_unadj"].between(PRICE_MIN, PRICE_MAX)
            & d["adv20"].notna()
            & np.isfinite(d["gap"])
            & np.isfinite(d["c2c"])
        ]
        frames.append(keep.copy())
        logger.info("  batch %d/%d: %s symbols -> %s universe rows (running %s)",
                    i, len(batches), f"{len(batch):,}", f"{len(keep):,}",
                    f"{sum(len(f) for f in frames):,}")

    u = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    u = u.reset_index(drop=True)
    logger.info("universe: %s symbol-days | %s symbols | %s .. %s",
                f"{len(u):,}", f"{u['symbol'].nunique():,}",
                str(u["date"].min())[:10], str(u["date"].max())[:10])
    return u


def report_by_year(u: pd.DataFrame) -> None:
    """Split-flag rates per year. 2021 had far more consolidations than 2017,
    and a detector that behaves differently across eras invalidates a
    cross-era backtest before any strategy is applied to it."""
    y = u.assign(year=u["date"].dt.year).groupby("year")
    rep = pd.DataFrame({
        "rows": y.size(),
        "symbols": y["symbol"].nunique(),
        "rev_split_%": 100 * y["is_suspect_split"].mean(),
        "any_split_%": 100 * y["any_split"].mean(),
        "fwd_split_%": 100 * y["is_suspect_fwd_split"].mean(),
        "med_adv20": y["adv20"].median(),
    })
    logger.info("per-year composition:\n%s",
                rep.to_string(float_format=lambda v: f"{v:,.2f}"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--batch-size", type=int, default=900)
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "forensic_universe.parquet"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and report, but do not overwrite the universe file.")
    args = ap.parse_args(argv)

    out = Path(args.out)
    u = build_universe(args.start_year, args.end_year, args.batch_size)

    missing = sorted(set(REQUIRED_COLUMNS) - set(u.columns))
    assert not missing, (
        f"rebuild would drop {len(missing)} columns that downstream studies "
        f"read: {missing} -- every split filter would silently become a no-op"
    )
    report_by_year(u)

    if args.dry_run:
        logger.info("dry run -- %s not written", out)
        return 0

    if out.exists():
        backup = out.with_suffix(".parquet.bak")
        if not backup.exists():
            out.replace(backup)
            logger.info("existing universe preserved at %s", backup)
    u.to_parquet(out, index=False)
    logger.info("wrote %s (%s rows)", out, f"{len(u):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
