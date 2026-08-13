"""forensic_precursors.py — every pattern, measured against its own base rate.

The whole point of this file
----------------------------
Ten names were selected **because they moved**, so any pattern found in them is
conditioned on success. Reading dossiers and reporting what recurs is how one
manufactures a story: with 719,863 symbol-days and a 0.17% base rate, a
"precursor" present in 7 of 10 movers is worthless if it also fires on 40% of
ordinary days.

So each candidate precursor is defined as a function computable for **every**
symbol-day from data strictly prior to it, and reported as:

    hit rate in the ten  |  base rate in the universe  |  P(move | precursor)
    |  lift = P(move | precursor) / P(move)

A precursor is worth building only if lift is large **and** it fires often enough
to produce trades. Both numbers are reported for every candidate, including the
ones that fail -- which is expected to be most of them.

Point-in-time throughout: filings on ``accepted_et`` strictly before 09:35,
trailing statistics shifted, and no same-day price information anywhere.

    python research/penny/scripts/forensic_precursors.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("precursors")

MOVE_THRESHOLD = 0.50
NS_DAY = 86_400_000_000_000

FOREIGN_FORMS = {"6-K", "20-F", "40-F", "F-1", "F-3", "F-4", "F-1/A", "F-3/A",
                 "6-K/A", "20-F/A", "F-6", "F-10"}
SHELF_FORMS = {"S-3", "S-1", "F-3", "F-1", "S-3/A", "S-1/A", "S-1MEF", "S-3MEF"}


def add_daily_precursors(u: pd.DataFrame) -> pd.DataFrame:
    """Trailing price/volume state, all shifted so nothing sees its own day."""
    u = u.sort_values(["symbol", "date"]).copy()
    g = u.groupby("symbol", sort=False)

    # Every window ends at t-1. The shift is the point-in-time guarantee.
    vol = g["volume"].shift(1)
    u["vol5"] = vol.groupby(u["symbol"]).transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    u["vol20"] = vol.groupby(u["symbol"]).transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    u["vol_ratio_5_20"] = u["vol5"] / u["vol20"].replace(0, np.nan)

    close_prev = g["close"].shift(1)
    u["ret_20d_prior"] = close_prev / close_prev.groupby(u["symbol"]).shift(20) - 1.0
    u["ret_5d_prior"] = close_prev / close_prev.groupby(u["symbol"]).shift(5) - 1.0

    hi90 = g["high"].shift(1).groupby(u["symbol"]).transform(
        lambda s: s.rolling(90, min_periods=30).max())
    lo90 = g["low"].shift(1).groupby(u["symbol"]).transform(
        lambda s: s.rolling(90, min_periods=30).min())
    u["pct_off_90d_high"] = close_prev / hi90 - 1.0
    u["pct_above_90d_low"] = close_prev / lo90 - 1.0

    prior_move = (g["close"].shift(1) / g["close"].shift(2) - 1.0) >= MOVE_THRESHOLD
    u["prior_50_90d"] = prior_move.groupby(u["symbol"]).transform(
        lambda s: s.rolling(90, min_periods=1).sum())
    return u


def add_filing_precursors(u: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Filing state strictly before 09:35 on each symbol-day."""
    f = filings.dropna(subset=["accepted_et"]).copy()
    f["is_foreign"] = f["form"].isin(FOREIGN_FORMS)
    f["is_shelf"] = f["form"].isin(SHELF_FORMS)

    by_symbol: Dict[str, Dict[str, np.ndarray]] = {}
    for sym, grp in f.sort_values("accepted_et").groupby("symbol", sort=False):
        by_symbol[sym] = {
            "t": grp["accepted_et"].dt.tz_convert("UTC").astype("int64").to_numpy(),
            "foreign": grp["is_foreign"].to_numpy(),
            "shelf": grp["is_shelf"].to_numpy(),
        }

    decision = (pd.to_datetime(u["date"].dt.strftime("%Y-%m-%d") + " 09:35")
                .dt.tz_localize("America/New_York").dt.tz_convert("UTC")
                .astype("int64").to_numpy())

    n = len(u)
    same_morning = np.zeros(n, dtype=bool)
    filings_30d = np.zeros(n, dtype=float)
    foreign_365 = np.zeros(n, dtype=bool)
    shelf_180 = np.zeros(n, dtype=float)
    has_cov = np.zeros(n, dtype=bool)

    symbols = u["symbol"].to_numpy()

    # Group row indices by symbol once, then binary-search within that symbol.
    idx_by_symbol: Dict[str, List[int]] = {}
    for i, sym in enumerate(symbols):
        idx_by_symbol.setdefault(sym, []).append(i)

    for sym, idxs in idx_by_symbol.items():
        arrays = by_symbol.get(sym)
        if arrays is None:
            continue
        t = arrays["t"]
        for i in idxs:
            now = decision[i]
            cut = int(np.searchsorted(t, now, side="left"))
            has_cov[i] = True
            if cut == 0:
                continue
            midnight = now - (now % NS_DAY)
            s30 = int(np.searchsorted(t, now - 30 * NS_DAY, side="left"))
            s180 = int(np.searchsorted(t, now - 180 * NS_DAY, side="left"))
            s365 = int(np.searchsorted(t, now - 365 * NS_DAY, side="left"))
            same_morning[i] = bool(t[cut - 1] >= midnight)
            filings_30d[i] = cut - s30
            foreign_365[i] = bool(arrays["foreign"][s365:cut].any())
            shelf_180[i] = float(arrays["shelf"][s180:cut].sum())

    u = u.copy()
    u["edgar_covered"] = has_cov
    u["filing_same_morning"] = same_morning
    u["filings_30d"] = filings_30d
    u["is_foreign_issuer"] = foreign_365
    u["shelf_180d"] = shelf_180
    return u


def precursor_definitions() -> Dict[str, str]:
    """Name -> pandas query, each computable strictly before the move."""
    return {
        "volume contraction (5d/20d < 0.5)": "vol_ratio_5_20 < 0.5",
        "volume expansion (5d/20d > 2.0)": "vol_ratio_5_20 > 2.0",
        "fell >15% over 20 sessions": "ret_20d_prior < -0.15",
        "fell >40% over 20 sessions": "ret_20d_prior < -0.40",
        "within 20% of 90-day low": "pct_above_90d_low < 0.20",
        "more than 50% off 90-day high": "pct_off_90d_high < -0.50",
        "ADV under $50k": "adv20 < 50000",
        "ADV under $250k": "adv20 < 250000",
        "foreign private issuer": "is_foreign_issuer == True",
        "filing accepted same morning": "filing_same_morning == True",
        "no filings in 30 days": "edgar_covered == True and filings_30d == 0",
        ">=3 shelf registrations in 180d": "shelf_180d >= 3",
        "prior +50% day within 90 sessions": "prior_50_90d >= 1",
        "contraction AND fell >15%": "vol_ratio_5_20 < 0.5 and ret_20d_prior < -0.15",
        "contraction AND near 90d low": "vol_ratio_5_20 < 0.5 and pct_above_90d_low < 0.20",
        "contraction AND fell >15% AND ADV<$250k":
            "vol_ratio_5_20 < 0.5 and ret_20d_prior < -0.15 and adv20 < 250000",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Precursor base rates.")
    parser.add_argument("--out", default=str(bars.DATA_ROOT / "forensic_precursors.json"))
    args = parser.parse_args(argv)

    u = pd.read_parquet(bars.DATA_ROOT / "forensic_universe.parquet")
    sample = pd.read_parquet(bars.DATA_ROOT / "forensic_sample.parquet")
    filings = pd.read_parquet(bars.DATA_ROOT / "edgar_filings.parquet")

    logger.info("universe %s rows | building daily precursors ...", f"{len(u):,}")
    u = add_daily_precursors(u)
    logger.info("building filing precursors ...")
    u = add_filing_precursors(u, filings)

    u["is_move"] = u["c2c"] >= MOVE_THRESHOLD
    base = float(u["is_move"].mean())
    logger.info("base rate P(+50%%) = %.4f%% (%s of %s)",
                100 * base, f"{int(u['is_move'].sum()):,}", f"{len(u):,}")

    keys = set(zip(sample["symbol"], sample["date"]))
    in_sample = np.array([(s, d) in keys for s, d in zip(u["symbol"], u["date"])])
    u["in_sample"] = in_sample
    assert int(in_sample.sum()) == len(sample), (
        f"sample join found {int(in_sample.sum())} of {len(sample)} rows"
    )

    rows: List[Dict] = []
    for name, expr in precursor_definitions().items():
        try:
            mask = u.eval(expr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: %s", name, exc)
            continue
        mask = mask.fillna(False).to_numpy(dtype=bool)
        fires = int(mask.sum())
        if fires == 0:
            continue
        p_move_given = float(u.loc[mask, "is_move"].mean())
        hits = int((mask & in_sample).sum())
        rows.append({
            "precursor": name,
            "hits_in_10": hits,
            "fires_pct_of_universe": 100.0 * fires / len(u),
            "n_fires": fires,
            "p_move_given_pct": 100.0 * p_move_given,
            "lift": p_move_given / base if base > 0 else np.nan,
            "moves_captured_pct": 100.0 * float(
                (mask & u["is_move"].to_numpy()).sum()) / max(int(u["is_move"].sum()), 1),
        })

    table = pd.DataFrame(rows).sort_values("lift", ascending=False)
    logger.info("=" * 112)
    logger.info("PRECURSOR TABLE  (base rate %.3f%%; 'lift' is P(move|precursor) / P(move))\n%s",
                100 * base,
                table.to_string(index=False, float_format="%.3f"))
    logger.info("=" * 112)

    useful = table[(table["lift"] >= 2.0) & (table["n_fires"] >= 2000)]
    if useful.empty:
        logger.info("NO precursor both doubles the base rate and fires often "
                    "enough to trade.")
    else:
        logger.info("Precursors with lift >= 2x firing on >= 2,000 symbol-days:\n%s",
                    useful.to_string(index=False, float_format="%.3f"))

    Path(args.out).write_text(json.dumps(
        {"base_rate": base, "universe_rows": int(len(u)),
         "n_moves": int(u["is_move"].sum()), "precursors": rows},
        indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
