"""build_dataset.py — assemble the labelled trade table.

Joins, per candidate symbol-day:

* cross-day features from the daily panel (trailing, shifted -- causal)
* opening-window features from 09:30-09:34 minute bars
* the triple-barrier outcome from 09:35 onward
* every registered veto column

The output ``trades.parquet`` is the single input to the analysis notebook.

Ordering matters and is enforced: the observation features and the label are
built from **disjoint** slices of the same minute path, and the join key is
(symbol, date) only -- never anything derived from the outcome.

    python research/penny/scripts/build_dataset.py
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

from research.penny.etl import bars, candidates as cand  # noqa: E402
from research.penny.features import veto as veto_mod  # noqa: E402
from research.penny.labels import triple_barrier as tb  # noqa: E402
from research.penny.scripts.fetch_intraday import MINUTE_DIR, session_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset")

# Causal columns: every one is trailing/shifted, or uses only the 09:30 opening
# print. These are the ONLY columns a veto may be built from.
_CAUSAL_COLS = [
    "symbol", "date", "open", "close_unadj_prev", "prev_close",
    "gap", "ret_1d_prev", "ret_5d_prev",
    "adv_5", "adv_20", "adv_60", "rvol_prev",
    "max_ret_20d", "vol_20d", "hl_range_20d", "consec_up",
    "split_factor_180d",
]

# Columns that contain information from AT OR AFTER the decision, kept only for
# post-hoc diagnostics. They are prefixed so nothing can reach for them by
# accident during feature mining -- an earlier veto did exactly that, using the
# adjusted/unadjusted ratio, which encodes corporate actions occurring after
# the bar and therefore leaked future reverse splits into the search.
_FUTURE_COLS = {
    "close": "fut_close",
    "volume": "fut_volume",
    "close_unadj": "fut_close_unadj",
    "days_to_last_bar": "diag_days_to_last_bar",
}


def build_trades(candidates: pd.DataFrame, spec: tb.BarrierSpec) -> pd.DataFrame:
    """Label every candidate day for which minute data is cached."""
    days = sorted(candidates["date"].unique())
    wanted = candidates.groupby("date")["symbol"].apply(lambda s: set(s))

    frames: List[pd.DataFrame] = []
    missing_sessions = 0

    for i, day in enumerate(days, start=1):
        day_ts = pd.Timestamp(day)
        path = session_path(day_ts)
        if not path.exists():
            missing_sessions += 1
            continue

        minute = pd.read_parquet(path)
        # Restrict to that day's candidates: the minute cache may hold more.
        minute = minute[minute["symbol"].isin(wanted.loc[day])]
        if minute.empty:
            continue

        obs = tb.observation_features(minute)
        labels = tb.label_day(minute, day_ts, spec)
        if labels.empty or obs.empty:
            continue

        merged = labels.merge(obs, on="symbol", how="left")
        merged["date"] = day_ts
        frames.append(merged)

        if i % 100 == 0:
            logger.info("labelled %d/%d sessions (%d trades so far)",
                        i, len(days), sum(len(f) for f in frames))

    if missing_sessions:
        logger.warning(
            "%d/%d sessions had no cached minute data and were skipped",
            missing_sessions, len(days),
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the labelled trade table.")
    parser.add_argument("--profit-target", type=float, default=0.30)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--out", default=str(bars.DATA_ROOT / "trades.parquet"))
    args = parser.parse_args(argv)

    cand_path = bars.DATA_ROOT / "candidates.parquet"
    if not cand_path.exists():
        raise SystemExit("candidates.parquet not found; run build_candidates first")

    candidates = pd.read_parquet(cand_path)
    logger.info("candidates: %s rows over %d sessions",
                f"{len(candidates):,}", candidates["date"].nunique())

    spec = tb.BarrierSpec(profit_target=args.profit_target, stop_loss=args.stop_loss)
    logger.info("barriers: +%.0f%% / -%.0f%%, vertical %s",
                100 * spec.profit_target, 100 * spec.stop_loss, spec.vertical_minute)

    trades = build_trades(candidates, spec)
    if trades.empty:
        raise SystemExit("no trades produced; is minute data cached?")

    logger.info("labelled %s trades", f"{len(trades):,}")

    # Join the daily features. (symbol, date) only -- nothing outcome-derived.
    # `validate` makes a duplicated bar an error rather than a silent
    # double-count of trades.
    daily_cols = [c for c in _CAUSAL_COLS if c in candidates.columns]
    future_cols = {k: v for k, v in _FUTURE_COLS.items() if k in candidates.columns}
    joined = candidates[daily_cols + list(future_cols)].rename(columns=future_cols)
    trades = trades.merge(joined, on=["symbol", "date"], how="left", validate="one_to_one")

    before = len(trades)
    trades = trades[trades["outcome"] != "no_fill"]
    logger.info(
        "dropped %d unfillable trades (no print in the decision minute); %s remain",
        before - len(trades), f"{len(trades):,}",
    )

    trades = veto_mod.build_all(trades, include_diagnostics=True)
    errors = trades.attrs.get("veto_build_errors", {})
    if errors:
        logger.warning("vetoes that could not be built: %s", errors)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out_path, engine="pyarrow", index=False, compression="snappy")

    outcome_mix = trades["outcome"].value_counts(normalize=True).round(4).to_dict()
    summary = {
        "trades": int(len(trades)),
        "sessions": int(trades["date"].nunique()),
        "symbols": int(trades["symbol"].nunique()),
        "date_min": str(trades["date"].min().date()),
        "date_max": str(trades["date"].max().date()),
        "outcome_mix": outcome_mix,
        "mean_net_return": float(trades["net_return"].mean()),
        "median_net_return": float(trades["net_return"].median()),
        "mean_gross_return": float(trades["gross_return"].mean()),
        "mean_cost_bps": float(trades["cost_bps"].mean()),
        "pct_halt_suspect": float(trades["halt_suspect"].mean()),
        "pct_ambiguous_bar": float(trades["ambiguous_bar"].mean()),
        "mean_coverage": float(trades["coverage"].mean()),
        "veto_build_errors": errors,
        "veto_prevalence": {
            c: float(trades[c].mean())
            for c in veto_mod.veto_names(include_diagnostics=True)
            if c in trades.columns
        },
        "barriers": {"profit_target": spec.profit_target, "stop_loss": spec.stop_loss},
    }
    (bars.DATA_ROOT / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    logger.info("outcome mix: %s", outcome_mix)
    logger.info("mean net return: %.4f  (gross %.4f, costs %.0f bps)",
                summary["mean_net_return"], summary["mean_gross_return"],
                summary["mean_cost_bps"])
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
