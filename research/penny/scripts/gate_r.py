"""gate_r.py — does the reversal edge survive to the 09:35 decision?

The round-5 pool was chosen on an `open -> close` table: liquid low-priced names
gapping down 3-10% returned **+0.46%** intraday, against −0.43% for the +3..+10%
gappers rounds 1-4 actually bought.

But this study enters at **09:35**, five minutes after the open. If short-term
reversal is a liquidity-provision effect (Nagel 2012), the compensation may be
paid in the opening minutes and gone by the time this strategy could act. That
would make the entire pool untradable at this entry time, and it is cheaper to
find out on 264 sampled sessions than on 2,605.

Reports the same population at several entry times so the decay is visible
rather than inferred, and charges nothing: this is a gross measurement, and the
cost bar is applied downstream.

    python research/penny/scripts/gate_r.py
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
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.fetch_intraday import session_path  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("gate_r")

# Pre-registered threshold: PREREGISTRATION_R5.md section 4.
GATE_R_MIN_RETURN = 0.0020

ENTRIES = ["09:31", "09:35", "09:40", "09:45", "10:00", "10:30", "11:00"]
EXIT_MINUTE = "15:55"


def measure(panel: pd.DataFrame, entries: List[str], exit_minute: str) -> pd.DataFrame:
    """Entry-price at each candidate time, and the return to the exit bar."""
    m = panel[panel["minute_et"].isin(set(entries) | {exit_minute})]
    wide = m.pivot_table(index=["symbol", "date"], columns="minute_et",
                         values="open", aggfunc="first")
    if exit_minute not in wide.columns:
        return pd.DataFrame()
    # Exit on the 15:55 bar's close-equivalent: use its open, consistent with
    # the entry convention, so both legs are priced the same way.
    out = pd.DataFrame(index=wide.index)
    for e in entries:
        if e in wide.columns:
            out[f"ret_{e}"] = wide[exit_minute] / wide[e] - 1.0
    return out.reset_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gate R: does reversal survive to 09:35?")
    parser.add_argument("--candidates",
                        default=str(bars.DATA_ROOT / "candidates_rev_probe.parquet"))
    parser.add_argument("--out", default=str(bars.DATA_ROOT / "gate_r.json"))
    args = parser.parse_args(argv)

    cand = pd.read_parquet(args.candidates, columns=["symbol", "date", "gap", "adv_20"])
    days = sorted(cand["date"].unique())
    wanted = cand.groupby("date")["symbol"].apply(set)
    logger.info("probe: %s candidate-days over %d sessions",
                f"{len(cand):,}", len(days))

    frames: List[pd.DataFrame] = []
    missing = 0
    for day in days:
        p = session_path(pd.Timestamp(day))
        if not p.exists():
            missing += 1
            continue
        part = pd.read_parquet(p)
        if part.empty:
            missing += 1
            continue
        part = part[part["symbol"].isin(wanted.loc[day])]
        if not part.empty:
            frames.append(part)
    if not frames:
        raise SystemExit("no minute data for the probe sessions")
    panel = pd.concat(frames, ignore_index=True)
    logger.info("minute rows %s over %d sessions (%d missing)",
                f"{len(panel):,}", len(frames), missing)

    rets = measure(panel, ENTRIES, EXIT_MINUTE)
    df = cand.merge(rets, on=["symbol", "date"], how="inner")
    logger.info("matched %s candidate-days", f"{len(df):,}")

    rows: List[Dict] = []
    for e in ENTRIES:
        col = f"ret_{e}"
        if col not in df.columns:
            continue
        s = df[col].replace([np.inf, -np.inf], np.nan)
        ok = s.notna()
        if ok.sum() < 200:
            continue
        est = pstats.cluster_bootstrap_mean(s[ok], df.loc[ok, "date"], n_boot=4_000)
        rows.append({
            "entry": e, "n": int(est.n), "n_days": int(est.n_days),
            "mean": float(est.mean), "lo": float(est.lo), "hi": float(est.hi),
            "p_greater": float(est.p_value_greater),
            "hit_rate": float((s[ok] > 0).mean()),
            "fill_rate": float(ok.mean()),
        })

    table = pd.DataFrame(rows)
    logger.info("=" * 76)
    logger.info("RETURN TO %s BY ENTRY TIME (gross, no costs)\n%s",
                EXIT_MINUTE, table.to_string(index=False, float_format="%.4f"))

    decision = table[table["entry"] == "09:35"]
    if decision.empty:
        raise SystemExit("no 09:35 entries measured")
    d = decision.iloc[0]
    passed = bool(d["mean"] >= GATE_R_MIN_RETURN and d["lo"] > 0)

    logger.info("-" * 76)
    logger.info("GATE R: 09:35 entry mean %+.4f  CI [%+.4f, %+.4f]  n=%d over %d days",
                d["mean"], d["lo"], d["hi"], int(d["n"]), int(d["n_days"]))
    logger.info("        threshold %+.4f with CI excluding zero -> %s",
                GATE_R_MIN_RETURN,
                "PASS -- the edge survives to the decision instant"
                if passed else "FAIL -- the edge is spent before 09:35")

    # How much of the open->close edge is left at each entry, for the record.
    if "09:31" in set(table["entry"]):
        first = table[table["entry"] == "09:31"].iloc[0]["mean"]
        if abs(first) > 1e-9:
            logger.info("        decay: %.0f%% of the 09:31 edge remains at 09:35",
                        100 * d["mean"] / first)
    logger.info("=" * 76)

    Path(args.out).write_text(json.dumps(
        {"gate_r_pass": passed, "threshold": GATE_R_MIN_RETURN,
         "exit_minute": EXIT_MINUTE, "by_entry": rows,
         "probe_sessions": len(frames), "probe_rows": int(len(df))},
        indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
