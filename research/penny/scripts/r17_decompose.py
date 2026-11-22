"""r17_decompose.py -- DIAGNOSTICS for Round 17: where did R15's 1.21 go? (dev only)

Not cells. Nothing here is selectable; it explains the gap between R15's
legacy number (1.2138) and R17 cell A (PREREGISTRATION_R17 §3 says anything
else computed is a labelled diagnostic).

D1  accounting: cell A's causal trades booked the legacy way (carried at cost,
    P&L on the release day). Same trades, so the gap to cell A is accounting.
D2  the leak: candidates the legacy frame dropped vs those it kept, using the
    causal outcome of each (gross, before cost), on signals whose 61-bar window
    closes before the cutoff.
D3  path conventions: on candidates in both frames, legacy R_t100 vs the causal
    exit value.

    python research/penny/scripts/r17_decompose.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.dates import DEV_START, SEAL                              # noqa: E402
from research.penny.etl import bars                                           # noqa: E402
from research.penny.eval import benchmark as bm                                # noqa: E402
from research.penny.scripts.r15_matched_book import attach_sector as attach_sector_r15  # noqa: E402
from research.penny.scripts.reproduce_round14 import (                        # noqa: E402
    HOLD, TARGET, VETO_FRAC, VOL_CAP, build_frame, market_calendar)
from research.penny.scripts.sector_split import attach_sector, attach_sector_relative  # noqa: E402
from research.penny.strategy import causal                                     # noqa: E402
from research.penny.strategy.book import BookConfig, gross_return, run_book    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("r17d")
SEEDS = 15


def stepped(paths: causal.Paths, cost_guess: float) -> causal.Paths:
    """Marks that reproduce realised-only booking: flat at par (1 + c cancels the
    charge) until the exit session, then the exit value."""
    m = paths.marks.copy()
    w = m.shape[1]
    for i in range(len(m)):
        xo = paths.exit_off[i]
        last = xo if xo >= 0 else w
        finite = np.isfinite(m[i])
        m[i, :min(max(paths.hold, last), w)] = 1.0 + cost_guess
        if xo >= 0:
            m[i, max(paths.hold, xo):] = paths.exit_value[i]
        m[i, ~finite] = np.nan
    return causal.Paths(paths.filled, paths.entry_unadj, m, paths.exit_off, paths.kind,
                        paths.exit_value, paths.n_corr, paths.hold)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-d1", action="store_true", help="D1 already recorded (LOG §101)")
    args = ap.parse_args(argv)
    cal = market_calendar("dev")
    cutoff = cal[-1]
    u = attach_sector_relative(attach_sector(causal.load_panel(cutoff)), 5)
    df = causal.causal_frame(u, DEV_START, SEAL, 0.15, causal.frozen_signs())
    df["_path"] = np.arange(len(df))
    corr = causal.split_corrections(u, causal.load_vendor_splits(cutoff))
    paths = causal.holding_paths(u, df["_pos"].to_numpy(), HOLD, TARGET, None, corr)
    out = {}
    p_out = bars.DATA_ROOT / "r17" / "r17_dev_decompose.json"
    if p_out.exists():
        out = json.loads(p_out.read_text(encoding="utf-8"))

    # ---- D1: same causal trades, realised-only booking ------------------------
    # c varies by trade; the stepped mark holds value at par only if c matches,
    # so book with cost_mode none and subtract nothing -- compare like with like
    cfg_m = BookConfig(n_slots=60, hold=HOLD, vol_cap=VOL_CAP, veto_frac=VETO_FRAC, target=TARGET,
                       sector_cap=3, cost_mode="none", mark="mtm")
    st = stepped(paths, 0.0)
    sh_mtm, sh_step = [], []
    for s in ([] if args.skip_d1 else range(SEEDS)):
        a = run_book(df, cal, cfg_m, seed=s, paths=paths)
        b = run_book(df, cal, cfg_m, seed=s, paths=st)
        sh_mtm.append(bm.sharpe(a.daily.to_numpy()[1:]))
        sh_step.append(bm.sharpe(b.daily.to_numpy()[1:]))
    if not args.skip_d1:
        out["D1_gross_of_cost"] = {"mtm_sharpe": float(np.mean(sh_mtm)),
                                   "realised_booking_sharpe": float(np.mean(sh_step))}
        logger.info("D1 %s", out["D1_gross_of_cost"])
    else:
        out.setdefault("D1_gross_of_cost", {"mtm_sharpe": 0.7666768993634768,
                                            "realised_booking_sharpe": 1.634900397120932,
                                            "note": "recorded from the 2026-11-22 run (LOG §101)"})

    # ---- D2 / D3: candidate level, legacy frame vs causal frame ----------------
    leg = attach_sector_r15(build_frame("dev"))
    last_ok = cal[len(cal) - 1 - (HOLD + 2)]                      # 61-bar window inside dev
    c = df.assign(gross_causal=np.where(paths.filled, paths.exit_value - 1.0, np.nan),
                  kind=paths.kind)
    c = c[c["date"] <= last_ok]
    lg = leg[leg["date"] <= last_ok][["symbol", "date", "R_t100", "B7_realised_vol"]]
    m = c.merge(lg, on=["symbol", "date"], how="left", indicator=True, suffixes=("", "_leg"))
    kept, dropped = m[m["_merge"] == "both"], m[m["_merge"] == "left_only"]
    def stats(x):
        g = x["gross_causal"].dropna()
        return {"n": int(len(x)), "filled": int(len(g)), "mean_gross": float(g.mean()),
                "median_gross": float(g.median()), "tp_rate": float((x["kind"] == "tp").mean()),
                "le_minus50": float((g <= -0.5).mean()), "ge_plus100": float((g >= 1.0).mean())}
    out["D2_leak"] = {"kept_by_legacy": stats(kept), "dropped_by_legacy": stats(dropped),
                      "legacy_rows_not_in_causal": int(len(lg.merge(c[["symbol", "date"]], how="left",
                                                                     indicator=True)
                                                            .query("_merge == 'left_only'")))}
    logger.info("D2 %s", out["D2_leak"])
    both = kept.dropna(subset=["gross_causal", "R_t100"])
    d = both["gross_causal"] - both["R_t100"]
    out["D3_paths"] = {"n": int(len(both)), "mean_legacy": float(both["R_t100"].mean()),
                       "mean_causal": float(both["gross_causal"].mean()),
                       "mean_diff": float(d.mean()), "share_equal_1e-9": float((d.abs() < 1e-9).mean()),
                       "share_abs_diff_gt_5pct": float((d.abs() > 0.05).mean())}
    logger.info("D3 %s", out["D3_paths"])

    # ---- legacy per-trade, for scale -------------------------------------------
    cfg_l = BookConfig(n_slots=60, hold=HOLD, vol_cap=VOL_CAP, veto_frac=VETO_FRAC, target=TARGET,
                       sector_cap=3, cost_mode="size_aware")
    tl = pd.concat([run_book(leg, cal, cfg_l, seed=s).trades for s in range(SEEDS)])
    ta = pd.concat([run_book(df, cal, BookConfig(**{**cfg_m.__dict__, "cost_mode": "size_aware"}),
                             seed=s, paths=paths).trades for s in range(SEEDS)])
    out["per_trade"] = {"legacy": {"n_per_seed": len(tl) / SEEDS, "mean_net": float(tl["net"].mean()),
                                   "median_net": float(tl["net"].median()),
                                   "mean_gross": float(tl["gross"].mean())},
                        "causal_A": {"n_per_seed": len(ta) / SEEDS, "mean_net": float(ta["net"].mean()),
                                     "median_net": float(ta["net"].median()),
                                     "mean_gross": float(ta["gross"].mean())}}
    logger.info("per trade %s", out["per_trade"])
    p = bars.DATA_ROOT / "r17" / "r17_dev_decompose.json"
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
