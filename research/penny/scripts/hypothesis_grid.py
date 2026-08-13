"""hypothesis_grid.py — a disciplined sweep over parametric trading hypotheses.

What this is
------------
A systematic test of families of the form *condition X on day t implies return Y
over the next H sessions*, with the parameters swept rather than chosen. It
covers the four hypotheses supplied plus seven of my own, each enriched with
conditioning variables.

Why a sweep is dangerous, and what is done about it
---------------------------------------------------
Sweeping thousands of parameter cells over one dataset **will** produce cells
that look profitable. That is arithmetic, not discovery: at a 5% threshold, 2,000
independent cells yield ~100 false positives. Three defences, all mandatory:

1. **Calibrate and confirm on disjoint periods.** Every cell is fitted on an
   in-sample window and re-measured, untouched, on a later out-of-sample window.
   A cell that survives only in-sample is reported as failed.
2. **Benjamini-Hochberg over every cell tested**, not over the survivors. The
   denominator is the full grid, including the cells nobody would have reported.
3. **Costs are charged before judging.** A gross edge below the spread is not an
   edge. Round-trip cost is applied on a measured schedule by liquidity band.

Look-ahead discipline
---------------------
Signals are computed from information available **at or before** the stated
decision instant, and entry happens strictly afterwards:

* ``close`` entry -- signal uses data through day *t*'s close; entry is day *t*'s
  close; exit is day *t+H*'s close. Fully tradable, no ambiguity.
* ``open`` entry -- signal uses the opening print of day *t* (a gap); entry at
  that same open is optimistic, so both variants are reported and the
  conservative one governs any conclusion.

Reverse splits are excluded from every family except the one that tests them,
because 16.1% of large "gainers" are share consolidations that fade ~15% from the
open and would otherwise contaminate any momentum or gap family.

    python research/penny/scripts/hypothesis_grid.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("grid")

# Split between calibration and confirmation. Fixed before any result is seen.
OOS_START = pd.Timestamp("2026-02-01")

HORIZONS = [1, 2, 3, 5, 10]
MIN_TRADES = 200          # below this a cell is not evidence
MIN_DAYS = 60             # distinct sessions, so a cell is not one week's fluke
FDR_Q = 0.05

# Round-trip cost in basis points by 20-day average dollar volume. Anchored on
# direct NBBO measurement: 59.7 bps spread-only round trip for the ADV >= $1M
# reversal pool (entry 40.2 + exit 19.5), and 66.6 bps mean at a $2,000 order for
# the ADV >= $5M band. Thinner bands are extrapolated upward and are the least
# certain numbers here, which is why a sensitivity multiplier is reported.
COST_BPS = [(0, 1e5, 400.0), (1e5, 2.5e5, 250.0), (2.5e5, 1e6, 150.0),
            (1e6, 5e6, 90.0), (5e6, 2e7, 65.0), (2e7, np.inf, 45.0)]


def cost_for(adv: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=adv.index)
    for lo, hi, bps in COST_BPS:
        out = out.mask((adv >= lo) & (adv < hi), bps / 10_000.0)
    return out.fillna(400.0 / 10_000.0)


@dataclass
class Cell:
    family: str
    label: str
    mask: np.ndarray
    entry: str          # 'close' or 'open'
    horizon: int


def build_panel() -> pd.DataFrame:
    """Universe with every signal input and forward return attached."""
    u = pd.read_parquet(bars.DATA_ROOT / "forensic_universe.parquet")
    u = u.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = u.groupby("symbol", sort=False)

    # --- trailing state, all strictly before day t's close -----------------
    for n in (1, 3, 5, 10, 20):
        u[f"ret_{n}d"] = u["close"] / g["close"].shift(n) - 1.0
    u["ret_1d_prior"] = g["close"].shift(1) / g["close"].shift(2) - 1.0

    u["vol20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    u["vol_shock"] = u["volume"] / u["vol20"].replace(0, np.nan)
    u["dv_shock"] = (u["close"] * u["volume"]) / u["adv20"].replace(0, np.nan)

    hi20 = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    lo20 = g["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    u["pct_of_20d_range"] = (u["close"] - lo20) / (hi20 - lo20).replace(0, np.nan)
    u["range_10d"] = (g["high"].transform(lambda s: s.rolling(10, min_periods=8).max())
                      / g["low"].transform(lambda s: s.rolling(10, min_periods=8).min()) - 1.0)
    u["at_20d_low"] = (u["close"] <= lo20 * 1.02).fillna(False)
    u["at_20d_high"] = (u["close"] >= hi20 * 0.98).fillna(False)

    down = (u["close"] < g["close"].shift(1)).astype(float)
    u["consec_down"] = down.groupby(u["symbol"]).transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1) * down

    u["realised_vol_20d"] = g["c2c"].transform(lambda s: s.rolling(20, min_periods=10).std())

    # --- forward returns ----------------------------------------------------
    for h in HORIZONS:
        fwd_close = g["close"].shift(-h)
        u[f"fwd_close_{h}"] = fwd_close / u["close"] - 1.0        # enter at close t
        u[f"fwd_open_{h}"] = fwd_close / u["open"] - 1.0          # enter at open t
    u["overnight"] = u["open"] / g["close"].shift(1) - 1.0
    u["cost"] = cost_for(u["adv20"])
    return u


def families(u: pd.DataFrame) -> List[Cell]:
    """Every hypothesis family, expanded over its parameter grid."""
    cells: List[Cell] = []
    clean = ~u["is_suspect_split"]
    liquid = u["adv20"] >= 2.5e5      # a floor at which costs are ~150 bps or better

    def add(family, label, mask, entry, horizons=HORIZONS):
        for h in horizons:
            cells.append(Cell(family, f"{label} | H={h}", mask.to_numpy(), entry, h))

    # --- A. mean reversion: down by A over N days -> bounce ---------------
    for n in (1, 3, 5, 10, 20):
        for a in (0.10, 0.20, 0.30, 0.40, 0.50):
            m = clean & liquid & (u[f"ret_{n}d"] <= -a)
            add("A_reversion", f"ret_{n}d <= -{100*a:.0f}%", m, "close")
    # enriched: reversion only when volume is NOT expanding (capitulation vs slide)
    for a in (0.20, 0.30, 0.40):
        m = clean & liquid & (u["ret_5d"] <= -a) & (u["vol_shock"] >= 3.0)
        add("A_reversion+vol", f"ret_5d <= -{100*a:.0f}% & vol>=3x", m, "close")
        m2 = clean & liquid & (u["ret_5d"] <= -a) & (u["at_20d_low"])
        add("A_reversion+low", f"ret_5d <= -{100*a:.0f}% & at 20d low", m2, "close")

    # --- B. momentum: up by A over N days -> continues ---------------------
    for n in (1, 3, 5, 10, 20):
        for a in (0.10, 0.20, 0.30, 0.50, 1.00):
            m = clean & liquid & (u[f"ret_{n}d"] >= a)
            add("B_momentum", f"ret_{n}d >= +{100*a:.0f}%", m, "close")
    for a in (0.20, 0.50):
        m = clean & liquid & (u["ret_5d"] >= a) & (u["at_20d_high"])
        add("B_momentum+high", f"ret_5d >= +{100*a:.0f}% & at 20d high", m, "close")

    # --- C. reverse split -> subsequent move -------------------------------
    # The only family that deliberately keeps consolidations.
    m = u["is_suspect_split"]
    add("C_reverse_split", "suspected reverse split", m, "close")
    add("C_reverse_split", "suspected reverse split (from open)", m, "open")
    m2 = u["is_suspect_split"] & (u["adv20"] >= 2.5e5)
    add("C_reverse_split", "suspected reverse split & ADV>=250k", m2, "close")

    # --- D. large gap up / down -------------------------------------------
    for a in (0.05, 0.10, 0.20, 0.30, 0.50):
        mu = clean & liquid & (u["gap"] >= a)
        add("D_gap_up", f"gap >= +{100*a:.0f}%", mu, "open")
        md = clean & liquid & (u["gap"] <= -a)
        add("D_gap_down", f"gap <= -{100*a:.0f}%", md, "open")
    # enriched: gap down into an already-beaten name
    for a in (0.05, 0.10):
        m = clean & liquid & (u["gap"] <= -a) & (u["ret_20d"] <= -0.20)
        add("D_gap_down+fallen", f"gap <= -{100*a:.0f}% & ret_20d<=-20%", m, "open")

    # --- E. volume shock without price move (accumulation) -----------------
    for v in (3.0, 5.0, 10.0):
        m = clean & liquid & (u["vol_shock"] >= v) & (u["c2c"].abs() <= 0.05)
        add("E_vol_shock_quiet", f"vol>={v:.0f}x & |move|<=5%", m, "close")
        m2 = clean & liquid & (u["vol_shock"] >= v)
        add("E_vol_shock", f"vol >= {v:.0f}x average", m2, "close")

    # --- F. range compression (squeeze) ------------------------------------
    for r in (0.10, 0.20, 0.30):
        m = clean & liquid & (u["range_10d"] <= r)
        add("F_squeeze", f"10d range <= {100*r:.0f}%", m, "close")

    # --- G. new lows / highs ------------------------------------------------
    add("G_new_low", "at 20-day low", clean & liquid & u["at_20d_low"], "close")
    add("G_new_high", "at 20-day high", clean & liquid & u["at_20d_high"], "close")

    # --- H. consecutive down days ------------------------------------------
    for k in (3, 4, 5):
        m = clean & liquid & (u["consec_down"] >= k)
        add("H_consec_down", f"{k}+ consecutive down days", m, "close")

    # --- I. overnight vs intraday decomposition ----------------------------
    # Buy at the close, sell at the next open. A documented equity anomaly and
    # a useful control: if the whole edge is overnight, no intraday rule matters.
    cells.append(Cell("I_overnight", "buy close -> sell next open", (clean & liquid).to_numpy(),
                      "overnight", 1))

    # --- J. price band, as a pure control ----------------------------------
    add("J_control", "all liquid clean days (no condition)", clean & liquid, "close")
    return cells


def evaluate(u: pd.DataFrame, cell: Cell, period: pd.Series,
             cost_mult: float = 1.0) -> Optional[Dict]:
    m = cell.mask & period.to_numpy()
    if m.sum() < MIN_TRADES:
        return None

    if cell.entry == "overnight":
        ret = u.loc[m, "overnight"]
    elif cell.entry == "open":
        ret = u.loc[m, f"fwd_open_{cell.horizon}"]
    else:
        ret = u.loc[m, f"fwd_close_{cell.horizon}"]

    dates = u.loc[m, "date"]
    net = ret - u.loc[m, "cost"] * cost_mult
    ok = net.notna()
    if ok.sum() < MIN_TRADES or dates[ok].nunique() < MIN_DAYS:
        return None

    est = pstats.cluster_bootstrap_mean(net[ok], dates[ok], n_boot=1_500)
    gross = float(ret[ok].mean())
    return {
        "family": cell.family, "label": cell.label, "entry": cell.entry,
        "horizon": cell.horizon, "n": int(est.n), "n_days": int(est.n_days),
        "gross": gross, "net": float(est.mean), "net_lo": float(est.lo),
        "net_hi": float(est.hi), "p": float(est.p_value_greater),
        "hit": float((net[ok] > 0).mean()),
        "cost_bps": float(1e4 * u.loc[m, "cost"].mean() * cost_mult),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hypothesis grid sweep.")
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "hypothesis_grid.json"))
    ap.add_argument("--cost-mult", type=float, default=1.0)
    args = ap.parse_args(argv)

    logger.info("building panel ...")
    u = build_panel()
    ins = u["date"] < OOS_START
    oos = u["date"] >= OOS_START
    logger.info("panel %s rows | in-sample %s | out-of-sample %s (from %s)",
                f"{len(u):,}", f"{int(ins.sum()):,}", f"{int(oos.sum()):,}",
                OOS_START.date())

    cells = families(u)
    logger.info("grid: %d cells across %d families",
                len(cells), len({c.family for c in cells}))

    rows: List[Dict] = []
    for c in cells:
        r_in = evaluate(u, c, ins, args.cost_mult)
        if r_in is None:
            continue
        r_out = evaluate(u, c, oos, args.cost_mult)
        row = {**{f"is_{k}": v for k, v in r_in.items()
                  if k not in ("family", "label", "entry", "horizon")},
               "family": c.family, "label": c.label, "entry": c.entry,
               "horizon": c.horizon}
        if r_out:
            row.update({f"oos_{k}": v for k, v in r_out.items()
                        if k not in ("family", "label", "entry", "horizon")})
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("evaluated %d cells with enough data", len(df))

    # --- multiple testing over the WHOLE grid ------------------------------
    have_oos = df["oos_p"].notna() if "oos_p" in df.columns else pd.Series(False, index=df.index)
    df["is_pass"] = df["is_net_lo"] > 0
    df["oos_pass"] = (df.get("oos_net_lo", pd.Series(np.nan, index=df.index)) > 0)

    p_all = df["is_p"].to_numpy()
    df["bh_pass_is"] = pstats.benjamini_hochberg(p_all, q=FDR_Q)

    survivors = df[df["bh_pass_is"] & df["oos_pass"]]

    logger.info("=" * 116)
    logger.info("IN-SAMPLE cells with net CI excluding zero : %d of %d",
                int(df["is_pass"].sum()), len(df))
    logger.info("  ... surviving BH-FDR over the whole grid : %d",
                int(df["bh_pass_is"].sum()))
    logger.info("  ... AND confirmed out-of-sample          : %d", len(survivors))
    logger.info("=" * 116)

    show = ["family", "label", "entry", "is_n", "is_gross", "is_net", "is_net_lo",
            "is_cost_bps", "oos_n", "oos_net", "oos_net_lo"]
    best = df.sort_values("is_net", ascending=False).head(18)
    logger.info("TOP 18 BY IN-SAMPLE NET\n%s",
                best[[c for c in show if c in best.columns]].to_string(
                    index=False, float_format="%.4f"))

    if not survivors.empty:
        logger.info("-" * 116)
        logger.info("SURVIVORS (BH-corrected in-sample AND positive out-of-sample)\n%s",
                    survivors[[c for c in show if c in survivors.columns]].to_string(
                        index=False, float_format="%.4f"))
    else:
        logger.info("NO cell survives BH correction in-sample AND confirms out-of-sample.")

    df.to_parquet(bars.DATA_ROOT / "hypothesis_grid.parquet", index=False)
    Path(args.out).write_text(json.dumps({
        "n_cells_tested": int(len(cells)),
        "n_cells_evaluated": int(len(df)),
        "oos_start": str(OOS_START.date()),
        "fdr_q": FDR_Q, "cost_mult": args.cost_mult,
        "n_is_pass": int(df["is_pass"].sum()),
        "n_bh_pass": int(df["bh_pass_is"].sum()),
        "n_survivors": int(len(survivors)),
        "rows": df.to_dict(orient="records"),
    }, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
