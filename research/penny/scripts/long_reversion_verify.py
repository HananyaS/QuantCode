"""long_reversion_verify.py — does the calibrated long cell survive its own audit?

The cell that won the portfolio calibration
--------------------------------------------
    price $0.50-$2.00, ADV >= $250k
    fallen >= 40% over the previous five sessions
    buy, sell on a +50% limit, time stop after ten sessions
    ~30 concurrent slots

reported Sharpe 3.20, compounded +125.6%, max drawdown -10.4%, +5.89% per trade.

Why that number is not yet a result
------------------------------------
It is the best of 48 portfolio cells swept over one dataset, and the parameters
were chosen by looking at the answer. Everything below is the audit that the two
long candidates in §57 and §61 failed and the short strategy in §63 passed:

1. **Execution.** The signal is day *t*'s close and the entry is day *t*'s
   close. Re-run entering at day *t+1*'s **open**, which needs no such
   assumption. If the edge is a closing-print artifact it dies here.
2. **Concentration.** Drop the best trades. An edge carried by five names is a
   story about five names.
3. **Mean versus median.** The standing rule: mean > median means the result
   lives in a right tail.
4. **Period confirmation.** Both halves measured separately, portfolio-level,
   not just per trade.
5. **Parameter neighbourhood.** A true effect degrades smoothly; an overfit peak
   sits alone. The neighbours of the chosen cell are reported next to it.
6. **The trades themselves.** The largest winners are printed with their prices
   and volumes, because three times in this project that is what exposed a
   measurement error no statistic caught.
7. **Liquidity.** The ADV floor is raised and the edge re-measured. A microcap
   edge that only exists below $250k of daily volume is not tradable.

    python research/penny/scripts/long_reversion_verify.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.hypothesis_grid import OOS_START, build_panel  # noqa: E402
from research.penny.scripts.long_reversion_portfolio import (  # noqa: E402
    build_trades, simulate,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("verify")

DROP, HOLD, TARGET, SLOTS = 0.40, 10, 0.50, 30
PRICE_LO, PRICE_HI = 0.50, 2.00
LOOKBACK = 5


def select(u: pd.DataFrame, tr: pd.DataFrame, drop: float = DROP,
           adv_floor: float = 2.5e5, price_lo: float = PRICE_LO,
           price_hi: float = PRICE_HI) -> pd.DataFrame:
    """Signal rows, split-free over the whole holding window, non-overlapping."""
    m = (
        (u["adv20"] >= adv_floor).to_numpy()
        & (~u["any_split"]).to_numpy()
        & (u[f"ret_{LOOKBACK}d"] <= -drop).fillna(False).to_numpy()
        & (u["close"] >= price_lo).to_numpy()
        & (u["close"] < price_hi).to_numpy()
        & tr["exit_date"].notna().to_numpy()
        & tr["net"].notna().to_numpy()
        & (~tr["fwd_split"]).to_numpy()
    )
    sel = tr[m].sort_values(["date", "symbol"])
    keep, busy = [], {}
    for i, row in sel.iterrows():
        if row["symbol"] in busy and row["date"] <= busy[row["symbol"]]:
            continue
        busy[row["symbol"]] = row["exit_date"]
        keep.append(i)
    return sel.loc[keep]


def line(sel: pd.DataFrame, label: str) -> Dict:
    if len(sel) < 60:
        logger.info("  %-40s n=%-5d (too few)", label, len(sel))
        return {}
    e = pstats.cluster_bootstrap_mean(sel["net"], sel["date"], n_boot=3_000)
    star = "  <<<" if e.lo > 0 else ""
    logger.info("  %-40s n=%-5s %+6.2f%%  CI [%+.2f%%, %+.2f%%]  median %+6.2f%%  "
                "target hit %.0f%%%s",
                label, f"{len(sel):,}", 100 * e.mean, 100 * e.lo, 100 * e.hi,
                100 * sel["net"].median(), 100 * sel["took_target"].mean(), star)
    return {"label": label, "n": len(sel), "mean": float(e.mean),
            "lo": float(e.lo), "hi": float(e.hi),
            "median": float(sel["net"].median())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "long_reversion_verify.json"))
    args = ap.parse_args(argv)

    u = build_panel()
    calendar = np.sort(u.loc[(u["adv20"] >= 2.5e5).to_numpy(), "date"].unique())
    res: Dict[str, object] = {}

    tr_close = build_trades(u, HOLD, TARGET, entry_at="close")
    tr_open = build_trades(u, HOLD, TARGET, entry_at="next_open")
    sel_close, sel_open = select(u, tr_close), select(u, tr_open)

    logger.info("=" * 122)
    logger.info("TEST 1 -- EXECUTION: does the edge need the closing print?")
    res["exec_close"] = line(sel_close, "enter at day t close (as calibrated)")
    res["exec_open"] = line(sel_open, "enter at day t+1 OPEN (executable)")

    logger.info("-" * 122)
    logger.info("  the same two, as an account (slots=%d)", SLOTS)
    for lab, sel in (("close", sel_close), ("next_open", sel_open)):
        s = simulate(sel, calendar, SLOTS)
        logger.info("    %-10s Sharpe %5.2f  compounded %+8.1f%%  maxDD %6.1f%%  taken %d",
                    lab, s.sharpe, 100 * s.total_return, 100 * s.max_drawdown, s.n_taken)
        res[f"pf_{lab}"] = s.__dict__

    # everything below uses the conservative entry
    sel = sel_open

    logger.info("-" * 122)
    logger.info("TEST 2 -- CONCENTRATION (executable entry)")
    rs = sel["net"].sort_values()
    conc = {f"drop_top_{k}": float(rs.iloc[:-k].mean()) for k in (3, 5, 10, 25, 50)}
    for k, v in conc.items():
        logger.info("  %-14s %+6.2f%%", k, 100 * v)
    logger.info("  mean %+.2f%%  median %+.2f%%  -> %s", 100 * sel["net"].mean(),
                100 * sel["net"].median(),
                "RIGHT-TAILED, suspect" if sel["net"].mean() > sel["net"].median()
                else "left-tailed, mean is not tail-driven")
    res["concentration"] = conc

    logger.info("-" * 122)
    logger.info("TEST 3 -- PERIOD CONFIRMATION, per trade and as an account")
    for lab, per in (("in-sample", sel["date"] < OOS_START),
                     ("out-of-sample", sel["date"] >= OOS_START)):
        sub = sel[per.to_numpy()]
        res[f"period_{lab}"] = line(sub, lab)
        cal = calendar[calendar < OOS_START] if lab == "in-sample" else calendar[calendar >= OOS_START]
        if len(sub) >= 60:
            s = simulate(sub, cal, SLOTS)
            logger.info("      account: Sharpe %5.2f  compounded %+7.1f%%  maxDD %6.1f%%",
                        s.sharpe, 100 * s.total_return, 100 * s.max_drawdown)
            res[f"pf_{lab}"] = s.__dict__

    logger.info("-" * 122)
    logger.info("TEST 4 -- PARAMETER NEIGHBOURHOOD (executable entry, account Sharpe)")
    logger.info("  a real effect degrades smoothly; an overfit peak stands alone")
    nb: List[Dict] = []
    for drop in (0.30, 0.35, 0.40, 0.45, 0.50):
        s2 = select(u, tr_open, drop=drop)
        if len(s2) < 60:
            continue
        sm = simulate(s2, calendar, SLOTS)
        mark = "  <-- chosen" if abs(drop - DROP) < 1e-9 else ""
        logger.info("    drop>=%.0f%%  n=%-5d per-trade %+6.2f%%  Sharpe %5.2f  "
                    "compounded %+7.1f%%%s",
                    100 * drop, len(s2), 100 * s2["net"].mean(), sm.sharpe,
                    100 * sm.total_return, mark)
        nb.append({"drop": drop, "n": len(s2), "per_trade": float(s2["net"].mean()),
                   "sharpe": sm.sharpe, "total_return": sm.total_return})
    res["neighbourhood_drop"] = nb

    logger.info("-" * 122)
    logger.info("TEST 5 -- LIQUIDITY: raise the ADV floor")
    liq: List[Dict] = []
    for floor in (2.5e5, 5e5, 1e6, 2e6):
        s2 = select(u, tr_open, adv_floor=floor)
        if len(s2) < 60:
            logger.info("    ADV>=$%-8s n=%-5d (too few)", f"{floor:,.0f}", len(s2))
            continue
        e = pstats.cluster_bootstrap_mean(s2["net"], s2["date"], n_boot=2_000)
        sm = simulate(s2, calendar, SLOTS)
        logger.info("    ADV>=$%-9s n=%-5d %+6.2f%%  CI [%+.2f%%, %+.2f%%]  Sharpe %5.2f",
                    f"{floor:,.0f}", len(s2), 100 * e.mean, 100 * e.lo, 100 * e.hi, sm.sharpe)
        liq.append({"adv_floor": floor, "n": len(s2), "mean": float(e.mean),
                    "lo": float(e.lo), "hi": float(e.hi), "sharpe": sm.sharpe})
    res["liquidity"] = liq

    logger.info("-" * 122)
    logger.info("TEST 6 -- THE LARGEST WINNERS, read individually")
    top = sel.nlargest(12, "net")
    for _, x in top.iterrows():
        logger.info("    %-7s %s  entry $%-7.2f  ADV $%-12s net %+7.1f%%  target=%s",
                    x["symbol"], str(x["date"])[:10], x["close"],
                    f"{x['adv20']:,.0f}", 100 * x["net"], bool(x["took_target"]))
    res["top_trades"] = [
        {"symbol": r["symbol"], "date": str(r["date"])[:10], "entry": float(r["close"]),
         "adv20": float(r["adv20"]), "net": float(r["net"])}
        for _, r in top.iterrows()]

    logger.info("-" * 122)
    logger.info("TEST 7 -- SURVIVORSHIP: trades whose stock stopped trading mid-hold")
    logger.info("  these have no tenth forward bar, so `notna` silently DROPS them --")
    logger.info("  and a delisting is not a neutral outcome for a long position")
    g = u.groupby("symbol", sort=False)
    end = u["date"].max()
    base = (
        (u["adv20"] >= 2.5e5).to_numpy() & (~u["any_split"]).to_numpy()
        & (u[f"ret_{LOOKBACK}d"] <= -DROP).fillna(False).to_numpy()
        & (u["close"] >= PRICE_LO).to_numpy() & (u["close"] < PRICE_HI).to_numpy()
        & (~tr_open["fwd_split"]).to_numpy()
    )
    sig, trs = u[base], tr_open[base]
    incomplete = (trs["exit_date"].isna() | trs["net"].isna()).to_numpy()
    # A signal in the last few weeks simply has no forward data yet; that is not
    # survivorship, it is the panel edge. Only the rest are genuine departures.
    near_end = (sig["date"] > end - pd.Timedelta(days=25)).to_numpy()
    gone = incomplete & ~near_end
    logger.info("  signals %s | incomplete %d | panel edge %d | STOPPED TRADING %d",
                f"{len(sig):,}", int(incomplete.sum()), int((incomplete & near_end).sum()),
                int(gone.sum()))

    surv: Dict[str, object] = {"n_signals": int(len(sig)), "n_gone": int(gone.sum())}
    if gone.sum():
        last_close = g["close"].transform("last").to_numpy()[base][gone]
        last_date = g["date"].transform("last").to_numpy()[base][gone]
        entry_px = g["open"].shift(-1).to_numpy()[base][gone]
        r_gone = (last_close / entry_px - 1.0) - u["cost"].to_numpy()[base][gone]
        logger.info("  those %d exited at the symbol's last print: mean %+.2f%%",
                    int(gone.sum()), 100 * float(np.nanmean(r_gone)))
        add = trs[gone].copy()
        add["net"] = r_gone
        add["exit_date"] = last_date
        add["took_target"] = False
        merged = pd.concat([trs[~incomplete], add]).sort_values(["date", "symbol"])
        keep, busy = [], {}
        for i, row in merged.iterrows():
            if row["symbol"] in busy and row["date"] <= busy[row["symbol"]]:
                continue
            busy[row["symbol"]] = row["exit_date"]
            keep.append(i)
        merged = merged.loc[keep].dropna(subset=["net"])
        for lab, frame in (("delisted at last print", merged),
                           ("delisted = -100% (worst case)",
                            merged.assign(net=np.where(merged.index.isin(add.index),
                                                       -1.0, merged["net"])))):
            e = pstats.cluster_bootstrap_mean(frame["net"], frame["date"], n_boot=3_000)
            star = "  <<<" if e.lo > 0 else "  -- CI includes zero"
            logger.info("  %-32s n=%-5s %+6.2f%%  CI [%+.2f%%, %+.2f%%]%s",
                        lab, f"{len(frame):,}", 100 * e.mean, 100 * e.lo, 100 * e.hi, star)
            surv[lab] = {"n": len(frame), "mean": float(e.mean),
                         "lo": float(e.lo), "hi": float(e.hi)}
    res["survivorship"] = surv

    logger.info("=" * 122)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
