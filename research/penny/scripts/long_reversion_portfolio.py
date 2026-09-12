"""long_reversion_portfolio.py — the one surviving long cell, tested honestly.

The cell
--------
From `long_side_calibration.py`, three cells survived Benjamini-Hochberg over
3,006 and confirmed on disjoint periods. All three are the same idea at
different parameters:

    a stock priced $0.50-$2.00 that has fallen >=25% over five sessions
    -> buy at the close, sell on a +50% limit, time stop after 10 sessions

Headline: **+3.87% per trade**, in-sample +4.03%, out-of-sample +3.68%.

Three reasons not to believe that number yet
--------------------------------------------
1. **Median -1.42% against a mean of +3.87%.** The standing rule from LOG §57,
   §61 and §63 is that mean > median means the result lives in a right tail, and
   a right tail is where measurement errors hide.
2. **Reverse splits inside the holding window.** The signal day excludes splits,
   but the position is held ten sessions and a split during those ten fires the
   +50% limit on a share consolidation. This universe hides ~20% of its
   "gainers" that way, and $0.50-$2.00 is exactly the band where a
   listing-compliance consolidation happens. Measured: 54 rows (1.39%) with a
   mean of **+48.96%**.
3. **Overlapping windows counted as independent trades.** AZI entered on
   2026-06-17, 06-23 and 06-24 all reach the same $3.50 high -- one event
   booked three times. A day-clustered bootstrap clusters on the *entry* date
   and cannot see this, so it reports a confidence interval far tighter than the
   evidence supports.

This file fixes all three and then asks the only question that matters: **what
would the account have done?** Round 6 recorded why per-trade expectancy is not
enough -- a rule with a positive arithmetic mean compounded to -15% with a 98%
drawdown, because sigma^2/2 exceeded mu.

The simulation
--------------
Event-driven, capital-constrained, and therefore incapable of double-counting:

* one open position per symbol -- a second signal while already long is skipped,
  which is what removes the overlap;
* a fixed number of concurrent slots, each funded with an equal share of
  capital; a signal that arrives with every slot full is **skipped, not
  queued**, and the number skipped is reported;
* exit on the first session whose high touches +50%, else the close of session
  ten;
* proceeds return to capital on the exit date and compound;
* days with no position are flat and are counted, never dropped.

    python research/penny/scripts/long_reversion_portfolio.py

.. deprecated:: round 15
   **Do not use this simulator for new results.** It charges the cost
   inconsistently -- `R` from `cross_section.forward_return` already carries a
   full round trip and this loop subtracts another `0.5 * cost` on top (1.5x),
   while a `R_t*` target column is gross and ends up charged 0.5x. It also
   schedules exits on candidate-days rather than market sessions and discards
   the daily equity curve it computes. Use `research/penny/strategy/book.py`,
   which fixes all three and is unit-tested. See LOG 85.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.hypothesis_grid import OOS_START, build_panel  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("longpf")

HOLD = 10
TARGET = 0.50
PRICE_LO, PRICE_HI = 0.50, 2.00
DROP = 0.25
LOOKBACK = 5
ADV_FLOOR = 2.5e5


def build_trades(u: pd.DataFrame, hold: int = HOLD, target: float = TARGET,
                 stop: Optional[float] = None, entry_at: str = "close") -> pd.DataFrame:
    """One row per candidate entry, with its exit date and net return.

    Exit date is read from the symbol's own shifted date column rather than by
    adding sessions to a market calendar -- a thinly traded name does not print
    every session, and assuming it does would exit trades on days the stock
    never traded.

    ``entry_at`` decides how literally the signal is obeyed. The condition is
    computed from day *t*'s close, so entering at that same close assumes a
    market-on-close order placed with knowledge of the closing price -- a mild
    look-ahead that is standard but worth not depending on. ``next_open`` enters
    at day *t+1*'s open instead, which is unambiguously executable, and shifts
    the whole barrier window forward with it.
    """
    g = u.groupby("symbol", sort=False)
    off = 1 if entry_at == "next_open" else 0
    rng = range(1 + off, hold + 1 + off)
    hi = pd.concat([g["high"].shift(-k) for k in rng], axis=1)
    lo = pd.concat([g["low"].shift(-k) for k in rng], axis=1)
    dt = pd.concat([g["date"].shift(-k) for k in rng], axis=1)
    cl = pd.concat([g["close"].shift(-k) for k in rng], axis=1)
    sp = pd.concat([g["any_split"].shift(-k) for k in range(1, hold + 1 + off)], axis=1)
    hi.columns = lo.columns = dt.columns = cl.columns = range(hold)

    entry = (g["open"].shift(-1) if entry_at == "next_open" else u["close"]).to_numpy()
    hit = hi.to_numpy() >= entry[:, None] * (1.0 + target)
    stopped = (lo.to_numpy() <= entry[:, None] * (1.0 - stop)) if stop else np.zeros_like(hit)

    first_hit = np.where(hit.any(axis=1), hit.argmax(axis=1), hold - 1)
    first_stop = np.where(stopped.any(axis=1), stopped.argmax(axis=1), hold - 1)

    took_target = hit.any(axis=1) & (~stopped.any(axis=1) | (first_hit < first_stop))
    took_stop = stopped.any(axis=1) & ~took_target
    k = np.where(took_target, first_hit, np.where(took_stop, first_stop, hold - 1))

    rows = np.arange(len(u))
    gross = np.where(took_target, target,
                     np.where(took_stop, -(stop or 0.0),
                              cl.to_numpy()[rows, hold - 1] / entry - 1.0))
    out = pd.DataFrame({
        "symbol": u["symbol"].to_numpy(),
        "date": u["date"].to_numpy(),
        "exit_date": dt.to_numpy()[rows, k],
        "close": entry,
        "gross": gross,
        "net": gross - u["cost"].to_numpy(),
        "took_target": took_target,
        # a split on ANY held session, not just the exit session
        "fwd_split": sp.fillna(False).to_numpy().any(axis=1),
        "adv20": u["adv20"].to_numpy(),
    })
    return out


def signal_mask(u: pd.DataFrame) -> np.ndarray:
    return (
        (u["adv20"] >= ADV_FLOOR).to_numpy()
        & (~u["any_split"]).to_numpy()
        & (u[f"ret_{LOOKBACK}d"] <= -DROP).fillna(False).to_numpy()
        & (u["close"] >= PRICE_LO).to_numpy()
        & (u["close"] < PRICE_HI).to_numpy()
    )


@dataclass
class Sim:
    n_taken: int
    n_skipped_slot: int
    n_skipped_open: int
    mean_daily: float
    sharpe: float
    total_return: float
    max_drawdown: float
    pct_days_active: float
    worst_day: float
    avg_positions: float


def simulate(trades: pd.DataFrame, calendar: np.ndarray, slots: int) -> Sim:
    """Event-driven, capital-constrained. Skipped signals are counted, not queued."""
    trades = trades.sort_values(["date", "symbol"]).reset_index(drop=True)
    by_day: Dict[pd.Timestamp, pd.DataFrame] = dict(tuple(trades.groupby("date")))

    capital = 1.0
    open_pos: List[Tuple[pd.Timestamp, str, float, float]] = []  # exit, sym, invested, net
    equity, active, taken, sk_slot, sk_open, pos_count = [], [], 0, 0, 0, []

    for day in calendar:
        # --- exits first: capital freed today is available today -----------
        still = []
        for exit_date, sym, invested, net in open_pos:
            if exit_date is not None and exit_date <= day:
                capital += invested * (1.0 + net)
            else:
                still.append((exit_date, sym, invested, net))
        open_pos = still
        held = {p[1] for p in open_pos}

        # --- entries -------------------------------------------------------
        todays = by_day.get(day)
        if todays is not None and len(todays):
            for _, t in todays.iterrows():
                if t["symbol"] in held:
                    sk_open += 1
                    continue
                if len(open_pos) >= slots:
                    sk_slot += 1
                    continue
                size = capital / max(1, slots - len(open_pos))
                size = min(size, capital)
                if size <= 1e-12:
                    sk_slot += 1
                    continue
                capital -= size
                open_pos.append((t["exit_date"], t["symbol"], size, t["net"]))
                held.add(t["symbol"])
                taken += 1

        mtm = capital + sum(p[2] for p in open_pos)
        equity.append(mtm)
        active.append(1.0 if open_pos else 0.0)
        pos_count.append(len(open_pos))

    eq = pd.Series(equity, index=pd.Index(calendar, name="date"))
    daily = eq.pct_change().dropna()
    peak = eq.cummax()
    dd = float(((eq - peak) / peak).min())
    sd = float(daily.std())
    return Sim(
        n_taken=taken, n_skipped_slot=sk_slot, n_skipped_open=sk_open,
        mean_daily=float(daily.mean()),
        sharpe=float(daily.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0,
        total_return=float(eq.iloc[-1] - 1.0), max_drawdown=dd,
        pct_days_active=float(np.mean(active)),
        worst_day=float(daily.min()) if len(daily) else 0.0,
        avg_positions=float(np.mean(pos_count)),
    )


def _sym_of(invested, net):        # positions are keyed by object identity only
    return None                    # -- `held` is maintained explicitly below


def _syms(open_pos):
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "long_reversion_portfolio.json"))
    args = ap.parse_args(argv)

    u = build_panel()
    trades = build_trades(u)
    m = signal_mask(u) & trades["exit_date"].notna().to_numpy()
    t = trades[m].copy()
    logger.info("raw candidate entries: %s over %d sessions",
                f"{len(t):,}", t["date"].nunique())

    results: Dict[str, object] = {}

    logger.info("=" * 120)
    logger.info("STEP 1 -- remove reverse splits occurring INSIDE the holding window")
    for lab, sel in (("as measured", t), ("split-free", t[~t["fwd_split"]])):
        est = pstats.cluster_bootstrap_mean(sel["net"], sel["date"], n_boot=3_000)
        logger.info("  %-14s n=%-7s %+6.2f%%  CI [%+.2f%%, %+.2f%%]  median %+6.2f%%  "
                    "target hit %.1f%%",
                    lab, f"{len(sel):,}", 100 * est.mean, 100 * est.lo, 100 * est.hi,
                    100 * sel["net"].median(), 100 * sel["took_target"].mean())
        results[f"pertrade_{lab.replace(' ', '_')}"] = {
            "n": len(sel), "mean": float(est.mean), "lo": float(est.lo),
            "hi": float(est.hi), "median": float(sel["net"].median())}
    t = t[~t["fwd_split"]].copy()

    logger.info("-" * 120)
    logger.info("STEP 2 -- one position per symbol at a time (removes overlapping windows)")
    t = t.sort_values(["date", "symbol"])
    keep, busy = [], {}
    for i, row in t.iterrows():
        s, d = row["symbol"], row["date"]
        if s in busy and d <= busy[s]:
            continue
        busy[s] = row["exit_date"]
        keep.append(i)
    nt = t.loc[keep]
    est = pstats.cluster_bootstrap_mean(nt["net"], nt["date"], n_boot=3_000)
    logger.info("  non-overlapping n=%-7s %+6.2f%%  CI [%+.2f%%, %+.2f%%]  median %+6.2f%%  "
                "(dropped %s overlapping entries)",
                f"{len(nt):,}", 100 * est.mean, 100 * est.lo, 100 * est.hi,
                100 * nt["net"].median(), f"{len(t) - len(nt):,}")
    results["non_overlapping"] = {"n": len(nt), "mean": float(est.mean),
                                  "lo": float(est.lo), "hi": float(est.hi),
                                  "median": float(nt["net"].median())}

    logger.info("-" * 120)
    logger.info("STEP 3 -- concentration: is it a handful of trades?")
    rs = nt["net"].sort_values()
    conc = {f"drop_top_{k}": float(rs.iloc[:-k].mean()) for k in (5, 10, 25, 50)}
    for k, v in conc.items():
        logger.info("  %-14s %+6.2f%%", k, 100 * v)
    results["concentration"] = conc

    logger.info("-" * 120)
    logger.info("STEP 4 -- period stability")
    for lab, sel in (("in-sample", nt[nt["date"] < OOS_START]),
                     ("out-of-sample", nt[nt["date"] >= OOS_START])):
        if len(sel) < 50:
            continue
        e = pstats.cluster_bootstrap_mean(sel["net"], sel["date"], n_boot=3_000)
        logger.info("  %-14s n=%-6s %+6.2f%%  CI [%+.2f%%, %+.2f%%]",
                    lab, f"{len(sel):,}", 100 * e.mean, 100 * e.lo, 100 * e.hi)
        results[f"period_{lab}"] = {"n": len(sel), "mean": float(e.mean),
                                    "lo": float(e.lo), "hi": float(e.hi)}

    logger.info("-" * 120)
    logger.info("STEP 5 -- THE ACCOUNT: event-driven, capital-constrained, compounded")
    calendar = np.sort(u.loc[(u["adv20"] >= ADV_FLOOR).to_numpy(), "date"].unique())
    sims = []
    for slots in (3, 5, 10, 20):
        s = simulate(nt, calendar, slots)
        logger.info("  slots=%-3d taken=%-5d skipped(full)=%-5d skipped(already long)=%-5d  "
                    "mean/day %+.4f%%  Sharpe %5.2f  compounded %+8.1f%%  maxDD %6.1f%%  "
                    "active %3.0f%%  worst day %+.1f%%",
                    slots, s.n_taken, s.n_skipped_slot, s.n_skipped_open,
                    100 * s.mean_daily, s.sharpe, 100 * s.total_return,
                    100 * s.max_drawdown, 100 * s.pct_days_active, 100 * s.worst_day)
        sims.append({"slots": slots, **s.__dict__})
    results["portfolio"] = sims

    logger.info("-" * 120)
    logger.info("STEP 6 -- CALIBRATION at the portfolio level, which is where it matters")
    logger.info("  per-trade expectancy was positive at every setting below; most still lose money")
    logger.info("  %-6s %-5s %-6s %-6s %7s %7s %9s %8s %7s", "drop", "hold",
                "target", "slots", "mean/d", "Sharpe", "compound", "maxDD", "taken")
    sweep: List[Dict] = []
    for hold in (5, 10):
        for target in (0.20, 0.30, 0.50):
            tr = build_trades(u, hold=hold, target=target)
            for drop in (0.25, 0.40):
                base = (
                    (u["adv20"] >= ADV_FLOOR).to_numpy()
                    & (~u["any_split"]).to_numpy()
                    & (u[f"ret_{LOOKBACK}d"] <= -drop).fillna(False).to_numpy()
                    & (u["close"] >= PRICE_LO).to_numpy()
                    & (u["close"] < PRICE_HI).to_numpy()
                    & tr["exit_date"].notna().to_numpy()
                    & (~tr["fwd_split"]).to_numpy()
                )
                sel = tr[base].sort_values(["date", "symbol"])
                if len(sel) < 200:
                    continue
                keep, busy = [], {}
                for i, row in sel.iterrows():
                    s, d = row["symbol"], row["date"]
                    if s in busy and d <= busy[s]:
                        continue
                    busy[s] = row["exit_date"]
                    keep.append(i)
                sel = sel.loc[keep]
                pt = float(sel["net"].mean())
                for slots in (10, 20, 30, 50):
                    sm = simulate(sel, calendar, slots)
                    flag = "  <<<" if sm.sharpe > 1.0 and sm.total_return > 0 else ""
                    logger.info("  %-6s %-5d %-6s %-6d %+6.3f%% %7.2f %+8.1f%% %7.1f%% %7d%s",
                                f"{100*drop:.0f}%", hold, f"{100*target:.0f}%", slots,
                                100 * sm.mean_daily, sm.sharpe, 100 * sm.total_return,
                                100 * sm.max_drawdown, sm.n_taken, flag)
                    sweep.append({"drop": drop, "hold": hold, "target": target,
                                  "slots": slots, "per_trade": pt,
                                  "n_signals": len(sel), **sm.__dict__})
    results["calibration"] = sweep
    best = [s for s in sweep if s["sharpe"] > 1.0 and s["total_return"] > 0]
    logger.info("-" * 120)
    if not best:
        logger.info("NO calibration reaches Sharpe > 1.0 with a positive compounded return.")
    else:
        b = max(best, key=lambda s: s["sharpe"])
        logger.info("BEST: drop>=%.0f%% hold=%d target=%.0f%% slots=%d -> Sharpe %.2f, "
                    "compounded %+.1f%%, maxDD %.1f%%, per-trade %+.2f%%",
                    100 * b["drop"], b["hold"], 100 * b["target"], b["slots"],
                    b["sharpe"], 100 * b["total_return"], 100 * b["max_drawdown"],
                    100 * b["per_trade"])

    logger.info("=" * 120)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
