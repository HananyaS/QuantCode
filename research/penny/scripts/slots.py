"""slots.py — bound the position count directly instead of thinning the signal.

Why the calendar-cohort approach was wrong
-------------------------------------------
`feasibility.py` reduced positions by entering only every tau-th session. That
fails twice over: it discards 80-95% of the signal flow, and it collapses the
sample from 973 cohorts to 47, so the veto's contribution stops being
measurable (intervals 14 points wide). Duplicates also mean N falls far more
slowly than H/tau predicts -- at H=60 going from tau=1 to tau=5 took N_p90 from
707 to 487, not to 141.

The right structure
-------------------
Keep the daily signal, and bound the portfolio directly:

    N slots, each holding one name for H sessions.
    Each session: release slots whose holding period is up, then fill every free
    slot from today's post-veto survivors, best score first.

Turnover is then N/H entries per session -- for N=40, H=60 that is **0.67 orders
a day**, against 42 in round 11 -- and the weight per name is a fixed 1/N.

This is a *capacity* constraint, not a signal change. When more names qualify
than there are free slots the surplus is skipped, and the skip count is
reported: a strategy that silently drops most of its signals is a different
strategy from the one that was validated.

Leakage discipline (Amendment 2)
--------------------------------
Composite signs stay frozen at the pre-2020 fit. N, H and the fill rule are
chosen on development only. Slots are filled from information available that
session; a name entered on day t is priced from day t+1's open.

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
from research.penny.scripts.run_composite import add_composite  # noqa: E402
from research.penny.scripts.veto_holdout import frame  # noqa: E402
from research.penny.strategy.cross_section import MIN_CELL  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")

SPEC = json.loads(Path("research/penny/specs/VETO.json").read_text(encoding="utf-8"))
SIGNS = {k: float(v) for k, v in SPEC["composite_signs"].items()}
VETO_FRAC = SPEC["veto_fraction"]
from research.penny.dates import DEV_START                             # noqa: E402


@dataclass
class SlotConfig:
    n_slots: int
    H: int
    rule: str          # "top" | "random" | "noveto" (control: fill from ALL candidates)
    stagger: bool = True   # cap entries at ceil(N/H) per session


def run_slots(df: pd.DataFrame, cfg: SlotConfig, seed: int = 0) -> Optional[Dict]:
    rng = np.random.default_rng(seed)
    sess = np.sort(df["date"].unique())
    by_day = dict(tuple(df.groupby("date")))

    # A TRUE EQUITY SIMULATION, not a sum of return contributions.
    #
    # Booking r/N on the exit day and compounding the mean of that series is
    # only valid for small r. These are 60-session microcap returns with a
    # standard deviation near 70%, and the two methods disagreed by a factor of
    # two. Here `cash` and per-slot capital are tracked in currency: a slot
    # entered with `equity/N` returns `capital*(1+r)` to cash on exit, and the
    # next entry draws a fresh `equity/N`. That is equal-weight rebalancing at
    # entry, which is what the strategy actually does.
    # slot -> (exit_index, symbol, net_return, capital)
    open_slots: List[Optional[Tuple[int, str, float, float]]] = [None] * cfg.n_slots
    cash = 1.0
    equity = np.ones(len(sess))
    n_held, n_entered, n_skipped = [], 0, 0

    for i, d in enumerate(sess):
        # 1. release finished slots, returning capital-plus-P&L to cash
        for s in range(cfg.n_slots):
            p = open_slots[s]
            if p and p[0] <= i:
                cash += p[3] * (1.0 + p[2])
                open_slots[s] = None

        # 2. fill free slots -- but STAGGERED.
        #
        # Filling greedily is a trap at small N. With 30 slots and a 60-session
        # hold, all 30 fill on day one and exit together: the book becomes a
        # single synchronised cohort with maximum exposure to the common factor,
        # which is exactly the term diversification cannot remove. The variance
        # decomposition puts sd(mu_t) at 25-33% against sd(eta) of 41-61%, so a
        # synchronised book throws away most of the achievable Sharpe.
        #
        # Capping entries at ceil(N/H) per session spreads the book evenly over
        # H cohorts, so the common factor is averaged across H different entry
        # days instead of one.
        free = [s for s in range(cfg.n_slots) if open_slots[s] is None]
        if cfg.stagger:
            free = free[:max(1, int(np.ceil(cfg.n_slots / cfg.H)))]
        g = by_day.get(d)
        if free and g is not None and len(g) >= MIN_CELL:
            if cfg.rule == "noveto":
                pool = g
            else:
                keep = max(1, int(np.ceil((1.0 - VETO_FRAC) * len(g))))
                pool = g.nlargest(keep, "COMPOSITE")
            held_syms = {p[1] for p in open_slots if p}
            pool = pool[~pool["symbol"].isin(held_syms)]
            if len(pool):
                if cfg.rule == "random":
                    pool = pool.sample(min(len(pool), len(free)),
                                       random_state=int(rng.integers(1 << 31)))
                else:
                    pool = pool.nlargest(min(len(pool), len(free)), "COMPOSITE")
                deployed = sum(p[3] for p in open_slots if p)
                for s, (_, row) in zip(free, pool.iterrows()):
                    size = (cash + deployed) / cfg.n_slots
                    size = min(size, cash)
                    if size <= 1e-12:
                        break
                    exit_i = min(i + cfg.H, len(sess) - 1)
                    net = float(row["R"] - 0.5 * row["cost"])
                    cash -= size
                    open_slots[s] = (exit_i, row["symbol"], net, size)
                    n_entered += 1
                n_skipped += max(0, (len(g) - len(pool)))
        n_held.append(sum(1 for p in open_slots if p))
        equity[i] = cash + sum(p[3] for p in open_slots if p)

    if n_entered < 40:
        return None
    eq = pd.Series(equity, index=sess)
    r = eq.pct_change().fillna(0.0)
    years = len(sess) / 252.0
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    dd = float(((eq - eq.cummax()) / eq.cummax()).min())
    b = r.to_numpy()[rng.integers(0, len(r), size=(4000, len(r)))].mean(axis=1)
    return {"n_slots": cfg.n_slots, "H": cfg.H, "rule": cfg.rule,
            "entered": n_entered, "skipped": n_skipped,
            "orders_per_session": n_entered / len(sess),
            "N_med": float(np.median(n_held)), "N_max": int(np.max(n_held)),
            "weight_pct": 100.0 / cfg.n_slots,
            "ann": 100 * ann, "sharpe": sharpe, "maxdd": 100 * dd,
            "daily_lo": 252 * 100 * float(np.percentile(b, 2.5)),
            "daily_hi": 252 * 100 * float(np.percentile(b, 97.5))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "slots.json"))
    args = ap.parse_args(argv)

    rows: List[Dict] = []
    for H in (40, 60, 90):
        df = add_composite(frame(H, SPEC["population"]["crash_pct"], args.period),
                           SIGNS, "COMPOSITE")
        if args.period == "dev":
            df = df[df["date"] >= DEV_START]
        for n in (20, 30, 40, 60):
            for rule in ("top", "random", "noveto"):
                r = run_slots(df, SlotConfig(n, H, rule))
                if r:
                    rows.append(r)
        print(f"  ... H={H} done", flush=True)

    res = pd.DataFrame(rows)
    print("=" * 122)
    print(f"SLOT-CONSTRAINED PORTFOLIO  ({args.period}, composite signs frozen at the "
          f"pre-2020 fit)")
    print("noveto = the same slots filled from ALL candidates; it isolates the veto")
    print("=" * 122)
    print(f"  {'H':>3}{'slots':>7}{'rule':>9}{'wt%':>7}{'orders/day':>12}{'skipped':>9}"
          f"{'annual':>10}{'Sharpe':>8}{'maxDD':>8}{'95% CI on annual':>22}")
    for H in sorted(res["H"].unique()):
        for n in sorted(res["n_slots"].unique()):
            sub = res[(res.H == H) & (res.n_slots == n)]
            for _, r in sub.iterrows():
                star = "  <<<" if (r["rule"] == "top" and r["sharpe"] > 1.0
                                   and r["daily_lo"] > 0) else ""
                print(f"  {int(r.H):>3}{int(r.n_slots):>7}{r['rule']:>9}"
                      f"{r.weight_pct:>7.2f}{r.orders_per_session:>12.2f}"
                      f"{int(r.skipped):>9}{r['ann']:>+9.2f}%{r.sharpe:>8.2f}"
                      f"{r.maxdd:>7.1f}%  [{r.daily_lo:>+6.1f}%,{r.daily_hi:>+6.1f}%]{star}")
            print()
    res.to_json(args.out, orient="records", indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
