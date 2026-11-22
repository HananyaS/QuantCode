"""C3's dev break-even cost per round trip, in bp of entry notional (PAPER_C3 §4, G-cost).

A diagnostic of the closed L1 cell on the dev window L1 already read; not a new
trial. Each trade is charged a uniform total cost of x bp per round trip in
place of costs.BASE. The break-even is the x at which C3's dev Sharpe equals
SPY's buy-and-hold dev Sharpe (l1_dev.json). Prints a small table and writes
research/liquid/results/c3_breakeven.json.

    python -m research.liquid.scripts.c3_breakeven
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from research.liquid import cells, costs
from research.liquid.calendar import DEV_END, DEV_START
from research.liquid.data import load
from research.penny.eval import benchmark as bm

OUT = Path("research/liquid/results/c3_breakeven.json")
L1 = Path("data/research_liquid/l1_dev.json")


def sharpe_at(trades: list[cells.Trade], dates, bp: float) -> float:
    """Dev Sharpe with a uniform cost of `bp` per round trip (a fraction of entry notional)."""
    r = np.zeros(len(dates))
    pos = {d: i for i, d in enumerate(dates)}
    for t in trades:
        r[pos[t.date]] += t.weight * (t.gross - bp * 1e-4)
    return float(bm.sharpe(r[1:]))


def git_state() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    return {"sha": sha, "dirty": dirty}


def main() -> None:
    m = load("SPY", DEV_START, DEV_END)
    tr = cells.c3(m)
    tab = {r["index"]: r for r in json.loads(L1.read_text())["cells"]["C3"]["comparison"]["table"]}
    spy = float(tab["SPY"]["sharpe"])
    base_bp = float(np.mean([(t.gross - t.net(costs.BASE)) * 1e4 for t in tr]))
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sharpe_at(tr, m.dates, mid) > spy else (lo, mid)
    grid = {f"{x:.2f}": sharpe_at(tr, m.dates, x) for x in (0.0, 0.5, 1.0, base_bp, 1.5, 2.0, 2.5, 3.0)}
    out = {"trades": len(tr), "spy_dev_sharpe": spy, "base_model_mean_bp": base_bp,
           "base_model_sharpe": float(bm.sharpe(cells.daily(tr, m.dates, costs.BASE).to_numpy()[1:])),
           "breakeven_bp": (lo + hi) / 2, "sharpe_by_uniform_bp": grid,
           "mean_gross_bp": float(np.mean([t.gross for t in tr]) * 1e4),
           "mean_weight": float(np.mean([t.weight for t in tr])), "git": git_state()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
