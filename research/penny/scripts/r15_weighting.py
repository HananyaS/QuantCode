"""r15_weighting.py — H4 re-run, after the inverse-volatility arm was a no-op.

The defect
-----------
`risk_shaped.py:120` normalises `1/sigma` over the pool it just SAMPLED, and its
stagger admits `ceil(n_slots / H)` fills a session -- which is one for every
`n_slots <= H`. With a one-name pool `iv / iv.mean()` is identically 1, so
`weight="invvol"` returns exactly the equal-weight book. Every inverse-vol
result the project has reported at N <= 60 is an equal-weight result; the first
pass of H4 reproduced it to the digit (0.46/0.46 at N=15, 0.73/0.73 at N=30)
before the cause was found.

`strategy/book.py` now normalises against the day's **eligible** set, so the tilt
survives a one-name fill. This re-runs H4 on that.

Why inverse-vol is the one alternative worth testing
-----------------------------------------------------
§78 found E[R] flat in the score inside the survivors, which is exactly the
condition under which equal weight is optimal -- there is no expected-return
dispersion to tilt towards. What there IS is volatility dispersion, and the
validated effect is that low realised volatility predicts higher return in this
population (IC -0.0845, t -12.0). So inverse-vol is not a generic risk overlay
here: it doubles down on the one thing the project has actually validated.

Kelly is reported alongside as a scaling reference, not as a weighting scheme:
with a right-skewed barrier payoff the full-Kelly fraction is small and the
half-Kelly convention is what the literature recommends for estimation error
(MacLean, Thorp & Ziemba 2011).

    python research/penny/scripts/r15_weighting.py --period dev
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

from research.penny.etl import bars                                    # noqa: E402
from research.penny.scripts.r15_concurrency import (                   # noqa: E402
    attach_sector, traded_return)
from research.penny.scripts.reproduce_round14 import (                 # noqa: E402
    HOLD, TARGET, VETO_FRAC, VOL_CAP, build_frame, market_calendar)
from research.penny.strategy.book import BookConfig, run_book          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("h4")

SLOTS = [15, 30, 60, 150]
SEEDS = 15


def kelly_fraction(r: np.ndarray) -> float:
    """Full-Kelly growth-optimal fraction for the measured per-trade payoff.

    For a general payoff the growth-optimal `f` maximises E[log(1 + f*R)];
    solved numerically here rather than with the two-outcome shortcut, because
    this payoff is a barrier touch against a continuous time-stop tail.
    """
    r = r[np.isfinite(r) & (r > -1.0)]
    if len(r) < 100 or r.mean() <= 0:
        return 0.0
    grid = np.linspace(0.001, 1.0, 400)
    g = [np.mean(np.log1p(f * r)) for f in grid]
    return float(grid[int(np.argmax(g))])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.period == "holdout" and not args.unseal:
        raise SystemExit("refusing to read the sealed holdout without --unseal")

    cal = market_calendar(args.period)
    df = attach_sector(build_frame(args.period))
    r = traded_return(df).to_numpy()
    kf = kelly_fraction(r)
    logger.info("frame %s rows | per-trade mean %+.2f%% sd %.1f%%",
                f"{len(df):,}", 100 * np.nanmean(r), 100 * np.nanstd(r))

    print("=" * 104)
    print(f"H4 — WEIGHTING ({args.period}), inverse-vol normalised on the day's "
          f"ELIGIBLE set")
    print("=" * 104)
    print(f"  full-Kelly fraction on the measured payoff: {kf:.3f}  "
          f"(half-Kelly {kf/2:.3f})")
    print(f"  at N=60 the book already runs {1/60:.4f} of capital per name, so "
          f"position size is\n  far below Kelly and the binding constraint is "
          f"breadth, not leverage.\n")
    print(f"  {'N':>5}{'weight':>10}{'Sharpe':>9}{'+/-':>7}{'annual':>9}"
          f"{'maxDD':>9}{'wt sd':>9}{'delta':>8}")
    rows: List[Dict] = []
    for n in SLOTS:
        base = None
        for w in ("equal", "invvol"):
            sh, ann, dd, wsd = [], [], [], []
            for s in range(SEEDS):
                res = run_book(df, cal, BookConfig(
                    n_slots=n, hold=HOLD, vol_cap=VOL_CAP, veto_frac=VETO_FRAC,
                    target=TARGET, weight=w, cost_mode="size_aware"), seed=s)
                if res is None:
                    continue
                sh.append(res.sharpe); ann.append(res.ann)
                dd.append(res.max_drawdown)
                wsd.append(float(res.trades["weight"].std()) * n)
            if not sh:
                continue
            m = float(np.mean(sh))
            base = m if base is None else base
            row = {"n_slots": n, "weight": w, "sharpe": m,
                   "sharpe_sd": float(np.std(sh, ddof=1)),
                   "ann": float(np.mean(ann)), "maxdd": float(np.mean(dd)),
                   "weight_dispersion": float(np.mean(wsd))}
            rows.append(row)
            print(f"  {n:>5}{w:>10}{m:>9.2f}{row['sharpe_sd']:>7.2f}"
                  f"{row['ann']:>+8.1f}%{row['maxdd']:>8.1f}%"
                  f"{row['weight_dispersion']:>9.3f}"
                  f"{(m - base):>+8.2f}")
        print()
    print("=" * 104)

    out = args.out or str(bars.DATA_ROOT / f"r15_weighting_{args.period}.json")
    Path(out).write_text(json.dumps({"kelly_full": kf, "rows": rows}, indent=1,
                                    default=float), encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
