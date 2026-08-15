"""evaluate_strategy.py — score one strategy spec, print one number.

This is the verification command for the automated search. It reads a `Spec`
from JSON, evaluates it, prints a diagnostic block, and prints the metric on the
final line so a shell pipeline can take it.

The seal
--------
Signals on or after **2024-01-01 are refused** unless `--unseal` is passed. The
search optimises a metric; if that metric can see the holdout, the search
overfits the holdout and the final number means nothing. Every unseal is logged
loudly, and the number of times it has happened is the number of times the
holdout has been spent.

The metric
----------
`10000 x (lower bound of the 95% day-clustered bootstrap CI of net per-trade
return)`, in basis points, higher is better.

The **lower bound**, not the mean, because a search that maximises a point
estimate finds lucky tails -- the failure mode of §57, §61 and §65. Maximising
the lower bound demands the effect be both large and well-supported.

Two hard vetoes return the sentinel `-9999`:

* fewer than `MIN_TRADES` trades or `MIN_DAYS` distinct days -- an interval on
  40 trades is not evidence;
* **a compounded portfolio return at or below zero** -- §49, §56 and §65 all
  produced positive per-trade expectancy that compounded to a loss, because
  sigma^2/2 exceeded mu. A per-trade edge that an account cannot hold is not a
  strategy, and the search must not be able to win by finding one.

    python research/penny/scripts/evaluate_strategy.py --spec specs/foo.json
    python research/penny/scripts/evaluate_strategy.py --spec specs/foo.json --quiet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.etl.screens import adjustment_tripwire  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.long_reversion_portfolio import simulate  # noqa: E402
from research.penny.strategy.engine import Spec, build_trades  # noqa: E402

logger = logging.getLogger("eval")

SEAL_DATE = pd.Timestamp("2024-01-01")
MIN_TRADES = 200
MIN_DAYS = 100
SENTINEL = -9999.0
PANEL = bars.DATA_ROOT / "panel_strategy.parquet"

_panel_cache: Optional[pd.DataFrame] = None


def _selected_rows(u: pd.DataFrame, spec: Spec, trades: pd.DataFrame) -> np.ndarray:
    """Boolean mask of the exact (symbol, date) pairs that became trades."""
    key = pd.MultiIndex.from_arrays([u["symbol"], u["date"]])
    want = pd.MultiIndex.from_arrays([trades["symbol"], trades["date"]])
    return key.isin(want)


def load_panel() -> pd.DataFrame:
    global _panel_cache
    if _panel_cache is None:
        _panel_cache = pd.read_parquet(PANEL)
    return _panel_cache


def evaluate(spec: Spec, unseal: bool = False,
             panel: Optional[pd.DataFrame] = None) -> Dict:
    u = panel if panel is not None else load_panel()
    trades = build_trades(u, spec)

    period = "SEALED HOLDOUT (2024-01-01 onward)" if unseal else "development (< 2024-01-01)"
    if unseal:
        trades = trades[trades["date"] >= SEAL_DATE]
    else:
        trades = trades[trades["date"] < SEAL_DATE]

    res: Dict = {"spec": spec.name, "period": period, "n": int(len(trades)),
                 "metric": SENTINEL}
    if len(trades) < MIN_TRADES or trades["date"].nunique() < MIN_DAYS:
        res["veto"] = (f"only {len(trades)} trades on "
                       f"{trades['date'].nunique() if len(trades) else 0} days "
                       f"(need {MIN_TRADES}/{MIN_DAYS})")
        return res

    est = pstats.cluster_bootstrap_mean(trades["net"], trades["date"], n_boot=3_000)
    liquid = u.loc[(u["adv20"] >= spec.adv_floor).to_numpy(), "date"].unique()
    liquid = np.sort(liquid)
    lo_d, hi_d = trades["date"].min(), max(trades["exit_date"].max(), trades["date"].max())
    cal = liquid[(liquid >= np.datetime64(lo_d)) & (liquid <= np.datetime64(hi_d))]
    sim = simulate(trades, cal, spec.slots)

    res.update({
        "mean": float(est.mean), "lo": float(est.lo), "hi": float(est.hi),
        "median": float(trades["net"].median()),
        "n_days": int(est.n_days),
        "target_rate": float(trades["took_target"].mean()),
        "stop_rate": float(trades["took_stop"].mean()),
        "compounded": sim.total_return, "sharpe": sim.sharpe,
        "max_drawdown": sim.max_drawdown, "n_taken": sim.n_taken,
        "worst_trade": float(trades["net"].min()),
        "drop_best_25": float(trades["net"].sort_values().iloc[:-25].mean()),
        # Must be measured on the SELECTED rows. Matching on dates alone picks
        # up every symbol trading that day and reports the universe's split
        # rate, not the strategy's -- a tripwire that cannot fire usefully.
        "tripwire": adjustment_tripwire(u, _selected_rows(u, spec, trades)),
    })

    if sim.total_return <= 0:
        res["veto"] = (f"per-trade CI lower bound {100*est.lo:+.2f}% but the account "
                       f"compounds to {100*sim.total_return:+.1f}% -- sigma^2/2 > mu")
        return res

    res["metric"] = 1e4 * float(est.lo)
    return res


def report(res: Dict) -> None:
    print(f"  spec            : {res['spec']}")
    print(f"  period          : {res['period']}")
    print(f"  trades          : {res['n']:,}")
    if "veto" in res and "mean" not in res:
        print(f"  VETO            : {res['veto']}")
        return
    print(f"  per trade       : {100*res['mean']:+.3f}%  CI [{100*res['lo']:+.3f}%, "
          f"{100*res['hi']:+.3f}%]   median {100*res['median']:+.3f}%")
    print(f"  target / stop   : {100*res['target_rate']:.0f}% / {100*res['stop_rate']:.0f}%")
    print(f"  account         : compounded {100*res['compounded']:+.1f}%   "
          f"Sharpe {res['sharpe']:.2f}   maxDD {100*res['max_drawdown']:.1f}%")
    print(f"  robustness      : drop best 25 {100*res['drop_best_25']:+.3f}%   "
          f"worst trade {100*res['worst_trade']:+.1f}%")
    tw = res["tripwire"]
    flag = "  <-- LEAKAGE SUSPECTED" if tw["share_gt_1_5"] > 0.05 else ""
    print(f"  adj/unadj check : {100*tw['share_gt_1_5']:.1f}% of rows > 1.5x{flag}")
    if "veto" in res:
        print(f"  VETO            : {res['veto']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--unseal", action="store_true",
                    help="Evaluate on the SEALED holdout. Every use spends it.")
    ap.add_argument("--quiet", action="store_true", help="Print only the metric.")
    args = ap.parse_args(argv)

    spec = Spec.load(args.spec)
    if args.unseal:
        print("=" * 78, file=sys.stderr)
        print("!! UNSEALING THE HOLDOUT -- this is a one-shot resource !!", file=sys.stderr)
        print("=" * 78, file=sys.stderr)

    res = evaluate(spec, unseal=args.unseal)
    if not args.quiet:
        report(res)
    print(f"{res['metric']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
