"""tune_objective.py — the autoresearch verify command.

Prints ONE number: the objective to maximise. Everything else goes to stderr so
the metric pipeline stays clean.

The objective
-------------
**Lower bound of the 95% confidence interval on out-of-sample mean net return.**

Not the mean. The lower bound, and the choice is load-bearing:

* Maximising the *mean* is trivially gamed by filtering down to a handful of
  lucky trades. The lower bound falls as the sample shrinks, so over-filtering
  is self-punishing rather than rewarded.
* It is the quantity a decision actually depends on. "The mean is positive" is
  not a reason to trade; "the interval excludes zero" is.
* It rewards variance reduction as well as return, which is what a veto is
  supposed to deliver.

Protocol against overfitting
----------------------------
Three nested layers, and the loop can only ever see the outer two:

1. **Sealed holdout.** The final ``--holdout-frac`` of sessions is removed
   before anything else happens and is never touched by tuning. It is scored
   exactly once, by a separate script, at the very end.
2. **Walk-forward inside the tuning period.** Veto sets are selected on train
   days and scored on held-out days within each fold.
3. **Guards.** A configuration that survives by trading almost nothing, or by
   failing a leakage tripwire, is rejected outright.

Layer 1 is what makes the final number meaningful. Without it, tuning against
the walk-forward output would eventually overfit the walk-forward itself, and
there would be nothing left to check that against.

    python research/penny/scripts/tune_objective.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import conditional as cx  # noqa: E402
from research.penny.eval.folds import assert_folds_are_clean, walk_forward_folds  # noqa: E402
from research.penny.features import veto as veto_mod  # noqa: E402

logging.basicConfig(
    level=logging.INFO, stream=sys.stderr,
    format="%(levelname)-7s %(message)s",
)
logger = logging.getLogger("tune")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "tuning_params.json"
TRADES_PATH = bars.DATA_ROOT / "trades.parquet"

# A configuration that survives by trading almost nothing is not a strategy.
MIN_OOS_TRADES = 400
MIN_RETAINED_FRAC = 0.02

FAILURE_METRIC = -1.0


def load_params() -> Dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def split_holdout(trades: pd.DataFrame, holdout_frac: float):
    """Seal the final slice of sessions away from tuning entirely."""
    days = np.sort(trades["date"].unique())
    cut = int(len(days) * (1 - holdout_frac))
    tune_days, holdout_days = set(days[:cut]), set(days[cut:])
    return (
        trades[trades["date"].isin(tune_days)].copy(),
        trades[trades["date"].isin(holdout_days)].copy(),
    )


def evaluate(trades: pd.DataFrame, params: Dict) -> Dict:
    """Run the walk-forward and return the objective plus diagnostics."""
    # Rebuild vetoes with the current thresholds. Diagnostics are excluded:
    # they use future information and must never influence tuning.
    trades = veto_mod.build_all(trades, params.get("veto_params", {}),
                                include_diagnostics=False)
    # Inert vetoes would inflate the hypothesis count without adding
    # information, penalising the FDR correction for everything else.
    veto_cols, _ = cx.drop_inert_vetoes(trades, veto_mod.veto_names(include_diagnostics=False))
    if not veto_cols:
        return {"objective": FAILURE_METRIC, "reason": "no usable vetoes at these thresholds"}

    fold_cfg = params["folds"]
    search_cfg = params["search"]

    folds = walk_forward_folds(
        trades["date"].unique(),
        n_splits=fold_cfg["n_splits"],
        embargo_days=fold_cfg["embargo_days"],
        min_train_days=fold_cfg["min_train_days"],
    )
    assert_folds_are_clean(folds, embargo_days=fold_cfg["embargo_days"])

    summary, oos = cx.walk_forward_veto_selection(
        trades, veto_cols, folds,
        max_size=search_cfg["max_veto_set_size"],
        min_retained=search_cfg["min_retained"],
        min_trades=search_cfg["min_trades"],
    )

    if oos.empty or summary.empty:
        return {"objective": FAILURE_METRIC, "reason": "no out-of-sample trades"}

    result = cx.evaluate_population(oos, name="oos", n_boot=3_000)

    n_baseline = len(trades.dropna(subset=["net_return"]))
    retained = len(oos) / max(n_baseline, 1)

    guards = {
        "enough_trades": result.n_trades >= MIN_OOS_TRADES,
        "enough_retained": retained >= MIN_RETAINED_FRAC,
    }
    passed = all(guards.values())

    return {
        "objective": result.ci_lo if passed else FAILURE_METRIC,
        "guards": guards,
        "guards_passed": passed,
        "oos_mean": result.mean_net,
        "oos_ci_lo": result.ci_lo,
        "oos_ci_hi": result.ci_hi,
        "oos_n": result.n_trades,
        "oos_days": result.n_days,
        "oos_retained": retained,
        "oos_sharpe": result.sharpe,
        "oos_hit_rate": result.hit_rate,
        "mean_degradation": float(summary["degradation"].mean()),
        "folds": len(summary),
        "chosen": summary["chosen_vetoes"].tolist(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Autoresearch objective.")
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--report", action="store_true", help="Dump full diagnostics to stderr.")
    args = parser.parse_args(argv)

    if not TRADES_PATH.exists():
        print(FAILURE_METRIC)
        logger.error("trades.parquet missing -- run build_dataset first")
        return 1

    params = load_params()
    trades = pd.read_parquet(TRADES_PATH)
    tune, holdout = split_holdout(trades, args.holdout_frac)

    logger.info(
        "tuning on %s trades / %d sessions; %s trades / %d sessions SEALED",
        f"{len(tune):,}", tune["date"].nunique(),
        f"{len(holdout):,}", holdout["date"].nunique(),
    )

    try:
        result = evaluate(tune, params)
    except Exception as exc:  # noqa: BLE001
        print(FAILURE_METRIC)
        logger.error("evaluation failed: %s", exc)
        return 1

    if args.report or True:
        for key, value in result.items():
            if key != "chosen":
                logger.info("  %-18s %s", key, value)
        logger.info("  chosen per fold: %s", result.get("chosen"))

    # The metric, and nothing else, on stdout.
    print(f"{result['objective']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
