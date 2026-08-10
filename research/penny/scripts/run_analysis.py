"""run_analysis.py — produce every result the notebook renders.

Writes a single ``analysis.pkl`` so the notebook does no heavy computation and
stays a presentation layer. That separation is deliberate: a notebook that
recomputes on every run invites silent divergence between what was tested and
what is shown.

Order of operations is the order of the argument:

1. null tests FIRST -- if the tripwires fail, nothing downstream is believable
2. baseline (no veto) -- the honest denominator
3. single-veto marginals -- diagnostic
4. exhaustive in-sample search, FDR-corrected -- exploratory, reported WITH the
   search size
5. walk-forward selection -- the only quotable out-of-sample number
6. sealed-holdout evaluation -- touched exactly once

    python research/penny/scripts/run_analysis.py
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import conditional as cx, nulls, stats as pstats  # noqa: E402
from research.penny.eval.folds import assert_folds_are_clean, walk_forward_folds  # noqa: E402
from research.penny.features import veto as veto_mod  # noqa: E402
from research.penny.scripts.tune_objective import load_params, split_holdout  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analysis")

OUT_PATH = bars.DATA_ROOT / "analysis.pkl"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the full analysis.")
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--skip-nulls", action="store_true")
    args = parser.parse_args(argv)

    trades_path = bars.DATA_ROOT / "trades.parquet"
    if not trades_path.exists():
        raise SystemExit("trades.parquet missing; run build_dataset first")

    params = load_params()
    trades = pd.read_parquet(trades_path)
    trades = veto_mod.build_all(trades, params["veto_params"], include_diagnostics=True)

    veto_cols = veto_mod.veto_names(include_diagnostics=False)
    tune, holdout = split_holdout(trades, args.holdout_frac)

    logger.info(
        "trades %s | tuning %s (%d sessions) | holdout %s (%d sessions)",
        f"{len(trades):,}", f"{len(tune):,}", tune["date"].nunique(),
        f"{len(holdout):,}", holdout["date"].nunique(),
    )

    results: Dict = {
        "params": params,
        "veto_registry": veto_mod.registry_frame(include_diagnostics=True),
        "n_trades": int(len(trades)),
        "date_min": str(trades["date"].min().date()),
        "date_max": str(trades["date"].max().date()),
        "holdout_frac": args.holdout_frac,
    }

    # --- 1. nulls first -----------------------------------------------------
    if not args.skip_nulls:
        logger.info("running null tests ...")
        results["nulls"] = nulls.run_all(tune, veto_cols)
        if not results["nulls"]["passed"].all():
            logger.warning("NULL TESTS FAILED -- downstream results are not believable")

    # --- 2. baseline --------------------------------------------------------
    logger.info("evaluating baseline ...")
    baseline = cx.evaluate_population(tune, name="(no veto)", n_boot=5_000)
    results["baseline"] = baseline
    logger.info(
        "baseline: E[net]=%+.4f  CI [%+.4f, %+.4f]  n=%s  Sharpe=%.2f",
        baseline.mean_net, baseline.ci_lo, baseline.ci_hi,
        f"{baseline.n_trades:,}", baseline.sharpe,
    )

    # --- 3. single-veto marginals ------------------------------------------
    logger.info("single-veto table ...")
    results["single"] = cx.single_veto_table(tune, veto_cols, n_boot=2_000)

    # --- 4. exhaustive in-sample search ------------------------------------
    search_cfg = params["search"]
    logger.info("exhaustive veto search (max size %d) ...", search_cfg["max_veto_set_size"])
    table = cx.search_veto_sets(
        tune, veto_cols,
        max_size=search_cfg["max_veto_set_size"],
        min_retained=search_cfg["min_retained"],
        min_trades=search_cfg["min_trades"],
        n_boot=1_500,
    )
    results["search"] = table
    n_hyp = int(table["n_hypotheses"].iloc[0]) if not table.empty else 0
    logger.info("searched %d veto sets; %d pass FDR", n_hyp, int(table["q_pass"].sum()) if not table.empty else 0)

    # Placebo matched to the best set's selectivity -- the fair baseline.
    if not table.empty:
        best = table.iloc[0]
        results["placebo"] = cx.random_veto_placebo(
            tune, retained_frac=float(best["retained_frac"]), n_draws=400
        )
        chosen = tuple(v for v in best["vetoes"].split(", ") if v and v != "(none)")
        results["best_in_sample"] = best
        results["best_vs_baseline"] = cx.compare_to_baseline(tune, chosen)

    # --- 5. walk-forward: the only quotable OOS number ---------------------
    fold_cfg = params["folds"]
    logger.info("walk-forward selection ...")
    folds = walk_forward_folds(
        tune["date"].unique(),
        n_splits=fold_cfg["n_splits"],
        embargo_days=fold_cfg["embargo_days"],
        min_train_days=fold_cfg["min_train_days"],
    )
    assert_folds_are_clean(folds, embargo_days=fold_cfg["embargo_days"])
    results["folds"] = [f.describe() for f in folds]

    summary, oos = cx.walk_forward_veto_selection(
        tune, veto_cols, folds,
        max_size=search_cfg["max_veto_set_size"],
        min_retained=search_cfg["min_retained"],
        min_trades=search_cfg["min_trades"],
    )
    results["wf_summary"] = summary
    results["oos_trades"] = oos

    if not oos.empty:
        oos_result = cx.evaluate_population(oos, name="walk-forward OOS", n_boot=5_000)
        results["oos"] = oos_result
        curve = pstats.daily_equity_curve(oos)
        results["oos_curve"] = curve
        results["oos_dsr"] = pstats.deflated_sharpe_ratio(curve["ret"], n_trials=max(n_hyp, 1))
        logger.info(
            "OOS: E[net]=%+.4f  CI [%+.4f, %+.4f]  n=%s  Sharpe=%.2f  DSR=%.3f",
            oos_result.mean_net, oos_result.ci_lo, oos_result.ci_hi,
            f"{oos_result.n_trades:,}", oos_result.sharpe, results["oos_dsr"],
        )

    # --- 6. the sealed holdout, opened once --------------------------------
    if not holdout.empty and not summary.empty:
        modal = summary["chosen_vetoes"].mode()
        chosen = tuple(
            v.strip() for v in (modal.iloc[0] if len(modal) else "").split(",")
            if v.strip() and v.strip() != "(none)"
        )
        results["holdout_vetoes"] = chosen
        kept = holdout[cx.apply_vetoes(holdout, chosen)]
        results["holdout"] = cx.evaluate_population(
            kept, name="sealed holdout", vetoes=chosen, n_boot=5_000,
            baseline_n=len(holdout.dropna(subset=["net_return"])),
        )
        results["holdout_baseline"] = cx.evaluate_population(
            holdout, name="holdout baseline", n_boot=5_000
        )
        logger.info(
            "HOLDOUT (%s): E[net]=%+.4f  CI [%+.4f, %+.4f]  n=%s",
            ", ".join(chosen) or "(none)",
            results["holdout"].mean_net, results["holdout"].ci_lo,
            results["holdout"].ci_hi, f"{results['holdout'].n_trades:,}",
        )

    # Curves for the figures.
    results["baseline_curve"] = pstats.daily_equity_curve(tune)
    results["trades_sample"] = trades

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        pickle.dump(results, fh)
    logger.info("wrote %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
