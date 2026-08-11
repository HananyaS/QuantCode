"""run_model_study.py — the round-2 experiment.

Tests the pre-registered hypothesis:

    H1. A model score over the full feature set selects a sub-population whose
        walk-forward out-of-sample net return is positive, with the
        day-clustered 95% CI excluding zero, at >= 5% of candidates retained
        and >= 500 out-of-sample trades.

Every number produced here is out-of-sample by construction: the model is fitted
inside each fold and scores only that fold's held-out days.

The cost bar is *assumed*, not measured (2 x (118 bps half-spread + 100 bps
impact)). Measuring it was out of scope for this round, so results are reported
at 200 / 300 / 437 bps and the verdict states which of them it depends on.

    python research/penny/scripts/run_model_study.py
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars, edgar as edgar_mod  # noqa: E402
from research.penny.eval import conditional as cx, stats as pstats  # noqa: E402
from research.penny.eval.folds import assert_folds_are_clean, walk_forward_folds  # noqa: E402
from research.penny.labels import barrier_free as bf  # noqa: E402
from research.penny.model import features as F, train as T  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("study")

OUT_PATH = bars.DATA_ROOT / "analysis_model.pkl"

# The assumed bar, plus two more optimistic ones so the verdict's dependence on
# an unmeasured number is visible rather than buried.
COST_LEVELS = [0.0200, 0.0300, 0.0437]
TOP_FRACTIONS = [0.05, 0.10, 0.20, 0.30]

MIN_OOS_TRADES = 500
MIN_RETAINED = 0.05


def load_inputs(use_edgar: bool = True) -> pd.DataFrame:
    """Join trades, barrier-free outcomes, and EDGAR features."""
    trades = pd.read_parquet(bars.DATA_ROOT / "trades.parquet")

    path_stats = pd.read_parquet(bars.DATA_ROOT / "barrier_free.parquet")
    keep = ["symbol", "date", "ret_c2c", "mfe_full", "mae_full",
            "minutes_to_mfe", "mfe_before_mae", "n_scan_minutes"]
    df = trades.merge(path_stats[keep], on=["symbol", "date"], how="inner",
                      validate="one_to_one")

    # The barrier-free MFE is a superset of the truncated one by construction.
    bf.assert_dominates_truncated(df["mfe_full"], df["max_favourable"])

    edgar_df = None
    edgar_path = bars.DATA_ROOT / "edgar_features.parquet"
    if use_edgar and edgar_path.exists():
        edgar_df = pd.read_parquet(edgar_path)
        logger.info("edgar features: %s rows", f"{len(edgar_df):,}")
    elif use_edgar:
        logger.warning("edgar_features.parquet missing -- running without filing features")

    return F.build_matrix(df, edgar_df, include_edgar=edgar_df is not None)


def evaluate_configuration(
    scored: pd.DataFrame, fraction: float, cost: float
) -> Dict:
    """Performance of the top-scoring `fraction` at a given cost bar."""
    selected = T.select_top_fraction(scored, fraction)
    if selected.empty:
        return {"n": 0}

    selected = selected.copy()
    selected["net_at_cost"] = selected["ret_c2c"] - cost

    result = cx.evaluate_population(
        selected, name=f"top {fraction:.0%} @ {cost:.2%}",
        return_col="net_at_cost", n_boot=3_000,
        baseline_n=len(scored),
    )
    gross = cx.evaluate_population(
        selected, name="gross", return_col="ret_c2c", n_boot=1_000,
        baseline_n=len(scored),
    )

    # Does the ranker beat taking everything? Paired and day-clustered, so the
    # common market factor cancels. This is the only defensible evidence that
    # the score carries information -- the decile-monotonicity statistic is
    # far too noisy to serve (a shuffled-label null on this data produced mean
    # |rho| = 0.515 against a "real" 0.842).
    keys = set(zip(selected["symbol"], selected["date"]))
    mask = np.array(
        [(s, d) in keys for s, d in zip(scored["symbol"], scored["date"])]
    )
    lift = cx.compare_selection_to_baseline(
        scored, mask, return_col="ret_c2c", n_boot=3_000
    )

    return {
        "lift_gross": lift["delta"],
        "lift_lo": lift["lo"],
        "lift_p": lift["p_value"],
        "lift_helps": lift["helps"],
        "fraction": fraction,
        "cost": cost,
        "n": result.n_trades,
        "n_days": result.n_days,
        "retained": len(selected) / max(len(scored), 1),
        "mean_gross": gross.mean_net,
        "gross_ci_lo": gross.ci_lo,
        "mean_net": result.mean_net,
        "net_ci_lo": result.ci_lo,
        "net_ci_hi": result.ci_hi,
        "p_greater": result.p_greater,
        "hit_rate": result.hit_rate,
        "sharpe": result.sharpe,
        "meets_h1": bool(
            result.ci_lo > 0
            and result.n_trades >= MIN_OOS_TRADES
            and (len(selected) / max(len(scored), 1)) >= MIN_RETAINED
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Round-2 model study.")
    parser.add_argument("--no-edgar", action="store_true")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--min-train-days", type=int, default=250)
    args = parser.parse_args(argv)

    df = load_inputs(use_edgar=not args.no_edgar)
    feature_cols = F.feature_columns(df, include_edgar=not args.no_edgar)
    logger.info("trades %s | features %d", f"{len(df):,}", len(feature_cols))

    folds = walk_forward_folds(
        df["date"].unique(), n_splits=args.n_splits,
        embargo_days=args.embargo_days, min_train_days=args.min_train_days,
    )
    assert_folds_are_clean(folds, embargo_days=args.embargo_days)
    for f in folds:
        logger.info(f.describe())

    results: Dict = {
        "n_trades": int(len(df)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "folds": [f.describe() for f in folds],
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "cost_levels": COST_LEVELS,
        "top_fractions": TOP_FRACTIONS,
        "use_edgar": not args.no_edgar,
    }

    # --- baseline: the whole candidate population, no model ---------------
    for cost in COST_LEVELS:
        base = df.assign(net_at_cost=df["ret_c2c"] - cost)
        r = cx.evaluate_population(base, return_col="net_at_cost", n_boot=2_000)
        results.setdefault("baseline", {})[f"{cost:.4f}"] = {
            "mean_net": r.mean_net, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "n": r.n_trades,
        }
    logger.info(
        "baseline gross %.4f | net @437bps %.4f",
        df["ret_c2c"].mean(), results["baseline"]["0.0437"]["mean_net"],
    )

    # --- the model, one label per cost bar --------------------------------
    grid: List[Dict] = []
    scored_by_cost: Dict[str, pd.DataFrame] = {}
    importances: Dict[str, pd.DataFrame] = {}

    for cost in COST_LEVELS:
        label_col = "meta_label"
        df[label_col] = bf.meta_label(df["ret_c2c"], cost)
        positive_rate = float(df[label_col].mean())
        logger.info("cost %.2f%%: positive class %.1f%%", 100 * cost, 100 * positive_rate)

        scored, imp = T.walk_forward_scores(df, folds, feature_cols, label_col)
        if scored.empty:
            logger.warning("no scores produced at cost %.4f", cost)
            continue

        scored_by_cost[f"{cost:.4f}"] = scored
        importances[f"{cost:.4f}"] = imp

        table = T.score_decile_table(scored)
        rho = T.spearman_monotonicity(table)
        logger.info("cost %.2f%%: score-decile monotonicity rho = %+.3f", 100 * cost, rho)
        results.setdefault("decile_tables", {})[f"{cost:.4f}"] = table
        results.setdefault("monotonicity", {})[f"{cost:.4f}"] = rho

        for fraction in TOP_FRACTIONS:
            row = evaluate_configuration(scored, fraction, cost)
            if row.get("n", 0) == 0:
                continue
            grid.append(row)
            logger.info(
                "  top %4.0f%% @ %5.2f%%: n=%5d  gross %+.4f [%+.4f, ..]  "
                "lift %+.4f (p=%.3f)  net %+.4f  H1=%s",
                100 * fraction, 100 * cost, row["n"], row["mean_gross"],
                row["gross_ci_lo"], row["lift_gross"], row["lift_p"],
                row["mean_net"], row["meets_h1"],
            )

    grid_df = pd.DataFrame(grid)
    results["grid"] = grid_df
    results["importances"] = importances
    results["scored"] = scored_by_cost.get("0.0437", pd.DataFrame())

    # --- multiple testing over the whole grid -----------------------------
    if not grid_df.empty:
        n_trials = len(grid_df)
        grid_df["q_pass"] = pstats.benjamini_hochberg(
            grid_df["p_greater"].fillna(1.0).to_numpy(), q=0.05
        )
        results["n_trials"] = n_trials
        best = grid_df.sort_values("net_ci_lo", ascending=False).iloc[0]
        results["best"] = best.to_dict()

        curve = pstats.daily_equity_curve(
            T.select_top_fraction(scored_by_cost["0.0437"], float(best["fraction"]))
            .assign(net_at_cost=lambda d: d["ret_c2c"] - float(best["cost"])),
            return_col="net_at_cost",
        )
        results["best_curve"] = curve
        results["best_dsr"] = pstats.deflated_sharpe_ratio(curve["ret"], n_trials=n_trials)

    # --- regime breakdown --------------------------------------------------
    # The population is not stationary: candidate rate runs ~3.5/session before
    # 2020 and ~15.5 in 2025-26. Pooling those into one number can hide an edge
    # that exists in one regime, or manufacture one that exists in none.
    scored_437 = scored_by_cost.get("0.0437")
    if scored_437 is not None and not scored_437.empty:
        regime_rows = []
        rc = scored_437["regime_n_cand_60d"]
        if rc.notna().sum() > 100:
            terciles = pd.qcut(rc.rank(method="first"), 3, labels=["quiet", "normal", "busy"])
            scored_437 = scored_437.assign(regime=terciles)
            for name, group in scored_437.groupby("regime", observed=True):
                sel = T.select_top_fraction(group, 0.10)
                if len(sel) < 50:
                    continue
                all_g = pstats.cluster_bootstrap_mean(
                    group["ret_c2c"], group["date"], n_boot=1_500
                )
                sel_g = pstats.cluster_bootstrap_mean(
                    sel["ret_c2c"], sel["date"], n_boot=1_500
                )
                regime_rows.append(
                    {
                        "regime": str(name),
                        "n_all": all_g.n,
                        "gross_all": all_g.mean,
                        "n_top10": sel_g.n,
                        "gross_top10": sel_g.mean,
                        "top10_ci_lo": sel_g.lo,
                        "top10_ci_hi": sel_g.hi,
                        "lift": sel_g.mean - all_g.mean,
                    }
                )
        by_year = (
            scored_437.assign(year=scored_437["date"].dt.year)
            .groupby("year")
            .agg(n=("ret_c2c", "size"), gross=("ret_c2c", "mean"))
            .reset_index()
        )
        results["regime"] = pd.DataFrame(regime_rows)
        results["by_year"] = by_year
        if regime_rows:
            logger.info("regime breakdown (top 10% selection):")
            for r in regime_rows:
                logger.info(
                    "  %-7s all %+.4f (n=%5d) | top10 %+.4f [%+.4f, %+.4f] | lift %+.4f",
                    r["regime"], r["gross_all"], r["n_all"], r["gross_top10"],
                    r["top10_ci_lo"], r["top10_ci_hi"], r["lift"],
                )

    # --- the pre-registered verdict ---------------------------------------
    meets = grid_df["meets_h1"].any() if not grid_df.empty else False
    at_437 = (
        grid_df[(grid_df["cost"] == 0.0437) & grid_df["meets_h1"]]
        if not grid_df.empty else pd.DataFrame()
    )
    results["h1_met_any_cost"] = bool(meets)
    results["h1_met_at_assumed_cost"] = bool(len(at_437) > 0)

    logger.info("=" * 62)
    logger.info("H1 met at the assumed 437 bps : %s", results["h1_met_at_assumed_cost"])
    logger.info("H1 met at ANY tested cost bar : %s", results["h1_met_any_cost"])
    if not grid_df.empty:
        logger.info("trials in the grid            : %d", results["n_trials"])
        logger.info("best net CI lower bound       : %+.4f", grid_df["net_ci_lo"].max())
        logger.info("best gross                    : %+.4f", grid_df["mean_gross"].max())
    logger.info("=" * 62)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        pickle.dump(results, fh)
    logger.info("wrote %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
