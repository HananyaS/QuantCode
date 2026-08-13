"""portfolio_study.py — what an account would actually have earned.

Six rounds reported the mean return **per trade**. An account earns a return
**per day**, equal-weighted across whatever it held that session, and flat on the
days it sat out. The two coincide only if candidate count is uncorrelated with
candidate quality — which in a reversal strategy is exactly the assumption worth
checking, since candidates spike on turbulent sessions.

This re-evaluates the round-6 arms as portfolios: daily series, compounded
equity, Sharpe, drawdown and the losing-streak profile, under a realistic cap on
simultaneous positions.

**Exploratory, and it does not change any verdict.** Diversification reduces the
variance of a daily series, not its mean. Round 6's H1‴ result stands. This asks
a different question — whether a mean near zero is survivable in practice, and
whether per-trade averaging has been misreporting it.

    python research/penny/scripts/portfolio_study.py
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

from research.penny.etl import bars, quotes as Q  # noqa: E402
from research.penny.eval import portfolio as PF, stats as pstats  # noqa: E402
from research.penny.eval.folds import walk_forward_folds  # noqa: E402
from research.penny.model import cascade as CAS, train as T  # noqa: E402
from research.penny.scripts.run_round6_study import (  # noqa: E402
    CAPPED_SIZES, LABEL_COST, MAX_SPREAD_BPS, MIN_TRAIN_ROWS, TOP_FRACTIONS,
    apply_tradeability, feature_sets, load_inputs,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("portfolio")

# The sizes where round 6 left net at or above zero.
SIZES = [250, 500]
MAX_POSITIONS = [5, 10, 20, None]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Portfolio view of round-6 arms.")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--min-train-days", type=int, default=700)
    parser.add_argument("--out", default=str(bars.DATA_ROOT / "portfolio_study.json"))
    args = parser.parse_args(argv)

    df_all, windows = load_inputs()
    rows: List[Dict] = []
    weighting: List[Dict] = []
    curves: Dict[str, pd.Series] = {}

    for order in SIZES:
        costed = Q.attach_costs(df_all, windows, float(order))
        stages_cfg = feature_sets(costed)
        df, _ = apply_tradeability(costed, float(order))
        df = df.dropna(subset=["ret_c2c", "cost_realized"]).reset_index(drop=True)
        df["meta_label"] = (df["ret_c2c"] > LABEL_COST).astype(int)
        df["net_return"] = df["ret_c2c"] - df["cost_realized"]

        # The trading calendar the strategy could have acted on: every session
        # in the tradeable panel, so days it sat out are flat rather than absent.
        calendar = np.sort(df["date"].unique())

        folds = walk_forward_folds(df["date"].unique(), n_splits=args.n_splits,
                                   embargo_days=args.embargo_days,
                                   min_train_days=args.min_train_days)
        stages = [
            CAS.Stage("context", stages_cfg["context"], 0.50, "rank"),
            CAS.Stage("fundamentals", stages_cfg["fundamentals"], 0.40, "binary"),
            CAS.Stage("execution", stages_cfg["execution"], 0.25, "quantile"),
        ]

        selections: Dict[str, pd.DataFrame] = {}

        survivors, _ = CAS.run_cascade(df, folds, stages, "meta_label", "ret_c2c")
        if not survivors.empty:
            selections["cascade"] = survivors

        if order in CAPPED_SIZES and "cost_hat_bps" in df.columns:
            cheap = df[df["cost_hat_bps"] <= df["cost_hat_bps"].median()]
            cheap = cheap.reset_index(drop=True)
            if len(cheap) >= MIN_TRAIN_ROWS * 2:
                cf = walk_forward_folds(cheap["date"].unique(),
                                        n_splits=args.n_splits,
                                        embargo_days=args.embargo_days,
                                        min_train_days=args.min_train_days)
                s2, _ = CAS.run_cascade(cheap, cf, stages, "meta_label", "ret_c2c")
                if not s2.empty:
                    selections["cascade_capped"] = s2

        scored, _ = T.walk_forward_scores(df, folds, stages_cfg["execution"],
                                          "meta_label")
        if not scored.empty:
            for frac in (0.01, 0.02, 0.05):
                selections[f"flat_{frac:g}"] = T.select_top_fraction(scored, frac)

        # The honest comparator: hold every tradeable candidate, no model.
        selections["all_tradeable"] = df

        for name, sel in selections.items():
            if "net_return" not in sel.columns:
                sel = sel.assign(net_return=sel["ret_c2c"] - sel["cost_realized"])

            w = PF.compare_weightings(sel, "net_return")
            if w:
                weighting.append({"arm": name, "order": order, **w})

            for cap in MAX_POSITIONS:
                res = PF.evaluate_portfolio(
                    sel, "net_return", max_positions=cap,
                    rank_col="model_score" if "model_score" in sel.columns else None,
                    all_sessions=calendar,
                )
                if res is None:
                    continue
                est = pstats.cluster_bootstrap_mean(res.daily, pd.Series(
                    np.arange(len(res.daily))), n_boot=3_000)
                row = {"arm": name, "order": order,
                       "max_positions": cap if cap else 0, **res.to_dict(),
                       "daily_lo": float(est.lo), "daily_hi": float(est.hi),
                       "daily_p": float(est.p_value_greater)}
                rows.append(row)
                if cap == 10:
                    curves[f"{name}@${order}"] = res.equity

    table = pd.DataFrame(rows)
    wtable = pd.DataFrame(weighting)

    logger.info("=" * 100)
    logger.info("PER-TRADE vs PER-DAY WEIGHTING\n%s",
                wtable.to_string(index=False, float_format="%.5f"))
    logger.info("-" * 100)
    show = ["arm", "order", "max_positions", "n_days", "n_trades", "avg_positions",
            "pct_days_active", "mean_daily", "daily_lo", "sharpe", "total_return",
            "max_drawdown", "hit_rate_days", "max_consecutive_losing_days"]
    logger.info("PORTFOLIO VIEW\n%s",
                table[[c for c in show if c in table.columns]].to_string(
                    index=False, float_format="%.4f"))
    logger.info("=" * 100)

    winners = table[(table["daily_lo"] > 0) & (table["mean_daily"] > 0)]
    if winners.empty:
        logger.info("NO configuration has a daily mean with a CI excluding zero.")
    else:
        logger.info("Daily mean above zero with CI excluding zero:\n%s",
                    winners.to_string(index=False, float_format="%.4f"))

    Path(args.out).write_text(json.dumps(
        {"portfolio": rows, "weighting": weighting, "exploratory": True},
        indent=2, default=str), encoding="utf-8")
    with open(bars.DATA_ROOT / "portfolio_curves.pkl", "wb") as fh:
        pickle.dump(curves, fh)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
