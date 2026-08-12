"""diagnose_inversion.py — why round 4's model is anti-predictive.

The round-4 precision ladder came out monotone in the wrong direction: at a
$2,000 order the top 0.5% of the GBDT's scores returned **-3.32% gross** against
-0.19% for the whole tradeable panel, and every step toward *more* confidence
made the trade worse.

The suspected mechanism is visible in the same table. Mean realised cost falls
monotonically with selectivity -- 66.6 bps for the panel, 37.6 at the top decile,
24.9 at the top 0.5%. Round 4 labelled each trade ``ret_c2c > cost_hat`` and
handed the model ``cost_hat`` as a feature. The cheapest way to make that label
true is to pick trades with a *low bar*, so the model may simply have learned to
minimise the hurdle rather than maximise the return.

Three diagnostics, each of which could falsify that story:

1. **Ablate the cause.** Refit with a fixed cost label and no cost features, on
   the identical funnel and folds. If the inversion vanishes, the design is the
   cause. If it survives, the population is.
2. **Is the information real but signed wrong?** Score the *bottom* k. A model
   with genuine inverted skill has a positive bottom tail; a model that merely
   sorts on cost has a flat one. Exploratory and labelled as such -- flipping a
   sign after seeing the result is not a strategy, it is a hypothesis.
3. **Measure the coupling directly.** Rank correlation between model score and
   ``cost_hat``, per fold.

    python research/penny/scripts/diagnose_inversion.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars, quotes as Q  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.eval.folds import walk_forward_folds  # noqa: E402
from research.penny.model import features as F, train as T  # noqa: E402
from research.penny.scripts.run_round4_study import (  # noqa: E402
    MAX_SPREAD_BPS,
    apply_tradeability,
    load_inputs,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("diagnose")

ORDER = 2_000
FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]
FIXED_LABEL_COST = 0.0437  # the round-2/3 constant, for the ablation arm


def ladder(scored: pd.DataFrame, label: str, bottom: bool = False) -> List[Dict]:
    """Gross by selectivity, from either end of the score distribution."""
    rows = []
    s = scored.copy()
    if bottom:
        s["model_score"] = -s["model_score"]
    for frac in FRACTIONS:
        sel = T.select_top_fraction(s, frac)
        if len(sel) < 30:
            continue
        est = pstats.cluster_bootstrap_mean(sel["ret_c2c"], sel["date"], n_boot=2_000)
        rows.append({
            "arm": label, "end": "bottom" if bottom else "top",
            "fraction": frac, "n": int(est.n), "n_days": int(est.n_days),
            "gross": float(est.mean), "gross_lo": float(est.lo),
            "gross_hi": float(est.hi),
            "cost_hat_bps": float(10_000 * sel["cost_hat"].mean()),
            "hit_rate": float((sel["ret_c2c"] > 0).mean()),
        })
    return rows


def main() -> int:
    df_all, windows = load_inputs("trades_r4.parquet", "quote_windows_r4.parquet")
    costed = Q.attach_costs(df_all, windows, order_notional=float(ORDER))
    df, _ = apply_tradeability(costed, float(ORDER), MAX_SPREAD_BPS)
    df = df.dropna(subset=["cost_hat", "ret_c2c"]).reset_index(drop=True)

    folds = walk_forward_folds(df["date"].unique(), n_splits=4,
                               embargo_days=10, min_train_days=900)
    logger.info("panel %s trades | panel gross %+.4f | mean cost_hat %.1f bps",
                f"{len(df):,}", df["ret_c2c"].mean(), 10_000 * df["cost_hat"].mean())

    out: Dict = {"panel_gross": float(df["ret_c2c"].mean()),
                 "panel_n": int(len(df)), "order_size": ORDER}
    rows: List[Dict] = []

    # --- arm 1: round-4 as run (per-trade cost label + cost features) -------
    df["meta_label"] = (df["ret_c2c"] > df["cost_hat"]).astype(int)
    cols_cost = F.feature_columns(df, include_cost=True)
    scored_cost, _ = T.walk_forward_scores(df, folds, cols_cost, "meta_label")
    rows += ladder(scored_cost, "r4_as_run")
    rows += ladder(scored_cost, "r4_as_run", bottom=True)

    # --- arm 2: ablate BOTH suspected causes --------------------------------
    df["meta_label_fixed"] = (df["ret_c2c"] > FIXED_LABEL_COST).astype(int)
    cols_nocost = F.feature_columns(df, include_cost=False)
    scored_plain, _ = T.walk_forward_scores(df, folds, cols_nocost, "meta_label_fixed")
    rows += ladder(scored_plain, "fixed_label_no_cost_features")

    # --- arm 3: separate the two causes -------------------------------------
    scored_lbl, _ = T.walk_forward_scores(df, folds, cols_nocost, "meta_label")
    rows += ladder(scored_lbl, "per_trade_label_only")

    scored_feat, _ = T.walk_forward_scores(df, folds, cols_cost, "meta_label_fixed")
    rows += ladder(scored_feat, "cost_features_only")

    table = pd.DataFrame(rows)
    out["ladder"] = table.to_dict(orient="records")

    # --- the coupling, measured ---------------------------------------------
    from scipy.stats import spearmanr

    coupling = []
    for name, sc in (("r4_as_run", scored_cost),
                     ("fixed_label_no_cost_features", scored_plain),
                     ("per_trade_label_only", scored_lbl),
                     ("cost_features_only", scored_feat)):
        for fold, part in sc.groupby("fold"):
            rho, _ = spearmanr(part["model_score"], part["cost_hat"])
            coupling.append({"arm": name, "fold": int(fold), "rho_score_vs_cost": float(rho)})
    coup = pd.DataFrame(coupling)
    out["coupling"] = coup.to_dict(orient="records")

    logger.info("=" * 78)
    logger.info("GROSS BY SELECTIVITY (top end unless marked bottom)\n%s",
                table.to_string(index=False, float_format="%.4f"))
    logger.info("-" * 78)
    logger.info("SPEARMAN(model score, cost_hat) by arm\n%s",
                coup.groupby("arm")["rho_score_vs_cost"].agg(["mean", "min", "max"])
                .to_string())
    logger.info("=" * 78)

    path = bars.DATA_ROOT / "diagnose_inversion.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
