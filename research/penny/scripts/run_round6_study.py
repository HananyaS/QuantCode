"""run_round6_study.py — the cascade on the reversal pool, at retail size.

Separate from rounds 3-5 so their results stay reproducible.

The question
-----------
Round 5 established the pool: liquid low-priced names gapping down 3-10% return
**+0.31% gross at 09:35** (n=51,361 over 2,575 days, p=0.0295). Round 5 stopped
there, because unconditionally that does not cover cost.

Round 6 asks whether a **cascade** can concentrate it above cost. That is a
different question from rounds 1-4, which hunted subsets of populations whose
gross was *negative* -- searching for a pocket where the sign flips. Here the
sign is already right and the magnitude is the problem, which is the ordinary
situation a model is for.

Two findings from the partial cost measurement shape what follows, and both are
unfavourable (PREREGISTRATION_R6.md, Amendment 1):

* **Cost-aware selection buys nothing.** Gross is flat across cost quintiles
  while cost rises 35x. There is no cheap corner where the edge also lives.
* **The edge sits in names too thin to buy.** Requiring a $2,000 book cuts gross
  from +0.244% to +0.034%.

So order sizes below $2,000 are tested for the first time. Depth shortfall is
``(order/inside - 1)+ x half_spread`` and scales with size, so at $500 most of
this pool fits inside the quote and the depth term vanishes by construction. The
open question is only whether **gross survives the tradeability filter** at $500
the way it fails at $2,000.

    python research/penny/scripts/run_round6_study.py
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars, filings_flow as FF, quotes as Q  # noqa: E402
from research.penny.eval import (  # noqa: E402
    conditional as cx,
    precision as prec,
    stats as pstats,
)
from research.penny.eval.folds import assert_folds_are_clean, walk_forward_folds  # noqa: E402
from research.penny.features import market_regime as MR  # noqa: E402
from research.penny.model import cascade as CAS, features as F, train as T  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("round6")

OUT_PATH = bars.DATA_ROOT / "analysis_round6.pkl"

# ---- Constants, fixed in PREREGISTRATION_R6.md (af4110d) + Amendment 1 -------
ORDER_SIZES = [250, 500, 1_000, 2_000, 5_000, 10_000]
TOP_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]

MIN_OOS_TRADES = 300
MIN_OOS_DAYS = 150
MIN_TRAIN_ROWS = 2_000
MAX_SPREAD_BPS = 200.0
GATE_C_MAX_BPS = 150.0
GATE_C_SIZE = 2_000          # the gate refers to $2k, so it cannot be passed
                             # by shrinking the order until it clears its bar
FDR_Q = 0.05
PRIOR_TRIALS = 94

# Never per-trade. Round 4 labelled `ret > cost_hat`, the model learned to
# minimise the hurdle, and the precision ladder inverted to -3.32% at the top
# 0.5%.
LABEL_COST = 0.0437

CONFIRMATORY_ARMS = {"cascade", "flat_gbdt"}
LEVELS_PENALTIES = [1.0, 1.5, 2.0]


def _load(name: str) -> Optional[pd.DataFrame]:
    path = bars.DATA_ROOT / name
    if not path.exists():
        logger.warning("%s missing", name)
        return None
    return pd.read_parquet(path)


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trades, path stats, every feature family, regime, and quote windows."""
    trades = pd.read_parquet(bars.DATA_ROOT / "trades_rev.parquet")
    paths = pd.read_parquet(bars.DATA_ROOT / "barrier_free_rev.parquet")
    keep = ["symbol", "date", "ret_c2c", "mfe_full", "mae_full", "mfe_before_mae"]
    df = trades.merge(paths[keep], on=["symbol", "date"], how="inner",
                      validate="one_to_one")

    df = F.build_matrix(
        df,
        edgar=_load("edgar_features_rev.parquet"),
        premarket=_load("premarket_features_rev.parquet"),
        short_volume=_load("finra_features_rev.parquet"),
        float_data=_load("xbrl_features_rev.parquet"),
    )

    flow = _load("flow_features_rev.parquet")
    if flow is not None:
        before = len(df)
        overlap = (set(flow.columns) & set(df.columns)) - {"symbol", "date"}
        df = df.merge(flow.drop(columns=list(overlap)), on=["symbol", "date"],
                      how="left", validate="one_to_one")
        assert len(df) == before, "flow merge changed the row count"

    # Regime from the studied cross-section. The factor must be built on the
    # UNIVERSE, not the candidate set -- defining the market by the selection
    # under test would make it circular.
    cands = pd.read_parquet(bars.DATA_ROOT / "candidates_rev.parquet",
                            columns=["symbol", "date", "daily_ret", "gap"])
    market = MR.build_market_factor(cands)
    MR.assert_regime_is_causal(market, cands)
    df = MR.attach_regime(df, market)

    windows = pd.read_parquet(bars.DATA_ROOT / "quote_windows_rev.parquet")
    logger.info("trades %s | quote windows %s | features %d cols",
                f"{len(df):,}", f"{len(windows):,}", df.shape[1])
    return df, windows


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Per-stage feature lists, and the union the flat comparator sees."""
    present = lambda cols: [c for c in cols if c in df.columns]  # noqa: E731

    context = present(F.BASE_FEATURES) + present(MR.REGIME_FEATURES)
    context += [c for c in df.columns if c.startswith(("xs_rank_", "regime_"))]

    fundamentals = context + present(F.EDGAR_FEATURES) + present(FF.FLOW_FEATURES)
    fundamentals += present(F.FLOAT_FEATURES) + present(F.SHORT_FEATURES)

    from research.penny.features.premarket import PREMARKET_FEATURES

    execution = fundamentals + present(PREMARKET_FEATURES) + present(F.COST_FEATURES)

    dedup = lambda cols: list(dict.fromkeys(cols))  # noqa: E731
    sets = {"context": dedup(context), "fundamentals": dedup(fundamentals),
            "execution": dedup(execution)}

    banned = F._forbidden_columns(df)
    for name, cols in sets.items():
        leaked = sorted(set(cols) & banned)
        assert not leaked, f"stage {name}: forbidden columns reached features: {leaked}"
        logger.info("stage %-13s %3d features", name, len(cols))
    return sets


def apply_tradeability(df: pd.DataFrame, order: float) -> Tuple[pd.DataFrame, Dict]:
    """Drop what cannot be traded at this size, and report what was dropped."""
    quoted = pd.DataFrame({
        "half_spread_bps": df["half_spread_bps_decision"],
        "ask_notional": df["ask_notional_decision"],
    }, index=df.index)
    mask = Q.tradeable_mask(quoted, order, MAX_SPREAD_BPS)
    removed = df[~mask]
    report = {
        "order": float(order),
        "n_before": int(len(df)), "n_after": int(mask.sum()),
        "share_kept": float(mask.mean()),
        "kept_gross": float(df.loc[mask, "ret_c2c"].mean()) if mask.any() else np.nan,
        "removed_gross": float(removed["ret_c2c"].mean()) if len(removed) else np.nan,
        "cost_median_bps": float(10_000 * df.loc[mask, "cost_realized"].median())
        if mask.any() else np.nan,
        "cost_mean_bps": float(10_000 * df.loc[mask, "cost_realized"].mean())
        if mask.any() else np.nan,
    }
    logger.info("$%-6s tradeable %6s/%6s (%4.1f%%) | kept gross %+.4f vs removed "
                "%+.4f | cost med %5.1f mean %6.1f bps",
                f"{order:,.0f}", f"{report['n_after']:,}", f"{report['n_before']:,}",
                100 * report["share_kept"], report["kept_gross"],
                report["removed_gross"], report["cost_median_bps"],
                report["cost_mean_bps"])
    return df[mask].reset_index(drop=True), report


def evaluate(scored: pd.DataFrame, mask: np.ndarray, arm: str,
             fraction: float, order: int) -> Dict:
    """Gross, net and precision KPIs for one selection."""
    selected = scored[mask]
    if len(selected) < 30:
        return {}

    gross = pstats.cluster_bootstrap_mean(selected["ret_c2c"], selected["date"],
                                          n_boot=3_000)
    lift = cx.compare_selection_to_baseline(scored, np.asarray(mask, dtype=bool),
                                            return_col="ret_c2c", n_boot=2_000)
    net = selected["ret_c2c"] - selected["cost_realized"]
    ok = net.notna()

    row = {
        "arm": arm, "fraction": fraction, "order_size": order,
        "n": int(gross.n), "n_days": int(gross.n_days),
        "retained": float(len(selected) / max(len(scored), 1)),
        "gross": float(gross.mean), "gross_lo": float(gross.lo),
        "gross_hi": float(gross.hi),
        "lift": float(lift["delta"]), "lift_p": float(lift["p_value"]),
        "cost_mean_bps": float(10_000 * selected["cost_realized"].mean()),
    }
    if ok.sum() >= 30:
        est = pstats.cluster_bootstrap_mean(net[ok], selected.loc[ok, "date"],
                                            n_boot=3_000)
        row.update({
            "net": float(est.mean), "net_lo": float(est.lo), "net_hi": float(est.hi),
            "net_p": float(est.p_value_greater), "net_n": int(est.n),
            "net_days": int(est.n_days),
            "eligible_h1": bool(est.n >= MIN_OOS_TRADES and est.n_days >= MIN_OOS_DAYS),
        })
        kpi = prec.precision_kpis(selected.loc[ok].assign(net_return=net[ok]),
                                  "net_return")
        row.update({f"kpi_{k}": v for k, v in kpi.items()})
    else:
        row["eligible_h1"] = False
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Round-6 cascade study.")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--min-train-days", type=int, default=700)
    parser.add_argument("--skip-nulls", action="store_true")
    args = parser.parse_args(argv)

    df_all, windows = load_inputs()
    results: Dict = {
        "preregistration": "af4110d + amendment 1",
        "order_sizes": ORDER_SIZES, "prior_trials": PRIOR_TRIALS,
        "label_cost": LABEL_COST,
        "n_trades_raw": int(len(df_all)),
        "date_min": str(df_all["date"].min().date()),
        "date_max": str(df_all["date"].max().date()),
    }

    # --- Gate C: the pool's own cost bar, at $2,000 ------------------------
    costed_gate = Q.attach_costs(df_all, windows, float(GATE_C_SIZE))
    rt = costed_gate["cost_realized_bps"].dropna()
    gate_c = {
        "n": int(len(rt)), "median_bps": float(rt.median()),
        "mean_bps": float(rt.mean()), "p25_bps": float(rt.quantile(0.25)),
        "p75_bps": float(rt.quantile(0.75)),
        "share_below_150": float((rt < 150).mean()),
        "share_below_62": float((rt < 62).mean()),
        "threshold_bps": GATE_C_MAX_BPS,
        "pass": bool(rt.median() < GATE_C_MAX_BPS),
    }
    results["gate_c"] = gate_c
    logger.info("=" * 78)
    logger.info("GATE C @ $%s: median %.1f bps (mean %.1f, p25 %.1f, p75 %.1f) n=%s",
                f"{GATE_C_SIZE:,}", gate_c["median_bps"], gate_c["mean_bps"],
                gate_c["p25_bps"], gate_c["p75_bps"], f"{gate_c['n']:,}")
    logger.info("           %.1f%% below 150 bps | %.1f%% below 62 bps -> %s",
                100 * gate_c["share_below_150"], 100 * gate_c["share_below_62"],
                "PASS" if gate_c["pass"] else "FAIL")
    logger.info("=" * 78)
    if not gate_c["pass"]:
        logger.error("Gate C failed; no cascade is fitted, as pre-registered.")
        results["h1_met"] = False
        _save(results)
        return 0

    stages_cfg = feature_sets(df_all)
    grid: List[Dict] = []
    tradeability: Dict[int, Dict] = {}

    for order in ORDER_SIZES:
        logger.info("-" * 78)
        costed = Q.attach_costs(df_all, windows, float(order))
        df, report = apply_tradeability(costed, float(order))
        tradeability[order] = report
        if len(df) < MIN_TRAIN_ROWS * 2:
            logger.warning("$%s: only %d tradeable rows -- skipping", order, len(df))
            continue

        df = df.dropna(subset=["ret_c2c", "cost_realized"]).reset_index(drop=True)
        df["meta_label"] = (df["ret_c2c"] > LABEL_COST).astype(int)

        folds = walk_forward_folds(df["date"].unique(), n_splits=args.n_splits,
                                   embargo_days=args.embargo_days,
                                   min_train_days=args.min_train_days)
        assert_folds_are_clean(folds, embargo_days=args.embargo_days)

        # Filter-only baseline: every tradeable candidate, no model. Reported so
        # the model cannot take credit for the screen.
        base_net = (df["ret_c2c"] - df["cost_realized"]).dropna()
        if len(base_net) >= 30:
            b = pstats.cluster_bootstrap_mean(base_net, df.loc[base_net.index, "date"],
                                              n_boot=3_000)
            results.setdefault("filter_only_baseline", {})[order] = {
                "n": int(b.n), "gross": float(df["ret_c2c"].mean()),
                "net": float(b.mean), "net_lo": float(b.lo), "net_hi": float(b.hi),
            }
            logger.info("  filter-only baseline: gross %+.4f  net %+.4f  CI [%+.4f, %+.4f]",
                        df["ret_c2c"].mean(), b.mean, b.lo, b.hi)

        # --- arm 1: the cascade --------------------------------------------
        stages = [
            CAS.Stage("context", stages_cfg["context"], 0.50, "rank"),
            CAS.Stage("fundamentals", stages_cfg["fundamentals"], 0.40, "binary"),
            CAS.Stage("execution", stages_cfg["execution"], 0.25, "quantile"),
        ]
        survivors, trace = CAS.run_cascade(df, folds, stages, "meta_label", "ret_c2c")
        results.setdefault("cascade_trace", {})[order] = CAS.trace_table(trace).to_dict(
            orient="records")
        if not survivors.empty:
            keys = set(zip(survivors["symbol"], survivors["date"]))
            # Evaluate against the full tradeable panel so `lift` measures the
            # cascade against what it selected FROM, not against itself.
            mask = np.array([(s, d) in keys for s, d in zip(df["symbol"], df["date"])])
            row = evaluate(df.assign(cost_realized=df["cost_realized"]), mask,
                           "cascade", float("nan"), order)
            if row:
                grid.append(row)
                logger.info("  cascade: n=%d retained %.2f%% gross %+.4f net %+.4f",
                            row["n"], 100 * row["retained"], row["gross"],
                            row.get("net", np.nan))

        # --- arm 2: flat GBDT over the union, the required ablation ---------
        scored, _ = T.walk_forward_scores(df, folds, stages_cfg["execution"],
                                          "meta_label")
        if not scored.empty:
            for frac in TOP_FRACTIONS:
                sel = T.select_top_fraction(scored, frac)
                keys = set(zip(sel["symbol"], sel["date"]))
                m = np.array([(s, d) in keys
                              for s, d in zip(scored["symbol"], scored["date"])])
                row = evaluate(scored, m, "flat_gbdt", frac, order)
                if row:
                    grid.append(row)

    results["tradeability"] = tradeability
    grid_df = pd.DataFrame(grid)
    results["grid"] = grid_df
    if grid_df.empty:
        logger.error("no results produced")
        results["h1_met"] = False
        _save(results)
        return 1

    # --- multiple testing ---------------------------------------------------
    conf = grid_df["arm"].isin(CONFIRMATORY_ARMS)
    elig = (grid_df["eligible_h1"].fillna(False).astype(bool)
            if "eligible_h1" in grid_df.columns
            else pd.Series(False, index=grid_df.index))
    eligible = conf & elig
    passed = np.zeros(len(grid_df), dtype=bool)
    if eligible.any():
        p = grid_df.loc[eligible, "net_p"].to_numpy()
        padded = np.concatenate([p, np.ones(PRIOR_TRIALS)])
        passed[np.flatnonzero(eligible.to_numpy())] = pstats.benjamini_hochberg(
            padded, q=FDR_Q)[: len(p)]
    grid_df["meets_h1"] = passed
    results["grid"] = grid_df
    results["n_eligible"] = int(eligible.sum())
    results["h1_met"] = bool(passed.any())

    _report(grid_df, results)
    _save(results)
    return 0


def _report(grid_df: pd.DataFrame, results: Dict) -> None:
    show = [c for c in ("arm", "fraction", "order_size", "n", "n_days", "retained",
                        "gross", "gross_lo", "net", "net_lo", "cost_mean_bps",
                        "kpi_hit_rate", "eligible_h1", "meets_h1")
            if c in grid_df.columns]
    logger.info("=" * 78)
    logger.info("ROUND 6 GRID\n%s", grid_df[show].to_string(index=False,
                                                            float_format="%.4f"))
    if "net" in grid_df.columns and grid_df["net"].notna().any():
        best = grid_df.loc[grid_df["net"].idxmax()]
        logger.info("-" * 78)
        logger.info("best net %+.4f  (%s, $%s, n=%d, days=%d)  CI [%+.4f, %+.4f]  H1=%s",
                    best["net"], best["arm"], f"{int(best['order_size']):,}",
                    best["n"], best["n_days"], best.get("net_lo", np.nan),
                    best.get("net_hi", np.nan), best.get("meets_h1", False))
    logger.info("H1''' met: %s  (eligible %d + prior trials %d)",
                results["h1_met"], results["n_eligible"], PRIOR_TRIALS)
    logger.info("=" * 78)


def _save(results: Dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        pickle.dump(results, fh)
    logger.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    sys.exit(main())
