"""run_round3_study.py — the round-3 experiment.

Separate from ``run_model_study.py`` so round-2 results stay reproducible: a
study that silently rewrites its own baseline cannot be compared against itself.

Structure
---------
1. Feature-set ablation, each step scored out-of-sample:
   base -> +EDGAR -> +premarket -> +short volume -> +float
2. **Gate B**: does premarket lift walk-forward gross by >= 50 bps? If not, the
   sequence model is not built -- if aggregates of the pre-open path carry
   nothing, an encoder over the same bars almost certainly carries nothing.
3. Sequence encoders (CNN / GRU / transformer), blended with the GBDT by rank.
4. Evaluation against **measured** costs, which are a function of order size,
   not the flat 437 bps every previous round assumed.

Costs
-----
Gate A measured a floor of ~77 bps (crossing the spread twice, paid at any
size) rising steeply with order size as the order exceeds the resting quote.
Results are therefore reported per order size rather than against one number.

    python research/penny/scripts/run_round3_study.py
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

from research.penny.etl import bars, quotes as Q  # noqa: E402
from research.penny.eval import conditional as cx, stats as pstats  # noqa: E402
from research.penny.eval.folds import assert_folds_are_clean, walk_forward_folds  # noqa: E402
from research.penny.labels import barrier_free as bf  # noqa: E402
from research.penny.model import features as F, train as T  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("round3")

OUT_PATH = bars.DATA_ROOT / "analysis_round3.pkl"

ORDER_SIZES = [2_000, 5_000, 10_000]
TOP_FRACTIONS = [0.05, 0.10, 0.20]
LABEL_COST = 0.0437          # the label threshold, held fixed across rounds

# Gate B: premarket must add at least this much walk-forward gross.
GATE_B_MIN_LIFT = 0.0050

MIN_OOS_TRADES = 500
MIN_RETAINED = 0.05

# Cumulative across rounds 2 and 3, for the Deflated Sharpe.
PRIOR_TRIALS = 27


def _load(name: str) -> Optional[pd.DataFrame]:
    path = bars.DATA_ROOT / name
    if not path.exists():
        logger.warning("%s missing", name)
        return None
    return pd.read_parquet(path)


def load_inputs() -> pd.DataFrame:
    """Join every data source onto the labelled trade table."""
    trades = pd.read_parquet(bars.DATA_ROOT / "trades.parquet")

    path_stats = pd.read_parquet(bars.DATA_ROOT / "barrier_free.parquet")
    keep = ["symbol", "date", "ret_c2c", "mfe_full", "mae_full", "mfe_before_mae"]
    df = trades.merge(path_stats[keep], on=["symbol", "date"], how="inner",
                      validate="one_to_one")
    bf.assert_dominates_truncated(df["mfe_full"], df["max_favourable"])

    return F.build_matrix(
        df,
        edgar=_load("edgar_features.parquet"),
        premarket=_load("premarket_features.parquet"),
        short_volume=_load("finra_features.parquet"),
        float_data=_load("xbrl_features.parquet"),
    )


def attach_measured_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Per-trade cost at each order size, from the Gate A measurement.

    Trades without a quote match keep NaN rather than inheriting a median: a
    made-up cost on an unmeasured trade is exactly the error this round set out
    to fix.
    """
    summary = _load("quote_summary.parquet")
    if summary is None:
        logger.warning("no quote summary; falling back to the assumed 437 bps")
        for size in ORDER_SIZES:
            df[f"cost_{size}"] = 0.0437
        return df

    merged = df.merge(
        summary[["symbol", "date", "half_spread_bps", "ask_notional"]],
        on=["symbol", "date"], how="left", validate="one_to_one",
    )
    for size in ORDER_SIZES:
        part = (size / merged["adv_20"]).clip(0, 1)
        impact = Q.impact_bps(merged["vol_20d"], part)
        depth = Q.depth_shortfall_bps(
            pd.Series(float(size), index=merged.index),
            merged["ask_notional"], merged["half_spread_bps"],
        )
        merged[f"cost_{size}"] = (
            2 * (merged["half_spread_bps"] + impact) + depth
        ) / 10_000

    coverage = merged["half_spread_bps"].notna().mean()
    logger.info("measured cost coverage: %.1f%% of trades", 100 * coverage)
    return merged


def evaluate_selection(
    scored: pd.DataFrame, fraction: float, label: str
) -> Dict:
    """Gross and net-at-each-order-size for one selectivity."""
    selected = T.select_top_fraction(scored, fraction)
    if len(selected) < 50:
        return {}

    gross = pstats.cluster_bootstrap_mean(
        selected["ret_c2c"], selected["date"], n_boot=3_000
    )

    keys = set(zip(selected["symbol"], selected["date"]))
    mask = np.array([(s, d) in keys for s, d in zip(scored["symbol"], scored["date"])])
    lift = cx.compare_selection_to_baseline(
        scored, mask, return_col="ret_c2c", n_boot=2_000
    )

    row = {
        "feature_set": label,
        "fraction": fraction,
        "n": gross.n,
        "n_days": gross.n_days,
        "retained": len(selected) / max(len(scored), 1),
        "gross": gross.mean,
        "gross_lo": gross.lo,
        "gross_hi": gross.hi,
        "lift": lift["delta"],
        "lift_p": lift["p_value"],
    }

    for size in ORDER_SIZES:
        col = f"cost_{size}"
        if col not in selected.columns:
            continue
        net = selected["ret_c2c"] - selected[col]
        ok = net.notna()
        if ok.sum() < 50:
            continue
        est = pstats.cluster_bootstrap_mean(net[ok], selected.loc[ok, "date"], n_boot=3_000)
        row[f"net_{size}"] = est.mean
        row[f"net_lo_{size}"] = est.lo
        row[f"net_p_{size}"] = est.p_value_greater
        row[f"net_n_{size}"] = int(est.n)
        # Fraction of selected trades with no measured cost. Reported because
        # dropping them tilts the estimate toward names with firm quotes --
        # i.e. optimistically.
        row[f"cost_coverage_{size}"] = float(ok.mean())
        # Provisional only. The family-wise decision is made after FDR
        # correction across the whole grid; a per-cell CI is one of ~72 tests
        # and "any of 72 excludes zero" is likely under a pure null.
        row[f"eligible_h1_{size}"] = bool(
            est.n >= MIN_OOS_TRADES and row["retained"] >= MIN_RETAINED
        )
    return row


def run_feature_set(
    df: pd.DataFrame, folds, cols: Sequence[str], label: str
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Walk-forward scoring plus evaluation for one feature set."""
    scored, importance = T.walk_forward_scores(df, folds, cols, "meta_label")
    if scored.empty:
        return scored, []
    rows = [r for f in TOP_FRACTIONS if (r := evaluate_selection(scored, f, label))]
    return scored, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Round-3 study.")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--embargo-days", type=int, default=10)
    # Raised from 250: fold 0 previously trained on 240 days / 610 rows, which
    # is not evidence of anything and diluted the pooled result.
    parser.add_argument("--min-train-days", type=int, default=900)
    parser.add_argument("--skip-sequence", action="store_true")
    parser.add_argument(
        "--force-sequence", action="store_true",
        help="Run the sequence encoders even if Gate B fails. Results are "
             "recorded as EXPLORATORY and excluded from the H1 verdict.",
    )
    args = parser.parse_args(argv)

    df = load_inputs()
    df = attach_measured_costs(df)
    df["meta_label"] = bf.meta_label(df["ret_c2c"], LABEL_COST)
    logger.info("trades %s | positive class %.1f%%",
                f"{len(df):,}", 100 * df["meta_label"].mean())

    folds = walk_forward_folds(
        df["date"].unique(), n_splits=args.n_splits,
        embargo_days=args.embargo_days, min_train_days=args.min_train_days,
    )
    assert_folds_are_clean(folds, embargo_days=args.embargo_days)
    for f in folds:
        logger.info(f.describe())

    # --- feature-set ablation ---------------------------------------------
    ladder = [
        ("base", dict(include_edgar=False, include_premarket=False,
                      include_short=False, include_float=False)),
        ("+edgar", dict(include_edgar=True, include_premarket=False,
                        include_short=False, include_float=False)),
        ("+premarket", dict(include_edgar=True, include_premarket=True,
                            include_short=False, include_float=False)),
        ("+short", dict(include_edgar=True, include_premarket=True,
                        include_short=True, include_float=False)),
        ("+float", dict(include_edgar=True, include_premarket=True,
                        include_short=True, include_float=True)),
    ]

    results: Dict = {
        "n_trades": int(len(df)),
        "folds": [f.describe() for f in folds],
        "order_sizes": ORDER_SIZES,
        "label_cost": LABEL_COST,
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
    }
    grid: List[Dict] = []
    scored_by_set: Dict[str, pd.DataFrame] = {}
    gross_by_set: Dict[str, float] = {}

    for label, kwargs in ladder:
        cols = F.feature_columns(df, **kwargs)
        scored, rows = run_feature_set(df, folds, cols, label)
        if scored.empty:
            continue
        scored_by_set[label] = scored
        grid.extend(rows)

        at10 = [r for r in rows if r["fraction"] == 0.10]
        gross_by_set[label] = at10[0]["gross"] if at10 else np.nan
        logger.info(
            "%-11s %2d features | top10%% gross %+.4f  lift %+.4f (p=%.3f)",
            label, len(cols),
            gross_by_set[label],
            at10[0]["lift"] if at10 else np.nan,
            at10[0]["lift_p"] if at10 else np.nan,
        )

    results["grid"] = pd.DataFrame(grid)
    results["gross_by_set"] = gross_by_set

    # --- Gate B ------------------------------------------------------------
    premarket_lift = gross_by_set.get("+premarket", np.nan) - gross_by_set.get("+edgar", np.nan)
    gate_b = bool(np.isfinite(premarket_lift) and premarket_lift >= GATE_B_MIN_LIFT)
    results["premarket_lift"] = float(premarket_lift)
    results["gate_b_pass"] = gate_b
    logger.info("=" * 68)
    logger.info("GATE B: premarket lift %+.4f (threshold %+.4f) -> %s",
                premarket_lift, GATE_B_MIN_LIFT,
                "PASS -- build the sequence model" if gate_b else "FAIL -- skip it")
    logger.info("=" * 68)

    # --- sequence models ---------------------------------------------------
    run_sequence = (gate_b or args.force_sequence) and not args.skip_sequence
    if run_sequence and not gate_b:
        logger.warning(
            "running sequence encoders despite Gate B failing -- results are "
            "EXPLORATORY and are excluded from the H1 verdict"
        )
    results["sequence_exploratory"] = bool(run_sequence and not gate_b)

    if run_sequence:
        from research.penny.features.premarket import DECISION_MINUTE  # noqa: F401
        from research.penny.model import sequence as S
        from research.penny.scripts.fetch_premarket import session_path

        logger.info("loading pre-open bars for the sequence encoder ...")
        panels = []
        for day in sorted(df["date"].unique()):
            p = session_path(pd.Timestamp(day))
            if p.exists():
                part = pd.read_parquet(p)
                if not part.empty:
                    panels.append(part)
        panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()

        if panel.empty:
            logger.warning("no pre-open bars available; skipping sequence models")
        else:
            x, keys = S.build_sequences(panel, df[["symbol", "date"]])
            # The boolean masks below index `x` and `df` positionally, so one
            # tensor row per trade row, in the same order, is load-bearing. If
            # build_sequences ever starts dropping unmatched keys this silently
            # pairs the wrong sequence with the wrong label.
            assert len(x) == len(df), (
                f"sequence tensor has {len(x)} rows for {len(df)} trades"
            )
            assert (keys["symbol"].to_numpy() == df["symbol"].to_numpy()).all(), (
                "sequence keys are not in trade-table order"
            )
            logger.info("sequence tensor %s", x.shape)
            best_cols = F.feature_columns(df)

            for kind in ("cnn", "gru", "transformer"):
                seq_scores = np.full(len(df), np.nan)
                for fold in folds:
                    tr = df["date"].isin(fold.train_dates).to_numpy()
                    te = df["date"].isin(fold.test_dates).to_numpy()
                    if tr.sum() < 500 or te.sum() == 0:
                        continue
                    seq_scores[te] = S.train_sequence_model(
                        x[tr], df.loc[tr, "meta_label"].to_numpy(), x[te], kind=kind
                    )

                base = scored_by_set.get("+float")
                if base is None:
                    continue
                merged = df.assign(seq_score=seq_scores).merge(
                    base[["symbol", "date", "model_score", "fold"]],
                    on=["symbol", "date"], how="inner", validate="one_to_one",
                )
                merged = merged.dropna(subset=["seq_score"])
                if merged.empty:
                    continue
                # Rank within fold. Ranking over the pooled panel would make
                # each score depend on the distribution of later years.
                merged["model_score"] = S.blend_scores(
                    merged["model_score"].to_numpy(), merged["seq_score"].to_numpy(),
                    groups=merged["fold"].to_numpy(),
                )
                rows = [r for f in TOP_FRACTIONS
                        if (r := evaluate_selection(merged, f, f"blend:{kind}"))]
                grid.extend(rows)
                at10 = [r for r in rows if r["fraction"] == 0.10]
                logger.info("blend:%-12s top10%% gross %+.4f",
                            kind, at10[0]["gross"] if at10 else np.nan)

    grid_df = pd.DataFrame(grid)

    # --- multiple testing, applied before any verdict ----------------------
    # Every (configuration x order size) cell is a separate test. Counting only
    # grid ROWS would undercount by a factor of len(ORDER_SIZES), and taking
    # the maximum over uncorrected CIs would declare success on noise.
    cells: List[Dict] = []
    for i, row in grid_df.iterrows():
        for size in ORDER_SIZES:
            p_col, elig_col = f"net_p_{size}", f"eligible_h1_{size}"
            if p_col not in grid_df.columns or not np.isfinite(row.get(p_col, np.nan)):
                continue
            cells.append({
                "row": i, "size": size,
                "p": float(row[p_col]),
                "eligible": bool(row.get(elig_col, False)),
                "net": float(row.get(f"net_{size}", np.nan)),
                "net_lo": float(row.get(f"net_lo_{size}", np.nan)),
            })

    # Exploratory arms (sequence encoders run despite a failed Gate B) are
    # excluded from the H1 decision. They were not pre-registered, and letting
    # them into the family would both inflate the correction and allow a
    # post-hoc arm to carry the verdict.
    if results.get("sequence_exploratory") and not grid_df.empty:
        exploratory = grid_df["feature_set"].str.startswith("blend:")
        cells = [c for c in cells if not exploratory.iloc[c["row"]]]
        logger.info("excluded %d exploratory cells from the H1 family",
                    int(exploratory.sum()) * len(ORDER_SIZES))

    cells_df = pd.DataFrame(cells)
    n_tests = len(cells_df) + PRIOR_TRIALS
    results["n_trials"] = n_tests
    results["n_cells"] = len(cells_df)

    if not cells_df.empty:
        eligible = cells_df["eligible"].to_numpy()
        passed = np.zeros(len(cells_df), dtype=bool)
        if eligible.any():
            # Correct only among eligible cells; ineligible ones were never
            # candidates and must not inflate the correction for the rest.
            sub = pstats.benjamini_hochberg(
                cells_df.loc[eligible, "p"].to_numpy(), q=0.05
            )
            passed[np.flatnonzero(eligible)] = sub
        cells_df["q_pass"] = passed
        results["cells"] = cells_df
        any_h1 = bool(passed.any())
    else:
        results["cells"] = cells_df
        any_h1 = False

    for size in ORDER_SIZES:
        col = f"meets_h1_{size}"
        grid_df[col] = False
    if not cells_df.empty:
        for _, c in cells_df[cells_df.get("q_pass", False) == True].iterrows():
            grid_df.at[c["row"], f"meets_h1_{int(c['size'])}"] = True

    results["grid"] = grid_df
    results["h1_met"] = any_h1

    logger.info("=" * 68)
    if not grid_df.empty:
        best = grid_df.loc[grid_df["gross"].idxmax()]
        logger.info("best gross %+.4f  (%s, top %.0f%%, n=%d)",
                    best["gross"], best["feature_set"], 100 * best["fraction"], best["n"])
        for size in ORDER_SIZES:
            col = f"net_{size}"
            if col in grid_df.columns and grid_df[col].notna().any():
                row = grid_df.loc[grid_df[col].idxmax()]
                logger.info("  best net @ $%-6s %+.4f  CI low %+.4f  H1=%s",
                            f"{size:,}", row[col], row.get(f"net_lo_{size}", np.nan),
                            row.get(f"meets_h1_{size}", False))
    logger.info("H1 met at any order size: %s", any_h1)
    logger.info("cumulative trials: %d", results["n_trials"])
    logger.info("=" * 68)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        pickle.dump(results, fh)
    logger.info("wrote %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
