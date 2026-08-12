"""run_round4_study.py — precision inside the liquid band.

Separate from ``run_round3_study.py`` so round-3 results stay reproducible: that
script's git state *is* the round-3 pre-registration, and a study that rewrites
its own baseline cannot be compared against itself.

What changed, and why
---------------------
Round 3 concluded that microstructure is the binding constraint. Two defects
made that look more final than the evidence supports.

**The cost table was mis-specified.** It sized every order at 1% of ADV, so
liquid names mechanically received larger orders and appeared more expensive.
At a *fixed* order size the ranking inverts -- 395 bps at ADV<=$300k against
63 bps above $20M, a 6.3x span. The "irreducible 77 bps floor" was a whole-panel
median. Cost is a filter, not a wall.

**The funnel was narrow and the cut shallow.** ``gap >= +10%`` with
``adv_20 >= $100k`` gave 20,558 candidates and selection took the top 10%. That
is a recall posture on a precision problem.

So round 4 widens the funnel into the liquid band (``adv_20 >= $5M``,
``gap >= +3%`` -> 44,136 candidates) and cuts to 0.5%. ``MIN_RETAINED`` is gone:
recall is an output here, not a setting.

Three arms, declared in PREREGISTRATION_R4.md before any result was computed:

``gbdt_liquid``       the rank cut, carried over as the comparator
``ev_rule``           take when calibrated E[r] clears that trade's own cost
``blend_transformer`` promoted from round 3's exploratory arm, where it was the
                      only configuration whose gross CI excluded zero

    python research/penny/scripts/run_round4_study.py
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
from research.penny.eval import (  # noqa: E402
    conditional as cx,
    precision as prec,
    stats as pstats,
)
from research.penny.eval.folds import assert_folds_are_clean, walk_forward_folds  # noqa: E402
from research.penny.labels import barrier_free as bf  # noqa: E402
from research.penny.model import calibrate as C, features as F, train as T  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("round4")

OUT_PATH = bars.DATA_ROOT / "analysis_round4.pkl"

# ---- Constants, fixed in PREREGISTRATION_R4.md (commit 2a9a428) -------------
ORDER_SIZES = [2_000, 5_000, 10_000]
TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]

MIN_OOS_TRADES = 300
MIN_OOS_DAYS = 150
MIN_TRAIN_ROWS = 2_000
MAX_SPREAD_BPS = 200.0
FDR_Q = 0.05

# Cumulative across rounds 2 and 3. Unlike round 3, this is folded into the BH
# input rather than merely reported: round 3 computed n_trials = cells + 27 and
# then ran BH on the cells alone, so the prior trials never harshened anything.
PRIOR_TRIALS = 72

# Explicit registry, not a prefix on a label. Round 3 excluded exploratory rows
# by `feature_set.str.startswith("blend:")` and only when Gate B had failed --
# so a passing gate would have silently admitted them to the confirmatory family.
CONFIRMATORY_ARMS = {"gbdt_liquid", "ev_rule", "blend_transformer"}

# Sensitivity: the depth model is crude by its own admission, so a conclusion
# that survives only at the gentlest setting is not a conclusion.
LEVELS_PENALTIES = [1.0, 1.5, 2.0]


def _load(name: str) -> Optional[pd.DataFrame]:
    path = bars.DATA_ROOT / name
    if not path.exists():
        logger.warning("%s missing", name)
        return None
    return pd.read_parquet(path)


def load_inputs(trades_name: str, windows_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_parquet(bars.DATA_ROOT / trades_name)
    path_stats = pd.read_parquet(bars.DATA_ROOT / "barrier_free_r4.parquet")
    keep = ["symbol", "date", "ret_c2c", "mfe_full", "mae_full", "mfe_before_mae"]
    df = trades.merge(path_stats[keep], on=["symbol", "date"], how="inner",
                      validate="one_to_one")
    bf.assert_dominates_truncated(df["mfe_full"], df["max_favourable"])

    df = F.build_matrix(
        df,
        edgar=_load("edgar_features_r4.parquet"),
        premarket=_load("premarket_features_r4.parquet"),
        short_volume=_load("finra_features_r4.parquet"),
        float_data=_load("xbrl_features_r4.parquet"),
    )
    windows = pd.read_parquet(bars.DATA_ROOT / windows_name)
    return df, windows


def apply_tradeability(
    df: pd.DataFrame, order_notional: float, max_spread_bps: float = MAX_SPREAD_BPS
) -> Tuple[pd.DataFrame, Dict]:
    """Drop what cannot be traded, and report what was dropped.

    Not a bias to apologise for: a name with no firm quote, a book smaller than
    the order, or a ruinous spread is outside the strategy by construction. But
    the removed population's own return is reported, because a filter that
    happens to remove the losers is doing the work the model claims to do.
    """
    quoted = df.rename(columns={
        "half_spread_bps_decision": "half_spread_bps",
        "ask_notional_decision": "ask_notional",
    })
    mask = Q.tradeable_mask(quoted, order_notional, max_spread_bps)
    removed = df[~mask]
    report = {
        "order_notional": float(order_notional),
        "n_before": int(len(df)),
        "n_after": int(mask.sum()),
        "share_kept": float(mask.mean()),
        "removed_gross_mean": (
            float(removed["ret_c2c"].mean()) if len(removed) else float("nan")
        ),
        "kept_gross_mean": (
            float(df.loc[mask, "ret_c2c"].mean()) if mask.any() else float("nan")
        ),
        "no_quote": int(df["half_spread_bps_decision"].isna().sum()),
    }
    logger.info(
        "tradeable @ $%s: %s of %s kept (%.1f%%) | removed gross %+.4f vs kept %+.4f",
        f"{order_notional:,.0f}", f"{report['n_after']:,}", f"{report['n_before']:,}",
        100 * report["share_kept"], report["removed_gross_mean"], report["kept_gross_mean"],
    )
    return df[mask].reset_index(drop=True), report


def evaluate_selection(
    scored: pd.DataFrame,
    mask: np.ndarray,
    label: str,
    fraction_label: float,
    order_size: int,
) -> Dict:
    """Gross, net and precision KPIs for one selection at one order size."""
    selected = scored[mask]
    if len(selected) < 30:
        return {}

    gross = pstats.cluster_bootstrap_mean(
        selected["ret_c2c"], selected["date"], n_boot=3_000
    )
    lift = cx.compare_selection_to_baseline(
        scored, np.asarray(mask, dtype=bool), return_col="ret_c2c", n_boot=2_000
    )

    net = selected["ret_c2c"] - selected["cost_realized"]
    ok = net.notna()
    row = {
        "arm": label,
        "fraction": fraction_label,
        "order_size": order_size,
        "n": int(gross.n),
        "n_days": int(gross.n_days),
        "retained": float(len(selected) / max(len(scored), 1)),
        "gross": float(gross.mean),
        "gross_lo": float(gross.lo),
        "gross_hi": float(gross.hi),
        "lift": float(lift["delta"]),
        "lift_p": float(lift["p_value"]),
        "cost_mean_bps": float(10_000 * selected["cost_realized"].mean()),
        "cost_coverage": float(ok.mean()),
    }

    if ok.sum() >= 30:
        est = pstats.cluster_bootstrap_mean(net[ok], selected.loc[ok, "date"], n_boot=3_000)
        row.update({
            "net": float(est.mean),
            "net_lo": float(est.lo),
            "net_hi": float(est.hi),
            "net_p": float(est.p_value_greater),
            "net_n": int(est.n),
            "net_days": int(est.n_days),
        })
        # Provisional only: the family decision is made after FDR correction.
        # "Any of ~N cells excludes zero" is likely under a pure null.
        row["eligible_h1"] = bool(est.n >= MIN_OOS_TRADES and est.n_days >= MIN_OOS_DAYS)

        kpi = prec.precision_kpis(
            selected.loc[ok].assign(net_return=net[ok]), "net_return"
        )
        row.update({f"kpi_{k}": v for k, v in kpi.items()})
    else:
        row["eligible_h1"] = False
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Round-4 precision study.")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--min-train-days", type=int, default=900)
    parser.add_argument("--trades", default="trades_r4.parquet")
    parser.add_argument("--windows", default="quote_windows_r4.parquet")
    parser.add_argument("--skip-sequence", action="store_true")
    parser.add_argument("--n-boot", type=int, default=3_000)
    args = parser.parse_args(argv)

    df_all, windows = load_inputs(args.trades, args.windows)
    logger.info("trades %s | quote windows %s",
                f"{len(df_all):,}", f"{len(windows):,}")

    results: Dict = {
        "preregistration_commit": "2a9a428",
        "order_sizes": ORDER_SIZES,
        "top_fractions": TOP_FRACTIONS,
        "confirmatory_arms": sorted(CONFIRMATORY_ARMS),
        "prior_trials": PRIOR_TRIALS,
        "n_trades_raw": int(len(df_all)),
        "date_min": str(df_all["date"].min().date()),
        "date_max": str(df_all["date"].max().date()),
    }

    grid: List[Dict] = []
    tradeability: Dict[int, Dict] = {}
    ladder_moves: List[str] = []

    for order_size in ORDER_SIZES:
        logger.info("=" * 72)
        logger.info("ORDER SIZE $%s", f"{order_size:,}")

        costed = Q.attach_costs(df_all, windows, order_notional=float(order_size))
        df, report = apply_tradeability(costed, float(order_size))
        tradeability[order_size] = report
        if df.empty:
            ladder_moves.append(f"${order_size}: no tradeable candidates")
            continue

        # The label is now per trade: "would this have cleared ITS OWN cost".
        # Round 3 trained against a flat 437 bps while true cost spanned
        # 34 -> 2000+ bps, so the model was learning the wrong question.
        df["meta_label"] = (df["ret_c2c"] > df["cost_hat"]).astype(int)
        df = df.dropna(subset=["cost_hat"]).reset_index(drop=True)
        logger.info("labelled %s trades | positive class %.1f%%",
                    f"{len(df):,}", 100 * df["meta_label"].mean())

        folds = walk_forward_folds(
            df["date"].unique(), n_splits=args.n_splits,
            embargo_days=args.embargo_days, min_train_days=args.min_train_days,
        )
        assert_folds_are_clean(folds, embargo_days=args.embargo_days)
        for f in folds:
            n_train = int(df["date"].isin(f.train_dates).sum())
            logger.info("%s | %s train rows", f.describe(), f"{n_train:,}")
            assert n_train >= MIN_TRAIN_ROWS, (
                f"fold {f.index} trains on {n_train} rows, below "
                f"MIN_TRAIN_ROWS={MIN_TRAIN_ROWS}; round 3's fold 0 trained on "
                f"610 rows and fabricated the round-2 headline"
            )

        cols = F.feature_columns(df, include_cost=True)
        logger.info("%d features (cost features enabled)", len(cols))

        # --- arm 1: rank cut, the comparator ------------------------------
        scored, _ = T.walk_forward_scores(df, folds, cols, "meta_label")
        if scored.empty:
            continue
        for frac in TOP_FRACTIONS:
            sel = T.select_top_fraction(scored, frac)
            keys = set(zip(sel["symbol"], sel["date"]))
            mask = np.array([(s, d) in keys
                             for s, d in zip(scored["symbol"], scored["date"])])
            row = evaluate_selection(scored, mask, "gbdt_liquid", frac, order_size)
            if row:
                grid.append(row)

        # --- arm 2: expected value against each trade's own cost ----------
        ev_scored = C.walk_forward_ev(df, folds, cols, "meta_label")
        if not ev_scored.empty:
            ev_mask = C.select_by_expected_value(ev_scored, "cost_hat").to_numpy()
            row = evaluate_selection(ev_scored, ev_mask, "ev_rule",
                                     float("nan"), order_size)
            if row:
                grid.append(row)
            results.setdefault("calibration", {})[order_size] = (
                C.calibration_table(ev_scored, "meta_label").to_dict(orient="records")
            )
            logger.info("ev_rule selected %d of %s (%.2f%%)",
                        int(ev_mask.sum()), f"{len(ev_scored):,}",
                        100 * ev_mask.mean())

        # --- arm 3: transformer blend, promoted to confirmatory -----------
        if not args.skip_sequence:
            rows = run_transformer(df, folds, scored, cols, order_size)
            grid.extend(rows)

    results["tradeability"] = tradeability
    grid_df = pd.DataFrame(grid)
    results["grid"] = grid_df

    if grid_df.empty:
        logger.error("no results produced")
        results["h1_met"] = False
        _save(results)
        return 1

    # --- multiple testing --------------------------------------------------
    confirmatory = grid_df["arm"].isin(CONFIRMATORY_ARMS)
    eligible = confirmatory & grid_df.get("eligible_h1", False).fillna(False)
    logger.info("cells: %d total, %d confirmatory, %d eligible",
                len(grid_df), int(confirmatory.sum()), int(eligible.sum()))

    passed = np.zeros(len(grid_df), dtype=bool)
    if eligible.any():
        p = grid_df.loc[eligible, "net_p"].to_numpy()
        # Pad with the prior trials as p = 1. BH's threshold depends on the
        # RANK of each p among all tests performed, so adding trials that cannot
        # pass still makes the correction stricter -- which is the point, and
        # what round 3 reported but did not do.
        padded = np.concatenate([p, np.ones(PRIOR_TRIALS)])
        sub = pstats.benjamini_hochberg(padded, q=FDR_Q)[: len(p)]
        passed[np.flatnonzero(eligible.to_numpy())] = sub
    grid_df["meets_h1"] = passed
    results["grid"] = grid_df
    results["n_cells"] = int(len(grid_df))
    results["n_eligible"] = int(eligible.sum())
    results["n_trials"] = int(eligible.sum()) + PRIOR_TRIALS
    results["h1_met"] = bool(passed.any())
    results["ladder_moves"] = ladder_moves

    _report(grid_df, results)
    _save(results)
    return 0


def run_transformer(df, folds, gbdt_scored, cols, order_size) -> List[Dict]:
    """The arm promoted from round 3, where its gross CI was the only one
    excluding zero. Same code, new funnel and a declared confirmatory status."""
    try:
        from research.penny.model import sequence as S
        from research.penny.scripts.fetch_premarket import session_path
    except ImportError as exc:  # noqa: BLE001
        logger.warning("sequence models unavailable: %s", exc)
        return []

    panels = []
    for day in sorted(df["date"].unique()):
        p = session_path(pd.Timestamp(day))
        if p.exists():
            part = pd.read_parquet(p)
            if not part.empty:
                panels.append(part)
    if not panels:
        logger.warning("no pre-open bars; skipping the transformer arm")
        return []

    panel = pd.concat(panels, ignore_index=True)
    x, keys = S.build_sequences(panel, df[["symbol", "date"]])
    assert len(x) == len(df), f"sequence tensor has {len(x)} rows for {len(df)} trades"
    assert (keys["symbol"].to_numpy() == df["symbol"].to_numpy()).all(), (
        "sequence keys are not in trade-table order"
    )

    seq = np.full(len(df), np.nan)
    for fold in folds:
        tr = df["date"].isin(fold.train_dates).to_numpy()
        te = df["date"].isin(fold.test_dates).to_numpy()
        if tr.sum() < MIN_TRAIN_ROWS or te.sum() == 0:
            continue
        seq[te] = S.train_sequence_model(
            x[tr], df.loc[tr, "meta_label"].to_numpy(), x[te], kind="transformer"
        )

    merged = df.assign(seq_score=seq).merge(
        gbdt_scored[["symbol", "date", "model_score", "fold"]],
        on=["symbol", "date"], how="inner", validate="one_to_one",
    ).dropna(subset=["seq_score"])
    if merged.empty:
        return []

    # Rank within fold: ranking over the pooled panel would make each score
    # depend on the distribution of later years (the look-ahead fixed in round 3).
    merged["model_score"] = S.blend_scores(
        merged["model_score"].to_numpy(), merged["seq_score"].to_numpy(),
        groups=merged["fold"].to_numpy(),
    )

    rows = []
    for frac in TOP_FRACTIONS:
        sel = T.select_top_fraction(merged, frac)
        keyset = set(zip(sel["symbol"], sel["date"]))
        mask = np.array([(s, d) in keyset
                         for s, d in zip(merged["symbol"], merged["date"])])
        row = evaluate_selection(merged, mask, "blend_transformer", frac, order_size)
        if row:
            rows.append(row)
    return rows


def _report(grid_df: pd.DataFrame, results: Dict) -> None:
    logger.info("=" * 72)
    logger.info("PRECISION LADDER (net, per arm and selectivity)")
    show = [c for c in ("arm", "fraction", "order_size", "n", "n_days", "retained",
                        "gross", "net", "net_lo", "cost_mean_bps",
                        "kpi_hit_rate", "kpi_trades_per_month", "eligible_h1",
                        "meets_h1") if c in grid_df.columns]
    logger.info("\n%s", grid_df[show].to_string(index=False, float_format="%.4f"))

    if "net" in grid_df.columns and grid_df["net"].notna().any():
        best = grid_df.loc[grid_df["net"].idxmax()]
        logger.info("-" * 72)
        logger.info("best net %+.4f  (%s, frac=%.3f, $%s, n=%d, days=%d)",
                    best["net"], best["arm"], best["fraction"],
                    f"{int(best['order_size']):,}", best["n"], best["n_days"])
        logger.info("  CI [%+.4f, %+.4f]  p=%.4f  H1=%s",
                    best.get("net_lo", np.nan), best.get("net_hi", np.nan),
                    best.get("net_p", np.nan), best.get("meets_h1", False))
    logger.info("=" * 72)
    logger.info("H1' met: %s   (eligible cells %d + prior trials %d)",
                results["h1_met"], results["n_eligible"], PRIOR_TRIALS)
    logger.info("=" * 72)


def _save(results: Dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        pickle.dump(results, fh)
    logger.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    sys.exit(main())
