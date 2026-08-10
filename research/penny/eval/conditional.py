"""conditional.py — the conditional-expectation surface. The study's deliverable.

The question, precisely
-----------------------
Not "can a model predict explosions" but: **does there exist a veto set V such
that E[net return | not V] > 0**, at a non-trivial retained population and a
capacity worth having?

That framing matters. The published evidence against penny day trading
(Eraker & Ready 2015; SEC DERA 2016; Bali, Cakici & Whitelaw 2011) measures
*unconditional* expectation. A high-precision veto can move a conditional mean
even when the unconditional one is deeply negative. This module measures
exactly that, and prices in the search cost of having looked.

Three guards against fooling ourselves
--------------------------------------
1. **Selection happens inside the fold.** Veto sets are chosen on training days
   and scored on held-out days. Choosing and scoring on the same data is the
   classic overfit and would "discover" a filter for pure noise.
2. **The baseline is matched on selectivity.** Any veto retaining X% of the
   population is compared against *random* vetoes retaining X%. A filter that
   merely trades less is not a filter that trades better -- and a smaller
   sample alone widens intervals enough to look impressive.
3. **The search size is carried through.** Every reported p-value is FDR-
   corrected against the true number of subsets examined, not the number that
   happened to look good.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.penny.eval import stats as pstats
from research.penny.eval.folds import Fold, split_frame

logger = logging.getLogger(__name__)


@dataclass
class VetoResult:
    """Performance of one veto set on one population."""

    name: str
    vetoes: Tuple[str, ...]
    n_trades: int
    n_days: int
    retained_frac: float
    mean_net: float
    ci_lo: float
    ci_hi: float
    p_value: float
    median_net: float
    hit_rate: float
    p_target: float
    p_stop: float
    sharpe: float
    max_dd: float
    total_return: float
    avg_holding_min: float
    q_pass: bool = False

    def to_dict(self) -> Dict:
        return {**self.__dict__, "vetoes": ", ".join(self.vetoes) or "(none)"}


def _kpis(trades: pd.DataFrame, return_col: str) -> Dict[str, float]:
    """Portfolio-level KPIs from a set of trades."""
    curve = pstats.daily_equity_curve(trades, return_col=return_col)
    if curve.empty:
        return {
            "sharpe": np.nan, "max_dd": np.nan,
            "total_return": np.nan, "n_days": 0,
        }
    return {
        "sharpe": pstats.sharpe_ratio(curve["ret"]),
        "max_dd": pstats.max_drawdown(curve["equity"]),
        "total_return": float(curve["equity"].iloc[-1] - 1.0),
        "n_days": int(len(curve)),
    }


def evaluate_population(
    trades: pd.DataFrame,
    name: str = "baseline",
    vetoes: Tuple[str, ...] = (),
    return_col: str = "net_return",
    date_col: str = "date",
    n_boot: int = 2_000,
    baseline_n: Optional[int] = None,
    seed: int = 42,
) -> VetoResult:
    """Full statistics for one (already filtered) population of trades."""
    df = trades.dropna(subset=[return_col])
    if df.empty:
        return VetoResult(
            name, vetoes, 0, 0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        )

    est = pstats.cluster_bootstrap_mean(
        df[return_col].to_numpy(), df[date_col].to_numpy(), n_boot=n_boot, seed=seed
    )
    kpi = _kpis(df, return_col)
    outcomes = df["outcome"] if "outcome" in df.columns else pd.Series(dtype=object)

    return VetoResult(
        name=name,
        vetoes=vetoes,
        n_trades=int(len(df)),
        n_days=int(kpi["n_days"]),
        retained_frac=float(len(df) / baseline_n) if baseline_n else 1.0,
        mean_net=est.mean,
        ci_lo=est.lo,
        ci_hi=est.hi,
        p_value=est.p_value,
        median_net=float(df[return_col].median()),
        hit_rate=float((df[return_col] > 0).mean()),
        p_target=float((outcomes == "target").mean()) if len(outcomes) else np.nan,
        p_stop=float((outcomes == "stop").mean()) if len(outcomes) else np.nan,
        sharpe=kpi["sharpe"],
        max_dd=kpi["max_dd"],
        total_return=kpi["total_return"],
        avg_holding_min=(
            float(df["holding_minutes"].mean()) if "holding_minutes" in df.columns else np.nan
        ),
    )


def apply_vetoes(trades: pd.DataFrame, vetoes: Sequence[str]) -> pd.Series:
    """Boolean mask of trades that SURVIVE every veto.

    A veto column is True when the condition is present and the trade should be
    *rejected*, so survival is the negation of their union.
    """
    if not vetoes:
        return pd.Series(True, index=trades.index)
    reject = pd.Series(False, index=trades.index)
    for col in vetoes:
        reject |= trades[col].fillna(False).astype(bool)
    return ~reject


def single_veto_table(
    trades: pd.DataFrame,
    veto_cols: Sequence[str],
    return_col: str = "net_return",
    n_boot: int = 2_000,
) -> pd.DataFrame:
    """Marginal effect of each veto applied alone.

    Diagnostic, not a selection procedure -- reading the best row off this table
    and calling it a result is precisely the error the fold machinery exists to
    prevent.
    """
    baseline_n = len(trades.dropna(subset=[return_col]))
    rows = []
    for col in veto_cols:
        survive = apply_vetoes(trades, [col])
        kept = evaluate_population(
            trades[survive], name=f"not {col}", vetoes=(col,),
            return_col=return_col, n_boot=n_boot, baseline_n=baseline_n,
        )
        removed = evaluate_population(
            trades[~survive], name=f"{col}", vetoes=(),
            return_col=return_col, n_boot=n_boot, baseline_n=baseline_n,
        )
        rows.append(
            {
                "veto": col,
                "n_kept": kept.n_trades,
                "retained_frac": kept.retained_frac,
                "mean_kept": kept.mean_net,
                "ci_lo_kept": kept.ci_lo,
                "ci_hi_kept": kept.ci_hi,
                "p_kept": kept.p_value,
                "sharpe_kept": kept.sharpe,
                "n_removed": removed.n_trades,
                "mean_removed": removed.mean_net,
                "lift": kept.mean_net - removed.mean_net,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_kept", ascending=False)


def enumerate_veto_sets(veto_cols: Sequence[str], max_size: int = 4) -> List[Tuple[str, ...]]:
    """All veto subsets up to ``max_size`` (including the empty set)."""
    sets: List[Tuple[str, ...]] = [()]
    for size in range(1, max_size + 1):
        sets.extend(itertools.combinations(veto_cols, size))
    return sets


def search_veto_sets(
    trades: pd.DataFrame,
    veto_cols: Sequence[str],
    max_size: int = 3,
    min_retained: float = 0.02,
    min_trades: int = 200,
    return_col: str = "net_return",
    n_boot: int = 1_000,
    q: float = 0.05,
) -> pd.DataFrame:
    """Exhaustive search over veto sets, FDR-corrected.

    ``min_retained`` and ``min_trades`` are enforced *before* significance is
    considered. A filter that keeps eleven trades can post a spectacular mean;
    it is not a strategy, and letting such rows into the multiple-testing pool
    would also inflate the correction for everyone else.
    """
    df = trades.dropna(subset=[return_col])
    baseline_n = len(df)
    candidates = enumerate_veto_sets(veto_cols, max_size)

    results: List[VetoResult] = []
    for vetoes in candidates:
        survive = apply_vetoes(df, vetoes)
        kept = df[survive]
        if len(kept) < min_trades or len(kept) / baseline_n < min_retained:
            continue
        name = " & ".join(f"~{v}" for v in vetoes) if vetoes else "(no veto)"
        results.append(
            evaluate_population(
                kept, name=name, vetoes=vetoes, return_col=return_col,
                n_boot=n_boot, baseline_n=baseline_n,
            )
        )

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame([r.to_dict() for r in results])
    out["n_hypotheses"] = len(results)
    out["q_pass"] = pstats.benjamini_hochberg(out["p_value"].fillna(1.0).to_numpy(), q=q)
    return out.sort_values("mean_net", ascending=False).reset_index(drop=True)


def random_veto_placebo(
    trades: pd.DataFrame,
    retained_frac: float,
    n_draws: int = 200,
    return_col: str = "net_return",
    date_col: str = "date",
    seed: int = 7,
) -> Dict[str, float]:
    """Distribution of means from random filters of the same selectivity.

    This is the fair baseline. A real veto must beat *random* filtering that
    keeps the same fraction of trades; otherwise the apparent improvement is
    just a smaller, noisier sample.

    Whole days are dropped rather than individual trades, matching how a real
    veto behaves -- many vetoes (a pending offering, a deficiency notice) are
    day-level facts, so a trade-level placebo would be an easier bar to clear.
    """
    df = trades.dropna(subset=[return_col])
    days = df[date_col].unique()
    rng = np.random.default_rng(seed)
    keep_n = max(1, int(round(len(days) * retained_frac)))

    means = []
    for _ in range(n_draws):
        picked = rng.choice(days, size=keep_n, replace=False)
        sample = df[df[date_col].isin(picked)][return_col]
        if len(sample):
            means.append(float(sample.mean()))

    if not means:
        return {"placebo_mean": np.nan, "placebo_p95": np.nan, "placebo_std": np.nan}

    arr = np.asarray(means)
    return {
        "placebo_mean": float(arr.mean()),
        "placebo_p05": float(np.percentile(arr, 5)),
        "placebo_p95": float(np.percentile(arr, 95)),
        "placebo_std": float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
        "n_draws": len(arr),
    }


def compare_to_baseline(
    trades: pd.DataFrame,
    vetoes: Sequence[str],
    return_col: str = "net_return",
    date_col: str = "date",
    n_boot: int = 4_000,
    seed: int = 23,
) -> Dict[str, float]:
    """Paired, day-clustered bootstrap of (veto mean - baseline mean).

    Two questions look alike and are not:

    * "Is the surviving population profitable?"  -- tests the subset against 0.
    * "Does the veto *help*?"                    -- tests the subset against the
      unfiltered baseline.

    Both are needed. In a universe whose baseline is strongly negative a veto
    can add real value while still leaving a losing population; conversely a
    subset can clear zero without the veto having contributed anything. The
    resampling is paired -- the same bootstrapped days feed both arms -- so the
    common market factor cancels instead of inflating the variance of the
    difference.
    """
    df = trades.dropna(subset=[return_col])
    if df.empty:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "p_value": np.nan}

    survive = apply_vetoes(df, vetoes)
    days = df[date_col].to_numpy()
    unique_days = np.unique(days)

    base_by_day, kept_by_day = {}, {}
    for d in unique_days:
        on_day = days == d
        base_by_day[d] = df.loc[on_day, return_col].to_numpy()
        kept_by_day[d] = df.loc[on_day & survive.to_numpy(), return_col].to_numpy()

    def delta(day_sample) -> float:
        base = np.concatenate([base_by_day[d] for d in day_sample])
        kept = np.concatenate([kept_by_day[d] for d in day_sample])
        if base.size == 0 or kept.size == 0:
            return np.nan
        return float(kept.mean() - base.mean())

    observed = delta(unique_days)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(unique_days, size=len(unique_days), replace=True)
        draws[b] = delta(picked)

    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    centred = draws - np.nanmean(draws)
    p = float(np.nanmean(np.abs(centred) >= abs(observed)))

    return {
        "delta": observed,
        "lo": float(lo),
        "hi": float(hi),
        "p_value": p,
        "helps": bool(lo > 0),
    }


def walk_forward_veto_selection(
    trades: pd.DataFrame,
    veto_cols: Sequence[str],
    folds: List[Fold],
    max_size: int = 3,
    min_retained: float = 0.05,
    min_trades: int = 100,
    return_col: str = "net_return",
    date_col: str = "date",
    selection_metric: str = "mean_net",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select a veto set on each fold's train days, score it on its test days.

    This is the only number in the study that may be quoted as an out-of-sample
    result. Everything produced by :func:`search_veto_sets` on the full panel is
    in-sample and exists for exploration and for honest reporting of the search
    size.

    Returns (per-fold summary, stitched out-of-sample trades).
    """
    per_fold: List[Dict] = []
    oos_frames: List[pd.DataFrame] = []

    for fold in folds:
        train, test = split_frame(trades, fold, date_col=date_col)
        if train.empty or test.empty:
            continue

        table = search_veto_sets(
            train, veto_cols, max_size=max_size, min_retained=min_retained,
            min_trades=min_trades, return_col=return_col, n_boot=400,
        )
        if table.empty:
            continue

        best = table.sort_values(selection_metric, ascending=False).iloc[0]
        chosen = tuple(v for v in best["vetoes"].split(", ") if v and v != "(none)")

        survive = apply_vetoes(test, chosen)
        oos = test[survive].copy()
        oos["fold"] = fold.index
        oos_frames.append(oos)

        test_res = evaluate_population(
            oos, name=best["name"], vetoes=chosen, return_col=return_col,
            n_boot=1_000, baseline_n=len(test.dropna(subset=[return_col])),
        )
        per_fold.append(
            {
                "fold": fold.index,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "chosen_vetoes": ", ".join(chosen) or "(none)",
                "train_mean": float(best[selection_metric]),
                "train_n": int(best["n_trades"]),
                "test_mean": test_res.mean_net,
                "test_ci_lo": test_res.ci_lo,
                "test_ci_hi": test_res.ci_hi,
                "test_n": test_res.n_trades,
                "test_retained": test_res.retained_frac,
                "test_sharpe": test_res.sharpe,
                "test_hit_rate": test_res.hit_rate,
                "degradation": float(best[selection_metric]) - test_res.mean_net,
            }
        )

    oos_all = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    return pd.DataFrame(per_fold), oos_all
