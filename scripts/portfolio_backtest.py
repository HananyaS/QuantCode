"""portfolio_backtest.py — real portfolio-level historical performance.

scripts/validate_full_scale.py and scripts/run_full_hpo.py both measure raw
walk-forward ranking IC, which does NOT directly translate to portfolio
economics — position sizing, trailing stops, rank-based entry/exit, and
transaction costs all matter and none of them have been exercised yet for
either candidate config.

This stitches together genuinely out-of-sample predictions across all 5
walk-forward folds into ONE continuous prediction series (no lookahead —
each fold's test predictions came only from a model trained strictly
before that period), then runs that through the real
PortfolioAgent -> MultiAssetBacktestAgent -> MultiAssetEvaluationAgent
chain for an honest Sharpe/drawdown/turnover number, benchmarked against
SPY, over the full ~2017-2024 out-of-sample window.

Usage:
    python scripts/portfolio_backtest.py --config a
    python scripts/portfolio_backtest.py --config b

Never hits the network — loads directly from data/cache/, same approach as
scripts/validate_full_scale.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.cs_feature_agent import CrossSectionalFeatureAgent
from agents.cs_labeling_agent import CrossSectionalLabelingAgent
from agents.multi_backtest_agent import MultiAssetBacktestAgent
from agents.multi_evaluation_agent import MultiAssetEvaluationAgent
from agents.portfolio_agent import PortfolioAgent
from utils.data_cache import load_or_fetch
from utils.walk_forward import _fit, generate_folds

_CACHE_DIR = Path("data/cache")
_N_SPLITS = 5
_START, _END = "2016-06-01", "2024-12-20"  # cache-safe buffer, see validate_full_scale.py

# Config A: autoresearch-converged (tag research/config-a-autoresearch-baseline)
_CONFIG_A = {
    "features": {"returns_windows": [1, 5], "vol_window": 20, "rsi_period": 14, "sma_windows": [20]},
    "labeling": {"forward_period": 3},
    "model": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
}

# Config B: HPO-found (scripts/run_full_hpo.py output) — feature/labeling/model
# only; portfolio/backtest params from that search were unvalidated and one
# was inconsistent (entry_rank > exit_rank), so this uses the same sane,
# manually-chosen portfolio params as Config A for a fair comparison.
_CONFIG_B = {
    "features": {"returns_windows": [1, 5, 10], "vol_window": 29, "rsi_period": 19, "sma_windows": [10, 50]},
    "labeling": {"forward_period": 20},
    "model": {"n_estimators": 136, "max_depth": 2, "learning_rate": 0.005737078437736713},
}

_PORTFOLIO = dict(
    max_positions=10, entry_rank=10, exit_rank=30, min_score=0.0,
    trailing_stop_atr_mult=1.5, atr_period=14, score_weighting=False,
)
_BACKTEST = dict(initial_capital=100_000.0, transaction_cost=0.001)


def _no_network_fetch(missing, start, end):
    return {}


def load_universe(benchmark: str = "SPY") -> tuple[dict, pd.DataFrame]:
    candidates = sorted(p.stem for p in _CACHE_DIR.glob("*.parquet") if p.stem != benchmark)
    universe_data = load_or_fetch(candidates, _START, _END, _no_network_fetch, cache_dir=_CACHE_DIR)
    assert len(universe_data) > 0, "0 tickers loaded — check cache date-range coverage"
    bm = load_or_fetch([benchmark], _START, _END, _no_network_fetch, cache_dir=_CACHE_DIR)
    assert benchmark in bm, f"{benchmark} not found in cache for [{_START}, {_END}]"
    return universe_data, bm[benchmark]


def stitched_out_of_sample_predictions(cs_features, cs_labels, model_cfg, n_splits=_N_SPLITS, purge_days=None) -> pd.Series:
    """Fit a model per walk-forward fold, predict on that fold's test dates
    only, and concatenate all folds' test predictions into one continuous,
    non-overlapping, chronologically-ordered series — a genuine
    out-of-sample simulation with no lookahead at any point."""
    combined = cs_features.join(cs_labels, how="inner").dropna()
    X = combined.drop(columns=["label"])
    y = combined["label"]
    dates = X.index.get_level_values("date")

    folds = generate_folds(dates, n_splits=n_splits, purge_days=purge_days)
    all_preds = []
    for fold in folds:
        train_mask = dates.isin(fold.train_dates)
        test_mask = dates.isin(fold.test_dates)
        model = _fit(
            X[train_mask], y[train_mask],
            model_type="lightgbm",
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            learning_rate=model_cfg["learning_rate"],
            random_state=42,
        )
        preds = pd.Series(model.predict(X[test_mask].values), index=X[test_mask].index, name="score")
        all_preds.append(preds)

    return pd.concat(all_preds).sort_index()


def run(config_name: str, cfg: dict) -> None:
    universe_data, benchmark_data = load_universe()
    print(f"[{config_name}] Loaded {len(universe_data)} tickers")

    ctx = {"universe_data": universe_data}
    ctx = CrossSectionalFeatureAgent(
        returns_windows=cfg["features"]["returns_windows"],
        vol_window=cfg["features"]["vol_window"],
        rsi_period=cfg["features"]["rsi_period"],
        sma_windows=cfg["features"]["sma_windows"],
        cross_sectional=True,
    ).run(ctx)
    ctx = CrossSectionalLabelingAgent(forward_period=cfg["labeling"]["forward_period"]).run(ctx)

    print(f"[{config_name}] Fitting {_N_SPLITS} walk-forward folds and stitching out-of-sample predictions...")
    cs_predictions = stitched_out_of_sample_predictions(
        ctx["cs_features"], ctx["cs_labels"], cfg["model"],
        n_splits=_N_SPLITS, purge_days=cfg["labeling"]["forward_period"],
    )
    test_dates = cs_predictions.index.get_level_values("date")
    print(f"[{config_name}] Out-of-sample window: {test_dates.min().date()} to {test_dates.max().date()} "
          f"({test_dates.nunique()} trading days)")

    ctx["cs_predictions"] = cs_predictions
    ctx = PortfolioAgent(**_PORTFOLIO).run(ctx)
    ctx = MultiAssetBacktestAgent(**_BACKTEST).run(ctx)
    ctx["benchmark_data"] = benchmark_data
    ctx = MultiAssetEvaluationAgent().run(ctx)

    m = ctx["multi_metrics"]
    print(f"\n=== [{config_name}] Portfolio-level results ===")
    print(f"  Sharpe:              {m['sharpe']:.3f}")
    print(f"  Max drawdown:        {m['max_drawdown']:.2%}")
    print(f"  Annualized return:   {m['annualized_return']:.2%}")
    print(f"  Final equity:        ${m['final_equity']:,.2f}  (from ${_BACKTEST['initial_capital']:,.0f})")
    print(f"  Avg daily turnover:  {m['avg_daily_turnover']:.2%}")
    print(f"  N rebalances:        {m['n_rebalances']}")
    if "benchmark" in m:
        b = m["benchmark"]
        print(f"  --- SPY benchmark ---")
        print(f"  SPY Sharpe:          {b['sharpe']:.3f}")
        print(f"  SPY max drawdown:    {b['max_drawdown']:.2%}")
        print(f"  SPY annualized ret:  {b['annualized_return']:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()

    if args.config in ("a", "both"):
        run("Config A", _CONFIG_A)
    if args.config in ("b", "both"):
        run("Config B", _CONFIG_B)


if __name__ == "__main__":
    main()
