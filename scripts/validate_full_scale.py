"""validate_full_scale.py — the real Phase 1 exit-criterion check.

scripts/measure_strategy.py is a fast 80-ticker/50-estimator proxy built for
rapid autoresearch iteration. This is the actual validation that matters
before trusting a config: walk-forward IC against the FULL cached S&P 500
universe with the production model size (n_estimators from
configs/universe.yaml, currently 300), using whatever feature/labeling
config is currently in configs/universe.yaml.

Never hits the network — relies entirely on the local parquet cache
(data/cache/), which already covers the full current S&P 500 universe.
"""
from __future__ import annotations

import os
import sys

# Allow running as `python scripts/validate_full_scale.py` (project root not
# auto-added to sys.path in that case, unlike root-level scripts).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from agents.cs_feature_agent import CrossSectionalFeatureAgent
from agents.cs_labeling_agent import CrossSectionalLabelingAgent
from utils.data_cache import load_or_fetch
from utils.walk_forward import walk_forward_validate

_CACHE_DIR = Path("data/cache")


def _no_network_fetch(missing, start, end):
    """Never hits the network — the Alpaca paper-account credentials in
    .env are returning 401s (stale/regenerated keys, unrelated to this
    validation), so this run is scoped to whatever is already fully cached
    rather than debugging auth. Any ticker missing full coverage is simply
    dropped from the universe for this run."""
    return {}


def main() -> None:
    with open("configs/universe.yaml") as fh:
        cfg = yaml.safe_load(fh)
    u, f, lb, m = cfg["universe"], cfg["features"], cfg["labeling"], cfg["model"]

    # Load directly from whatever's already cached rather than resolving
    # "sp500" live (Wikipedia + Alpaca) — sidesteps both the network
    # dependency and the current Alpaca auth issue entirely.
    benchmark = u.get("benchmark", "SPY")
    candidates = sorted(p.stem for p in _CACHE_DIR.glob("*.parquet") if p.stem != benchmark)
    # Buffered on both ends: cache starts 2016-01-04 and the last bulk fetch
    # landed on 2024-12-30 (not 2024-12-31, config's declared end_date) —
    # _covers() requires the FULL range covered per-ticker, so an exact
    # boundary date silently drops every ticker that's a day short.
    start_date = "2016-06-01"
    end_date = "2024-12-20"
    print(f"Loading {len(candidates)} cached tickers ({start_date} to {end_date})...")
    universe_data = load_or_fetch(
        candidates, start_date, end_date, _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    n_tickers = len(universe_data)
    assert n_tickers > 0, (
        f"0/{len(candidates)} tickers had full cache coverage for "
        f"[{start_date}, {end_date}] — check the requested range against "
        f"actual cached date bounds (e.g. inspect data/cache/AAPL.parquet)"
    )
    n_days = len(next(iter(universe_data.values())))
    print(f"Loaded {n_tickers} tickers x {n_days} trading days "
          f"({len(candidates) - n_tickers} dropped for incomplete coverage)")

    ctx = {"universe_data": universe_data}

    print(f"Computing features (returns={f['returns']}, sma={f['sma_windows']}, "
          f"vol_window={f['volatility_window']}, rsi_period={f['rsi_period']})...")
    ctx = CrossSectionalFeatureAgent(
        returns_windows=f["returns"],
        vol_window=f["volatility_window"],
        rsi_period=f["rsi_period"],
        sma_windows=f["sma_windows"],
        cross_sectional=True,
    ).run(ctx)

    print(f"Computing labels (forward_period={lb['forward_period']})...")
    ctx = CrossSectionalLabelingAgent(forward_period=lb["forward_period"]).run(ctx)

    print(f"Running walk-forward validation (n_splits=5, n_estimators={m['n_estimators']}, "
          f"max_depth={m['max_depth']}, learning_rate={m['learning_rate']})...")
    result = walk_forward_validate(
        ctx["cs_features"], ctx["cs_labels"],
        n_splits=5,
        purge_days=lb["forward_period"],
        n_estimators=m["n_estimators"],
        max_depth=m["max_depth"],
        learning_rate=m["learning_rate"],
        random_state=m["random_state"],
    )

    print("\n" + result.to_string(index=False))
    mean_test_ic = result["test_ic"].mean()
    mean_ic_gap = result["ic_gap"].abs().mean()
    n_positive = (result["test_ic"] > 0).sum()
    print(f"\nmean_test_ic={mean_test_ic:.4f}  mean_ic_gap={mean_ic_gap:.4f}  "
          f"positive_folds={n_positive}/{len(result)}")


if __name__ == "__main__":
    main()
