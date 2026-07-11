"""run_full_hpo.py — Optuna HPO against the full cached universe.

Follow-on to scripts/validate_full_scale.py: that run confirmed a real (if
weak) out-of-sample signal (mean_test_ic=0.0081, 4/5 positive folds) with a
substantial train/test overfitting gap (mean_ic_gap=0.0533). This searches
the full feature/model/labeling space (configs/hpo_params.yaml) at full
scale via utils.hpo.run_hpo to find a config that closes that gap —
n_splits=3 during search (vs. 5 for final validation) to keep per-trial
cost down. The winning config is then re-validated at n_splits=5 for a
clean, comparable final report.

Never hits the network — loads directly from data/cache/, same approach as
scripts/validate_full_scale.py (buffered date range; Alpaca fallback
disabled since the .env paper-account keys are currently returning 401s).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from dotenv import load_dotenv

load_dotenv()
optuna.logging.set_verbosity(optuna.logging.WARNING)

from agents.cs_feature_agent import CrossSectionalFeatureAgent
from agents.cs_labeling_agent import CrossSectionalLabelingAgent
from utils.data_cache import load_or_fetch
from utils.hpo import run_hpo
from utils.walk_forward import walk_forward_validate

_CACHE_DIR = Path("data/cache")
_N_TRIALS = 15
_SEARCH_SPLITS = 3
_FINAL_SPLITS = 5


def _no_network_fetch(missing, start, end):
    return {}


def load_universe(benchmark: str = "SPY", start: str = "2016-06-01", end: str = "2024-12-20") -> dict:
    candidates = sorted(p.stem for p in _CACHE_DIR.glob("*.parquet") if p.stem != benchmark)
    universe_data = load_or_fetch(candidates, start, end, _no_network_fetch, cache_dir=_CACHE_DIR)
    assert len(universe_data) > 0, "0 tickers loaded — check cache date-range coverage"
    return universe_data


def _unflatten(flat_params: dict) -> dict:
    """Reconstruct {section: {name: value}} from Optuna's 'section.name' keys,
    converting tuple-valued categoricals (e.g. returns_windows) back to lists."""
    nested: dict = {}
    for key, value in flat_params.items():
        section, name = key.split(".", 1)
        nested.setdefault(section, {})[name] = list(value) if isinstance(value, tuple) else value
    return nested


def main() -> None:
    universe_data = load_universe()
    print(f"Loaded {len(universe_data)} tickers")

    print(f"Running Optuna HPO: {_N_TRIALS} trials, n_splits={_SEARCH_SPLITS} (search phase)...")
    study = run_hpo(universe_data, "configs/hpo_params.yaml", n_trials=_N_TRIALS, n_splits=_SEARCH_SPLITS, seed=42)

    print(f"\nBest trial #{study.best_trial.number}: score={study.best_value:.4f}")
    print(f"  mean_test_ic={study.best_trial.user_attrs.get('mean_test_ic'):.4f}  "
          f"mean_ic_gap={study.best_trial.user_attrs.get('mean_ic_gap'):.4f}")
    print(f"  params={study.best_params}")

    trials_sorted = sorted(
        (t for t in study.trials if t.value is not None), key=lambda t: t.value, reverse=True
    )
    print("\nTop 5 trials:")
    for t in trials_sorted[:5]:
        print(f"  #{t.number}: score={t.value:.4f}  "
              f"mean_test_ic={t.user_attrs.get('mean_test_ic'):.4f}  "
              f"mean_ic_gap={t.user_attrs.get('mean_ic_gap'):.4f}")

    # Re-validate the winner at full rigor (n_splits=5), matching
    # validate_full_scale.py's setup for a direct comparison.
    print(f"\nRe-validating best config at n_splits={_FINAL_SPLITS}...")
    best = _unflatten(study.best_params)
    ctx = {"universe_data": universe_data}
    ctx = CrossSectionalFeatureAgent(
        returns_windows=best["features"]["returns_windows"],
        vol_window=best["features"]["volatility_window"],
        rsi_period=best["features"]["rsi_period"],
        sma_windows=best["features"]["sma_windows"],
        cross_sectional=True,
    ).run(ctx)
    ctx = CrossSectionalLabelingAgent(forward_period=best["labeling"]["forward_period"]).run(ctx)
    final = walk_forward_validate(
        ctx["cs_features"], ctx["cs_labels"],
        n_splits=_FINAL_SPLITS,
        purge_days=best["labeling"]["forward_period"],
        n_estimators=best["model"]["n_estimators"],
        max_depth=best["model"]["max_depth"],
        learning_rate=best["model"]["learning_rate"],
        random_state=42,
    )
    print("\n" + final.to_string(index=False))
    print(f"\nFINAL: mean_test_ic={final['test_ic'].mean():.4f}  "
          f"mean_ic_gap={final['ic_gap'].abs().mean():.4f}  "
          f"positive_folds={(final['test_ic'] > 0).sum()}/{len(final)}")
    print(f"\nWinning feature/labeling/model config: {best}")


if __name__ == "__main__":
    main()
