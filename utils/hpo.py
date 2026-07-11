"""hpo.py — Optuna hyperparameter search wired against configs/hpo_params.yaml.

Why walk-forward IC as the objective, not a single split
-----------------------------------------------------------
Tuning against a single train/test split (as RankingModelAgent does for
production fits) risks fitting hyperparameters to one lucky period. The HPO
objective instead maximizes mean out-of-sample IC across multiple
walk-forward folds (utils/walk_forward.py), penalized by the mean
train/test IC gap — a direct overfitting signal — so trials that look great
in-sample but generalize poorly score worse.

Search space format
--------------------
configs/hpo_params.yaml defines, per section (features/labeling/model/
portfolio/backtest), one entry per parameter with `type` (int/float/bool/
categorical), `distribution` (uniform/log_uniform, for int/float), and
`low`/`high` or `choices`. See that file for the authoritative schema and
the constraints enforced by `sample_config` below.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import optuna
import pandas as pd
import yaml

from agents.cs_feature_agent import CrossSectionalFeatureAgent
from agents.cs_labeling_agent import CrossSectionalLabelingAgent
from utils.logger import get_logger
from utils.walk_forward import walk_forward_validate

logger = get_logger(__name__)


def load_search_space(path: str) -> Dict[str, Any]:
    """Load the HPO parameter search space from a YAML file.

    Args:
        path: Path to a YAML file shaped like configs/hpo_params.yaml
              (top-level key "parameters", one sub-dict per pipeline section).

    Returns:
        Dict mapping section name -> {param_name: spec_dict}.
    """
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    return raw["parameters"]


def _sample_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    ptype = spec["type"]

    if ptype in ("categorical", "bool"):
        choices = spec["choices"]
        # optuna requires hashable choices; lists (e.g. [1, 5, 10]) aren't.
        hashable = [tuple(c) if isinstance(c, list) else c for c in choices]
        chosen = trial.suggest_categorical(name, hashable)
        return list(chosen) if isinstance(chosen, tuple) else chosen

    log_scale = spec.get("distribution", "uniform") == "log_uniform"
    if ptype == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=log_scale)
    if ptype == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=log_scale)

    raise ValueError(f"Unknown param type {ptype!r} for {name!r}")


def sample_config(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Sample one full pipeline config from the search space via an Optuna trial.

    Enforces the constraints documented in configs/hpo_params.yaml:
      entry_rank <= exit_rank
      max_positions <= entry_rank
    (Constraint violations are clamped rather than re-sampled, keeping the
    Optuna parameter distributions well-defined.)
    """
    config: Dict[str, Dict[str, Any]] = {}
    for section, params in search_space.items():
        config[section] = {
            name: _sample_param(trial, f"{section}.{name}", spec)
            for name, spec in params.items()
        }

    portfolio = config.get("portfolio", {})
    if "entry_rank" in portfolio and "exit_rank" in portfolio:
        portfolio["exit_rank"] = max(portfolio["exit_rank"], portfolio["entry_rank"])
    if "max_positions" in portfolio and "entry_rank" in portfolio:
        portfolio["max_positions"] = min(portfolio["max_positions"], portfolio["entry_rank"])

    return config


def build_objective(
    universe_data: Dict[str, pd.DataFrame],
    search_space: Dict[str, Any],
    n_splits: int = 3,
    overfit_penalty: float = 0.5,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective: walk-forward mean test IC, penalized by overfit gap.

    Purge days always equal the trial's sampled forward_period (matching
    MultiAssetOrchestrator's production convention) so training labels never
    overlap the test window regardless of what forward_period is sampled.
    """

    def objective(trial: optuna.Trial) -> float:
        config = sample_config(trial, search_space)
        f, lb, m = config["features"], config["labeling"], config["model"]

        ctx = {"universe_data": universe_data}
        ctx = CrossSectionalFeatureAgent(
            returns_windows=f["returns_windows"],
            vol_window=f["volatility_window"],
            rsi_period=f["rsi_period"],
            sma_windows=f["sma_windows"],
            cross_sectional=True,
        ).run(ctx)
        ctx = CrossSectionalLabelingAgent(forward_period=lb["forward_period"]).run(ctx)

        fold_metrics = walk_forward_validate(
            ctx["cs_features"],
            ctx["cs_labels"],
            n_splits=n_splits,
            purge_days=lb["forward_period"],
            n_estimators=m["n_estimators"],
            max_depth=m["max_depth"],
            learning_rate=m["learning_rate"],
        )

        mean_test_ic = float(fold_metrics["test_ic"].mean())
        mean_gap = float(fold_metrics["ic_gap"].abs().mean())
        score = mean_test_ic - overfit_penalty * mean_gap

        trial.set_user_attr("mean_test_ic", mean_test_ic)
        trial.set_user_attr("mean_ic_gap", mean_gap)
        return score

    return objective


def run_hpo(
    universe_data: Dict[str, pd.DataFrame],
    search_space: Dict[str, Any] | str,
    n_trials: int = 20,
    n_splits: int = 3,
    seed: int = 42,
    overfit_penalty: float = 0.5,
) -> optuna.Study:
    """Run Optuna HPO and return the completed study.

    Args:
        universe_data: Dict[ticker, OHLCV DataFrame] — same shape as
            context['universe_data'] elsewhere in the pipeline.
        search_space: Either a pre-loaded search-space dict or a path to a
            YAML file in the configs/hpo_params.yaml format.
        n_trials: Number of Optuna trials.
        n_splits: Walk-forward folds per trial.
        seed: Seed for the TPE sampler (deterministic given fixed n_trials).
        overfit_penalty: Weight on the train/test IC gap penalty term.
    """
    if isinstance(search_space, str):
        search_space = load_search_space(search_space)

    objective = build_objective(universe_data, search_space, n_splits=n_splits, overfit_penalty=overfit_penalty)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "HPO: %d trials complete | best_score=%.4f | best_params=%s",
        len(study.trials), study.best_value, study.best_params,
    )
    return study
