"""Tests for utils/hpo.py — Optuna HPO wired against configs/hpo_params.yaml."""
import numpy as np
import optuna
import pytest

from utils.hpo import build_objective, load_search_space, run_hpo, sample_config

optuna.logging.set_verbosity(optuna.logging.WARNING)

_REAL_SEARCH_SPACE_PATH = "configs/hpo_params.yaml"

# A tiny search space for fast objective/run_hpo tests (the real
# configs/hpo_params.yaml ranges up to n_estimators=600 which is too slow
# for unit tests).
_TINY_SEARCH_SPACE = {
    "features": {
        "returns_windows": {"type": "categorical", "choices": [[1, 5], [1, 5, 10]]},
        "volatility_window": {"type": "int", "distribution": "uniform", "low": 5, "high": 10},
        "rsi_period": {"type": "int", "distribution": "uniform", "low": 5, "high": 8},
        "sma_windows": {"type": "categorical", "choices": [[5, 10], [10, 20]]},
    },
    "labeling": {
        "forward_period": {"type": "int", "distribution": "uniform", "low": 3, "high": 5},
    },
    "model": {
        "n_estimators": {"type": "int", "distribution": "uniform", "low": 5, "high": 10},
        "max_depth": {"type": "int", "distribution": "uniform", "low": 2, "high": 3},
        "learning_rate": {"type": "float", "distribution": "log_uniform", "low": 0.05, "high": 0.2},
    },
    "portfolio": {
        "max_positions": {"type": "int", "distribution": "uniform", "low": 2, "high": 4},
        "entry_rank": {"type": "int", "distribution": "uniform", "low": 2, "high": 3},
        "exit_rank": {"type": "int", "distribution": "uniform", "low": 3, "high": 5},
        "score_weighting": {"type": "bool", "choices": [True, False]},
    },
}


# ---------------------------------------------------------------------------
# load_search_space
# ---------------------------------------------------------------------------

def test_load_search_space_has_all_sections():
    space = load_search_space(_REAL_SEARCH_SPACE_PATH)
    for section in ("features", "labeling", "model", "portfolio", "backtest"):
        assert section in space


# ---------------------------------------------------------------------------
# sample_config
# ---------------------------------------------------------------------------

def _new_trial():
    study = optuna.create_study(direction="maximize")
    return study.ask()


def test_sample_config_respects_int_range():
    trial = _new_trial()
    config = sample_config(trial, _TINY_SEARCH_SPACE)
    assert 5 <= config["model"]["n_estimators"] <= 10


def test_sample_config_categorical_returns_list_not_tuple():
    trial = _new_trial()
    config = sample_config(trial, _TINY_SEARCH_SPACE)
    assert isinstance(config["features"]["returns_windows"], list)


def test_sample_config_enforces_entry_rank_le_exit_rank():
    for _ in range(20):
        trial = _new_trial()
        config = sample_config(trial, _TINY_SEARCH_SPACE)
        assert config["portfolio"]["entry_rank"] <= config["portfolio"]["exit_rank"]


def test_sample_config_enforces_max_positions_le_entry_rank():
    for _ in range(20):
        trial = _new_trial()
        config = sample_config(trial, _TINY_SEARCH_SPACE)
        assert config["portfolio"]["max_positions"] <= config["portfolio"]["entry_rank"]


# ---------------------------------------------------------------------------
# build_objective / run_hpo
# ---------------------------------------------------------------------------

def test_build_objective_returns_finite_float(universe_data):
    objective = build_objective(universe_data, _TINY_SEARCH_SPACE, n_splits=2)
    trial = _new_trial()
    value = objective(trial)
    assert np.isfinite(value)


def test_run_hpo_returns_study_with_best_trial(universe_data):
    study = run_hpo(
        universe_data,
        search_space=_TINY_SEARCH_SPACE,
        n_trials=3,
        n_splits=2,
        seed=42,
    )
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 3
    assert study.best_trial is not None


def test_run_hpo_deterministic_with_seed(universe_data):
    study1 = run_hpo(universe_data, search_space=_TINY_SEARCH_SPACE, n_trials=3, n_splits=2, seed=7)
    study2 = run_hpo(universe_data, search_space=_TINY_SEARCH_SPACE, n_trials=3, n_splits=2, seed=7)
    assert study1.best_value == pytest.approx(study2.best_value)
