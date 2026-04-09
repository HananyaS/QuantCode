"""Tests for CriticAgent — leakage detection and pipeline validation."""
import pandas as pd
import pytest

from agents.critic_agent import CriticAgent, DataLeakageError


def _model_info(train_end: str, test_start: str) -> dict:
    return {
        "estimator": None,
        "features": [],
        "train_start": "2020-01-01",
        "train_end": train_end,
        "test_start": test_start,
        "test_end": "2022-12-31",
    }


def _predictions(dates: pd.DatetimeIndex) -> dict:
    import numpy as np
    return {
        "values": pd.Series(1, index=dates, name="prediction"),
        "train_accuracy": 0.6,
        "test_accuracy": 0.55,
        "feature_importances": {},
    }


# ---------------------------------------------------------------------------
# Happy path — valid context passes without error
# ---------------------------------------------------------------------------

def test_valid_context_passes():
    train_dates = pd.bdate_range("2020-01-01", periods=200)  # noqa: F841
    test_dates = pd.bdate_range("2021-01-01", periods=50)
    ctx = {
        "model": _model_info("2020-12-31", "2021-01-01"),
        "predictions": _predictions(test_dates),
        "features": pd.DataFrame({"SMA_5": [1.0]}, index=[train_dates[0]]),
        "metrics": {"sharpe": 0.5, "max_drawdown": -0.1, "annualized_return": 0.08},
    }
    result = CriticAgent().run(ctx)
    assert result is ctx  # returns same context


def test_valid_context_no_model_passes():
    """CriticAgent must be tolerant of missing optional keys."""
    ctx = {"model": None, "predictions": None, "features": None, "metrics": None}
    CriticAgent().run(ctx)  # no exception


# ---------------------------------------------------------------------------
# Temporal leakage checks
# ---------------------------------------------------------------------------

def test_raises_when_train_end_equals_test_start():
    test_dates = pd.bdate_range("2021-06-01", periods=50)
    ctx = {
        "model": _model_info("2021-06-01", "2021-06-01"),  # same date — leakage
        "predictions": _predictions(test_dates),
        "features": None,
        "metrics": None,
    }
    with pytest.raises(DataLeakageError, match="[Tt]emporal"):
        CriticAgent().run(ctx)


def test_raises_when_train_end_after_test_start():
    test_dates = pd.bdate_range("2021-01-01", periods=50)
    ctx = {
        "model": _model_info("2021-06-01", "2021-01-01"),  # train_end > test_start
        "predictions": _predictions(test_dates),
        "features": None,
        "metrics": None,
    }
    with pytest.raises(DataLeakageError):
        CriticAgent().run(ctx)


# ---------------------------------------------------------------------------
# Label-in-features check
# ---------------------------------------------------------------------------

def test_raises_when_label_in_features():
    import pandas as pd
    dates = pd.bdate_range("2020-01-01", periods=5)
    ctx = {
        "model": None,
        "predictions": None,
        "features": pd.DataFrame({"SMA_5": [1.0] * 5, "label": [0] * 5}, index=dates),
        "metrics": None,
    }
    with pytest.raises(DataLeakageError, match="[Ll]abel"):
        CriticAgent().run(ctx)


# ---------------------------------------------------------------------------
# Predictions-overlap-train check
# ---------------------------------------------------------------------------

def test_raises_when_predictions_in_train_period():
    # predictions start BEFORE or ON train_end
    train_dates = pd.bdate_range("2020-01-01", "2021-12-31")
    test_dates = pd.bdate_range("2021-06-01", periods=30)  # inside training window
    ctx = {
        "model": _model_info("2021-12-31", "2022-01-01"),
        "predictions": _predictions(test_dates),
        "features": None,
        "metrics": None,
    }
    with pytest.raises(DataLeakageError, match="[Pp]rediction"):
        CriticAgent().run(ctx)


# ---------------------------------------------------------------------------
# Missing metrics
# ---------------------------------------------------------------------------

def test_raises_when_required_metric_missing():
    ctx = {
        "model": None,
        "predictions": None,
        "features": None,
        "metrics": {"sharpe": 0.5},  # missing max_drawdown and annualized_return
    }
    with pytest.raises(AssertionError, match="[Mm]issing"):
        CriticAgent().run(ctx)


def test_passes_with_all_required_metrics():
    ctx = {
        "model": None,
        "predictions": None,
        "features": None,
        "metrics": {
            "sharpe": 0.5,
            "max_drawdown": -0.1,
            "annualized_return": 0.08,
        },
    }
    CriticAgent().run(ctx)  # no exception
