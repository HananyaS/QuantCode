"""Tests for RankingModelAgent."""
import numpy as np
import pandas as pd
import pytest

from agents.ranking_model_agent import RankingModelAgent


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_context_keys_written(cs_features_and_labels):
    features, labels = cs_features_and_labels
    agent = RankingModelAgent(n_estimators=10, random_state=42, test_size=0.2)
    ctx = agent.run({"cs_features": features, "cs_labels": labels})
    assert ctx["cs_model"] is not None
    assert ctx["cs_predictions"] is not None


def test_predictions_are_series_with_multiindex(cs_features_and_labels):
    features, labels = cs_features_and_labels
    agent = RankingModelAgent(n_estimators=10, random_state=42, test_size=0.2)
    ctx = agent.run({"cs_features": features, "cs_labels": labels})
    preds = ctx["cs_predictions"]
    assert isinstance(preds, pd.Series)
    assert isinstance(preds.index, pd.MultiIndex)
    assert preds.index.names == ["date", "ticker"]


def test_model_record_has_expected_keys(cs_features_and_labels):
    features, labels = cs_features_and_labels
    agent = RankingModelAgent(n_estimators=10, random_state=42)
    ctx = agent.run({"cs_features": features, "cs_labels": labels})
    model_record = ctx["cs_model"]
    for key in ("estimator", "features", "train_start", "train_end",
                "test_start", "test_end", "ic"):
        assert key in model_record, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def test_no_temporal_leakage(cs_features_and_labels):
    """All prediction dates must be strictly after train_end."""
    features, labels = cs_features_and_labels
    agent = RankingModelAgent(n_estimators=10, random_state=42)
    ctx = agent.run({"cs_features": features, "cs_labels": labels})

    model_record = ctx["cs_model"]
    train_end = pd.Timestamp(model_record["train_end"])
    pred_dates = ctx["cs_predictions"].index.get_level_values("date")
    assert (pred_dates > train_end).all(), (
        "Some prediction dates fall within or before the training window"
    )


def test_train_end_before_test_start(cs_features_and_labels):
    features, labels = cs_features_and_labels
    agent = RankingModelAgent(n_estimators=10, random_state=42)
    ctx = agent.run({"cs_features": features, "cs_labels": labels})
    m = ctx["cs_model"]
    assert pd.Timestamp(m["train_end"]) < pd.Timestamp(m["test_start"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_output(cs_features_and_labels):
    features, labels = cs_features_and_labels
    agent = RankingModelAgent(n_estimators=10, random_state=42)
    ctx1 = agent.run({"cs_features": features, "cs_labels": labels})
    ctx2 = agent.run({"cs_features": features, "cs_labels": labels})
    pd.testing.assert_series_equal(ctx1["cs_predictions"], ctx2["cs_predictions"])


# ---------------------------------------------------------------------------
# IC sanity
# ---------------------------------------------------------------------------

def test_ic_is_finite(cs_features_and_labels):
    features, labels = cs_features_and_labels
    agent = RankingModelAgent(n_estimators=10, random_state=42)
    ctx = agent.run({"cs_features": features, "cs_labels": labels})
    assert np.isfinite(ctx["cs_model"]["ic"])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_raises_if_features_missing(cs_features_and_labels):
    _, labels = cs_features_and_labels
    agent = RankingModelAgent(n_estimators=5)
    with pytest.raises(AssertionError, match="cs_features"):
        agent.run({"cs_labels": labels})


def test_raises_if_labels_missing(cs_features_and_labels):
    features, _ = cs_features_and_labels
    agent = RankingModelAgent(n_estimators=5)
    with pytest.raises(AssertionError, match="cs_labels"):
        agent.run({"cs_features": features})
