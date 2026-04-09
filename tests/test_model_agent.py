"""Tests for ModelAgent — walk-forward split, determinism, no temporal leakage."""
import numpy as np
import pandas as pd
import pytest

from agents.model_agent import ModelAgent


def _make_context(features, labels):
    return {"features": features, "labels": labels}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_model_and_predictions_stored(features_and_labels):
    features, labels = features_and_labels
    ctx = ModelAgent().run(_make_context(features, labels))
    assert ctx["model"] is not None
    assert ctx["predictions"] is not None


def test_predictions_are_series(features_and_labels):
    features, labels = features_and_labels
    ctx = ModelAgent().run(_make_context(features, labels))
    assert isinstance(ctx["predictions"]["values"], pd.Series)


def test_predictions_binary(features_and_labels):
    features, labels = features_and_labels
    ctx = ModelAgent().run(_make_context(features, labels))
    vals = ctx["predictions"]["values"]
    assert set(vals.unique()).issubset({0, 1})


def test_train_end_before_test_start(features_and_labels):
    """Core no-leakage guarantee: train must end before test begins."""
    features, labels = features_and_labels
    ctx = ModelAgent().run(_make_context(features, labels))
    model_info = ctx["model"]
    train_end = pd.Timestamp(model_info["train_end"])
    test_start = pd.Timestamp(model_info["test_start"])
    assert train_end < test_start


def test_accuracy_keys_present(features_and_labels):
    features, labels = features_and_labels
    ctx = ModelAgent().run(_make_context(features, labels))
    preds = ctx["predictions"]
    assert "train_accuracy" in preds
    assert "test_accuracy" in preds


def test_accuracy_in_range(features_and_labels):
    features, labels = features_and_labels
    ctx = ModelAgent().run(_make_context(features, labels))
    preds = ctx["predictions"]
    assert 0.0 <= preds["train_accuracy"] <= 1.0
    assert 0.0 <= preds["test_accuracy"] <= 1.0


def test_feature_importances_present(features_and_labels):
    features, labels = features_and_labels
    ctx = ModelAgent().run(_make_context(features, labels))
    fi = ctx["predictions"]["feature_importances"]
    assert isinstance(fi, dict)
    assert len(fi) == len(features.columns)


def test_feature_importances_sum_to_one(features_and_labels):
    features, labels = features_and_labels
    ctx = ModelAgent().run(_make_context(features, labels))
    fi = ctx["predictions"]["feature_importances"]
    assert abs(sum(fi.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_predictions(features_and_labels):
    """Two runs with the same seed must produce identical predictions."""
    features, labels = features_and_labels
    ctx1 = ModelAgent(random_state=42).run(_make_context(features, labels))
    ctx2 = ModelAgent(random_state=42).run(_make_context(features, labels))
    pd.testing.assert_series_equal(
        ctx1["predictions"]["values"],
        ctx2["predictions"]["values"],
    )


def test_different_seeds_may_differ(features_and_labels):
    """Different seeds should (with high probability) produce different predictions."""
    features, labels = features_and_labels
    ctx1 = ModelAgent(random_state=0).run(_make_context(features, labels))
    ctx2 = ModelAgent(random_state=99).run(_make_context(features, labels))
    # Not guaranteed to differ, but extremely likely on real data
    vals1 = ctx1["predictions"]["values"].values
    vals2 = ctx2["predictions"]["values"].values
    # Just check both ran successfully — we can't assert they differ
    assert len(vals1) == len(vals2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_small_test_size_raises(features_and_labels):
    """If test_size would yield < MIN_TEST_ROWS, raise AssertionError."""
    features, labels = features_and_labels
    # With 500 rows and test_size=0.001 we get ~0 test rows
    with pytest.raises(AssertionError, match="[Tt]est set too small"):
        ModelAgent(test_size=0.001).run(_make_context(features, labels))


def test_missing_features_raises():
    ctx = {"features": None, "labels": pd.Series([0, 1])}
    with pytest.raises(AssertionError):
        ModelAgent().run(ctx)


def test_missing_labels_raises(features_and_labels):
    features, _ = features_and_labels
    ctx = {"features": features, "labels": None}
    with pytest.raises(AssertionError):
        ModelAgent().run(ctx)
