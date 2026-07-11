"""Tests for utils/experiment_tracker.py — structured per-run experiment log."""
import json

import pandas as pd
import pytest

from utils.experiment_tracker import ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    return ExperimentTracker(log_path=str(tmp_path / "experiments.jsonl"))


# ---------------------------------------------------------------------------
# log_run
# ---------------------------------------------------------------------------

def test_log_run_creates_file(tracker):
    tracker.log_run("run1", config={"model": {"n_estimators": 100}}, metrics={"sharpe": 1.2})
    assert tracker.log_path.exists()


def test_log_run_appends_valid_json_line(tracker):
    tracker.log_run("run1", config={"a": 1}, metrics={"sharpe": 1.2})
    lines = tracker.log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_name"] == "run1"
    assert record["config"] == {"a": 1}
    assert record["metrics"] == {"sharpe": 1.2}
    assert "timestamp" in record


def test_log_run_appends_multiple_runs(tracker):
    tracker.log_run("run1", config={"a": 1}, metrics={"sharpe": 1.0})
    tracker.log_run("run2", config={"a": 2}, metrics={"sharpe": 2.0})
    lines = tracker.log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_log_run_returns_record(tracker):
    record = tracker.log_run("run1", config={"a": 1}, metrics={"sharpe": 1.0})
    assert record["run_name"] == "run1"
    assert "timestamp" in record


# ---------------------------------------------------------------------------
# load_runs
# ---------------------------------------------------------------------------

def test_load_runs_empty_file_returns_empty_dataframe(tracker):
    df = tracker.load_runs()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_load_runs_returns_flattened_dataframe(tracker):
    tracker.log_run("run1", config={"model": {"n_estimators": 100}}, metrics={"sharpe": 1.2, "max_drawdown": -0.1})
    tracker.log_run("run2", config={"model": {"n_estimators": 200}}, metrics={"sharpe": 1.5, "max_drawdown": -0.2})
    df = tracker.load_runs()
    assert len(df) == 2
    assert "config.model.n_estimators" in df.columns
    assert "metrics.sharpe" in df.columns
    assert list(df["metrics.sharpe"]) == [1.2, 1.5]


def test_load_runs_sorted_by_timestamp_ascending(tracker):
    tracker.log_run("run1", config={}, metrics={"sharpe": 1.0})
    tracker.log_run("run2", config={}, metrics={"sharpe": 2.0})
    df = tracker.load_runs()
    assert list(df["run_name"]) == ["run1", "run2"]
