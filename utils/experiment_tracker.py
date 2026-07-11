"""experiment_tracker.py — structured per-run experiment log.

Why this exists
----------------
Before this, run results only surfaced via stdout logging and in-memory
context dicts — nothing persisted across runs, so configs could not be
compared over time. This appends one JSON line per run (config + metrics)
to a log file, and reads it back as a flat DataFrame for comparison.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Appends and reads structured run records (config + metrics) as JSONL."""

    def __init__(self, log_path: str = "outputs/experiments.jsonl") -> None:
        self.log_path = Path(log_path)

    def log_run(
        self,
        run_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append one run record and return it.

        Args:
            run_name: Human-readable label for this run.
            config: The full config used (or the subset worth comparing).
            metrics: Result metrics (e.g. sharpe, max_drawdown, ic).
            extra: Any additional metadata to attach (e.g. git commit, notes).
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "config": config,
            "metrics": metrics,
        }
        if extra:
            record["extra"] = extra

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        logger.info("ExperimentTracker: logged run %r to %s", run_name, self.log_path)
        return record

    def load_runs(self) -> pd.DataFrame:
        """Read all logged runs into a flat DataFrame, sorted by timestamp ascending.

        Nested config/metrics keys are flattened with dot-separated column
        names (e.g. "config.model.n_estimators", "metrics.sharpe").
        """
        if not self.log_path.exists():
            return pd.DataFrame()

        records = []
        with self.log_path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return pd.DataFrame()

        df = pd.json_normalize(records)
        return df.sort_values("timestamp").reset_index(drop=True)
