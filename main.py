"""main.py — pipeline entry point.

Usage:
    python main.py
    python main.py --config configs/experiment.yaml
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import yaml

from agents.orchestrator import Orchestrator
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """Load and return the YAML experiment config."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r") as fh:
        return yaml.safe_load(fh)


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print the final evaluation metrics."""
    print("\n" + "=" * 55)
    print("  EXPERIMENT RESULTS")
    print("=" * 55)
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_val in value.items():
                _print_metric(f"    {sub_key}", sub_val)
        else:
            _print_metric(f"  {key}", value)
    print("=" * 55 + "\n")


def _print_metric(label: str, value: Any) -> None:
    if isinstance(value, float):
        print(f"{label:<30} {value:>10.4f}")
    else:
        print(f"{label:<30} {value!s:>10}")


def main(config_path: str = "configs/experiment.yaml") -> dict:
    """Run the full pipeline and return the final context."""
    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)

    logger.info(
        "Starting experiment '%s'  ticker=%s  benchmark=%s",
        config["experiment"]["name"],
        config["data"]["ticker"],
        config["data"]["benchmark"],
    )

    orch = Orchestrator(config)
    context = orch.run()

    print_metrics(context["metrics"])
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant AI Lab pipeline runner")
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
