"""main_multi.py — multi-asset pipeline entry point.

Usage:
    python main_multi.py
    python main_multi.py --config configs/universe.yaml
"""
import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

from agents.multi_orchestrator import MultiAssetOrchestrator
from utils.logger import get_logger
from utils.visualizer import plot_multi_results

logger = get_logger(__name__)


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively replace ${VAR} placeholders with environment variables."""
    if isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r") as fh:
        return _resolve_env_vars(yaml.safe_load(fh))


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 55)
    print("  MULTI-ASSET EXPERIMENT RESULTS")
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


def main(config_path: str = "configs/universe.yaml") -> dict:
    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)

    tickers_cfg = config["universe"]["tickers"]
    universe_desc = tickers_cfg if isinstance(tickers_cfg, str) else f"{len(tickers_cfg)} tickers"
    logger.info(
        "Starting multi-asset experiment '%s'  universe=%s  benchmark=%s",
        config["experiment"]["name"],
        universe_desc,
        config["universe"]["benchmark"],
    )

    orch = MultiAssetOrchestrator(config)
    context = orch.run()

    print_metrics(context["multi_metrics"])
    plot_multi_results(context)
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantCode multi-asset pipeline runner")
    parser.add_argument(
        "--config",
        default="configs/universe.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
