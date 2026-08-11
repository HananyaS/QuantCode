"""sweep_barriers.py — evaluate several barrier geometries.

Barrier geometry cannot be tuned by the ordinary loop: it is baked into the
labels, so each candidate requires a full dataset rebuild. This runs that sweep
explicitly.

Why it is likely the dominant lever
-----------------------------------
A +30%/-10% geometry needs a 25% win rate merely to break even *before* costs.
The observed rate is 23.6%. Lowering the target raises the hit rate but shrinks
the payoff, so there is a genuine optimum to find rather than a monotone
improvement to chase -- and it may well be that no geometry clears costs, which
is itself the answer.

Each geometry is scored by the same out-of-sample objective as the tuning loop:
the lower bound of the 95% CI on out-of-sample mean net return.

    python research/penny/scripts/sweep_barriers.py
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep")

PROJECT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "tuning_params.json"
PYTHON = sys.executable

# Reward/risk pairs spanning the plausible range. The break-even win rate for a
# target t and stop s (ignoring costs) is s / (t + s).
GEOMETRIES: List[Tuple[float, float]] = [
    (0.30, 0.10),   # break-even 25.0%  -- the current default
    (0.20, 0.10),   # break-even 33.3%
    (0.15, 0.07),   # break-even 31.8%
    (0.10, 0.05),   # break-even 33.3%
    (0.50, 0.15),   # break-even 23.1%  -- swing for the tail
    (0.30, 0.05),   # break-even 14.3%  -- tight stop, many small losses
]


def set_barriers(target: float, stop: float) -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["barriers"]["profit_target"] = target
    cfg["barriers"]["stop_loss"] = stop
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def run_in_process(target: float, stop: float) -> Dict:
    """Rebuild the dataset and score it without spawning a subprocess.

    The first version shelled out with a one-hour timeout. The machine slept
    mid-sweep, the wall clock advanced past the timeout while the child was
    suspended, and the whole run died having completed one geometry -- after
    which the `finally` block "restored" a config that had already been
    mutated. In-process avoids the timeout entirely and is faster besides,
    since pandas is imported once rather than twice per geometry.
    """
    from research.penny.scripts import build_dataset, tune_objective

    build_dataset.main(["--profit-target", str(target), "--stop-loss", str(stop)])

    params = tune_objective.load_params()
    trades = pd.read_parquet(bars.DATA_ROOT / "trades.parquet")
    tune, _ = tune_objective.split_holdout(trades, 0.20)
    return tune_objective.evaluate(tune, params)


def load_existing(path: Path) -> List[Dict]:
    """Resume support: a sweep interrupted halfway keeps what it measured."""
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sweep barrier geometries.")
    parser.add_argument("--out", default=str(bars.DATA_ROOT / "barrier_sweep.csv"))
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    original = {"profit_target": 0.30, "stop_loss": 0.10}
    rows: List[Dict] = load_existing(out_path)
    done = {(round(r["profit_target"], 4), round(r["stop_loss"], 4)) for r in rows}
    if done:
        logger.info("resuming: %d geometries already measured", len(done))

    try:
        for target, stop in GEOMETRIES:
            if (round(target, 4), round(stop, 4)) in done:
                logger.info("skip +%.0f%%/-%.0f%% (cached)", 100 * target, 100 * stop)
                continue

            breakeven = stop / (target + stop)
            logger.info(
                "=== target +%.0f%% / stop -%.0f%%  (break-even win rate %.1f%%) ===",
                100 * target, 100 * stop, 100 * breakeven,
            )
            set_barriers(target, stop)

            try:
                result = run_in_process(target, stop)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  geometry failed: %s", str(exc)[:300])
                continue

            summary = json.loads(
                (bars.DATA_ROOT / "dataset_summary.json").read_text(encoding="utf-8")
            )

            row = {
                "profit_target": target,
                "stop_loss": stop,
                "breakeven_win_rate": breakeven,
                "objective": result.get("objective", float("nan")),
                "oos_mean": result.get("oos_mean", float("nan")),
                "oos_n": result.get("oos_n", float("nan")),
                "oos_retained": result.get("oos_retained", float("nan")),
                "trade_mean_net": summary["mean_net_return"],
                "trade_mean_gross": summary["mean_gross_return"],
                "p_target": summary["outcome_mix"].get("target", 0.0),
                "p_stop": summary["outcome_mix"].get("stop", 0.0),
                "p_vertical": summary["outcome_mix"].get("vertical", 0.0),
                "cost_bps": summary["mean_cost_bps"],
            }
            rows.append(row)
            # Persist after every geometry so an interruption keeps the work.
            pd.DataFrame(rows).to_csv(out_path, index=False)
            logger.info(
                "  objective %.6f | net %.4f | gross %.4f | P(target) %.3f",
                row["objective"], row["trade_mean_net"],
                row["trade_mean_gross"], row["p_target"],
            )
    finally:
        # Always restore, so a crash mid-sweep cannot leave the repo configured
        # for a geometry nobody chose.
        set_barriers(original["profit_target"], original["stop_loss"])
        logger.info("restored original barriers: %s", original)

    if not rows:
        logger.error("no geometries completed")
        return 1

    df = pd.DataFrame(rows).sort_values("objective", ascending=False)
    df.to_csv(args.out, index=False)
    logger.info("\n%s", df.to_string(index=False))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
