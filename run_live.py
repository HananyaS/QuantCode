"""run_live.py — daily paper-trading pipeline entry point.

PAPER TRADING ONLY — see agents/execution_agent.py and ROADMAP.md Phase 4.
This is the "unit of work" for one trading day: run the research pipeline,
submit paper orders, persist state. It is deliberately a single blocking
call per invocation rather than an always-on daemon, so it can be driven by
either:

  1. An OS scheduler (Windows Task Scheduler / cron), invoking it once a day:
         python run_live.py

  2. An in-process APScheduler loop (utils/scheduler.py), for a box that's
     left running:
         python run_live.py --schedule --hour 16 --minute 30

On any failure, an alert is sent via utils/alerting.py (ALERT_WEBHOOK_URL
env var) before the exception propagates — an unattended job failing
silently is worse than a noisy one.
"""
from __future__ import annotations

import argparse
import os
import traceback

from dotenv import load_dotenv

load_dotenv()

from agents.live_orchestrator import LiveOrchestrator
from main_multi import load_config
from utils.alerting import send_alert
from utils.logger import add_file_handler, get_logger
from utils.scheduler import build_daily_scheduler

logger = get_logger(__name__)


def run_once(config_path: str = "configs/universe.yaml") -> dict:
    """Run the live paper-trading pipeline once and return the final context."""
    try:
        config = load_config(config_path)
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]

        orch = LiveOrchestrator(config, api_key=api_key, secret_key=secret_key)
        context = orch.run()

        orders = context.get("execution_orders") or []
        n_submitted = sum(1 for o in orders if o.get("qty", 0) > 0)
        logger.info(
            "run_live: complete | %d orders submitted | equity=%.2f",
            n_submitted,
            (context.get("execution_account") or {}).get("equity", float("nan")),
        )
        return context
    except Exception as exc:
        logger.error("run_live: FAILED — %s", exc)
        send_alert(f"QuantCode run_live failed: {exc}\n{traceback.format_exc()[-1500:]}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantCode daily paper-trading runner")
    parser.add_argument("--config", default="configs/universe.yaml", help="Path to the YAML config file")
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run daily via an in-process APScheduler loop instead of once immediately (blocks).",
    )
    parser.add_argument("--hour", type=int, default=16, help="Scheduled hour (America/New_York), used with --schedule")
    parser.add_argument("--minute", type=int, default=30, help="Scheduled minute, used with --schedule")
    args = parser.parse_args()

    add_file_handler()

    if args.schedule:
        scheduler = build_daily_scheduler(
            lambda: run_once(args.config), hour=args.hour, minute=args.minute,
        )
        logger.info("run_live: scheduled daily at %02d:%02d America/New_York", args.hour, args.minute)
        scheduler.start()  # blocks
    else:
        run_once(args.config)


if __name__ == "__main__":
    main()
