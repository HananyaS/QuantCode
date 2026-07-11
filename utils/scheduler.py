"""scheduler.py — daily cron-style job scheduling via APScheduler.

Why a wrapper instead of raw APScheduler in run_live.py
-----------------------------------------------------------
Keeps the scheduling policy (one job, one time-of-day, US/Eastern by
default) in one place and testable, separate from the CLI entry point.
`BlockingScheduler` is used deliberately: run_live.py's --schedule mode is
meant to be the whole process (invoked once via `python run_live.py
--schedule`, left running) rather than embedded in a larger app.
"""
from __future__ import annotations

from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.logger import get_logger

logger = get_logger(__name__)


def build_daily_scheduler(
    run_fn: Callable[[], None],
    hour: int = 16,
    minute: int = 30,
    timezone: str = "America/New_York",
    job_id: str = "daily_pipeline",
    misfire_grace_time: int = 3600,
) -> BlockingScheduler:
    """Build (but do not start) a scheduler that runs `run_fn` once a day.

    Args:
        run_fn: Zero-argument callable to invoke at the scheduled time.
        hour, minute: Time of day (in `timezone`) to run.
        timezone: IANA timezone name. Defaults to US market hours' zone.
        job_id: Scheduler job id (useful for introspection/testing).
        misfire_grace_time: Seconds of tolerance if the trigger fires late
            (e.g. the machine was asleep) before the run is skipped.
    """
    scheduler = BlockingScheduler(timezone=timezone)
    scheduler.add_job(
        run_fn,
        CronTrigger(hour=hour, minute=minute, timezone=timezone),
        id=job_id,
        misfire_grace_time=misfire_grace_time,
    )
    logger.info(
        "Scheduler: job %r set for %02d:%02d %s daily", job_id, hour, minute, timezone
    )
    return scheduler
