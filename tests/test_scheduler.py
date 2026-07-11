"""Tests for utils/scheduler.py — daily cron-style job scheduling via APScheduler."""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.scheduler import build_daily_scheduler


def test_returns_blocking_scheduler():
    scheduler = build_daily_scheduler(lambda: None, hour=16, minute=30)
    assert isinstance(scheduler, BlockingScheduler)


def test_adds_exactly_one_job():
    scheduler = build_daily_scheduler(lambda: None, hour=16, minute=30)
    assert len(scheduler.get_jobs()) == 1


def test_job_uses_cron_trigger_with_given_time():
    scheduler = build_daily_scheduler(lambda: None, hour=9, minute=45)
    job = scheduler.get_jobs()[0]
    assert isinstance(job.trigger, CronTrigger)
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert fields["hour"] == "9"
    assert fields["minute"] == "45"


def test_scheduler_not_started_by_default():
    scheduler = build_daily_scheduler(lambda: None, hour=16, minute=30)
    assert scheduler.running is False


def test_custom_job_id_used():
    scheduler = build_daily_scheduler(lambda: None, hour=16, minute=30, job_id="my_job")
    assert scheduler.get_job("my_job") is not None
