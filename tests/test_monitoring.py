"""Tests for monitoring/alerting: utils/logger.py file handler + utils/alerting.py webhook."""
import logging

import pytest

from utils.alerting import send_alert
from utils.logger import add_file_handler, get_logger


# ---------------------------------------------------------------------------
# Rotating file logging
# ---------------------------------------------------------------------------

def test_add_file_handler_creates_log_file(tmp_path):
    log_dir = tmp_path / "logs"
    handler = add_file_handler(log_dir=str(log_dir), filename="test.log")
    try:
        logger = get_logger("test_monitoring_logger_1")
        logger.info("hello world")
        for h in logging.getLogger().handlers:
            h.flush()
        log_file = log_dir / "test.log"
        assert log_file.exists()
        assert "hello world" in log_file.read_text()
    finally:
        logging.getLogger().removeHandler(handler)


def test_add_file_handler_idempotent_for_same_path(tmp_path):
    log_dir = tmp_path / "logs"
    h1 = add_file_handler(log_dir=str(log_dir), filename="test.log")
    h2 = add_file_handler(log_dir=str(log_dir), filename="test.log")
    try:
        assert h1 is h2
        root_file_handlers = [
            h for h in logging.getLogger().handlers
            if getattr(h, "baseFilename", None) == h1.baseFilename
        ]
        assert len(root_file_handlers) == 1
    finally:
        logging.getLogger().removeHandler(h1)


# ---------------------------------------------------------------------------
# Webhook alerting
# ---------------------------------------------------------------------------

def test_send_alert_posts_to_webhook():
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        class Resp:
            def raise_for_status(self):
                pass
        return Resp()

    ok = send_alert("something broke", webhook_url="https://example.com/hook", http_post=fake_post)
    assert ok is True
    assert len(calls) == 1
    assert calls[0][0] == "https://example.com/hook"
    assert calls[0][1] == {"text": "something broke"}


def test_send_alert_returns_false_without_webhook(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    ok = send_alert("something broke", webhook_url=None)
    assert ok is False


def test_send_alert_returns_false_on_http_error():
    def failing_post(url, json=None, timeout=None):
        raise ConnectionError("network down")

    ok = send_alert("something broke", webhook_url="https://example.com/hook", http_post=failing_post)
    assert ok is False


def test_send_alert_reads_webhook_from_env(monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/env-hook")
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(url)
        class Resp:
            def raise_for_status(self):
                pass
        return Resp()

    ok = send_alert("hi", http_post=fake_post)
    assert ok is True
    assert calls == ["https://example.com/env-hook"]
