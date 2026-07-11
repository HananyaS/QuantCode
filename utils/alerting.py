"""alerting.py — simple webhook-based failure alerting.

Why this exists
----------------
Previously the only feedback channel for a failed/missed run was stdout
logs, which nobody watches for an unattended daily job. `send_alert` posts
a Slack-compatible `{"text": ...}` payload to a webhook URL (Slack incoming
webhooks, Discord-via-Slack-compat, or any endpoint accepting that shape).

Never raises — a failing alert must not itself crash the pipeline run that
triggered it; failures are logged and `False` is returned instead.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)


def send_alert(
    message: str,
    webhook_url: Optional[str] = None,
    timeout: int = 10,
    http_post: Callable = requests.post,
) -> bool:
    """POST `message` to a Slack-compatible webhook.

    Args:
        message: Alert text.
        webhook_url: Webhook URL. Falls back to the ALERT_WEBHOOK_URL env
            var if not given.
        timeout: Request timeout in seconds.
        http_post: Injectable POST callable (for testing); defaults to
            `requests.post`.

    Returns:
        True if the alert was sent successfully, False otherwise (no
        webhook configured, or the request failed).
    """
    webhook_url = webhook_url or os.environ.get("ALERT_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("Alerting: no webhook configured, message not sent: %s", message)
        return False

    try:
        response = http_post(webhook_url, json={"text": message}, timeout=timeout)
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Alerting: failed to send alert: %s", exc)
        return False
