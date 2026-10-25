"""Deterministic, event-driven intraday trading research engine.

Design spec: docs/superpowers/specs/2026-10-26-intraday-system-design.md

Branch rule: this package lives on `research/intraday` and is NEVER merged to main.
"""
from __future__ import annotations

AGENT_PACKAGES: tuple[str, ...] = ("data", "state", "model", "risk", "exec", "ops")
