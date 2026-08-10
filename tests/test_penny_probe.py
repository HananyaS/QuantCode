"""Tests for the Phase 0 data-feasibility probe.

The network probes themselves are not unit-tested (they exist to make real
calls). What is tested here is the surrounding logic that decides whether the
programme may proceed: failure isolation, gate arithmetic, and verdict
rendering. A bug in that logic could silently green-light a project whose data
foundation does not exist.
"""
from datetime import date

import pytest

from research.penny.scripts.probe_data_feasibility import (
    ProbeResult,
    _recent_window,
    _safe,
    render_verdict,
)


# ---- Date window ----


def test_recent_window_is_entirely_in_the_past():
    start, end = _recent_window()
    assert date.fromisoformat(end) < date.today()
    assert date.fromisoformat(start) < date.fromisoformat(end)


def test_recent_window_lag_keeps_end_clear_of_the_sip_delay():
    # Free-tier SIP requires end to be >15 minutes old; a multi-day lag makes a
    # 403 unambiguously an entitlement problem rather than a recency problem.
    _, end = _recent_window(lag_days=3)
    assert (date.today() - date.fromisoformat(end)).days >= 3


def test_recent_window_respects_lookback():
    start, end = _recent_window(lookback_days=10, lag_days=2)
    assert (date.fromisoformat(end) - date.fromisoformat(start)).days == 10


# ---- Failure isolation ----


def test_safe_converts_exception_into_failed_result():
    def boom():
        raise RuntimeError("network is down")

    result = _safe("exploding_probe", True, boom)
    assert result.passed is False
    assert result.gating is True
    assert "RuntimeError" in result.detail
    assert "network is down" in result.detail


def test_safe_passes_through_successful_result():
    result = _safe("ok_probe", False, lambda: (True, "all good", {"n": 1}))
    assert result.passed is True
    assert result.detail == "all good"
    assert result.data == {"n": 1}


def test_status_label_distinguishes_gating_failure():
    assert ProbeResult("a", True, True, "").status == "PASS"
    assert ProbeResult("b", False, True, "").status == "FAIL (GATE)"
    assert ProbeResult("c", False, False, "").status == "fail"


# ---- Verdict rendering ----


def _results(gate_passed: bool):
    return [
        ProbeResult("alpaca_sip_volume", gate_passed, True, "ratio = 28.6x"),
        ProbeResult("edgar_fulltext", True, False, "ok"),
        ProbeResult("ibkr_borrow", False, False, "firewall blocked"),
    ]


def test_verdict_reports_pass_when_gating_probe_passes():
    text = render_verdict(_results(True))
    assert "**HARD GATE: PASS**" in text
    assert "STOP." not in text


def test_verdict_reports_stop_when_gating_probe_fails():
    text = render_verdict(_results(False))
    assert "**HARD GATE: FAIL**" in text
    assert "STOP." in text
    # A failed gate must not be reported as merely one of several failures.
    assert "RVOL cannot be computed" in text


def test_verdict_gate_ignores_non_gating_failures():
    # ibkr_borrow fails in both fixtures; it must never affect the gate.
    text = render_verdict(_results(True))
    assert "**HARD GATE: PASS**" in text
    assert "ibkr_borrow" in text


def test_verdict_lists_non_gating_failures_separately():
    text = render_verdict(_results(True))
    assert "## Non-gating failures" in text
    assert "firewall blocked" in text


def test_verdict_escapes_pipes_so_the_table_survives():
    results = [ProbeResult("p", True, True, "a|b|c")]
    text = render_verdict(results)
    assert r"a\|b\|c" in text


def test_verdict_records_credential_source():
    text = render_verdict(_results(True), cred_source="ALPACA_API_KEY_KELLY")
    assert "ALPACA_API_KEY_KELLY" in text


def test_verdict_handles_missing_credential_source():
    text = render_verdict(_results(True))
    assert "`none`" in text


def test_verdict_counts_passes():
    text = render_verdict(_results(True))
    assert "2/3 probes passed" in text


# ---- Isolation contract ----


def _module_ast():
    import ast
    import inspect

    from research.penny.scripts import probe_data_feasibility as mod

    return ast.parse(inspect.getsource(mod))


def test_probe_module_does_not_import_live_trading_code():
    """The research package must never reach into the production path."""
    import ast

    forbidden_modules = {
        "scripts.live_run",
        "agents.execution_agent",
        "utils.state_store",
        "utils.live_decision",
        "agents.live_orchestrator",
        "run_live",
    }

    imported: set[str] = set()
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    offenders = imported & forbidden_modules
    assert not offenders, f"research code must not import {offenders}"


def test_probe_module_never_builds_an_order_capable_client():
    """Only the read-only data client is permitted -- never TradingClient.

    Checked against identifiers in the AST rather than raw text, so that
    mentioning the name in a comment or docstring does not trip the test.
    """
    import ast

    identifiers: set[str] = set()
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            identifiers.update(alias.name for alias in node.names)

    assert "TradingClient" not in identifiers
    assert "StockHistoricalDataClient" in identifiers
