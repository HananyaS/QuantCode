"""Guards the live-trading workflow's trigger invariants.

Why this exists
---------------
The rule "run and notify exactly once per trading session" is enforced by
`scripts/preflight_check.py`, which is well covered by
`tests/test_preflight_check.py`. But *whether preflight is even consulted*
is decided by GitHub Actions expressions in the workflow YAML, and those
have no tests at all — a wrong expression there silently disables the
guard while every Python test stays green.

The specific footgun: the punctual trigger is an external scheduler
POSTing to the workflow_dispatch API, so a scheduler-driven run arrives as
`workflow_dispatch` — the same event as a human pressing the button. The
human path is deliberately un-guarded (`--manual`: always execute, always
notify). If the two are ever conflated, the scheduler bypasses the
once-per-session guard and the duplicate "triggered" pushes come straight
back. `inputs.source` is what separates them, and these tests hold that
line.

The second footgun: `dry_run` defaults to `true`. If a dry-run gate keyed
on the input alone, a scheduler payload that forgot to pass `dry_run`
would leave the bot previewing forever — silently never trading, while
every run still reported success.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "live-trading.yml"

_HUMAN_ONLY = "inputs.source == 'human'"


@pytest.fixture(scope="module")
def raw() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow(raw: str) -> dict:
    return yaml.safe_load(raw)


def _triggers(workflow: dict) -> dict:
    """The `on:` block. YAML 1.1 parses a bare `on` key as the boolean
    True, so accept either spelling rather than depending on the loader.
    """
    return workflow[True] if True in workflow else workflow["on"]


def _expressions(raw: str, containing: str) -> list:
    """Every ${{ ... }} expression in the workflow mentioning `containing`."""
    return [e for e in re.findall(r"\$\{\{(.*?)\}\}", raw, re.S) if containing in e]


def _step(workflow: dict, name: str) -> dict:
    for step in workflow["jobs"]["trade"]["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"no step named {name!r}")


# ---------------------------------------------------------------------------
# The scheduler must not be mistaken for a human
# ---------------------------------------------------------------------------

def test_workflow_dispatch_declares_a_source_input_defaulting_to_human():
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert "source" in inputs, "the scheduler/human distinction has no input to key on"
    assert inputs["source"]["default"] == "human", (
        "default must be 'human': a person at the Actions UI shouldn't have to "
        "pick anything, and an unset value must never silently mean 'scheduler'"
    )
    assert set(inputs["source"]["options"]) == {"human", "scheduler"}


def test_the_manual_bypass_is_reachable_only_by_a_human(raw: str):
    exprs = _expressions(raw, "'--manual'")
    assert exprs, "no expression sets --manual any more; preflight's manual path is unreachable"
    for expr in exprs:
        assert _HUMAN_ONLY in expr, (
            "--manual makes preflight always execute AND always notify. Gating it on "
            "workflow_dispatch alone hands that bypass to the external scheduler, "
            f"restoring the duplicate-notification bug. Offending expression: {expr.strip()}"
        )


def test_dry_run_is_a_human_only_affordance(raw: str):
    exprs = _expressions(raw, "inputs.dry_run")
    assert exprs, "no expression reads dry_run any more"
    for expr in exprs:
        assert _HUMAN_ONLY in expr, (
            "dry_run defaults to true, so a scheduler payload that omitted it would "
            "leave the bot previewing forever -- silently never trading while still "
            f"reporting success. Offending expression: {expr.strip()}"
        )


# ---------------------------------------------------------------------------
# The watchdog and the guard must stay wired up
# ---------------------------------------------------------------------------

def test_both_watchdog_crons_are_still_present(workflow: dict):
    crons = {entry["cron"] for entry in _triggers(workflow)["schedule"]}
    assert crons == {"45 14 * * 1-5", "45 15 * * 1-5"}, (
        "the crons are the watchdog behind the external trigger -- they are what "
        "trades the day anyway, and raise the missed-day alert, when the external "
        "scheduler dies. Removing them makes a dead trigger silent."
    )


@pytest.mark.parametrize("step_name", ["Kelly strategy", "Linear strategy"])
def test_trading_steps_are_gated_on_preflight(workflow: dict, step_name: str):
    condition = _step(workflow, step_name).get("if", "")
    assert "steps.preflight.outputs.trade == 'true'" in condition, (
        f"{step_name} would trade on every firing, including the watchdog crons "
        "behind a successful run"
    )


def test_the_notification_is_gated_on_preflight(workflow: dict):
    condition = _step(workflow, "Notify (ntfy.sh)").get("if", "")
    assert "steps.preflight.outputs.notify == 'true'" in condition
    assert "failure()" in condition, (
        "the failure() arm must stay: if preflight itself dies it never sets an "
        "output, and that is exactly when you need to be told"
    )


def test_a_cron_fallback_is_visible_in_the_notification(raw: str):
    assert 'TRIGGER: ${{ github.event_name }}' in raw
    assert '"$TRIGGER" = "schedule"' in raw, (
        "a run that traded via cron means the punctual external trigger never "
        "fired; unmarked, a dead trigger looks identical to a healthy day"
    )
