"""Tooling map: every agent definition names the skills and MCP tools its stage needs."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = REPO_ROOT / "docs/superpowers/specs/2026-10-26-intraday-system-design.md"
HEADING = "## Skills and tools"
BUILDERS = ("data", "model", "risk", "exec", "ops")
STAGE_SKILLS = {
    "superpowers:test-driven-development",
    "superpowers:verification-before-completion",
    "superpowers:systematic-debugging",
}


def _section(agent: str) -> str:
    text = (REPO_ROOT / ".claude/agents" / f"intraday-{agent}.md").read_text(encoding="utf-8")
    assert text.count(HEADING) == 1, f"intraday-{agent}: need exactly one {HEADING!r}"
    return text.split(HEADING, 1)[1]


def test_every_builder_definition_has_skills_section() -> None:
    for b in BUILDERS:
        assert _section(b).strip(), b


def test_builders_list_tdd_verification_debugging() -> None:
    for b in BUILDERS:
        sec = _section(b)
        for skill in STAGE_SKILLS:
            assert skill in sec, (b, skill)


def test_every_section_carries_the_two_closing_sentences() -> None:
    for a in (*BUILDERS, "architect", "hypothesis-designer", "critic"):
        sec = _section(a)
        assert "invoke a skill before acting when there is even a small chance it applies" in sec, a
        assert "report NEEDS_CONTEXT instead of skipping it" in sec, a


def test_designer_lists_experiment_design_and_scholar_search() -> None:
    sec = _section("hypothesis-designer")
    for needle in ("phd-skills:experiment-design", "phd-skills:literature-research", "scholar-search"):
        assert needle in sec, needle


def test_critic_lists_mlflow_and_compare() -> None:
    sec = _section("critic")
    for needle in ("mlflow-mcp", "phd-skills:compare", "phd-skills:reproduce", "phd-skills:experiment-design"):
        assert needle in sec, needle


def test_model_lists_research_and_plot_rows() -> None:
    sec = _section("model")
    for needle in ("mlflow-mcp", "intraday", "phd-skills:reproduce", "plotly", "dataviz"):
        assert needle in sec, needle


def test_architect_lists_design_and_planning_skills() -> None:
    sec = _section("architect")
    for needle in ("superpowers:brainstorming", "domain-modeling", "superpowers:writing-plans", "improve-plan"):
        assert needle in sec, needle


def test_spec_has_tooling_section() -> None:
    text = SPEC.read_text(encoding="utf-8")
    assert "## 6. Tooling per stage" in text
    sec = text.split("## 6. Tooling per stage", 1)[1]
    for needle in ("MLFLOW_ALLOW_FILE_STORE", "authenticat", "controller only", "superpowers:finishing-a-development-branch"):
        assert needle in sec, needle
