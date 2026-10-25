"""Per-service design documents: present, uniformly structured, and naming only real events.

A builder works from its own design doc plus the contracts, so a doc that drifts from the
contracts (an invented event name, a second emitter of ``OrderIntent``) costs a build.
"""
from __future__ import annotations

import re
from pathlib import Path

from research.intraday.contracts.events import EVENT_TYPES

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN = REPO_ROOT / "research" / "intraday" / "docs" / "design"

SERVICES = ("screener", "feed", "state", "predictor", "risk", "execution", "monitor",
            "recorder", "engine")
SECTIONS = ("Responsibility", "Consumes", "Emits", "State", "Configuration", "Invariants",
            "Failure modes", "Replay vs live", "Test plan", "Open questions", "Tracker")
EPICS: dict[str, int] = {"engine": 2, "screener": 3, "feed": 4, "state": 5, "predictor": 6,
                         "risk": 7, "execution": 8, "monitor": 9, "recorder": 10}

_H2 = re.compile(r"^## (.+?)\s*$")
_CAMEL = re.compile(r"`([A-Z][a-z][A-Za-z0-9]*)`")
_ISSUE = re.compile(r"https://github\.com/HananyaS/QuantCode-intraday-board/issues/(\d+)")
_TAGS = ("[blocks-build]", "[research]")


def _text(name: str) -> str:
    return (DESIGN / f"{name}.md").read_text(encoding="utf-8")


def _sections(text: str) -> dict[str, str]:
    """H2 heading -> body up to the next H2. Fenced code blocks are not special-cased."""
    out: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        m = _H2.match(line)
        if m:
            if current is not None:
                out[current] = "\n".join(lines)
            current, lines = m.group(1), []
        elif current is not None:
            lines.append(line)
    if current is not None:
        out[current] = "\n".join(lines)
    return out


def _headings(text: str) -> list[str]:
    return [m.group(1) for line in text.splitlines() if (m := _H2.match(line))]


def test_every_service_has_a_design_doc() -> None:
    assert (DESIGN / "TEMPLATE.md").is_file()
    for svc in SERVICES:
        assert (DESIGN / f"{svc}.md").is_file(), svc


def test_design_docs_have_exact_sections_in_order() -> None:
    for name in (*SERVICES, "TEMPLATE"):
        assert tuple(_headings(_text(name))) == SECTIONS, name
        for heading, body in _sections(_text(name)).items():
            assert body.strip(), f"{name}: section {heading!r} is empty"


def test_consumes_and_emits_name_only_real_event_types() -> None:
    for svc in SERVICES:
        sections = _sections(_text(svc))
        for heading in ("Consumes", "Emits"):
            for token in _CAMEL.findall(sections[heading]):
                assert token in EVENT_TYPES, f"{svc} {heading}: `{token}` is not an event type"


def test_only_risk_emits_order_intent() -> None:
    pattern = re.compile(r"\bOrderIntent\b")
    for svc in SERVICES:
        emits = _sections(_text(svc))["Emits"]
        if svc == "risk":
            assert pattern.search(emits), "risk.md must list OrderIntent under Emits"
        else:
            assert not pattern.search(emits), f"{svc}.md lists OrderIntent under Emits"


def test_tracker_section_links_an_issue() -> None:
    for svc in SERVICES:
        issues = [int(n) for n in _ISSUE.findall(_sections(_text(svc))["Tracker"])]
        assert issues == [EPICS[svc]], f"{svc}: tracker links {issues}, want [{EPICS[svc]}]"


def test_open_questions_are_tagged() -> None:
    for svc in SERVICES:
        body = _sections(_text(svc))["Open questions"]
        items = [ln for ln in body.splitlines() if ln.startswith("- ")]
        assert items, f"{svc}: no open questions listed"
        for item in items:
            assert sum(tag in item for tag in _TAGS) == 1, f"{svc}: untagged item {item!r}"
