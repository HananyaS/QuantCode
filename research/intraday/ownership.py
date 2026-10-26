"""Which builder agent may write which paths. See docs/adr and spec section 3."""
from __future__ import annotations

from collections.abc import Iterable

OWNERS: dict[str, tuple[str, ...]] = {
    "intraday-architect": (
        "research/intraday/__init__.py",
        "research/intraday/README.md",
        "research/intraday/contracts/",
        "research/intraday/engine/",
        "research/intraday/docs/adr/",
        "research/intraday/docs/design/",
        "research/intraday/ownership.py",
        "research/intraday/session.py",
        "tests/test_intraday_events.py",
        "tests/test_intraday_codec.py",
        "tests/test_intraday_bus.py",
        "tests/test_intraday_replay.py",
        "tests/test_intraday_guards.py",
        "tests/test_intraday_ownership.py",
        "tests/test_intraday_design_docs.py",
        "tests/test_intraday_tooling.py",
        "tests/test_intraday_integration.py",
    ),
    "intraday-data": ("research/intraday/data/", "tests/test_intraday_data"),
    "intraday-model": (
        "research/intraday/state/",
        "research/intraday/model/",
        "research/intraday/research/",
        "tests/test_intraday_state",
        "tests/test_intraday_model",
    ),
    "intraday-risk": ("research/intraday/risk/", "tests/test_intraday_risk"),
    "intraday-exec": ("research/intraday/exec/", "tests/test_intraday_exec"),
    "intraday-ops": (
        "research/intraday/ops/",
        "tests/test_intraday_recorder.py",
        "tests/test_intraday_ops",
    ),
    "intraday-hypothesis-designer": ("research/penny/PREREGISTRATION_R16.md",),
    "intraday-critic": ("research/intraday/docs/critiques/",),
}


def owner_of(path: str) -> str | None:
    """Return the single agent whose prefix matches ``path``, else None."""
    norm = path.replace("\\", "/")
    hits = {a for a, prefixes in OWNERS.items() if any(norm.startswith(p) for p in prefixes)}
    if len(hits) > 1:
        raise ValueError(f"{path!r} matches several owners: {sorted(hits)}")
    return next(iter(hits), None)


def unowned(agent: str, paths: Iterable[str]) -> list[str]:
    """Paths (in input order) that ``agent`` does not own."""
    return [p for p in paths if owner_of(p) != agent]
