"""Ownership map: disjoint prefixes, full coverage, agent definitions in sync.

Agent-definition format parsed here: a line that is exactly ``Owns:``, then the
following lines up to the first blank line, each formatted as ``- `<prefix>```.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from research.intraday import AGENT_PACKAGES
from research.intraday.ownership import OWNERS, owner_of, unowned

REPO_ROOT = Path(__file__).resolve().parents[1]
_ITEM = re.compile(r"^- `([^`]+)`$")


def _rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


def _owns_block(agent: str) -> list[str]:
    lines = (REPO_ROOT / ".claude/agents" / f"{agent}.md").read_text(encoding="utf-8").splitlines()
    assert lines.count("Owns:") == 1, f"{agent}: need exactly one 'Owns:' line"
    out: list[str] = []
    for line in lines[lines.index("Owns:") + 1:]:
        if not line.strip():
            break
        m = _ITEM.match(line)
        assert m, f"{agent}: malformed Owns line {line!r}"
        out.append(m.group(1))
    return out


def test_owned_prefixes_are_disjoint() -> None:
    items = [(a, p) for a, ps in OWNERS.items() for p in ps]
    for i, (a1, p1) in enumerate(items):
        for a2, p2 in items[i + 1:]:
            if a1 != a2:
                assert not p1.startswith(p2) and not p2.startswith(p1), (a1, p1, a2, p2)


def test_every_intraday_python_file_has_one_owner() -> None:
    for p in (REPO_ROOT / "research/intraday").rglob("*.py"):
        assert owner_of(_rel(p)) is not None, p


def test_every_agent_package_is_owned() -> None:
    for pkg in AGENT_PACKAGES:
        assert owner_of(f"research/intraday/{pkg}/__init__.py") is not None, pkg


def test_every_owner_has_an_agent_definition() -> None:
    for agent in OWNERS:
        assert (REPO_ROOT / ".claude/agents" / f"{agent}.md").is_file(), agent


def test_agent_definitions_list_exactly_their_owned_paths() -> None:
    for agent, prefixes in OWNERS.items():
        assert tuple(_owns_block(agent)) == prefixes, agent


def test_owner_of_normalises_backslashes_and_misses() -> None:
    assert owner_of(r"research\intraday\risk\sizing.py") == "intraday-risk"
    assert owner_of("README.md") is None


def test_owner_of_raises_on_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    import research.intraday.ownership as o
    monkeypatch.setattr(o, "OWNERS", {"a": ("x/",), "b": ("x/y",)})
    with pytest.raises(ValueError):
        o.owner_of("x/y/z.py")


def test_unowned_filters_to_foreign_paths() -> None:
    paths = ["research/intraday/risk/a.py", "tests/test_intraday_risk_sizing.py",
             "research/intraday/exec/b.py", "README.md"]
    assert unowned("intraday-risk", paths) == ["research/intraday/exec/b.py", "README.md"]


# ---- scripts/check_ownership.py sees the working tree (final-review fix A10) --
import importlib.util  # noqa: E402
import subprocess  # noqa: E402


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_ownership", REPO_ROOT / "scripts" / "check_ownership.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                    *args], cwd=root, check=True, capture_output=True)


def _toy_repo(root: Path) -> None:
    _git(root, "init", "-q")
    (root / "research/intraday/risk").mkdir(parents=True)
    (root / "research/intraday/risk/a.py").write_text("x = 1\n")
    (root / "old.txt").write_text("o\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")


def test_checker_reports_untracked_and_modified_files(tmp_path: Path, capsys) -> None:
    chk = _load_checker()
    _toy_repo(tmp_path)
    (tmp_path / "research/intraday/exec").mkdir(parents=True)
    (tmp_path / "research/intraday/exec/new.py").write_text("y = 2\n")   # untracked
    (tmp_path / "research/intraday/risk/a.py").write_text("x = 3\n")     # modified, owned
    _git(tmp_path, "mv", "old.txt", "renamed.txt")                       # staged rename
    paths = chk.changed_paths("HEAD", tmp_path)
    assert "research/intraday/exec/new.py" in paths
    assert "research/intraday/risk/a.py" in paths
    assert "renamed.txt" in paths
    assert paths == sorted(set(paths))
    rc = chk.main(["--agent", "intraday-risk", "--base", "HEAD", "--repo", str(tmp_path)])
    printed = capsys.readouterr().out.split()
    assert rc == 1
    assert "research/intraday/exec/new.py" in printed
    assert "research/intraday/risk/a.py" not in printed


def test_checker_clean_tree_passes(tmp_path: Path) -> None:
    chk = _load_checker()
    _toy_repo(tmp_path)
    assert chk.changed_paths("HEAD", tmp_path) == []
    assert chk.main(["--agent", "intraday-risk", "--base", "HEAD", "--repo", str(tmp_path)]) == 0


def test_checker_default_repo_is_script_repo() -> None:
    chk = _load_checker()
    assert Path(chk.repo_root()).resolve() == REPO_ROOT
