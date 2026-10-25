"""Reject a change set that strays outside an agent's owned paths.

Usage: python scripts/check_ownership.py --agent intraday-risk --base research/intraday
The change set is the union of committed changes (``base...HEAD``) and the
working tree (``git status``: staged, unstaged and untracked files, both sides
of a rename). Git runs at the repository root, so the result does not depend
on the caller's cwd. Prints each non-owned changed file, one per line; exit 1
if any, 2 on a bad agent.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from research.intraday.ownership import OWNERS, unowned  # noqa: E402


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def repo_root() -> Path:
    """Top level of the repository that contains this script."""
    return Path(_git(_SCRIPT_DIR, "rev-parse", "--show-toplevel").strip())


def _status_paths(root: Path) -> list[str]:
    tokens = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all").split("\0")
    paths: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if len(tok) < 4:
            continue
        paths.append(tok[3:])
        if tok[0] in "RC" and i < len(tokens):  # rename/copy: next token is the source
            paths.append(tokens[i])
            i += 1
    return paths


def changed_paths(base: str, root: Path) -> list[str]:
    """Sorted, de-duplicated repo-relative paths changed since base, incl. the working tree."""
    committed = _git(root, "diff", "--name-only", "-z", f"{base}...HEAD").split("\0")
    return sorted({p for p in (*committed, *_status_paths(root)) if p.strip()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--repo", default=None, help="repository root (default: this script's repo)")
    args = ap.parse_args(argv)
    if args.agent not in OWNERS:
        print(f"unknown agent {args.agent!r}; choose from: {', '.join(OWNERS)}", file=sys.stderr)
        return 2
    root = Path(args.repo) if args.repo else repo_root()
    bad = unowned(args.agent, changed_paths(args.base, root))
    for p in bad:
        print(p)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
