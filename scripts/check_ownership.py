"""Reject a diff that strays outside an agent's owned paths.

Usage: python scripts/check_ownership.py --agent intraday-risk --base research/intraday
Prints each non-owned changed file, one per line; exit 1 if any, 2 on a bad agent.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.intraday.ownership import OWNERS, unowned  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--base", required=True)
    args = ap.parse_args(argv)
    if args.agent not in OWNERS:
        print(f"unknown agent {args.agent!r}; choose from: {', '.join(OWNERS)}", file=sys.stderr)
        return 2
    out = subprocess.run(["git", "diff", "--name-only", f"{args.base}...HEAD"],
                         check=True, capture_output=True, text=True).stdout
    bad = unowned(args.agent, [ln for ln in out.splitlines() if ln.strip()])
    for p in bad:
        print(p)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
