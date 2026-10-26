"""Session log layout and the run manifest written beside each closed log.

A manifest pins what produced a log: the git commit (and whether the tree was
dirty), the config and model hashes, the log format and the log's own sha256.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Mapping

from research.intraday.contracts.codec import canonical_json
from research.intraday.ops.recorder import LOG_FORMAT, log_hash

LOG_ROOT: Path = Path("C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_intraday/logs")

_FORBIDDEN_IN_RUN_ID = ("/", "\\", "..", ":")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


def session_log_path(session_date: date, run_id: str, log_root: Path = LOG_ROOT) -> Path:
    """``log_root/<YYYY-MM-DD>/<run_id>.jsonl``; creates the date directory."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"run_id must be a non-empty str, got {run_id!r}")
    if any(tok in run_id for tok in _FORBIDDEN_IN_RUN_ID):
        raise ValueError(f"run_id must not contain / \\ .. or :, got {run_id!r}")
    day_dir = Path(log_root) / f"{session_date:%Y-%m-%d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{run_id}.jsonl"


def config_hash(config: Mapping[str, object]) -> str:
    """sha256 of the config's canonical JSON (key-order independent)."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def git_state(repo: Path) -> tuple[str, bool]:
    """(HEAD sha, working tree dirty) of the repository at ``repo``."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout
    sha = git("rev-parse", "HEAD").strip()
    assert _GIT_SHA.fullmatch(sha), f"unexpected git sha {sha!r}"
    return sha, bool(git("status", "--porcelain").strip())


def manifest_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}.manifest.json")


def write_manifest(log_path: Path, *, git_sha: str, git_dirty: bool,
                   config: Mapping[str, object], model_hash: str, unseal: bool) -> Path:
    """Write the manifest of a closed log (mode "x": never overwritten); return its path."""
    if not isinstance(git_sha, str) or not _GIT_SHA.fullmatch(git_sha):
        raise ValueError(f"git_sha must be 40 lowercase hex characters, got {git_sha!r}")
    if not isinstance(model_hash, str) or not model_hash:
        raise ValueError(f"model_hash must be a non-empty str, got {model_hash!r}")
    if type(git_dirty) is not bool or type(unseal) is not bool:
        raise ValueError("git_dirty and unseal must be bool")
    record: dict[str, object] = {
        "config_hash": config_hash(config),
        "git_dirty": git_dirty,
        "git_sha": git_sha,
        "log_format": LOG_FORMAT,
        "log_hash": log_hash(log_path),
        "model_hash": model_hash,
        "run_id": log_path.stem,
        "unseal": unseal,
    }
    out = manifest_path(log_path)
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical_json(record) + "\n")
    return out
