"""Phase-2 exit criterion: one recorded pre-holdout day end to end through every service.

ClockTick -> ScreenerAgent -> FeedAgent(ReplayFeed) -> scorer -> RiskAgent ->
ExecutionAgent(SimBroker) -> MonitorAgent, recorded by the ops recorder. The scorer is
the TEST STAND-IN from ``research/intraday/session.py`` (State/Predictor are built
elsewhere). Spec section 5; research/intraday/docs/design/engine.md, Test plan.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from research.intraday.contracts.events import Bar, Fill, OrderIntent
from research.intraday.contracts.protocols import Envelope
from research.intraday.data.clock import et_to_utc
from research.intraday.data.errors import SealedHoldoutError
from research.intraday.data.paths import DEFAULT_DATA_ROOT, DEFAULT_VOLUME_DAILY_ROOT
from research.intraday.data.rejects import read_rejects_jsonl
from research.intraday.ops.manifest import manifest_path
from research.intraday.ops.recorder import log_hash, read_log
from research.intraday.session import StandInTestScorer, run_session

REPO_ROOT = Path(__file__).resolve().parents[1]
D = date(2024, 3, 4)
HOLDOUT = datetime(2025, 8, 7, tzinfo=timezone.utc)

needs_data = pytest.mark.skipif(
    not (DEFAULT_DATA_ROOT / "minute").is_dir() or not DEFAULT_VOLUME_DAILY_ROOT.is_dir(),
    reason="intraday caches not present")


def _run(out_dir: Path) -> Path:
    return run_session(D, arm="screen", data_root=DEFAULT_DATA_ROOT,
                       volume_daily_root=DEFAULT_VOLUME_DAILY_ROOT, out_dir=out_dir,
                       scorer=StandInTestScorer(), run_id="itest")


@pytest.fixture(scope="module")
def two_logs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    if not (DEFAULT_DATA_ROOT / "minute").is_dir() or not DEFAULT_VOLUME_DAILY_ROOT.is_dir():
        pytest.skip("intraday caches not present")
    root = tmp_path_factory.mktemp("session")
    return _run(root / "a"), _run(root / "b")


@pytest.fixture(scope="module")
def envs(two_logs: tuple[Path, Path]) -> list[Envelope]:
    return list(read_log(two_logs[0]))


# ---- (a) determinism in one process ------------------------------------------
@needs_data
def test_recorded_day_end_to_end_twice_identical_hash(two_logs: tuple[Path, Path]) -> None:
    a, b = two_logs
    assert a != b
    assert log_hash(a) == log_hash(b)
    assert a.read_bytes() == b.read_bytes()


@needs_data
def test_manifest_written_after_close_matches_log(two_logs: tuple[Path, Path]) -> None:
    for log in two_logs:
        rec = json.loads(manifest_path(log).read_text(encoding="utf-8"))
        assert rec["log_hash"] == log_hash(log)
        assert rec["unseal"] is False
        assert rec["run_id"] == "itest"


@needs_data
def test_rejects_written_beside_the_log(two_logs: tuple[Path, Path]) -> None:
    a, b = two_logs
    ra, rb = (log.with_name(f"{log.stem}.rejects.jsonl") for log in (a, b))
    assert ra.read_bytes() == rb.read_bytes()
    assert all(r.session_date == D for r in read_rejects_jsonl(ra))


# ---- (b) determinism across processes and hash seeds ---------------------------
_CHILD = (
    "import sys\n"
    "from datetime import date\n"
    "from pathlib import Path\n"
    "from research.intraday.data.paths import DEFAULT_DATA_ROOT, DEFAULT_VOLUME_DAILY_ROOT\n"
    "from research.intraday.ops.recorder import log_hash\n"
    "from research.intraday.session import StandInTestScorer, run_session\n"
    "p = run_session(date(2024, 3, 4), arm='screen', data_root=DEFAULT_DATA_ROOT,\n"
    "                volume_daily_root=DEFAULT_VOLUME_DAILY_ROOT, out_dir=Path(sys.argv[1]),\n"
    "                scorer=StandInTestScorer(), run_id='itest')\n"
    "print(log_hash(p))\n"
)


@needs_data
def test_recorded_day_identical_hash_across_hash_seeds(
        tmp_path: Path, two_logs: tuple[Path, Path]) -> None:
    hashes = []
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", _CHILD, str(tmp_path / seed)], cwd=REPO_ROOT,
                             env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-3000:]
        hashes.append(out.stdout.strip().splitlines()[-1])
    assert hashes[0] == hashes[1]
    assert hashes[0] == log_hash(two_logs[0])


# ---- (c) the whole chain was exercised, and its invariants hold ---------------
@needs_data
def test_log_exercises_intent_fill_and_flatten(envs: list[Envelope]) -> None:
    counts = Counter(type(e.event).__name__ for e in envs)
    assert counts["Candidates"] == 1
    assert counts["Bar"] > 0 and counts["Signal"] > 0
    assert counts["OrderIntent"] >= 1
    fills = [e.event for e in envs if isinstance(e.event, Fill)]
    assert any(f.side == "buy" for f in fills)
    flatten = et_to_utc(D, "15:55")
    assert any(f.side == "sell" and f.ts >= flatten for f in fills), "no end-of-day flatten"
    net: Counter[str] = Counter()
    for f in fills:
        net[f.symbol] += f.qty if f.side == "buy" else -f.qty
    assert all(v == 0 for v in net.values()), f"positions left open at the close: {net}"


@needs_data
def test_every_order_intent_comes_from_risk(envs: list[Envelope]) -> None:
    sources = {e.source for e in envs if isinstance(e.event, OrderIntent)}
    assert sources == {"risk"}


@needs_data
def test_no_fill_on_the_bar_that_caused_its_intent(envs: list[Envelope]) -> None:
    by_seq = {e.seq: e for e in envs}
    intents = {e.event.intent_id: e for e in envs if isinstance(e.event, OrderIntent)}

    def bar_ancestor(env: Envelope) -> Envelope | None:
        while env.cause_seq is not None:
            env = by_seq[env.cause_seq]
            if isinstance(env.event, Bar):
                return env
        return None

    checked = 0
    for e in envs:
        if not isinstance(e.event, Fill) or e.event.side != "buy":
            continue
        intent_env = intents[e.event.intent_id]
        cause = by_seq[e.cause_seq]
        assert isinstance(cause.event, Bar)
        assert cause.event.bar_start >= intent_env.event.ts
        assert e.event.ts > intent_env.event.ts
        assert cause.seq != bar_ancestor(intent_env).seq  # type: ignore[union-attr]
        checked += 1
    assert checked >= 1


@needs_data
def test_no_event_inside_the_sealed_holdout(envs: list[Envelope]) -> None:
    assert max(e.event.ts for e in envs) < HOLDOUT


# ---- (d) the sealed holdout is refused -----------------------------------------
def test_session_runner_refuses_holdout_date_without_unseal(tmp_path: Path) -> None:
    with pytest.raises(SealedHoldoutError):
        run_session(date(2025, 8, 7), arm="screen", data_root=tmp_path / "none",
                    volume_daily_root=tmp_path / "none", out_dir=tmp_path / "out",
                    scorer=StandInTestScorer())
    assert not (tmp_path / "out").exists()


def test_stand_in_scorer_is_labelled() -> None:
    s = StandInTestScorer()
    assert "TEST STAND-IN" in (StandInTestScorer.__doc__ or "")
    assert "test-stand-in" in s.model_hash
