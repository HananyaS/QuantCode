# Intraday Phase 2 — Ops (MonitorAgent, Recorder extensions) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MonitorAgent (halt-log release, stale-feed kill, order-flow kill, kill latch) and extend the recorder sink with a compact `Window` encoding, a strict reader, the nightly-replay extractor, live flushing, the log path layout and the run manifest.

**Architecture:** The MonitorAgent is a pure `on_event` object fed by an injected halt source (replay: a recorded, possibly empty, halt log). The recorder stays a bus sink; it now writes a `Window` as references to the `seq`s of its already-logged `Bar` lines and expands them in `read_log`, so the event contract is unchanged, `read_log(write(x)) == x`, and `log_hash` stays deterministic (it is over the compact bytes). Manifest and path helpers live in a separate module beside the recorder.

**Tech Stack:** Python 3.13, stdlib only (`dataclasses`, `json`, `hashlib`, `zoneinfo` with the installed `tzdata`, `subprocess` for git), pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-intraday-system-design.md` (§2 agent table: MonitorAgent, RecorderAgent; §5 phase 2 exit criterion). Binding design docs: `research/intraday/docs/design/monitor.md`, `research/intraday/docs/design/recorder.md`; read with `engine.md`, `execution.md` (Halt/Kill, partial fills), `risk.md` (Halt/Kill, partial fills), the contracts (`research/intraday/contracts/*.py`) and ADR-001.

## Window log decision (closes recorder.md's OPEN DECISION)

- **Decided:** a `Window` line is written compactly when, and only when, **every** one of its bars equals a `Bar` already written to this log (looked up by `(symbol, bar_start)`, latest line wins, then checked for full equality). The compact event record replaces `"bars"` with `"bar_seqs"`, the `seq`s of those `Bar` lines, e.g. `{"bar_seqs":[0,1,2],"symbol":"TEST","ts":"2026-03-02T14:31:00+00:00","type":"Window","v":1}`. Otherwise the `Window` is written inline, exactly as in Phase 1.
- **Format/versioning:** no version field is added to log lines. Every non-`Window` line and every inline `Window` line is byte-identical to Phase 1, so Phase-1 logs remain readable and existing hashes of Window-free logs are unchanged. The compact form is self-identifying (the `"bar_seqs"` key). The format generation is recorded explicitly as `LOG_FORMAT = 2` in `research/intraday/ops/recorder.py` and in every run manifest (`"log_format": 2`); 1 = Phase-1 inline-only.
- **Contract:** unchanged (`Window.bars` is still `tuple[Bar, ...]`; `EVENT_TYPES`, `SCHEMA_VERSION`, codec untouched).
- **Size:** a compact `Window` line at `k = 30` is about 250 bytes instead of about 7 KB, so a 50-symbol session goes from about 135 MB to about 10 MB (bars plus compact windows).
- **Checked:** no existing test pins a literal hash; `tests/test_intraday_recorder.py` and `tests/test_intraday_replay.py` compare hashes only with each other and record no `Window`, so their bytes do not change. `run_replay` is untouched and keeps working.

## Global Constraints

- Execution worktree: `C:/Users/Shachar/Desktop/PythonProjects/QuantCode-intraday-ops`, branch `research/intraday-ops`, created from `research/intraday` (`git worktree add -b research/intraday-ops ../QuantCode-intraday-ops research/intraday`). Never push. Never merge to main.
- Write only `research/intraday/ops/`, `tests/test_intraday_recorder.py` (do not touch it in this plan), and `tests/test_intraday_ops*`. Never edit contracts, engine, design docs or another agent's tests.
- Never edit an existing test assertion. If a change makes an existing test fail, stop and report; do not plan or make a test edit.
- Interpreter: `PY="C:/Users/Shachar/AppData/Local/Programs/Python/Python313/python.exe"` (not on PATH). All commands run from the worktree root.
- Contract field types are strict: `type(x) is int` for ints, `float` (`0.0`, never `0`) for floats, non-empty `str` for symbols/ids, `tuple` (never list/deque) for tuple fields, `datetime` with `tzinfo is timezone.utc` (convert with `.astimezone(timezone.utc)`, never pass a `ZoneInfo`-aware datetime into an event).
- Never emit while iterating a `set`/`frozenset`; iterate sorted tuples.
- No wall clock (`datetime.now`, `time.time`, …), no `uuid4`/`random`, no `alpaca.trading` anywhere under `research/intraday/` (`tests/test_intraday_guards.py`).
- `research/intraday/ops/` imports only `research.intraday.contracts`, `research.intraday.ops`, stdlib. It never imports `research.intraday.engine` (engine imports `ops.recorder`; the reverse would be a cycle and a layering break). Tests may import anything.
- Time comes from `ClockTick`. Session times are America/New_York, converted with `zoneinfo.ZoneInfo("America/New_York")` per session date.
- Halt source in Phase 2 is the recorded halt log only (empty for days before recording began). The live Nasdaq halt RSS adapter and the "RSS unreachable → Kill" rule are Phase 5.
- Tests use `research.intraday.contracts.fixtures` (`T_DEV`, `make_*`) for every timestamp tied to a session; `T_DEV` = 2024-03-04 14:31 UTC = 09:31 EST, the first session tick.
- The full suite passes before each commit: `$PY -m pytest tests/ -q -k intraday` and then `$PY -m pytest tests/ -q` (all green).
- Before reporting done: `$PY scripts/check_ownership.py --agent intraday-ops --base research/intraday` prints nothing.
- Every commit message ends with these two lines (in this order):
  `Refs: HananyaS/QuantCode-intraday-board#25`
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **Daylight saving time.** A session in July opens at 13:30 UTC, not 14:30. Expected: the stale baseline and the feed-session window follow `America/New_York` for the session date. Pinned in Task 5, `test_stale_baseline_follows_dst`.
2. **Candidates arriving after the open** (a late screen). The feed emits nothing before `Candidates`, so a 09:30 baseline would kill on the first check. Expected: baseline = the later of 09:30 ET and `Candidates.ts`. Pinned in Task 5, `test_late_candidates_use_their_ts_as_baseline`.
3. **A `Window` whose bars were not all logged as `Bar` lines, or differ from them** (built from another source, or a corrected bar). Expected: written inline, still round-trips. Pinned in Task 2, `test_window_with_unlogged_or_differing_bar_written_inline`.
4. **Exit fills that carry the entry's `intent_id`** (the `sell` leg of a `buy` intent). Expected: they never count toward the entry overfill check; they are checked only by the net-position rule. Pinned in Task 6, `test_exit_fill_on_entry_intent_is_not_an_overfill`.
5. **An operator `Kill` published externally by the live runner.** Expected: the monitor latches and never emits a second `Kill` (at most one per session). Pinned in Task 7, `test_external_kill_silences_monitor`.

---

## File Structure

```
research/intraday/ops/recorder.py     MODIFY: LOG_FORMAT, EXTERNAL_SOURCE, compact Window write, strict read_log, external_envelopes, flush
research/intraday/ops/manifest.py     CREATE: LOG_ROOT, session_log_path, config_hash, git_state, manifest_path, write_manifest
research/intraday/ops/halts.py        CREATE: HaltRow, HaltSource protocol, RecordedHaltLog
research/intraday/ops/monitor.py      CREATE: MonitorAgent
tests/test_intraday_ops_recorder.py   CREATE: recorder + manifest tests
tests/test_intraday_ops_monitor.py    CREATE: monitor tests
```

`tests/test_intraday_recorder.py` stays byte-for-byte unchanged.

---

### Task 1: Strict reader, every-type round trip, nightly-replay extractor

**Files:**
- Modify: `research/intraday/ops/recorder.py` (`read_log`, `_parse_line`; add `EXTERNAL_SOURCE`, `external_envelopes`)
- Test: `tests/test_intraday_ops_recorder.py`

**Interfaces:**
- Consumes: `JsonlRecorder(path)`, `read_log(path)`, `log_hash(path)` (Phase 1); `EventBus` from `research.intraday.engine.bus` (tests only).
- Produces:
  - `EXTERNAL_SOURCE: str = "external"` (must equal `engine.bus.EXTERNAL`; ops may not import the engine, so a test pins the equality).
  - `read_log(path: Path) -> Iterator[Envelope]` now also raises `ValueError("malformed log line {n}: ...")` when line `n`'s `seq != n - 1`, or its `cause_seq` is not `None` and not `0 <= cause_seq < seq` (it must name an earlier line).
  - `external_envelopes(path: Path) -> Iterator[Envelope]`: the envelopes with `source == EXTERNAL_SOURCE`, in log order.

- [ ] **Step 1: Write the failing tests** in `tests/test_intraday_ops_recorder.py`. Shared helpers in the file:

Imports this file needs over Tasks 1–3: `json`, `re`, `deque`, `replace`, `date`, `datetime`, `timedelta`, `Path`, `Sequence`, `pytest`; from `research.intraday.contracts`: `EVENT_TYPES, SCHEMA_VERSION, Agent, Bar, Envelope, Event, Window`; `canonical_json` from `contracts.codec`; `T0, make_bar, make_candidates, make_clock_tick, make_fill, make_gap, make_halt, make_kill, make_order_intent, make_order_update, make_quote, make_signal, make_window` from `contracts.fixtures`; `EXTERNAL, EventBus` from `engine.bus`; `clock_ticks, run_replay` from `engine.replay`; the recorder and manifest names from `research.intraday.ops.recorder` and `research.intraday.ops.manifest`.

```python
class ToyAgent:  # emits one Signal per Bar
    name = "toy"
    subscribes: tuple[type[Event], ...] = (Bar,)
    def on_event(self, event: Event) -> tuple[Event, ...]:
        return (make_signal(ts=event.ts, symbol=event.symbol),)

def _publish_all(path: Path, events: Sequence[Event], agents: Sequence[Agent] = ()) -> list[Envelope]:
    with JsonlRecorder(path) as rec:
        bus = EventBus(list(agents), sink=rec)
        return [env for e in events for env in bus.publish(e)]

def _rewrite(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
```

Tests:

```python
def test_external_source_matches_bus() -> None:
    assert EXTERNAL_SOURCE == EXTERNAL  # engine.bus.EXTERNAL

def test_seq_contiguous_from_zero_in_log(tmp_path):
    p = tmp_path / "a.jsonl"
    bars = [make_bar(ts=T0 + i * MIN, bar_start=T0 + (i - 1) * MIN) for i in range(3)]
    envs = _publish_all(p, bars, [ToyAgent()])
    assert [e.seq for e in read_log(p)] == list(range(6))
    lines = p.read_text(encoding="utf-8").splitlines()
    q = tmp_path / "hole.jsonl"
    _rewrite(q, [lines[0], *lines[2:]])            # drop seq 1
    with pytest.raises(ValueError, match="line 2"):
        list(read_log(q))

def test_cause_seq_points_to_earlier_line(tmp_path):
    p = tmp_path / "a.jsonl"
    envs = _publish_all(p, [make_bar()], [ToyAgent()])
    assert envs[1].cause_seq == 0 and envs[0].cause_seq is None
    lines = p.read_text(encoding="utf-8").splitlines()
    for bad in (1, -1):                                 # itself; before the log
        rec = json.loads(lines[1]); rec["cause_seq"] = bad
        q = tmp_path / f"cause{bad}.jsonl"
        _rewrite(q, [lines[0], canonical_json(rec)])
        with pytest.raises(ValueError, match="line 2"):
            list(read_log(q))

def test_every_event_type_round_trips(tmp_path):
    w = make_window(3)                      # bars end T0-2m, T0-1m, T0; logged first
    inline_w = make_window(2, symbol="ZZZ") # its bars are never logged
    bracket = make_order_intent(intent_id="intent-2", limit_price=1.0, take_profit=1.2,
                                stop_loss=0.9, time_stop=T0 + 15 * MIN)
    events = [*w.bars, make_clock_tick(), make_candidates(), make_quote(), make_gap(),
              w, inline_w, make_signal(), make_order_intent(), bracket, make_order_update(),
              make_fill(), make_halt(), make_kill()]
    envs = _publish_all(tmp_path / "all.jsonl", events)
    assert {type(e.event).__name__ for e in envs} == set(EVENT_TYPES)
    assert list(read_log(tmp_path / "all.jsonl")) == envs

def test_external_events_extracted_in_order(tmp_path):
    p = tmp_path / "a.jsonl"
    b1, b2 = make_bar(symbol="AAA"), make_bar(symbol="BBB")
    _publish_all(p, [b1, b2], [ToyAgent()])
    ext = list(external_envelopes(p))
    assert [e.event for e in ext] == [b1, b2]
    assert [e.seq for e in ext] == [0, 2]
    assert all(e.source == EXTERNAL_SOURCE and e.cause_seq is None for e in ext)
```

(`MIN = timedelta(minutes=1)`; `T0` and the `make_*` factories come from `research.intraday.contracts.fixtures`.)

- [ ] **Step 2: Run them; expect failures.**
Run: `$PY -m pytest tests/test_intraday_ops_recorder.py -v`
Expected: collection fails with `ImportError` (`EXTERNAL_SOURCE`, `external_envelopes`). `test_every_event_type_round_trips` would already pass on Phase-1 code; it becomes the regression guard for Task 2.

- [ ] **Step 3: Implement** in `research/intraday/ops/recorder.py`: `EXTERNAL_SOURCE`; a `read_log` loop that tracks the expected next `seq` and raises through the existing `malformed log line {n}` path on a gap or a `cause_seq` outside `[0, seq)`; `external_envelopes` as a filter over `read_log`. `_parse_line`'s existing `ValueError` message format stays.

- [ ] **Step 4: Run the recorder tests plus the Phase-1 ones.**
Run: `$PY -m pytest tests/test_intraday_ops_recorder.py tests/test_intraday_recorder.py tests/test_intraday_replay.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, then commit.**
Run: `$PY -m pytest tests/ -q` → all pass.
```bash
git add research/intraday/ops/recorder.py tests/test_intraday_ops_recorder.py
git commit -m "feat(intraday-ops): strict read_log and the nightly-replay extractor" -m "Refs: HananyaS/QuantCode-intraday-board#25
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Compact Window encoding

**Files:**
- Modify: `research/intraday/ops/recorder.py` (`JsonlRecorder.__call__`, `read_log`; add `LOG_FORMAT`)
- Test: `tests/test_intraday_ops_recorder.py`

**Interfaces:**
- Consumes: Task 1's strict `read_log`.
- Produces:
  - `LOG_FORMAT: int = 2` (module docstring: "1 = Phase-1, inline Windows only; 2 = a Window may be written as `bar_seqs`").
  - `JsonlRecorder` keeps `self._bars: dict[tuple[str, datetime], tuple[int, Bar]]` (updated on every `Bar` envelope, latest wins) and writes a `Window` compactly iff every bar `b` has `self._bars[(b.symbol, b.bar_start)][1] == b`.
  - `read_log` keeps `bars_by_seq: dict[int, Bar]` and expands a compact `Window` record to the inline record (`"bars": [to_record(bars_by_seq[s]) for s in bar_seqs]`) before `from_record`, so all codec and contract validation is reused. A `bar_seqs` entry that is not the `seq` of an earlier `Bar` line raises `ValueError("malformed log line {n}: ...")`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_intraday_ops_recorder.py`). Helper agent:

```python
class WindowAgent:  # one symbol; emits the last k bars as a Window on each Bar
    name = "win"
    subscribes: tuple[type[Event], ...] = (Bar,)
    def __init__(self, k: int = 30) -> None:
        self._bars: deque[Bar] = deque(maxlen=k)
    def on_event(self, event: Event) -> tuple[Event, ...]:
        assert isinstance(event, Bar)
        self._bars.append(event)
        return (Window(ts=event.ts, symbol=event.symbol, bars=tuple(self._bars)),)

def _lines(p: Path) -> list[dict]:
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
```

```python
def test_window_line_references_bar_seqs(tmp_path):
    w = make_window(3)
    p = tmp_path / "w.jsonl"
    envs = _publish_all(p, [*w.bars, w])
    ev = _lines(p)[3]["event"]
    assert ev == {"bar_seqs": [0, 1, 2], "symbol": "TEST", "ts": T0.isoformat(),
                  "type": "Window", "v": SCHEMA_VERSION}
    assert list(read_log(p)) == envs

def test_window_with_unlogged_or_differing_bar_written_inline(tmp_path):
    w = make_window(3)
    p = tmp_path / "missing.jsonl"
    envs = _publish_all(p, [w.bars[0], w.bars[1], w])        # bars[2] never logged
    assert "bars" in _lines(p)[2]["event"] and "bar_seqs" not in _lines(p)[2]["event"]
    assert list(read_log(p)) == envs
    q = tmp_path / "differs.jsonl"
    other = replace(w.bars[2], close=1.03)                    # same key, different value
    envs = _publish_all(q, [w.bars[0], w.bars[1], other, w])
    assert "bars" in _lines(q)[3]["event"]
    assert list(read_log(q)) == envs

def test_compact_ref_to_non_bar_or_later_line_rejected(tmp_path):
    w = make_window(3)
    p = tmp_path / "w.jsonl"
    _publish_all(p, [*w.bars, make_signal(), w])              # Signal is seq 3, Window seq 4
    raw = p.read_text(encoding="utf-8").splitlines()
    for bad in ([0, 1, 3], [0, 1, 5]):                       # a Signal line; a later line
        rec = json.loads(raw[4]); rec["event"]["bar_seqs"] = bad
        q = tmp_path / f"bad{bad[-1]}.jsonl"
        _rewrite(q, [*raw[:4], canonical_json(rec)])
        with pytest.raises(ValueError, match="line 5"):
            list(read_log(q))

def test_compact_window_lines_are_small(tmp_path):
    bars = [make_bar(ts=T0 + i * MIN, bar_start=T0 + (i - 1) * MIN) for i in range(60)]
    p = tmp_path / "k30.jsonl"
    envs = _publish_all(p, bars, [WindowAgent(30)])
    win_lines = [s for s in p.read_text(encoding="utf-8").splitlines() if '"Window"' in s]
    assert len(win_lines) == 60
    assert max(len(s) for s in win_lines) < 400               # inline k=30 is ~7 KB
    assert list(read_log(p)) == envs

def test_replay_with_windows_identical_hash(tmp_path):
    def src() -> list[Bar]:
        return [make_bar(ts=T0 + i * MIN, bar_start=T0 + (i - 1) * MIN) for i in range(40)]
    h1 = run_replay([WindowAgent(30)], [clock_ticks(T0, T0 + 40 * MIN), src()], tmp_path / "a.jsonl")
    h2 = run_replay([WindowAgent(30)], [clock_ticks(T0, T0 + 40 * MIN), src()], tmp_path / "b.jsonl")
    assert h1 == h2
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert sum(isinstance(e.event, Window) for e in read_log(tmp_path / "a.jsonl")) == 40

def test_log_format_is_two() -> None:
    assert LOG_FORMAT == 2
```

(`replace` is `dataclasses.replace`; `run_replay`, `clock_ticks` from `research.intraday.engine.replay`.)

- [ ] **Step 2: Run; expect failures.**
Run: `$PY -m pytest tests/test_intraday_ops_recorder.py -v`
Expected: `ImportError: LOG_FORMAT`; once imported, `test_window_line_references_bar_seqs` and `test_compact_window_lines_are_small` FAIL (inline `bars`).

- [ ] **Step 3: Implement** the writer index and compact record, and the reader expansion, per the Interfaces block. The compact record is built as `{**to_record(window) without "bars", "bar_seqs": [seq, ...]}` and serialised with the existing `canonical_json`, so key order and float text match the codec exactly.

- [ ] **Step 4: Run recorder, replay and Phase-1 recorder tests.**
Run: `$PY -m pytest tests/test_intraday_ops_recorder.py tests/test_intraday_recorder.py tests/test_intraday_replay.py -v`
Expected: all PASS. If any Phase-1 test fails, stop and report (Global Constraints).

- [ ] **Step 5: Full suite, then commit.**
```bash
git add research/intraday/ops/recorder.py tests/test_intraday_ops_recorder.py
git commit -m "feat(intraday-ops): write a Window as refs to its logged bars (log format 2)" -m "Refs: HananyaS/QuantCode-intraday-board#25
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Live flushing, log path layout, run manifest

**Files:**
- Modify: `research/intraday/ops/recorder.py` (`JsonlRecorder.__init__` gains `flush`)
- Create: `research/intraday/ops/manifest.py`
- Test: `tests/test_intraday_ops_recorder.py`

**Interfaces:**
- Consumes: `log_hash`, `LOG_FORMAT` (recorder); `canonical_json` (contracts.codec).
- Produces:
  - `JsonlRecorder(path: Path, *, flush: bool = False)`: when `flush`, `self._fh.flush()` after every line (live). Default `False`, so `run_replay` is unchanged.
  - `LOG_ROOT: Path = Path("C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_intraday/logs")`.
  - `session_log_path(session_date: date, run_id: str, log_root: Path = LOG_ROOT) -> Path`: returns `log_root / f"{session_date:%Y-%m-%d}" / f"{run_id}.jsonl"`, creating the date directory (`parents=True, exist_ok=True`). `ValueError` if `run_id` is empty, or contains `/`, `\`, `..`, or `:`.
  - `config_hash(config: Mapping[str, object]) -> str`: `sha256(canonical_json(config).encode("utf-8")).hexdigest()`.
  - `git_state(repo: Path) -> tuple[str, bool]`: `(git rev-parse HEAD, git status --porcelain is non-empty)`, via `subprocess.run(..., check=True)`.
  - `manifest_path(log_path: Path) -> Path`: `log_path.with_name(f"{log_path.stem}.manifest.json")`.
  - `write_manifest(log_path: Path, *, git_sha: str, git_dirty: bool, config: Mapping[str, object], model_hash: str, unseal: bool) -> Path`: writes `canonical_json(record) + "\n"` with mode `"x"`, UTF-8, LF, where `record` has exactly the keys `config_hash, git_dirty, git_sha, log_format, log_hash, model_hash, run_id, unseal` (`run_id` = `log_path.stem`, `log_hash` = `log_hash(log_path)` read from the closed log). `ValueError` unless `git_sha` is 40 lowercase hex and `model_hash` is a non-empty `str`. Returns the manifest path.

- [ ] **Step 1: Write the failing tests.**

```python
def test_flush_makes_each_line_visible_before_close(tmp_path):
    p = tmp_path / "live.jsonl"
    env = EventBus([]).publish(make_bar())[0]
    rec = JsonlRecorder(p, flush=True)
    rec(env)
    assert p.read_text(encoding="utf-8").count("\n") == 1
    rec.close()

def test_session_log_path_layout(tmp_path):
    p = session_log_path(date(2024, 3, 4), "run-001", log_root=tmp_path)
    assert p == tmp_path / "2024-03-04" / "run-001.jsonl"
    assert p.parent.is_dir() and not p.exists()
    for bad in ("", "a/b", "a\\b", "..", "c:x"):
        with pytest.raises(ValueError):
            session_log_path(date(2024, 3, 4), bad, log_root=tmp_path)

def test_config_hash_is_key_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2.0}) == config_hash({"b": 2.0, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})

def test_manifest_records_sha_config_and_model_hash(tmp_path):
    p = tmp_path / "run-001.jsonl"
    _publish_all(p, [make_bar()])
    cfg = {"stale_after_min": 2, "arm": "screen"}
    m = write_manifest(p, git_sha="a" * 40, git_dirty=False, config=cfg,
                       model_hash="abc123", unseal=False)
    assert m == tmp_path / "run-001.manifest.json"
    assert json.loads(m.read_text(encoding="utf-8")) == {
        "config_hash": config_hash(cfg), "git_dirty": False, "git_sha": "a" * 40,
        "log_format": 2, "log_hash": log_hash(p), "model_hash": "abc123",
        "run_id": "run-001", "unseal": False}
    with pytest.raises(FileExistsError):
        write_manifest(p, git_sha="a" * 40, git_dirty=False, config=cfg,
                       model_hash="abc123", unseal=False)
    with pytest.raises(ValueError):
        write_manifest(tmp_path / "x.jsonl", git_sha="nothex", git_dirty=False,
                       config=cfg, model_hash="abc123", unseal=False)

def test_git_state_reports_sha_and_dirty() -> None:
    sha, dirty = git_state(Path(__file__).resolve().parents[1])
    assert re.fullmatch(r"[0-9a-f]{40}", sha) and isinstance(dirty, bool)
```

(The `ValueError` case on `x.jsonl` is raised before the log is read, so the missing file is irrelevant; validate arguments first.)

- [ ] **Step 2: Run; expect failures.**
Run: `$PY -m pytest tests/test_intraday_ops_recorder.py -v`
Expected: `ImportError` for `research.intraday.ops.manifest`; `TypeError` for `flush=`.

- [ ] **Step 3: Implement** `flush` in `JsonlRecorder` and the six names in `research/intraday/ops/manifest.py` per the Interfaces block.

- [ ] **Step 4: Run.**
Run: `$PY -m pytest tests/test_intraday_ops_recorder.py tests/test_intraday_recorder.py tests/test_intraday_replay.py tests/test_intraday_guards.py -v`
Expected: all PASS (the guard confirms no wall clock / no engine import in ops).

- [ ] **Step 5: Full suite, then commit.**
```bash
git add research/intraday/ops/recorder.py research/intraday/ops/manifest.py tests/test_intraday_ops_recorder.py
git commit -m "feat(intraday-ops): live flush, session log path and run manifest" -m "Refs: HananyaS/QuantCode-intraday-board#25
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Halt source and halt release

**Files:**
- Create: `research/intraday/ops/halts.py`, `research/intraday/ops/monitor.py`
- Test: `tests/test_intraday_ops_monitor.py`

**Interfaces:**
- Produces (`halts.py`):
  - `@dataclass(frozen=True, slots=True) class HaltRow: ts: datetime; symbol: str; halted: bool; reason: str`, validating in `__post_init__` with `ValueError`: `ts` a `datetime` with `tzinfo is timezone.utc`, `symbol` non-empty `str`, `type(halted) is bool`, `reason` a `str`.
  - `class HaltSource(Protocol): def due(self, ts: datetime) -> tuple[HaltRow, ...]` — every row not yet returned whose `ts <= ts`, in source order.
  - `class RecordedHaltLog: __init__(self, rows: Sequence[HaltRow])`; `due` walks a cursor forward while `rows[i].ts <= ts` (no sorting; order is checked by the monitor). `RecordedHaltLog(())` is the replay default for days before recording began. (A halt-log file loader is Phase 5, when recording starts.)
- Produces (`monitor.py`), the agent's construction surface used by Tasks 5–7:
  ```python
  class MonitorAgent:
      name = "monitor"
      subscribes = (ClockTick, Candidates, Bar, Gap, OrderIntent, OrderUpdate, Fill, Kill)
      def __init__(self, halt_source: HaltSource, *, stale_after_min: int = 2,
                   session_open_et: time = time(9, 30), session_close_et: time = time(16, 0),
                   infer_halts_from_gaps: bool = False, gap_halt_min: int = 5) -> None
      def on_event(self, event: Event) -> tuple[Event, ...]
      @property
      def counts(self) -> dict[str, int]   # copy; keys "skipped_resume", "skipped_duplicate_halt"
  ```
  Construction asserts (`ValueError`): `stale_after_min` and `gap_halt_min` positive `int`, `session_open_et < session_close_et`.
  Halt release on `ClockTick(T)`: for each row of `halt_source.due(T)` in order: `ValueError("halt log out of time order ...")` if `row.ts < ` the last released row's `ts` or `row.ts > T`; a resume (`halted=False`) for a symbol not halted is skipped and counted `skipped_resume`; a halt for a symbol already halted is skipped and counted `skipped_duplicate_halt`; otherwise emit `Halt(ts=T, symbol=row.symbol, halted=row.halted, reason=row.reason)` and update the halted flag. Rows are released for every tick, in or out of the feed session, and for any symbol.

- [ ] **Step 1: Write the failing tests** in `tests/test_intraday_ops_monitor.py`. Imports over Tasks 4–7: `datetime, timedelta, timezone`, `Any, Sequence`, `pytest`; from `research.intraday.contracts`: `Bar, Candidates, ClockTick, Event, Gap, Halt, Kill`; `T_DEV, make_bar, make_candidates, make_fill, make_gap, make_order_intent, make_order_update` from `contracts.fixtures`; `clock_ticks, run_replay` from `engine.replay`; `read_log` from `ops.recorder`; `HaltRow, RecordedHaltLog` from `ops.halts`; `MonitorAgent` from `ops.monitor`. Shared helpers:

```python
MIN = timedelta(minutes=1)
OPEN = T_DEV - MIN                     # 2024-03-04 14:30 UTC = 09:30 EST, first bar_start

def at(m: int) -> datetime: return OPEN + m * MIN      # at(1) == T_DEV, the 09:31 tick
def tick(m: int) -> ClockTick: return ClockTick(ts=at(m))
def bar(sym: str, m: int) -> Bar: return make_bar(symbol=sym, ts=at(m), bar_start=at(m - 1))
def gap(sym: str, m: int) -> Gap: return make_gap(symbol=sym, ts=at(m), missing_bar_start=at(m - 1))
def cands(*syms: str, m: int = -5) -> Candidates:
    return make_candidates(ts=at(m), symbols=tuple(sorted(syms)))
def monitor(rows: Sequence[HaltRow] = (), **kw: Any) -> MonitorAgent:
    return MonitorAgent(RecordedHaltLog(tuple(rows)), **kw)
def feed(mon: MonitorAgent, *events: Event) -> list[Event]:
    return [out for e in events for out in mon.on_event(e)]
```

```python
def test_halt_rows_released_at_first_tick_at_or_after_row_time():
    rows = [HaltRow(at(10) + timedelta(seconds=30), "AAA", True, "LUDP"),
            HaltRow(at(15), "AAA", False, "LUDP")]
    mon = monitor(rows)
    assert feed(mon, tick(10)) == []
    assert feed(mon, tick(11)) == [Halt(ts=at(11), symbol="AAA", halted=True, reason="LUDP")]
    assert feed(mon, tick(14)) == []
    assert feed(mon, tick(15)) == [Halt(ts=at(15), symbol="AAA", halted=False, reason="LUDP")]

def test_halt_ts_equals_tick_ts():
    mon = monitor([HaltRow(at(-20), "BBB", True, "T1")])     # before the first tick seen
    (h,) = feed(mon, tick(-5))
    assert isinstance(h, Halt) and h.ts == at(-5)

def test_resume_without_halt_skipped():
    mon = monitor([HaltRow(at(2), "AAA", False, "LUDP"),
                   HaltRow(at(3), "AAA", True, "LUDP"), HaltRow(at(3), "AAA", True, "LUDP")])
    assert feed(mon, tick(2)) == []
    assert [e.halted for e in feed(mon, tick(3))] == [True]
    assert mon.counts == {"skipped_resume": 1, "skipped_duplicate_halt": 1}

def test_out_of_order_halt_log_raises():
    mon = monitor([HaltRow(at(5), "AAA", True, "x"), HaltRow(at(3), "BBB", True, "x")])
    with pytest.raises(ValueError, match="out of time order"):
        feed(mon, tick(5))

def test_halt_row_validates_fields():
    for bad in (dict(ts=at(1).replace(tzinfo=None)), dict(symbol=""), dict(halted=1)):
        with pytest.raises(ValueError):
            HaltRow(**{**dict(ts=at(1), symbol="AAA", halted=True, reason="x"), **bad})
```

- [ ] **Step 2: Run; expect failures.**
Run: `$PY -m pytest tests/test_intraday_ops_monitor.py -v`
Expected: `ModuleNotFoundError: research.intraday.ops.halts`.

- [ ] **Step 3: Implement** `halts.py` and the `MonitorAgent` skeleton (constructor, `counts`, `ClockTick` → halt release; every other subscribed type returns `()` for now).

- [ ] **Step 4: Run.**
Run: `$PY -m pytest tests/test_intraday_ops_monitor.py tests/test_intraday_guards.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, then commit.**
```bash
git add research/intraday/ops/halts.py research/intraday/ops/monitor.py tests/test_intraday_ops_monitor.py
git commit -m "feat(intraday-ops): MonitorAgent halt-log release" -m "Refs: HananyaS/QuantCode-intraday-board#25
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Stale-feed kill

**Files:**
- Modify: `research/intraday/ops/monitor.py`
- Test: `tests/test_intraday_ops_monitor.py`

**Interfaces:**
- Consumes: Task 4's `MonitorAgent` surface.
- Produces (behaviour; no new public names):
  - **Session date** = the America/New_York date of the first `ClockTick` or `Candidates` seen. A later `ClockTick` on another ET date raises `ValueError("one run is one session ...")`. A second `Candidates` raises `ValueError` (mirrors the FeedAgent: exactly one per session).
  - `open_utc` = `datetime.combine(session_date, session_open_et, ZoneInfo("America/New_York")).astimezone(timezone.utc)`; `close_utc` likewise. The stale check runs on ticks with `open_utc + 1 min <= T <= close_utc` (09:31–16:00 ET).
  - **Baseline** = `max(open_utc, Candidates.ts)`, used as `last` for a candidate until its first `Bar`/`Gap`. `last` for a symbol = the `ts` of its latest `Bar` or `Gap`.
  - At tick `T` in the window, stale symbols = candidates (in `Candidates.symbols` order, already sorted) with `(T - 1 min) - last >= stale_after_min` minutes. If any: emit, after that tick's `Halt`s, one `Kill(ts=T, reason=...)` with reason exactly `f"stale_feed: {', '.join(f'{s} last={last.isoformat()}' for s in stale)} at {T.isoformat()} (>= {stale_after_min} min)"`, and latch (Task 7 tests the silence).
  - No `Candidates` yet, or an empty one: no stale check. Halted symbols are not exempt (the feed emits `Gap`s for them).

- [ ] **Step 1: Write the failing tests.**

```python
def test_first_session_tick_does_not_kill():
    mon = monitor()
    assert feed(mon, tick(-5), cands("AAA"), tick(1)) == []   # no bar has arrived yet

def test_stale_feed_emits_kill():
    mon = monitor()
    evs = [tick(-5), cands("AAA", "BBB")]
    for m in range(1, 6):
        evs += [tick(m), bar("AAA", m), gap("BBB", m)]
    assert feed(mon, *evs) == []
    assert feed(mon, tick(6), bar("BBB", 6), tick(7)) == []
    out = feed(mon, tick(8))                                     # AAA last = at(5): at(7) - at(5) = 2
    assert out == [Kill(ts=at(8), reason=(f"stale_feed: AAA last={at(5).isoformat()} "
                                          f"at {at(8).isoformat()} (>= 2 min)"))]

def test_stale_check_at_tick_uses_previous_minute():
    mon = monitor(stale_after_min=1)       # evaluated against T it would kill at every tick
    evs = [tick(-5), cands("AAA")] + [e for m in range(1, 31) for e in (tick(m), bar("AAA", m))]
    assert feed(mon, *evs) == []
    out = feed(monitor(stale_after_min=2), tick(-5), cands("AAA"), tick(1), tick(2), tick(3))
    assert [(type(e), e.ts) for e in out] == [(Kill, at(3))]   # at(2) - at(0) = 2; not at tick(2)

def test_full_coverage_never_kills():
    mon = monitor()
    evs = [tick(m) for m in range(-5, 1)] + [cands("AAA", "BBB", m=0)]
    for m in range(1, 391):                # bars 09:30..15:59, ticks to 16:00
        evs += [tick(m), bar("AAA", m), gap("BBB", m) if m % 7 == 0 else bar("BBB", m)]
    evs += [tick(391), tick(392), tick(393)]   # after the session: no check
    assert feed(mon, *evs) == []

def test_stale_baseline_follows_dst():
    open_ = datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc)   # 09:30 EDT
    t = lambda m: ClockTick(ts=open_ + m * MIN)
    mon = monitor()
    c = make_candidates(ts=open_ - 5 * MIN, symbols=("AAA",))
    assert feed(mon, t(-5), c, t(1), t(2)) == []
    assert [type(e) for e in feed(mon, t(3))] == [Kill]

def test_late_candidates_use_their_ts_as_baseline():
    mon = monitor()
    assert feed(mon, tick(60), cands("AAA", m=60), tick(61), tick(62)) == []
    (k,) = feed(mon, tick(63))
    assert k.reason.startswith(f"stale_feed: AAA last={at(60).isoformat()}")

def test_second_candidates_raises():
    mon = monitor()
    feed(mon, tick(-5), cands("AAA"))
    with pytest.raises(ValueError):
        feed(mon, cands("BBB", m=-4))

def test_second_session_date_raises():
    mon = monitor()
    feed(mon, tick(1))
    with pytest.raises(ValueError, match="one session"):
        feed(mon, ClockTick(ts=at(1) + timedelta(days=1)))
```

- [ ] **Step 2: Run; expect failures.**
Run: `$PY -m pytest tests/test_intraday_ops_monitor.py -v`
Expected: the stale tests FAIL (no `Kill`), the two raise tests FAIL (no raise); Task 4 tests still PASS.

- [ ] **Step 3: Implement** the session, baseline and stale check per the Interfaces block. Keep per-symbol `last` in a `dict[str, datetime]` updated on every `Bar`/`Gap`; read it only through the candidates tuple.

- [ ] **Step 4: Run.**
Run: `$PY -m pytest tests/test_intraday_ops_monitor.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, then commit.**
```bash
git add research/intraday/ops/monitor.py tests/test_intraday_ops_monitor.py
git commit -m "feat(intraday-ops): stale-feed kill evaluated against the previous minute" -m "Refs: HananyaS/QuantCode-intraday-board#25
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Order-flow consistency kill

**Files:**
- Modify: `research/intraday/ops/monitor.py`
- Test: `tests/test_intraday_ops_monitor.py`

**Interfaces:**
- Consumes: Task 4's surface; the kill latch from Task 5.
- Produces (behaviour). Books: per `intent_id` → `(symbol, side, qty, entry_filled, closed)`; per symbol → net position (`buy` +qty, `sell` −qty, over all fills). Rules, each emitting `Kill(ts=<cause ts>, reason)` and latching; the reason prefix is exact:
  - `OrderIntent` whose `intent_id` was seen → `"duplicate_intent: {intent_id}"`.
  - `OrderUpdate` with status `cancelled`, `expired` or `rejected` → mark the intent `closed` (no emission; an update for an unknown id is ignored; `accepted` changes nothing).
  - `Fill` checks in this order:
    1. unknown `intent_id` → `"unknown_intent: {intent_id}"`;
    2. `fill.side == intent.side` (an entry fill) on a `closed` intent → `"fill_after_close: {intent_id}"`;
    3. entry fill making `entry_filled > intent.qty` → `"overfill: {intent_id} filled {entry_filled} > qty {qty}"`;
    4. net position of `fill.symbol` below zero after the fill → `"short_sell: {symbol} position {pos}"`.
    Fills whose side differs from the intent's side are exits: rules 2–3 do not apply to them.
  - The filled qty stays booked after `cancelled` (partial-fill-then-cancel).

- [ ] **Step 1: Write the failing tests.**

```python
def test_fill_for_unknown_intent_kills():
    (k,) = feed(monitor(), make_fill(ts=T_DEV, intent_id="ghost"))
    assert k == Kill(ts=T_DEV, reason="unknown_intent: ghost")

def test_overfill_kills():
    mon = monitor()
    assert feed(mon, make_order_intent(ts=T_DEV, qty=100), make_fill(ts=T_DEV, qty=60)) == []
    (k,) = feed(mon, make_fill(ts=T_DEV + MIN, qty=50))
    assert k == Kill(ts=T_DEV + MIN, reason="overfill: intent-1 filled 110 > qty 100")

def test_partial_fill_then_cancel_books_filled_qty():
    mon = monitor()
    assert feed(mon, make_order_intent(ts=T_DEV, qty=100),
                make_fill(ts=T_DEV + MIN, qty=40),                       # Fill before cancel,
                make_order_update(ts=T_DEV + MIN, status="cancelled"),  # same cascade
                make_fill(ts=T_DEV + 5 * MIN, side="sell", qty=40)) == []   # exit of the 40
    (k,) = feed(mon, make_fill(ts=T_DEV + 6 * MIN, side="sell", qty=1))
    assert k.reason == "short_sell: TEST position -1"

def test_entry_fill_after_cancel_kills():
    mon = monitor()
    feed(mon, make_order_intent(ts=T_DEV), make_order_update(ts=T_DEV, status="cancelled"))
    (k,) = feed(mon, make_fill(ts=T_DEV + MIN, qty=10))
    assert k.reason == "fill_after_close: intent-1"

def test_exit_fill_on_entry_intent_is_not_an_overfill():
    mon = monitor()
    assert feed(mon, make_order_intent(ts=T_DEV, qty=100),
                make_fill(ts=T_DEV + MIN, qty=100),
                make_fill(ts=T_DEV + 9 * MIN, side="sell", qty=100)) == []

def test_short_creating_sell_kills():
    mon = monitor()
    feed(mon, make_order_intent(ts=T_DEV, intent_id="s-1", side="sell", qty=10))
    (k,) = feed(mon, make_fill(ts=T_DEV + MIN, intent_id="s-1", side="sell", qty=10))
    assert k == Kill(ts=T_DEV + MIN, reason="short_sell: TEST position -10")

def test_duplicate_intent_id_kills():
    mon = monitor()
    assert feed(mon, make_order_intent(ts=T_DEV)) == []
    (k,) = feed(mon, make_order_intent(ts=T_DEV + MIN))
    assert k == Kill(ts=T_DEV + MIN, reason="duplicate_intent: intent-1")
```

- [ ] **Step 2: Run; expect failures.**
Run: `$PY -m pytest tests/test_intraday_ops_monitor.py -v`
Expected: six of the seven new tests FAIL (no `Kill`); `test_exit_fill_on_entry_intent_is_not_an_overfill` passes trivially until the books exist and guards against an over-eager overfill rule.

- [ ] **Step 3: Implement** the books and rules per the Interfaces block (positions are `int`; a dataclass per intent is fine).

- [ ] **Step 4: Run.**
Run: `$PY -m pytest tests/test_intraday_ops_monitor.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, then commit.**
```bash
git add research/intraday/ops/monitor.py tests/test_intraday_ops_monitor.py
git commit -m "feat(intraday-ops): order-flow kill on filled-qty books" -m "Refs: HananyaS/QuantCode-intraday-board#25
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Kill latch, gap-inferred halts, replay determinism

**Files:**
- Modify: `research/intraday/ops/monitor.py`
- Test: `tests/test_intraday_ops_monitor.py`

**Interfaces:**
- Consumes: Tasks 4–6.
- Produces (behaviour):
  - Latch: after the monitor's own `Kill`, or on receiving any `Kill` (an operator kill published by the live runner; the bus never delivers the monitor its own), `on_event` returns `()` for everything, `Halt`s included, for the rest of the session.
  - With `infer_halts_from_gaps=True`: on the `gap_halt_min`-th consecutive `Gap` of a symbol that is not halted, emit `Halt(ts=gap.ts, symbol, halted=True, reason=f"inferred: {gap_halt_min} consecutive gaps")` and mark it halted-by-inference; on that symbol's next `Bar`, emit `Halt(ts=bar.ts, symbol, halted=False, reason="inferred: bar after gaps")`. A `Bar` resets the consecutive-gap count. Inferred resumes happen only for inferred halts; halt-log rows and inferred halts share the halted flag. Default `False`: `Gap`s never produce a `Halt`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_at_most_one_kill_and_silence_after():
    mon = monitor([HaltRow(at(5), "AAA", True, "x")])
    assert len(feed(mon, make_fill(ts=at(1), intent_id="ghost"))) == 1
    assert feed(mon, make_fill(ts=at(2), intent_id="ghost2"), cands("AAA", m=2),
                *[tick(m) for m in range(3, 10)]) == []          # no 2nd Kill, no Halt

def test_external_kill_silences_monitor():
    mon = monitor()
    assert feed(mon, Kill(ts=at(1), reason="operator")) == []
    assert feed(mon, make_fill(ts=at(2), intent_id="ghost")) == []

def test_gap_inference_off_by_default():
    mon = monitor()
    evs = [tick(-5), cands("AAA")] + [e for m in range(1, 8) for e in (tick(m), gap("AAA", m))]
    assert feed(mon, *evs) == []

def test_gap_inference_emits_halt_and_resume():
    mon = monitor(infer_halts_from_gaps=True, gap_halt_min=5)
    evs = [tick(-5), cands("AAA")] + [e for m in range(1, 6) for e in (tick(m), gap("AAA", m))]
    assert feed(mon, *evs) == [Halt(ts=at(5), symbol="AAA", halted=True,
                                    reason="inferred: 5 consecutive gaps")]
    assert feed(mon, tick(6), bar("AAA", 6)) == [
        Halt(ts=at(6), symbol="AAA", halted=False, reason="inferred: bar after gaps")]

def test_monitor_in_replay_identical_hash(tmp_path):
    rows = (HaltRow(at(3), "AAA", True, "LUDP"), HaltRow(at(8), "AAA", False, "LUDP"))
    def sources() -> list:
        return [clock_ticks(at(-5), at(12)), [cands("AAA", "BBB")],
                [b for m in range(1, 12) for b in (bar("AAA", m), bar("BBB", m))]]
    h1 = run_replay([MonitorAgent(RecordedHaltLog(rows))], sources(), tmp_path / "a.jsonl")
    h2 = run_replay([MonitorAgent(RecordedHaltLog(rows))], sources(), tmp_path / "b.jsonl")
    assert h1 == h2
    halts = [e.event for e in read_log(tmp_path / "a.jsonl") if isinstance(e.event, Halt)]
    assert [(h.ts, h.halted) for h in halts] == [(at(3), True), (at(8), False)]
    assert not any(isinstance(e.event, Kill) for e in read_log(tmp_path / "a.jsonl"))
```

(The bars source is sorted by `ts` then symbol, so it is ts-monotone for `merge_sources`.)

- [ ] **Step 2: Run; expect failures.**
Run: `$PY -m pytest tests/test_intraday_ops_monitor.py -v`
Expected: `test_external_kill_silences_monitor` and `test_gap_inference_emits_halt_and_resume` FAIL; `test_at_most_one_kill_and_silence_after` FAILs if Tasks 5–6 did not gate halt release on the latch; the others may already pass (regression guards).

- [ ] **Step 3: Implement** the `Kill` subscription latch and gap inference per the Interfaces block.

- [ ] **Step 4: Run the whole intraday suite and the ownership check.**
Run: `$PY -m pytest tests/ -q -k intraday` → all PASS.
Run: `$PY scripts/check_ownership.py --agent intraday-ops --base research/intraday` → prints nothing.

- [ ] **Step 5: Full suite, then commit.**
```bash
git add research/intraday/ops/monitor.py tests/test_intraday_ops_monitor.py
git commit -m "feat(intraday-ops): kill latch, gap-inferred halts, replay determinism" -m "Refs: HananyaS/QuantCode-intraday-board#25
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Hand-offs (not this builder's to write)

- `research/intraday/docs/design/recorder.md` (architect-owned): replace the OPEN DECISION and the "about 135 MB" State line with the decision above (`LOG_FORMAT = 2`, compact `bar_seqs`, inline fallback); add `test_intraday_ops_recorder.py`'s new test names to its Test plan.
- `research/intraday/docs/design/monitor.md` (architect-owned): record the resolutions listed in the hand-back (Kill subscription, late-Candidates baseline, fill-after-close rule, duplicate-halt skip, reason formats).
