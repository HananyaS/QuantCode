# Intraday Phase 1 — Contracts and Engine Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen event contracts, a deterministic in-process event bus, a replay driver and a hash-stable event recorder, plus the ownership map and agent definitions. After this, the data, risk, exec and ops builders can work in parallel against the contracts.

**Architecture:** Agents are pure `on_event(event) -> tuple[Event, ...]` objects and never see each other. The `EventBus` assigns sequence numbers, dispatches breadth-first in registration order, and enforces two rules: (1) an emitted event carries its cause's timestamp, and (2) source events are time-monotone. Every envelope goes to a JSONL recorder whose canonical encoding makes "same day replayed twice" a byte-equality check.

**Tech Stack:** Python 3.13, stdlib only (`dataclasses`, `json`, `hashlib`, `heapq`, `ast`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-intraday-system-design.md` (§2 runtime architecture, §3 development workflow, §5 phase 1).

## Global Constraints

- Branch `research/intraday` in worktree `C:\Users\Shachar\Desktop\PythonProjects\QuantCode-intraday`. Never merged to main.
- All code under `research/intraday/`. Tests are flat: `tests/test_intraday_<area>.py`.
- Interpreter: `PY="C:/Users/Shachar/AppData/Local/Programs/Python/Python313/python.exe"`. All commands below run from the worktree root.
- Type hints required. Frozen dataclasses. No `datetime.now()` / `time.time()` anywhere under `research/intraday/` (§2: "Time comes from a Clock event").
- Nothing under `research/intraday/` imports `alpaca.trading` or `*TradingClient`. The only future exception is `research/intraday/exec/alpaca_paper.py`, which does not exist in this phase.
- An agent package may import only `research.intraday.contracts`, its own package, stdlib, third-party libraries, and `research.penny`. It never imports another agent package.
- The full `tests/` suite passes before each commit. Tests are never edited to make them pass.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Tracker:** `HananyaS/QuantCode-intraday-board`, which is private. Phase 1 Tasks 1–8 are issues #12–#19, in order. Each commit message names its issue (`Refs: HananyaS/QuantCode-intraday-board#<n>`). When a task passes review, comment its commit SHA on the issue and close it (`gh issue close <n> -R HananyaS/QuantCode-intraday-board -c "<sha>: <summary>"`). The branch is local-only, so GitHub won't link commits on its own.
- **Service epics, for the `## Tracker` sections in Task 8:** contracts #1, engine #2, screener #3, feed #4, state #5, predictor #6, risk #7, execution #8, monitor #9, recorder #10, round-16 research #11.

## Review Focus

1. **Naive or non-UTC timestamps** (for example a tz-naive `datetime` from pandas). Expected: construction raises `ValueError`. Pinned in Task 2, `test_event_rejects_naive_and_non_utc_ts`.
2. **What a bar's timestamp means.** Alpaca stamps a bar at its *start*, but a bar is only knowable at its *end*. Expected: `Bar.ts == bar_start + 1 minute`, enforced, so a model can never act on a bar before it has closed. Pinned in Task 2, `test_bar_ts_is_end_of_bar`.
3. **An agent emitting an event with a timestamp other than its cause's**, which is how future information or latency fakery would get in. Expected: the bus raises. Pinned in Task 4, `test_emitted_event_must_carry_cause_ts`.
4. **Two agents ping-ponging events forever.** Expected: the bus raises after `max_cascade` events instead of hanging. Pinned in Task 4, `test_runaway_cascade_raises`.
5. **NaN or inf prices, and float round-trip in the log.** Expected: NaN/inf are rejected at construction, and a float written then read is bit-identical. Pinned in Task 3, `test_codec_round_trip_is_exact` and in Task 2, `test_bar_rejects_nan`.

---

## File Structure

```
research/intraday/__init__.py            package marker + docstring
research/intraday/README.md              what the package is, how to run tests
research/intraday/ownership.py           OWNERS: agent -> owned path prefixes (single source)
research/intraday/contracts/__init__.py  re-exports
research/intraday/contracts/events.py    SCHEMA_VERSION, Event base and all event types
research/intraday/contracts/codec.py     to_record / from_record / canonical_json
research/intraday/contracts/protocols.py Agent protocol, Envelope
research/intraday/contracts/fixtures.py  make_bar/make_quote/... factories for every workstream's tests
research/intraday/engine/__init__.py
research/intraday/engine/bus.py          EventBus
research/intraday/engine/replay.py       clock_ticks, merge_sources, run_replay
research/intraday/ops/__init__.py
research/intraday/ops/recorder.py        JsonlRecorder, read_log, log_hash
research/intraday/{data,state,model,risk,exec}/__init__.py   empty owned packages
research/intraday/docs/adr/0001-event-driven-single-process.md
scripts/check_ownership.py               lists files changed vs a base that an agent does not own
.claude/agents/intraday-{architect,data,model,risk,exec,ops,hypothesis-designer,critic}.md
tests/test_intraday_guards.py
tests/test_intraday_events.py
tests/test_intraday_codec.py
tests/test_intraday_bus.py
tests/test_intraday_recorder.py
tests/test_intraday_replay.py
tests/test_intraday_ownership.py
```

---

### Task 1: Package skeleton and architectural guards

**Files:**
- Create: `research/intraday/__init__.py`, and `__init__.py` in `contracts/`, `engine/`, `ops/`, `data/`, `state/`, `model/`, `risk/`, `exec/`
- Test: `tests/test_intraday_guards.py`

**Interfaces:**
- Produces: `AGENT_PACKAGES: tuple[str, ...] = ("data", "state", "model", "risk", "exec", "ops")` in `research/intraday/__init__.py`. Task 1's guard tests and Task 7's ownership test import it from there.

- [ ] **Step 1: Write the failing tests.** Use an AST walk over `research/intraday/**/*.py`, the same pattern as `tests/test_breakout_etl.py:44-60`.

```python
INTRADAY = REPO_ROOT / "research" / "intraday"
TRADING_ALLOWLIST = {"research/intraday/exec/alpaca_paper.py"}

def test_intraday_package_exists():
    assert (INTRADAY / "contracts" / "__init__.py").is_file()

def test_no_trading_client_import_outside_allowlist():
    # offenders = modules importing alpaca.trading* or *TradingClient, minus TRADING_ALLOWLIST
    assert offenders == []

def test_no_wall_clock_calls():
    # flags ast.Call on attribute chains ending in now/utcnow/today on datetime/date,
    # and time.time / time.monotonic
    assert offenders == []

def test_agent_packages_do_not_import_each_other():
    # for each file under research/intraday/<pkg>/ with pkg in AGENT_PACKAGES:
    # any import of research.intraday.<other> where other in AGENT_PACKAGES and other != pkg
    assert offenders == []
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_guards.py -v`. Expected: `test_intraday_package_exists` FAILS. The others pass trivially on an empty tree, which is correct because they are tripwires.
- [ ] **Step 3: Create the package files.** `research/intraday/__init__.py` holds a docstring that names the spec path and the "never merge to main" rule, plus `AGENT_PACKAGES`. Every other `__init__.py` is empty.
- [ ] **Step 4: Run** `"$PY" -m pytest tests/test_intraday_guards.py -v`. Expected: 4 PASS. Then run `"$PY" -m pytest tests/ -q`. Expected: all pass.
- [ ] **Step 5: Commit** with message `feat(intraday): package skeleton and import/clock/isolation guards`.

---

### Task 2: Event types

**Files:**
- Create: `research/intraday/contracts/events.py`
- Test: `tests/test_intraday_events.py`

**Interfaces:**
- Produces, all `@dataclass(frozen=True, slots=True, kw_only=True)`, all subclassing `Event`:
  - `SCHEMA_VERSION: int = 1`
  - `Event(ts: datetime)`. `__post_init__` requires `ts.tzinfo` and `ts.utcoffset() == timedelta(0)`, otherwise `ValueError`.
  - `ClockTick(ts)`
  - `Candidates(ts, screen_id: str, symbols: tuple[str, ...])`. Symbols are unique and sorted.
  - `Bar(ts, symbol: str, bar_start: datetime, open: float, high: float, low: float, close: float, volume: int, vwap: float, trade_count: int, feed: Literal["sip", "iex"])`. Invariants: `ts == bar_start + timedelta(minutes=1)`; all prices finite and > 0; `low <= min(open, close)`; `high >= max(open, close)`; `volume >= 0`.
  - `Quote(ts, symbol, bid: float, ask: float, bid_size: int, ask_size: int)`. Invariants: finite, `0 < bid <= ask`, sizes >= 0.
  - `Gap(ts, symbol, missing_bar_start: datetime)`
  - `Window(ts, symbol, bars: tuple[Bar, ...])`. Invariants: non-empty, every `bar.symbol == symbol`, `bar_start` strictly increasing, `bars[-1].ts <= ts`.
  - `Signal(ts, symbol, score: float, horizon_min: int, model_hash: str)`. Invariants: score finite, `horizon_min > 0`, model_hash non-empty.
  - `OrderIntent(ts, intent_id: str, symbol, side: Literal["buy","sell"], qty: int, limit_price: float | None, take_profit: float | None, stop_loss: float | None, time_stop: datetime | None, reason: str)`. Invariants: `qty > 0`. For a buy with both brackets set, `stop_loss < take_profit`. `time_stop` is UTC-aware and `> ts` if set.
  - `OrderUpdate(ts, intent_id, status: Literal["accepted","rejected","cancelled","expired"], detail: str)`
  - `Fill(ts, intent_id, symbol, side, qty: int, price: float, cost_bps: float)`. `qty > 0`, `price > 0`, `cost_bps >= 0`.
  - `Halt(ts, symbol, halted: bool, reason: str)`
  - `Kill(ts, reason: str)`
  - `EVENT_TYPES: dict[str, type[Event]]`, keyed by class name. Used by the codec.

- [ ] **Step 1: Write the failing tests.**

```python
T0 = datetime(2026, 3, 2, 14, 31, tzinfo=timezone.utc)

def test_event_rejects_naive_and_non_utc_ts():
    with pytest.raises(ValueError): ClockTick(ts=datetime(2026, 3, 2, 14, 31))
    with pytest.raises(ValueError): ClockTick(ts=T0.astimezone(ZoneInfo("America/New_York")))

def test_bar_ts_is_end_of_bar():
    with pytest.raises(ValueError): make_bar(ts=T0, bar_start=T0)  # stamped at start
    assert make_bar(ts=T0, bar_start=T0 - timedelta(minutes=1)).ts == T0

def test_bar_rejects_nan():
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError): make_bar(close=bad)

def test_bar_rejects_inconsistent_ohlc():
    with pytest.raises(ValueError): make_bar(open=2.0, high=1.9, low=1.8, close=1.85)

def test_quote_rejects_crossed_book():
    with pytest.raises(ValueError): make_quote(bid=1.01, ask=1.00)

def test_window_rejects_future_and_unsorted_bars(): ...
def test_order_intent_bracket_and_time_stop_validation(): ...
def test_events_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError): make_bar().close = 9.0
def test_event_types_registry_covers_every_subclass():
    assert set(EVENT_TYPES.values()) == {c for c in all_subclasses(Event)}
```

  This test file uses `make_bar` and `make_quote` from Task 3's fixtures, so write the fixtures module in this task as well. Put the `make_*` factories in `research/intraday/contracts/fixtures.py` now. Signature pattern: `make_bar(**overrides) -> Bar`, with valid defaults (`symbol="TEST"`, `bar_start=T0-1min`, `ts=T0`, `open=1.00, high=1.05, low=0.98, close=1.02, volume=10_000, vwap=1.01, trade_count=120, feed="sip"`). There are `make_*` factories for `Bar`, `Quote`, `Window`, `Signal`, `OrderIntent` and `Fill`. `make_window(n, symbol, end_ts)` builds n consecutive valid bars.

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_events.py -v`. Expected: FAIL, because `research.intraday.contracts.events` cannot be imported.
- [ ] **Step 3: Implement `events.py` and `fixtures.py`.** Invariants are checked in `__post_init__` and raise `ValueError` with a message that names the field. Use a small helper `_finite_positive(name, x)`. Re-export everything from `contracts/__init__.py`.
- [ ] **Step 4: Run** the events tests (expected: all PASS), then the full suite.
- [ ] **Step 5: Commit** with message `feat(intraday): frozen event contracts with invariant checks`.

---

### Task 3: Canonical codec

**Files:**
- Create: `research/intraday/contracts/codec.py`
- Test: `tests/test_intraday_codec.py`

**Interfaces:**
- Consumes: `EVENT_TYPES` and `SCHEMA_VERSION` from Task 2.
- Produces:
  - `to_record(event: Event) -> dict[str, object]`. The record is `{"type": <class name>, "v": SCHEMA_VERSION, **fields}`. Datetimes are written with `.isoformat()` and the `+00:00` suffix. Tuples become lists, and a tuple of `Bar` becomes a list of nested records.
  - `from_record(record: Mapping[str, object]) -> Event`. Raises `ValueError` on an unknown `type` or a `v != SCHEMA_VERSION`.
  - `canonical_json(record: Mapping[str, object]) -> str`, which is `json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)`.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("event", [make_bar(), make_quote(), make_window(3), make_signal(),
                                   make_order_intent(), make_fill(), ClockTick(ts=T0),
                                   Kill(ts=T0, reason="x"), Halt(ts=T0, symbol="A", halted=True, reason="LUDP")])
def test_codec_round_trip_is_exact(event):
    assert from_record(json.loads(canonical_json(to_record(event)))) == event

def test_codec_preserves_awkward_floats():
    bar = make_bar(open=0.1 + 0.2, high=1.0000000000000002, low=0.30000000000000004, close=0.5)
    assert from_record(json.loads(canonical_json(to_record(bar)))).open == 0.1 + 0.2

def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

def test_from_record_rejects_unknown_type_and_version(): ...
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_codec.py -v`. Expected: FAIL (import error).
- [ ] **Step 3: Implement.** Drive the codec off `dataclasses.fields()` and the field type hints (`typing.get_type_hints`), so that new event types need no codec changes. Datetime fields are parsed with `datetime.fromisoformat`. Fields of type `tuple[Bar, ...]` recurse through `from_record`. Fields of type `tuple[str, ...]` become a `tuple`.
- [ ] **Step 4: Run** the codec tests (expected: PASS), then the full suite.
- [ ] **Step 5: Commit** with message `feat(intraday): canonical JSON codec for events`.

---

### Task 4: Agent protocol and EventBus

**Files:**
- Create: `research/intraday/contracts/protocols.py`, `research/intraday/engine/bus.py`
- Test: `tests/test_intraday_bus.py`

**Interfaces:**
- Consumes: `Event` (Task 2).
- Produces:
  - `class Agent(Protocol)` with `name: str`, `subscribes: tuple[type[Event], ...]`, and `def on_event(self, event: Event) -> tuple[Event, ...]: ...`
  - `@dataclass(frozen=True, slots=True) class Envelope: seq: int; cause_seq: int | None; source: str; event: Event`. `source` is the emitting agent's name, or `"external"` for published events.
  - `class EventBus` with:
    - `__init__(self, agents: Sequence[Agent], sink: Callable[[Envelope], None] | None = None, max_cascade: int = 10_000)`. Raises `ValueError` on duplicate agent names.
    - `publish(self, event: Event) -> list[Envelope]`. Returns every envelope produced by this publish, in dispatch order, the root envelope included.

Dispatch algorithm (not determined by the signature, so it is specified here):

```
publish(e): require e.ts >= self._last_external_ts, else ValueError("non-monotone source"); update it.
  queue = deque([Envelope(next_seq, None, "external", e)])
  while queue:
      env = queue.popleft(); sink(env); out.append(env)
      if len(out) > max_cascade: raise RuntimeError("cascade exceeded")
      for agent in agents (registration order):
          if isinstance(env.event, agent.subscribes):
              for emitted in agent.on_event(env.event):
                  if emitted.ts != env.event.ts: raise ValueError(f"{agent.name} emitted ts != cause ts")
                  queue.append(Envelope(next_seq, env.seq, agent.name, emitted))
```

  An agent never receives its own emitted events. The check is `env.source != agent.name`, which prevents trivial self-loops.

- [ ] **Step 1: Write the failing tests.** Toy agents are defined inside the test module, for example `Echo` (Bar → Signal) and `Counter` (records what it saw).

```python
def test_dispatch_is_breadth_first_in_registration_order(): ...   # asserts exact (seq, source, type) list
def test_agent_receives_only_subscribed_types(): ...
def test_agent_never_receives_its_own_emissions(): ...
def test_emitted_event_must_carry_cause_ts():
    # agent emits Signal with ts = cause.ts + 1min  -> ValueError naming the agent
def test_runaway_cascade_raises():
    # A: Signal->Kill, B: Kill->Signal  with max_cascade=50 -> RuntimeError
def test_non_monotone_external_publish_raises(): ...
def test_equal_ts_external_publish_is_allowed(): ...
def test_sink_sees_every_envelope_with_increasing_seq(): ...
def test_duplicate_agent_names_rejected(): ...
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_bus.py -v`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `protocols.py` and `bus.py` as specified above. Use `collections.deque`. Sequence numbers start at 0 and increase across publishes for the bus's lifetime.
- [ ] **Step 4: Run** the bus tests (expected: PASS), then the full suite.
- [ ] **Step 5: Commit** with message `feat(intraday): deterministic event bus with ts-causality and cascade guards`.

---

### Task 5: Recorder sink

**Files:**
- Create: `research/intraday/ops/recorder.py`
- Test: `tests/test_intraday_recorder.py`

**Interfaces:**
- Consumes: `Envelope` (Task 4); `to_record`, `from_record`, `canonical_json` (Task 3).
- Produces:
  - `class JsonlRecorder`. `__init__(self, path: Path)` opens the file in mode `"x"`, so an existing log is never overwritten. It also has `__call__(self, env: Envelope) -> None`, which writes one line, `canonical_json({"seq", "cause_seq", "source", "event": to_record(...)})` followed by `"\n"`. Plus `close()` and the context-manager methods.
  - `read_log(path: Path) -> Iterator[Envelope]`
  - `log_hash(path: Path) -> str`, the hex sha256 of the file bytes.
- Decision, recorded in ADR-001: the recorder is a bus **sink**, not an `Agent`, because it needs `seq`, `cause_seq` and `source`, which an agent never sees. This deviates from the spec's table on purpose.

- [ ] **Step 1: Write the failing tests.**

```python
def test_recorder_round_trips_envelopes(tmp_path): ...       # write via bus with sink, read_log == envelopes
def test_recorder_refuses_to_overwrite(tmp_path):
    p = tmp_path / "day.jsonl"; p.write_text("")
    with pytest.raises(FileExistsError): JsonlRecorder(p)
def test_log_is_lf_utf8_without_bom(tmp_path):
    raw = p.read_bytes(); assert not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw
def test_log_hash_changes_with_content(tmp_path): ...
```

  The newline test matters on Windows. Open the file with `open(path, "x", encoding="utf-8", newline="\n")`.
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_recorder.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `recorder.py`.
- [ ] **Step 4: Run** the recorder tests (expected: PASS), then the full suite.
- [ ] **Step 5: Commit** with message `feat(intraday): append-only JSONL event recorder and log hash`.

---

### Task 6: Replay driver and determinism

**Files:**
- Create: `research/intraday/engine/replay.py`
- Test: `tests/test_intraday_replay.py`

**Interfaces:**
- Consumes: `EventBus` (Task 4), `JsonlRecorder` and `log_hash` (Task 5), `ClockTick` (Task 2).
- Produces:
  - `clock_ticks(start: datetime, end: datetime, step: timedelta = timedelta(minutes=1)) -> Iterator[ClockTick]`. Inclusive of `start`, exclusive of `end`. Both must be UTC-aware.
  - `merge_sources(*sources: Iterable[Event]) -> Iterator[Event]`. A heap-merge by `(ts, source_index, position_in_source)`. Each source must be ts-monotone, otherwise `ValueError` naming the source index.
  - `run_replay(agents: Sequence[Agent], sources: Sequence[Iterable[Event]], log_path: Path) -> str`. Builds the bus with a `JsonlRecorder` sink, publishes `merge_sources(*sources)`, closes the log, and returns `log_hash(log_path)`.
- Ordering rule: at equal `ts`, a lower source index publishes first. The convention, documented in ADR-001, is that `clock_ticks` is passed as source 0, so that time-driven logic such as time stops and gap checks sees the tick before that minute's bars.

- [ ] **Step 1: Write the failing tests.**

```python
def test_clock_ticks_bounds_and_utc(): ...
def test_merge_orders_by_ts_then_source_index(): ...
def test_merge_rejects_non_monotone_source(): ...
def test_same_day_replayed_twice_gives_identical_hash(tmp_path):
    sources = lambda: [clock_ticks(T0, T0 + 30min), synthetic_bars("AAA", 30), synthetic_bars("BBB", 30)]
    h1 = run_replay([EchoAgent()], sources(), tmp_path / "a.jsonl")
    h2 = run_replay([EchoAgent()], sources(), tmp_path / "b.jsonl")
    assert h1 == h2
def test_changed_input_changes_hash(tmp_path): ...       # perturb one bar's close by 1e-9
```

  `synthetic_bars(symbol, n, seed)` is a helper in the test module. It builds valid consecutive bars with `make_bar` from a seeded `numpy.random.default_rng`.
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_replay.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `replay.py` with `heapq.merge` over keyed tuples, or an explicit heap. Check monotonicity per source inside a generator wrapper.
- [ ] **Step 4: Run** the replay tests (expected: PASS), then the full suite.
- [ ] **Step 5: Commit** with message `feat(intraday): replay driver -- one day replayed twice hashes identically`.

---

### Task 7: Ownership map, checker, agent definitions, ADR-001, README

**Files:**
- Create: `research/intraday/ownership.py`, `scripts/check_ownership.py`, `research/intraday/docs/adr/0001-event-driven-single-process.md`, `research/intraday/README.md`, and the eight files `.claude/agents/intraday-*.md`
- Test: `tests/test_intraday_ownership.py`

**Interfaces:**
- Produces:
  - `OWNERS: dict[str, tuple[str, ...]]`, repo-relative path prefixes with POSIX separators:
    - `intraday-architect`: `research/intraday/contracts/`, `research/intraday/engine/`, `research/intraday/docs/adr/`, `research/intraday/ownership.py`, `tests/test_intraday_events.py`, `tests/test_intraday_codec.py`, `tests/test_intraday_bus.py`, `tests/test_intraday_replay.py`, `tests/test_intraday_guards.py`, `tests/test_intraday_ownership.py`
    - `intraday-data`: `research/intraday/data/`, `tests/test_intraday_data`
    - `intraday-model`: `research/intraday/state/`, `research/intraday/model/`, `research/intraday/research/`, `tests/test_intraday_state`, `tests/test_intraday_model`
    - `intraday-risk`: `research/intraday/risk/`, `tests/test_intraday_risk`
    - `intraday-exec`: `research/intraday/exec/`, `tests/test_intraday_exec`
    - `intraday-ops`: `research/intraday/ops/`, `tests/test_intraday_recorder.py`, `tests/test_intraday_ops`
    - `intraday-hypothesis-designer`: `research/penny/PREREGISTRATION_R16.md`
    - `intraday-critic`: `research/intraday/docs/critiques/`
  - `owner_of(path: str) -> str | None`, returns the agent whose prefix matches.
  - `scripts/check_ownership.py`, used as `"$PY" scripts/check_ownership.py --agent intraday-risk --base research/intraday`. It prints every file in `git diff --name-only <base>...HEAD` that the agent doesn't own, and exits 1 if there are any.

- [ ] **Step 1: Write the failing tests.**

```python
def test_owned_prefixes_are_disjoint(): ...                 # no prefix of one agent is a prefix of another's
def test_every_intraday_python_file_has_one_owner():
    for p in (REPO_ROOT / "research/intraday").rglob("*.py"): assert owner_of(rel(p)) is not None, p
def test_every_agent_package_is_owned():                    # AGENT_PACKAGES from Task 1
def test_every_owner_has_an_agent_definition():
    for agent in OWNERS: assert (REPO_ROOT / ".claude/agents" / f"{agent}.md").is_file()
def test_agent_definitions_list_exactly_their_owned_paths():
    # each .md contains a line "Owns:" followed by the OWNERS prefixes, verbatim, one per line
```

  The `research/intraday/__init__.py` marker belongs to the architect. Add `research/intraday/__init__.py` to the architect's prefixes, and also `research/intraday/README.md`.
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_ownership.py -v`. Expected: FAIL.
- [ ] **Step 3: Write `ownership.py` and `check_ownership.py`.** The script uses `subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"], check=True)`.
- [ ] **Step 4: Write the eight agent definitions.** Use the frontmatter format of `.claude/agents/breakout-critic.md` on the `research/breakout` branch (read it with `git show research/breakout:.claude/agents/breakout-critic.md`). Each body contains:
  - one line naming the agent's single responsibility, taken from spec §2 and §3
  - an `Owns:` block that lists the prefixes verbatim
  - the rules: "read the contracts, never edit them; a contract change is a request to `intraday-architect` with a reason"
  - "TDD; never edit tests to pass"
  - "run `scripts/check_ownership.py` before reporting done"
  - the Global Constraints from this plan
  - for `intraday-critic`: `tools: Read, Write, Grep, Glob, Bash`, and its gates are G1 (design), G2 (result) and G3 (before paper). It adapts the checklist from `breakout-critic.md` (adjustment basis, decision-time leakage, controls, the account rather than per-trade results, the trial count) and adds checks for bar timestamp semantics and halt fills.
  - for `intraday-hypothesis-designer`: writes only, runs nothing, and must state each round-16 factor and the bar to pass from spec §4.
- [ ] **Step 5: Write ADR-001 and the README.** ADR-001 is about one page and covers: the context; the decision (one process, a bus whose transport can be swapped, pure agents); the ts-causality rule; the recorder being a sink; the tick-first ordering; and the consequences, including that a services split later means replacing the bus transport only. The README covers the layout, how to run `tests/test_intraday_*`, and a link to the spec.
- [ ] **Step 6: Run** `"$PY" -m pytest tests/test_intraday_ownership.py -v` (expected: PASS), then `"$PY" -m pytest tests/ -q` (expected: all pass). Then run `"$PY" scripts/check_ownership.py --agent intraday-architect --base research/penny-veto` and confirm it lists only the files this phase did not assign to the architect.
- [ ] **Step 7: Commit** with message `feat(intraday): ownership map, checker, agent definitions, ADR-001`.

---

### Task 8: Per-service design documents

Each service gets its own design document before its builder starts. A builder works from its own doc plus the contracts, and from nothing else.

**Files:**
- Create: `research/intraday/docs/design/TEMPLATE.md`, plus one doc per service, `research/intraday/docs/design/{screener,feed,state,predictor,risk,execution,monitor,recorder,engine}.md`
- Modify: `research/intraday/ownership.py`. `research/intraday/docs/design/` goes under `intraday-architect`.
- Test: `tests/test_intraday_design_docs.py`

**Interfaces:**
- Produces: every design doc has exactly these H2 sections, in this order: `## Responsibility` (one sentence), `## Consumes`, `## Emits` (event types from `contracts/events.py`, by exact class name), `## State` (what it holds and how big it can get), `## Configuration` (each parameter with its default and its source: spec, R16, or "builder's call"), `## Invariants`, `## Failure modes` (each with its detection and response: raise, emit `Kill`, emit `Gap`, or skip), `## Replay vs live` (what differs, which must be adapters only), `## Test plan` (test names), `## Open questions` (each tagged `blocks-build` or `research`), `## Tracker` (the GitHub issue URL of the service's epic).

- [ ] **Step 1: Write the failing tests.**

```python
SERVICES = ("screener","feed","state","predictor","risk","execution","monitor","recorder","engine")
SECTIONS = ("Responsibility","Consumes","Emits","State","Configuration","Invariants",
            "Failure modes","Replay vs live","Test plan","Open questions","Tracker")

def test_every_service_has_a_design_doc(): ...
def test_design_docs_have_exact_sections_in_order(): ...
def test_consumes_and_emits_name_only_real_event_types():
    # every backticked CamelCase token under Consumes/Emits is a key of EVENT_TYPES
def test_only_risk_emits_order_intent():
    # 'OrderIntent' appears under '## Emits' in risk.md and in no other doc
def test_tracker_section_links_an_issue():
    # matches https://github.com/HananyaS/QuantCode-intraday-board/issues/\d+
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_design_docs.py -v`. Expected: FAIL.
- [ ] **Step 3: Write the template and the nine docs from spec §2 and §4.** Notes per doc:
  - **risk.md**: TP/SL and the exit grid are R16 factors, so configuration lists them as `research` rather than as fixed values. The quote-size cap and the daily loss limit get defaults, marked "builder's call".
  - **execution.md**: SimBroker fill rules come from spec §4: one-bar latency, stops filled at the next bar's price including gaps, no fills during halts.
  - **predictor.md**: the model is a frozen artifact identified by `model_hash`. k and h are R16 factors.
  - **feed.md**: `ReplayFeed` comes now and `AlpacaStreamFeed` after Gate I. Missing minutes are emitted as `Gap`, never filled in.
- [ ] **Step 4: Run** the design-doc tests (expected: PASS), then the full suite.
- [ ] **Step 5: Commit** with message `docs(intraday): per-service design documents and their structure test`.

---

## Phase 1 exit check

- `"$PY" -m pytest tests/ -q` is green.
- `tests/test_intraday_replay.py::test_same_day_replayed_twice_gives_identical_hash` passes.
- The guards pass: no trading import, no wall clock, no imports between agent packages.
- `git log research/penny-veto..research/intraday --oneline` shows 8 commits, all touching only `research/intraday/`, `tests/test_intraday_*`, `scripts/check_ownership.py`, `.claude/agents/intraday-*`, and `docs/superpowers/`.

What comes after, and not in this plan: R16 pre-registration in parallel with critic G1, and then the four parallel builder plans (data, risk, exec(Sim), ops monitor), each written against these contracts.
