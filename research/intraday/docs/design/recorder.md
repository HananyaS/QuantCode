# Recorder sink design

Owner: `intraday-ops`. Module: `research/intraday/ops/recorder.py` (built in Phase 1:
JsonlRecorder, `read_log`, `log_hash`). Spec §2 (RecorderAgent), ADR-001. The spec's
"RecorderAgent" is **not an Agent**: it is a bus sink, because it must see `seq`, `cause_seq` and
`source`, which the bus assigns and an agent never receives (ADR-001).

## Responsibility

Append every envelope the bus delivers to an append-only, canonical JSONL log, which makes any session replayable and hashable.

## Consumes

Every event type (all of `EVENT_TYPES`), each wrapped in the bus envelope (seq, cause_seq,
source, event). The sink is called by the bus for each envelope before it is dispatched to
agents, so the log order is the dispatch order.

## Emits

Nothing on the bus. One line per envelope on disk.

## State

- One open file handle (mode `"x"`, UTF-8, LF). No in-memory buffer beyond the file object's.
- Size on disk: one line per event, a few hundred bytes each, except `Window`: a `Window` line
  repeats its `k` bars, so the log grows with `k`. At 50 symbols and `k = 30` a session is about
  135 MB, almost all of it `Window` lines (see the OPEN DECISION under Open questions).

## Configuration

| Parameter | Default | Source |
|---|---|---|
| line format | canonical JSON of `{seq, cause_seq, source, event}` (sorted keys, no spaces), LF, no BOM | spec §2 (same hash for two runs); built |
| open mode | `"x"` (never overwrite) | builder's call; built |
| log path | `<log_root>/<session_date>/<run_id>.jsonl` | builder's call |
| `log_root` | `C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_intraday/logs` | builder's call (data lives in the main checkout, gitignored) |
| run manifest | `<run_id>.manifest.json` beside the log: git SHA, dirty flag, config hash, `model_hash`, log hash | builder's call |
| live flushing | flush after every line | builder's call |

## Invariants

- Every envelope the bus delivers is written exactly once, in delivery order.
- `seq` runs 0, 1, 2, … with no hole; `cause_seq` is `None` for external events and otherwise a
  smaller `seq` already in the log.
- A log is never overwritten or edited; a second run gets a new path.
- `read_log` round-trips every line to an equal envelope; the bytes are platform-independent, so
  `log_hash` is stable across machines.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| The log path exists | `open(..., "x")` | raise (`FileExistsError`) |
| Write after close | file object | raise (`ValueError`) |
| Disk full or I/O error | `OSError` from write | raise (the bus propagates it; the run is void) |
| A malformed line on read | JSON or codec error | raise (`ValueError` with the line number) |
| A sink exception mid-cascade | the bus propagates it | raise (ADR-001: the run is abandoned and replayed from the start) |

## Replay vs live

The same sink. Live flushes each line so a crash keeps the log up to the last event; replay may
buffer. The live log is the input of the nightly replay: its external envelopes (`source ==
"external"`) are re-published in order, and the replayed log is diffed against the live one. Live
bars are FeedAgent emissions, not external, so that replay also needs a log-backed feed adapter
(`engine.md`, Open questions).

## Test plan

Existing, `tests/test_intraday_recorder.py`: `test_recorder_round_trips_envelopes`,
`test_recorder_refuses_to_overwrite`, `test_log_is_lf_utf8_without_bom`,
`test_log_hash_changes_with_content`, `test_write_after_close_raises`,
`test_read_log_rejects_malformed_line`.

New, `tests/test_intraday_ops_recorder.py`:
- `test_seq_contiguous_from_zero_in_log`
- `test_cause_seq_points_to_earlier_line`
- `test_every_event_type_round_trips` (one fixture per `EVENT_TYPES` entry)
- `test_manifest_records_sha_config_and_model_hash`
- `test_external_events_extracted_in_order` (the nightly-replay input)

## Open questions

- **[blocks-build]** OPEN DECISION (blocks the state/model build and the R16 cost estimate; does
  not block the data, risk, exec or ops builds): `Window` log size, about 135 MB per session at 50
  symbols and `k = 30`, because every `Window` line repeats its bars. Preferred: the recorder writes
  a `Window` compactly as references to the `seq`s of its already-logged bars and expands them on
  `read_log`; the contract is unchanged and the hash stays deterministic (it is over the compact
  bytes). To be decided in the ops Phase-2 plan; until then the full `Window` is written.
- **[research]** The manifest's contents and the nightly diff's format are needed only by Phase 5.
  The Phase 2 build writes the manifest fields above; the diff report is designed with the live runner.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/10
