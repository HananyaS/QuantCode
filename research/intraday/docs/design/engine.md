# Engine design (bus and replay driver)

Owner: `intraday-architect` (Ruling 9: the engine stays with the architect). Package:
`research/intraday/engine/` (`bus.py`: EventBus; `replay.py`: `clock_ticks`, `merge_sources`,
`run_replay`), built in Phase 1. Spec §2 (design rules), §5 phase 2 exit criterion, ADR-001.
The engine is not an Agent and is the only coupling between agents.

## Responsibility

Deliver every event to the agents that subscribe to it, in one deterministic order, and drive a recorded session from its sources so that two runs give the same log hash.

## Consumes

External events published by the driver, in `ts` order: the clock's `ClockTick`s (source 0) and
any other recorded source (in Phase 2 the FeedAgent produces market data from ticks, so ticks are
usually the only source). Agents' returned events re-enter the queue in the same cascade.

## Emits

- `ClockTick`, from `clock_ticks(start, end, step)`: one per `step` in `[start, end)`, UTC.
- Every agent-emitted event, wrapped in an envelope with the next `seq`, `cause_seq` = the cause's
  `seq` and `source` = the emitting agent's name, delivered to subscribers and to the sink.

## State

- EventBus: the agent tuple (frozen at construction), the sink, `seq`, the last external `ts`,
  the failure that poisoned it (if any), and during a publish the breadth-first queue, bounded by
  `max_cascade`.
- `merge_sources`: one look-ahead event per source (a lazy heap merge).
- Size: O(agents + cascade); nothing grows with the session.

## Configuration

| Parameter | Default | Source |
|---|---|---|
| `max_cascade` | `10_000` envelopes per publish | builder's call; built |
| tick `step` | 1 minute | spec §2 (time comes from the clock); built |
| session tick range | `[09:25, 16:02)` ET, converted to UTC per session date | builder's call: the screen fires at 09:25, the last bar (15:59) is released at 16:00, and the end-of-session pass runs at 16:01 |
| source order | `clock_ticks(...)` is source 0 | ADR-001 (tick-first ordering); built |
| agent order | the order given to `EventBus` | builder's call: screener, feed, state, predictor, risk, execution, monitor |
| `holdout_start` | `2025-08-07` | spec §4 (last 12 months sealed) / R16; provisional value per controller Ruling 10 until `PREREGISTRATION_R16.md` fixes it |
| `unseal` | `False` | spec §4 (the sealed holdout is read only by an explicit, recorded decision) |
| one run | one trading session, fresh agents, a new log path | builder's call (ADR-001 determinism; nothing carries across days except explicit inputs such as starting equity) |

Phase 2 real-data replays use only session dates before `holdout_start`. The session runner
(integration) refuses a date `>= holdout_start` unless `unseal=True`, and so does every data adapter
(screener, feed). `unseal` is recorded in the run manifest. `run_replay` itself stays data-agnostic,
because it sees events, not dates; the guard sits where dates enter.

Agent order matters only among agents subscribed to the same envelope: dispatch is breadth-first,
so every subscriber of an envelope sees it before any event it caused is delivered.

## Invariants

All built and tested in Phase 1:
- An emitted event's `ts` equals its cause's `ts` (ts causality: no agent can stamp the future).
- External publishes have non-decreasing `ts`.
- At equal `ts`, a lower source index publishes first, so a minute's tick precedes that minute's
  other external events.
- Dispatch is breadth-first in registration order; an agent never receives its own emissions; an
  agent receives only the types in its `subscribes`.
- The sink sees every envelope, `seq` strictly increasing by 1 from 0.
- `run_replay` over the same agents and sources gives a byte-identical log, hence the same hash,
  in the same process and across processes with different `PYTHONHASHSEED`s.
- No agent is named `external` (reserved for external publishes).
- After any exception inside a cascade the bus is poisoned: every later `publish` raises
  `RuntimeError` (ADR-001).
- No wall clock anywhere in `research/intraday/` (`tests/test_intraday_guards.py`).

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| External event older than the last external `ts` | `publish` check | raise (`ValueError`) |
| A source not ts-monotone | `merge_sources` check | raise (`ValueError`) |
| Emitted `ts` differs from the cause's | dispatch check | raise (`ValueError`) |
| Runaway cascade | count above `max_cascade` | raise (`RuntimeError`) |
| An agent or the sink raises | not caught | raise (propagates; the bus is poisoned, the log is closed by `run_replay`, the run is void and is replayed from the start) |
| `publish` after a failed cascade | poison flag | raise (`RuntimeError`) |
| An agent named `external` | `EventBus.__init__` | raise (`ValueError`) |

## Replay vs live

The bus is the same. Replay: `run_replay` merges `clock_ticks` with any recorded sources and
publishes as fast as it can. Live (Phase 5): a live runner, outside the agents, publishes a
`ClockTick` each minute from the wall clock (the only place a wall clock is read, outside
`research/intraday/`'s guarded code) and lets adapters release buffered data on ticks. The nightly
check replays the live log through fresh agents and diffs the logs; live bars are emitted by the
FeedAgent (envelope `source` = the feed's name), not published as `external`, so that replay needs
a log-backed feed adapter (Open questions). A later services split replaces only the transport
(ADR-001).

## Test plan

Existing, `tests/test_intraday_bus.py`: `test_dispatch_is_breadth_first_in_registration_order`,
`test_agent_receives_only_subscribed_types`, `test_agent_never_receives_its_own_emissions`,
`test_emitted_event_must_carry_cause_ts`, `test_runaway_cascade_raises`,
`test_cascade_counter_resets_per_publish`, `test_non_monotone_external_publish_raises`,
`test_equal_ts_external_publish_is_allowed`, `test_sink_sees_every_envelope_with_increasing_seq`,
`test_duplicate_agent_names_rejected`, `test_empty_agent_name_rejected`,
`test_agents_frozen_at_construction`, `test_sink_exception_propagates`,
`test_agent_named_external_rejected`, `test_bus_is_poisoned_after_a_failed_cascade`,
`test_bus_is_poisoned_after_sink_failure`, `test_bus_is_poisoned_after_runaway_cascade`,
`test_non_monotone_rejection_does_not_poison`.

Existing, `tests/test_intraday_replay.py`: `test_clock_ticks_bounds_and_utc`,
`test_merge_orders_by_ts_then_source_index`, `test_merge_rejects_non_monotone_source`,
`test_same_day_replayed_twice_gives_identical_hash`, `test_changed_input_changes_hash`,
`test_replay_log_closed_on_agent_error`,
`test_replay_hash_identical_across_processes_and_hash_seeds` (the same replay fixture run in two
subprocesses with `PYTHONHASHSEED=0` and `=12345`; identical log hash and bytes; part of the normal
suite, under a second).

Integration (main session, after the Phase 2 merges), `tests/test_intraday_replay.py`:
`test_recorded_day_end_to_end_twice_identical_hash` (all seven agents on one recorded day),
`test_session_runner_refuses_holdout_date_without_unseal`, and the same end-to-end day run in two
subprocesses with `PYTHONHASHSEED=0` and `=12345`, asserting an identical `log_hash`
(`test_recorded_day_identical_hash_across_hash_seeds`).

## Open questions

- **[blocks-build]** (Phase 5 only) Nightly replay of live mode. Live bars come from the FeedAgent
  (envelope `source` = the feed's name), not from `external`, so re-publishing only the external
  envelopes would replay ticks without market data. The nightly check needs a log-backed feed
  adapter that serves the live log's feed-emitted `Bar`/`Quote`/`Gap` events to a FeedAgent in
  replay. Does not block Phase 2 (replay feeds read the caches). Decided with the live runner.
- **[research]** Session calendar: half days (13:00 ET close) and holidays need the tick range and
  the feed session per date. Phase 2 uses full days only; the calendar source is chosen with the
  integration replay.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/2
