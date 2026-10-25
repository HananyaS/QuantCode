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
  and during a publish the breadth-first queue, bounded by `max_cascade`.
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
| one run | one trading session, fresh agents, a new log path | builder's call (ADR-001 determinism; nothing carries across days except explicit inputs such as starting equity) |

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
- `run_replay` over the same agents and sources gives a byte-identical log, hence the same hash.
- No wall clock anywhere in `research/intraday/` (`tests/test_intraday_guards.py`).

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| External event older than the last external `ts` | `publish` check | raise (`ValueError`) |
| A source not ts-monotone | `merge_sources` check | raise (`ValueError`) |
| Emitted `ts` differs from the cause's | dispatch check | raise (`ValueError`) |
| Runaway cascade | count above `max_cascade` | raise (`RuntimeError`) |
| An agent or the sink raises | not caught | raise (propagates; the log is closed by `run_replay`, the run is void and is replayed from the start) |

## Replay vs live

The bus is the same. Replay: `run_replay` merges `clock_ticks` with any recorded sources and
publishes as fast as it can. Live (Phase 5): a live runner, outside the agents, publishes a
`ClockTick` each minute from the wall clock (the only place a wall clock is read, outside
`research/intraday/`'s guarded code) and lets adapters release buffered data on ticks. The nightly
check replays the live log's external envelopes through fresh agents and diffs the logs. A later
services split replaces only the transport (ADR-001).

## Test plan

Existing, `tests/test_intraday_bus.py`: `test_dispatch_is_breadth_first_in_registration_order`,
`test_agent_receives_only_subscribed_types`, `test_agent_never_receives_its_own_emissions`,
`test_emitted_event_must_carry_cause_ts`, `test_runaway_cascade_raises`,
`test_cascade_counter_resets_per_publish`, `test_non_monotone_external_publish_raises`,
`test_equal_ts_external_publish_is_allowed`, `test_sink_sees_every_envelope_with_increasing_seq`,
`test_duplicate_agent_names_rejected`, `test_empty_agent_name_rejected`,
`test_agents_frozen_at_construction`, `test_sink_exception_propagates`.

Existing, `tests/test_intraday_replay.py`: `test_clock_ticks_bounds_and_utc`,
`test_merge_orders_by_ts_then_source_index`, `test_merge_rejects_non_monotone_source`,
`test_same_day_replayed_twice_gives_identical_hash`, `test_changed_input_changes_hash`,
`test_replay_log_closed_on_agent_error`.

Integration (main session, after the Phase 2 merges), `tests/test_intraday_replay.py`:
`test_recorded_day_end_to_end_twice_identical_hash` (all seven agents on one recorded day).

## Open questions

- **[research]** The bus has no poison flag: after an exception it would accept further publishes
  although `seq` and the last external `ts` have advanced and queued events were dropped (ADR-001
  makes discarding it the caller's duty). Not blocking: `run_replay` abandons the run on any
  exception. Decide before the Phase 5 live runner whether the bus refuses publishes after a failure
  (an architect change with a test).
- **[research]** Session calendar: half days (13:00 ET close) and holidays need the tick range and
  the feed session per date. Phase 2 uses full days only; the calendar source is chosen with the
  integration replay.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/2
