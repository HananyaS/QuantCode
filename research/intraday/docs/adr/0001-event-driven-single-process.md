# ADR-001: Event-driven, single-process engine

Status: Accepted
Date: 2026-10-26

## Context

The intraday system must be deterministic (two runs of one recorded day give event logs
with the same hash), testable one agent at a time, and buildable by several agents in
parallel without drifting apart. A services split is attractive but would cost
infrastructure before any edge is shown (spec section 2).

## Decision

- **One process.** All agents run in a single process behind an `EventBus`.
- **Transport-swappable bus.** The bus sits behind an interface; in-memory today, ZeroMQ or
  Redis possible later. Agents see only `publish` and subscriptions.
- **Pure agents returning events.** An agent receives an event and returns the events it
  emits. It holds no reference to another agent, and the bus is the only coupling.
- **Timestamp causality.** An emitted event's `ts` equals the `ts` of its cause. External
  publishes (feeds, clock) have monotone non-decreasing `ts`. Time comes from `Clock`
  events, never the wall clock.
- **Tick-first ordering.** `clock_ticks` is source 0, so at equal `ts` the clock tick is
  delivered before any other source's events.
- **The recorder is a bus sink, not an `Agent`.** It must see `seq`, `cause_seq` and
  `source`, which the bus assigns and an agent never receives.
- **Fail loud.** Any exception during dispatch (non-monotone ts, ts-causality violation,
  cascade overflow) propagates to the caller. After one, the bus is unusable: `seq` and the
  last external `ts` have already advanced and queued events were dropped, so the run is
  abandoned and replayed from the start. A half-delivered event is never silently skipped.
  (The bus does not yet refuse further publishes after a failure; discarding it is the
  caller's duty.)

## Consequences

- Replay is deterministic, and per-agent tests feed recorded event fixtures and assert on
  what is emitted.
- Ownership can be disjoint (`research/intraday/ownership.py`): contracts are the only
  shared code, and a contract change goes through `intraday-architect` with an ADR.
- A later services split replaces only the bus transport; agents, contracts and tests do
  not change.
- The cost is that ordering rules (ts causality, tick-first) are load-bearing and are
  enforced by the bus and tested, not left to convention. Any ordering change needs a new ADR.
