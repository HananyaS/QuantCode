# MonitorAgent design

Owner: `intraday-ops`. Package: `research/intraday/ops/`. Spec §2 (agent table: halts, stale feed,
reconciliation mismatch, kill switch), §5 phase 5. Read with `research/intraday/contracts/` and
ADR-001.

## Responsibility

Watch the session for trading halts and for conditions under which trading must stop, and announce them as `Halt` and `Kill` events.

## Consumes

- `ClockTick`: releases halt-log entries due by the tick, and runs the stale-feed check.
- `Candidates`: the symbols the feed must cover each minute.
- `Bar`, `Gap`: per-symbol feed coverage (one of them per candidate per minute is expected).
- `OrderIntent`, `OrderUpdate`, `Fill`: the consistency checks on the order flow.

Halt data is not an event on the way in: an injected halt-source adapter supplies
`(ts, symbol, halted, reason)` rows.

## Emits

- `Halt` with `ts` = the releasing tick's `ts`, for each halt-source row with row ts in
  `(previous tick ts, tick ts]`: `symbol`, `halted` (true at the halt, false at the resumption),
  `reason` from the source (e.g. the halt code).
- `Kill` with `ts` = the cause's `ts` and a `reason` naming the rule and the evidence, at most
  once per session, on:
  - **stale feed**: at a tick `T` during the feed session, a candidate has had no `Bar` or `Gap`
    for `stale_after_min` minutes;
  - **order-flow inconsistency**: a `Fill` whose `intent_id` was never seen in an order intent;
    total fills for an `intent_id`'s entry side above the intent's `qty`; a `sell` fill that would
    take the symbol's net position below zero; an order intent whose `intent_id` repeats.

After its `Kill` the agent emits nothing else for the session.

## State

- Candidate set; per symbol, the `ts` of the latest `Bar`/`Gap`; per symbol, halted flag.
- Per `intent_id`: intent qty and filled qty by side; per symbol: net position from fills.
- The halt-source cursor; `killed` flag.
- Size: O(candidates + intents per session).

## Configuration

| Parameter | Default | Source |
|---|---|---|
| `stale_after_min` | `2` | builder's call |
| feed session for the stale check | ticks `09:31`–`16:00` ET (bars `09:30`–`15:59`) | builder's call (must match the FeedAgent) |
| halt source, replay | the recorded halt log; empty for days before recording began | spec §2 ("replay of the halt log") |
| halt source, live | Nasdaq Trader halt RSS, polled, buffered to the next tick | spec §2 |
| `infer_halts_from_gaps` | `False` | builder's call (see Open questions) |
| `gap_halt_min` (when inference is on) | `5` consecutive `Gap`s | builder's call (an LULD pause is 5 minutes) |

Reuse: `utils/alerting.py` (on main and this branch) for the live alert on `Kill`, called from
the live runner, never from `on_event`.

## Invariants

- At most one `Kill` per session, and nothing after it.
- Every `Halt(halted=False)` for a symbol follows a `Halt(halted=True)` for it.
- Halt rows are released at the first tick at or after their time, never earlier.
- In replay the agent reads no clock but `ClockTick`; the halt log is a fixed input.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| A candidate goes `stale_after_min` without a `Bar`/`Gap` | tick check | emit `Kill` |
| Fill for an unseen `intent_id`, overfill, or a short-creating sell | order-flow books | emit `Kill` |
| Repeated `intent_id` in an `OrderIntent` | seen-id set | emit `Kill` |
| Halt source row out of time order | cursor check | raise (the halt log is corrupt) |
| A resumption with no prior halt for the symbol | halted flag | skip the row (a log that starts mid-halt), and count it |
| Live: the halt RSS cannot be fetched for `stale_after_min` | adapter reports it | emit `Kill` (trading without halt data is not allowed live) |

## Replay vs live

The agent is the same; the halt-source adapter differs (recorded log versus the Nasdaq RSS). In
replay the stale-feed rule should never fire, since ReplayFeed emits a `Bar` or `Gap` for every
candidate minute; it firing in replay means a feed bug. Live, it is the disconnect detector. An
operator kill in live is an external `Kill` published by the live runner, not something this
agent decides.

## Test plan

`tests/test_intraday_ops_monitor.py`:
- `test_halt_rows_released_at_first_tick_at_or_after_row_time`
- `test_halt_ts_equals_tick_ts`
- `test_resume_without_halt_skipped`
- `test_stale_feed_emits_kill`
- `test_full_coverage_never_kills`
- `test_fill_for_unknown_intent_kills`
- `test_overfill_kills`
- `test_short_creating_sell_kills`
- `test_duplicate_intent_id_kills`
- `test_at_most_one_kill_and_silence_after`
- `test_out_of_order_halt_log_raises`
- `test_gap_inference_off_by_default`

## Open questions

- **[research]** Halt history for replay. No free historical halt feed exists for the days before
  recording starts, so for 2016–2026 the replay halt log is empty. The build does not depend on it:
  "nothing fills during a halt" still holds in replay because a halted symbol prints no bars and
  SimBroker fills only on bars, so the first fill after a halt is at the reopening bar's open, gap
  included. What is lost is the cancellation of working entries at the halt. Optional fallback:
  `infer_halts_from_gaps` (5 consecutive `Gap`s), off by default because thin names skip minutes
  without being halted. R16 states which is used; recording real halts starts with the live feed.
- **[blocks-build]** Broker reconciliation mismatch has no carrying event in the contracts (the
  broker's position snapshot never reaches the bus). Blocks the live reconciliation rule
  (Phase 5) only; the Phase 2 build implements the order-flow checks above. Needs an ADR and a
  contract change by `intraday-architect`.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/9
