# StateAgent design

Owner: `intraday-model`. Package: `research/intraday/state/`. Spec §2 (agent table), §4 (factor k).
Read with `research/intraday/contracts/` and ADR-001. Builds after the R16 pre-registration is
committed (spec §3).

## Responsibility

Keep the last k bars per candidate symbol and emit them as a point-in-time `Window` each time a new bar closes.

## Consumes

- `Candidates`: the symbol set. Bars for any other symbol are ignored.
- `Bar`: appended to its symbol's buffer; may trigger a `Window`.
- `Gap`: recorded against its symbol's buffer (it is not a bar and is never turned into one).

## Emits

- `Window` on a `Bar` for a candidate symbol, once that symbol's buffer holds `k` bars and the
  span rule below passes. Fields: `ts` = the bar's `ts`, `symbol`, `bars` = the last `k` bars in
  `bar_start` order, the newest being the bar just received. Nothing else is emitted.

`Window` carries bars only. "Point-in-time features" (spec §2) are computed by the
PredictorAgent from `Window.bars` alone; every bar in a window has `ts <= Window.ts` (the contract
checks it), so no feature can see a later bar.

## State

- `symbols: frozenset[str]` from `Candidates`.
- Per symbol, a `collections.deque(maxlen=k)` of `Bar`s and the count of `Gap`s since the oldest
  bar in the deque.
- Size: at most `len(symbols) × k` bars, so `20 × 30 = 600` at the largest R16 cell.

`symbols` is for membership tests only. Never emit while iterating a set or frozenset (its order
changes with `PYTHONHASHSEED`, and so would the log hash); sort first. `Window.bars` must be built
as `tuple(deque)` at emit: the contract requires a `tuple` of `Bar` (a deque or a list is
rejected), and the copy freezes the window so later appends cannot change an emitted event.

## Configuration

| Parameter | Default | Source |
|---|---|---|
| `k` (bars in a window) | `15` (tests and smoke runs only) | R16 (k ∈ {5, 15, 30}, spec §4) |
| `max_span_min` (newest `bar_start` − oldest `bar_start` + 1 min, at most) | `2 × k` | builder's call |
| `require_contiguous` | `False` | builder's call (microcaps skip minutes; requiring contiguity would drop most windows) |

A window whose span exceeds `max_span_min` is not emitted; the buffer is kept, and a later bar
may produce a valid window once old bars fall out.

The bars are on the traded (unadjusted) basis (`feed.md`); the agent never rescales them.

Reuse: `research/penny/model/sequence.py` (`SequenceConfig`, `build_sequences`) shows the training-side
window layout. The training windows built by research scripts must use the same `k`, the same
span rule, the same gap handling and the same bar basis (the cache converted as ReplayFeed converts
it) as this agent, or training and serving differ.

## Invariants

- A `Window` holds exactly `k` bars of one symbol, strictly increasing `bar_start`, newest bar's
  `ts == Window.ts`.
- A `Window` is emitted only in response to a `Bar`, never to a `Gap` or a tick.
- No bar is invented, repeated or reordered; a `Gap` never becomes a bar.
- At most one `Window` per symbol per minute.
- `Window.bars` is a fresh `tuple` copied from the deque at emit; nothing is emitted from a set
  iteration.
- Bars from before `Candidates` or for non-candidates never enter a buffer.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| A bar's `bar_start` is not after the buffer's newest | comparison on append | raise (the feed broke ordering; the run is void) |
| A bar arrives for a symbol not in `Candidates` | set lookup | skip |
| Window span exceeds `max_span_min` | span check | skip the emission (keep the buffer) |
| A `Bar` arrives before any `Candidates` | state check | skip |
| A second `Candidates` in one session | state check | raise |

## Replay vs live

None. The agent is the same code in both, with no adapter: its inputs are bus events only.

## Test plan

`tests/test_intraday_state.py`:
- `test_no_window_until_k_bars`
- `test_window_has_last_k_bars_newest_last`
- `test_window_ts_equals_triggering_bar_ts`
- `test_gap_never_becomes_a_bar`
- `test_gap_does_not_emit_window`
- `test_span_over_max_suppresses_window`
- `test_non_candidate_bar_ignored`
- `test_out_of_order_bar_raises`
- `test_one_window_per_symbol_per_minute`
- `test_state_bounded_by_k_per_symbol`
- `test_window_bars_is_tuple_snapshot_of_deque`

## Open questions

- **[research]** k is an R16 factor (5, 15 or 30 bars). The builder takes it from configuration and
  tests with small k; R16 runs every level.
- **[research]** The gap policy (`require_contiguous`, `max_span_min`) changes which minutes can
  trade at all. R16 should pre-register it with the factors; the default above is used meanwhile.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/5
