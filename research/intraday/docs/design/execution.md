# ExecutionAgent design

Owner: `intraday-exec`. Package: `research/intraday/exec/`. Spec §2 (agent table), §4 (simulated
fills), Constraints (only the live adapter imports `alpaca.trading`). Read with
`research/intraday/contracts/` and ADR-001. Built against `FakeTradingClient`
(`tests/test_execution_agent.py`, on main and on this branch).

## Responsibility

Turn each `OrderIntent` into fills and order updates through a broker adapter, managing the bracket, idempotency and order state.

## Consumes

- `OrderIntent`: a new order. Acknowledged at once; filled no earlier than the next bar.
- `Bar`: the fill trigger (prices, the bracket trigger check) for the bar's symbol.
- `Quote`: the latest inside per symbol, for the size cap at fill and the spread in the cost.
- `ClockTick`: time-stop triggers, the entry wait limit, and the end-of-session pass.
- `Halt`: blocks fills in the symbol until `halted=False`; cancels a working entry.
- `Kill`: cancels working entries, exits every open position, rejects any later intent.

## Emits

- `OrderUpdate` with `ts` = the cause's `ts`:
  - `accepted` on a valid new intent;
  - `rejected` on a duplicate `intent_id`, an intent after `Kill`, a `sell` larger than the
    position, or a broker refusal (`detail` says which);
  - `cancelled` for a working entry on `Halt`/`Kill`, or for the unfilled remainder after a
    size-capped partial fill. In that case, in the same cascade, the `Fill` for the capped qty
    comes first and the `OrderUpdate(cancelled)` for the remainder after it, so the RiskAgent and
    the MonitorAgent book the filled partial position before they see the cancel; the cancel
    drops only the working remainder;
  - `expired` for an entry with no bar within `entry_max_wait_min`, or an exit still unfilled at the
    end of the session (only possible when the symbol is halted).
- `Fill` with `ts` = the filling `Bar`'s `ts` (or the end-of-session tick's `ts`), `intent_id` of the
  entry, `side` (`buy` for the entry, `sell` for its exit), `qty`, `price` = the bar reference
  price **before** costs, and `cost_bps` = the leg's cost, charged separately so it stays auditable.

### SimBroker fill rules (spec §4)

- **One-bar latency.** A decision at `ts = T` (an intent, a bracket trigger, a time stop, a kill)
  fills at the **open** of the first bar with `bar_start >= T`. Because a bar is stamped at its
  end, the bar that caused the decision (`bar_start = T − 1 min`) can never fill it.
- **Stops fill at the next bar's price, gaps included, never at the stop level.** After an entry
  fills, every bar of the symbol from the fill bar onward is checked: for a long, `low <= stop_loss`
  triggers the stop and `high >= take_profit` triggers the target, each decided at that bar's
  `ts` and filled at the open of the next bar, whatever that open is. If one bar triggers both, the
  stop wins (conservative). TP uses the same latency (builder's call: one latency rule for every
  exit).
- **Time stop.** The first `ClockTick` with `ts >= time_stop` triggers the exit; it fills at the
  open of the bar starting at that tick.
- **Nothing fills during a halt.** Between `Halt(halted=True)` and `Halt(halted=False)` for a symbol
  no fill happens; a working entry is cancelled; a pending exit fills at the open of the first bar
  with `bar_start >=` the resume `ts`.
- **Missing minutes.** A `Gap` minute has no bar, so the fill moves to the next bar that exists.
  Entries wait at most `entry_max_wait_min`, then expire; exits wait until the session ends.
- **Size cap.** Fill qty = `min(remaining qty, floor(quote_size_frac × ask_size))` for a buy (bid
  size for a sell) from the latest `Quote` within `quote_max_age_min`; with no fresh quote, the cap
  is `floor(bar_participation_cap × volume)` of the filling bar. An entry remainder is cancelled
  (the `Fill` is emitted before that `OrderUpdate(cancelled)`, same cascade); the filled part is a
  position with the entry's bracket. An exit remainder stays pending for the next bar.
- **Cost.** `cost_bps` = one leg of the refit cost model: half-spread + impact + depth shortfall
  (`research/penny/etl/quotes.py::leg_cost_bps`). Half-spread and inside notional come from the
  fresh quote if one exists, else from the cost inputs for the symbol-day (below).
- **End of session.** On the first tick at or after `eod_et`, any position whose exit has not
  filled and whose symbol is not halted fills at the last bar's close seen for it (`Fill.ts` = the
  tick's `ts`); a halted one gets `OrderUpdate(expired, "exit unfilled: halted at session end")` and
  stays open in the record, marked at its last price by evaluation.
- **Sell intents.** A `sell` intent closes up to the current position at the next bar's open; it
  is rejected if it exceeds the position (no shorts).

## State

- Orders by `intent_id`: the intent, status, filled qty, the entry fill price, bracket levels,
  pending exit (decision ts) if any.
- Per symbol: latest `Quote`, last close, halted flag; global `killed` flag.
- Seen `intent_id`s for idempotency.
- Size: O(intents per session), bounded by the RiskAgent's limits (tens per day).

## Configuration

| Parameter | Default | Source |
|---|---|---|
| fill latency | next bar's open after the decision | spec §4 |
| stop fill price | next bar's open, gaps included | spec §4 |
| fills during a halt | none | spec §4 |
| `quote_size_frac` | `1.0` | spec §4 ("size capped by the shares at the best quote") |
| `quote_max_age_min` | `5` | builder's call (must equal the RiskAgent's) |
| `bar_participation_cap` | `0.02` | builder's call (must equal the RiskAgent's) |
| `entry_max_wait_min` | `2` | builder's call |
| `eod_et` | `16:01` | builder's call (one tick after the last bar, 15:59, is released) |
| both TP and SL in one bar | stop wins | builder's call |
| `impact_coefficient`, `levels_penalty` | `1.0`, `1.0` | spec §4 ("costs from the refit cost model"); the refit may change them |
| `default_half_spread_bps` (no quote and no cost inputs) | `100.0` | builder's call; placeholder until the refit |
| broker | SimBroker now; AlpacaPaperBroker after Gate I with `paper=True` asserted | spec §2 |

Cost inputs are not events: an injected per-symbol-day adapter supplies `sigma_daily` (trailing 20
sessions before the session date), `adv_shares` (same window), and the 09:34 quote summary's
`half_spread_bps`, `ask_notional`, `bid_notional` where cached. Participation = `fill qty / adv_shares`
(`adv_shares` as defined in `risk.md`: the mean of the 20 prior sessions, all on one share basis via
`f_{D-1}`). The 09:34 summary covers quotes through 09:34 ET, so it is causal only for fills at or
after 09:35 ET; the builder asserts it (`Fill.ts >= 09:35 ET` whenever the summary is the spread or
notional source). An earlier fill with no fresh quote uses `default_half_spread_bps` and the bar
cap.
Every level in it is **unadjusted** (shares and dollars as they traded; penny §72): `adv_shares` is
converted from the adjusted daily cache with the per-day factor (`feed.md`), while `sigma_daily` is a
return statistic and may come from the adjusted series. Bar prices and quote sizes already arrive on
the traded basis.

**Volume correction (data plan D1, verified on the cache):** Alpaca `adjustment="all"` adjusts share volume for splits only, not dividends (BRSP 2024-03-04: `f` = 1.345, adjusted volume == raw). Share volume is therefore converted with the split-only factor `s = volume / volume_unadj` from `data/research_breakout/daily`, used only where that file's price factor agrees with the penny daily `f` within 2%; otherwise the symbol-day is rejected as `no_volume_factor`. Wherever this doc says "volume ÷ f", read "volume ÷ s". Daily ADV in dollars is `close_unadj × volume_unadj`.

Reuse (under `research/penny/`): `etl/quotes.py` `leg_cost_bps`, `impact_bps`,
`depth_shortfall_bps`, `CostModel` (scalar wrappers are fine; a parity test pins them to the frame
functions), `summarise_quotes` for the summary inputs. From main and this branch:
`tests/test_execution_agent.py::FakeTradingClient` for the live adapter's tests,
`utils/state_store.py` for live order-state persistence.

## Invariants

- No fill on the bar that caused the decision; every fill is on a bar with `bar_start >=` the
  decision ts, or at the end-of-session tick.
- An SL exit's price is a bar open (or the end-of-session close), never the stop level unless the
  open happens to equal it.
- No fill in a symbol while it is halted.
- Total filled qty per `intent_id` never exceeds the intent's `qty`; an exit never exceeds the
  position.
- Each `intent_id` is processed once; a duplicate gets `rejected` and no fill.
- After `Kill`, no new entry fill.
- A partial entry fill's `Fill` precedes the remainder's `OrderUpdate(cancelled)` in the cascade.
- The 09:34 quote summary is used only for fills at or after 09:35 ET (asserted).
- Only the live adapter module (`research/intraday/exec/alpaca_paper.py`, the guard's allowlist)
  imports `alpaca.trading`.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| Duplicate `intent_id` | seen-id set | skip the order, emit `OrderUpdate(rejected)` |
| Intent after `Kill` | killed flag | emit `OrderUpdate(rejected)` |
| Entry gets no bar within `entry_max_wait_min` | tick check | emit `OrderUpdate(expired)` |
| Fill size capped to zero | cap computation | entry: `OrderUpdate(cancelled)`; exit: retry on the next bar |
| A bar for a symbol arrives out of order | `bar_start` not after the last seen | raise |
| Live: broker position differs from the agent's books | reconciliation after each fill and every `reconcile_every_min` | see Open questions (needs a contract decision) |
| Live: AlpacaPaperBroker built with `paper` not `True` | assertion at construction | raise |

## Replay vs live

The agent's `on_event` is the same; the broker adapter differs. SimBroker applies the rules above
to bus `Bar`s. AlpacaPaperBroker (Phase 5) submits the bracket to Alpaca paper with `paper=True`
asserted, uses `client_order_id = intent_id` for idempotency, and turns broker callbacks into
`Fill`/`OrderUpdate` released on the next `ClockTick`. Live fills are the broker's, so the
nightly replay of the same day measures the gap between SimBroker and paper.

## Test plan

`tests/test_intraday_exec.py`:
- `test_new_intent_accepted_with_intent_ts`
- `test_entry_fills_at_next_bar_open_not_causing_bar`
- `test_stop_fills_at_next_bar_open_including_gap_down`
- `test_stop_never_fills_at_stop_level`
- `test_take_profit_fills_at_next_bar_open`
- `test_stop_wins_when_bar_hits_both`
- `test_time_stop_exit_at_bar_starting_at_tick`
- `test_no_fill_while_halted`
- `test_entry_cancelled_on_halt`
- `test_exit_fills_after_resume_at_first_bar_open`
- `test_gap_minute_moves_fill_to_next_bar`
- `test_entry_expires_after_max_wait`
- `test_fill_capped_by_shares_at_best_quote`
- `test_partial_entry_remainder_cancelled`
- `test_partial_fill_precedes_remainder_cancel`
- `test_0934_summary_not_used_before_0935`
- `test_cost_bps_matches_leg_cost_bps`
- `test_duplicate_intent_rejected_no_second_fill`
- `test_kill_cancels_entries_and_exits_positions`
- `test_intent_after_kill_rejected`
- `test_end_of_session_closes_open_positions`
- `test_oversized_sell_rejected`

## Open questions

- **[research]** Quotes in replay are cached only as per-day summaries and windows
  (`quote_summary*.parquet`, `quotes_rev/`), not a full NBBO stream, so most fills use the bar-volume
  size cap and the symbol-day's 09:34 summary for the spread. That is the replay fallback; R16 must
  state it, and report how often each source was used.
- **[research]** The refit cost model (spec §1: the method transfers, the 66.6 bps number does not):
  `impact_coefficient`, `levels_penalty` and `default_half_spread_bps` are placeholders until it is
  refit on this round's population.
- **[blocks-build]** Live reconciliation has no event: a broker-position mismatch cannot reach the
  MonitorAgent through the current contracts. Blocks the AlpacaPaperBroker build (Phase 5) only,
  not the Phase 2 SimBroker build. Needs an ADR and a contract change by `intraday-architect`
  (e.g. a reconciliation event the MonitorAgent turns into `Kill`).

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/8
