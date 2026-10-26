# RiskAgent design

Owner: `intraday-risk`. Package: `research/intraday/risk/`. Spec §2 (agent table: sizing, position
limits, quote-size liquidity cap, TP/SL selection, daily loss limit), §4 (exit factor, fill rules).
Read with `research/intraday/contracts/` and ADR-001. Built against fixture `Signal`s
(`research/intraday/contracts/fixtures.py`); it needs no model.

## Responsibility

Decide whether a `Signal` becomes an order, and if so its size, its TP/SL/time-stop bracket and its id; it is the only agent that emits an `OrderIntent`.

## Consumes

- `Signal`: the entry trigger (rules below).
- `Bar`: the latest close per symbol (reference price, mark-to-market), the latest volume (the
  bar-volume liquidity fallback) and the recent true ranges (ATR exits). In the bus's
  breadth-first order a minute's `Bar` reaches this agent before the `Signal` derived from it.
- `Quote`: the latest inside sizes and its `ts` per symbol (the quote-size cap).
- `Fill`: positions, cash, realised P&L. A size-capped partial entry fill is a position: in the
  same cascade the ExecutionAgent emits the `Fill` first and then `OrderUpdate(cancelled)` for the
  remainder, so by the time the cancel arrives the filled qty is already booked.
- `OrderUpdate`: `rejected`, `cancelled` or `expired` frees the working slot of that `intent_id`.
  After a partial fill it drops only the working remainder: the filled partial position is kept
  (it still counts toward `max_open_positions` and the one-entry-per-symbol rule) and exits on its
  bracket.
- `Halt`: `halted=True` blocks new entries in the symbol until a `halted=False` for it.
- `Kill`: no further `OrderIntent` this session.

## Emits

- `OrderIntent`, at most one per `Signal`, with `ts` = the `Signal`'s `ts` and:
  - `intent_id` = `f"{session_date:%Y%m%d}-{symbol}-{n}"`, `n` a 1-based per-session counter over
    all intents, so ids are unique and deterministic;
  - `side = "buy"` (entries are long only), `qty` (sizing below), `limit_price` = None (market);
  - `take_profit`, `stop_loss` per `exit_mode` (both None for `"time_stop"`);
  - `time_stop = min(ts + hold_min, flatten_et on the session date)`;
  - `reason` = `f"entry score={score:.6g} thr={entry_threshold:.6g} cap={cap}"`, `cap` being
    `quote`, `bar` or `notional`, whichever bound the size.

Entry rules, all of which must hold, checked in this order (the first failure skips, silently
but counted):
1. not killed, and the daily loss limit has not tripped;
2. the symbol is not halted;
3. `Signal.ts` in ET is before `last_entry_et`;
4. `score >= entry_threshold`;
5. no open position and no working intent in the symbol, and fewer than
   `max_entries_per_symbol` entries in it this session;
6. open positions plus working intents < `max_open_positions`;
7. a latest `Bar` exists for the symbol (its `close` is `ref_price`);
8. sizing gives `qty >= 1`, and the bracket is valid (`0 < stop_loss < ref_price < take_profit`).

Sizing: `qty = floor(min(position_notional_usd, cash) / ref_price)`, then capped by liquidity:
- if a `Quote` for the symbol with `ts` within `quote_max_age_min` of the `Signal` exists:
  `qty <= floor(quote_size_frac × ask_size)` (shares at the best ask);
- otherwise (the replay fallback, see Open questions): `qty <= floor(bar_participation_cap × volume)`
  of the latest `Bar`.

Brackets, from `ref_price`:
- `"time_stop"`: no TP or SL;
- `"fixed_pct"`: `take_profit = ref_price × (1 + tp_pct)`, `stop_loss = ref_price × (1 − sl_pct)`;
- `"atr"`: ATR = mean true range of the symbol's last `atr_period` bars; `take_profit = ref_price +
  tp_atr × ATR`, `stop_loss = ref_price − sl_atr × ATR`; fewer than `atr_period` bars held skips.

Phase 2 emits buys only. Exits are the bracket legs the ExecutionAgent manages; a closing `sell`
intent is reserved for later use (e.g. flatten on the loss limit) and is not emitted now.

## State

- Per symbol: latest `Bar` (close, volume), a `deque(maxlen=atr_period)` of true ranges, latest
  `Quote` (ask size, ts), position qty and average price, entries this session, halted flag.
- Working intents: `intent_id -> (symbol, remaining qty)` until fully filled, rejected, cancelled
  or expired; a fill reduces the remaining qty and adds to the position.
- Account: `cash`, realised P&L, `start_equity_usd`, `killed`, `loss_limit_tripped`, the intent
  counter.
- Size: O(candidates × atr_period), about `20 × 14` true ranges plus a few scalars per symbol.

## Configuration

| Parameter | Default | Source |
|---|---|---|
| `start_equity_usd` | `25_000` | builder's call (the R15 sizing check covered 25k/50k/100k) |
| `position_notional_usd` | `2_000` | builder's call (the scale of the spec §1 cost measurement) |
| `max_open_positions` | `5` | builder's call |
| `max_entries_per_symbol` | `1` | builder's call |
| `entry_threshold` | `0.5` (tests only) | R16 (depends on the scorer; the constant control scores 1.0) |
| `exit_mode` | `"time_stop"` | R16 (exit ∈ {time stop only, TP+SL fixed in percent, TP+SL scaled by ATR}, spec §4) |
| `tp_pct`, `sl_pct` | none; set by R16 | R16 (research: the levels are pre-registered, not chosen here) |
| `tp_atr`, `sl_atr` | none; set by R16 | R16 (research) |
| `atr_period` | `14` bars | builder's call |
| `hold_min` | `Signal.horizon_min` | spec §4 (the model predicts the move over the next h minutes) |
| `quote_size_frac` | `1.0` | spec §4 ("size capped by the shares at the best quote") |
| `quote_max_age_min` | `5` | builder's call |
| `bar_participation_cap` | `0.02` of the latest bar's volume | builder's call (fallback when no fresh quote exists) |
| `daily_loss_limit_pct` | `0.02` of `start_equity_usd`, on marked equity | builder's call |
| `last_entry_et` | `15:30` | builder's call |
| `flatten_et` | `15:55` | builder's call (also caps every `time_stop`) |

Every level the agent uses is on the traded (unadjusted) basis: `ref_price`, the bar volume in the
fallback cap and the quote sizes arrive that way from the FeedAgent (`feed.md` converts the
adjusted cache). Dollar sizing, the price band and the shares-at-best-quote cap must never be fed
an adjusted level (penny §72).

`adv_shares` (wherever the agent or the cost inputs use it) is the mean of the 20 prior sessions'
daily share volume, every session converted to one share basis, the traded basis of the session
date `D`, with `f_{D-1}` (the factor of the last daily row before `D`, `screener.md`): each adjusted
volume is divided by `f_{D-1}`. Mixing per-day factors would put a split inside the window on two
bases. Only prior sessions enter it; `D`'s own volume never does.

**Volume correction (data plan D1, verified on the cache):** Alpaca `adjustment="all"` adjusts share volume for splits only, not dividends (BRSP 2024-03-04: `f` = 1.345, adjusted volume == raw). Share volume is therefore converted with the split-only factor `s = volume / volume_unadj` from `data/research_breakout/daily`, used only where that file's price factor agrees with the penny daily `f` within 2%; otherwise the symbol-day is rejected as `no_volume_factor`. Wherever this doc says "volume ÷ f", read "volume ÷ s". Daily ADV in dollars is `close_unadj × volume_unadj`.

Construction asserts `last_entry_et < flatten_et` (an entry window that ends after the flatten
time would give an intent whose `time_stop` is capped at or before its own `ts`).

Marked equity = `cash + Σ qty × latest close`. Cash moves on each `Fill`: a buy costs
`price × qty × (1 + cost_bps / 10_000)`, a sell returns `price × qty × (1 − cost_bps / 10_000)`.
The loss limit is checked after every `Bar` and `Fill`; once tripped it stays tripped for the
session and stops new entries only (open positions keep their brackets).

Reuse: `research/penny/etl/quotes.py::tradeable_mask` encodes the same "you can only trade what you
can quote" rule on summary frames; the agent applies it per order on events. `research/penny/eval/portfolio.py`
(`daily_returns`, `evaluate_portfolio`) evaluates the compounded account from the recorded fills.

## Invariants

- Only this agent emits `OrderIntent` (the design-doc test checks the docs; the integration replay
  test checks the log's `source` field).
- At most one `OrderIntent` per `Signal`; `intent_id`s are unique within the session.
- `qty × ref_price <= position_notional_usd` and `qty` never exceeds the liquidity cap in force.
- Open positions plus working intents never exceed `max_open_positions`.
- No intent after `Kill`, after the loss limit trips, for a halted symbol, or at or after `last_entry_et`.
- Every `time_stop` is at or before `flatten_et`.
- Position bookkeeping matches the sum of `Fill`s exactly, partial fills included.
- `last_entry_et < flatten_et` (asserted at construction).

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| Any entry rule fails | the rule checks above | skip (count by reason) |
| A `Fill` for an unknown `intent_id` | lookup in working intents and positions | raise (the books cannot be trusted; the MonitorAgent also kills on it in live) |
| A sell `Fill` larger than the position | position check | raise |
| An ATR or percent bracket gives `stop_loss <= 0` or crosses `ref_price` | bracket check | skip |
| Marked equity falls through the loss limit | check after `Bar`/`Fill` | stop new entries for the session (not a `Kill`: only the MonitorAgent emits `Kill`) |

## Replay vs live

None in the agent. The only practical difference is what arrives: in replay `Quote`s are rare
(cached summaries only), so most entries are capped by the bar-volume fallback; live sees a quote
stream and the quote cap binds.

## Test plan

`tests/test_intraday_risk.py`:
- `test_only_buy_intents_with_signal_ts`
- `test_intent_ids_unique_and_deterministic`
- `test_below_threshold_signal_skipped`
- `test_size_capped_by_notional`
- `test_size_capped_by_shares_at_best_ask`
- `test_stale_quote_falls_back_to_bar_volume_cap`
- `test_zero_qty_after_cap_skipped`
- `test_fixed_pct_bracket_levels`
- `test_atr_bracket_needs_atr_period_bars`
- `test_time_stop_capped_at_flatten_time`
- `test_one_entry_per_symbol_per_session`
- `test_max_open_positions_respected`
- `test_rejected_or_expired_update_frees_slot`
- `test_halted_symbol_blocked_until_resume`
- `test_no_intent_after_kill`
- `test_loss_limit_stops_new_entries`
- `test_no_entry_at_or_after_last_entry_time`
- `test_unknown_fill_raises`
- `test_books_match_fills`
- `test_partial_fill_then_cancel_keeps_filled_position`
- `test_adv_shares_uses_prior_sessions_on_one_basis`
- `test_last_entry_must_precede_flatten`

## Open questions

- **[research]** The TP/SL levels and the exit grid are R16 factors (time stop only, TP+SL fixed in
  percent, TP+SL scaled by ATR). The agent implements all three modes; R16 pre-registers
  `tp_pct`, `sl_pct`, `tp_atr`, `sl_atr` and runs every cell, recording the trial count.
- **[research]** Quotes in replay. Historical NBBO is cached only in summary form
  (`research/penny/etl/quotes.py`, a few minutes per day), so in replay most entries fall back to
  `bar_participation_cap × bar volume`. The fallback is a proxy for the spec's quote-size cap,
  not the cap itself; R16 must state it, and report how many entries each cap bound. If the
  critic rules the proxy insufficient, the fix is a per-minute quote cache in the feed's ETL,
  with no change to this agent.
- **[research]** Whether the loss limit should also flatten open positions (it only stops entries
  now). Deciding it needs a sell `OrderIntent` path; the ExecutionAgent already accepts one.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/7
