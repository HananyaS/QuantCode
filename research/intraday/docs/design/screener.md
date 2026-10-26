# ScreenerAgent design

Owner: `intraday-data`. Package: `research/intraday/data/`. Spec §2 (agent table), §4 (main arm
and controls). Read with `research/intraday/contracts/` and ADR-001.

## Responsibility

Once per session, at the screen time, turn daily and pre-market data known before that time into one `Candidates` list.

## Consumes

- `ClockTick`: the only trigger. The first tick whose time in America/New_York is at or after
  `screen_time_et` fires the screen; every other tick is ignored.

The daily and pre-market inputs are not events. They come through an injected read-only adapter
(the ScreenInputs protocol, defined by the builder in `research/intraday/data/`), which is asked
for data strictly before the firing tick's `ts`.

## Emits

- `Candidates`: exactly one per session, with `ts` = the firing tick's `ts`,
  `symbols` = the selected symbols sorted and de-duplicated (the contract requires sorted, unique),
  and `screen_id` = `f"{arm}:{config_hash[:12]}:{session_date:%Y-%m-%d}"`, where `config_hash` is
  the sha256 of the canonical JSON of this agent's configuration. An empty `symbols` tuple is
  valid and means the day trades nothing.

The rank order is lost when the symbols are sorted. Nothing downstream may rely on it; the
cut to `top_n` happens before sorting.

## State

- `fired: bool`, set after the one emission.
- `session_date` (ET date of the first tick seen).
- Nothing per symbol is kept after the emission. Inputs are read once, inside the firing
  `on_event`, and dropped. Size: O(universe) for the duration of one call, O(1) afterwards.

## Configuration

Times are ET wall-clock times, converted with `zoneinfo.ZoneInfo("America/New_York")` against the
tick's date (the `tzdata` package supplies the database on Windows; it is installed).

| Parameter | Default | Source |
|---|---|---|
| `screen_time_et` | `09:25` | spec (Context: "the screen runs at 09:00–09:25 ET") |
| `premarket_start_et` | `04:00` | builder's call (Alpaca's extended-hours start, as in `research/penny/scripts/fetch_premarket.py`) |
| `arm` | `"screen"`; also `"no_selection"` | spec §4 (main arm and the no-selection control) |
| universe filter: `min_price`, `max_price` on the **unadjusted** prior close | `1.00`, `20.00` | R16 ("liquid low-priced names"); these defaults are placeholders until the pre-registration is committed |
| universe filter: `min_adv_usd` (20-day median of `close_unadj` × unadjusted share volume, days before the session) | `1_000_000` | R16; placeholder default |
| ranking feature(s) for `arm="screen"` | `pm_gap` then `pm_dollar_volume`, descending, ties broken by symbol | R16 ("ranked by pre-market features"); placeholder default |
| `top_n` | `20` | R16; placeholder default |
| `exclude_split_days` | `True` | builder's call (a raw-price gap on a split day is fabricated); detected causally, see the adjustment rule |
| `split_log_step_threshold` (flag a split when `abs(log(f_{D-1} / f_D))` exceeds it) | `log(1.25)` | builder's call (an ex-dividend step is a few percent, a split is at least 5:4, so dividends are not flagged) |
| `holdout_start` | `2025-08-07` | spec §4 (last 12 months sealed) / R16; provisional value per controller Ruling 10 until `PREREGISTRATION_R16.md` fixes it |
| `unseal` | `False` | spec §4 (the sealed holdout is read only by an explicit, recorded decision) |
| `data_root` | `C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_penny` | builder's call (main checkout, gitignored) |
| `restrict_to_minute_cache` | `True` | user decision 2026-10-26 (covered stocks only, every arm) |

`arm="no_selection"` returns every symbol that passes the universe filter, with no ranking and
no `top_n` cut. The other two controls (random score, screen with no model) live in the
PredictorAgent's `scorer` setting, not here.

Reuse (all on this branch under `research/penny/`):
- `features/premarket.py`: `premarket_features`, `add_relative_features` for the feature
  formulas. Feed them pre-market bars converted to the traded basis (the adjustment rule below),
  never the cache as stored.
- `features/premarket.py`: `assert_no_post_decision_bars` checks a hard-coded 09:35 cutoff, which
  is later than this screen's. Keep it as a second guard, and add the agent's own assertion at
  `screen_time_et` (every bar's `bar_start + 1 min <= firing tick ts`).
- `etl/corporate_actions.py`: **not** at screen time. `flag_reverse_splits` and `flag_forward_splits`
  read the session's open and its full-session range (`corporate_actions.py:82,98,146`), which are
  unknowable at 09:25. Use them only for after-the-fact audits.
- `etl/bars.py:226-238`: the causal split-event factor (the step of the adjustment ratio between
  consecutive days), the precedent for the split-day flag below.
- `etl/bars.py`: `load_years` for daily bars carrying `close_unadj` (raw) beside the adjusted series.
- `etl/alpaca_fetch.py`: no new fetch (user decision 2026-10-26: no downloads). Replay reads the
  existing caches only; the live pre-market fetch is a Phase-5 adapter question.

**The adjustment rule (spec §1, penny §57–§99).** Every price, share-volume or dollar-volume
**level** used in a filter or a feature is on the traded (unadjusted) basis. The caches under
`data_root` were fetched with `adjustment="all"` (`research/penny/etl/alpaca_fetch.py:127,184`):
the minute cache and the pre-market cache (`<data_root>/premarket/`) are split- and
dividend-adjusted, so `pm_dollar_volume` straight from them is exactly the withdrawn
`M_mean_trade_usd` leak. Decision, causal at 09:25 on session date `D`:
- **Conversion factor.** Pre-market bars are converted with `f_{D-1} = close_unadj / close` of the
  last daily row **before** `D` (prices × `f_{D-1}`, share volume ÷ `f_{D-1}`). That is exact on every
  day without a corporate action at `D`'s open; split days are excluded below, so it is exact for
  every symbol the screen keeps. The screener never reads the session date's daily row for
  conversion or consistency. That check belongs to the FeedAgent's adapter (`feed.md`), where the
  `D` row is used only as a scale factor.
- **Split-day flag.** A split at `D`'s open is detected from the factor step `f_{D-1} / f_D`
  (the `bars.py:226-238` event factor; ex-dates are announced ahead, so the step is knowable before
  the open), or from a corporate-actions calendar if one is added. Only the ratio of the two factors
  is used, never a session price. Only a step with `abs(log(f_{D-1} / f_D)) > split_log_step_threshold`
  (default `log(1.25)`) is a split: the adjusted series also steps on every ex-dividend date, and
  those small steps must not exclude the symbol. If the `D` row is missing, no split is known and the symbol is
  **kept**. Never `flag_reverse_splits`/`flag_forward_splits` (session open and range).
- **Failing symbol-days stay in `Candidates`.** A symbol whose session-`D` data later turns out
  missing or inconsistent (halted all day, delisted that day, a factor mismatch) is not dropped by
  the screen, since dropping it would select on the outcome. The FeedAgent turns its minutes into
  `Gap`s and records the reject.
- Daily filters use the `*_unadj` columns of rows before `D`; daily share volume is converted with
  that row's own factor before any dollar volume is formed. Cross-day ratios (gap versus prior
  close) are raw-to-raw (`pm_close × f_{D-1} / close_unadj_{D-1}`); a raw level is never paired with an
  adjusted one.
- **Cache coverage (user decision, 2026-10-26: covered stocks only, no download).** The minute
  cache (median 54 symbols a day) and the pre-market cache (median 13 a day) hold only the union of
  earlier rounds' candidate pools, while the daily caches are complete for the universe. Order of
  operations, for every arm including `no_selection`: (1) the causal universe filter on daily rows
  before `D`; (2) intersect with the symbols the minute cache covers on `D`
  (`restrict_to_minute_cache`); (3) for `arm="screen"`, rank on pre-market features, skipping symbols
  without pre-market bars. Cache membership is never the universe itself, because it may encode the
  outcome (`feed.md`, Open questions). The count removed at step 2 is recorded per arm.
- **Sealed holdout.** The adapter refuses any session date `>= holdout_start` (raise) unless
  `unseal=True` is passed. `unseal` is part of `config_hash`, so it shows in `screen_id`.

**Volume correction (data plan D1, verified on the cache):** Alpaca `adjustment="all"` adjusts share volume for splits only, not dividends (BRSP 2024-03-04: `f` = 1.345, adjusted volume == raw). Share volume is therefore converted with the split-only factor `s = volume / volume_unadj` from `data/research_breakout/daily`, used only where that file's price factor agrees with the penny daily `f` within 2%; otherwise the symbol-day is rejected as `no_volume_factor`. Wherever this doc says "volume ÷ f", read "volume ÷ s". Daily ADV in dollars is `close_unadj × volume_unadj`.

## Invariants

- At most one `Candidates` per session; it is emitted on a `ClockTick` and nothing else.
- No input row has a timestamp at or after the firing tick's `ts`. Daily rows read by the screener
  are for dates strictly before the session date `D`. The one exception is the split-day flag, which
  reads only the factor ratio `f_{D-1} / f_D`, a stand-in for the announced ex-date calendar and never
  a price. Pre-market bars end at or before the tick.
- The screen never drops a symbol because of session-`D` data being missing or inconsistent.
- No session date `>= holdout_start` is screened unless `unseal=True`.
- Every price or dollar-volume level in a filter is unadjusted.
- The same inputs and configuration give the same `Candidates` (no randomness, no wall clock).
- `screen_id` changes when any configuration value changes.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| The adapter returns a bar or daily row at or after the cutoff | the agent's own timestamp assertion | raise (a leak; the run is void) |
| No daily history for the session (missing cache year) | adapter returns no rows for the prior date | raise (a data error, not an empty day) |
| Session date `>= holdout_start` and `unseal` is false | date check in the adapter | raise |
| A symbol lacks pre-market bars | absent from the pre-market panel | skip the symbol (it has no pre-market activity, so it cannot rank) |
| The first tick of the session is after `screen_time_et` + 5 min | tick time check | raise (the run was started late; a late screen sees more data than the design allows) |
| A second session date appears in the ticks | date of tick differs from `session_date` | raise (one run is one session) |

## Replay vs live

The agent code is identical. Only the ScreenInputs adapter differs: replay reads the caches under
`data_root` for the session date, converted to the traded basis; live reads the same caches for the prior days
plus a pre-market fetch completed before 09:25 ET. The live adapter must still return only rows
before the firing tick, and the agent's assertion checks it either way.

## Test plan

`tests/test_intraday_data_screener.py`:
- `test_emits_one_candidates_at_first_tick_at_or_after_screen_time`
- `test_ignores_ticks_before_screen_time_and_after_firing`
- `test_candidates_ts_equals_firing_tick_ts`
- `test_symbols_sorted_unique_and_cut_to_top_n_before_sorting`
- `test_no_selection_arm_returns_whole_filtered_universe`
- `test_universe_is_causal_filter_intersected_with_cache`
- `test_cache_symbol_outside_causal_filter_never_selected`
- `test_price_filter_uses_unadjusted_prior_close`
- `test_split_day_symbol_excluded`
- `test_split_flag_reads_factor_step_not_session_prices`
- `test_ex_dividend_factor_step_is_not_a_split`
- `test_missing_session_daily_row_keeps_symbol`
- `test_conversion_uses_prior_day_factor`
- `test_holdout_date_refused_without_unseal`
- `test_premarket_bars_converted_to_traded_basis`
- `test_dollar_volume_uses_unadjusted_price_and_shares`
- `test_input_row_at_cutoff_raises`
- `test_late_first_tick_raises`
- `test_screen_id_changes_with_config`
- `test_same_inputs_same_candidates`
- `test_empty_universe_emits_empty_candidates`

## Open questions

- **[research]** The screen itself (universe bounds, ranking features, `top_n`) is R16's to
  pre-register. The builder implements the placeholder defaults above behind configuration and
  changes no code when R16 sets the values.
- **[research]** (for R16) Cache membership may be outcome-selected (see `feed.md`). The universe is
  the causal filter intersected with the cache, never the cache's symbol list. The pre-market cache is
  thinner than the minute cache (median 13 against 54 a day), so the screen arm ranks fewer names than
  the no-selection arm can trade. R16 states both coverages; there is no top-up fetch.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/3
