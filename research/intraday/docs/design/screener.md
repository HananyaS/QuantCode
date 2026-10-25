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
| `exclude_split_days` | `True` | builder's call (a raw-price gap on a split day is fabricated) |
| `data_root` | `C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_penny` | builder's call (main checkout, gitignored) |

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
- `etl/corporate_actions.py`: `flag_reverse_splits`, `flag_forward_splits` to drop split days.
- `etl/bars.py`: `load_years` for daily bars carrying `close_unadj` (raw) beside the adjusted series.
- `etl/alpaca_fetch.py`: `make_data_client`, `fetch_with_rejects` for any new fetch.

**The adjustment rule (spec §1, penny §57–§99).** Every price, share-volume or dollar-volume
**level** used in a filter or a feature is on the traded (unadjusted) basis. The caches under
`data_root` were fetched with `adjustment="all"` (`research/penny/etl/alpaca_fetch.py:127,184`):
the minute cache and the pre-market cache (`<data_root>/premarket/`) are split- and
dividend-adjusted, so `pm_dollar_volume` straight from them is exactly the withdrawn
`M_mean_trade_usd` leak. Decision: the screener's input adapter converts pre-market bars with the
same per-`(symbol, date)` factor and consistency check as ReplayFeed (`feed.md`: prices × `f`,
share volume ÷ `f`, `f = close_unadj / close` of the session date's daily row; a failing
symbol-day is dropped from the screen and recorded). The factor is a scale only; the session
date's daily price levels never enter a feature. Daily filters use `close_unadj` (and the other
`*_unadj` columns); daily share volume is converted the same way before any dollar volume is formed.
Cross-day ratios (gap versus prior close) are raw-to-raw with split days excluded; a raw level is
never paired with an adjusted one.

## Invariants

- At most one `Candidates` per session; it is emitted on a `ClockTick` and nothing else.
- No input row has a timestamp at or after the firing tick's `ts`: daily rows are for dates
  before the session date, pre-market bars end at or before the tick.
- Every price or dollar-volume level in a filter is unadjusted.
- The same inputs and configuration give the same `Candidates` (no randomness, no wall clock).
- `screen_id` changes when any configuration value changes.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| The adapter returns a bar or daily row at or after the cutoff | the agent's own timestamp assertion | raise (a leak; the run is void) |
| No daily history for the session (missing cache year) | adapter returns no rows for the prior date | raise (a data error, not an empty day) |
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
- `test_price_filter_uses_unadjusted_prior_close`
- `test_split_day_symbol_excluded`
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
- **[research]** The pre-market cache was requested only for earlier rounds' candidates
  (`scripts/fetch_premarket.py` records requested symbols per day), so it may not cover R16's whole
  low-priced universe. The no-selection arm needs the universe filter only (daily data), but the
  screen arm ranks on pre-market features. Meanwhile: build and test on fixture caches; R16 states
  the coverage, and a top-up fetch (raw, through `fetch_with_rejects`) fills the gap.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/3
