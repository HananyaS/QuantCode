# FeedAgent design

Owner: `intraday-data`. Package: `research/intraday/data/`. Spec §2 (agent table, design rules),
§5 phases 2 and 5. Read with `research/intraday/contracts/` and ADR-001.

## Responsibility

Turn market data for the session's candidates into one `Bar` or one `Gap` per symbol per minute, plus `Quote`s where quote data exists, released on the clock.

## Consumes

- `Candidates`: fixes the symbol set for the session. Exactly one is expected.
- `ClockTick`: the release trigger. On a tick at `ts = T` inside the session, the agent releases
  everything for the minute that has just closed, `bar_start = T - 1 min`.

Market data is not an event on the way in. It comes from an injected bar-and-quote source adapter:
ReplayFeed (cached SIP minute bars) now, AlpacaStreamFeed (a websocket buffer) after Gate I.

## Emits

On a tick at `T` with `T - 1 min` inside `[session_open_et, session_close_et)`, for each candidate
symbol in sorted order:
- `Bar` with `ts = T`, `bar_start = T - 1 min` (the contract stamps a bar at its end), unadjusted
  `open/high/low/close/vwap`, `volume`, `trade_count`, and `feed` = the source's feed
  (`"sip"` in replay).
- or, when the source has no bar for that symbol and minute, `Gap` with `ts = T`,
  `symbol`, `missing_bar_start = T - 1 min`. **A missing minute is never filled in**: no
  forward-filled bar, no zero-volume bar.
- then, if the source holds a quote for that symbol stamped in `(T - 1 min, T]`, one `Quote`
  with `ts = T` (the latest such quote; sizes in **shares**).

Ticks outside the session emit nothing. Ticks before `Candidates` arrives emit nothing.

## State

- `symbols: tuple[str, ...]` from `Candidates`.
- The source adapter's cursor. ReplayFeed loads one session's bars for the candidates into a
  dict keyed by `(symbol, bar_start)` when `Candidates` arrives: at most
  `len(symbols) × 390` bars (about 7,800 for `top_n = 20`).
- For live, the stream buffer holds at most two minutes per symbol (the open minute and the one
  being released).

## Configuration

| Parameter | Default | Source |
|---|---|---|
| `session_open_et` | `09:30` (first `bar_start`) | spec (regular session; the model reads bars "after the open", §4) |
| `session_close_et` | `16:00` (last `bar_start` is 15:59) | builder's call |
| replay `feed` | `"sip"` | spec §2 (`ReplayFeed (cached SIP)`) |
| live `feed` | `"iex"` | spec §1 (free real-time is IEX only); the paid SIP choice is Phase 4's decision |
| emitted bar basis | **unadjusted** (prices and share volume as they traded) | spec Constraints ("unadjusted prices in every filter"); decided here (below) |
| `data_root` | `C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_penny` | builder's call (the data lives in the main checkout, gitignored, not in the worktree) |
| minute cache | `<data_root>/minute/dt=YYYY-MM-DD/minute.parquet` (2016-02-17 .. 2026-08-07, 2,628 days, 09:25–16:05 ET; columns `symbol, date, minute_et` ('HH:MM' ET), `open, high, low, close, volume` (float), `trade_count` (float), `vwap`) | existing cache |
| daily cache | `<data_root>/daily/bars_<year>.parquet` (adjusted OHLCV plus `open_unadj, high_unadj, low_unadj, close_unadj`) | existing cache (`research/penny/etl/bars.py`) |
| `unadjust_tolerance` | `0.02` | builder's call |
| `raw_minute_dir` (top-up cache) | `<data_root>/minute_raw/dt=YYYY-MM-DD/minute.parquet` | builder's call |
| `holdout_start` | `2025-08-07` | spec §4 (last 12 months sealed) / R16; provisional value per controller Ruling 10 until `PREREGISTRATION_R16.md` fixes it |
| `unseal` | `False` | spec §4 (the sealed holdout is read only by an explicit, recorded decision) |
| quote size unit | shares | contract (`Quote.bid_size/ask_size` are shares) |

**The minute cache is adjusted, and ReplayFeed un-adjusts it (decision).** The cache was fetched
by `research/penny/etl/alpaca_fetch.py::fetch_minute_for_day`, which hard-codes
`adjustment="all"` (lines 127 and 184): split- and dividend-adjusted. ATOS on 2016-02-17 prints
about $2,025. A live feed is unadjusted, so emitting these bars as they are would skew every
price level, dollar volume and share volume between training and serving, and the levels encode
corporate actions that had not happened yet (penny §72). Within one day the factor is constant,
so intraday returns are unaffected; levels are not.

ReplayFeed therefore converts each `(symbol, date)` to the traded basis with the day's factor
`f = close_unadj / close` from the daily cache for the same date:
`open, high, low, close, vwap` × `f`; `volume` ÷ `f`, rounded to the nearest int; `trade_count`
unchanged (cast to int). The factor is only a scale; no daily price level enters a bar.
Consistency check per symbol-day: the converted first regular-session open must be within
`unadjust_tolerance` of `open_unadj`, and `f` from `close` must agree with `f` from `open`
(`open_unadj / open`) within the same tolerance (the two caches were fetched at different times;
a corporate action between the fetches would make their adjustments differ). A symbol-day that fails,
or has no daily row, emits no bars: every minute of it is a `Gap`, and the symbol-day is written to
the ETL's rejects file with the reason. The pre-market cache (`<data_root>/premarket/`) was written
by the same function and gets the same conversion. The fallback, if too many symbol-days fail, is a
fresh fetch with `adjustment="raw"` through `fetch_with_rejects`; it changes the cache, not the agent.

Reuse (all under `research/penny/`):
- `etl/alpaca_fetch.py`: `make_data_client` (never a trading client), `fetch_with_rejects` for
  batching and rejected symbols. Do not call `fetch_minute_for_day` for new data: it hard-codes
  `adjustment="all"`. A new fetch uses the same request shape with `adjustment="raw"`, `feed="sip"`.
- `etl/bars.py`: `load_years` for the daily raw/adjusted pair that gives the factor.
- `etl/quotes.py`: `summarise_quotes` and the cached `quote_summary*.parquet`, `quote_windows_*.parquet`,
  `quotes_rev/` under `data_root`. These are **per-day summaries and windows, not an NBBO
  stream**. `summarise_quotes` multiplies sizes by 100 ("sizes are in round lots"); convert so
  `Quote` carries shares, and verify the unit against current Alpaca docs before trusting it.

The converted replay cache layout, if the builder materialises one, is the builder's call; it lives
under `data_root`, never in the worktree.

## Invariants

- For every candidate and every session minute, exactly one of `Bar` or `Gap` is emitted, at
  `ts = bar_start + 1 min`.
- Bars are never synthesised. A `Bar` always corresponds to one source row.
- No event is emitted for a minute that has not closed (`Bar.ts == T`, never later than the tick).
- Prices and share volume are on the traded (unadjusted) basis; `feed` matches the source.
- Output order within a tick is sorted symbol, then `Bar`/`Gap`, then `Quote`.
- Two replays of the same session emit identical events.
- No session date `>= holdout_start` is replayed unless `unseal=True`.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| No source bar for a candidate minute | key lookup misses | emit `Gap` |
| A symbol-day fails the un-adjustment check, or has no daily row for the factor | check at load | emit `Gap` for each of its minutes (and record it in the rejects file) |
| A cached row violates the `Bar` contract (e.g. `low > min(open, close)`) | `Bar.__post_init__` raises | raise (the ETL must have rejected it; a bad cache voids the run) |
| Duplicate rows for one `(symbol, bar_start)` in the cache | checked at load | raise |
| A cached quote is crossed or zero-sized | `Quote.__post_init__` raises, or size check at load | skip that quote (summarise_quotes already drops non-markets; a stray one is not a reason to abort) |
| A second `Candidates` in one session | state check | raise |
| A candidate has **zero** rows in both minute caches for the session (a coverage reject, not illiquidity) | check when `Candidates` loads | emit `Gap` for each of its minutes, and record a coverage reject (`kind="coverage"`, symbol, date, `screen_id`) separately from per-minute `Gap`s. The run summary counts coverage rejects **per arm**, so the arms are compared on what was actually covered |
| Session date `>= holdout_start` and `unseal` is false | date check when the adapter opens a session | raise |
| Live: a bar arrives after its minute was already released as `Gap` | buffer sees a `bar_start` older than the last release | skip (drop it and count it); the log keeps the `Gap`, so live and its nightly replay agree |
| Live: the stream disconnects | no data for whole minutes | emit `Gap` per symbol per minute; the MonitorAgent's stale-feed rule decides on `Kill` |

## Replay vs live

The agent's `on_event` is the same. The adapter differs: ReplayFeed answers lookups from the SIP minute
cache converted to the traded basis; AlpacaStreamFeed (Phase 5, after Gate I) subscribes to the websocket for the candidate
symbols when `Candidates` arrives and buffers until the tick. Live `feed` is `"iex"` unless
Phase 4 decides otherwise. Only ExecutionAgent's live adapter may import `alpaca.trading`; the feed
uses `alpaca.data` only.

## Test plan

`tests/test_intraday_data_feed.py`:
- `test_tick_releases_bar_for_minute_just_closed`
- `test_bar_ts_is_bar_start_plus_one_minute`
- `test_missing_minute_emits_gap_not_filled_bar`
- `test_every_candidate_minute_gets_exactly_one_bar_or_gap`
- `test_nothing_emitted_before_candidates`
- `test_nothing_emitted_outside_session`
- `test_output_order_sorted_symbol_then_bar_then_quote`
- `test_quote_sizes_are_shares`
- `test_invalid_cached_bar_raises`
- `test_duplicate_cached_bar_raises`
- `test_second_candidates_raises`
- `test_replay_feed_is_deterministic`
- `test_unadjust_scales_prices_by_factor_and_volume_inversely`
- `test_unadjusted_first_open_matches_daily_open_unadj`
- `test_factor_mismatch_symbol_day_becomes_gaps`
- `test_missing_daily_row_symbol_day_becomes_gaps`
- `test_intraday_returns_unchanged_by_unadjust`
- `test_zero_row_candidate_is_coverage_reject_not_only_gaps`
- `test_coverage_rejects_counted_per_arm`
- `test_raw_cache_rows_skip_unadjust`
- `test_raw_cache_wins_over_adjusted_for_same_symbol_day`
- `test_holdout_date_refused_without_unseal`
- `test_raw_fetch_requests_unadjusted_sip` (any new ETL request carries `adjustment="raw"`, `feed="sip"`)

## Open questions

- **[research]** Live feed IEX versus SIP. IEX is about 2.5% of consolidated volume, so live bars
  differ from the SIP replay. Phase 4 replays the passing strategy on IEX versus SIP bars over the
  same days, after checking current Alpaca terms, and the feed is chosen from that number. Until
  then live defaults to `"iex"`.
- **[research]** Historical quotes in replay. NBBO is cached only as per-minute summaries at a few
  minutes (`research/penny/etl/quotes.py`: 09:34, 09:35, 15:55), not every minute. ReplayFeed emits a
  `Quote` only where a cached summary exists (as a synthetic quote: bid/ask from mid ± half-spread,
  sizes from the median inside sizes, stamped at the end of its window). The RiskAgent and the
  ExecutionAgent have a fallback for the minutes without one (see their docs). Whether R16 needs a
  per-minute quote fetch for the candidates is for the pre-registration to decide.

- **[research]** How many symbol-days fail the un-adjustment check. If the failure rate in the
  R16 development window is material (say above 2% of candidate symbol-days), refetch that window
  raw instead; the agent does not change. The rate is reported with the R16 run.

- **[research]** Minute-cache coverage. `<data_root>/minute/` is the union of earlier penny rounds'
  candidate pools (the per-day `requested.json` lists the requested symbols): 2,628 days, 207,594
  symbol-days, 3,507 unique symbols, symbols per day min 1 / median 54 / max 819, minutes 09:25–16:05
  ET. The pre-market cache (`<data_root>/premarket/`) is the same kind of union: 2,621 days, 452,243
  symbol-days, 3,555 symbols, median 13 a day, 04:00–09:34 ET. No other minute data exists on disk.
  Daily bars, by contrast, are complete for the universe (`<data_root>/daily/bars_YYYY.parquet`, and
  `data/research_breakout/daily/bars_YYYY.parquet`, about 6–7k symbols a year). Most of an R16
  screen's names, and nearly all of the no-selection arm's, have no minute bars there. Without
  action ReplayFeed would emit all-`Gap` symbol-days that look like illiquid names, and the arms
  would silently not be comparable. Remedy: a top-up fetch of the missing symbol-days through
  `fetch_with_rejects` with `adjustment="raw"`, `feed="sip"`, into `raw_minute_dir`. The directory
  tells the two sources apart, so each row's basis is known. Rows from `minute/` are adjusted and get
  the un-adjust step. Rows from `minute_raw/` are already on the traded basis and are emitted as they
  are (the un-adjust step is skipped, not run with `f = 1`). If both hold a symbol-day, `minute_raw/`
  wins. Until the top-up exists, the coverage rejects above are counted and reported per arm, and R16
  states the coverage it ran on.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/4
