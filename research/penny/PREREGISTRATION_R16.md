# Round 16 Pre-Registration: the user's intraday design, tested against its own controls

**Status: DRAFT for critic G1. Nothing has been run.** This document is committed before any
round-16 result exists, and its commit hash is the timestamp. Constants may not be edited in
place once G1 signs off. A change after sign-off is a dated **Amendment** appended at the end,
and an amendment made after any R16 number has been seen counts as a trial (§6.3).

Binding inputs: `docs/superpowers/specs/2026-10-26-intraday-system-design.md` (§1 priors, §4
round 16), and the design docs `research/intraday/docs/design/{screener,feed,state,predictor,risk,execution,monitor}.md`.
Every key those docs mark `R16` or `[research]` is fixed by name in §5.

---

## 1. Purpose

The question is whether **the user's design** makes money net of realistic costs in cached
history. The design is a pre-open screen for intraday-rise potential at 09:25 ET, ranked on
pre-market features. A light model then reads the last k one-minute bars after the open and
predicts the move over the next h minutes, and the RiskAgent enters with a TP/SL bracket.
"Makes money" means the compounded account, not the per-trade mean (LOG §49). The design has
to beat three controls that remove one part each: the screen, the model's information, and the
model itself.

This is the first test in the project of a model that reads **live bars after the open** and
predicts **minutes ahead** (spec §1, row 4). Earlier rounds are priors for it, not verdicts
(spec, user decision 2026-10-25/26). §2 states each prior, what it was measured on, and how
this round could overturn it.

The round has one decision, Gate I (§11). If it passes, the live parts get built (spec §5
phases 4–5). If it fails, the intraday track stops and nothing is re-tuned.

---

## 2. Priors, with provenance

Each row gives the measured number, what it was measured on, why it may not transfer, and the
prediction it implies for R16. A prediction that fails overturns the prior **for this
population and this horizon only**.

| # | Prior | Measured on | Why it may not transfer | R16 prediction (overturned if…) |
|---|---|---|---|---|
| P1 | **Gap-ups lose open→close.** Mean open→close falls monotonically with the opening gap: +3..+10% → −0.43%, +10..+25% → −1.07%, >+25% → −3.68% (hit rate 31.6%) | LOG §38 / R5 §1: 1,455,390 daily symbol-days, ADV ≥ $1M, prior unadjusted close $0.50–$10, 2016–2026. **Daily bars, hold open→close, no model, no exit rule** | R16 enters after k minutes, holds h minutes, and lets a model choose the minute. An open→close mean says nothing about which 5–30-minute windows inside the day are positive | The screen-only control (C_c) has **negative** net compounded return, below the no-selection baseline (C_0). *Overturned if* C_c − C_0 has a day-clustered CI above zero |
| P2 | **Exit policy is not the lever.** Nine policies, gross −0.09% to −0.33% | LOG §16: round-2 gap-up population, **no predictive signal**, hold-to-close horizon | §1 of the spec: with no edge, no exit rule can create one (optional stopping on a martingale). With a model, exits are a real lever | Within the main arm, the fixed-% and ATR bracket arms do not beat the time-stop arm. *Overturned if* a bracket arm beats its time-stop sibling, same (k, h, model), with a paired CI above zero |
| P3 | **Every stop hurts.** Monotone worse from −50% to −15% stops | LOG §70: the **retracted** 10-session daily long (§72, an adjusted-price look-ahead sample). Daily closes, stops −15..−50% | Different horizon (minutes, not sessions), different population, a poisoned sample. Only the mechanism transfers: a stop ejects positions that later recover, and gap-through fills make it worse | Same as P2. A stop at about 1σ of the h-minute move is predicted to cut the hit rate more than it cuts the loss tail |
| P4 | **No minute-bar model cleared costs.** R3: best gross +1.22% (exploratory transformer blend), pre-registered best +0.71%, net −0.90% at $2k, cost ≈ 162 bps. R15 H3: minute features chose names held 60 days and inverted on the holdout (dev +0.53..+0.65 Sharpe, holdout −0.89..−1.46) | LOG §§26–27: the **pre-open path only**, entry 09:35, held to close. LOG §§95–97: **features of the signal session** used to pick 60-session holds | Neither tested a model reading bars **after** the open and predicting **minutes ahead** | Model cells show **gross** skill over the random control (H3) but **no** cell clears cost (H1 fails). *Overturned if* H1 passes |
| P5 | **Short-term reversal is real but marginal and sub-cost.** Reversal pool 09:35→15:55 +0.31%, CI [−0.03%, +0.63%] | R5 Amendment 2: 51,361 symbol-days, gap ∈ [−10%, −3%], hold to 15:55, no model | R16 conditions on minute paths and may concentrate the reversal into minutes | Covered by H3 and H1 |
| P6 | **Costs.** Round trip 66.6 bps at $2k (ADV ≥ $5M). Reversal pool (ADV ≥ $1M): spread-only round trip 59.7 bps, all-in ≈ 136.9 bps at $2k. Entry half-spread median 40.2 bps (p90 168), entry ask inside book median $2,120 (p25 $880) | LOG §33 (round 4), §§43–44 (round 5), measured at 09:34/09:35 and 15:55 only | The method transfers, the number does not (spec §1). Intraday entries at other minutes see different spreads | Refit on this round's population (§9.3). Predicted median round trip at $2k: **80–150 bps** |
| P7 | **Volatility drag.** Positive mean daily return compounded to −15% (σ ≈ 4.2%/day against an 8 bps edge) | LOG §49: round-6 cascade arms, 1–2 positions per day | It applies to any thin book; R16's book is at most 5 concurrent positions | Reported: mean against geometric mean daily return for every cell |
| P8 | **A random control moves Sharpe by +0.06 to +0.42** | LOG §97: R15 holdout, random fill rule | Different estimand; the lesson is that a single draw needs a seeded control | C_b runs 20 seeds and its dispersion is reported beside every Sharpe |

**What R16 is predicted to find, stated before the run:** Gate I **fails**. The most likely
pattern: (i) model cells beat the random control in **gross** terms at h = 5 (bid-ask bounce and
liquidity-imbalance reversal inside the hour; Heston, Korajczyk & Sadka 2010); (ii) the edge is
smaller than the refit round trip at every cell; (iii) the screen arm does worse than the
no-selection arm, because ranking on `pm_gap` selects the losing tail of P1. Each of these
predictions can be wrong, and §11 says what follows if one is.

---

## 3. Hypotheses

All hypotheses are judged on **walk-forward out-of-sample development days** (§10.1). Only H1
is confirmed on the sealed holdout (§10.6). Estimand throughout: the daily return of a
compounded account (§10.2).

### H1 (main, the user's design)

> At least one of the 54 main-arm cells (§6) has (a) a compounded net account return whose
> 95% day-clustered CI excludes zero on the positive side, (b) beats each of controls C_a, C_b
> and C_c (§7), (c) meets the multiple-testing bar of §10.4 at the declared trial count, and
> (d) passes the robustness and null conditions of §11. The cell chosen by the rule in §10.5
> then confirms on the sealed holdout.

- *Predicted:* fails (P4, P6).
- *Falsified (H1 rejected) if:* no cell meets (a)–(d) on development, or the chosen cell fails
  the holdout reading. Either result stops the intraday track (§11.3).

### H2 (does the screen add value?)

> The 09:25 screen raises the account return over the no-selection universe on the same days.
> Two contrasts: C_c − C_0 (screen versus none, no model, same entry rule), and Main − C_a
> (screen versus none, same model).

- *Predicted:* both contrasts ≤ 0 (P1).
- *Falsified (prediction overturned) if:* either paired day-clustered CI lies wholly above zero.

### H3 (does the model carry information?)

> At some (k, h), the model's ranked windows have higher **gross** forward h-minute returns than
> the random control's, at matched entry rate.

- Diagnostics: out-of-fold AUC of the classifier against the martingale-calibrated null of §10.7,
  and the gross paired difference Main(gross) − C_b(gross).
- *Predicted:* positive at h = 5, shrinking with h; net of cost, not enough (P4).
- *Falsified if:* at no (k, h, model) does the gross paired difference against C_b have a CI above
  zero. Then the minute window carries no usable information here, whatever the costs.

### H4 (are exits a lever once there is a signal?)

> Within the main arm, at fixed (k, h, model), a TP+SL bracket (fixed-% or ATR) beats the time
> stop.

- *Predicted:* no (P2, P3). The brackets add gap-through stop losses at the next open and cut
  right-tail winners.
- *Falsified if:* any bracket arm beats its time-stop sibling with a paired CI above zero. H4 is
  read inside the 54-cell family and adds no trials.

H2–H4 are explanatory. **Only H1 decides Gate I.** A positive H2, H3 or H4 with H1 failing does
not keep the track alive (§11.3).

---

## 4. Population and universe construction

### 4.1 Data on disk (user decision: no downloads)

`data_root = C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_penny`.

| Cache | Coverage (measured) | Basis |
|---|---|---|
| minute `minute/dt=YYYY-MM-DD/minute.parquet` + `requested.json` | 2,628 days (2016-02-17 .. 2026-08-07), 207,594 symbol-days, 3,507 symbols, per day min 1 / median 54 / max 819, 09:25–16:05 ET | `adjustment="all"` (split- and dividend-adjusted) |
| pre-market `premarket/dt=…` + `requested.json` | 2,621 days, 452,243 symbol-days, 3,555 symbols, median 13 per day, 04:00–09:34 ET | `adjustment="all"` |
| daily `daily/bars_YYYY.parquet` | complete universe, about 6–7k symbols a year; adjusted OHLCV plus `*_unadj` | raw and adjusted |
| quote summaries `quote_summary*.parquet`, `quote_windows_*.parquet`, `quotes_rev/`, `quotes_r4/` | per symbol-day summaries at 09:34, 09:35 and 15:55 only, for rounds 3–5 pools | NBBO summaries, sizes in round lots ×100 |

No minute, pre-market or quote data will be fetched for R16. That includes no top-up of
coverage and no per-minute quote fetch, which resolves the feed.md open question: none.

### 4.2 The cache is not a universe, and why

The minute and pre-market caches are the **union of earlier rounds' requested pools**.
`requested.json` records `sorted(have | needed)` (`scripts/fetch_intraday.py:157`), so it records
*that* a symbol was requested but not *which round* requested it. Pool files still on disk let
the provenance be rebuilt by joining on `(symbol, date)`:

| Pool file | Rows | Rule | Membership known at | Class |
|---|---|---|---|---|
| `candidates.parquet` (R1–3) | 20,558 | opening gap ≥ +10%, prior unadjusted close $0.50–$10, ADV ≥ $100k (`etl/candidates.py`) | **09:30** (opening print; `etl/candidates.py:10`) | C1 |
| `candidates_r4.parquet` | 44,136 | opening gap ≥ +3%, round-4 liquidity band | 09:30 | C1 |
| `candidates_rev.parquet`, `candidates_rev_probe.parquet` (R5) | 68,030 / 6,881 | gap ∈ [−10%, −3%], $0.50–$10, ADV ≥ $1M | 09:30 | C1 |
| `candidates_r15.parquet` | 98,378 | fell ≥ 15% over 5 sessions, $0.50–$10, ADV ≥ $250k. Minute bars were fetched for the **signal session** (features up to its close) and for **entry sessions** (H5 entry prices) | signal session: **its own close** (outcome of D); entry session: before D's open | C2 on signal sessions, C0 on entry sessions |
| `hyp_pool.parquet` (§60 pre-market study) | 420,446 | ADV $100k–$5M on `forensic_universe` (2025-02 .. 2026-08), split days removed | before D's open (the split flag is a pre-announced event) | C0 |
| `forensic_fetch.parquet`, `forensic_sample.parquet` | 23 / 10 | the ten most recent ≥ +50% close-to-close movers (`forensic_sample.py`, `forensic_movers.py`) | **after D's close** | C2 |
| `top30_fetch.parquet`, `top30_sample.parquet` | 37 / 30 | the 30 largest close-to-close gainers of two months | after D's close | C2 |

Classes: **C0** means membership is a function of data before D's open. **C1** means it is a
function of D's 09:30 opening print, which is known before R16's earliest decision (09:35 at
k = 5; §8.1), so it does not leak a return. **C2** means membership is a function of D's close
or later, so it is outcome-selected. **U** means present in `requested.json` but in no pool file;
its origin is unknown.

### 4.3 The provenance audit (runs first, reads no return)

Script `research/intraday/research/r16_provenance.py`. Its output
`data/research_intraday/r16_provenance.parquet` has its sha256 logged as an MLflow tag
(§12). For every `(symbol, D)` in the minute cache on dev days, and separately for the
pre-market cache:

1. Join every pool file in §4.2 on `(symbol, date)`.
2. For `candidates_r15`, recompute the R15 rule from daily rows (drop ≥ 15% over the five
   sessions ending at close D−1 ⇒ entry-session, C0; ending at close D ⇒ signal-session, C2).
   This uses daily data only and reads no minute return.
3. Assign each symbol-day the set of its classes.
4. **Admissible** iff it holds at least one C0 or C1 attribution. A symbol-day in C1 and also in
   C2 (a forensic mover that also gapped ≥ +3%) is admissible, because the C1 pool would have put
   it in the cache anyway, so its presence does not depend on the outcome. Symbol-days that are
   **only** C2 or U are **excluded** from every arm.
5. Report counts by class, year and pool, and the share of each day's cache that is excluded.

The audit reads only `symbol`, `date` and daily rows. It must be committed and its counts recorded
in Amendment 0 **before** any minute return is read.

### 4.4 Universe each day D (identical for every arm, in this order)

1. **Causal filter** on daily rows strictly before D, all on the unadjusted basis
   (`screener.md`, adjustment rule):
   - `min_price = 1.00`, `max_price = 10.00` on `close_unadj` of D−1;
   - `min_adv_usd = 1_000_000`: 20-session median of `close_unadj × (volume ÷ f)` over the
     20 sessions before D, each row converted with its own factor `f = close_unadj / close`;
   - at least 20 prior daily rows;
   - `exclude_split_days = True`, detected from the factor step `f_{D−1} / f_D` only.
2. **Admissible minute coverage on D** (§4.3) (`restrict_to_minute_cache = True`).
3. **Excluded sessions:** half days (13:00 close) and any session `>= holdout_start` unless
   unsealed (§10.6).

The universe is never "every symbol in the cache that day". Steps 1 and 2 are counted
separately per day and per arm. The count removed at step 2 and the count of coverage rejects
(`feed.md`) are logged.

*Why $1–$10 and $1M.* The covered pools all use prior unadjusted close $0.50–$10, so an upper
bound above $10 would claim coverage the cache does not have. Sub-dollar names have a tick that
is ≥ 1% of price and the thinnest quote data. $1M ADV is round 5's standard. It is the band where
the $2,000 order roughly fits the median inside book (LOG §43). None of these numbers was chosen
on an R16 result.

### 4.5 What the universe actually is (stated so it is not over-read)

Because the admissible cache is made of the C1 gap pools and the C0 crash-entry and `hyp_pool`
pools, the R16 universe is in effect **"liquid $1–$10 names that opened with |gap| ≳ 3%, plus
post-crash entry sessions and (from 2025-02) a broad ADV-band pool"**. It is **not** the whole
liquid low-priced market. P1's daily table shows the |gap| < 3% bucket holds about 1.29M of
1.46M symbol-days, and almost none of it is cached. The no-selection control (C_a, C_0) is
therefore "no selection **within the covered pools**". Every result is labelled
`universe = cache-pool-gated`.

Two consequences, both binding:
- **Live reproducibility.** A live system cannot see cache membership. If Gate I passes, G3
  must specify an explicit open-time pool gate that reproduces the C0/C1 rules on live data
  (the 09:30 gap rule, the crash-entry rule) before any paper trading. Deciding that is outside
  R16 (§13), but a pass is conditional on it.
- **Composition drift.** Pools cover different years: the R5 pool spans 2016–2026, the gap ≥ +10%
  and +3% pools were built over different spans, and `hyp_pool` starts in 2025-02. Results are
  reported per pool and per year.

### 4.6 Sensitivities (reported, not trials, cannot rescue a failure)

- S1: universe including C2 and U symbol-days. If this looks materially better than the primary,
  that is the measured size of the outcome contamination.
- S2: C1 pools only (pure open-gap gated).
- S3: per pool (gap-up C1, gap-down C1, crash-entry C0, `hyp_pool` C0).
- S4: symbol-days failing the un-adjustment check (`feed.md`, tolerance 0.02) reported per arm.
  If the failure rate exceeds 2% of candidate symbol-days, it is weighed in the G2 report.
  There is no refetch.

---

## 5. Fixed configuration (every R16-tagged key, by name)

Values not listed here keep their design-doc defaults, which are frozen at G1 by the
`config_hash` in every `screen_id` and MLflow run.

### 5.1 ScreenerAgent (`screener.md`)

| Key | Value | Reason |
|---|---|---|
| `screen_time_et` | `09:25` | spec |
| `premarket_start_et` | `04:00` | design default |
| `arm` | `"screen"` (main, C_b, C_c); `"no_selection"` (C_a, C_0) | spec §4 |
| `min_price`, `max_price` | **`1.00`, `10.00`** (placeholder was `1.00`, `20.00`) | §4.4 |
| `min_adv_usd` | `1_000_000` | §4.4 |
| ranking features | `screen_score` = mean of the day's cross-sectional percentile ranks of `pm_gap` and `pm_rvol`, descending; ties by symbol | see below |
| `top_n` | `10` | see below |
| `exclude_split_days` | `True` | design |
| `holdout_start` | **`2025-08-07`** (§10.6) | confirmed |
| `unseal` | `False` until the single holdout read | spec |
| `restrict_to_minute_cache` | `True`, **on the admissible set of §4.3** | user decision plus §4.3 |

- `pm_gap = pm_close_last × f_{D−1} / close_unadj_{D−1} − 1`, with `pm_close_last` the close of
  the last pre-market bar with `bar_start + 1 min ≤ 09:25`. Raw-to-raw.
- `pm_rvol = Σ_{04:00 ≤ bar_start < 09:25} (close × volume)` on the traded basis, divided by the
  `min_adv_usd` statistic of §4.4 (unadjusted). It is a dimensionless ratio.
- Both features are invariant to `price × c`, `volume ÷ c` (§99). Symbols with no admissible
  pre-market bars are skipped (`screener.md`). Skips are counted separately as "requested, no
  bars" (no pre-market activity) and "not requested" (coverage gap).
- *Why these two.* "Rise potential" before the open is, in the day-trading folklore the user's
  design comes from and in the attention literature, a gap plus unusual pre-market volume (the
  "in play" idea round 1 used). The two are combined by rank-average so neither scale dominates.
  This is one level, not a factor, so nothing is tuned.
- *Why `top_n = 10`.* Pre-market coverage is median 13 per day before admissibility, so 20
  would rarely bind. 10 keeps the screen a selection on most days and gives the
  `max_open_positions = 5` book a choice. The share of days on which `top_n` binds is reported.
  If it binds on fewer than half of OOS days, the G2 report must say the "screen" was mostly
  cache membership.

### 5.2 FeedAgent (`feed.md`)

| Key | Value |
|---|---|
| `session_open_et`, `session_close_et` | `09:30`, `16:00` |
| replay `feed` | `"sip"` |
| emitted basis | unadjusted, factor `f = close_unadj / close` of the same date, `unadjust_tolerance = 0.02` |
| `holdout_start` / `unseal` | `2025-08-07` / `False` |
| per-minute quote fetch | **none** (no downloads). `Quote`s only where a cached summary exists (09:34, 09:35, 15:55), as synthetic quotes per `feed.md` |
| un-adjustment failure handling | symbol-day → `Gap`s plus a rejects entry. Rate reported per arm (S4) |

### 5.3 StateAgent (`state.md`)

| Key | Value |
|---|---|
| `k` | factor: `{5, 15, 30}` |
| `max_span_min` | `2 × k` (frozen design default) |
| `require_contiguous` | `False` |

Training windows in research scripts use the same k, span rule, gap handling and bar basis as
the agent (`state.md`). A parity test on 20 fixed dev days asserts they are identical.

### 5.4 PredictorAgent (`predictor.md`)

| Key | Value |
|---|---|
| `scorer` | `"model"` (main, C_a); `"random"` (C_b); `"constant"` (C_c, C_0) |
| `h` (`horizon_min`) | factor: `{5, 15, 30}` |
| `k` | equals the StateAgent's |
| model family | factor: **`gbdt`** (LightGBM on window aggregates, `model/train.py::ModelConfig` defaults: 300 trees, lr 0.03, 15 leaves, depth 4, `min_child_samples` 80, subsample 0.7, colsample 0.7, λ 10, `random_state` 42) or **`cnn`** (the 1-D CNN head of `model/sequence.py`, `SequenceConfig` defaults: hidden 32, 40 epochs, patience 6, batch 256, lr 1e-3, wd 1e-4, dropout 0.3, seed 42; validation = the last 20% of **training dates**, date-blocked) |
| `random_seed` (C_b) | `0 … 19` (20 seeds) |

*Level choice inside the spec's pairs.* GBDT rather than logistic, because it is the project's
tested tabular learner and handles interactions between aggregates without hand-made terms. A
CNN rather than a transformer, because it is smaller and deterministic on CPU, and it is
well-posed at k = 5. R3's transformer advantage was an exploratory, post-hoc result (LOG §26)
and is not a reason to prefer it. Logistic and transformer are **not run** (§13).

**Label (per (k, h); exit-independent).** For the window ending at `ts = T`: entry
`p_in = open` of the bar with `bar_start = T` (one-bar latency, §8.1), exit `p_out = open` of
the bar with `bar_start = T + h` (the time-stop fill rule), or the 15:55 rule if `T + h > 15:55`.
`y = 1{p_out / p_in − 1 > c_label}`, with **`c_label` a single fixed constant**: the median refit
round trip at $2,000 over the dev no-selection universe (§9.3). It is computed once, before any
model is fit, and recorded in Amendment 0. It is never per-trade (LOG §34). Score =
`P(y = 1)`. Windows with no `p_out` bar are dropped from training and never synthesised.

**Features.** From `Window.bars` only. All are ratios and must pass the joint rescale test
(`price × 750`, `volume ÷ 750`, §99).
- GBDT aggregates: log returns of the newest close over the last 1, 3 and k bars; (high − low)
  / close of the newest bar and over the window; newest close / window VWAP − 1; newest bar
  volume / window median volume; fraction of up-bars; count of missing minutes inside the span;
  minutes since 09:30 / 390.
- CNN raw channels per bar: log(open, high, low, close, vwap / newest close); log(1 + volume /
  window median volume); minutes since 09:30 / 390.
- No price, share-volume or dollar-volume level. No pre-market or daily feature in the model
  (`predictor.md`).

**Training population.** Windows of the **no-selection** universe on the fold's training dates.
The same frozen artifact scores the main arm and C_a, so Main − C_a isolates the screen.

### 5.5 RiskAgent (`risk.md`)

| Key | Value | Reason |
|---|---|---|
| `start_equity_usd` | `25_000` | frozen design default |
| `position_notional_usd` | `2_000` | scale of the cost measurements (P6) |
| `max_open_positions` | `5` | frozen |
| `max_entries_per_symbol` | `1` | frozen |
| `entry_threshold` | model: **in-fold 0.90 quantile** of the training-fold scores over training windows (the top decile). random: **`0.90`** (the same entry-eligible rate in expectation, with no data used). constant: **`1.0`** (first window) | matched entry rates without looking at test data |
| `exit_mode` | factor: `{"time_stop", "fixed_pct", "atr"}` | spec |
| `hold_min` | `h` | spec §4 |
| `time_stop` | `min(ts + h, 15:55)` in **every** exit arm | the exit factor changes only the bracket |
| `tp_pct`, `sl_pct` (by h) | h=5: `0.0375`, `0.025`; h=15: `0.06375`, `0.0425`; h=30: `0.09`, `0.06` | below |
| `tp_atr`, `sl_atr` (by h) | h=5: `2.10`, `1.40`; h=15: `3.63`, `2.42`; h=30: `5.13`, `3.42` | below |
| `atr_period` | **`5`** bars (design default 14 overridden) | below |
| `quote_size_frac` | **`0.5`** | §9.2 |
| `quote_max_age_min` | `5` | frozen |
| `bar_participation_cap` | `0.02` | frozen; the replay binding cap (§9.2) |
| `daily_loss_limit_pct` | `0.02` | frozen; stops entries only |
| `last_entry_et`, `flatten_et` | `15:30`, `15:55` | frozen |

**TP/SL derivation, fixed before any R16 data is read.**
- *Scale.* The round-2 gap-up population had daily σ ≈ 21.6% (LOG §16), so σ₁ ≈ 21.6%/√390 ≈
  1.09% per minute. The h-minute scale is taken as `s_h = 1.1% × √h`, so `s_h` = 2.46%, 4.26% and
  6.02% at h = 5, 15 and 30. The model's horizon is h, so a bracket much wider than `s_h` would
  never trigger and one much narrower would be bid-ask noise.
- *Geometry.* `SL = 1.0 × s_h`, `TP = 1.5 × s_h`, both arms, every h. A 1.5 : 1 bracket breaks
  even before costs at a 40% target-first rate under a driftless walk. The stop sits at about 5–15
  entry half-spreads (P6: median 40 bps), outside the bounce band. The same multipliers are used
  in both bracket arms, so **fixed versus ATR isolates one thing: a population-constant scale
  against a symbol-adaptive scale.**
- *Fixed %:* `sl_pct = s_h` rounded to 0.25%: 2.50%, 4.25%, 6.00%. `tp_pct = 1.5 × sl_pct`.
- *ATR:* for a Brownian minute, E[high − low] = 2√(2/π) σ₁ ≈ 1.60 σ₁, so σ₁ ≈ ATR₁ / 1.6 and
  `sl_atr = √h / 1.6` (1.40, 2.42, 3.42), `tp_atr = 1.5 × sl_atr`. ATR = mean true range of the
  last `atr_period` one-minute bars. The first bar's TR is high − low.
- *Why `atr_period = 5`.* With 14, the ATR arm would skip every k = 5 entry before 09:44 while
  the other arms took them. The exit factor would then also change the entry set. With 5, every
  k level has an ATR at its first window.
- No level is revisited after results. A bracket that never triggers is a result, not a reason
  to tighten it.

### 5.6 ExecutionAgent (`execution.md`) and MonitorAgent (`monitor.md`)

| Key | Value |
|---|---|
| fill latency | open of the first bar with `bar_start ≥` decision `ts` |
| stop and target fill | open of the next bar after the triggering bar, gaps included, **never the stop level**; both in one bar ⇒ stop |
| fills during halt | none |
| `quote_size_frac` | `0.5` (must equal the RiskAgent's) |
| `quote_max_age_min`, `bar_participation_cap` | `5`, `0.02` |
| `entry_max_wait_min` | `2` |
| `eod_et` | `16:01` |
| `impact_coefficient` | `1.0` (Y of the square-root law; not refittable without fills, stated as an assumption) |
| `levels_penalty` | `1.0` |
| `default_half_spread_bps` | replaced by the refit model's prediction (§9.3); never the 100 bps placeholder |
| broker | `SimBroker` only |
| `infer_halts_from_gaps` | **`False`** primary. Sensitivity run with `True`, `gap_halt_min = 5` (reported, not a trial) |

Without inference, "nothing fills during a halt" holds by construction: a halted minute is a
`Gap` and has no bar to fill on, so a pending exit fills at the first bar after the halt. The
historical halt log is empty (`monitor.md`), and that is stated in the G2 report.

---

## 6. Factors and trial count

### 6.1 Experiment matrix (main arm)

Full factorial, single-variable contrasts read inside it:

| Factor | Levels |
|---|---|
| k (bars seen) | 5, 15, 30 |
| h (horizon, min) | 5, 15, 30 |
| model | `gbdt`, `cnn` |
| exit | `time_stop`, `fixed_pct`, `atr` |

**3 × 3 × 2 × 3 = 54 main-arm cells.** Cell id `M-k{k}-h{h}-{model}-{exit}`, e.g.
`M-k15-h5-gbdt-atr`. Models trained: 3 × 3 × 2 = 18 per fold × 6 folds = 108 fits (the exit
factor needs no refit).

### 6.2 Controls (run per cell; not selectable, so not trials)

| Control | Cells | Varies |
|---|---|---|
| C_a no selection + model | 54 | `arm = no_selection` |
| C_b random score | 54 × 20 seeds | `scorer = random` |
| C_c screen only | 27 (k, h, exit) | `scorer = constant` |
| C_0 baseline: no selection, no model | 27 | `arm = no_selection`, `scorer = constant` |

### 6.3 Trial count, declared

- **R16's own family: N₁₆ = 54.** Only main-arm cells can be selected.
- **Prior project trials:** 287 through R15 Part I (LOG §88: `search_trials.jsonl` 195 +
  `PRIOR_TRIALS` 72 + R15 cells). R15 Part II (H2–H5) added more, and its exact count is not
  tabulated in the log. **`PRIOR_TRIALS_R16 = 350`** is a conservative upper bound (it allows up
  to 63 Part II cells; a larger N only deflates harder).
- **Deflated Sharpe trial count N = 350 + 54 = 404** (binding). DSR at N = 54 is also reported.
- Every amendment made after an R16 number has been seen adds 1. Every holdout read is
  recorded (the R16 holdout is read at most once).
- The breakout lab's trials (`research/breakout`) are a different question on a different branch
  and are excluded. This is flagged for the critic.

### 6.4 Resource estimate

GBDT fits are seconds each. The CNN takes minutes per fit on CPU, so 54 fits is a few hours.
Replays: (54 + 54 + 1,080 + 27 + 27) = 1,242 cell-runs × about 1,870 OOS days. The engine runs
one session per run and is too slow for the full grid, so the grid uses a **vectorised research
simulator** that implements §8 exactly. It must pass a **parity test against the engine**
before any grid result is read: on 50 fixed dev days per arm (seeded draw, seed 16), fills,
quantities, costs and daily P&L must be identical to the cent. The selected cell and its four
controls are then replayed **in the engine** on every OOS day, and the engine numbers are the
ones Gate I reads. The event-log hashes come from these engine replays. Storage: under 5 GB
under `data/research_intraday/`.

### 6.5 Execution order

1. G1 sign-off on this document.
2. Provenance audit (§4.3), then Amendment 0: class counts, `c_label`, cost-refit
   coefficients, coverage per year. No return is read.
3. Cost refit (§9.3), using quotes only and no strategy returns.
4. Null battery and leakage guards on one fold (§10.7).
5. Vectorised-to-engine parity test.
6. Full dev walk-forward grid: main arm and all controls.
7. Engine replay of the selected cell and its controls, then the G2 report to the critic.
8. After G2 sign-off: the single holdout read (§10.6), then the Gate I decision.

---

## 7. Controls

All controls run on **the same days, the same universe steps 1–2, the same fills, costs, risk
limits, folds and (where they apply) the same frozen model artifact** as the main cell they are
paired with.

- **C_a, no selection.** Every name passing §4.4 (the spec's "all candidates passing the causal
  liquid low-priced filter"), with no ranking and no `top_n`, scored by the same model. Main −
  C_a = the value of the screen given the model.
- **C_b, random feature as the score.** The screen arm with `score` = a deterministic
  uniform(seed, symbol, ts) (`predictor.md`) and `entry_threshold = 0.90`, the same top-decile
  entry rate as the model in expectation. 20 seeds. Main − C_b = the value of the model's
  information at matched turnover.
- **C_c, screen alone.** `scorer = constant`: each screened name is entered at its first window
  (09:30 + k), then the same exit. Main − C_c = the value of the model given the screen.
- **C_0, nothing.** No selection, constant scorer: what an account earns by buying covered liquid
  names at 09:30 + k. It is the anchor for H2 and P1.
- **SPY.** The project rule requires a SPY comparison. Report the same-day SPY open→close return
  from the daily cache (it is reported only and is not a control the gate reads, since the account
  is flat overnight), and each arm's daily correlation and alpha against it. If SPY is absent from
  the daily cache, that is reported and nothing is fetched.

---

## 8. Fill model

### 8.1 Timing

A `Window` is emitted at `ts = T` (the end of the newest bar). The `Signal` and `OrderIntent`
are at T. The entry fills at the **open of the first bar with `bar_start ≥ T`**. The bar that
caused the decision can never fill it. Earliest decision 09:35 (k = 5), 09:45 (k = 15), 10:00
(k = 30). C1 pool membership (09:30 open) is known by then.

### 8.2 Exits

- Bracket check from the fill bar on: `low ≤ SL` or `high ≥ TP` decided at that bar's `ts`, filled
  at the **next bar's open, whatever it is**. A gap-through stop fills below the stop and a
  gap-through target fills above it. Both in one bar ⇒ the stop.
- Time stop: first tick ≥ `time_stop` ⇒ the open of the bar starting there.
- `Gap` minutes have no bar, so the fill moves to the next existing bar. Entries expire after
  `entry_max_wait_min = 2`. Exits wait until the session ends. At `eod_et` an unfilled
  non-halted position fills at its last seen close. A halted one is marked at its last price and
  reported.
- Bid-ask bounce: a TP "touch" on a single print at the ask, followed by an open back inside the
  spread, is filled at that open. This is conservative by construction and stated as such.

### 8.3 Size (cap fraction stated)

`qty = floor(min($2,000, cash) / ref_price)`, then capped:
- with a cached quote no older than 5 min: `qty ≤ floor(0.5 × ask_size)` at entry
  (`0.5 × bid_size` at exit). **Cap fraction 0.5** of the shares at the best quote. The haircut
  allows for the quote fading over the one-bar latency between decision and fill.
- otherwise (most replay minutes): `qty ≤ floor(0.02 × volume)` of the decision bar at intent and
  of the filling bar at fill. An entry remainder is cancelled. An exit remainder carries to the
  next bar.
- The G2 report states how many entries each cap bound (quote, bar, notional), per arm.

---

## 9. Cost model

### 9.1 Formula

One leg (`etl/quotes.py::leg_cost_bps`) = half-spread + `Y σ √participation` + depth shortfall,
with `Y = 1.0`, `levels_penalty = 1.0`, `σ` = the trailing 20-session daily return σ before D, and
participation = fill qty / 20-session median unadjusted share volume before D. Every level is
unadjusted (`execution.md`). Entry pays on the ask side and exit on the bid side. Costs are
charged once per leg, separately from the bar reference price, and auditable per fill.

### 9.2 Inputs

Half-spread and inside notional come from a cached quote where one exists for that symbol-day
and minute window. Elsewhere they come from the refit model (§9.3). Half-spread is floored at half
a tick: `0.5 × $0.01 / price × 10⁴` bps.

### 9.3 The refit (on this round's population, before any strategy result)

- **Fit set:** dev-period symbol-days in the R16 **no-selection universe** (§4.4) that have cached
  quote summaries at 09:34/09:35 and/or 15:55. Coverage is reported.
- **Models (OLS, specification fixed now):** `log(half_spread_bps)`, `log(ask_notional)` and
  `log(bid_notional)` each regressed on `log(close_unadj_{D−1})`, `log(adv20_usd)`,
  `log(σ_20d)`, a window indicator (open = 09:34/35, close = 15:55), and year dummies.
  All regressors are known before D.
- **Time of day:** a leg at minute t uses the open-window prediction for t ≤ 09:35, the
  close-window prediction for t ≥ 15:55, and **linear interpolation in clock time** in between.
  Spreads fall through the morning, so interpolation over-charges midday relative to an L-shaped
  profile. That is conservative and stated as such.
- **Independent cross-check, reported only:** Corwin & Schultz (2012) and Roll (1984) spread
  estimates from the minute bars of the same symbol-days, compared by time-of-day bucket. If
  either exceeds the refit by more than 50% at the median in a bucket, the G2 report flags it.
- **Fixed outputs:** coefficients, R², the median round trip at $2,000 (which becomes
  `c_label`), recorded in Amendment 0.
- **Sensitivity, reported and never a trial:** all legs × 1.5.

---

## 10. Evaluation

### 10.1 Folds

`eval/folds.py::walk_forward_folds(dev_dates, n_splits=6, embargo_days=5, expanding=True,
min_train_days=504)` over the unique dev dates (2016-02-17 .. 2025-08-06, half days removed),
then `assert_folds_are_clean`. Labels are same-session (h ≤ 30 min, flat by 15:55), so labels never
straddle a fold boundary. The 5-day embargo covers the 20-day daily statistics behind the
universe filter and cost inputs. The in-fold threshold, the CNN's early-stopping split and every
calibrator are fit on training dates only. The OOS record is the six test blocks concatenated
(about 1,870 sessions, 2018 → 2025-08).

### 10.2 Primary metric: the compounded account

The account starts at $25,000 on the first OOS day, with fixed $2,000 notional per position, at
most 5 open positions, flat every night. Daily return `r_d = E_d / E_{d−1} − 1`, flat days 0
(days with an empty universe are included as zero). Primary statistic: the **geometric mean daily
net return** `g = (Π(1 + r_d))^{1/T} − 1` and the total compounded return.
**CI:** day-clustered bootstrap, resampling OOS days with replacement and compounding each
resample, B = 10,000, seed 16, 95% percentile interval. A stationary bootstrap (mean block 5
days; Politis & Romano 1994) is reported beside it.

### 10.3 Reported for every cell and control

- mean and median per-trade net and gross; hit rate; mean against geometric daily return (P7);
- the 20 largest trades by P&L read individually (symbol, date, entry/exit minute, fill source,
  whether a gap-through), and the compounded return after removing the top 1% of trades and the
  top 10 trades;
- trades, distinct trading days, positions per active day, % of days active;
- Sharpe (annualised √252), Mertens/Lo SE, PSR, MinTRL, max drawdown against its Magdon-Ismail &
  Atiya expectation (`eval/battery.py`);
- cap binding counts (§8.3), the cost decomposition per leg, quote-source usage, coverage
  rejects, the un-adjustment failure rate, and how often `top_n` binds;
- per year, per pool (S3), and gross alongside net;
- the SPY comparison (§7).

### 10.4 Multiple testing

- For each of the 54 main cells, a one-sided day-clustered bootstrap p-value for mean daily net
  return > 0. **Holm** step-down over the 54 at family-wise α = 0.05.
- **Deflated Sharpe** (Bailey & López de Prado 2014) at **N = 404**. `Var(SR_n)` = the variance
  of annualised Sharpe **across the 54 main cells** (the harsher choice, LOG §88), with the
  seed-to-seed variance of C_b reported as the alternative. Gate threshold: DSR ≥ 0.95.
- H2–H4 contrasts are explanatory and are reported with BH-FDR (q = 0.05) within their own
  families. They do not enter the gate.

### 10.5 Selection rule (fixed now)

Among cells meeting every development condition of §11.1, choose the **highest DSR**. Ties go to
`gbdt` over `cnn`, then smaller k, then `time_stop` over brackets. No other cell may be
substituted later.

### 10.6 Holdout protocol

- **`holdout_start = 2025-08-07`** is confirmed. It is exactly 12 months before the last cached
  day (2026-08-07), matching spec §4 and Ruling 10. It is a regular full session (a Thursday).
  Nothing argues for moving it: R16 has read no data, and moving the boundary after the
  provenance audit would itself be a choice made on data. Development uses dates <
  2025-08-07 only.
- **Disclosure:** this window was **not** clean in the project's history. R5 Gate R read
  09:31–11:00 → 15:55 returns on all reversal-pool sessions through 2026, and R15 H3/H5 read
  minute features and 10:00 entry prices in 2024–2026. None asked R16's question (a minutes-ahead
  model with brackets), so it is sealed **for this question**, as the spec states.
- **Read exactly once**, after the critic's G2 sign-off on the development result, and only if
  some cell passed development. Read: the selected cell and its controls C_a, C_b (20 seeds), C_c
  and C_0. The model is retrained once on **all** dev dates with the frozen configuration, the
  threshold quantile is recomputed on those dev dates, and both are frozen with `model_hash` before
  `unseal=True`. The unseal is recorded (who, when, `config_hash`) in MLflow and as an
  amendment.
- **Holdout pass (all required):** (i) the compounded net return's day-clustered 95% CI excludes
  zero on the positive side; (ii) the point estimate of the paired daily difference against each
  of C_a, C_b (seed mean) and C_c is > 0; (iii) the compounded return stays > 0 after removing
  the top 1% of trades. The CI-on-differences requirement is applied on development and not
  repeated on about 250 holdout days, where it would be underpowered. This is stated now, not
  relaxed later.
- If no cell passes development, **the holdout is not read**. It stays sealed and the round stops.

**Forward paper trading is the final holdout.** It is pre-registered at **63 sessions**, extended
to `min(MinTRL, 252)` sessions if the holdout MinTRL at 95% exceeds 63. Pass: the net compounded
paper return > 0, and the nightly live-versus-replay diff within G3's tolerance. It runs only
after Gate I passes and G3 signs off. It is not part of this round's work (§13).

### 10.7 Nulls and guards (preconditions; a failure voids the run, not the hypothesis)

- `eval/nulls.py::run_all` on one fold per (k, h, model): shuffled label, date shift, the feature
  timing tripwire and constant return must destroy performance. A null "failing" by improving
  means negative skill, not leakage (LOG §35), and is reported as such.
- **Martingale-calibrated IC/AUC null** (memory: rank IC against a forward return is not centred
  on zero): the classifier's out-of-fold AUC is compared with its AUC on synthetic minute paths
  that keep each symbol-day's realised minute σ but have zero drift, so sigma-correlated scores do
  not read as skill.
- Joint scale-invariance test on every feature (`price × 750`, `volume ÷ 750`).
- Assertions: no input after the decision `ts`; no daily row ≥ D in a filter except the factor
  ratio for the split flag; no session ≥ `holdout_start` without `unseal`.
- Determinism: the engine replay of 5 fixed days run twice gives identical event-log hashes.

---

## 11. Gate I decision rule

### 11.1 Development conditions (all required, for at least one main cell)

1. **Account:** compounded net geometric-mean daily return has a day-clustered 95% CI with lower
   bound > 0.
2. **Controls:** paired daily-difference CI lower bound > 0 against **each** of C_a, C_b (the
   20-seed mean daily return) and C_c. Also, the cell's compounded return exceeds the **95th
   percentile** of the 20 C_b seeds' compounded returns.
3. **Multiple testing:** Holm-adjusted p < 0.05 over the 54 cells, **and** DSR ≥ 0.95 at N = 404.
4. **Robustness:** the compounded return point estimate stays > 0 after removing the top 1% of
   trades by P&L. The median trade is reported, and a negative median with a positive mean is
   flagged in the G2 report.
5. **Power floor:** ≥ 300 OOS trades on ≥ 150 distinct days.
6. **Validity:** §10.7 passes, the parity test passes, and the engine replay of the cell agrees
   with the vectorised result.

### 11.2 Then

Choose the cell by §10.5 → critic G2 → the single holdout read → §10.6 pass conditions.
**Gate I passes iff development and holdout both pass.**

### 11.3 What stops the intraday track

Any of the following ends the track. The result is recorded in LOG and in an amendment. There is
**no re-tuning, no new level, no other cell, no second holdout read, no new pool definition and no
change of cost assumption**, and Phases 4–5 are not built:

- no main cell meets §11.1 on development; or
- the selected cell fails §10.6 on the holdout.

A result of the form "H3 shows gross skill but nothing clears cost" is a **stop**, not a reason
to continue. Whether some later question deserves a new pre-registration (for example, after the
user's data-budget decision) is the user's call, and it would start a new round with this one's
trials counted.

### 11.4 Infeasibility (neither pass nor fail)

If Amendment 0 shows the admissible dev universe cannot reach the §11.1.5 power floor even for
C_0 (for example, a median below 3 admissible names a day after step 1), the round is declared
**infeasible on cached data**. No return is read and the question goes to the user's data-budget
decision (spec, user decision: the data budget is decided after the research). This is checked
on counts only.

---

## 12. Tracking

- MLflow experiment **`intraday`**, local file store. `MLFLOW_ALLOW_FILE_STORE=true` is set in the
  tracker's constructor, not at import. The experiment is created on the first logged run.
- One run per (cell or control, fold) plus one aggregate run per cell. **Params:** every key in
  §5, cell id, arm, scorer, seed, fold index, `config_hash`, `screen_id`, `model_hash`,
  `c_label`, cost-refit coefficient hash, prereg commit hash. **Metrics:** everything in §10.2–10.4.
  **Tags:** the **event-log hash** (sha256 over the ordered per-session log hashes of the engine
  replay; for vectorised runs, the parity-test log hash plus the simulator's trade-file sha256),
  the provenance-audit sha256, git commit, `universe = cache-pool-gated`, `period = dev | holdout`.
- The trial ledger `data/research_intraday/r16_trials.jsonl` gets one line per main cell. Its
  line count must equal 54 at G2.

---

## 13. What is NOT done this round

- No data download of any kind: no minute, pre-market or quote fetch, and no coverage top-up.
- No logistic model and no transformer (the unselected levels of the spec's pairs).
- No hyperparameter, threshold, TP/SL, `top_n`, price band or ADV tuning. No level is added after
  results.
- No short side, no overnight holds, no leverage, no more than one entry per symbol per day.
- No pre-market or daily features inside the model.
- No use of C2 or U symbol-days in any primary number (they appear only in S1).
- No holdout read before G2, and at most one ever.
- No live feed, no broker connection, no paper trading, no open-time pool gate build (G3's job if
  Gate I passes).
- No per-trade-mean verdicts. The account decides.

---

## 14. Open risks

1. **The universe is gated by open-time pool rules, not by the screen** (§4.5). Replay results
   apply to a universe a live system would have to rebuild explicitly at 09:30. If the critic
   rules this a mismatch with the user's design, the only fixes are data (a download) or an
   explicit pool gate. Neither is in this round.
2. **Pre-market coverage is thin and changes over time** (median 13 per day; `hyp_pool` densifies
   it only from 2025-02). The screen may mostly rank cache membership. Mitigated by reporting the
   `top_n`-binding share and S3, not removed.
3. **Quotes exist only at 09:34/09:35 and 15:55.** The bar-volume cap is a proxy for the
   quote-size cap (`risk.md`), and midday spreads are interpolated. Both are conservative by
   design, and both could be wrong in either direction for thin names.
4. **Bid-ask bounce in the label.** Minute opens and closes are trade prints on either side of
   the spread. The model can learn bounce (HKS 2010 show sub-hour reversal is partly bounce). The
   cost model charges half-spread on top of the open print, which double-counts on average when
   the print is already at the ask. That is conservative, but it is a source of apparent gross
   skill in H3 that is not tradable.
5. **Halts are invisible** in historical replay (an empty halt log). They appear as gaps, and
   fills move to the next bar. LULD halts on the names that move most make the stop fill price
   the most model-dependent number in the round.
6. **Prior trial count is a bound, not a tally** (§6.3). If the critic requires the exact R15
   Part II count, it must be tabulated before the run, and only an increase is allowed.
7. **Survivorship.** The cache keeps delisted symbols only where they were requested before
   delisting. Coverage of names that later delisted is reported per year.
8. **Holdout non-virginity** (§10.6). Disclosed, not fixable.
9. **Cost Y = 1.0 is assumed.** It is not measurable without fills. The × 1.5 sensitivity is
   reported.

---

## 15. Literature

Found with the `scholar-search` MCP this session (year and venue as returned):

- Gao, L., Han, Y., Li, S. Z. & Zhou, G. (2018). Market intraday momentum. *Journal of Financial
  Economics*. Index-level: the first half-hour return predicts the last half-hour. It motivates
  that the opening minutes carry information about the rest of the day; it is not evidence for
  single microcaps.
- Heston, S. L., Korajczyk, R. A. & Sadka, R. (2010). Intraday patterns in the cross-section of
  stock returns. *Journal of Finance* 65, 1369–1407. Half-hour periodicity; short-term reversal
  driven by liquidity imbalances lasting under an hour **and bid-ask bounce**. This is the basis
  for the H3 prediction and risk 4.
- Lou, D., Polk, C. & Skouras, S. (2019). A tug of war: overnight versus intraday expected
  returns. *Journal of Financial Economics*. Intraday and overnight components of anomalies
  differ. The open→close prior (P1) is an intraday-component fact.
- Eraker, B. & Ready, M. (2015). Do investors overpay for stocks with lottery-like payoffs? An
  examination of the returns of OTC stocks. The scholar-search record is the SSRN version (DOI
  10.2139/SSRN.1733225); the tool did not return the journal venue. Lottery-like low-priced
  stocks earn poor average returns, which supports a prior against buying the screen's gap-ups.
- Kaminski, K. M. & Lo, A. W. (2014). When do stop-loss rules stop losses? (The tool returned no
  venue.) Stop-losses add value only when returns have momentum or regime structure. Under a
  random walk they lower expected return. This is the theory behind H4's prediction.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient
  market. *Journal of Finance* 39, 1127–1139. Bounce-induced negative autocorrelation; the spread
  cross-check of §9.3.
- Corwin, S. A. & Schultz, P. H. (2012). A simple way to estimate bid-ask spreads from daily high
  and low prices. *Journal of Finance* 67, 719–760. The spread cross-check of §9.3.
- Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection
  bias, backtest overfitting, and non-normality. *Journal of Portfolio Management* 40. §10.4.
- Nagel, S. (2012 working-paper record "Evaporating liquidity"; the tool returned 2011 and no
  venue). Reversal as liquidity provision, behind P5.
- Barber, B., Lee, Y.-T., Liu, Y.-J. & Odean, T. (2009 in the record's journal field; tool year
  2007). Just how much do individual investors lose by trading? *Review of Financial Studies* 22,
  609–632. Also "Do individual day traders make money? Evidence from Taiwan" (2004 working paper).
  The base rate for retail-style day trading is negative net of costs.
- Zarattini, C., Aziz, A. & Barbon, A. (2024). Beat the market: an effective intraday momentum
  strategy for S&P500 ETF (SPY). *SSRN*. A practitioner intraday momentum design with dynamic
  trailing exits on the index. It shows the user's design family has published index-level
  analogues, not microcap ones.
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform
  deep learning on tabular data? arXiv (cs.LG). This is why GBDT is the aggregate learner and the
  CNN is the raw-window learner.

Cited from the project's earlier reference lists. Each was searched for this session and **not
retrieved** by scholar-search, so the year and venue below are as recorded in
`PREREGISTRATION_R15.md`:
- Bali, T., Cakici, N. & Whitelaw, R. (2011). Maxing out: stocks as lotteries and the
  cross-section of expected returns. *Journal of Financial Economics* 99(2).
- Bogousslavsky, V. (2021). The cross-section of intraday and overnight returns. *Journal of
  Financial Economics* 141(1).
- Politis, D. & Romano, J. (1994). The stationary bootstrap. *JASA* 89(428).
- Magdon-Ismail, M. & Atiya, A. (2004). An analysis of the maximum drawdown. *Quantitative
  Finance* 4(5).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley (ch. 7, purging and
  embargo).

**Gap this round sits in.** The intraday predictability literature found is index-level
(Gao et al.; Zarattini et al.) or large-cross-section at half-hour frequency (HKS). Nothing found
tests minute-ahead prediction **inside liquid sub-$10 names with realistic quote-capped fills**.
That is the absence the user's design probes, and it is also why the priors from this project's
own rounds weigh more than the literature here.

---

## Amendments

*(none; Amendment 0 records the provenance-audit counts, `c_label` and the cost-refit
coefficients before any return is read)*
