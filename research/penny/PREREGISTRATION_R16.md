# Round 16 Pre-Registration: the user's intraday design, tested against its own controls

**Status: REVISION 1, for critic G1 re-check. Nothing has been run.** The first draft (commit
32c06c5) was blocked by G1 (`research/intraday/docs/critiques/G1-R16.md`, findings B1, B2, W1–W10,
N1–N9). This revision applies G1's required edits E1–E20 and one controller correction (V1, the
volume basis). The changes are listed at the end ("Changes after G1"). This document is committed
before any round-16 result exists, and its commit hash is the timestamp. Constants may not be
edited in place once G1 signs off. A change after sign-off is a dated **Amendment** appended at
the end, and an amendment made after any R16 number has been seen counts as a trial (§6.3).

Binding inputs: `docs/superpowers/specs/2026-10-26-intraday-system-design.md` (§1 priors, §4
round 16), and the design docs `research/intraday/docs/design/{screener,feed,state,predictor,risk,execution,monitor}.md`.
Every key those docs mark `R16` or `[research]` is fixed by name in §5.

> ### Conditional on U1 (user decision, pending)
>
> **U1.** Is "09:25 pre-market score → 09:30 open-gap gate G(D) → `top_n` inside G" still the
> user's design? This document is written **assuming U1 = yes**. The user forbade downloads, and
> on cached data the gated variant is the only admissible universe (§4.2–§4.5, G1 ruling 1).
>
> - **Every R16 result is reported as conditional on U1.** Each MLflow run carries the tag
>   `conditional_on = U1`, and every table in the G2 report states it.
> - **No R16 result is Gate-I evidence until the user confirms U1.** A development or holdout
>   pass before that confirmation is recorded and nothing is built on it.
> - **The holdout is not unsealed until the user has confirmed U1.**
> - **If the user rejects U1, R16 is void as a test of the user's design.** Its numbers stay
>   recorded as a test of the gated variant only. Testing the ungated design needs a new round with
>   a download of pre-market and minute bars on the whole filtered universe, its own
>   pre-registration and budget, and this round's trials counted.
>
> **Strata that ARE tested** (each only inside G(D) ∩ cache ∩ the §4.4 filter, i.e. $1–$10 prior
> unadjusted close and ADV ≥ $1M, and only in the years its source pool covers):
> - gap ≥ +10% (source pool R1–3; its own ADV ≥ $100k is implied by the $1M filter);
> - +3% ≤ gap < +10% with ADV ≥ $5M (R4 definition);
> - −10% ≤ gap ≤ −3% (R5 pool, ADV ≥ $1M);
> - post-crash entry sessions: a fall ≥ 15% over the five sessions ending at close D−1 (R15
>   rule). This bracket drops out if the R15 pool fails the completeness check of §4.3 step 4a
>   (U2, information only).
>
> **Strata that are NOT tested:**
> - |gap| < 3%, except post-crash entry sessions: about 1.29M of 1.46M liquid low-priced symbol-days (P1), which is most of the market;
> - +3% ≤ gap < +10% with ADV $1M–$5M;
> - gap < −10%, except names that also meet the post-crash rule;
> - the live 09:25 screen ranking the **whole** filtered market. Here the percentiles and `top_n`
>   are taken inside G at 09:30 (§5.1), and `Candidates` is stamped 09:30, not 09:25.

---

## 1. Purpose

The question is whether **the user's design** makes money net of realistic costs in cached
history. The design is a pre-open screen for intraday-rise potential at 09:25 ET, ranked on
pre-market features. A light model then reads the last k one-minute bars after the open and
predicts the move over the next h minutes, and the RiskAgent enters with a TP/SL bracket.
"Makes money" means the compounded account, not the per-trade mean (LOG §49). The design has
to beat three controls that remove one part each: the screen, the model's information, and the
model itself. As replayed, the screen ranks only inside the 09:30 open-gap gate G(D) (§4.4
step 1b, §5.1), so every result is conditional on U1 (box above).

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
| `candidates_r4.parquet` | 44,136 | opening gap ≥ +3%, ADV ≥ $5M (round-4 definition; LOG.md:970) | 09:30 | C1 |
| `candidates_rev.parquet`, `candidates_rev_probe.parquet` (R5) | 68,030 / 6,881 | gap ∈ [−10%, −3%], $0.50–$10, ADV ≥ $1M | 09:30 | C1 |
| `candidates_r15.parquet` | 98,378 | fell ≥ 15% over 5 sessions, $0.50–$10, ADV ≥ $250k. Minute bars were fetched for the **signal session** (features up to its close) and for **entry sessions** (H5 entry prices) | signal session: **its own close** (outcome of D); entry session: before D's open | C2 on signal sessions, C0 on entry sessions |
| `hyp_pool.parquet` (§60 pre-market study) | 420,446 | ADV $100k–$5M on `forensic_universe` (2025-02 .. 2026-08), split days removed. A **pre-market** pool: minute bars were not fetched for it (`hypothesis_premarket.py:26-31`) | ADV band: before D's open. Split exclusion: `any_split` from D's open and open-to-high (`build_universe.py:124-129`) | Rows with open_unadj / prev_close_unadj ≥ 1.8 are **C2**; all other rows are **C0** |
| `forensic_fetch.parquet`, `forensic_sample.parquet` | 23 / 10 | the ten most recent ≥ +50% close-to-close movers (`forensic_sample.py`, `forensic_movers.py`) | **after D's close** | C2 |
| `top30_fetch.parquet`, `top30_sample.parquet` | 37 / 30 | the 30 largest close-to-close gainers of two months | after D's close | C2 |

Classes: **C0** means membership is a function of data before D's open. **C1** means it is a
function of D's 09:30 opening print, which is known before R16's earliest decision (09:35 at
k = 5; §8.1), so it does not leak a return. **C2** means membership is a function of D's close
or later, so it is outcome-selected. **U** means present in `requested.json` but in no pool file;
its origin is unknown.

*Note on `hyp_pool` (B2).* The split detector behind `any_split` reads D's open-to-high
(`corporate_actions.py:49-60,94-105`). For gap multiples ≥ 1.8 it keeps the days that ran and
drops the inert ones, so those rows are outcome-selected and are classed C2. Below a 1.8 multiple
the detector does not depend on D's session, so those rows stay C0.

*Note on `forensic_universe`.* It requires a finite close on D (`build_universe.py:152-157`).
That is a same-day survivorship condition. Its effect is negligible (a symbol-day with no D close
has no session to trade), but it is recorded here, and the G2 report states it.

### 4.3 The provenance audit (runs first, reads no return)

Script `research/intraday/research/r16_provenance.py`. Its output
`data/research_intraday/r16_provenance.parquet` has its sha256 logged as an MLflow tag
(§12). For every `(symbol, D)` in the minute cache on dev days, and separately for the
pre-market cache:

1. Join every pool file in §4.2 on `(symbol, date)`.
1a. **Pool reproducibility:** for each pool file, recompute its rule from the daily cache and
   report the symmetric difference. A pool whose file differs from its rule by more than 1% of
   rows is labelled U for the differing rows. (The R5 and R15 builders are not in the repository,
   so for those pools this check is the only evidence of their rule.)
2. For `candidates_r15`, recompute the R15 rule from daily rows (drop ≥ 15% over the five
   sessions ending at close D−1 ⇒ entry-session, C0; ending at close D ⇒ signal-session, C2).
   This uses daily data only and reads no minute return.
3. Assign each symbol-day the set of its classes.
4. **Admissible** iff it holds at least one C0 or C1 attribution. A symbol-day in C1 and also in
   C2 (a forensic mover that also gapped ≥ +3%) is admissible, because the C1 pool would have put
   it in the cache anyway, so its presence does not depend on the outcome. Symbol-days that are
   **only** C2 or U are **excluded** from every arm.
4a. **Pool completeness:** for each C0/C1 pool, the share of its members on dev date D that
   appear in that day's `requested.json` (or, where the sidecar is missing, in the parquet), by
   pool and year. A C0/C1 attribution rescues a C2 or U symbol-day only in pool-years with
   completeness ≥ 99%. Otherwise, such symbol-days are excluded from the primary universe and
   appear in S1.
5. Report counts by class, year and pool, and the share of each day's cache that is excluded.

The audit reads only `symbol`, `date` and daily rows. It must be committed and its counts recorded
in Amendment 0 **before** any minute return is read.

### 4.4 Universe each day D (identical for every arm, in this order)

1. **Causal filter** on daily rows strictly before D, all on the unadjusted basis
   (`screener.md`, adjustment rule):
   - `min_price = 1.00`, `max_price = 10.00` on `close_unadj` of D−1;
   - `min_adv_usd = 1_000_000`: 20-session median of `close_unadj × volume_unadj` (raw ×
     raw) over the 20 sessions before D. `volume_unadj` comes from
     `data/research_breakout/daily`, the only cache that stores it (see the volume-basis rule
     below);
   - at least 20 prior daily rows;
   - `exclude_split_days = True`, detected from the factor step `f_{D−1} / f_D` only.
1b. **Open-time pool gate G(D)**, evaluated on daily rows before D plus D's official open only:
   G(D) = [gap ≥ +10%, ADV(R3 def.) ≥ $100k] ∪ [gap ≥ +3%, ADV(R4 def.) ≥ $5M] ∪ [−10% ≤ gap ≤
   −3%, ADV(R5 def.) ≥ $1M] ∪ [the R15 crash rule at close D−1], where gap = open_D / close_{D−1}
   − 1 on one basis, and each bracket uses its source pool's own ADV definition and year span as
   recorded by the reproducibility check (E7). If E7 shows a pool's true rule differs, the
   recorded rule replaces the bracket in Amendment 0, before any return is read.
2. **Admissible minute coverage on D** (§4.3) (`restrict_to_minute_cache = True`).
   admissible = passes step 1 ∧ G(D) ∧ in the cache ∧ C0/C1 provenance (§4.3).
   A symbol-day whose pre-market data was *not requested* (as opposed to requested with no bars)
   is removed from the universe of **every** arm, so that Main and C_a differ only by ranking and
   `top_n`. "Requested, no bars" stays, and is skipped by the screen as a live screen would skip
   it. A symbol-day whose minute presence is admissible but whose pre-market presence is only C2
   or U (after step 4a) is removed from every arm and counted.
2b. **Basis check:** if the pre-market cache's last close before 09:25, × f_{D−1}, differs by
   more than a factor of 1.5 from the D minute-cache first regular open × f_D, or if the
   `feed.md` un-adjustment check fails, the symbol-day is excluded from every arm (both inputs
   are known by 09:30) and counted per year (S4).
3. **Excluded sessions:** half days (13:00 close) and any session `>= holdout_start` unless
   unsealed (§10.6).

The universe is never "every symbol in the cache that day". Steps 1, 1b, 2 and 2b are counted
separately per day and per arm. The count removed at each step, the count of coverage rejects
(`feed.md`) and the count of `no_volume_factor` rejects (below) are logged.

**Volume-basis rule (V1, controller correction).** Alpaca `adjustment="all"` adjusts **volume
for splits only, not for dividends**. So converting share volume with the price factor
`f = close_unadj / close` is wrong whenever a dividend sits between D and the fetch date (for
example BRSP 2024-03-04: f = 1.345, yet adjusted volume equals raw volume, 1,034,853 shares; the
f-rule misses raw volume by more than 2% on 58%, 44% and 35% of symbol-days in 2017, 2021 and
2024). Everywhere in this document:
- the **split-only volume factor** is `s = volume / volume_unadj`, read from
  `data/research_breakout/daily`. Raw shares = adjusted shares ÷ s;
- s is used only where that file's price factor (`close_unadj / close`) agrees with the penny
  daily f within 2% on the same symbol-date. Otherwise, or where the row is missing, the
  symbol-day is rejected as **`no_volume_factor`** in every arm. Rejects are counted per year
  and per arm and reported;
- dollar ADV is raw × raw (`close_unadj × volume_unadj`), with no factor at all;
- share counts used inside D (participation, minute and pre-market bar volume) are put on D's
  traded basis with the single factor s_{D−1} (no split on D, by step 1).
The `research_breakout` cache is used here as a volume-basis input only. Its trials are not in
R16's family (§6.3).

*Why $1–$10 and $1M.* The covered pools all use prior unadjusted close $0.50–$10, so an upper
bound above $10 would claim coverage the cache does not have. Sub-dollar names have a tick that
is ≥ 1% of price and the thinnest quote data. $1M ADV is round 5's standard. It is the band where
the $2,000 order roughly fits the median inside book (LOG §43). None of these numbers was chosen
on an R16 result.

### 4.5 What the universe actually is (stated so it is not over-read)

The R16 universe is exactly G(D) ∩ cache under the §4.4 filter: gap ≥ +10% (any ADV ≥ $1M);
+3% ≤ gap < +10% only with ADV ≥ $5M; −10% ≤ gap ≤ −3%; and post-crash entry sessions.
`hyp_pool` contributes pre-market coverage only. Its minute contribution is measured in
Amendment 0. |gap| < 3% (except post-crash entry sessions) and +3–10% gaps with ADV of $1M–$5M are not in the universe. G3
implements this G(D) exactly and may not redefine it. A different gate is a new round.

Amendment 0 reports |G ∩ cache| / |G| per year and per bracket (daily data only), and reports
the minute and pre-market attributions separately. It is **not** the whole liquid low-priced
market: P1's daily table shows the |gap| < 3% bucket holds about 1.29M of 1.46M symbol-days. The
no-selection control (C_a, C_0) is therefore "no selection **within G(D) ∩ cache**". Every result
is labelled `universe = G(D)-gated` and `conditional_on = U1`.

Two consequences, both binding:
- **Live reproducibility.** The replayed system is "09:25 pre-market scores, 09:30 gate G, `top_n`
  by score inside G" (§5.1). If Gate I passes, G3 builds exactly that G(D) on live data. G is
  fixed here, before any result, so the live gate is not a post-result choice.
- **Composition drift.** Pools cover different years: the R5 pool spans 2016–2026, the gap ≥ +10%
  and +3% pools were built over different spans, and `hyp_pool` starts in 2025-02. Results are
  reported per pool and per year.

### 4.6 Sensitivities (reported, not trials, cannot rescue a failure)

- S1: universe including C2 and U symbol-days. If this looks materially better than the primary,
  that is the measured size of the outcome contamination.
- S2: the three gap brackets of G only (the crash bracket removed).
- S3: per G bracket (gap ≥ +10%, +3..+10% at ADV ≥ $5M, −10..−3%, crash-entry), and the
  `hyp_pool` pre-market attribution reported separately.
- S4: symbol-days failing the un-adjustment check (`feed.md`, tolerance 0.02) or the step-2b
  basis check, reported per arm and per year, together with the `no_volume_factor` rejects (V1).
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
| ranking features | `screen_score` = mean of the cross-sectional percentile ranks of `pm_gap` and `pm_rvol`, descending; ties by symbol. `screen_score` is computed at 09:25 from pre-market bars. The cross-sectional percentile ranks and the `top_n` cut are taken **within the G(D)-admissible set at 09:30**. Candidates are emitted at 09:30. The first decision is at 09:30 + k as before. | see below |
| `top_n` | `10` | see below |
| `exclude_split_days` | `True` | design |
| `holdout_start` | **`2025-08-07`** (§10.6) | confirmed |
| `unseal` | `False` until the single holdout read | spec |
| `restrict_to_minute_cache` | `True`, **on the admissible set of §4.4 step 2** (G(D) ∩ cache ∩ C0/C1) | user decision plus §4.3, §4.4 |

The `Candidates` event for the replay is stamped 09:30, and that is the system the live build must
reproduce.

- `pm_gap = pm_close_last × f_{D−1} / close_unadj_{D−1} − 1`, with `pm_close_last` the close of
  the last pre-market bar with `bar_start + 1 min ≤ 09:25`. Raw-to-raw.
- `pm_rvol = Σ_{04:00 ≤ bar_start < 09:25} (close × volume)` on the traded basis (prices × f_{D−1},
  shares ÷ s_{D−1}; V1, §4.4), divided by the `min_adv_usd` statistic of §4.4 (raw × raw). It is a
  dimensionless ratio. Dollar volume is invariant to a split, so the s-conversion matters only
  through the dividend case V1 describes.
- Both features are invariant to `price × c`, `volume ÷ c` (§99). Symbols with no admissible
  pre-market bars are skipped (`screener.md`). Skips are counted separately as "requested, no
  bars" (no pre-market activity; stays in the universe, skipped by the screen) and "not requested"
  (coverage gap; removed from every arm, §4.4 step 2).
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

**Label (per (k, h); exit-independent).** For the window ending at `ts = T`: `p_in` = open of
the first existing bar with `bar_start ≥ T` within `entry_max_wait_min = 2` (otherwise the window
is dropped, as the entry would expire). `p_out` = open of the first existing bar with
`bar_start ≥ min(T + h, 15:55)`. If none exists before `eod_et`, the last seen close (the
execution EOD rule). These are the engine's fill rules applied to the label.
`y = 1{p_out / p_in − 1 > c_label}`, with **`c_label` a single fixed constant**: the median refit
round trip at $2,000 over the dev no-selection universe (§9.3). It is computed once, before any
model is fit, and recorded in Amendment 0. It is never per-trade (LOG §34). Score =
`P(y = 1)`. No price is ever synthesised.

**CNN determinism.** `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(1)`,
seeds fixed for Python, NumPy and torch.

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
| `entry_threshold` | model: the 0.90 quantile of scores on the inner validation block (the last 20% of the fold's training dates, date-blocked; the same split the CNN uses), from a model fit on the first 80%. The deployed fold model is then refit on 100% of the training dates with the threshold held fixed; the CNN's 100% refit trains for the epoch count chosen by early stopping on the 80/20 split. random: **`0.90`** (the same entry-eligible rate in expectation, with no data used). constant: **`1.0`** (first window) | matched entry rates without looking at test data |
| `exit_mode` | factor: `{"time_stop", "fixed_pct", "atr"}` | spec |
| `hold_min` | `h` | spec §4 |
| `time_stop` | `min(ts + h, 15:55)` in **every** exit arm | the exit factor changes only the bracket |
| `tp_pct`, `sl_pct` (by h) | h=5: `0.0375`, `0.025`; h=15: `0.06375`, `0.0425`; h=30: `0.09`, `0.06` | below |
| `tp_atr`, `sl_atr` (by h) | h=5: `2.10`, `1.40`; h=15: `3.63`, `2.42`; h=30: `5.13`, `3.42` | below |
| `atr_period` | **`5`** bars (design default 14 overridden) | below |
| `atr_floor` | one tick (`$0.01`) on the traded basis. ATR = max(mean TR, $0.01), so an ATR bracket is never invalid where the fixed-% bracket is valid. Entries skipped for an invalid bracket are counted per exit arm. They must be 0 for the ATR arm at G2. | equal entry sets across exit arms (`risk.md:50,140`) |
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
- **Prior project trials are tabulated before the run** and recorded in Amendment 0, from:
  `search_trials.jsonl` line count; `PRIOR_TRIALS` constants and cell counts per round
  (`run_round3/4/6_study.py`, `compare_strategies.py`); the §56 grid (441 cells) unless it is
  shown to be inside `search_trials.jsonl`; rounds 11–14 sweeps; and R15 Part II from the
  `cumulative_trials` field of the R15 result artifacts (`r15_intraday.py:306`).
  `PRIOR_TRIALS_R16 = max(350, tally)`. Binding N = PRIOR_TRIALS_R16 + 54. The tabulation reads
  ledgers and result metadata only, never a return.
- **Tally made for this revision** (ledger lines, constants, cell counts and `*trial*` fields of
  result JSONs only; no return field was read). Counting generously is the conservative
  direction, so no overlap is assumed between rows unless a source says so:

  | # | Source | Trials | Evidence |
  |---|---|---|---|
  | 1 | Rounds 2–3 | 72 | `PREREGISTRATION_R4.md:70`, LOG:814 (R3 itself: 27 prior + its cells, `run_round3_study.py:66`) |
  | 2 | Round 4 eligible cells | 17 | `PREREGISTRATION_R5.md:141` (72 → 89) |
  | 3 | Round 5 escalation + horizon study | 2 | `PREREGISTRATION_R5.md:230` (89 → 90), `PREREGISTRATION_R6.md:99` (→ 91) |
  | 4 | Round 6 added sizes and follow-up | 5 | `PREREGISTRATION_R6.md:190,261` (91 → 94 → 96; `run_round6_study.py:77`) |
  | 5 | Round 6 eligible cells | 30 | `PREREGISTRATION_R6.md:208,286` |
  | 6 | §56 hypothesis grid | 441 | LOG:1653; `hypothesis_grid.json` `n_cells_tested` = 441. Not inside `search_trials.jsonl` (441 > 195; a different round and engine) |
  | 7 | §65 long-side triple-barrier calibration | 3,300 | LOG:2005 ("across 3,300 cells"). It selected the long strategy later retracted in §72, so it is inside the family |
  | 8 | Round 9 automated search | 195 | `search_trials.jsonl`: 195 lines (LOG:2604) |
  | 9 | Round 12 configurations | 84 | LOG:2989 ("2 of 84 configurations admissible") |
  | 10 | R15 Part I own cells | 20 | 287 − 195 − 72 (LOG:3467; `battery_dev.json` `n_trials` = 287) |
  | 11 | R15 Part II H3 | 28 | `r15_intraday_dev.json` `cumulative_trials` = 315 = 287 + 28 |
  | | **Tally (lower bound)** | **4,194** | |

  **Not tabulable from disk, so not in the tally:** rounds 10, 11 (the veto), 13 and 14 (slot
  and target sweeps; `compare_strategies.py:68-71` names them, but no count is recorded), R15
  Part II H1, H2, H4 and H5 (their result files carry no trial field), and the §60–§63 and §70–§71
  side studies. The tally is therefore a **lower bound**. A later tabulation may only raise it,
  and it must be done before the run.
- **`PRIOR_TRIALS_R16 = max(350, 4,194) = 4,194`. Binding Deflated Sharpe trial count
  N = 4,194 + 54 = 4,248.** DSR at N = 54 is also reported. Deflation grows like √(2 ln N): the
  expected-maximum term is about 4.08 at N = 4,248, against 3.46 at the first draft's N = 404.
- Every amendment made after an R16 number has been seen adds 1. Every holdout read is
  recorded (the R16 holdout is read at most once).
- **Family principle.** The trial family is every search on the `research_penny` data lineage
  whose winner could have been deployed as a penny/intraday strategy. The breakout lab
  (`research/breakout`, `data/research_breakout`) is outside it. Sensitivity, reported and never
  gating: DSR with the breakout ledger's line count added to N (the count is recorded in
  Amendment 0). Using `data/research_breakout/daily` as a volume-basis input (V1) does not bring
  its searches into the family: no breakout spec competes with an R16 cell.

### 6.4 Resource estimate

GBDT fits are seconds each. The CNN takes minutes per fit on CPU, so 54 fits is a few hours.
Replays: (54 + 54 + 1,080 + 27 + 27) = 1,242 cell-runs × about 1,870 OOS days. The engine runs
one session per run and is too slow for the full grid, so the grid uses a **vectorised research
simulator** that implements §8 exactly. It must pass a **parity test against the engine**
before any grid result is read.

Parity configurations: every (exit × h × scorer ∈ {model-gbdt, model-cnn, random seed 0,
constant} × arm ∈ {screen, no_selection}) at k = 15, plus each k at (h = 15, time_stop, gbdt,
screen). Parity days: 50 dev days (seed 16), of which at least 10 each contain a gap-through
stop, a partial fill, an EOD-forced exit and an un-adjustment reject, drawn by stratified seeded
sampling. Pass = fills, quantities, costs and daily P&L identical to the cent on every
configuration-day.

The selected cell and its four controls are then replayed **in the engine** on every OOS day,
and the engine numbers are the ones Gate I reads. If the full-period engine replay differs on any
day, the grid is void. The simulator is fixed, re-verified on the parity set plus every
mismatching day, and the whole grid is rerun with §10.5 unchanged. This is logged as an
amendment. It counts as a trial only if the selected cell changes, in which case both selections
are reported. The event-log hashes come from these engine replays. Storage: under 5 GB
under `data/research_intraday/`.

### 6.5 Execution order

1. G1 sign-off on this document.
2. Provenance audit (§4.3, including pool reproducibility and completeness), G(D) coverage
   |G ∩ cache| / |G| per year and bracket, the `no_volume_factor` and basis-check counts, the
   prior-trial tabulation (§6.3), the SPY span (§7), then Amendment 0: class counts, `c_label`,
   cost-refit coefficients, coverage per year. No return is read.
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
- **C_c, screen alone.** `scorer = constant`: each screened name becomes entry-eligible at its
  first window (09:30 + k), then the same exit. Main − C_c = the value of the model given the
  screen.
- **C_0, nothing.** No selection, constant scorer. It is the anchor for H2 and P1.
- **Entry order in C_c and C_0 (N4).** C_c/C_0 enter in sorted-symbol order as the 5 slots free
  up. Ties in any arm break by symbol. So C_0 enters the alphabetically first 5 admissible names
  at 09:30 + k and later names only as positions close. It is not "buying every covered name at
  09:30 + k". The rule is deterministic and independent of outcomes.
- **Random-control behaviour check (N5).** C_b OOF AUC and IC against the label are reported per
  seed and must lie within the shuffled-label null band; a seed outside it voids the run.
- **SPY.** The project rule requires a SPY comparison. Report the same-day SPY open→close return
  (it is reported only and is not a control the gate reads, since the account is flat overnight),
  and each arm's daily correlation and alpha against it. SPY is not in `research_penny/daily`
  (that cache's universe drops ARCA and names containing 'ETF/Trust', `etl/universe.py:44,112`).
  Use `C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/cache/SPY.parquet`, which is already
  on disk. Its date span and columns are recorded in Amendment 0 from the schema and the min/max
  date only. SPY open→close is basis-invariant within a day. Dev days it does not cover are listed.
  Nothing is fetched.
  - *Recorded for this revision (schema and index min/max only):* columns `Open`, `High`, `Low`,
    `Close`, `Volume` (all float64) with a pandas `timestamp[ns]` index; 2,263 rows; index span
    **2016-01-04 .. 2024-12-30**. So SPY covers the dev period through 2024-12-30 only. Dev
    sessions 2025-01-02 .. 2025-08-06 and the **whole holdout** (from 2025-08-07) are not covered.
    For those days the SPY comparison is reported as missing. Nothing is fetched. The SPY
    comparison is reported only and does not gate, so the gap changes no decision.
  - *Controller addition (R8, symbol/date columns only):* `data/research_breakout/daily/bars_2025.parquet`
    holds SPY for 2025-01-02 .. 2025-08-06 (rows before the holdout). Those 2025 dev sessions use
    it (open→close, basis-invariant within a day), so SPY covers the whole dev period. No fetch.

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
participation = fill qty / the 20-session median share volume before D, every session converted
to D's traded basis with a single factor (`risk.md:103-107`). Under V1 (§4.4) that factor is the
split-only volume factor s_{D−1}, not f_{D−1}: shares_t on D's basis = volume_t / s_{D−1}, with
`volume` the split-adjusted daily volume and s_{D−1} = volume_{D−1} / volume_unadj_{D−1}, both
read from `data/research_breakout/daily`. A symbol-day without a valid s is a `no_volume_factor` reject.
Every price level is unadjusted (`execution.md`). Minute-bar volume used by
`bar_participation_cap` is converted the same way (÷ s_{D−1}). Entry pays on the ask side and exit on the bid side. Costs are
charged once per leg, separately from the bar reference price, and auditable per fill.

### 9.2 Inputs

Half-spread and inside notional come from a cached quote where one exists for that symbol-day
and minute window. Elsewhere they come from the refit model (§9.3). Half-spread is floored at half
a tick: `0.5 × $0.01 / price × 10⁴` bps.

### 9.3 The refit (on this round's population, before any strategy result)

- **Fit set:** dev-period symbol-days in the R16 **no-selection universe** (§4.4) that have cached
  quote summaries at 09:34/09:35 and/or 15:55. Coverage is reported.
- **Models (OLS, specification fixed now):** `log(half_spread_bps)`, `log(ask_notional)` and
  `log(bid_notional)` each regressed on `log(close_unadj_{D−1})`, `log(adv20_usd)` (raw × raw, V1),
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
- **Year dummies:** holdout legs use the 2025 dummy (the last dev year). There is no refit on
  holdout quotes. `c_label` and the refit use all dev years, including later test folds. This is
  cost data, not returns, and is fixed in Amendment 0.
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
  rejects, the un-adjustment failure rate, the basis-check and `no_volume_factor` rejects, and
  how often `top_n` binds;
- realised entry-eligible window rate, per arm and per cell;
- C_b's OOF AUC and IC per seed against the shuffled-label null band (§7);
- count and age distribution of EOD-forced fills, per arm;
- invalid-bracket entry skips, per exit arm (§5.5 `atr_floor`);
- per year, per pool (S3), and gross alongside net;
- the SPY comparison (§7).

### 10.4 Multiple testing

- For each of the 54 main cells, a one-sided day-clustered bootstrap p-value for mean daily net
  return > 0. **Holm** step-down over the 54 at family-wise α = 0.05.
- Holm uses arithmetic-mean p-values while the account gate uses the geometric mean; both must
  pass.
- **Deflated Sharpe** (Bailey & López de Prado 2014) at **N = 4,248** (§6.3). `Var(SR_n)` = the variance
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
  minute features and 10:00 entry prices in 2024–2026. Also read in the holdout window: §60
  (`hypothesis_premarket.py`), pre-market dollar volume / ADV against open→close, 2025-02 to
  2026-08, which is R16's `pm_rvol` concept (result negative); and the forensic / top-30 mover
  studies (2026 outcomes). `pm_rvol` was chosen with that result known. None asked R16's question
  (a minutes-ahead model with brackets), so it is sealed **for this question**, as the spec states.
- **Read exactly once**, after the critic's G2 sign-off on the development result, and only if
  some cell passed development. The holdout is not unsealed until the user has confirmed U1.
  Before the unseal, the quote-size unit is made date-dependent: Alpaca reports CTA/UTP sizes in
  round lots until 2025-11-02 and in shares from 2025-11-03 (procedural; it changes no development
  number, since every development date is in round lots). Read: the selected cell and its controls C_a, C_b (20 seeds), C_c
  and C_0. The model is retrained once on **all** dev dates with the frozen configuration. The
  threshold is the 0.90 quantile on the inner validation block of the dev dates (their last 20%,
  from a fit on the first 80%), by the same rule as in each fold (§5.5). Both are frozen with `model_hash` before
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

- R16-specific nulls, implemented in `research/intraday/research/r16_nulls.py` (because
  `eval/nulls.py::run_all` is a veto-set battery with a fixed 09:35 tripwire and does not apply):
  (a) labels permuted within day, model retrained, OOF AUC and grid account must fall to the null;
  (b) labels taken from the day 20 sessions later, same symbol slot; (c) a per-window tripwire:
  max bar `ts` in the window ≤ the window's `ts`, and no input after the decision `ts`; (d)
  constant return through the simulator gives zero dispersion across cells. They run on one fold
  per (k, h, model). A null "failing" by improving means negative skill, not leakage (LOG §35),
  and is reported as such.
- **Martingale-calibrated IC/AUC null** (memory: rank IC against a forward return is not centred
  on zero): the classifier's out-of-fold AUC is compared with its AUC on synthetic minute paths
  that keep each symbol-day's realised minute σ but have zero drift, so sigma-correlated scores do
  not read as skill.
- Joint scale-invariance test on every feature (`price × 750`, `volume ÷ 750`).
- Assertions: no input after the decision `ts`; no daily row ≥ D in a filter except the factor
  ratio for the split flag, f_D for step 2b and f_D, s_D for the feed un-adjustment of D's own bars (they rebuild the tape as it traded; no other field of D is read), and D's official open in G(D) (§4.4 step 1b); no field of D other than
  the open is read by G; no session ≥ `holdout_start` without `unseal`.
- Determinism: the engine replay of 5 fixed days run twice gives identical event-log hashes.

---

## 11. Gate I decision rule

### 11.1 Development conditions (all required, for at least one main cell)

1. **Account:** compounded net geometric-mean daily return has a day-clustered 95% CI with lower
   bound > 0.
2. **Controls:** paired daily-difference CI lower bound > 0 against **each** of C_a, C_b (the
   20-seed mean daily return) and C_c. Also, the cell's compounded return exceeds the **95th
   percentile** of the 20 C_b seeds' compounded returns.
3. **Multiple testing:** Holm-adjusted p < 0.05 over the 54 cells, **and** DSR ≥ 0.95 at N = 4,248.
4. **Robustness:** the compounded return point estimate stays > 0 after removing the top 1% of
   trades by P&L. The median trade is reported, and a negative median with a positive mean is
   flagged in the G2 report.
5. **Power floor:** ≥ 300 OOS trades on ≥ 150 distinct days.
6. **Validity:** §10.7 passes, the parity test passes, and the engine replay of the cell is
   identical to the cent, on every OOS day, to the vectorised result. If it differs on any day,
   §6.4's void-and-rerun rule applies.

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
  the provenance-audit sha256, git commit, `universe = G(D)-gated`, `conditional_on = U1`,
  `period = dev | holdout`.
- Every run asserts `git status --porcelain` is empty for tracked code under `research/`.
  Otherwise it refuses to run. The SHA and the clean flag are MLflow tags.
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
- No live feed, no broker connection, no paper trading, no live build of the open-time gate. If
  Gate I passes (and U1 is confirmed), G3 builds exactly the G(D) of §4.4 step 1b and may not
  redefine it.
- No per-trade-mean verdicts. The account decides.

---

## 14. Open risks

1. **The universe is the open-time gate G(D), not the whole market** (§4.4 step 1b, §4.5). G is
   written down now, so a live system can rebuild it at 09:30 exactly. Whether the gated variant
   is still the user's design is U1 (box at the top). If the user rejects U1, R16 is void as a
   test of that design, and only a download round can test it.
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
6. **The prior trial tally (4,194) is a lower bound** (§6.3). Rounds 10, 11, 13, 14, most of R15
   Part II and several side studies have no count on disk. A later tabulation may only raise N,
   and only before the run.
10. **The volume basis depends on a second cache** (V1, §4.4). Symbol-days where
    `research_breakout/daily` has no row, or disagrees with the penny factor by more than 2%, are
    rejected. If that reject share is large in a year, it is reported per year and per arm. It is
    not repaired.
11. **SPY covers dev only through 2024-12-30** (§7). The SPY comparison is missing for 2025 dev
    days and for the whole holdout. It does not gate.
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

*(none; Amendment 0 records the provenance-audit counts, pool reproducibility and completeness,
G(D) coverage per year and bracket, the basis-check and `no_volume_factor` counts, the prior-trial
tabulation, the breakout ledger line count, the SPY span, `c_label` and the cost-refit
coefficients, all before any return is read)*

### Amendment 0 (2026-10-26)

**In plain English.** Before the model sees a single minute of price movement, this amendment
writes down two kinds of thing. First, the facts the counting audit found: how many stock-days
are usable, how much of each gap bracket the cache covers, what trading costs look like, and the
cost hurdle a trade must clear to count as a win. Second, about thirty small decisions that the
pre-registration left open and the code had to settle (for example, exactly how "random" draws
are seeded, or how a check that should find nothing is scored so that it really finds nothing).
None of these decisions was made after seeing a result, because no result exists yet. Writing
them down now means they cannot be adjusted later to make the strategy look better. Everything
here stays conditional on the user confirming the design (U1).

**Status and trial effect.** This amendment is recorded after G1 sign-off (cff3c76) and before
any minute return, label or R16 metric has been read. Under §6.3, only an amendment made after an
R16 number has been seen adds a trial, so it adds **0 trials**. N stays 4,248 unless §A0.1's
prior-trial check raises the tally (it does not). §A0.1 is filled from the audit's
`amendment0_inputs.json`. The frozen text above is unchanged.
Where a ruling below changes a frozen sentence, the ruling governs from this amendment on.

**Conditional on U1.** U1 is still pending. Every rule in the header box (all results conditional
on U1, no Gate-I evidence and no unseal before confirmation, void as a test of the user's design
if U1 is rejected) applies unchanged. Nothing in this amendment depends on the answer to U1.

#### A0.1 Audit inputs (from the audit run; daily-level outputs only, no return was read)

| Field | Value | Source / check |
|---|---|---|
| audit git sha (code used by smoke and full) | `0457541` (output dir `data/research_intraday/r16/04575419fd76/audit`) | git HEAD of the `audit --refit` run. An earlier audit at 4155601 was superseded by the P3-10 and P3-11 code changes and is not used |
| audit output sha256 | `2e80cc0e89ccec032762c5316ff8f466882c21cda4efca6d4bcba14d83f499ab` | sha256 of the bytes of `data/research_intraday/r16/amendment0_inputs.json` (53,616 bytes, byte-identical to the copy under `04575419fd76/audit/`). `run_r16` preflight refuses smoke and full unless this hex appears in the prereg at HEAD under this heading |
| audit scope | counts, cost refit and c_label | as required (§A0.5) |
| dev sessions | 2,358, 2016-02-17 .. 2025-08-06 | `dev_last` < 2025-08-07; `day_errors` empty |
| `c_label` | 0.005537 (55.4 bps) | §5.4, §9.3 |
| refit digest | `b7da07e5e958265b76c1d67b92a882488a8892f8bae5c6085b529ffa45228c6c` | the §12 cost-refit coefficient hash; years 2016–2025, base year 2016; coefficients below |
| refit rows | fit 94,988; books 34,051; CostInputs 34,051 | no-selection symbol-days 34,051, of which 32,014 have a quote summary (share 0.940) |
| infeasibility median (final admissible count per day) | 16 (after step 1 only: 484) | §11.4 as amended (§A0.4 item 10): floor 3 |
| C_0 trade bound | 34,051 symbol-days on 2,305 days | §11.4 floor: 300 trades on 150 days |
| §11.4 verdict | **feasible** (`infeasible` = False) | `r16_amendment0.infeasible` on the file |
| brackets dropped (P3-9/P3-11) | `crash` in 2020, 2021, 2022, 2023, 2024, 2025 (R15 pre-market completeness < 0.99) | to S1 |
| `top_n` binding share | 0.329 of days | §10.3 |
| half days removed | 19: 2016-11-25, 2017-07-03, 2017-11-24, 2018-07-03, 2018-11-23, 2018-12-24, 2019-07-03, 2019-11-29, 2019-12-24, 2020-11-27, 2020-12-24, 2021-11-26, 2022-11-25, 2023-07-03, 2023-11-24, 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03 | §A0.4 item 11 |
| SPY span | cache 2016-01-04 .. 2024-12-30 (2,263 rows); 2025 from `research_breakout/daily` 2025-01-02 .. 2025-08-06 (148 rows) | §7, R8 |
| prior-trial tabulation | `search_trials.jsonl` 195 lines; `hypothesis_grid` 441; `battery_dev` 287; `r15_intraday_dev` 315. All four match the prereg | §6.3: `PRIOR_TRIALS_R16` = 4,194, binding **N = 4,248**, unchanged |
| breakout ledger lines | 103 | §6.3 non-gating sensitivity: N = 4,248 + 103 = 4,351 |
| provenance sha256 | combined `a0e34d2c992499facbdf96fe88fd3560628622ecb31cab221deb10571ce5d5a2`; minute `27f7a28da5dd15bf53319f433b240930664ba97fcd89a3559d6076cff84da512`; pre-market `9958a7f9398598ebc05ffb9dbc7dd7dce5009d5fd9631110f525f38a03d694e4` | §12 tag |

**Provenance classes (§4.3 step 5).** A symbol-day holding several classes is counted in each.

| Cache | symbol-days | admissible | C0 | C1 | C2 | U |
|---|---|---|---|---|---|---|
| minute | 160,599 | 136,817 | 50,368 | 100,723 | 71,390 | 566 |
| premarket | 183,617 | 183,312 | 140,790 | 50,259 | 10,317 | 57 |

| Cache: admissible / n | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| minute | 4,025 / 4,025 | 2,756 / 2,756 | 3,297 / 3,297 | 3,713 / 3,713 | 28,078 / 32,098 | 15,357 / 18,027 | 22,545 / 27,524 | 14,886 / 18,208 | 20,457 / 25,161 | 21,703 / 25,790 |
| premarket | 2,048 / 2,048 | 1,545 / 1,545 | 1,734 / 1,734 | 1,838 / 1,838 | 10,846 / 10,982 | 5,862 / 5,875 | 5,631 / 5,672 | 4,446 / 4,493 | 6,275 / 6,340 | 143,087 / 143,090 |

**Pool reproducibility (§4.3 step 1a, E7; pass iff frac_diff ≤ 0.01).** The same check serves both caches.

| Pool | in minute cache | in pre-market cache | span | n_file | n_rule | symmetric diff | frac_diff | 1% rule |
|---|---|---|---|---|---|---|---|---|
| `candidates` | 16,405 | 16,107 | 2016-02-17 .. 2025-08-06 | 16,667 | 16,667 | 0 | 0.000 | reproduces |
| `candidates_r15` | 115,756 | 17,236 | 2020-01-02 .. 2025-08-06 | 72,255 | 152,024 | 80,907 | 1.120 | **does not reproduce** |
| `candidates_r4` | 35,586 | 35,586 | 2016-02-17 .. 2025-08-06 | 35,747 | 35,747 | 0 | 0.000 | reproduces |
| `candidates_rev` | 53,533 | 3,367 | 2016-02-17 .. 2025-08-06 | 54,062 | 55,368 | 1,306 | 0.024 | **does not reproduce** |
| `candidates_rev_probe` | 5,023 | 368 | 2016-02-26 .. 2025-07-23 | 5,034 | 54,658 | 49,624 | 9.858 | **does not reproduce** |
| `hyp_pool` | 10,541 | 137,901 | 2025-02-03 .. 2025-08-06 | 138,995 | 238,826 | 99,831 | 0.718 | **does not reproduce** |

**Pool completeness (§4.3 step 4a; ≥ 0.99 lets the pool rescue a C2/U symbol-day).** Blank = no pool rows that year.

| Pool : cache | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| `candidates` : minute | 0.991 | 0.963 | 0.972 | 0.988 | 0.995 | 0.993 | 0.997 | 0.993 | 0.995 | 0.995 |
| `candidates` : premarket | 0.958 | 0.944 | 0.948 | 0.958 | 0.972 | 0.986 | 0.970 | 0.975 | 0.978 | 0.993 |
| `candidates_r15` : minute |  |  |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `candidates_r15` : premarket |  |  |  |  | 0.122 | 0.064 | 0.047 | 0.037 | 0.048 | 0.543 |
| `candidates_r4` : minute | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `candidates_r4` : premarket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `candidates_rev` : minute | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `candidates_rev` : premarket | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.389 |
| `candidates_rev_probe` : minute | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `candidates_rev_probe` : premarket | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.388 |
| `hyp_pool` : minute |  |  |  |  |  |  |  |  |  | 0.076 |
| `hyp_pool` : premarket |  |  |  |  |  |  |  |  |  | 1.000 |

**G(D) coverage per year and bracket (§4.5): share |G ∩ cache| / |G| (counts in brackets).** `gap10` = gap ≥ +10%; `gap3` = +3% to +10% with ADV ≥ $5M; `rev` = −10% to −3%; `crash` = post-crash entry session (R15 pool, 2020 on). — = no gate members.

| Year | gap10 | gap3 | rev | crash |
|---|---|---|---|---|
| 2016 | 0.989 (188 / 190) | 0.998 (1,584 / 1,587) | 0.999 (1,695 / 1,696) | — |
| 2017 | 0.980 (195 / 199) | 0.996 (847 / 850) | 0.998 (834 / 836) | — |
| 2018 | 0.991 (222 / 224) | 0.993 (915 / 921) | 0.996 (1,092 / 1,096) | — |
| 2019 | 0.989 (262 / 265) | 0.998 (1,048 / 1,050) | 0.999 (1,392 / 1,393) | — |
| 2020 | 0.999 (1,613 / 1,614) | 1.000 (8,297 / 8,300) | 1.000 (8,432 / 8,434) | 0.669 (7,822 / 11,692) |
| 2021 | 0.979 (911 / 931) | 0.989 (4,460 / 4,510) | 0.991 (4,861 / 4,907) | 0.479 (3,841 / 8,016) |
| 2022 | 0.994 (470 / 473) | 0.996 (4,160 / 4,175) | 0.993 (7,273 / 7,322) | 0.559 (7,187 / 12,846) |
| 2023 | 0.989 (450 / 455) | 0.997 (2,679 / 2,687) | 0.994 (2,862 / 2,879) | 0.551 (3,807 / 6,907) |
| 2024 | 0.997 (774 / 776) | 0.998 (3,543 / 3,551) | 0.995 (4,048 / 4,068) | 0.582 (4,940 / 8,493) |
| 2025 | 0.994 (719 / 723) | 0.997 (3,831 / 3,843) | 0.998 (5,451 / 5,464) | 0.654 (5,156 / 7,889) |

Final no-selection counts per bracket are not disjoint: after the P3-11 drop, every remaining `crash` symbol-day also qualifies for another bracket (it counts in both).

| Year | admissible (before the crash drop): gap10 / gap3 / rev / crash | final no-selection: gap10 / gap3 / rev / crash |
|---|---|---|
| 2016 | 188 / 1,584 / 1,695 / 0 | 180 / 1,482 / 0 / 0 |
| 2017 | 195 / 847 / 834 / 0 | 190 / 820 / 0 / 0 |
| 2018 | 222 / 915 / 1,092 / 0 | 219 / 876 / 0 / 0 |
| 2019 | 262 / 1,048 / 1,392 / 0 | 253 / 1,021 / 0 / 0 |
| 2020 | 1,613 / 8,297 / 8,431 / 7,822 | 1,472 / 8,028 / 0 / 1,859 |
| 2021 | 911 / 4,459 / 4,858 / 3,841 | 856 / 4,262 / 0 / 734 |
| 2022 | 470 / 4,160 / 7,271 / 7,187 | 438 / 4,041 / 0 / 864 |
| 2023 | 450 / 2,679 / 2,858 / 3,807 | 419 / 2,614 / 0 / 320 |
| 2024 | 774 / 3,543 / 4,045 / 4,940 | 736 / 3,467 / 0 / 576 |
| 2025 | 719 / 3,828 / 5,448 / 5,156 | 684 / 3,731 / 1,707 / 1,002 |

**Universe steps per year (§4.4, symbol-days).** `step 1`, `admissible`, `final` and `screen` are the counts left after a stage; every other column is the count removed at that step. `step 1` = passes §4.4 step 1. `admissible` = step 1 minus short history, `zero_adj_volume`, `no_volume_factor`, gate, the crash-bracket drop (P3-9/P3-11), cache and provenance (I2). `final` = the no-selection universe, i.e. `admissible` minus the pre-market, basis and feed steps. `screen` = after `top_n`.

| Year | step 1 | short history (< 21 rows) | `zero_adj_volume` | `no_volume_factor` | gate | crash bracket dropped | cache | provenance | admissible | pm not requested | pm C2/U | basis check | feed reject | final | screen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 64,593 | 3 | 210 | 2,228 | 58,955 | 0 | 4 | 0 | 3,193 | 1,608 | 0 | 0 | 14 | 1,571 | 1,091 |
| 2017 | 69,284 | 12 | 392 | 2,429 | 64,710 | 0 | 8 | 0 | 1,733 | 806 | 0 | 0 | 5 | 922 | 831 |
| 2018 | 71,755 | 12 | 409 | 2,263 | 66,987 | 0 | 10 | 0 | 2,074 | 1,064 | 0 | 0 | 7 | 1,003 | 838 |
| 2019 | 83,646 | 9 | 501 | 2,325 | 78,285 | 0 | 4 | 0 | 2,522 | 1,350 | 0 | 0 | 21 | 1,151 | 961 |
| 2020 | 121,837 | 21 | 425 | 4,188 | 92,757 | 7,648 | 6 | 1 | 16,791 | 8,191 | 67 | 0 | 74 | 8,459 | 2,145 |
| 2021 | 130,364 | 92 | 1,505 | 4,611 | 108,511 | 6,302 | 89 | 4 | 9,250 | 4,664 | 3 | 1 | 44 | 4,538 | 2,023 |
| 2022 | 134,112 | 48 | 414 | 3,898 | 107,969 | 10,350 | 48 | 2 | 11,383 | 7,116 | 12 | 0 | 41 | 4,214 | 1,713 |
| 2023 | 127,379 | 22 | 165 | 3,903 | 111,827 | 5,865 | 23 | 3 | 5,571 | 2,761 | 11 | 0 | 15 | 2,784 | 1,736 |
| 2024 | 133,409 | 22 | 19 | 4,478 | 114,115 | 7,118 | 25 | 2 | 7,630 | 3,876 | 9 | 0 | 19 | 3,726 | 2,017 |
| 2025 | 97,014 | 39 | 36 | 3,484 | 78,300 | 5,884 | 22 | 5 | 9,244 | 3,544 | 0 | 0 | 17 | 5,683 | 1,456 |
| total | 1,033,393 | 280 | 4,076 | 33,807 | 882,416 | 43,167 | 239 | 17 | 69,391 | 34,980 | 102 | 1 | 257 | 34,051 | 14,811 |

**Daily medians per year.**

| Median per day | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| step 1 | 287 | 278 | 286 | 338.5 | 489.5 | 514 | 531 | 514 | 535 | 659 |
| admissible | 11 | 6 | 6 | 7 | 32 | 24 | 28 | 17 | 22 | 36 |
| final no-selection | 5 | 3 | 3 | 4 | 13.5 | 11 | 8 | 8 | 11 | 25 |
| screen | 4 | 3 | 3 | 3 | 10 | 10 | 8 | 8 | 10 | 10 |

**Cost refit (§9.3; OLS in logs; base year 2016; `window_close` = 1 for the 15:55 window).**

| Regressor | log half-spread (bps) | log ask notional | log bid notional |
|---|---|---|---|
| `const` | +8.8694 | -2.6772 | -2.4512 |
| `log_adv20` | -0.2882 | +0.6437 | +0.6351 |
| `log_close_prev` | -0.3041 | -0.7508 | -0.7679 |
| `log_sigma` | +0.2813 | -0.7471 | -0.7270 |
| `window_close` | -0.5355 | +1.5073 | +1.4640 |
| `year_2017` | +0.0638 | -0.0091 | +0.0009 |
| `year_2018` | +0.0479 | -0.2776 | -0.2507 |
| `year_2019` | +0.0151 | -0.3149 | -0.2612 |
| `year_2020` | +0.0872 | -0.5900 | -0.5767 |
| `year_2021` | +0.1243 | -0.5682 | -0.5347 |
| `year_2022` | +0.0807 | -0.5443 | -0.5307 |
| `year_2023` | +0.0292 | -0.5763 | -0.5430 |
| `year_2024` | +0.1152 | -0.6221 | -0.5735 |
| `year_2025` | +0.3249 | -0.8528 | -0.8360 |
| R² | 0.496 | 0.565 | 0.552 |

**What the counts say, in plain English.** The development period has 2,358 sessions, and about a
million stock-days pass the price and liquidity filter, but only 34,051 stock-days on 2,305
sessions survive into the test universe, because the gate keeps only names that gapped at the
open and the cache holds only names an earlier study saved. The primary test is therefore mostly
gap-up stocks: post-crash names enter only when they also gapped (the post-crash bracket itself
moves to the S1 side study), and gap-down names appear only in 2025, the one year with their
pre-market data. A typical day has 3 to 5 tradable names in 2016–2019 and 8 to 25 from 2020 on.
That clears the §11.4 floor (median admissible 16 ≥ 3; at most 34,051 C_0 trades on 2,305 days,
against 300 on 150), so the round is **feasible on cached data**. The audit's three open findings
(O1–O3) are closed by controller rulings P3-8 to P3-11 (§A0.3).

#### A0.2 Plan ambiguities (plan `2026-10-26-intraday-p3-model-r16.md`, items 1–15)

Each is pinned by a test in the model worktree. "Why" applies to the whole row.

| # | Ruling | Why | Fills / changes |
|---|---|---|---|
| 1 | Training subsample: per fit, the `GBDT_TRAIN_CAP` = 1,000,000 / `CNN_TRAIN_CAP` = 200,000 training windows with the smallest `subsample_key(symbol, ts)` (splitmix64, seed 16); `CNN_VAL_CAP` = 50,000 early-stopping windows by the same rule. Inference and thresholds use every window. | Compute on a 16 GB host. The key ignores outcomes, so the sample cannot select on the label; the samples are nested across folds. | fills §5.4 training population, §6.4 |
| 2 | Training, validation, threshold and AUC use only windows with `ts` < 15:30 ET (`last_entry_et`). | A later window can never trade, so it would only dilute the label. | fills §5.4, §5.5 |
| 3 | `ret_n = log(close[-1] / open[-n])`; `vol_rel` uses the window median volume, else the mean if the median is 0, else 0.0. | Keeps the k-bar return inside the window and defines the zero-volume case. | fills §5.4 features |
| 4 | `c_label` round trip: entry leg = open-window refit prediction (ask side), exit leg = close-window prediction (bid side), at $2,000, as in `etl/quotes.py::round_trip_cost_bps`; median over dev no-selection symbol-days. | §9.3 names the median but not the legs. | fills §5.4, §9.3 |
| 5 | Time-of-day interpolation is linear in level (bps, $ notional), not in log, in ET clock minutes between 09:35 and 15:55. | §9.3 says "linear interpolation in clock time" without the scale. | fills §9.3 |
| 6 | C_b, C_c and C_0 do not depend on the model factor: 27 × 20 = 540 unique C_b runs serve all 54 main cells, reported as the 1,080 pairings. | The gbdt and cnn twins share every control input, so their controls are the same runs. | fills §6.2, §6.4 |
| 7 | Null (a): permuted-label OOF AUC must fall to the null; null (b): the label of the same symbol and ET minute on the dev session 20 sessions later, dropped if absent; a session ≥ `holdout_start` is never read. The AUC statistic and the account half are **superseded by P3-2 and P3-4** (§A0.3). | §10.7 says "fall to the null" with no test. | fills §10.7 (a), (b) |
| 8 | C_b null band (§7 N5): per seed, \|z\| ≤ Φ⁻¹(1 − 0.01/80) = 3.66 for AUC and for IC (Spearman of score against y); z from a day-clustered bootstrap SE (B = 200); Bonferroni over 20 seeds × 2 statistics. One-sided p (§10.4) = (1 + #{b: m*_b − m ≥ m}) / (B + 1), centred bootstrap. Scope: §A0.4 item 9. | §7 N5 names a band but not its width. | fills §7 N5, §10.4 |
| 9 | DSR: `Var(SR_n)` across the 54 main cells is on the annualised scale and is passed to `eval/stats.py::deflated_sharpe_ratio` as var / 252. | That function works per period. | fills §10.4 |
| 10 | Ties after §10.5's list: `fixed_pct` over `atr`, then smaller h. The C_b 95th percentile is `np.percentile(…, 95)` (linear). | §10.5 stops before breaking every tie. | fills §10.5, §11.1.2 |
| 11 | Gross account = the net run's fills with `cost_bps` = 0 over the same E_{d−1}; trades are identical. | Gross and net must differ only by cost. | fills §10.3 |
| 12 | Smoke runs are pipeline tests only: experiment `intraday-smoke`, their own folds, 60 dev sessions, no trial-ledger line, never cited as R16 evidence. | A smoke run must not become a trial. | fills §12 |
| 13 | The engine full-period replay (§6.4, §11.1.6) runs only if some cell meets §11.1.1–5 (smoke: SMOKE_CELL always). If none does, H1 fails on development and the parity test is the validity evidence. The halt sensitivity and the engine-only parts of §10.7 run on that replay. | Nothing would be validated by replaying a failed grid. | fills §6.4, §11.1.6; see P3-4 |
| 14 | S1, S2 and cost × 1.5 are re-simulated grids. S3 attributes the primary trades to G brackets with no re-simulation. | S3 is an attribution, not a new universe. | fills §4.6, §9.3 |
| 15 | The pool rules of G(D) are frozen constants (`POOL_RULES`). If §A0.1's reproducibility shows a different true rule (§4.4 1b, E7), the change is a new reviewed code task recorded in a further amendment before any return, not a config edit. | Keeps G(D) out of reach of configuration. | fills §4.4 1b |

#### A0.3 Controller rulings P3-1 .. P3-11

- **P3-1 (process).** The audit runs once as `audit --refit` (a superset of the counts-only
  audit), and only that run's sha goes in §A0.1. *Why:* the counts-only audit leaves `c_label` and
  the refit null, and the costs and windows stages then refuse.
- **P3-2 (controller ruling, pending user review). Nulls (a) and (b) comparator.** The statistic
  is the within-stratum AUC: strata are (session date, symbol, 30-min ET bucket of the decision
  time); only strata holding both classes count; AUC = ΣU / ΣP, the pair-weighted mean of the
  stratum AUCs (ties count ½). SE comes from a day-clustered bootstrap (B = 200, seed 16, each day a
  cluster). z = (AUC − 0.5)/SE. The null passes iff |z| ≤ 3. z < −3 is reported as negative skill
  and does not pass. Strata, pair and day counts are recorded; the pooled AUC is report-only.
  Fills §10.7 (a), (b) (the "falls to the null" test) and replaces ambiguity 7's pooled
  statistic.
  *Why, in plain words:* the pooled AUC compares every window with every other window across all
  days, symbols and times. A model can then score above 0.5 on null labels with no leakage at all.
  Under null (a), each day keeps its own share of winners, and busy days have both more
  big-moving windows and more windows that clear `c_label`; a model that learns "volatile
  window ⇒ volatile day" ranks windows of high-share days above those of low-share days, so the
  pooled AUC beats 0.5. Under null (b), the label comes from the same symbol and minute 20
  sessions later, and both a symbol's volatility and the early-session volatility persist, so a
  model that recognises "volatile symbol" or "first half hour" predicts the shifted label honestly.
  All of these use only past and present information, so they are not leakage, but a pooled AUC
  test would wrongly void the run. Comparing windows only inside the same session, symbol and
  half hour removes the day base rate, symbol volatility and coarse time of day, so what is left
  is what a null should show: nothing. *If wrong:* the nulls are too lenient. If the user changes
  this before any null result is read, the change is a further amendment and adds no trial;
  after a null result is seen, it adds one (§6.3).
- **P3-3. Null (b) labels** are read from the full windows-stage index at session d+20 (same
  symbol, same ET minute) and require `valid_h`. Drops are counted under `beyond_last_session`,
  `no_window_at_target` and `target_label_invalid`. *Why:* reading the training subsample would
  drop most targets. Fills §10.7 (b).
- **P3-4. Null (a)'s account half and the §5.6 halt sensitivity are run, not dropped.** *Why:*
  §10.7 (a) and §5.6 require them. Null (a) account: the last fold, one null model per (k, h,
  model) refit on 100% of the training dates, threshold = the 0.90 quantile of its own scores on
  the training rows of the inner-validation block (in-sample for the null model), scored on the
  last fold's OOS days. For every main cell, the paired daily-return difference against that
  cell's C_b seed mean on the same days, B = `BOOT_B`, seed 16, iid days, passes iff its 95% CI
  contains 0. In full mode a failure voids the run (exit 5). Halt sensitivity: the replayed main
  cell (smoke: SMOKE_CELL) is replayed again with `infer_halts_from_gaps = True`, `gap_halt_min =
  5`, report-only, never voiding. It therefore exists in full mode only when ambiguity 13's replay
  runs. Fills §10.7 (a), §5.6.
- **P3-5. Short history.** Names with fewer than **21** daily rows strictly before D in the
  45-calendar-day lookback are excluded right after step 1 and counted as `short_history`
  (count in §A0.1). A missing CostInputs for any primary or S1 universe symbol-day, either arm,
  fails the costs stage. *Why:* the 20-session statistics (ADV, σ_20d, and 20 returns for the
  cost regressors) need 21 rows. §4.4 step 1 says "at least 20", which leaves σ_20d undefined.
  Changes §4.4 step 1 (20 → 21) and adds a count to §4.4. A name whose 20 prior closes are equal
  (σ_20d = 0) fails the costs stage by design, and smoke will show whether that happens.
- **P3-6. §12 MLflow layout.** One run per (cell or control, fold) plus an aggregate. Params: the
  flattened §5 keys (`risk.*`, `exec.*`), `config_hash`, `model_hash` (or `random:<seed>` /
  `constant`), fold, threshold. `event_log_hash` = sha256 over every parity config-day's engine log
  hash. `git_clean` = HEAD equals the run's sha and the tree is clean. Replayed cells are tagged
  `engine_log_hashes_sha`. Fills §12.
- **P3-8 (closes O1, pool reproducibility).** Four pools fail the 1% rule of §4.3 step 1a:
  `candidates_r15` (1.120), `candidates_rev` (0.024), `candidates_rev_probe` (9.858) and
  `hyp_pool` (0.718). In each, the coded rule produces more rows than the file holds. Ruling: the
  coded `POOL_RULES` stay as G(D), because they are the gate of the user's design. Rows that
  differ between file and rule are U per §4.3 step 1a (minute U = 566). Coverage below 100%
  (`crash` 0.48–0.67) is disclosed in §A0.1, and results are also reported per bracket. *Why:*
  §4.4 1b's "true rule" clause would change the user's gate to fit the cache. *If wrong:* the
  covered part of a bracket is a non-random subset of G, and the per-bracket table exposes that.
  This fills §4.4 1b and closes ambiguity 15 with no code change.
- **P3-9 / P3-11 (closes O2, completeness and the U2 drop).** The header box's U2 drop of the
  post-crash bracket is tested on the **pre-market** cache, because the screen needs pre-market
  data and §4.4 step 2 removes symbol-days whose pre-market data was not requested from every
  arm. Only the post-crash bracket is dropped as a bracket, and only in years where R15
  pre-market completeness is < 0.99. That is data-driven, and here it covers 2020–2025, every
  year the pool exists. The dropped bracket goes to S1. A symbol-day that also qualifies for
  another bracket stays. In 2016–2024 the drop removes nothing from the final universe, because
  the surviving post-crash members also qualify for another bracket. In 2025 it removes 1,139.
  `gap10`, `gap3` and `rev` stay governed per symbol-day by §4.3 step 4a and §4.4 step 2.
  `candidates_rev` has no pre-market data before 2025, so the `rev` bracket's final count is 0 in
  2016–2024. **Added to the untested strata of the header box and §4.5:** −10% ≤ gap ≤ −3% before
  2025, and post-crash entry sessions as a bracket in every dev year (S1 only). *Why:* the header
  box names only the R15 pool, so a generic bracket drop would go beyond the frozen text
  (P3-11 narrows P3-9). This fills the header box (U2) and §4.3 step 4a.
- **P3-10 (closes O3, cost coverage).** New exclusion **`zero_adj_volume`**, applied right after
  `short_history` in every arm: *a symbol is excluded from G(D) in every arm and variant, and
  counted as `zero_adj_volume`, if the median of its last 20 daily split-adjusted volumes
  (`volume_v`, rows strictly before D) is not positive.* *Why:* the 15 no-selection symbol-days
  that lacked CostInputs in the 4155601 audit all had adjusted volume rounded to 0 after very large
  reverse splits. The rule uses only rows before D, so it is causal. At 0457541 it removes 4,076
  step-1 symbol-days (§A0.1, which also shows it ahead of the gate). Books and CostInputs now
  match (34,051 each). This adds a step to §4.4 step 1.

#### A0.4 Other rulings from the model reviews (`model-review-2.md`, `model-report-3.md`, `model-rereview-2.md`)

1. **Gate ADV versus screen ADV (M4).** The gate and pool ADV (adjusted close × volume, mean of
   the last 20 rows including D−1, min_periods 5, which is the R3/R4 builders' `adv_20`) is used
   **only** to replicate the source pools in G(D) and the audit. It is never used for
   `min_adv_usd`. The screen's `min_adv_usd` is always the §4.4 step-1 statistic (20-session
   median of `close_unadj × volume_unadj`, raw × raw, V1). *Why:* the pools were built with their
   own ADV, but adjusted prices encode the future and are never a screen. Fills §4.4 1b.
2. **Seeding (M-5).** No stdlib `random` is used anywhere (a guard test flags any call), so no
   Python seed is set. The CNN worker calls `set_torch_determinism(42)` (with §5.4's
   deterministic flags and one thread), and its epoch shuffles use `torch.Generator` seeded
   42 + epoch. Every other draw uses a seeded `numpy.random.Generator`: bootstrap 16 (§10.2 and
   all clustered SEs), nulls 16 (training labels) and 17 (OOF labels), subsample 16, parity days
   16, martingale paths 16 + i. C_b scores are splitmix64(symbol, ts, seed), seeds 0–19. Fills
   §5.4 "seeds fixed for Python, NumPy and torch".
3. **Quote-size unit (Q1).** Dev uses `quote_size_scale = 1.0`: every dev date is before
   2025-11-03, so the cached sizes are in round lots and are read as the feed defines them. This
   was checked against the cache: the implied median touch is about 704 shares (~$2.8k) under the
   lot reading. Holdout scaling (lots → shares from 2025-11-03) is deferred to the pre-unseal step
   of §10.6 and changes no dev number.
4. **Null (d).** "Exactly zero" means |pnl| ≤ 1e-9 × E_{d−1}, on one cell per (exit arm ×
   gbdt / cnn / random / constant) over the first 5 OOS days. *Why:* constant prices with zero cost
   leave about 4e-12 from the engine-identical cash arithmetic. Fills §10.7 (d).
5. **Martingale AUC null** is report-only, not a void condition: every 10th OOS day of the last
   fold, path seeds 16 + i. *Why:* §10.7 lists it with no pass rule. Fills §10.7.
6. **Null retrain training set (M7).** Nulls (a) and (b) train on `store.train_rows`, the
   union of the fit, full and validation caps, which is a superset of the fold model's own capped
   set. *Why:* a larger set makes the null harder to fail by under-fitting. Fills §10.7.
7. **Threshold population.** Every primary tradable window of the inner-validation block,
   including label-invalid ones, scored per day. Fills §5.5 `entry_threshold`.
8. **Grid threshold shortcut (M1)** is report-only: the grid passes each model cell the minimum
   threshold over folds, which is equivalent because `score_oos` stores only scores at or above
   their own fold's threshold. `below_threshold` skip counts therefore differ from the engine's,
   and that count is not a parity field.
9. **C_b band scope (M13).** Last fold only, on the 200k smallest-key no-selection OOF windows,
   |z| ≤ 3.66 (ambiguity 8). C_b itself trades the screen arm. Fills §7 N5.
10. **Infeasibility floor (I2).** §11.4's median is taken over the **final** admissible count
    (step 1 ∧ short history ∧ `zero_adj_volume` ∧ `no_volume_factor` ∧ G after the P3-11 bracket
    drop ∧ cache ∧ C0/C1), not the step-1 count of the §11.4 example.
    Changes §11.4.
11. **Half days (M-1).** Taken from the NYSE early-close calendar (`data.sessions`), 19 dev days
    (§A0.1 `half_days`), instead of a minute-file 13:00 rule, which finds none because the cache
    holds after-hours bars. This defines the dev-session list and the folds. Fills §4.4 step 3,
    §10.1.
12. **Non-finite Sharpe (M2).** A non-finite Sharpe in any main cell makes `Var(SR_n)` NaN, so
    §11.1.3 fails for every cell. The failing cells are named in `evaluation.json`
    `dsr_failure`. Fills §10.4.

#### A0.5 Process

- Amendment 0 is written from an `audit --refit` run at the code SHA that smoke and full will
  use (0457541, §A0.1). The audit at 6057180 predates P3-5, and the audit at 4155601 predates
  P3-10 and P3-11; neither is used, and Amendment 0 was refreshed in place for 0457541 before any
  minute return was read. Under P3-7 it is not merged into the model branch until §A0.1 is filled,
  because its heading unskips the real-day parity test. If any later
  code change affects an audit output (universe, provenance, gate, costs, `c_label`), the audit is
  rerun at the new SHA and §A0.1 is replaced in a dated amendment before any minute return is
  read. Preflight enforces the link: smoke and full refuse unless this heading carries the sha256
  of the `amendment0_inputs.json` they load.
- **Autoresearch (user decision, 2026-10-26).** Autoresearch is used only as a performance loop
  (peak memory and wall time per stage, on the fixed 60-session smoke slice, ≤ 2024-12-31). A
  change is kept only if every stage output sha256 is bit-identical to the baseline at the fixed
  SHA and the full test suite passes. It never loops on returns, inner-fold AUC or any metric of
  §10, so it selects nothing and adds **no trials** to the DSR count. A kept speed change that
  touches audit inputs triggers the audit rerun above.

---

## Changes after G1

Source: `research/intraday/docs/critiques/G1-R16.md` (BLOCK; B1, B2, W1–W10, N1–N9) and one
controller correction (V1). Amendments made after G1 sign-off still go under "Amendments" above.

| Edit | Section changed | Change |
|---|---|---|
| U1 | header box, §1, §4.5, §12, §13, §14.1 | "Conditional on U1" box: every result conditional on U1, not Gate-I evidence until confirmed, void if U1 is rejected; tested and untested strata listed |
| E1 | §4.4 step 1b, step 2 | Open-time gate G(D) written as an explicit formula; admissible = step 1 ∧ G(D) ∧ cache ∧ C0/C1 (B1) |
| E2 | §5.1 | Percentiles and `top_n` taken within G at 09:30; `Candidates` stamped 09:30 (B1) |
| E3 | §4.5, §4.2 R4 row, §4.6 S2/S3 | Universe stated as exactly G(D) ∩ cache; `hyp_pool` is pre-market only; G3 may not redefine G; coverage per year and bracket (B1, W2) |
| E4 | §4.4 step 2, §5.1 | "Not requested" pre-market symbol-days removed from every arm; "requested, no bars" stays |
| E5 | §4.4 step 2b, §4.6 S4 | Pre-market vs minute basis check (factor 1.5), excluded from every arm, counted (W1) |
| E6 | §4.2 `hyp_pool` row, note | `hyp_pool` rows with open/prev close ≥ 1.8 are C2, the rest C0; `forensic_universe` finite-close note (B2) |
| E7 | §4.3 steps 1a, 4a | Pool reproducibility (1% rule) and completeness (≥ 99% to rescue C2/U) (W3) |
| E8 | §6.3 | Prior trials tabulated from disk: tally 4,194 (lower bound); `PRIOR_TRIALS_R16 = max(350, tally)` = 4,194; binding N = 4,248 (W7) |
| E9 | §6.3 | Trial-family principle written; breakout ledger added to N as a non-gating sensitivity |
| E10 | §5.5 | `atr_floor` = $0.01; invalid-bracket skips counted, 0 required for ATR arm |
| E11 | §5.5, §10.3, §10.6 | Threshold from the inner validation block, model refit on 100%; realised entry-eligible rate reported (W4) |
| E12 | §5.4 | Label uses the engine's fill rules (next existing bar, 2-min entry expiry, EOD close) (W5) |
| E13 | §6.4, §11.1.6 | Parity configurations and stratified days defined; identical to the cent; void-and-rerun rule (W6) |
| E14 | §10.6 | Disclosure adds §60 `pm_rvol` study and forensic/top-30 studies (W8) |
| E15 | §9.3 | Holdout uses the 2025 year dummy, no refit on holdout quotes; dev-wide cost data stated (W9, N3) |
| E16 | §7 | SPY from `QuantCode/data/cache/SPY.parquet`; span 2016-01-04 .. 2024-12-30 recorded; nothing fetched (N2) |
| E17 | §9.1 | Participation share volume on D's basis with a single factor (`risk.md:103-107`), as amended by V1 (N1) |
| E18 | §10.7, §5.4 | R16-specific nulls in `r16_nulls.py` replace `eval/nulls.py::run_all`; torch determinism flags named (W10) |
| E19 | §7, §10.3, §10.4 | Sorted-symbol entry order for C_c/C_0; C_b null-band check; Holm-arithmetic vs geometric note; EOD-forced fill counts (N4–N7) |
| E20 | §12 | Runs refuse a dirty tree under `research/`; SHA and clean flag tagged (N9) |
| V1 | §4.4, §5.1, §9.1, §9.3, §4.6, §14 | Volume converted with split-only s = volume / volume_unadj from `research_breakout/daily` (price factor must agree within 2%, else `no_volume_factor`); ADV = close_unadj × volume_unadj |
| — | header, §11.1.3, §10.4 | Status set to Revision 1; N = 404 replaced by N = 4,248 throughout |
| R8 | §7 | SPY 2025 dev sessions from `research_breakout/daily` (no fetch) |
| C1–C5 | box, §4.4, §4.5, §5.5, §10.6, §10.7 | G1 re-check changes: no unseal before U1; f_D in the assertion; C2/U pre-market removed; CNN refit epochs; post-crash exception |
| Q1 | §10.6 | Quote-size unit date-dependent before any holdout read (Alpaca: lots → shares on 2025-11-03); no dev effect |
