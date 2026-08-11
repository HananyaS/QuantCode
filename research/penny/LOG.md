# Penny-Stock Veto Study — Process Log

A running record of what was decided, what was tried, what broke, and what the
data said. Written so the study can be audited by someone who was not here.

Branch: `research/penny-veto` (never merged to `main`).
Companion data recorder: `HananyaS/QuantCode-penny-data` (private).

---

## 0. The question, and why it is not the obvious one

The prior consultation concluded that penny/microcap day trading has strongly
negative expected return, citing Eraker & Ready (2015), the SEC DERA outcomes
study (2016), Bali/Cakici/Whitelaw (2011), and the day-trader survival
literature (Barber et al.; Chague et al.).

The user's objection was correct and is the reason this study exists: **all of
that measures *unconditional* expectation.** A filter with high precision on a
toxic sub-population can move `E[r | survived]` even when `E[r]` is deeply
negative. Several of the cited papers effectively hand over the veto
conditions.

So the question under test is:

> Does there exist a veto set `V`, computable from free data at 09:35 ET with
> no look-ahead, such that `E[net return | not V] > 0` with a confidence
> interval excluding zero after multiple-testing correction, at a non-trivial
> retained population and a capacity worth having?

The honest prior is "no". The apparatus is built so that "no" is a cheap, fast,
publishable answer and "yes" is trustworthy.

### Fixed scope (user decisions)

| Parameter | Value |
|---|---|
| Universe | Nasdaq/NYSE/AMEX **listed** (no OTC) |
| Decision point | 09:35 ET |
| Horizon | flat by 15:55 ET, same session |
| Direction | long-only (short deferred) |
| Data budget | $0 |

---

## 1. Data feasibility (Phase 0)

Established earlier, and it gates everything: **the free Alpaca tier serves
consolidated SIP volume** — measured 28.6x IEX volume on the same window, and
within 0.4% of yfinance consolidated. IEX alone is ~2.5% of consolidated
volume, so an IEX-only feed would have made relative volume meaningless and
ended the project rather than degrading it.

Probes run for this phase:

| Question | Answer |
|---|---|
| How deep do daily bars go? | ≥ 2016 |
| Are minute bars available? | Yes, 391/session |
| Can minute bars be fetched multi-symbol per day? | Yes — 8 symbols, 3,394 rows, 0.8s |
| Do microcaps have minute bars? | Yes, but **sparse** — one $3 name printed 52 of 391 minutes |

That last row is not a footnote. Sparse prints are the executability problem
made concrete, and they drive assumption 2 of the labelling design.

---

## 2. Survivorship — the decision that most affects the answer

Today's Nasdaq symbol directory lists only what is *currently* listed. Every
microcap that delisted during the study window is absent, and those are exactly
the names that went to zero. Building the universe from it would have
conditioned the whole study on survival.

**Fix:** Alpaca's asset registry exposes `status=inactive`. Probed and confirmed:

- active: 14,207 US equities
- inactive: 19,201 (16,342 OTC, the rest listed)
- **inactive on a major exchange: 2,725**

After filtering to plausible common stock the study universe is **6,827
symbols — 5,795 active and 1,129 delisted.**

**Residual bias, stated rather than assumed away:** the registry carries no
listing/delisting *dates*, so membership is reconstructed from the bar record
(a symbol is treated as listed on days it printed bars). Names that both listed
and delisted before Alpaca's coverage began are still missing. Both effects
push toward optimism, so measured edges are upper bounds.

---

## 3. Raw vs adjusted — why both are fetched

They answer different questions and using one for both silently corrupts the
study.

- **Raw (unadjusted)** drives the price screen. A stock at $2 in 2022 that
  later reverse-splits 1-for-20 appears at $40 adjusted. An adjusted screen
  would exclude it on precisely the days it *was* a penny stock, and admit it
  on days it was not.
- **Adjusted** drives every cross-day feature — gap, trailing ADV, 20-day MAX.
  On a split date the raw series contains a fabricated overnight move that an
  unguarded gap feature reads as real, and reverse splits are endemic here.

Cost: roughly double the fetch. Worth it.

---

## 4. Labelling design

Triple-barrier (Lopez de Prado 2018, ch. 3), evaluated on a **minute path**
rather than a daily series, with an explicit executability model.

Timeline: `09:30–09:34` observation → `09:35` decision and fill →
`09:36–15:55` barrier scan.

Five assumptions, each pessimistic by construction:

1. **Ambiguous minutes resolve against the trade.** If one bar's high crosses
   the target *and* its low crosses the stop, the bar cannot say which came
   first. We assume the stop. For microcaps — whose minute ranges are enormous
   — the optimistic choice would manufacture edge out of bar granularity alone.
2. **No print means no fill.**
3. **Stops slip, targets do not.** A triggered stop is a market order crossing
   a falling book; a target is a resting limit.
4. **Entry pays the spread** plus square-root impact. The printed price is not
   a fill.
5. **Data gaps are suspicion, not absence** — flagged `halt_suspect`, because
   free halt history does not exist and during a halt a position cannot be
   exited at all.

One convention stated plainly: the scan starts at 09:36, so a barrier touched
inside the entry bar is not detected. This is the single place where the
pessimism is not one-directional.

---

## 5. Evaluation design

**Folds.** Date-blocked walk-forward with an embargo. Splits are on *whole
days* — same-day trades share a market factor, so dividing a day across train
and test would let the model learn that day's regime from one half and
"predict" the other.

A deliberate distinction: labels here are same-session, so label horizons do
**not** overlap and the classic purge for overlapping labels is unnecessary.
The embargo addresses serial correlation of regimes, which does exist. (The
audit in §7 sharpened this: an embargo cannot address "feature-window bleed" in
a strictly sequential design, and the docstring claiming otherwise was wrong.)

**Inference.** Three corrections, answering different questions:

- day-clustered bootstrap (trades within a session are not independent)
- Benjamini-Hochberg FDR across the search, reported **with the search size**
- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), because penny returns
  are violently right-skewed and the sample Sharpe flatters them

**Two questions kept separate.** "Is the surviving population profitable?"
(test against 0) and "does the veto help?" (test against the unfiltered
baseline) are different, and they come apart precisely when the baseline is
strongly negative. Both are reported.

**Fair baseline.** Any veto retaining X% is compared against *random* filters
retaining X%, dropping whole days to match how real vetoes behave. A filter
that merely trades less is not a filter that trades better.

---

## 6. Bug found by the test suite: the shuffled-label null

The shuffled-label null test failed on first run — **100% of veto sets passed
FDR on shuffled data.**

Diagnosis: `search_veto_sets` tests each subset against zero, which is the
right question for the real study. Under a shuffle every subset simply inherits
the panel's overall mean; since that mean is non-zero (and for penny stocks
emphatically negative), every subset read as "significant" and the null failed
for a reason unrelated to leakage.

Fix: demean returns after shuffling, so the null asks what it means to ask —
can the vetoes still *discriminate*. The same failure prompted adding
`compare_to_baseline`.

This is logged because it is instructive: a null test that fails for the wrong
reason is worse than no null test, since the natural response is to weaken it.

---

## 7. Adversarial leakage audit — one critical finding

An independent audit was run against the ETL, labelling, fold and selection
code, with instructions to assume a real mistake existed. It found one.

### CRITICAL — `veto_extreme_adjustment` used future information

The veto compared adjusted to unadjusted close, documented as detecting *past*
split activity. It detected the opposite. With `adjustment="all"` the adjusted
series is anchored to the **fetch date**, so `close_t / close_unadj_t` is the
cumulative factor for corporate actions occurring **after** day *t*. A 1-for-20
reverse split at date *T* sets that ratio to ~20 on every bar *before* *T*.

The veto therefore read "will reverse-split soon" — and in this universe
reverse splits follow price collapse. It was registered as tradable.

**Why walk-forward could not catch it:** the leak travels inside a feature
column present identically in train and test. It would have been selected on
train (the leak works there too) and then inflated the out-of-sample number —
the one figure the study says may be quoted.

**Fix:** differencing the ratio recovers the per-day event factor, which *is*
knowable (ex-dates are announced). `split_factor_180d` is the trailing product
of those factors and measures only splits that have already happened.

### Also fixed

| Finding | Mechanism |
|---|---|
| `ret_1d_prev` | `.shift(1)` applied **outside** the groupby, so each symbol's first row inherited the previous symbol's last return — years in the future |
| `consec_up` | `cumcount` counts rows, not True values; a NaN yesterday neither qualified nor reset yet still incremented, inflating streaks after any gap |
| entry spread | estimated from the **whole day's** path, charging days that later went wild a wider spread and coupling the label to future volatility |
| `assert_folds_are_clean` | compared `gap >= min(embargo, gap)`, true for every value — a dead tripwire inside the function whose job is to fail loudly |
| `_DAILY_COLS` | shipped current-session close/volume into the analysis table under innocent names; now `fut_`/`diag_` prefixed |
| diagnostics guard | `search_veto_sets` now structurally refuses look-ahead vetoes rather than relying on convention |

Two further bugs surfaced when the new regression tests first exercised the
feature builder on real shapes: `groupby.apply` for ADV returned a frame that
could not be assigned back, and `np.prod` over a window containing the
shift-induced NaN returned NaN — which a trailing `fillna(1.0)` disguised as
"no split activity".

**Audit verdict on everything else:** shift-before-rolling throughout, the
raw/adjusted split handled correctly, observation and label windows disjoint,
zero-padded `HH:MM` string comparison order-isomorphic and safe, selection
confined to train folds, and the placebo correctly day-level.

---

## 8. Parameter tuning protocol

Tuning uses the autoresearch loop (modify → commit → verify → keep/discard),
with one adaptation that matters: **the objective is out-of-sample.**

**Objective: the lower bound of the 95% CI on out-of-sample mean net return.**
Not the mean — maximising a mean is trivially gamed by filtering down to a
handful of lucky trades, whereas the lower bound falls as the sample shrinks,
so over-filtering punishes itself. It is also the quantity a decision actually
depends on.

Three nested layers, of which the loop sees only the outer two:

1. **Sealed holdout** — the final 20% of sessions, removed before anything
   else and scored exactly once, at the end.
2. **Walk-forward** inside the tuning period.
3. **Guards** — minimum out-of-sample trades and retained fraction; a
   configuration surviving by trading almost nothing is rejected.

---

---

## 9. Study period extended to 2016

Initially scoped 2021-2026. An early probe suggested minute bars did not exist
before ~2021 — but that probe had accidentally sampled **weekends** (2018-06-10
and 2023-06-10 are a Sunday and a Saturday) and read the empty responses as
missing history. Re-probed on known Wednesdays: minute bars are available with
full 391-bar coverage back to **2016**.

Extending mattered for power, not tidiness. At ~8 candidates per session, five
years yields ~9k trades — thin for an FDR-corrected search over ~1,000 veto
subsets. Eleven years roughly triples it.

### Fetch integrity

Two fetch processes ended up running in parallel (a `pgrep` guard failed to see
Windows processes). This was checked rather than assumed, because a 429 would
not match the "invalid symbol" pattern and would have been bisected down and
silently recorded as *rejected symbols* — real data loss that leaves no error.

Result: **0 rejected in every year**, 250-251 sessions per year, symbol counts
consistent. No rate-limit damage.

| Panel | |
|---|---|
| rows | 11,202,209 |
| symbols | 6,778 |
| period | 2016-01-04 .. 2026-08-07 |
| unadjusted close missing | 0.00% |

### Candidate screen

| Cumulative filter | rows |
|---|---|
| has ≥30 prior bars | 11,000,517 |
| price \$0.50-\$10 (previous **unadjusted** close) | 3,407,083 |
| trailing 20d dollar volume ≥ \$100k | 2,655,397 |
| opening gap ≥ +10% | **20,558** |

**20,558 symbol-days over 2,555 sessions (8.0/day), 2,960 unique symbols.**

The per-year distribution is itself a finding:

| | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| candidates | 556 | 796 | 898 | 899 | 3475 | 1703 | 1572 | 1760 | 2715 | 3859 | 2325 |
| per session | 2.9 | 3.4 | 3.8 | 3.8 | 13.8 | 6.8 | 6.3 | 7.1 | 10.8 | 15.5 | 15.5 |

*2026 is partial (150 sessions).

The 2016-2019 baseline sits near 3.5 gap-ups per session. 2020 spikes to 13.8,
and 2025-2026 run at 15.5 — roughly four times the pre-2020 rate. That is
consistent with the ramp-and-dump wave FINRA and the FBI flagged (the FBI
reported a >300% increase in complaints in 2025 over 2024). It also means the
population is **not stationary**, which is exactly why fold-level stability is
reported rather than a single pooled number.

---

---

## 10. Labelled dataset

| | |
|---|---|
| candidates labelled | 20,367 |
| dropped as unfillable (no print at 09:35) | 2,977 (14.6%) |
| **trades in the study** | **17,390** |
| tuning period | 10,893 trades / 2,017 sessions |
| sealed holdout | 6,497 trades / 505 sessions |

**Outcome mix:** stop **52.8%**, time-exit **40.2%**, target **7.0%**.

**Mean net return −5.14%**, gross **−0.92%**, mean cost **437 bps**.

Two things stand out immediately.

First, the 14.6% unfillable rate is not a data problem, it is the finding: in
one trade out of seven there was **no print at all** in the decision minute.
Assumption 2 of the labelling design is doing real work.

Second, and more important: **the population has no gross edge either.** Gross
is −0.92%, so even at zero cost the strategy loses money. Costs then remove a
further 4.4 points. This is precisely the Avramov/Chordia/Goyal and
Novy-Marx/Velikov mechanism — the effects live where the frictions are largest.

---

## 11. First full result

All three null tests **PASS** (constant-return spread 3.5e-18; shuffled labels
0.000; date-shift 0.000).

| Population | E[net] | 95% CI | n |
|---|---|---|---|
| baseline (no veto) | **−5.26%** | [−5.55%, −4.97%] | 10,893 |
| best walk-forward OOS | **−4.58%** | [−5.12%, −4.04%] | 3,128 |
| **sealed holdout** | **−4.58%** | [−5.11%, −4.04%] | 1,725 |

- veto sets searched: **299**; passing one-sided FDR: **0**
- Deflated Sharpe: **0.000**
- selection degradation train→test: **0.0036** (small — selection is stable)
- vetoes chosen: `veto_high_max` and `veto_thin_adv` in **all four folds**,
  `veto_open_reversal` in three

**H0 is not rejected.** The vetoes measurably help — 0.68 percentage points,
consistently, and stably across folds — and the surviving population is still
firmly unprofitable. This is Lemma 2 exactly: *improvement is not
profitability*, and the two come apart precisely when the baseline is as
negative as it is here.

### A reporting bug worth recording

The first run reported "**299 of 299 veto sets pass FDR**". True, and useless:
the p-value was two-sided against zero, and in a universe where returns are
almost uniformly negative *every* subset differs significantly from zero. It
read as a wall of discoveries when nothing was profitable.

FDR is now applied to the **one-sided** hypothesis (mean > 0), which is what
the study asks. The corrected count is 0 of 299.

### A performance bug worth recording

The first analysis run did not finish. The day-clustered bootstrap concatenated
each drawn day's array inside a Python loop — exact, but hundreds of millions
of list operations across the search. Resampling per-day (sum, count) pairs is
algebraically identical and runs in seconds. The equivalence is pinned by a test
against the naive estimator using the same draw sequence, because "it's faster
now" is not evidence that it is still the same estimator.

---

## 12. Tuning

Objective: the lower bound of the 95% CI on out-of-sample mean net return.
Baseline **−0.051154**.

| # | Change | Metric | Status |
|---|---|---|---|
| 0 | baseline | −0.051154 | — |
| 1 | `min_adv` 1M → 5M | −0.054060 | discard — n fell 3,128 → 2,145 and the interval widened |
| 2 | `profit_target` 30% → 15% | −0.051154 | **no-op — see below** |

Iteration 2 returned *exactly* the baseline, which was itself a bug rather than
a null result. The harness rebuilds vetoes from config, but barrier geometry is
baked into the labels at dataset-build time — so the change was silently
ignored, and an experiment that never ran was about to be recorded as evidence
of "no effect".

`tune_objective` now refuses to run when configured barriers disagree with the
built dataset, and `sweep_barriers.py` performs that sweep properly with a full
rebuild per geometry.

### 12.1 Barrier sweep

Six geometries, each requiring a full dataset rebuild.

| target / stop | break-even win rate | objective | net | **gross** | P(target) | P(stop) |
|---|---|---|---|---|---|---|
| **+10% / −5%** | 33.3% | **−0.04752** | −4.65% | **−0.39%** | 15.5% | 75.9% |
| +30% / −5% | 14.3% | −0.04797 | −4.71% | −0.46% | 4.0% | 81.0% |
| +15% / −7% | 31.8% | −0.05048 | −4.93% | −0.69% | 13.2% | 67.2% |
| +20% / −10% | 33.3% | −0.05099 | −5.09% | −0.86% | 12.0% | 51.6% |
| +30% / −10% | 25.0% | −0.05115 | −5.14% | −0.92% | 7.0% | 52.8% |
| +50% / −15% | 23.1% | −0.05280 | −5.09% | −0.87% | 4.0% | 31.5% |

**The decisive column is gross.** It is negative at *every* geometry. This is
not only a cost story: there is no directional edge to harvest at any
reward/risk ratio. Tighter stops help (they cap the loss) but cannot create an
edge that is not there.

### 12.2 The sweep died once, and how

The first attempt shelled out to `build_dataset` with a one-hour subprocess
timeout. The machine slept mid-sweep; the wall clock advanced past the timeout
while the child was suspended; the run died having completed one geometry. Worse,
the `finally` block then "restored" a config that had already been mutated,
leaving configured barriers (0.15) disagreeing with the built labels (0.30).

The guard added in §12 caught exactly that on the next run — which is the only
reason it was noticed rather than silently producing results for a geometry
nobody chose. The sweep now runs in-process, persists after every geometry, and
resumes from what it already measured.

### 12.3 Full iteration log

| # | Change | Metric | Status |
|---|---|---|---|
| 0 | baseline | −0.051154 | — |
| 1 | `min_adv` 1M → 5M | −0.054060 | discard — n fell 3,128 → 2,145, CI widened |
| 2 | `profit_target` 30% → 15% | −0.051154 | no-op — harness ignored barriers (bug, fixed) |
| 3 | barrier geometry → +10%/−5% | **−0.047518** | **keep** |
| 4 | `max_ret_threshold` 0.50 → 0.30 | **−0.045802** | **keep** |
| 5 | `min_close_position` 0.35 → 0.50 | −0.046184 | discard — mean improved, n fell, CI widened |
| 6 | `max_hl_range` 0.20 → 0.12 | **−0.045324** | **keep** |
| 7 | `min_open_dollar` 250k → 1M | −0.045324 | no-op — that veto is in no selected set |
| 8 | veto sets of size 4 | −0.046348 | discard — worse, **and selection became unstable across folds** |

Net movement: **−0.051154 → −0.045324**, a 0.58 percentage-point improvement in
the out-of-sample CI lower bound. Real, and nowhere near enough.

Iteration 8 is worth noting as a diagnostic rather than a failure: allowing
size-4 sets made the chosen vetoes differ in every fold, which is the textbook
signature of selection overfitting. The objective correctly penalised it.

---

## 13. Final result

Configuration: barriers +10%/−5%, `max_ret_threshold` 0.30, `max_hl_range` 0.12,
veto sets up to size 3.

| Population | E[net] per trade | 95% CI | n |
|---|---|---|---|
| baseline (no veto) | **−4.71%** | [−4.85%, −4.57%] | 10,893 |
| best walk-forward OOS | **−4.28%** | [−4.54%, −4.03%] | 2,374 |
| **sealed holdout** | **−4.48%** | [−4.81%, −4.15%] | 1,213 |

- veto sets searched: **299**; passing one-sided FDR: **0**
- Deflated Sharpe Ratio: **0.000**
- all three null tests: **PASS**
- holdout vetoes: `veto_thin_adv`, `veto_wide_spread`, `veto_open_reversal`

**H0 is not rejected.**

### The pre-registered kill criterion

> "Phase 4: no veto set achieving net E[r] > 0 at 95% after FDR correction, with
> ≥2% of the population retained → null not rejected. Stop and report the
> negative result."

Met. This is that report.

### Mechanically, why

1. **No gross edge.** Best-geometry mean gross return is **−0.39%** — negative
   before any cost, at every geometry. Buying a gapping microcap at 09:35 and
   exiting the same session is not a positive-expectation bet spoiled by
   frictions; it is a negative-expectation bet that frictions worsen.
2. **Costs remove ~4.4 more points.** 437 bps round trip, estimated from the
   opening window's own range plus square-root impact at 1% participation.
3. **One trade in seven was never enterable.** 2,977 of 20,367 candidates
   (14.6%) had no print in the decision minute.

### What the vetoes *did* do

They worked in the sense of Lemma 1 — identifying genuinely worse
sub-populations, consistently and stably. `veto_thin_adv` was selected in every
fold; train→test degradation was **0.0036**. They are not overfitted artefacts.
They are simply an order of magnitude too small, which is Lemma 2.

### Honest limits of this result

- **Long-only, and the evidence points short.** Every effect found runs the same
  direction: high-MAX, illiquid, wide-spread names underperform. That is a short
  thesis, gated on borrow data the companion recorder is now accruing.
- **No filing-based vetoes.** Pending dilution (S-3/424B5), Nasdaq deficiency
  notices (8-K 3.01) and going-concern language are free from EDGAR and were not
  in this run. They target the mechanism most directly and are the highest-value
  next step.
- **Two vetoes were inert.** The short-volume pair never fired (no FINRA data
  wired) and were excluded from the search rather than silently inflating the
  hypothesis count.
- **Marketable entry only.** All cost is paid crossing the spread at 09:35. A
  resting-limit entry would change the cost structure entirely, at the price of
  non-fills — which this dataset can measure.
- **Survivorship is improved, not solved.** 1,129 delisted names are included,
  but the registry has no listing dates, so membership is reconstructed from the
  bar record. Residual bias points toward optimism.

---

## 14. Bugs found and fixed during this run

Recorded because in this domain every one of them would have produced a
*better* number.

| Where | Bug | Found by |
|---|---|---|
| `veto_extreme_adjustment` | used **future** splits (adjusted series is anchored to the fetch date) | adversarial audit |
| `ret_1d_prev` | `.shift(1)` outside the groupby → cross-symbol bleed from the future | audit |
| `consec_up` | `cumcount` counted NaN rows, inflating streaks after gaps | audit |
| entry spread | estimated from the **whole day**, coupling the label to future vol | audit |
| `assert_folds_are_clean` | embargo assertion was a tautology that could never fail | audit |
| `_DAILY_COLS` | shipped current-session close/volume under innocent names | audit |
| shuffled-label null | tested against zero, so a non-zero panel mean failed it for the wrong reason | own test suite |
| FDR reporting | two-sided test flagged 299/299 in an all-negative universe | reading the output |
| bootstrap | per-day concatenation in a Python loop; analysis never finished | runtime |
| `groupby.apply` for ADV | returned a frame that could not be assigned back | new tests |
| `np.prod` over NaN | shift-induced NaN disguised as "no split activity" by `fillna(1.0)` | new tests |
| barrier tuning | harness ignored barrier params entirely; a non-experiment was about to be logged as evidence | metric identical to baseline |
| `nulls.run_all` | `status` is a property, so `__dict__` omitted it | notebook execution |

Test suite: **116 penny tests**, all passing; **437** across the whole repo.
