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

---

# ROUND 2 — model-based selection

Round 1 asked whether a *veto filter* could work. Round 2 asks whether a
*learned ranker* over the full continuous feature set, plus filing data that
observes the supply mechanism directly, can close the gap.

## 15. Round 1's excursion statistics were an artifact

`max_favourable` was measured only until a barrier fired. That is the right
quantity for describing a trade and the wrong one for describing an
opportunity, and round 1 conflated them.

| | truncated (round 1) | **barrier-free (true)** |
|---|---|---|
| P(MFE ≥ 10%) | 0.156 | **0.394** |
| P(MFE ≥ 20%) | 0.004 | **0.187** |
| P(MFE ≥ 50%) | 0.000 | **0.054** |
| P(MFE ≥ 100%) | 0.000 | **0.0145** |

The truncated numbers said "price never goes anywhere". The truth is that it
goes a long way — **1.45% of candidates double intraday** — and the exit rule
threw it away. The user's original observation was correct all along.

## 16. Exit policy is not the lever either

Nine policies on 2,803 real paths — hold-to-close, targets to +50%, no stop,
trailing 10/15/20%, wide stops:

**Every one produced gross between −0.09% and −0.33%.**

The large MFE is what a driftless walk with 21.6% daily volatility produces,
not exploitable drift. A trailing stop gives back precisely what a fixed target
would have captured. Selection, exit policy and barrier geometry have now all
been tested; none of them is the binding constraint.

## 17. A null test invalidated the headline diagnostic before it was reported

The model's score-decile monotonicity read **ρ = +0.842**, which looks like
strong evidence that the score orders outcomes.

Under shuffled labels the same statistic averaged **|ρ| = 0.515** and reached
**+0.709**. With ten bins, Spearman ρ has a standard deviation near 0.33, so a
"real" 0.842 is not distinguishable from noise.

This was nearly reported as the round-2 headline. Evidence now comes from
`compare_selection_to_baseline` — a paired, day-clustered bootstrap of the
return difference — which is the only statistic here that discriminates.

## 18. What the model actually achieved

Meta-labelling LightGBM, walk-forward, model fitted strictly inside each fold.

| features | best gross | lift vs baseline | p |
|---|---|---|---|
| price/volume only | +0.14% | +1.17pp | 0.025 |
| **+ EDGAR filings** | **+1.51%** | **+2.54pp** | 0.032 |

The filing features are the largest single improvement in the whole project —
roughly a tenfold increase in achievable gross. That is the round's one genuine
positive finding, and it supports the prior: features that observe supply
directly beat features that proxy it through price.

Prevalence in the candidate population: **18.6%** have a listing-deficiency
notice (8-K item 3.01) within 180 days; **48.9%** a dilutive 8-K; **86.5%** have
filed a 424B takedown at some point.

## 19. The regime finding, and why it fails

Edge appeared to concentrate in busy regimes:

| regime | all | top 10% | lift |
|---|---|---|---|
| quiet | −1.54% | −0.28% | +1.25pp |
| normal | −0.64% | +1.29% | +1.93pp |
| **busy** | −0.91% | **+2.11%** | **+3.03pp** |

+2.11% gross clears the *optimistic* 200 bps bar. It does not survive:

- **No CI excludes zero** at any selectivity (best p = 0.056, uncorrected)
- **Only 2 of 4 folds contain busy-regime days**, and they disagree wildly:
  +0.34% versus +3.00%
- **Deflated Sharpe = 0.127** at 27 trials. The gross Sharpe of 1.52 looks
  excellent and does not survive correction for having looked 27 times
- Net at 200 bps: +0.11%, CI [−2.48%, +3.85%]

A single period (fold 3, 2024–26) drives it. This is what an overfit looks like
when the machinery is honest enough to say so.

## 20. Round 2 verdict

**H1 is not met at any tested cost level (200 / 300 / 437 bps).**

| | |
|---|---|
| baseline gross | −1.08% |
| best model gross (with EDGAR) | **+1.51%** |
| best gross 95% CI | [−0.56%, ...] — includes zero |
| best net @ 437 bps | −2.86% |
| best net @ 200 bps (optimistic) | −1.74% |
| trials in the grid | 12 (27 including the regime search) |

The model is real and the lift is statistically significant (p = 0.001–0.03
across configurations). It moves gross from −1.08% to +1.51%, a genuine
**+2.6 percentage points**. The cost bar is 200–437 bps and the confidence
interval on the best configuration still includes zero.

**Two rounds now agree, by different mechanisms.** Round 1: no veto set works.
Round 2: a learned ranker with filing data works *as a ranker* and still cannot
reach the bar. The binding constraint is that the candidate population's gross
expectancy is roughly −1%, and the best available selection buys ~2.5 points of
that back — against a friction cost of at least 2 points and probably 4.4.

## 21. What would actually change this

Ranked by how much of the remaining gap each could close:

1. **Measure the cost bar.** It is assumed, never measured: 2 × (118 bps
   half-spread + 100 bps impact), both by convention. The best model gross of
   +1.51% is *profitable* if true round-trip cost is below ~150 bps. This is now
   the single highest-value experiment in the project, and it was explicitly
   out of scope this round. Alpaca's free tier should serve historical NBBO
   quotes, which would settle it in a day.
2. **More filing features.** EDGAR gave the largest lift of anything tried.
   Untested: going-concern language, shares-outstanding growth from XBRL, ATM
   program size, warrant overhang.
3. **A different population.** Every result here is conditional on "gap ≥ 10%
   microcap". Intraday breakouts, halt-resumptions, and post-news setups are
   different populations that this apparatus could screen without modification.

## 22. Round 2 bug ledger

| Where | Bug | Found by |
|---|---|---|
| `max_favourable` | truncated by the exit; reported as opportunity | barrier-free re-measurement |
| decile monotonicity | ρ = +0.84 not distinguishable from a 0.515 null | shuffled-label test |
| `build_pit_features` | O(symbols × filings) scan; would not have finished | running it on 1.4M rows |
| EDGAR searchsorted | tz-aware `.to_numpy()` gives object dtype, cannot compare | tests after the rewrite |
| `select_top_fraction` | `groupby.apply` deprecation, silently changes grouping behaviour | pandas warning |

---

# ROUND 3 — measured costs, four new data sources, sequence models

## 23. Gate A: the cost bar, finally measured

Rounds 1-2 charged a flat **437 bps** that nobody had measured. Gate A replaced
both constants with measurements from NBBO quotes on **9,794** candidate
symbol-days.

| component | assumed | **measured** |
|---|---|---|
| half-spread | 118 bps | **37.7 bps** |
| impact @1% ADV | 100 bps | **71.8 bps** |
| **depth shortfall** | **0 bps** | **398.4 bps** |
| **round trip** | 437 bps | **728.9 bps** |

The assumption was wrong in both directions and they partly cancelled. Spread
was ~3x too harsh; **depth was ignored entirely**, and it is the largest term.
The median inside quote is ~$1,900 while 1% of ADV is ~$19,000 — roughly **10x
the resting size**, so the order walks the book.

I had predicted, from a two-name probe, that costs would come in ~5x *lower*.
That probe was right about the term I looked at and blind to the term that
dominates. Worth recording as a reasoning failure, not just a number.

**Cost is a function of order size, which no previous round modelled:**

| order | round-trip |
|---|---|
| $2,000 | ~162 bps |
| $5,000 | ~257 bps |
| $10,000 | ~387 bps |
| $25,000 | ~739 bps |

with an **irreducible floor of 77 bps** — crossing the spread twice — paid at
any size.

Re-measured at 3.8x the sample (2,585 -> 9,794 symbol-days), the numbers moved
by under 1 bp. The measurement is stable.

## 24. Four new data sources

| source | coverage | notes |
|---|---|---|
| **premarket 04:00-09:34** | 99.8% | median **83 bars** vs 5 before; median **$1.94m** pre-open dollar volume |
| **FINRA short volume** | 84.8% since 2018-08 | lagged one session |
| **XBRL shares outstanding** | 86.7% | float proxy; 25.6% of candidate-days show >5% QoQ share growth |
| **EDGAR continuous recency** | 94.5% | binaries replaced with days-since |

Premarket features carry real univariate signal — `pm_range` IC **−0.178** with
a cleanly monotone decile gradient (−0.25% to −3.85%). As everywhere in this
project, the signal is in *avoiding the worst*, not finding winners: the best
decile is only **+0.08%**.

## 25. Adding features made the model worse

The ablation is monotonically **degrading**:

| feature set | n features | top-10% gross |
|---|---|---|
| base | 33 | **+0.71%** |
| +EDGAR | 36 | +0.43% |
| +premarket | 52 | +0.31% |
| +short volume | 55 | +0.19% |
| +float | 60 | +0.13% |

**Gate B fails**: premarket lift is **−0.12%** against a +0.50% threshold.

This directly contradicts round 2, where EDGAR appeared to give a 10x
improvement. The difference is the fold restructuring. Round 2's fold 0 trained
on **610 rows**; round 3's minimum training window is 900 sessions, so fold 0
now trains on **2,682**. With a sane baseline the base model scores +0.71%
instead of round 2's +0.14% — and the apparent EDGAR benefit largely
disappears.

**Round 2's headline improvement was substantially an artifact of a degenerate
fold.** Fixing the fold structure was worth more than any feature added since.

The mechanism for the degradation is ordinary: 33 -> 60 features against
2,682-10,613 training rows is feature bloat, and each addition buys more
variance than signal.

## 26. The sequence models — the one thing that helped

Gate B failed, so under the pre-registered protocol the sequence encoders
should not have been built. They were run anyway as an **explicitly exploratory
arm**, excluded from the H1 family and from the multiple-testing correction.

| model | top-10% gross | gross 95% CI | lift | p |
|---|---|---|---|---|
| base GBDT | +0.71% | [−0.62%, …] | +1.60pp | 0.017 |
| blend:cnn | +0.56% | [−0.39%, …] | +1.45pp | 0.002 |
| blend:gru | +0.65% | [−0.46%, …] | +1.53pp | 0.004 |
| **blend:transformer** | **+1.22%** | **[+0.14%, …]** | **+2.10pp** | **<0.0001** |

The transformer blend is **the first configuration in the entire project whose
gross confidence interval excludes zero.** The pre-open path evidently carries
shape information that ~16 hand-made aggregates destroy — which is exactly the
argument for using a sequence model at all, and exactly why it was *not*
justified on the tabular features (Grinsztajn et al. 2022).

It remains exploratory. Gate B was pre-registered and it failed; reporting a
post-hoc arm as confirmatory would be the error the whole apparatus exists to
prevent.

## 27. Round 3 verdict

**H1 is not met at any order size.**

| | |
|---|---|
| best gross (exploratory) | **+1.22%** — CI [+0.14%, …] |
| best gross (pre-registered) | +0.71% — CI includes zero |
| net @ $2,000 | **−0.90%** |
| net @ $5,000 | −3.47% |
| net @ $10,000 | −6.95% |
| H1 family | 45 cells, BH-FDR corrected |
| cumulative trials (rounds 2+3) | 72 |

The measured cost floor is ~162 bps at a $2,000 order. The best gross the
project has produced is **122 bps**, and that from an exploratory arm. Even at
the smallest tradable size, and with the best model, **gross does not reach the
cost floor.**

## 28. Round 3 bug ledger

Nine issues from an independent adversarial audit; three were real, and one of
my own tripwires caught a genuine data fault.

| Where | Bug | Found by |
|---|---|---|
| `blend_scores` | ranked over the pooled panel, so scores depended on later folds — a real look-ahead I introduced | audit |
| `compare_selection_to_baseline` | mask truncated after `dropna`, pairing mask entries with wrong trades | audit |
| H1 verdict | unadjusted maximum over ~72 correlated CIs; trial count ignored the order-size dimension | audit |
| FINRA raw files | **2,262 duplicate (symbol, date) rows** — `shift(1)` would return the same session, not the prior one | my own duplicate guard |
| FINRA tripwire | asserted "differs from today" and fired on real, correctly-lagged data (ACAMU, 100% short twice running) | running it |
| `edgar.py` comment | claimed acceptance times are Eastern when they are UTC; following it would have created a 4-5 hour look-ahead | audit |
| premarket test fixture | generated bars past 09:35 — the boundary guard caught my own test | running it |
| allowlist | cost/score/fold columns not in the forbidden set | audit |

Audit items verified **correct**: FINRA lag placement, XBRL searchsorted
arithmetic and tz handling, premarket causality (checked in cache: max minute
09:34, zero bars ≥09:35), sequence left-padding and truncation, chronological
validation split, one-to-one merges, and the feature allowlist.

---

# Round 4 — precision inside the liquid band

## 29. Round 3's cost conclusion was drawn from a mis-specified table

Round 3 measured a median round-trip of **728.9 bps**, dominated by a **398.4 bps
depth shortfall**, and concluded the binding constraint is microstructure rather
than prediction. That conclusion was premature, for a reason visible only on
re-reading the code that produced it.

`build_cost_table` sizes every order at **1% of ADV** (`etl/quotes.py:196`). A
more liquid name therefore receives a proportionally *larger* order and comes out
looking more expensive. That is not how the strategy would trade — a real book
places a fixed dollar amount. Recomputed at a fixed order size, the ranking
inverts, and the spread is enormous:

| ADV bucket | n | RT @ $2k | RT @ $10k | uncond. gross |
|---|---|---|---|---|
| ≤ $300k | 1,501 | 395.1 bps | 1034.6 | −1.24% |
| $300k–1M | 2,085 | 259.4 | 683.8 | −0.80% |
| $1–5M | 2,731 | 149.4 | 371.6 | −1.00% |
| $5–20M | 1,956 | **98.6** | 215.5 | −0.45% |
| > $20M | 1,521 | **63.1** | **124.0** | −0.59% |

A **6.3× span**. The "irreducible 77 bps floor" reported in §23 is a whole-panel
median, not a floor — in the liquid tail the round trip is 63 bps at a $2,000
order. **Cost is a filter, not a wall**, and `adv_20` is ex-ante, already in the
panel, and free.

Two further facts made the round-3 framing worse than it needed to be:

- **Cost and edge were never crossed.** Selection is a rank cut on score alone
  (`run_round3_study.py:133`); measured cost is subtracted *after* selection
  (`:165`); and every quote-derived column is banned from the feature list
  (`model/features.py:249-251`) — by convention, with no leakage rationale
  recorded anywhere in the code, comments or tests.
- **The screen contradicted the project's own veto.** `adv_20 >= $100k`
  (`etl/candidates.py:62`) admits names ten times thinner than the `$1M` floor
  round 1's `_veto_thin_adv` used (`features/veto.py:84`).

## 30. Three cost-accounting defects, all of which flattered the result

Fixed **before** testing the promising hypothesis, which is the only thing that
distinguishes this from motivated reasoning. All three push costs up.

**Depth was charged once for a round trip that crosses the book twice.**
`2 × (half_spread + impact) + depth` (`etl/quotes.py:204`) assumes the 15:55 exit
fills entirely at the inside. Since depth was 398 of the 728 bps, the single
largest term was half-counted. Each leg is now charged against its own measured
book.

**Only the ask was ever measured.** A long exit sells into the *bid*. Round 3
never fetched bid depth, so the exit leg was priced off the wrong side of a book
whose two sides routinely differ by an order of magnitude.

**The estimate peeked at the entry minute.** The measured half-spread is the
median over `[09:35:00, 09:36:00)` (`etl/quotes.py:85-86`) while entry is the
09:35 bar **open** (`labels/triple_barrier.py:172`). Harmless while cost was
evaluation-only; disqualifying the moment cost informs the decision, which is
exactly what round 4 does. The two are now separate quantities:

| | window | role |
|---|---|---|
| `cost_hat` | `[09:34:00, 09:35:00)` | decision input — what was knowable |
| `cost_realized` | 09:35 ask + 15:55 bid | charged — what actually happened |

**Coverage, stated plainly for the first time.** Round 3's cost table covered
**56.3%** of trades (9,794 of 17,390) and *dropped* the remainder from the
bootstrap rather than defaulting them. The code concedes the direction of the
bias — "tilts the estimate toward names with firm quotes — i.e. optimistically"
(`run_round3_study.py:174`) — but §24's coverage table omits the cost table
entirely, so the published net figures carry an unstated 44% exclusion. One
mitigating fact, verified: the quote-match rate is flat across ADV buckets
(53.0 / 56.3 / 56.9 / 57.7 / 57.2 %), so the gap is missing-at-random with
respect to liquidity — under-powered rather than skewed.

## 31. Five caches that would have silently supplied round-3 coverage

Found while pointing the pipeline at the rebuilt funnel. The same latent bug in
five places, and in every case the symptom is a log line reading `cached`, not an
exception.

| Fetcher | Skipped when | Consequence for round 4 |
|---|---|---|
| `fetch_intraday` | session parquet exists | 2,553 sessions skipped, round-3 symbols only |
| `fetch_premarket` | session parquet exists | same |
| `fetch_edgar` | `edgar_filings.parquet` exists | **436 symbols** with no filing features |
| `fetch_finra` | `finra_short_volume.parquet` exists | **277 symbols** with no short volume |
| `fetch_xbrl` | `xbrl_shares.parquet` exists | 483 symbols with no float proxy |

Each cache is keyed on a path that encodes nothing about which symbols it
covers. That was safe for three rounds because the candidate screen never
changed; round 4's funnel overlaps round 3's by only 2,212 of 2,488 symbols.

Each now computes the symbol complement and fetches only that. The bar fetchers
key on a **requested**-symbols sidecar rather than the returned set, because a
thin name legitimately has no prints and keying on what came back would refetch
it forever. FINRA merges with a plain concat plus an explicit disjointness
assertion, deliberately **not** `drop_duplicates`: its consolidated files carry
several rows per (symbol, date), one per market, which `build_lagged_features`
sums — a content-based dedupe there would silently understate short volume,
which is the same duplicate-row property §28 had to handle in the other
direction.

Two more caught while running:

- **Alpaca's `end` is an inclusive instant.** A request for
  `[09:34:00, 09:35:00)` returns the quote stamped exactly `09:35:00.000` — a
  quote from the entry instant itself. The new `cost_hat` tripwire fired on ~3%
  of sessions. This is the same trap that made rounds 1–2 drop each target day's
  own bar (§7). Now trimmed before asserting, so the assertion verifies the trim
  rather than restating the request.
- **A BOM in my own tooling.** A `Set-Content -Encoding utf8` while editing
  `fetch_xbrl.py` wrote a UTF-8 BOM and mangled an em-dash, breaking the module.
  Windows PowerShell 5.1's `utf8` means *with* BOM. Round 3 hit the BOM problem
  reading Nasdaq's halt RSS (§8); this time it was self-inflicted.

**`build_edgar_features.py` did not exist.** Round 3 produced
`edgar_features.parquet` ad hoc, with no script in the repo that regenerates it —
so the input underneath round 2's largest claimed lift could not be rebuilt from
the codebase. Written now.
