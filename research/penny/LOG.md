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

## 32. The liquidity screen changed what population is under study

Worth stating before any result, because the ADV floor would otherwise redefine
the question quietly.

| | round 3 | round 4 |
|---|---|---|
| screen | gap ≥ +10%, ADV ≥ $100k | gap ≥ +3%, ADV ≥ $5M |
| candidates | 20,558 | **44,136** |
| labelled trades | 17,390 | **41,895** |
| symbols | 2,785 | 2,419 |
| mean gross return | −0.39% | **−0.22%** |
| modelled cost | 437 bps | **318 bps** |
| minute coverage | 82.7% | **92.2%** |
| EDGAR coverage | ~85% | **95.5%** |
| median float | microcap | **$437M** |
| 8-K deficiency within 180d | 18.6% | **11.1%** |
| reached the +30% target | 15.5% | **1.9%** |
| ran to the 15:55 close | 8.6% | **78.3%** |

Three things follow, and only the first is good news.

**The population is less toxic.** Unconditional gross is roughly half as
negative, data coverage is materially better because liquid names actually
print, and the filing-distress rate nearly halves.

**It is also far less explosive.** Only 1.9% of trades reach +30% against 15.5%
in round 3, and 78% simply run to the close. The right tail that motivated the
whole study — the user's original observation that these names "explode every
day with tens or hundreds of percent" — is largely *not present* in the liquid
band. Less violence means less to lose to a stop, but also less to win.

**Median float is $437M and median share count 111M.** These are low-*priced*
stocks of real companies, not the pump-and-dump microcaps rounds 1–3 studied.
Round 4 therefore answers "is there an edge in liquid low-priced gappers", which
is a genuinely different question from "is there an edge in penny stocks". The
answer is worth having either way, but it is not a continuation of the same
measurement, and no round-4 number should be quoted as though it were.

One consequence for the labels: the `+30% / −10%` barrier geometry was chosen
for the violent population and is badly matched to this one, which is why 78% of
outcomes are `vertical`. It does not drive the verdict — the study evaluates
`ret_c2c` from the barrier-free path, not the barrier outcome — but the barrier
sweep of §12 should not be assumed to transfer.

## 33. Gate A′: the cost bar, measured honestly — and it is not the constraint

Every session, both books, both legs. 2,582 sessions, 43,792 symbol-days,
**99.9% of trades matched** against round 3's 56.3%.

| | round 3 | round 4 (liquid band) |
|---|---|---|
| entry half-spread | 37.7 bps | **21.2** |
| exit half-spread | *assumed equal to entry* | **12.2 measured** |
| entry ask inside book | $1,900 | **$3,624** (p25 $1,632) |
| exit bid inside book | *never measured* | **$20,280** |
| depth shortfall @ $2k | 398.4 bps | **≈ 0** |
| round trip, mean @ $2k | 728.9 bps | **66.6** |
| trade coverage | 56.3% | **99.9%** |

An **11× reduction**, from three corrections compounding: sizing at a fixed
dollar amount instead of 1% of ADV, screening to the liquid band, and measuring
the exit rather than assuming it. At a $2,000 order the median entry book
($3,624) *contains the whole order*, so the depth term — 398 of round 3's 728
bps — goes to zero. The cost-sensitivity sweep confirms this is not an artefact
of the crude depth model: net moves by 12 bps across `levels_penalty` in
{1.0, 1.5, 2.0}, because there is no shortfall for the penalty to multiply.

**One correction to section 30, stated plainly.** I flagged round 3's 60-second
look-ahead as a defect. It was real and worth fixing, but the measured
entry-minus-decision half-spread drift is **+0.0 bps at the median**. The bug
had no magnitude. A fixed bug should not be allowed to imply a rescued result.

**This overturns the round-3 verdict.** Section 27 concluded "the binding
constraint is microstructure, not prediction". With costs measured honestly the
constraint is **prediction**: mean gross on the tradeable panel is **−0.19%**
against a 66.6 bps round trip, and no model configuration below finds a positive
sub-population.

## 34. The precision ladder inverted, and round 4's own design caused it

H1′ is not met. 17 eligible cells, none survive BH-FDR with the 72 prior trials
folded into the input. But the interesting result is the shape, not the verdict.

**GBDT, $2,000 order:**

| selectivity | n | gross | mean cost | hit rate |
|---|---|---|---|---|
| top 0.5% | 117 | **−3.32%** | 24.9 bps | 20.5% |
| top 1% | 232 | −2.77% | 25.8 | 25.0% |
| top 2% | 466 | −2.08% | 29.0 | 28.8% |
| top 5% | 1,165 | −1.35% | 35.4 | 33.3% |
| top 10% | 2,330 | −0.73% | 37.6 | 37.2% |
| whole panel | 27,287 | −0.19% | 66.6 | — |

Monotone in the **wrong** direction: more confidence, worse outcome. Round 3's
shallowest cut was 10%, where this reads as mild underperformance (−0.73%). The
0.5% rung is what makes it unmissable. **The deep ladder earned its place by
finding a defect, not an edge.**

Mean cost falls monotonically with selectivity in the same table — 66.6 → 37.6 →
24.9 bps — which names the suspect. `diagnose_inversion.py` tests it by refitting
on the identical funnel and folds with each cause ablated:

| arm | rho(score, cost_hat) | gross @ 0.5% | gross @ 10% |
|---|---|---|---|
| round 4 as run | **−0.476** | −3.32% | −0.73% |
| per-trade label, no cost features | **−0.388** | −2.40% | −0.85% |
| fixed label, with cost features | **+0.366** | +0.42% | +0.09% |
| fixed label, no cost features | **+0.355** | +0.59% | −0.21% |

**The per-trade cost label is the cause; the cost features are not.** Labelling
`ret_c2c > cost_hat` makes a cheap trade mechanically easier to call positive, so
the model learned to rank by *low cost* — and did so even without cost features,
because other features proxy for liquidity. With the label fixed at a constant,
the correlation flips positive (a volatile, expensive name is what clears a fixed
437 bps bar) and the inversion disappears.

Round 4 introduced the per-trade label to "align the objective with the
decision". It did the opposite: it let the model optimise the *hurdle* instead of
the *return*.

**The sign flip is not a strategy.** A model with genuine inverted skill has a
positive bottom tail. This one does not — bottom 0.5% is −2.28%, bottom 2% is
−2.82%. Both ends of the cost distribution lose (cheap names at ~27 bps, dear
ones at ~205 bps) while the panel averages −0.19%. The score sorts on cost, and
cost is non-monotone in return.

**Fixing the bug yields no skill, not positive skill.** The best repaired arm is
+0.59% at the top 0.5%, n=117, CI **[−1.13%, +2.23%]**.

## 35. A null "failed", and the failure is evidence for section 34

The date-shift null returned lift **+0.59%, p=0.045** on the rank arm:
scrambling the features *improved* the result.

Read naively that is a leakage signature. It is not. A null test assumes
destroying information removes an edge — which presumes skill is non-negative.
When skill is *below* zero, destroying the information removes the harm, and the
null "fails" by getting better. That is precisely the model characterised in
section 34.

The other three nulls pass (shuffled-label on both arms, date-shift on the EV
rule). Recorded here rather than quietly re-run, because "one null failed" is
exactly the sort of line that gets dropped from a write-up.

## 36. Round 4 verdict

**H1′ NOT MET.** 17 eligible cells of 33, zero surviving BH-FDR at q=0.05 with
72 prior trials folded in.

| | |
|---|---|
| filter-only baseline, $2k (no model) | −0.87%, CI [−1.12%, −0.64%] |
| best net, any arm | +0.25% — n=45, 39 days, CI [−1.07%, +1.74%], p=0.36 |
| best *eligible* net | −0.67% (EV rule, $2k, n=2,373) |
| measured round trip @ $2k | **66.6 bps** |
| panel gross | **−0.19%** |

Three things are now settled that were not before.

1. **Cost is not the binding constraint.** 66.6 bps at a $2,000 order, robust to
   a doubled depth penalty. Round 3's 728.9 bps figure was inflated ~11x by
   order-sizing, population and an unmeasured exit.
2. **Prediction is the binding constraint.** Panel gross is −0.19%; the honest
   arms are flat; the best-repaired configuration's CI spans zero.
3. **The filter is not a free lunch.** Trading every tradeable candidate loses
   0.87% net per trade, and the tradeability screen keeps the *worse* half by
   gross (−0.19% kept vs −0.08% removed) — it buys cheap execution, not alpha.

The liquid band is cheap enough to trade and has no edge to trade on. That is a
different answer from round 3's, arrived at with better data, and it closes the
cost hypothesis rather than the study.

## 37. Round 4 bug ledger

Ten, of which five were the same latent defect in five places.

| Where | Bug | Found by |
|---|---|---|
| `fetch_{intraday,premarket,edgar,finra,xbrl}` | cache keyed on file existence, not symbol coverage — would have reported "cached" while silently serving round-3 coverage to 436 EDGAR and 277 FINRA symbols | pointing the pipeline at a new funnel |
| `measure_costs_r4` | Alpaca's `end` is an inclusive instant: `[09:34, 09:35)` returns the 09:35:00.000 quote | the new `cost_hat` tripwire, on ~3% of sessions |
| `fetch_xbrl.py` | UTF-8 BOM written by my own `Set-Content -Encoding utf8`, breaking the module | running it |
| `build_edgar_features` | called `assert_no_future_filings(candidates, filings)`; the signature is `(filings, decision)` | reading before running |
| `run_round4_study` | `walk_forward_folds` raises `ValueError`, not `AssertionError` — an under-length panel killed the whole run | smoke test on partial data |
| `run_round4_study` | fold construction fatal rather than adaptive | smoke test |
| **study design** | **per-trade cost label taught the model to minimise the hurdle, not maximise the return** | **the 0.5% rung of the precision ladder** |

The last is the one worth remembering. It was introduced *this round*, as an
improvement, and it was only visible because the ladder went four times deeper
than round 3's shallowest cut.

---

# Round 5 — the cascade, and the screen that was wrong all along

## 38. The gap screen was selecting the losing tail

Rounds 1–3 screened `gap ≥ +10%`; round 4 screened `gap ≥ +3%`. The rule came
from round 1, justified as selecting "in play" names, and **was never tested
against its own alternative in four rounds.**

Measured across 1,455,390 liquid low-priced symbol-days (ADV ≥ $1M, prior
unadjusted close $0.50–$10, 2016–2026), intraday open→close return is
**monotonically decreasing in the opening gap**:

| gap bucket | n | open→close | hit rate | P(intraday high ≥20%) |
|---|---|---|---|---|
| < −10% | 9,292 | −0.00% | 48.5% | 15.5% |
| **−10% to −3%** | **68,737** | **+0.46%** | **50.9%** | 3.9% |
| −3% to 0% | 702,143 | −0.05% | 46.2% | 0.7% |
| 0% to +3% | 586,182 | −0.14% | 44.2% | 0.7% |
| +3% to +10% | 77,386 | −0.43% | 41.0% | 4.0% |
| +10% to +25% | 8,831 | −1.07% | 37.7% | 15.7% |
| > +25% | 2,819 | **−3.68%** | 31.6% | 32.8% |

A one-sided `gap ≥ X` screen selects the **losing tail**. Rounds 1–3 bought the
−1.07% bucket; round 4 bought −0.43% and −1.07%. **Four rounds of negative gross
are substantially explained by the screen itself**, not by the model, the
features or the cost bar.

The user's original observation was right and its interpretation was wrong: names
that explode *are* concentrated in the big gappers (P(high ≥20%) rises to 32.8%
above +25%), but so are the losses, and the mean is deeply negative. The right
tail is real; it is bought at a price that exceeds it.

The gap-**down** bucket is the first positive-gross pool the project has
produced: day-clustered **+0.461%, CI [+0.146%, +0.766%]**, 10 of 11 years
positive, robust across seven cut variants, stronger excluding 2020. This is
short-term reversal (Lehmann 1990; Lo & MacKinlay 1990; Nagel 2012), and buying
oversold names is long-only, so it respects the standing constraint.

## 39. Gate R: the edge survives to 09:35, but not past its own cost

`open → close` is a daily-bar proxy. The study enters at 09:35. Gate R measured
the difference before any modelling, first on a 264-session probe (**FAIL on
power**: +0.37%, CI [−0.39%, +1.06%]), then — under a single pre-declared
escalation, with the threshold and entry time fixed in writing beforehand — on
all 2,605 sessions and 19.1M minute rows:

| entry | n | mean | CI | p(>0) | hit | fill |
|---|---|---|---|---|---|---|
| 09:31 | 50,360 | +0.30% | [−0.03%, +0.64%] | 0.043 | 49.7% | 75.6% |
| **09:35** | **51,361** | **+0.31%** | **[−0.03%, +0.63%]** | **0.0295** | 49.6% | 77.1% |
| 09:45 | 50,377 | +0.28% | [−0.05%, +0.61%] | 0.047 | 49.3% | 75.6% |
| 10:00 | 50,424 | +0.17% | [−0.13%, +0.44%] | 0.113 | 48.6% | 75.7% |
| 11:00 | 47,046 | +0.04% | [−0.17%, +0.27%] | 0.346 | 47.8% | 70.6% |

**GATE R FAILS.** The point estimate clears +0.20%; the CI lower bound is
−0.03% and does not exclude zero. Round 5 stops, as Amendment 1 required.

Three findings, on the best-powered measurement in the project:

1. **The daily proxy overstated the edge by a third** — +0.46% open→close
   against **+0.31%** actual at 09:35. This is why §4 promised the proxy would
   never be used as a result.
2. **The probe's sharp intraday decay was noise.** On full data the edge is flat
   through 09:45 and only fades after 10:00.
3. **The economics fail independently of the statistics.** 31 bps gross against
   a round trip measured at **66.6 bps** (round 4, ADV ≥ $5M — and this pool
   reaches down to $1M, so dearer). The edge is about half what it must be.

No cascade was fitted on this pool. `model/cascade.py` — staged selection with
per-stage feature sets, lambdarank grouped by session, and quantile regression
on the right tail — is built and tested but deliberately unused: hunting for a
profitable subset of a population whose gross is half its cost is exactly the
exercise rounds 1–4 already performed.

## 40. Holding longer does not help, and the control explains why

Every round held one intraday session, a choice fixed in round 1 and never
varied. Since cost is paid once per round trip, a longer hold amortises it — so
if 1-day reversal is 31 bps and 5-day reversal were 150 bps, the second would be
tradable on the same cost base.

It is not. Reversal is a one-day effect and decays immediately:

| pool | h=1 | h=2 | h=3 | h=5 | h=20 |
|---|---|---|---|---|---|
| reversal ADV≥$1M | **+0.42%** (p=0.010) | +0.26% | +0.19% | +0.38% | +0.73% |
| reversal deep | **+0.52%** (p=0.019) | −0.26% | −0.36% | −0.43% | −0.44% |
| reversal ADV≥$5M | +0.13% | +0.06% | −0.09% | −0.14% | +0.18% |

Extending the hold multiplies variance (sd 0.127 → 0.584 from h=1 to h=20)
without adding mean. Net is negative at every horizon on every reversal pool.

**The trap in this table, named explicitly.** Two rows clear cost with a CI
excluding zero, and **both are controls at 20 days**:

| pool | h | mean | CI | net |
|---|---|---|---|---|
| CONTROL gap-up | 20 | +1.37% | [+0.12%, +2.65%] | +0.70% |
| **CONTROL no screen** | 20 | **+1.08%** | [+0.67%, +1.51%] | +0.42% |

The second applies **no selection whatsoever** — 650,865 symbol-days, p < 0.0001.
That is equity drift, roughly +0.05%/day, not alpha. Reporting it as a finding
would be the single most misleading thing in this log, which is why it is here.

It reframes the whole study: an intraday strategy in this universe is not
competing against zero. It competes against **+0.05%/day available for free**,
at one round trip per month rather than one per day.

## 41. Features that turned out not to matter

**Insider and institutional flow.** `etl.edgar` filters Forms 3/4/144 out
deliberately — correctly, for a *supply* question — so 738,623 filings sat
unused for four rounds: 568,896 Form 4, 73,880 SC 13G, 22,580 SC 13D. Insider
buying is among the better-documented small-cap anomalies (Lakonishok & Lee
2001; Cohen, Malloy & Pomorski 2012), so this looked like the largest untapped
signal in the cache.

Built as `etl/filings_flow.py` with counts normalised against each symbol's own
trailing rate (a firm with twelve insiders files twelve Form 4s per board
decision; the *departure* is the signal). Coverage 95.5%; an insider transaction
in the trailing 30 days on 39.8% of candidate-days.

Measured univariately on the round-4 panel:

| feature | q1 → q5 gross | spread | ρ |
|---|---|---|---|
| `passive_365d` | −0.38% → −0.01% | +0.37pp | +0.60 |
| `insider_txn_90d` | −0.23% → −0.01% | +0.22pp | **+0.90** |
| `activist_365d` | +0.17% → −0.33% | −0.50pp | −0.50 |

`insider_txn_90d` is almost perfectly monotone across quintiles, and the entire
spread is 22 bps — an order of magnitude too small, with every top-quintile CI
including zero. This was *in-sample over the whole panel*, so it is an upper
bound. Recorded as a negative because a monotone ρ=+0.90 is exactly the sort of
diagnostic that gets published as a signal when its magnitude is ignored.

**Market regime.** No index symbol exists in the daily cache — the universe
fetch was microcap-focused — so no regime feature was available in four rounds.
`features/market_regime.py` builds the factor from the studied cross-section
instead, which for this segment is the better benchmark anyway, and derives
`gap_vs_market` to separate an idiosyncratic gap-down from a systematic one.
Built and tested; unused, because Gate R stopped the round before stage 2.

## 42. Round 5 bug ledger

| Where | Bug | Found by |
|---|---|---|
| `filings_flow` | counts returned NaN rather than 0 for a covered symbol with no prior filings — "no insider filed" is an observation, not an absence of one | its own tests |
| `filings_flow` | ratio-to-baseline damped small-count symbols hardest, exactly where a burst is most informative; replaced with a Poisson-style standardised excess | its own tests |
| `cascade` | feature validation ran *after* the row-count guard, so a mistyped column surfaced as a thin-fold warning instead of an error | its own tests |
| test fixtures (×3) | two panels sharing a start date, an undersized fold, and index arithmetic that mismatched fits to folds | running them |

None reached a result. Worth recording only because all four were caught by
tests written before the code was run on real data.

---

# Round 6 — the cascade, at retail size

## 43. Gate C: the reversal pool's own cost bar, measured

Round 5 extrapolated this pool's cost from round 4's ADV ≥ $5M measurement. It
is now measured directly: 2,604 sessions, 66,598 symbol-days, 90–94% window
coverage.

| | median | p90 |
|---|---|---|
| decision half-spread | 41.2 bps | 166.7 |
| entry half-spread | 40.2 bps | 168.2 |
| exit half-spread | 19.5 bps | 84.3 |
| entry ask inside book | **$2,120** | p25 $880 |
| exit bid inside book | $8,130 | p25 $1,850 |

**The spread-only round trip is 59.7 bps**, against +27.8% ÷ 100 = **+0.278%**
unconditional gross — so the pool is roughly 2× underwater before depth, before
the model, at any order size. Gate C passes on its own terms (median at $2,000
is below 150 bps) but the number that matters is the floor, not the gate.

The entry book at $2,120 median explains round 5's tradeability problem exactly:
a $2,000 order barely fits at the median and fails outright at p25.

## 44. Small order sizes work, directionally

Amendment 1 predicted cost near 45 bps at $500 once depth vanishes; measured is
59.7. The direction was right and the magnitude optimistic.

Net improves monotonically as size falls, exactly as depth-scaling predicts:

| order | flat GBDT best net | cost |
|---|---|---|
| $250 | **+0.04%** | 89.7 bps |
| $500 | **+0.15%** | 98.0 |
| $1,000 | −0.54% | 106.7 |
| $2,000 | −0.80% | 136.9 |
| $5,000 | −1.26% | 198.8 |
| $10,000 | −2.45% | 296.5 |

**One Amendment 1 claim is retracted.** It asserted the edge sits in names too
thin to buy, from partial data showing gross falling +0.244% → +0.034% under the
tradeability filter. On the full measurement, kept gross is **+0.29% against
removed +0.27%** — indistinguishable. That gap was noise, and the retraction is
recorded because the claim shaped the round's design.

## 45. The cascade concentrates — the first model result in six rounds

| order | cascade gross | CI low | vs pool (+0.28%) |
|---|---|---|---|
| $250 | **+1.15%** | **+0.54%** | **4.1×** |
| $500 | **+1.17%** | **+0.38%** | 4.2× |
| $1,000 | +0.75% | +0.07% | 2.7× |

**This is the first time in six rounds that a model has produced gross with a
confidence interval excluding zero, and it does so at three order sizes.** The
pre-registration asked for 3–5× concentration and called it "strong but not
absurd"; it arrived at 4.1×.

The required ablation favours staging: the cascade beats a flat single-stage
GBDT over the union of all features on gross at every order size, at ~4.6%
retention against the flat model's 1–2%.

## 46. And it fails for a nameable reason: it buys expensive names

| $250 | gross | cost | net |
|---|---|---|---|
| cascade | +1.15% | **127.5 bps** | −0.12% |
| flat GBDT top-1% | +0.95% | 91.5 | +0.04% |

Stage 3 uses a quantile objective on the right tail, and high-upside names carry
wide spreads. Concentrating the tail concentrates costly execution — round 4's
pathology reached by a different route, and this time via a deliberate design
choice rather than a bug.

`cascade_capped` tested the fix directly: drop candidates whose 09:34 `cost_hat`
exceeds the pool median, **as a filter, not a label** (round 4's failure came
from labelling `ret > cost_hat`, which let the model optimise the hurdle).

| $250 | gross | cost | net | net CI |
|---|---|---|---|---|
| cascade | +1.15% | 127.5 | −0.12% | [−0.75%, …] |
| **cascade_capped** | **+0.67%** | **60.0** | **+0.07%** | **[−0.72%, …]** |

Cost fell 53%; gross fell 42%; net moved 19 bps, onto zero. At $500 the cap is
actively harmful (gross +0.12%, net −0.47%).

**60 bps is a structural floor.** `cascade_capped` pays 60.0 bps against a
measured spread-only round trip of 59.7. The cap has already driven cost to the
spread and no further — there is nothing left to remove without removing the
trade. Profit requires gross reliably above ~60 bps, and the best confirmed
gross is +0.67% with a CI including zero.

## 47. Round 6 verdict

**H1‴ NOT MET.** 30 eligible cells, none surviving BH-FDR against 96 prior
trials.

| | |
|---|---|
| pool gross | +0.28% |
| best cascade gross | **+1.15%**, CI [+0.54%, …] — 4.1× concentration |
| measured spread floor | **59.7 bps** |
| best net, any arm | **+0.15%** (flat, $500, top-2%), CI [−1.25%, +1.58%] |
| next best | +0.07% (cascade_capped $250), +0.04% (flat $250) |

**The honest summary: six rounds have moved this from −4.5% per trade to
breakeven.** Round 1's sealed holdout was −4.48%; round 6's best cells sit at
0.00% ± 1.2pp. That is real progress and it is not profit. Every configuration
at or above zero has a confidence interval roughly ±1.2pp wide, which is another
way of saying the result is indistinguishable from doing nothing.

What is now established rather than assumed:

1. **The pool is right.** Gap-downs, not gap-ups — established in round 5 on
   1.46M symbol-days and confirmed here: this population's gross is positive.
2. **The model works.** The cascade concentrates 4.1× with a gross CI excluding
   zero, and beats a flat model at every size.
3. **The floor is the spread.** 59.7 bps, structural, already reached by the
   cost cap. Not reducible by better selection.
4. **The gap is roughly 2×.** Reliable gross of ~60 bps is needed; confirmed
   gross is ~67 bps with a CI touching zero.

## 48. Round 6 bug ledger

| Where | Bug | Found by |
|---|---|---|
| `run_round6_study` | market factor built from the **candidate set**, not the universe — circular, and at 26 names/session it blanked 2,013 of 2,605 sessions, leaving regime features almost entirely null while appearing computed | a load-time check before the run |
| `run_round6_study` | `feature_sets()` ran **before** `attach_costs()`, so every cost feature was filtered out as non-existent and stage 3 had the identical 65 features as stage 2 — a no-op stage reported as a real one | the same check |
| tooling | `Set-Content -Encoding ascii` mangled em-dashes to `???` | running it |

The first two would not have raised and would have produced a plausible result.
Both were caught by loading the study's inputs and printing feature counts
before running it — a habit worth keeping.

## 49. The portfolio view: why per-trade expectancy was the wrong object

Six rounds reported the mean return **per trade**. An account earns a return
**per day**, equal-weighted across what it held, and flat on days it sat out.
§47 named portfolio construction as the remaining lever, on the argument that
thirty small positions a day would diversify away idiosyncratic variance.

**That argument was wrong, and the measurement says so in two ways.**

### There is no diversification available

| arm | avg positions per active day | % of days active |
|---|---|---|
| cascade @ $250 | **2.10** | 39.4% |
| cascade_capped @ $250 | 1.78 | 22.4% |
| flat top-1% @ $250 | 2.43 | 6.8% |
| flat top-2% @ $250 | 2.56 | 12.9% |
| all_tradeable @ $250 | 18.04 | 100% |

The cascade keeps 4.6% of a pool averaging 26 names a session — about **1.2
names a day**, and ~2 once folds are pooled. Position caps of 5, 10 and 20 make
no difference because the cap never binds. The thirty-position portfolio I
proposed does not exist at any selectivity that produces an edge; the only arm
holding meaningful numbers is `all_tradeable`, which has no edge at all.

### Volatility drag destroys the edge even where the mean is positive

| arm @ $500 | mean daily | Sharpe | **compounded** | max DD |
|---|---|---|---|---|
| flat top-2% | **+0.08%** | 0.31 | **−15.2%** | −98.0% |
| flat top-1% | +0.00% | 0.02 | −60.5% | −95.1% |
| flat top-5% | +0.01% | 0.03 | −98.6% | −99.9% |
| cascade | −0.16% | −0.26 | −100% | −100% |
| all_tradeable | −0.90% | −3.29 | **−100%** | −100% |

The best arm has a **positive arithmetic mean and compounds to −15%.** Implied
daily σ is ≈4.2% against an 8 bps edge, so σ²/2 exceeds μ and the geometric
return is negative. An account compounds; it does not earn the arithmetic mean.

**This is a stronger negative than "the edge does not clear costs".** It says
that at the one point where costs *are* covered, the variance is roughly 50× the
edge, so the equity curve still falls. No amount of selection fixes that, because
selecting harder shrinks positions per day and raises variance further — visible
above as Sharpe falling from 0.31 at top-2% to 0.02 at top-1%.

**No configuration has a daily mean whose CI excludes zero.** Best is flat top-2%
at $500: +0.08%/day, CI [−0.07%, …].

### What this settles

The per-trade framing used for six rounds was not merely incomplete, it was
measuring something no account could have collected. The correct object — daily
compounded return — is negative everywhere, including in the cells where
per-trade net was positive.

Recorded as exploratory: it changes no round-6 verdict, since diversification
alters variance rather than mean, and H1‴ was already not met.

---

# Forensic study — the ten most recent +50% movers

## 50. Method: case studies, disciplined by base rates

Six rounds averaged over tens of thousands of candidates. Averaging is exactly
what destroys a precursor present in 2% of cases, so this inverts the method:
read ten explosions closely, then hand every pattern back to the population to be
measured against its own base rate.

Sample: the ten most recent close-to-close ≥+50% events in the penny universe,
selected mechanically (most recent N meeting a fixed condition) so it cannot
drift toward whichever names tell the nicest story. Universe: **719,863
symbol-days, 2025-02 to 2026-08, 1,213 events, base rate 0.169%.**

The split is 5 gap-driven and 5 intraday runners, and ADV spans **$6,795 to
$21M** — a 3,100× range within ten names.

## 51. The blind spot: the biggest movers never entered any pool

| symbol | gap | open→high |
|---|---|---|
| CLRO | +217% | +6.9% |
| MB | +281% | +38% |
| **YJ** | **+12%** | **+912%** |
| **XHLD** | **+1.6%** | **+569%** |
| **ZYBT** | **−1.6%** | **+240%** |

Every round screened on the **opening gap** and entered at 09:35. The intraday
runners are invisible to that: `gap ≥ +3%` excludes them and the round-5 reversal
screen `gap ∈ [−10%, −3%]` excludes them too. The caches prove it — EDGAR and
FINRA covered all twelve names checked, but minute and premarket data covered
only the seven that happened to be candidates. **The names that exploded are the
ones no screen ever looked at.**

## 52. The case study got it backwards, and the base rates caught it

Reading three dossiers, volume had *contracted* before every one — XHLD 0.29×,
YJ 0.49×, CLRO 0.35× (5-day against 20-day). That looked like a counterintuitive
finding and it was written up as one.

Measured across the universe:

| | lift on P(+50%) | fires on |
|---|---|---|
| volume **contraction** (5d/20d < 0.5) | **1.19** | 15.2% of days |
| volume **expansion** (5d/20d > 2.0) | **5.57** | 6.5% of days |

**The strongest precursor is the opposite of what three hand-read dossiers
suggested.** Generalising from three of ten produced a sign error. This is the
single clearest demonstration in the project of why the base-rate step is not
optional.

The same discipline caught a second error: `filing accepted the same morning` has
**lift 5.21 — the second-strongest precursor — and zero hits in the ten.** A pure
case study would have missed it entirely. And ultra-low ADV, flagged from MB's
$6,795, has lift **0.81** — mildly *protective* against a +50% move.

## 53. Predicting explosions and making money are opposite things

Lift measures P(+50% move). It says nothing about expected return, and the two
dissociate completely:

| precursor | n | lift | E[open→close] | CI |
|---|---|---|---|---|
| **ADV under $50k** | 74,379 | **0.81** | **+0.279%** | **[+0.200%, +0.354%]** |
| volume expansion | 46,739 | **5.57** | +0.094% | [−0.087%, +0.286%] |
| fell >40% in 20d | 29,273 | 4.50 | −0.074% | [−0.350%, +0.205%] |
| prior +50% within 90d | 70,970 | 3.20 | −0.173% | [−0.378%, +0.019%] |
| **filing same morning** | 26,754 | **5.21** | **−0.396%** | **[−0.602%, −0.189%]** |
| **≥3 shelf regs in 180d** | 35,245 | 2.26 | **−0.602%** | **[−0.824%, −0.377%]** |

Unconditional open→close over the universe is **−0.014%** — flat.

**The best explosion predictors lose money from the open.** A same-morning
filing predicts a +50% day 5.2× more often than chance *and* returns −0.40% from
the open, because the move happens in the gap: CLRO gapped +217% and then fell
16% open-to-close. You cannot buy what has already happened.

**The one precursor with a significantly positive open→close return does not
predict explosions at all.** ADV under $50k: +0.279%, CI excluding zero, on
74,379 symbol-days — with lift 0.81.

## 54. Two real vetoes, from mechanism rather than search

Round 1 searched **299 veto sets** and none survived FDR. Two here do, and both
have confidence intervals excluding zero on open-to-close return:

- **≥3 shelf registrations in 180 days → −0.602%** [−0.824%, −0.377%], 35,245
  symbol-days. A company registering shares repeatedly is diluting into any
  strength.
- **Filing accepted the same morning → −0.396%** [−0.602%, −0.189%], 26,754
  symbol-days. The news is in the gap; the open is the exit, not the entry.

The difference from round 1 is that these came from reading what actually
preceded specific explosions, not from enumerating combinations and testing all
of them. A mechanism proposed first and tested second survives correction in a
way that a search does not.

**What they are worth is limited.** The universe they would filter has an
unconditional open→close of −0.014%; removing the worst names lifts that to
perhaps +0.05%, still far under the ~60 bps spread floor measured in round 6.
A veto helps only if the remaining population is tradable, and it is not.

## 55. What the forensic study establishes

1. **Every screen in six rounds was structurally blind to half the movers.**
   The intraday runners never gapped, so no gap-based screen could see them.
   Trading them needs an intraday trigger, which does not exist in this codebase.
2. **Explosion prediction is not the same problem as profit.** The strongest
   predictors of +50% days have significantly *negative* open-to-close returns.
   Six rounds spent optimising the wrong target would still have failed even with
   a perfect explosion predictor.
3. **Two vetoes are real** and were found by mechanism rather than search.
4. **The one positive-return precursor is ADV < $50k** — +0.279% open-to-close,
   CI excluding zero — and it sits in names too thin to trade, which is the same
   wall rounds 5 and 6 hit from the other direction.
5. **Case studies mislead without base rates.** Three of ten dossiers produced a
   sign error on the strongest precursor, and the second-strongest precursor
   appeared in none of the ten.

---

# Hypothesis grid — a swept parameter search, and what it caught

## 56. The grid: 441 cells, 17 families, zero survivors

Four hypothesis families were supplied (reversion, momentum, reverse split, gap)
and seven added (volume shock, quiet-volume accumulation, range compression, new
high/low, consecutive down days, overnight decomposition, and a no-condition
control). Each was swept over its parameters and five holding horizons.

Discipline fixed before running, because a sweep of this size manufactures
winners by arithmetic:

- **calibrate and confirm on disjoint periods** — all cells fitted before
  2026-02-01, re-measured untouched after it;
- **Benjamini-Hochberg over the whole grid**, not over the survivors;
- **costs charged before judging**, on a schedule anchored to the measured
  59.7 bps spread-only round trip for ADV ≥ $1M.

| | |
|---|---|
| cells tested | 441 |
| cells with enough data | 422 |
| in-sample net CI excluding zero | **2** |
| surviving BH-FDR across the grid | **0** |
| confirmed out-of-sample | **0** |

Two of 422 cells is *below* what chance alone would produce at a 5% threshold.
The families as stated — down-A-then-up-B, up-A-continues, gap-A-then-B — are
not profitable at any parameterisation tested, once costs are charged.

## 57. The one candidate, and why it was an artifact

One cell was positive in both windows with intervals excluding zero:

> **volume ≥ 5× its 20-day average while the close moves less than 5%**,
> buy the close, sell the next close

| | n | gross | net | CI |
|---|---|---|---|---|
| in-sample | 836 | +3.71% | +2.51% | [+0.54%, …] |
| out-of-sample | 527 | — | +2.96% | [+0.44%, …] |

It looked strong. The threshold sweep was **monotone** — net −0.42% at 2×,
+0.50% at 3×, +1.75% at 4×, +2.68% at 5×, +5.71% at 7×, +10.37% at 10× — which
is the signature of a real effect rather than a fitted parameter. It was stable
across 2025, 2026, and both halves. The placebo (same liquidity, no condition)
returned −0.97%.

The portfolio view then produced **+31,677% compounded at Sharpe 2.26**, which is
not a number any real microcap strategy produces. That implausibility is what
prompted the audit rather than a write-up.

**Cause one: splits on the exit day.** `is_suspect_split` was applied to the
*signal* day, but the forward return runs to day *t+1* — so a reverse split
occurring on the exit day was being booked as a return. Only **18 of 1,363
rows (1.3%)** were affected, and they carried **58% of the edge**:

| | n | net | CI |
|---|---|---|---|
| as run | 1,363 | +2.68% | [+1.09%, +4.44%] |
| splits excluded from the window | 1,345 | **+1.13%** | **[−0.21%, +2.71%]** |

**Cause two: the detector's own blind spot.** What remained at ≥7× (+2.80%)
dissolved on inspection. Of its ten largest contributors, **five are corporate
actions the detector missed** — it requires an inert session (open→high < 35%),
and a reverse split followed by a genuine move shows 274%, 70%, 56%, 48%, 47%.
The consequence runs backwards through this log too: **the 16.1%
split-contamination rate reported in §50 is a lower bound.**

**Cause three: concentration and structure.** Even taking the residual at face
value, the top **5 trades of 698 produce 55% of the total return**; dropping
them leaves +1.83%, dropping ten leaves +0.99%. The median forward return is
**0.00%** and the hit rate 47%. And **129% of the mean return is the overnight
gap** — the intraday component is negative, so the "signal" is a lottery ticket
on overnight events, which in this universe are largely corporate actions,
halt resumptions and offerings.

## 58. Verdict on the hypothesis families

| family | best parameterisation | net | verdict |
|---|---|---|---|
| A. down A% → up in C days | ret_5d ≤ −40%, H=5 | +0.40% | CI spans zero; −1.83% out-of-sample |
| B. up A% → continues | none positive after cost | — | rejected |
| C. reverse split → rise | — | negative | rejected; splits *fade* −15% from the open |
| D. large gap down → bounce | gap ≤ −30%, H=5 | +7.55% | n=221, CI [−11.3%, …] — noise |
| D. large gap up → continues | none positive | — | rejected |
| E. volume shock + quiet close | vol ≥ 5×, H=1 | +1.13% clean | artifact; see §57 |
| F. range compression | none positive | — | rejected |
| G. new 20-day low / high | none positive | — | rejected |
| H. 3+ consecutive down days | none positive | — | rejected |
| I. overnight vs intraday | — | — | control only |

**No profitable strategy was found.** The single candidate that survived the
statistical filters failed the forensic one, and it failed for reasons that are
specific and checkable rather than a matter of judgement.

The methodological point is the durable one: the candidate passed a monotone
threshold sweep, a placebo, a period split, and a cost-sensitivity test. It was
caught by asking why a Sharpe of 2.26 appeared in a universe where six rounds had
found nothing, and then reading the ten largest trades. **Statistical validation
did not catch it; inspecting the trades did.**

## 59. Contamination revised upward: 20.4%, not 16.1%

Widening the detector to a tight round ratio *without* the inert-session
requirement — the test §57 showed was missing split-then-rally cases — raises the
count:

| detector | events flagged | share of +50% gainers |
|---|---|---|
| strict (round ratio **and** inert session) | 195 | 16.1% |
| wide (tight round ratio only) | 163 | 13.4% |
| **either** | **247** | **20.4%** |

Whole-universe false-positive pressure stays negligible at 0.05%.

The three cohorts behave distinctly, which is itself evidence the split is real
rather than a screening artefact:

| cohort | n | median gap | open→close |
|---|---|---|---|
| strict-flagged | 195 | +131% | **−15.22%** [−17.64%, −12.82%] |
| wide-only | 52 | +152% | **+27.24%** [+10.76%, +47.19%] |
| clean by both | 966 | +20% | **+62.31%** [+55.08%, +72.64%] |

The wide-only group is a consolidation followed by a genuine move — it gaps like
a split and then trades like news. Both flags are kept separately rather than
merged, because the strict one is the conservative estimate and the wide one is
the one to use when excluding rows from a return study, where a false positive
costs far less than a false negative.

**One in five "top gainers" in this universe is a share consolidation.**

## 60. The premarket hypothesis, rejected at the movers' own size

Round 4 tested premarket dollar volume on ADV ≥ $5M names and found it negative.
That did not settle the question, because the thirty largest gainers have a
**median ADV of $548,901** and 60% sit below $1M — a population the $5M floor
structurally excludes. So premarket bars were fetched for the band where the
explosions actually occur: **ADV $100k–$5M, 420,395 symbol-days, 379 sessions,
2,441 tickers, 5.59M premarket bars**, corporate actions excluded on both flags.

| premarket $ / ADV | n | open→close | CI |
|---|---|---|---|
| whole pool | 420,395 | −0.048% | [−0.196%, +0.100%] |
| **<0.01× — did not trade pre-open** | 125,226 | **+0.056%** | [−0.077%, +0.184%] |
| 0.1 – 1× | 49,016 | −0.074% | [−0.283%, +0.147%] |
| 1 – 5× | 2,740 | −0.260% | [−0.873%, +0.350%] |
| **≥20×** | 1,128 | **−2.928%** | **[−4.304%, −1.512%]** |
| **H: ≥1× ADV** | 4,838 | **−0.769%** | **[−1.390%, −0.146%]** |
| ≥1× ADV **and** fell >25% over 20d | 1,059 | **−1.726%** | **[−3.077%, −0.331%]** |

**The hypothesis is rejected, and not for want of power.** The sign is negative
with an interval excluding zero on 420k observations, at both size ranges tested.
The prior-decline interaction — the one condition that might have rescued it —
makes the result *worse*.

The quietest cell is the best one: names that did not trade at all before the
bell returned **+0.056%**, the highest in the panel.

This closes the most visible precursor in the top-30 study. Seventy per cent of
the winners traded at least their whole ADV before the open; across the
population that same condition loses money at every threshold, and loses more the
louder it gets. The signal marks the move as *already priced*, not as beginning.

## 61. The last candidate, and the complete family verdict

`D_gap_down` (gap ≤ −30%, buy the open) was the only family besides the
already-refuted accumulation cell with materially positive gross. It fails the
same way, and more starkly:

| | value |
|---|---|
| n | 342 clean, 206 sessions, 292 tickers |
| mean | +1.51% |
| **median** | **−6.26%** |
| hit rate | 37% |
| after dropping top 3 trades | **−6.11%** |
| after dropping top 5 trades | **−6.90%** |
| out-of-sample | **−6.29%**, CI [−12.96%, +0.78%] |

The mean is one trade: **WW on 2025-05-07 returned +2,294%** — WeightWatchers
around its restructuring, which is a corporate action rather than a trade. Remove
the three largest and the rule loses 6%. A positive mean sitting on a −6.26%
median is not an edge, it is a lottery ticket priced as one.

### Every family, final

| family | best gross | verdict |
|---|---|---|
| E. volume shock + quiet close | +9.90% | artefact — splits on the exit day, §57 |
| D. gap down ≤ −30% | +8.40% | one trade; median −6.26%; OOS −6.29% |
| A. reversion (down A% → up) | +1.31% | CI spans zero; −1.83% out-of-sample |
| H. consecutive down days | +0.32% | gross positive, net −0.67% |
| F. range compression | +0.24% | net negative |
| G. new 20-day low | +0.18% | net negative |
| **I. overnight (buy close, sell open)** | **+0.08%** | **no overnight anomaly here; OOS −0.03%** |
| **J. control (no condition)** | **+0.01%** | the base population is flat gross |
| B. momentum (up A% → continues) | −0.04% | gross negative |
| G. new 20-day high | −0.16% | gross negative |
| E. volume shock (no quiet filter) | −0.69% | gross negative |
| **D. gap up ≥ +5%** | **−2.06%** | **strongly gross-negative — gap-ups fade** |
| **C. reverse split** | **−10.72%** | **splits crater; the supplied hypothesis is backwards** |

Only **77 of 422** cells have positive in-sample gross at all, and the control
shows why: the base population's gross is **+0.011%** against a ~98 bps cost.

Two results here are worth keeping independently of the search:

**The overnight anomaly does not exist in this universe.** Buying the close and
selling the next open — one of the better-documented effects in large-cap equity
— returns +0.076% in-sample and **−0.034% out-of-sample** across 317,873
observations. Whatever compensates overnight risk-bearing elsewhere is absent
here, which is consistent with a universe where overnight events are offerings
and consolidations rather than earnings.

**Two of the four supplied hypotheses are directionally backwards.** Reverse
splits do not rise, they crater (−10.72% gross). Large gap-ups do not continue,
they fade (−2.06% gross). Both are cleanly measured and both are the opposite of
the stated conjecture.

## 62. Final verdict on the search

**No profitable strategy was found**, across 441 swept cells, 17 families, six
research rounds and a forensic study of thirty explosions.

The two candidates that reached the final filter were killed by the same
technique and not by statistics:

| candidate | passed | killed by |
|---|---|---|
| volume ≥5× on a quiet close | monotone thresholds, placebo, period split, cost sensitivity, out-of-sample | 1.3% of rows carrying 58% of the edge; 5 of 10 top trades were missed splits |
| gap ≤ −30% | positive mean in-sample | median −6.26%; one trade (WW +2,294%) carrying the mean |

Both were found by **reading the largest trades**. Neither would have been caught
by any statistical test in this repository, and the accumulation cell in
particular passed every one of them.

That is the transferable result. In a universe where roughly one symbol-day in
1,400 is an unadjusted corporate action, and where returns are this
right-skewed, **trade-level inspection is not a supplement to statistical
validation — it is the binding constraint on what can be believed.**

---

# 63. A profitable strategy — on the short side

## The reversal of framing

Every reliable effect this project found is **negative**: reverse splits −10.72%
gross, gap-ups −2.06%, premarket ≥20× ADV −2.93%, shelf registrations −0.60%,
same-morning filings −0.40%. All were used as *vetoes*, because a long-only
constraint was set early and never revisited. The strongest evidence in the study
pointed one way and the search was pointed the other.

## The strategy

> **Short a suspected reverse split at the open, cover at the close, with a hard
> stop.** Signal: the unadjusted overnight multiple sits on a round consolidation
> ratio. Filter: ADV ≥ $250k. Costs: 150 bps round trip plus 50% annualised
> borrow.

All prices **unadjusted** — what actually printed. This matters enormously; see
§64.

| stop | per trade | CI | worst trade | hit |
|---|---|---|---|---|
| none | +7.53% | [+2.80%, +11.79%] | **−195.40%** | 69% |
| **+30%** | **+6.32%** | **[+2.93%, +9.55%]** | **−31.70%** | 62% |
| +50% | +6.43% | [+2.60%, +10.17%] | −51.70% | 65% |
| +100% | +6.87% | [+2.31%, +11.26%] | −101.70% | 68% |

Period stability, independently measured:

| | n | per trade | CI |
|---|---|---|---|
| in-sample (< 2026-02) | 118 | +7.02% | [+0.26%, +12.83%] |
| out-of-sample (≥ 2026-02) | 83 | **+8.25%** | [+1.13%, +14.81%] |

**Portfolio** — 20% position sizing, +50% stop, equal weight, flat when nothing
fires, compounded:

| | |
|---|---|
| mean daily | **+0.536%** |
| Sharpe | **2.56** |
| compounded, 379 sessions | **+516.5%** |
| max drawdown | **−31.5%** |
| worst day | −10.3% |
| days active | 40% |
| trades | 201 (~11/month) |

## Why this survives what killed the others

The two long candidates died on concentration and on tail-driven means. This one
inverts both:

- **median +9.41% exceeds mean +7.53%** — the distribution is *left*-skewed, so
  the mean is dragged down by losses rather than propped up by a few wins. Both
  failed long candidates had medians far below their means (0.00% and −6.26%);
- **dropping the best 10 trades still leaves +4.85%**;
- it holds at every liquidity floor (no floor +7.04%, ≥$250k +7.53%, ≥$1M +7.19%)
  and at every borrow assumption from 20% to 150% annualised;
- the control — short every liquid day — returns **−1.00%**, so this is not
  "shorting works".

There is also a literature: post-reverse-split underperformance is documented
(Desai & Jain 1997; Kim, Klein & Rosenfeld 2008). The mechanism is not mysterious
— a consolidation is usually a listing-compliance action by a company whose price
has been falling, and the decline resumes.

## What stands between this and a live trade

Stated plainly, because these are not minor:

1. **Borrow availability is unmeasured, and it is the binding constraint.**
   Post-reverse-split microcaps are among the hardest-to-borrow securities in the
   market. IBKR's borrow feed is unreachable from this network and no free
   substitute exists, so every number above assumes a locate that may not exist.
   **A meaningful fraction of these 201 trades were probably unborrowable.**
2. **The signal is my own heuristic, not a corporate-actions feed.** Trading it
   means trading its classification errors, and the false positives — genuine
   gappers on a round multiple — are precisely the names that squeeze.
3. **The stop assumes a fill.** In a post-split squeeze the tape can gap through
   +30% without trading there.
4. **n = 201**, and the detector was developed on the same period it is measured
   over. The in/out split mitigates but does not remove this.
5. **Unbounded loss without the stop.** Five trades exceeded −100%; the strategy
   is only survivable with both a hard stop and ≤20% position sizing.

**Verdict: profitable in the data, conditional on borrow.** The honest next step
is not more backtesting — it is checking locate availability and rates with a
broker on the next twenty flagged names, forward, before any capital moves.

## 64. The measurement error that nearly inverted the answer

The first short test mixed price series: the split flag was computed on
**unadjusted** prices while the return was computed on **adjusted** ones. Reading
the extreme trades exposed it:

| symbol | unadjusted open | adjusted open | volume |
|---|---|---|---|
| WOK | $4.80 | **$4,800,000** | **116 shares** |
| GRI | $9.33 | **$261.24** | 435,282 |
| MSS | $3.79 | **$189.50** | 1,869,048 |
| ALDFW | — | — | **9,719 shares** |

The apparent *gains* were adjusted/unadjusted mismatches and untradeable
near-zero-volume names; the apparent *losses* (SPRB, PROK, ELBM, SUNE — 24M to
343M shares) were real squeezes. Recomputing everything on unadjusted prices —
what actually printed — the effect survives at +7.53%, but the number arrived at
honestly rather than by accident.

**Third time this session that reading the largest trades changed the
conclusion**, after §57 and §61. In this universe that check is not optional.

---

# 65. The four supplied hypotheses, calibrated on the LONG side

## Why this section exists

§63 found a profitable short and treated the question as answered. It was not.
The four hypotheses supplied were all phrased **long** -- "raise by B",
"continues to raise", "raise by A" -- and answering with a short strategy answers
a different question. Worse, the 441-cell grid had tested them as *mean* effects,
but they were stated as **thresholds**. "Rise by B in the next C days" asks
whether the price *touches* +B%, not where it closes on day C. Those are
different questions, and the threshold version is the tradable one: a limit order
at +B% with a time stop captures a touch.

So every family was re-run as a **long triple barrier** -- buy, sell on a +B%
limit, time-stop at C sessions -- across 3,300 cells. Where both barriers fall in
one session the **stop is assumed to fill first**, because daily bars cannot
order intraday events and the favourable assumption manufactures edge out of
missing information.

## The result

| family | best condition | h | tgt | net | 95% CI | median | n |
|---|---|---|---|---|---|---|---|
| **H11 price band** | **down>=25%/5d, $0.50-$2** | 10 | 50% | **+3.87%** | **[+2.75%, +4.96%]** | -1.42% | 3,897 |
| H10 overnight | vol shock >=5x | - | - | +0.14% | [-0.81%, +1.36%] | -1.50% | 6,022 |
| H5 capitulation | down>=25%/5d, vol>=3x, at 20d low | 1 | 50% | +0.43% | [-1.21%, +2.35%] | -2.81% | 252 |
| H2 momentum | up>=100%/3d | 1 | 50% | -0.14% | [-1.37%, +1.12%] | -10.45% | 877 |
| H1 reversion | down>=25%/3d | 1 | 50% | -0.57% | [-0.96%, -0.21%] | -2.94% | 7,834 |
| H7 quiet decline | down>=35%/10d, low volume | 1 | 50% | -0.91% | [-1.28%, -0.56%] | -1.88% | 8,873 |
| **CONTROL** | no condition | 1 | 50% | **-0.98%** | [-1.15%, -0.79%] | -1.23% | 489,756 |
| H9 dip in trend | up>=20%/20d, down>=5%/3d | 1 | 50% | -1.03% | [-1.34%, -0.72%] | -1.63% | 13,169 |
| H6 confirmation | down>=25%/20d then a green day | 10 | 50% | -1.05% | [-2.08%, +0.03%] | -3.36% | 19,967 |
| H4 gap up | overnight >= +5% | 1 | 50% | -1.09% | [-1.51%, -0.54%] | -2.57% | 15,357 |
| H4 gap down | overnight <= -10% | 1 | 50% | -1.42% | [-1.82%, -1.00%] | -3.84% | 4,715 |
| **H3 split (long)** | reverse split | 1 | 50% | **-4.67%** | [-6.26%, -3.10%] | -10.90% | 306 |

**Direct answers to the four supplied hypotheses:**

1. *down by A -> rise by B* — **true, but only inside a price band.**
   Unconditional reversion is −0.57%; the same rule restricted to $0.50–$2.00 is
   +3.87%. The band is the hypothesis, not the drop.
2. *up by A -> continues to rise* — **false.** −0.14% at best, and the median of
   the strongest momentum cell is **−10.45%**. Buying strength here loses.
3. *reverse split -> rise* — **false, and inverted.** −4.67% long, which is §63's
   short read from the other side. Of the four, this one is exactly backwards.
4. *large gap -> rise* — **false in both directions.** Gap up −1.09%, gap down
   −1.42%, both worse than the unconditional control.

Three cells survived Benjamini-Hochberg over 3,006 and confirmed on disjoint
periods. All three were H11 at different parameters.

## Then the account lost money anyway

Per-trade expectancy positive, compounded return **−82%**. The §49/§56 trap
again, and it took three corrections to reach the real number:

| step | n | per trade | CI |
|---|---|---|---|
| as measured | 3,923 | +4.16% | [+3.03%, +5.31%] |
| less reverse splits **inside the holding window** | 3,865 | +3.49% | [+2.39%, +4.60%] |
| less **overlapping** entries on the same symbol | **1,443** | **+2.34%** | [+0.84%, +3.86%] |

The overlap correction is the one a day-clustered bootstrap cannot make: AZI
entered 2026-06-17, 06-23 and 06-24 all reach the same $3.50 high — one event
booked three times. Clustering is on the *entry* date, so it never sees it.
**2,422 of 3,865 entries were re-entries into a position already open.**

And then, event-driven with real capital:

| slots | Sharpe | compounded | maxDD |
|---|---|---|---|
| 3 | −1.32 | **−82.2%** | −82.6% |
| 5 | −0.66 | −57.0% | −64.2% |
| 10 | −0.33 | −27.6% | −41.5% |
| 20 | +0.75 | +31.7% | −33.3% |

σ²/2 exceeded μ at every concentrated setting. **Diversification is not a
refinement here, it is the difference between +32% and −82%.**

## Calibrating on the account, which is where it matters

Sweeping drop × hold × target × slots on the *account* rather than on per-trade
expectancy, the surface is monotone and the peak is a plateau:

> **drop ≥40%, hold 10 sessions, target +50%, ~30 slots**

| drop | per trade | Sharpe | compounded |
|---|---|---|---|
| ≥30% | +3.40% | 2.01 | +83.0% |
| ≥35% | +4.37% | 2.61 | +102.8% |
| **≥40%** | **+5.56%** | **3.06** | **+109.4%** |
| ≥45% | +6.96% | 3.10 | +102.9% |
| ≥50% | +3.48% | 1.36 | +27.4% (n=217) |

Smooth degradation on both sides — the signature of an effect rather than of a
fitted spike.

## The audit

| test | result |
|---|---|
| entry at t+1 **open**, not t's close | +5.56% vs +5.89% — **not a closing-print artifact** |
| in-sample | +4.52% CI [+0.21%, +8.94%], Sharpe 2.60 |
| out-of-sample | **+6.76%** CI [+2.18%, +11.44%], Sharpe 4.23 |
| ADV ≥ $2M | +4.60% CI [+0.41%, +8.99%], Sharpe 1.94 |
| **survivorship**, delisted at last print | **+4.97%** CI [+1.74%, +8.14%] |
| **survivorship**, delisted = −100% | **+3.59%** CI [+0.03%, +7.02%] |
| concentration, drop best 50 of 421 | **−0.35%** |
| mean vs median | +5.56% vs −0.05% — **right-tailed** |

The largest winners read as real events in liquid names, not artifacts: **WOLF**
($133M ADV), **BYND**, **GRML** ($157M), **CIRC**, **DVLT**, **HUBC**, **AIFF**,
**BIYA** — ADVs of $20M to $157M. This is the check that killed §57, §61 and the
first §63 run, and it is the first time a long candidate has passed it.

**Survivorship was a real hole and is now closed.** 16 trades had no tenth
forward bar because the stock stopped trading; `notna()` was silently dropping
them, and a delisting is not a neutral outcome for a long. Exited at the last
print they average **−20.12%**. The result survives including them, and survives
assuming a total loss on every one.

## Verdict

**A profitable long exists.** Buy a $0.50–$2.00 stock with ADV ≥ $250k that has
fallen ≥40% in five sessions; enter at the next open; sell on a +50% limit or at
the close of the tenth session; run ~30 concurrent positions.

**+4.97% per trade, CI [+1.74%, +8.14%], n=429. Sharpe ~3.0, compounded +109%,
max drawdown −10.2%. Out-of-sample better than in-sample.**

**Its weakness is convexity, and it is structural rather than a defect.** The
target hits 27% of the time; the median trade is flat; dropping the best 50 of
421 turns the whole thing negative. That is what a capped-upside lottery payoff
looks like, and it dictates the operating rules: **take every signal**, never
size up, and hold ~30 positions. Cherry-picking this strategy destroys it —
the opposite of how a discretionary trader would be tempted to run it.

**What is not established:** the parameters were chosen by looking at this data,
and the out-of-sample window is six months. The honest confirmation is forward,
not another backtest.

# 66. Standing lessons added

- **Test the hypothesis in the form it was stated.** "Rise by B" is a barrier
  question. Testing it as a mean effect answered a different question and hid a
  real effect for a full round.
- **Day-clustered bootstraps do not fix overlapping holding periods.** Clustering
  is on the entry date; two entries days apart share a forward window. 63% of
  entries here were re-entries into an already-open position.
- **Exclude corporate actions over the whole holding window, not just the signal
  day.** Repeat of §57's bug at a longer horizon.
- **A truncated forward window is not missing data, it is a delisting.**
  `notna()` drops exactly the worst outcomes for a long.
- **Calibrate on the account, not on per-trade expectancy.** The same rule ran
  from −82% to +110% purely on the number of concurrent positions.

---

# 67. The first live screen, and a correction to §63

## The screen

`scripts/screen_today.py` fetches the last ~48 sessions for all 5,795 active
listed symbols, raw **and** adjusted, and applies both rules to the most recent
completed session. Run on **2026-08-13**: 8 long, 2 short.

| long | close | 5d | ADV20 | what it actually is |
|---|---|---|---|---|
| TENX | $1.83 | −86.3% | $27.7M | Phase 3 LEVEL trial failed; **$117M cash, no debt** |
| ACH | $1.10 | −62.3% | $2.6M | Accendra Health (NYSE) |
| RETO | $1.20 | −57.9% | $0.5M | ReTo Eco-Solutions — near-zero volume until 08-10 |
| CRIS | $1.87 | −57.6% | $0.7M | Curis Inc |
| LESL | $0.77 | −49.3% | $2.4M | Leslie's Inc — −41.7% on 08-13 alone |
| MEDS | $1.49 | −43.6% | $0.4M | DataMeds AI — 179k shares, thin |
| GLND | $1.25 | −43.4% | $2.5M | Greenland Energy — −44.5% on 14M shares |
| WFF | $1.63 | −40.2% | $4.3M | WF Holding — 08-07 ran $1.72→$10.53→$2.30 on 31M shares |

Rate check: 8 in one session against a studied rate of ~1.1/session. Elevated,
consistent with a wave of small-cap breakdowns that week rather than a bug.

**Two tickers had to be identified before they could be trusted.** ACH fell
**exactly −50.00%**, the signature of an ADR ratio change — Aluminum Corp of
China's old ticker. Chalco delisted its ADRs from the NYSE in 2022 and the
symbol was **reassigned to Accendra Health**. Real move, different company. And
TENX's −89.8% is close to a 10:1 factor, but the session had a 27% intraday
range (a unit change is inert), the vendor applied no adjustment, and the
catalyst is documented. Both survive; neither was obvious from the tape.

## The correction: borrow data existed, in a file already cached

§63 stated that borrow availability "is NOT modelled ... no free substitute
exists". **That was wrong.** Alpaca's asset registry — cached since round 0 at
`data/research_penny/asset_registry.parquet` — carries **`shortable`** and
**`easy_to_borrow`** per symbol. The claim was never checked against a file
already on disk.

Both of 2026-08-13's short candidates are flagged **not shortable**:

| short | open | prev close | ratio | shortable |
|---|---|---|---|---|
| KIDZ | $4.28 | $0.27 | 15.7x | **False** |
| BNRG | $3.10 | $0.47 | 6.6x | **False** |

Both are genuine consolidations (KIDZ 15:1, volume collapsed 1.83M → 217k
shares; BNRG 6:1). Had they been shortable: KIDZ +3.5% gross, BNRG **−11.6%**
gross — BNRG kept running after the split, which is exactly the false-positive
mode §63 named.

## What the flag does to the short strategy

Of 342 historical short signals (ADV ≥$250k), only **55 (16.1%)** are currently
flagged shortable. Splitting the measurement on that flag:

| | n | per trade | 95% CI | median | hit |
|---|---|---|---|---|---|
| all signals, as published | 342 | +8.61% | [+6.28%, +10.93%] | +7.24% | 65% |
| **borrowable only** | **55** | **+4.32%** | **[−0.38%, +9.12%]** | +0.52% | 55% |
| not borrowable | 287 | +9.43% | [+6.79%, +12.11%] | +7.58% | 67% |

**The edge lives almost entirely in the names that cannot be shorted.** On the
borrowable subset the confidence interval includes zero, the median falls from
+7.24% to +0.52%, and the hit rate from 65% to 55%.

This is economically coherent rather than surprising: a name is hard to borrow
*because* the demand to short it is high, and that demand exists precisely where
the expected decline is largest. The short strategy was measuring an
opportunity that other people had already taken.

**Caveat on the caveat:** the registry is a *current* snapshot, not
point-in-time. A name shortable today may not have been at signal time and vice
versa, and 2026 flags applied to 2025 signals are anachronistic. The direction
is clear and the mechanism is sound, but the magnitude is an estimate.

## Revised verdict

**§63's short strategy is downgraded from "profitable, conditional on borrow" to
"not demonstrated".** The condition turned out to be most of the result. What
survives is a much smaller claim: on the ~16% of consolidations that are
borrowable, the point estimate is +4.32% with an interval that includes zero at
n=55. That is a hypothesis, not a strategy.

**§65's long strategy is unaffected** — a long position needs no locate. It
remains the only demonstrated result in the project.

# 68. The lesson

**Check whether the data exists before declaring it unavailable.** The IBKR FTP
probe failed in round 0, and "no borrow data" was carried as an assumption for
seven rounds and written into a published document — while the field sat in a
parquet file this project fetched itself. One `grep` of the registry columns
would have caught it at any point.

The pattern is the same one behind every other error in this log: **a fact was
established once, under one set of conditions, and never re-tested when the
question changed.** The gap screen (§53), the cost table (§37), the fold sizes
(§30) and now the borrow feed all failed that way.

---

# 69. Two bugs the first live screen found

## A forward split is invisible to a reverse-split detector

`flag_reverse_splits` fires on `open / prev_close >= 1.8` -- price going **up**.
A *forward* split divides the price, so a 1-for-10 prints as a **90% overnight
decline**: indistinguishable from a catastrophe on price alone, and ranked by a
buy-the-crash screen as its most extreme signal. Added `flag_forward_splits`,
and excluded it across the **whole lookback** rather than the signal day alone
-- a split three sessions back makes `ret_5d` read -90% while the signal day
itself looks ordinary.

**Writing the test caught a design flaw before it shipped.** The first version
reused the reverse detector's inert-session test, `|open->high| < 35%`. That is
the wrong measure here. Tenax Therapeutics fell 89.8% overnight on a failed
Phase 3 -- a real catalyst -- and opened at $1.79 against a high of $1.83, an
open-to-high of **2.2%**, which reads as perfectly inert. Its low was $1.31, so
the session **range** was 39.7%. Open-to-high can only see a stock that runs up;
a name that opens and collapses looks quiet to it. Inertness is now measured on
the high-low range, and `test_inertness_uses_range_not_open_to_high` pins it.

## `date.today()` is the machine's date, not the market's

The Saturday run returned **zero rows** after 90 minutes. Not throttling: this
machine reports UTC+9, so `date.today()` gave `2026-08-15` while the US market
was still in Friday's session. `end=2026-08-15` asked for data hours in the
future and every request was rejected as "recent SIP data" -- and
`fetch_with_rejects` then binary-searched each failing batch hunting for a bad
symbol, which is why it ran 7x slower and still produced nothing.

Two traps stacked, both returning *empty* rather than raising:

1. the local date can be a day ahead of US/Eastern;
2. Alpaca reads a bare date as **midnight UTC** = 20:00 ET the *previous* day,
   so passing the target date silently excludes that date's own bar.

Both are fixed by anchoring to an instant -- `now(UTC) - 20 minutes` -- which
also clears the free tier's 15-minute delay on recent SIP data.

**This is the third time this exact bug has appeared** (§Phase-1 recorder, the
01:00 UTC cron, and now the screener). It is written down each time and
reintroduced anyway, because `date.today()` is the obvious thing to type. The
durable fix is that no script in this project should call `date.today()` when it
means "the last completed US session".

## What the first two live screens looked like

| session | long | short |
|---|---|---|
| 2026-08-13 | 8 | 2 |
| 2026-08-14 | 9 | 4 |

Against a studied rate of ~1.1 long/session. Elevated, and consistent with a
broad small-cap breakdown that week rather than a defect -- both sessions'
candidates verify as genuine crashes with real intraday ranges and heavy volume.

**ADV20 can be dominated by the explosion itself.** YJ shows ADV20 of $9.4M, but
that is one day: 2026-08-07 traded $174M on a +176% move. By Friday it traded
**$148k**. RETO is the same shape -- $533k ADV20, $191k on the day, and 438
shares on 08-06. The liquidity filter is backward-looking and a name can pass it
while being effectively untradeable now. The backtest used the same filter, so
the measured result is internally consistent; the constraint binds on execution,
and how hard depends on account size.

---

# 70. Should the long carry a stop loss? No.

The calibrated rule has **no stop** -- and that was never a decision.
`build_trades` defaults to `stop=None` and the portfolio sweep varied drop,
hold, target and slot count while leaving the stop untouched. Tested properly
(`scripts/long_reversion_stops.py`), every stop makes it **worse, monotonically**:

| stop | per trade | 95% CI | median | target hit | stopped out | compounded | Sharpe |
|---|---|---|---|---|---|---|---|
| **none** | **+5.56%** | [+2.43%, +8.81%] | -0.05% | **27%** | - | **+109.4%** | **3.06** |
| -50% | +5.11% | [+2.17%, +8.41%] | -0.07% | 27% | 5% | +97.0% | 2.84 |
| -40% | +4.43% | [+1.43%, +7.38%] | -0.46% | 26% | 12% | +81.1% | 2.39 |
| -30% | +3.42% | [+0.58%, +6.28%] | -3.07% | 25% | 23% | +60.0% | 1.92 |
| -25% | +3.09% | [+0.55%, +5.82%] | -5.29% | 24% | 34% | +53.2% | 1.76 |
| -20% | +1.92% | [-0.64%, +4.36%] | -10.75% | 21% | 45% | +30.1% | 1.13 |
| -15% | +1.97% | [-0.21%, +4.14%] | -15.45% | 19% | 57% | +34.6% | 1.35 |

The mechanism is visible in the last two columns. At a -15% stop, **57% of
trades are stopped out and the target hit rate falls from 27% to 19%** -- the
stop is not cutting losers, it is ejecting positions that go on to double. These
names routinely fall 30% and then recover; the drawdown *is* the entry.

At -20% the confidence interval includes zero. A stop tight enough to feel
protective destroys the strategy outright.

**Downside control here is position sizing, not stops.** 1/30 per name means a
-100% costs 3.3% of the account. The worst single trade in 421 was **-66.2%**,
so nothing reached zero inside ten sessions.

This is the mirror image of §63's short, where a hard stop was *mandatory*
because five trades exceeded -100%. A short's loss is unbounded and a long's is
not, so the same instrument is essential in one and destructive in the other.

## First forward observation

Thursday 2026-08-13's eight signals, entered at Friday's open, marked to
Friday's close -- **1 of 10 sessions elapsed**:

| | entry | Fri close | P&L |
|---|---|---|---|
| RETO | 1.23 | 1.28 | +4.1% |
| TENX | 1.77 | 1.80 | +1.7% |
| GLND | 1.30 | 1.28 | -1.5% |
| CRIS | 1.83 | 1.80 | -1.6% |
| MEDS | 1.53 | 1.48 | -3.3% |
| WFF | 1.57 | 1.44 | -8.3% |
| ACH | 1.08 | 0.95 | -12.0% |
| LESL | 0.78 | 0.65 | -16.8% |

Equal-weight basket **-4.72%**, 2 of 8 green, 0 targets hit. Exactly the shape
the backtest describes -- most positions drift while the return waits on a
minority reaching +50% -- and a useful reminder that a -15% stop would already
have ejected LESL on day one of ten.

---

# 71. Should the entry be the signal day's close instead of the next open?

## The proposal

Since the strategy uses only limit orders and no premarket data, enter at the
**signal day's close** via a market-on-close order rather than waiting for the
next open -- and collect the overnight session instead of forfeiting it. The
look-ahead objection is weak: an MOC placed at 15:55 acts on a price minutes
old, not on a price from the future.

The reasoning is sound, and the headline supports it. §65's own TEST 1:

| entry | per trade | 95% CI | Sharpe | compounded |
|---|---|---|---|---|
| **close (MOC)** | **+5.89%** | [+2.66%, +8.87%] | **3.20** | **+125.6%** |
| next open (MOO) | +5.56% | [+2.43%, +8.81%] | 3.06 | +109.4% |

## What the overnight session actually pays

Measured directly on the 427 signal days (`long_reversion_entry_timing.py`):

| | |
|---|---|
| mean | +0.87% |
| **95% CI** | **[-0.56%, +2.63%]** -- includes zero |
| **median** | **+0.00%** |
| standard deviation | **17.28%** |
| gaps **up** | **43%** of the time |
| percentiles | p1 -25.7%, p5 -12.9%, p50 +0.0%, p95 +11.2%, p99 +49.2% |
| **drop the best 10 gaps** | **-0.97%** |

The overnight move on these names is **not distinguishable from zero**. The
median is exactly flat, it gaps *down* 57% of the time, and the entire positive
mean comes from about ten enormous gap-ups -- drop them and it is negative. This
is the same right-tail signature that killed the candidates in §57 and §61,
appearing in a component rather than in a result.

And the risk taken to collect it is not small: **17.28% standard deviation**, on
every position, overnight, for an expected value that a confidence interval
cannot separate from nothing.

## What it costs to capture

The close entry's advantage is **+0.33% per trade**, so it survives only up to
**33 bps of slippage**:

| MOC slippage | per trade | compounded | verdict |
|---|---|---|---|
| 0 bps | +5.89% | +125.6% | better than MOO |
| 25 bps | +5.64% | +117.8% | better than MOO |
| **50 bps** | +5.39% | +110.2% | **MOO wins** |
| 100 bps | +4.89% | +95.8% | MOO wins |

The measured round-trip cost in the $250k-$1M ADV band is **150 bps**, so a
half-spread is ~75 bps. A market-on-close order in a 65-cent microcap does not
fill at the print, and the realistic slippage is roughly double the entire
advantage.

## And the period split disagrees with the headline

| entry | in-sample | out-of-sample |
|---|---|---|
| close | +4.28% **CI [-0.28%, +8.53%]** | +7.73% [+3.49%, +12.19%] |
| open | +4.52% **CI [+0.21%, +8.98%]** | +6.76% [+2.18%, +11.46%] |

In-sample, the close entry's interval **includes zero** and the open entry's
does not. The close entry's better headline rests on its out-of-sample half.

## Verdict: keep the next open

Not because entering at the close is wrong in principle -- it is executable and
it does score higher -- but because the thing it collects **is not reliably
there** and costs more to collect than it pays. The +0.33% is noise on a
component whose own CI spans zero, and 33 bps is inside the bid-ask spread of
every name on the list.

**A liquidity-split version is the one variant worth testing later**: TENX at
$27M ADV would slip perhaps 10-20 bps on an MOC, where RETO at $191k/day would
slip far more. Entering the liquid names at the close and the thin ones at the
open might capture the difference -- but that is a rule change requiring its own
walk-forward, not an adjustment to make live.

---

# 72. RETRACTION — §65's long strategy was look-ahead bias

## The gate failed

Phase 1 of the continuation plan re-tested the §65 long over the eleven years of
daily bars that were already on disk and never used. It does not survive. Worse,
the reason it does not survive is that **it was never real**.

## The bug: the price band was filtered on adjusted prices

`long_reversion_verify.select()` screened the entry band with

```python
& (u["close"] >= price_lo)      # ADJUSTED close
& (u["close"] < price_hi)
```

`close` is the **split-adjusted** series. Adjusted prices are computed backwards
from every corporate action that happens *afterwards*, so a stock printing $0.60
today that reverse-splits 10-for-1 next month carries an adjusted price of $6.00
**today**. Screening `$0.50 <= price < $2.00` on that series therefore removes
every name that will later consolidate.

And reverse splits are not random. A company consolidates because its price has
kept falling and it needs to hold a $1.00 listing bid. So the filter was
silently selecting:

> crashed stocks that will not need a reverse split over the coming months

which is a **survivorship screen built from future information**, and it removes
precisely the worst outcomes in the population.

The contamination is not marginal. Over the §65 span, **46.8% of candidate rows
have an adjusted/unadjusted ratio above 1.5**. WOK's ratio is **1,000,000**.

## Isolating it, one change at a time

Re-implemented from scratch (`scripts/long_walkforward.py`, positional forward
paths), then changing a single thing per row:

| configuration | n | per trade | 95% CI | median |
|---|---|---|---|---|
| **§65 as published** — adjusted band, delisted dropped | 430 | **+5.32%** | [+2.12%, +8.41%] | −0.20% |
| + exclude forward splits | 429 | +5.58% | [+2.57%, +8.70%] | −0.05% |
| + keep delisted (exit at last print) | 466 | +4.84% | [+2.04%, +7.88%] | −0.45% |
| **honest** — unadjusted band, the price you can actually see | **808** | **−4.00%** | **[−6.70%, −1.38%]** | **−7.13%** |

The first row reproduces §65's published n=421, +5.56% to within the delisting
and forward-split handling, which establishes that the rewrite is faithful. The
last row changes **only which price series defines the band**. The sample nearly
doubles and the sign flips, significantly.

## The eleven-year picture

Frozen §65 parameters (drop ≥40%, hold 10, target +50%, 30 slots), honest band:

| span | n | per trade | 95% CI | compounded | maxDD |
|---|---|---|---|---|---|
| §65's own window | 802 | **−4.27%** | [−6.95%, −1.64%] | −72.6% | −76.8% |
| full 2016–2026 | 2,157 | **−2.20%** | [−4.03%, −0.30%] | **−93.6%** | −94.0% |

Per year, only **2019 and 2020** are positive — and 2020 is the COVID rebound,
when everything oversold bounced.

## The walk-forward

Parameters refit on an expanding window each year, trading the next
(`long_walkforward.py`, 45 parameter sets x 4 slot counts per fit):

- **six of nine years had no cell with a positive fit at all** — the procedure
  declined to trade 2018, 2019, 2020, 2024, 2025 and 2026;
- the three years it did trade: **n=838, −2.17%, CI [−4.46%, +0.15%]**, median
  −5.72%, target hit 13% (against §65's claimed 27%);
- 1 of 3 traded years profitable;
- dropping the best 50 trades gives −5.43%.

**Verdict: not distinguishable from zero, with a negative point estimate.**

## Why every audit test passed

§65 ran concentration, mean-versus-median, period confirmation, next-open entry,
liquidity floors, parameter-neighbourhood smoothness, and a survivorship check.
All passed. §70 added a stop-loss sweep and §71 an entry-timing study, both
internally consistent.

**Every one of them ran on an already-poisoned sample.** The bias was in the
*selection filter*, upstream of everything, so each test faithfully described a
population that had been chosen with future information. A smooth parameter
surface and a clean out-of-sample split are properties of the biased sample, not
evidence against bias.

The one check that would have caught it — comparing the two price series in the
filter itself — was the check §57 and §64 had already forced twice. Both of
those found adjusted/unadjusted mixing in a *return* calculation, and the fix
was applied there. Nobody looked at the *filter*.

## Where the project now stands

| candidate | status |
|---|---|
| intraday veto sets (rounds 1–4) | dead — negative gross before costs |
| gap screens (round 5) | dead — the screen selected the losing tail |
| 441-cell hypothesis grid | dead — 0 survived BH |
| short reverse splits (§63) | **not demonstrated** — the edge is in unborrowable names (§67) |
| long crashed microcaps (§65) | **retracted — look-ahead bias (this section)** |

**No profitable strategy has been demonstrated in this universe.** That is the
answer to the original question, and it is now supported by eleven years rather
than eighteen months.

## Consequences for the live screener

`screen_today.py` filters on `close_unadj` and is therefore **correct** — it was
written after §64 and never inherited the bug. But the rule it screens for has
no demonstrated edge, so the 2026-08-13 and 2026-08-14 lists are **not
actionable**. The Phase 3 paper-trading bot is cancelled; there is nothing to
forward-test.

The ADV fix made alongside this section (trailing, excluding the signal day)
stands on its own and is now consistent with the backtest.

## 73. The lesson, stated as a rule

**Adjusted prices encode the future. Never use them in a selection filter.**

Adjusted series exist to make *returns* comparable across corporate actions.
Every adjusted price before an action is computed from information that did not
exist at the time, so any screen that conditions on an adjusted *level* —
a price band, a market-cap band, a moving-average crossing, a 52-week-high
distance — is conditioning on the future. Returns can use adjusted data; filters
must use the tape.

This is the fourth adjusted/unadjusted failure in this project (§57, §64, §67's
cousin, and now §72), and the most expensive, because it survived seven
subsequent sections of audit. The generalisation that would have prevented all
four: **a study should assert, once and centrally, which series each column is
allowed to touch** — and the assertion belongs in the universe builder, not in
each analysis.

---

# 74. Round 9 — an automated search, a data-continuity bug, and the COVID crash

## What was searched

The proposal was a different structure to §65's: take stocks that **fell 50%+ in
a week**, wait until they **start moving up**, put the **stop where the bounce
is**, and take profit on a **retrace of the gap down**. Rather than hand-code it,
this round built a declarative engine and searched **195 specs** over the eleven
year panel.

Three pieces of infrastructure, all of which outlive the result:

* **`etl/screens.py`** — `assert_unadjusted()` raises if a *filter* references an
  adjusted price level. §72's bug is now a thrown exception rather than a thing
  to remember. `adjustment_tripwire()` reports what share of **selected** rows
  have adj/unadj > 1.5.
* **`strategy/engine.py`** — a strategy is a `Spec` dataclass, and there is one
  implementation of selection, barriers, exits and portfolio simulation. Ties go
  to the stop; gaps through the stop fill at the open; corporate actions are
  excluded over lookback *and* holding window; a delisting exits at the last
  print rather than being dropped.
* **A sealed holdout.** Signals from 2024-01-01 are refused without `--unseal`.
  The search optimises a metric, so a metric that can see the holdout overfits
  it.

## Three findings that survive

**1. Every stop hurts.** The proposed structural stop at the bounce low fired on
**74%** of trades and returned −2.27%. Fixed stops at −20% and −35% were no
better. These names routinely break the bounce low and recover: the drawdown
*is* the entry. This independently reproduces §70 on a completely different
entry rule.

**2. The metric was wrong, and changing it changed the answer.** The search
initially maximised the **lower CI bound of per-trade return**. That selected a
lottery: +41.4% per trade with **negative medians in six of eight years**,
Sharpe 0.90 and a −39.1% drawdown — in 2020 it earned +65% per trade and
compounded to **+14.3%**, because σ²/2 exceeded μ. Switched to **portfolio
Sharpe**, gated on the per-trade CI excluding zero *and* a positive compounded
return. §49, §56 and §65 all died of this and the search walked straight into it
again within three iterations.

**3. On a portfolio metric the confirmation and the target earn their place;
the stop never does.** On per-trade mean, dropping the confirmation and the
target scores *higher* (+58.5%). On Sharpe, the confirmed and targeted version
wins. The proposal's shape was right in two of three parts.

## The bug that invalidated the first answer

The search produced a candidate that passed everything: dev Sharpe 1.55,
compounded +920%, and a **sealed-holdout** result of **+36.25% per trade, CI
[+24.54%, +49.23%], Sharpe 2.40, maxDD −11.4%, three of three years positive**.

Then the individual trades were read.

**MI** showed 2026-03-17 at **$0.48** and the next panel row at **$10.01** on
2026-07-02. **GDHG**: $0.42 then $5.80. Those are not consecutive sessions.

`forensic_universe.parquet` is **pre-filtered** — it keeps only rows whose
previous close sits in $0.50–$10.00 with a computable ADV. Every `shift()` and
every forward path computed on that frame walks over **surviving rows, not
sessions**. A stock that falls below the $0.50 floor vanishes from the frame and
reappears weeks later at a different price, and the time-series operations step
silently across the hole.

Measured:

| | first panel | corrected panel |
|---|---|---|
| rows | 3,431,765 | **6,710,204** |
| consecutive-row gaps > 10 days | **13,279** | **168** |
| gaps > 90 days | 3,815 | 156 |
| signals whose 5-day window spanned > 10 calendar days | **886 of 4,670 (19%)** | 0 |

MI alone had **73 of 91 sessions invisible** to its forward path. A 90-bar hold
skipped the months the stock spent at its lows and booked the recovery.

**The fix is a rule, not a patch: filter at signal time; never filter the series
a shift walks over.** Eligibility is now a boolean column (`in_universe`) on a
continuous panel, and a signal whose lookback spans more than `1.6 x window + 10`
calendar days is rejected outright.

## What was left after the fix

| | contaminated | corrected |
|---|---|---|
| per trade (dev) | +29.27% | +19.87% |
| excess over date-matched control | +25.17% (7.1x) | +12.16% (**2.6x**) |
| drop the best 50 trades | +7.09% | **−3.51%** |

And the top ten trades named the real mechanism — **seven of them fall between
2020-03-10 and 2020-03-13, and every one is an oil and gas name**: MTDR, OVV,
SM, PR, QEP, APA, CLB. That is the bottom of the COVID crash.

Leave-one-year-out is unambiguous:

| removed | n | per trade | 95% CI | Sharpe |
|---|---|---|---|---|
| nothing | 274 | +19.87% | [+5.24%, +33.44%] | 0.74 |
| 2016 / 2017 / 2018 / 2019 | ~250 | +19.9% to +21.5% | all exclude zero | ~0.73 |
| 2021 / 2022 / 2023 | ~230 | +22.7% to +25.0% | all exclude zero | ~0.85 |
| **2020** | **204** | **+1.31%** | **[−6.46%, +9.26%]** | **0.10** |

And precisely:

> **Removing 2020-02-15 to 2020-06-30 — 59 trades of 274, four and a half months
> of eleven years — leaves +1.17%, CI [−6.36%, +8.79%], median −8.08%, Sharpe
> 0.09, compounded +3.7%.**

## Verdict

**No strategy. The candidate was one market event.** Buying beaten-down energy
equities at the exact bottom of the COVID crash worked; nothing else did. A rule
that requires a once-in-a-decade dislocation to be profitable is a description of
that dislocation, not a strategy.

The holdout number of +36.25% is **void** — it was computed on the buggy panel
and measures the gap artefact, not the rule. It is recorded here so the sequence
is legible, not as a result.

## 75. The lesson

**Filter at signal time. Never filter the series a shift walks over.**

Every derived quantity in a panel — a lagged close, an N-day return, a rolling
low, a forward path — assumes its neighbours are the adjacent *sessions*. Apply
a row filter first and that assumption silently becomes false, with no error and
no missing values, because the frame is still perfectly well-formed. This is
strictly worse than the §72 adjusted-price bug: that one at least left a
detectable fingerprint in a ratio.

It also cost the third data-integrity failure in a row that was caught **only by
reading individual trades** (§57, §64, §72, now §74). The statistics never
objected: the contaminated version had a tighter confidence interval, a higher
Sharpe, a cleaner drawdown and a passing sealed holdout. Four separate
bias checks — leave-one-year-out, a date-matched control, a delisting
worst-case, and a split-exclusion audit — all passed on corrupted data.

**Reading ten rows of raw tape has now overturned four conclusions that survived
every aggregate test put to them.** It belongs in the pipeline, not in the
post-mortem.

---

# 76. Sector conditioning — the population really is two populations, and it still isn't a strategy

## The proposal

Use a sector classifier as a pre-model: a stock down 50% while its **sector is
flat** fell for its own reasons (a failed trial, a fraud); one down 50% while its
**sector fell too** fell because a factor moved. Nine rounds pooled the two and
reported the average, which describes neither. §74's ten largest trades were
seven oil and gas names on two consecutive days — the systematic case, presented
as the general one.

Sector was not in any cached file, but EDGAR publishes an SIC code in the same
`submissions` payload `etl/edgar.py` already fetches. One pass over the 2,879
known CIKs gave **98.9% coverage** (2,847 symbols) at no new data source.

## The split is real, and it is enormous

Round 9's arm, split on the sector's own median 5-day return (computed
point-in-time from the same session's cross-section, leave-one-out so a thin
sector is not compared against a median the stock itself set):

| arm | n | per trade | 95% CI | median | Sharpe |
|---|---|---|---|---|---|
| **systematic** — sector ≤ −10% | 57 | **+91.24%** | **[+68.36%, +107.50%]** | **+91.22%** | 1.83 |
| pooled (§74's arm) | 274 | +19.87% | [+5.24%, +33.44%] | +1.50% | 0.74 |
| **idiosyncratic** — sector ≥ −5% | 183 | **+1.38%** | **[−6.26%, +9.75%]** | −7.67% | 0.10 |

A **66×** separation, and the systematic arm's median ≈ its mean, so it is not
tail-driven. The hypothesis was correct: these are two different populations.

## But the systematic arm was one event, again

| sector threshold | n | inside COVID | outside |
|---|---|---|---|
| ≤ −10% | 57 | **52** | 5 |
| ≤ −15% | 41 | **41** | **0** |

The conjunction was mis-specified: demanding a **−50% individual** crash inside a
sector down only 10% is close to a contradiction. Relaxing the individual
requirement produced a genuinely testable population — and, excluding COVID
entirely:

| individual drop | sector | n (ex-COVID) | per trade | 95% CI | median |
|---|---|---|---|---|---|
| ≥20% | ≤ −10% | **286** | **+8.41%** | **[+4.06%, +12.98%]** | **+18.75%** |
| ≥20% | ≤ −5% | 779 | +3.04% | [−0.70%, +6.16%] | +17.90% |

Left-tailed, Sharpe 0.80, maxDD −8.2%, and the sector condition is visibly doing
the work — at −5% the interval covers zero, at −10% it does not.

Then the same shape as always:

- **232 of 286 trades fall in 2022.** Without 2022: n=54, +7.49%, **CI [−4.34%,
  +19.73%]**.
- All 286 trades occur on **48 distinct days across eight years** — about six
  days a year. Four of the twelve largest share one date, 2022-06-14.

## The test that settles it

If a sector-wide crash is the signal, the honest question is whether picking the
*crashed* names beats picking *anything* on those same days. Paired by date, so
the day effect cancels:

| control | control return | paired difference | |
|---|---|---|---|
| any liquid stock, same day | +6.69% [+2.30%, +14.74%] | **−1.06%** [−12.09%, +9.77%] | not distinguishable |
| **same sector**, same day | +7.01% [+2.49%, +14.03%] | **−1.96%** [−14.39%, +9.70%] | not distinguishable |

**The stock selection adds nothing — and the point estimate is negative.** On a
day when a sector is down 10%, buying anything returns about +6.7%; buying the
specific names that crashed hardest returns about the same. The unpaired "+2.19%
excess" was a day-mix artefact that the pairing removes.

## Verdict

**Sector conditioning is a real discovery about the data and not a strategy.** It
correctly identifies that the population is two populations, and it correctly
isolates the one that moves. But what it isolates is a **timing** signal —
sector-wide dislocations are followed by broad rebounds — with **no selection**
content whatsoever.

And a timing signal here has the sample size of the events themselves: sector
crashes deep enough to matter occurred on 48 tradeable days in eleven years,
concentrated in 2020 and 2022. That is the n≈4 regime problem arriving through
the sector door rather than the front one. The rebound is real; there is simply
no way to establish it is repeatable from four or five instances, and nothing in
the stock-level data adds to it.

## 77. What this rules in and out

**Ruled out, now with a mechanism rather than just a null:** the crash-and-bounce
family, in every parameterisation tried across rounds 8, 9 and 10. When it works,
it works because the whole market or sector is rebounding, and then it does not
beat buying at random on the same day.

**Ruled in as method:** splitting a population beats thresholding one. The
idiosyncratic/systematic split produced the single largest effect separation in
the project (66×) on the first attempt. The lesson is not that sector is the
right variable — it is that nine rounds of *univariate thresholds on a pooled
population* were asking a question that could not have a clean answer.

**The standing check that keeps paying:** a date-matched, paired control. §53
recorded it as mandatory after equity drift was mistaken for alpha; it has now
killed two further candidates that survived leave-one-year-out, concentration
tests and confidence intervals. Any result whose trades cluster on few days is
a day effect until a paired control says otherwise.

---

# 78. Round 11 — the first validated alpha: a cross-sectional veto

## Why ten rounds could not have found this

Rounds 1–10 all estimated the same object — the unconditional mean of a screened
set. Decomposing the forward return as `R = mu_t + lambda_{t,g} + eta`, a screen
that fires on few days puts nearly all of its variance into `mu_t`. §76 measured
it: the sector arm placed 286 trades on **48 days** and the paired control
returned −1.06% with a confidence interval **22 points wide**. "No effect" and
"an effect too small to see" were indistinguishable. Every prior round shares
this defect.

Demeaning **within the day** removes `mu_t` algebraically — no control group, no
matching. Amendment 1 fixed cells as days rather than day×sector, because
day×sector cells have median size 1–2. That single change took the design from
**48 clusters to 1,834**.

## The cross-section has real structure

58,016 candidates, 1,834 cells, development only. The random control returns
IC **−0.0016, CI [−0.0140, +0.0109]** — the estimator is calibrated. Then:

| feature | IC | t | reading |
|---|---|---|---|
| realised volatility | **−0.0845** | **−12.0** | low-vol wins — the MAX effect (Bali–Cakici–Whitelaw 2011) |
| prior 50%+ spikes | −0.0748 | −9.8 | lottery names underperform |
| distance from 20-day low | −0.0565 | −8.2 | short-horizon reversal (Jegadeesh 1990, Lehmann 1990) |
| dollar-volume shock | +0.0545 | +8.0 | attention / capitulation (Barber–Odean 2008) |
| log ADV | +0.0383 | +5.4 | liquidity conditioning (Avramov–Chordia–Goyal 2006) |

Every sign matches the literature the formulation named in advance.

**The random control's long-short net of −1.06% is the cost floor** at gross
exposure 1, so features must be read against that rather than zero. The best
single gross spread is about +0.95% against a 1.06% round trip: real signal,
eaten by cost.

## The signal is monotone but left-tailed

Combining seven features (signs fitted on development **before 2020**, reported
on development **from 2020**) lifts composite IC from 0.067 at h=5 to **0.133**
at h=60 with ADV >= $5M. But the tradable form was not the obvious one:

| construction | excess over the cell average | 95% CI |
|---|---|---|
| long the top decile | −1.18% | [−3.20%, +0.97%] |
| long the top decile, **random feature** | +0.77% | [−1.73%, +3.41%] |
| **reverse the signs** (long the bottom) | **−5.63%** | [−7.86%, −3.27%] |

The top decile does not beat the cell average, and neither does a random
feature — the apparent "+5.22% long-only" was entirely `mu_t`. But reversing the
signs is **strongly** negative. The relationship is monotone and concentrated in
the **left** tail: the composite identifies losers powerfully and winners not at
all, which is exactly how the MAX effect is documented.

**So the tradable form is a veto** — which is what round 0 set out to find.

## The veto

> Population: eligible symbol-days that fell >=15% over five sessions, $0.50–$10,
> ADV >= $250k, in a day-cell of at least five names. Score each name by the
> composite, **drop the worst 25% within the day**, equal-weight the rest, hold
> 60 sessions, enter at the next open, one-sided measured cost.

| | development (>=2020) | **sealed holdout 2024–2026** |
|---|---|---|
| drop worst 10% | +0.67% [+0.43%, +0.90%] | **+1.53%** [+1.27%, +1.78%] |
| **drop worst 25%** | **+1.07%** [+0.56%, +1.56%] | **+2.43%** [+1.91%, +2.91%] |
| random control | −0.11% [−0.33%, +0.11%] | −0.06% [−0.33%, +0.19%] |

Every leave-one-year-out interval excludes zero. Excluding the COVID window it
**improves** (+0.72% and +1.24%). 62.8% of cells are positive.

### As a portfolio

| | base (all crashed names) | **veto (drop worst 25%)** |
|---|---|---|
| dev >=2020, per 60d | +6.40% | **+7.47%** |
| dev annualised / Sharpe | +10.23% / 1.35 | **+14.93% / 1.92** |
| **holdout, per 60d** | **+0.62%** | **+3.05%** |
| **holdout annualised / Sharpe** | **−4.12% / −0.88** | **+5.94% / 1.21** |

| holdout year | base | veto | excess |
|---|---|---|---|
| 2024 | −0.40% | **+2.18%** | +2.58% [+1.69%, +3.40%] |
| 2025 | +3.56% | **+6.19%** | +2.63% [+1.81%, +3.45%] |
| 2026 | −2.60% | −0.77% | +1.83% [+0.94%, +2.65%] |

**The base population LOSES money on the holdout (−4.12% annualised). The veto
turns it into +5.94%.** The excess is positive and significant in all three
holdout years, and it grew while the beta vanished — the strongest evidence
available that it is independent of the market effect.

## Honest limits

1. **The magnitude is modest.** +5.94% annualised at Sharpe 1.21 — not the
   headline numbers earlier rounds produced and then retracted.
2. **It is a veto, not a selector.** It cannot pick winners; it removes losers.
   The long leg still holds ~58 names and inherits whatever the population does.
3. **The base is negative on the holdout**, so the whole return is the veto. A
   period where the veto stops working is a losing period outright — 2026 was
   still −0.77% even after vetoing.
4. **Execution is untested** at ~58 simultaneous microcap positions with
   60-session rotation; the one-sided measured schedule is assumed, not verified
   at that breadth.
5. Signs were fitted once on pre-2020 data and never refitted. That is the right
   discipline, and it also means no adaptation has been tested.

# 79. Why this one is different

Every retracted result in this project shared a signature: a large headline, a
tight interval, and a sample concentrated in one event or corrupted upstream.
This one has the opposite signature — **a small headline, intervals that survive
every split, a random control that behaves, and a holdout that came in stronger
than development while the market effect went the other way.**

The methodological lesson generalises past this project: **when a screen fires on
few days, the day effect is the estimate.** Demeaning within the day is not a
refinement of the earlier designs; it is the difference between measuring the
idiosyncratic term and measuring the market.

---

# 80. Round 12 — making it feasible, and the simplification that fell out

## The problem

Round 11's veto was validated and unimplementable: **787 concurrent positions**,
42 new orders a session, **0.040% of capital per name** — $40 on a $100k account.
Amendment 2 fixed the constraint and the identity

    N = (H / tau) * K * (1 - delta)

## Four findings, in the order they closed off options

**1. Calendar cohorts are the wrong lever.** Entering every tau-th session
discards 80–95% of the signal *and* collapses the sample from 973 cohorts to 47,
so the veto stops being measurable (intervals 14 points wide). Duplicates also
mean `N` falls far more slowly than `H/tau` predicts — at H=60, going tau 1→5
took N_p90 from 707 to 487, not to 141. **2 of 84 configurations admissible**,
best Sharpe 0.70.

**2. The horizon lever is closed.** The veto excess is a *threshold*, not an
accrual:

| H | 10 | 20 | 30 | 40 | **60** | 90 | 120 |
|---|---|---|---|---|---|---|---|
| excess | −0.09% | −0.42% | +0.07% | +0.32% | **+1.07%** | +0.97% | +1.54% |
| CI excludes 0 | no | no | no | no | **yes** | yes | yes |

H must be ≥60, which puts a floor under the position count.

**3. An accounting bug, found and fixed.** `slots.py` booked `r/N` as a return
contribution on the exit day and compounded the mean of that series. That is
only valid for small `r`, and these are 60-session microcap returns with sigma
near 70% — it disagreed with the §78 method by a **factor of two**. Replaced
with a true equity simulation in currency: a slot entered with `equity/N`
returns `capital*(1+r)` to cash on exit. Every number below uses that.

**4. Breadth is irreducible, and the arithmetic says why.** The edge is +1.07%
per 60 sessions against single-name sigma near 70%, so a portfolio of `N` names
estimates it with `SE = sigma/sqrt(N)`: at N=30 the standard error is ~13%,
five times the signal. Measured breadth curve agrees exactly — **the veto's
Sharpe contribution is negative or nil below 150 slots and turns positive only
at 250–400.**

## The lever that was missing was sigma, not N

Sharpe is mu/sigma and nothing had touched the denominator. Keeping only names
below the **30th percentile of 20-day realised volatility** cuts drawdown by a
third and lifts Sharpe at every breadth:

| slots | vol cap | dev Sharpe | dev maxDD |
|---|---|---|---|
| 250 | none | 0.72 | −43.8% |
| 250 | 0.30 | 0.67 | −34.1% |
| 400 | none | 0.86 | −35.2% |
| **400** | **0.30** | **1.05** | **−24.4%** |
| 800 | 0.50 | 1.71 | −16.2% |

## V2, chosen on development and then held out

> Crashed microcaps (≥15% over five sessions, $0.50–$10, ADV ≥$250k). Keep the
> **lowest 30% by 20-day realised volatility** within the day. 400 slots, equal
> weight, enter at the next open, hold 60 sessions.

| | round 11 | **V2** |
|---|---|---|
| concurrent positions | 787 | **400** |
| new orders per session | 42 | **5.7** |
| weight per name | 0.040% | **0.25%** |
| dev Sharpe / maxDD | — | 1.05 / −24.4% |
| **holdout Sharpe** | — | **2.61** |
| **holdout annual** | — | **+18.3%** |
| **holdout maxDD** | — | **−13.7%** |

## The simplification: the composite *was* the volatility effect

On the holdout, with the volatility cap in place, the seven-feature composite
adds **nothing**:

| 400 slots, H=60 | holdout Sharpe |
|---|---|
| vol cap 0.30 **+ veto** | 2.61 |
| vol cap 0.30, **no veto** | **2.63** |
| no vol cap, + veto | 1.66 |
| no vol cap, no veto | 0.76 |

The veto and the volatility cap are **substitutes**. That is not a surprise in
hindsight — realised volatility was the composite's strongest component by a
wide margin (IC −0.0845, t = −12.0, against +0.038 for the next best). The
composite was a seven-feature wrapper around one effect.

**Engineering consequence:** the strategy needs **one feature**, not seven, and
**no fitted signs at all** — which removes the entire sign-fitting leakage
surface along with six features of implementation.

**Research consequence:** the round-11 result stands, but its interpretation
changes. What was validated is not "a composite veto" — it is the **low-volatility
effect inside crashed microcaps**, which is Bali–Cakici–Whitelaw's MAX finding
holding in the one corner of the market where it should be strongest.

## Honest limits

1. **400 positions is fewer, not few.** At 0.25% per name a $2,000 minimum
   position implies an **$800k account**. Order flow is the easy part (6 a day);
   capital is the constraint.
2. **The holdout is much better than development** (2.61 vs 1.05). The
   consistent direction across §78 and here is that 2024–2026 suited this
   strategy; that is a period property, not a guarantee.
3. **This is the second valid holdout read**, declared in Amendment 2 before it
   was taken. It tested portfolio construction on an already-validated signal —
   but it is a second read and the multiple-comparison cost is real.
4. **Execution at 400 simultaneous microcap positions remains unmeasured.** The
   one-sided cost schedule is assumed at that breadth.
5. The volatility cap quantile (0.30) was chosen on development from three
   values. Not a large search, but not zero either.

---

# 82. Round 13 — 787 positions to 30, and what it costs

## Two findings did the work

**1. The simulation had a design flaw, not the strategy.** Filling slots greedily
means that at N=30 with a 60-session hold, **all 30 fill on day one and exit
together**. The book is one synchronised cohort with maximum exposure to the
common factor — the term diversification cannot remove. Capping entries at
`ceil(N/H)` per session spreads the book over 60 cohorts.

That alone took N=30 from **Sharpe 0.48 to 1.08** on development. Every earlier
conclusion that "breadth is irreducible" was measuring my own fill rule.

The variance decomposition says why, and it was there to be read:

$$R_i = \mu_t + \eta_i,\qquad
\operatorname{Var}(\text{portfolio}) = \operatorname{Var}(\mu_t) + \frac{\operatorname{Var}(\eta)}{N}$$

| target | sd(mu_t) | sd(eta) | ceiling = mean/sd(mu_t) | N* | N for 90% of ceiling |
|---|---|---|---|---|---|
| none | 33.4% | 61% | 0.93 | 3 | **14** |
| +50% | 19.2% | 32% | 1.01 | 3 | **12** |
| +100% | 24.9% | 41% | 1.11 | 3 | **12** |

**Twelve to fourteen positions reach 90% of everything achievable.** The earlier
N=400 requirement was an artifact, not arithmetic.

**2. A profit target raises information ratio per position**, which is precisely
the quantity that sets required breadth. A +100% target cuts sigma from 70% to
49% while giving up 1.6 points of mean: **IR/pos 0.217 → 0.273**.

Stops do the reverse at every level tested (IR 0.217 → 0.129 at −20%),
independently reproducing §70 on a different population, a different horizon and
a different filter.

## V3

> Crashed microcaps (≥15% over five sessions, $0.50–$10, ADV ≥$250k). Keep the
> lower half by 20-session realised volatility, drop the worst quarter by
> composite. **30 slots. Enter at most one name per session.** Equal weight,
> 3.33% each. Exit at **+100%** or after 60 sessions.

| | round 11 | V2 | **V3** |
|---|---|---|---|
| concurrent positions | 787 | 400 | **30** |
| orders per session | 42 | 5.7 | **0.51** |
| weight per name | 0.040% | 0.25% | **3.33%** |
| **dev** Sharpe / annual / maxDD | — | 1.05 / +9.1% / −24.4% | **1.54 / +28.7% / −30.7%** |
| **holdout** Sharpe / annual / maxDD | — | 2.61 / +18.3% / −13.7% | **0.99 / +15.9% / −23.5%** |

On a $50,000 account that is **$1,667 a position and one trade every other day**.

## What feasibility costs, stated plainly

**The seed-to-seed standard deviation of Sharpe at N=30 is 0.75.** With five
seeds the standard error on the mean is 0.34, so the holdout figure is
**0.99 ± 0.34** — against **1.54 ± 0.09** on development, where the same
dispersion was a tenth as large.

That dispersion is not only estimation noise. At 30 positions *which names you
happen to draw* is a large part of the outcome, and a live trader gets exactly
one draw. **Concentration converts a diversified statistical edge into a path
that may or may not resemble its own expectation.** That is the real price of
going from 787 positions to 30, and it does not show up in a mean.

Two further honest points:

* On the holdout the **no-veto** variant scored higher (1.31 vs 0.99). Once the
  volatility cap is applied the composite keeps adding nothing — the same
  conclusion as §80, now at retail breadth.
* Holdout (0.99) is well below development (1.54). The direction is the opposite
  of §78 and §80, where the holdout came in stronger. With ±0.34 on the estimate
  the two are not distinguishable, which is itself the point.

**This is the third valid holdout read.** The accumulated selection across
rounds 11–13 is real and is recorded here rather than discovered later.

## The frontier, since there is no single answer

| positions | weight/name | orders/day | holdout Sharpe | account for $2k positions |
|---|---|---|---|---|
| 30 | 3.33% | 0.5 | ~1.0 ± 0.34 | **$60k** |
| 40 | 2.50% | 0.7 | ~1.5 ± 0.48 | $80k |
| 400 | 0.25% | 5.7 | 2.61 | $800k |

Breadth buys precision. The retail end of that frontier is now reachable; it is
simply a noisier place to stand.

---

# 84. Round 14 — the cost model was backwards, and it changes the answer

## The question that broke it open

"Does the $2,000 per position really matter?"

It was my assumption, never a measurement, and it was **inverted**. Two errors
were hiding behind it.

**Error 1 — the round trip was charged at half.** `COST_BPS` is documented as a
*round trip*, and every slot simulation charged `0.5 * cost`. A long-only
strategy that buys and later sells pays **both** sides.

**Error 2 — cost depends on dollar size, and these strategies do not trade
$2,000.** The measured NBBO gives a median inside-quote notional of **$1,900**,
and round 4's model is

$$\text{round trip} = 2\,h\left(1 + \left(\tfrac{\text{order}}{\text{inside}}-1\right)^{+}\right)$$

so cost is **flat below the inside quote and linear above it**:

| position | $100 | $250 | $1,000 | $2,000 | $5,000 | $25,000 |
|---|---|---|---|---|---|---|
| round trip | 76 bps | 76 | 83 | 100 | 199 | **873** |

The two errors nearly cancel at $250 a name (75 charged, 76 true) and do not at
$3,333 (75 charged, 144 true). **The flat schedule flattered precisely the
low-position-count configurations** — the ones the feasibility argument was
built on.

## The consequence is the opposite of the intuition

**More positions is cheaper.** A fixed account split more ways puts each order
further inside the quote:

| | 30 positions | 400 positions |
|---|---|---|
| $50k account | $1,667/name, **75 bps** | $125/name, **75 bps** |
| $100k account | $3,333/name, **144 bps** | $250/name, **76 bps** |

The "$800k account needed for 400 positions" in §80 was wrong in direction as
well as magnitude. **Position count was never a capital constraint.** It is an
operational one — how many line items a person will actually track.

## Re-run with size-aware, full round-trip cost

`strategy/cost_model.py` fits median half-spread and inside-quote notional by
ADV decile from the 9,794 measured symbol-days, then charges both sides at the
actual position size. $50,000 account, +100% target, vol cap 0.50:

| slots | $/name | cost | **dev** Sharpe | **holdout** Sharpe | holdout annual | maxDD | orders/day |
|---|---|---|---|---|---|---|---|
| 15 | $3,333 | 143 bps | 0.52 | 1.21 | +30.2% | −20.6% | 1.0 |
| 30 | $1,667 | 75 bps | 1.45 | **0.87** | +13.9% | −24.0% | 1.0 |
| **60** | **$833** | **75 bps** | **1.36** | **1.39** | **+15.8%** | **−21.6%** | **1.0** |
| 150 | $333 | 75 bps | 1.52 | 2.31 | +18.1% | −14.5% | 3.0 |
| 400 | $125 | 75 bps | 1.77 | 2.66 | +13.4% | −14.2% | 7.0 |

**N=15 is genuinely bad**, and at a $25k account (where $1,667 a name still fits
inside the quote, 75 bps) it is *still* bad — Sharpe 0.66. So the floor at 15 is
a **breadth** floor, not only a cost one, exactly as the variance decomposition
in §82 predicted.

## The recommendation changes from 30 to 60

**N=60 costs the same operational effort as N=30 — one order per session — and
is the only setting that is stable across both periods.**

| | dev | holdout |
|---|---|---|
| N=30 | 1.45 | **0.87** |
| **N=60** | **1.36** | **1.39** |
| N=150 | 1.52 | 2.31 |

N=30's dev figure was the best of the low-N row and did not survive. N=60 did.
That is what selecting on a development period and then testing is for.

## The rule this yields

$$N_{\min} \;=\; \max\!\left(\underbrace{60}_{\text{breadth floor}},\;
\frac{\text{account}}{\text{inside-quote notional}\;(\approx \$1{,}700)}\right)$$

Below the breadth floor the estimate is too noisy; above `account/inside` the
orders walk the book. For accounts up to about **$100,000, sixty positions
satisfies both** — and costs one trade a day.

## Honest limits

1. **This is the fourth holdout read.** It corrects a cost model rather than
   selecting a new strategy, but it is a fourth look and the accumulated
   selection is real.
2. The size-aware model is fitted from 9,794 measured symbol-days and applied by
   ADV decile. It is a model, not a per-name quote.
3. Sixty microcap positions entered one a day and held three months is still
   real administrative work — sixty line items, sixty exits to schedule.
4. Holdout Sharpe rises monotonically with N above 60 (1.39 → 2.31 → 2.66).
   **Anyone who can automate should hold more, not fewer.** The retail answer is
   a constraint being satisfied, not an optimum being found.

### §84 addendum — the sizing rule, confirmed at three account sizes

The background sweep finished after §84 was written. It validates
`N_min = max(60, account / 1700)` empirically rather than by assertion.
Development, +100% target, vol cap 0.50, size-aware full round trip:

| account | N=15 | N=30 | **N=60** | N=150 |
|---|---|---|---|---|
| **$25k** | $1,667 / 75bps / **0.66** | $833 / 75bps / 1.47 | $417 / 75bps / **1.36** | $167 / 75bps / 1.52 |
| **$50k** | $3,333 / 143bps / **0.52** | $1,667 / 75bps / 1.45 | $833 / 75bps / **1.36** | $333 / 75bps / 1.52 |
| **$100k** | $6,667 / 285bps / **0.21** | $3,333 / 143bps / 1.28 | $1,667 / 75bps / **1.33** | $667 / 75bps / 1.52 |

Two boundaries are visible and they are different things:

* **The cost boundary bites when position size passes the inside quote
  (~$1,700).** N=15 pays 75 bps at $25k, 143 at $50k, **285 at $100k**, and
  Sharpe falls 0.66 → 0.52 → 0.21 with it. N=30 crosses the same boundary
  between $50k and $100k (75 → 143 bps, 1.45 → 1.28).
* **The breadth boundary is fixed and account-independent.** N=15 is poor even
  at $25k where its orders still fit inside the quote — 0.66 against 1.36 at
  N=60. No account size rescues it.

**N=60 is the only setting that pays 75 bps at every account size tested**
($417 to $1,667 a name, all inside the quote) while sitting above the breadth
floor. That is why the rule takes a maximum of the two terms rather than either
alone.

---

# 85. Round 15, Part I — the baseline was not reproducible, and it was mis-costed

## Why this came first

The goal for this round was to attack the position count and to build a proper
evaluation battery before doing so. Exploring the code to wire the battery in
turned up something that outranked both.

**The round-14 table is not reproducible from committed code.** `bf2ccd4` ("the
cost model was backwards — 60 positions, one trade a day") touched exactly three
files: `LOG.md`, `specs/FINAL_RETAIL.json` and `strategy/cost_model.py`.
`fdc7d26`, the §84 addendum carrying the $25k/$50k/$100k table, touched `LOG.md`
alone. No simulation script was added or modified, and **`cost_model.py` is
imported by nothing in the repo**. The newest artefact in `data/research_penny/`
from that work predates the round-14 commits by a day.

So the N=15/30/60/150/400 table — the basis of the current recommendation — came
from an uncommitted, in-session variant of `scripts/risk_shaped.py:run`.

## The cost defect, verified

`cross_section.forward_return` returns a value **already net of one full round
trip**:

```python
r = cl[rows, ...] / entry - 1.0 - u["cost"].to_numpy()[pos]   # cross_section.py:88
```

`risk_shaped.attach_target_returns` returns a **gross** value:

```python
out[f"R_t{int(100*t)}"] = px / entry - 1.0                    # risk_shaped.py:171
```

Every simulator then subtracts the same half a round trip from whichever column
it was handed:

```python
net = float(row[col] - 0.5 * row["cost"])   # slots.py:135, risk_shaped.py:119,
                                            # veto_holdout.py:72
```

**The time-stop arm was charged 1.5 round trips and the profit-target arm 0.5** —
a threefold difference between two arms of the same sweep, in the direction that
favours the target. §84's premise ("every slot simulation charged `0.5 * cost`")
is correct for the target arm it was written about and backwards for the other.

This matters beyond arithmetic. §82 compares "no target" against "+100% target"
and concludes the target raises information ratio per position. Those two arms
were charged three times apart from each other.

## What replaced it

`research/penny/strategy/book.py` — one simulator, replacing three that
disagreed about position sizing, entry staggering, exit scheduling and cost.
Four defects fixed, each unit-tested in `tests/test_penny_book.py`:

1. **Cost charged exactly once.** The book works in gross return space and
   applies one size-aware round trip from `cost_model.round_trip_bps` at the
   position size actually traded. The sharpest statement of the fix is a test:
   undoing the embedded charge and re-applying one *flat* round trip must land
   back on `R` itself.
2. **Exits scheduled on the market calendar**, not on candidate days. The old
   loops index `sess = np.sort(df["date"].unique())`, which holds only days with
   at least `MIN_CELL` candidates — so a 60-session hold was 60 *candidate* days
   and the equity path ran on a clock that speeds up when candidates are scarce.
3. **A real no-selection control.** `slots.py`'s "noveto" arm skips the veto trim
   but still fills with `nlargest(..., "COMPOSITE")`, so `breadth_curve.py`
   compares veto+random fill against noveto+**top-score** fill — two changes at
   once. `fill` and `veto_frac` are now independent knobs. (`risk_shaped.py`'s
   `veto` flag did this correctly, so §80/§82's version of "the veto and the vol
   cap are substitutes" stands and `breadth_curve`'s does not.)
4. **The daily equity curve is returned.** All three old loops build one and
   throw it away, which is why no distributional metric has ever been computed
   on this strategy.

## The rebuild, and what the honest cost does

15 seeds, $50,000 account, H=60, vol cap 0.50, veto 0.25, target +100%.
LEGACY reproduces the old charge exactly (gross target column minus half a flat
round trip); HONEST charges one size-aware round trip.

**Development**

| N | $/name | LEGACY | HONEST | §84 published |
|---|---|---|---|---|
| 15 | 3,333 | 0.67 ± 0.37 | **0.46** | 0.52 |
| 30 | 1,667 | 0.84 ± 0.40 | **0.73** | 1.45 |
| 60 | 833 | 1.46 ± 0.24 | **1.37** | 1.36 |
| 150 | 333 | 1.68 ± 0.19 | **1.60** | 1.52 |
| 400 | 125 | 1.76 ± 0.07 | **1.67** | 1.77 |

**Sealed holdout** (the fifth read of it — recorded as such)

| N | LEGACY | HONEST | §84 published |
|---|---|---|---|
| 15 | 1.29 ± 0.54 | **1.07** | 1.21 |
| 30 | 1.09 ± 0.56 | **0.98** | 0.87 |
| 60 | 1.62 ± 0.45 | **1.51** | 1.39 |
| 150 | 2.33 ± 0.37 | **2.17** | 2.31 |
| 400 | 3.02 ± 0.21 | **2.79** | 2.66 |

Three conclusions.

**1. The reconstruction is faithful, except at N=30.** Nine of the ten LEGACY
cells land within about half a seed-standard-error of the published number, and
N=400 lands on it exactly (1.76 vs 1.77 dev, 3.02 vs 2.66 holdout). **N=30 on
development does not reproduce: 0.84 against a published 1.45.** The published
value is also the one anomaly in its own column — it is higher than N=60's 1.36,
breaking a curve that is otherwise monotone in N. The rebuild is monotone
throughout. §84 used exactly that N=30 figure to argue "N=30's dev figure was the
best of the low-N row and did not survive"; in the rebuild N=30 was never best.

**2. The honest cost does not overturn the recommendation.** At N=60 the charge
roughly doubles (48 → 81 bps) and Sharpe falls 6%, dev 1.46 → 1.37 and holdout
1.62 → 1.51. §84's *direction* was right and its conclusion survives its own
correction. Cost bites hardest exactly where §84 said it would: at N=15 the
charge goes 49 → 174 bps, because $3,333 a name walks past the ~$1,900 inside
quote, and Sharpe falls 0.67 → 0.46.

**3. The orders-per-day column was the cap, not the rate.** §84 reports
1.0/1.0/1.0/3.0/7.0 across N=15/30/60/150/400. Those are `ceil(N/H)` — the
per-session stagger limit. The **realised** rate is what the book actually does,
which in steady state is `N/H`: **0.25 / 0.51 / 0.97 / 2.40 / 3.97** on
development. So the claim that "N=60 costs the same operational effort as N=30 —
one order per session" is wrong. N=30 is half an order a session; N=60 is one.
The Sharpe ranking is unaffected, but the effort argument that justified moving
from 30 to 60 was double-counting.

## 86. The lesson

**A result that cannot be re-run is not a result yet.** §84 was written
carefully, pre-registered its constants, declared its holdout read and stated its
limits — and still shipped a headline recommendation produced by code that was
never committed. Everything that made it checkable was recorded except the thing
that produced it.

The operational rule: **the commit that reports a number must contain the code
that computed it.** A spec file and a log table are not enough; `cost_model.py`
sat in the same commit as the table it supposedly produced and was imported by
nothing.

---

# 87. Round 15, Part I — the evaluation battery, and what it says about the evidence

## What was built

`research/penny/eval/battery.py`, tested in `tests/test_penny_battery.py` — 38
tests, each pinning a metric against a case whose answer is known in closed form.
Five tiers: the Sharpe with a standard error and a deflation, the shape of the
loss, breadth, backtest-overfitting diagnostics, and execution realism. The full
mathematics and references are in `PREREGISTRATION_R15.md` Part I.

Two things about it are worth recording here rather than in the spec.

**`stats.deflated_sharpe_ratio` has existed since round 4 and had never been
called with a real trial count.** It was written, tested, and left holding a
default of 1. The battery now feeds it the project's actual ledger:
`search_trials.jsonl` (195) plus `PRIOR_TRIALS = 72` plus the round's own cells.

**The Hill estimator needed a correction that only shows up on this strategy.** A
profit target puts a point mass at the barrier: with a +100% target a large share
of trades return exactly +1.00 less cost. Hill sees a wall of ties, drives its
mean log excess to zero, and reports `alpha -> inf` — which reads as "no tail"
when what actually happened is "the tail was truncated by construction". The
right tail is now estimated only on gains below the barrier, with the truncated
share reported beside it. First run, the uncorrected version reported tail
indices of 220, 672, 1034 and `inf` across four adjacent cells, which is how it
was caught.

## The first finding: 972 trades are 17 observations

`average_uniqueness` on the N=60 development book:

| | |
|---|---|
| nominal trades | 972 |
| mean concurrency | 57.3 |
| average uniqueness | **0.0174** |
| **effective sample** | **16.9** |

At a 60-session hold with an entry a day, roughly 57 positions are live at any
moment and every one of them overlaps the others. The information content is
bounded by how many non-overlapping 60-session windows fit in the period, which
over 1,004 sessions is about 17.

**Stated carefully, because it is easy to over-read.** This measures *temporal*
overlap and treats two concurrent positions in different symbols as fully
redundant. They are not — they share `mu_t`, not `eta`. So 17 is the bound on
independent *time* observations, not on independent bets, and the truth lies
between 17 and 972. The correlation-adjusted breadth
`BR_eff = N / (1 + (N-1) * rho_bar)` is what speaks to the cross-section, and at
a modest `rho_bar = 0.2` it puts 60 nominal positions at under 5 effective ones.

Either way the direction is the same and it is uncomfortable: **every interval in
rounds 11-14 was computed on a nominal count that overstates the evidence.**

## The second finding: three uncertainty measures disagree by 5x

The daily returns of a 60-session book are strongly autocorrelated, and they stop
being so at almost exactly the hold length. Measured on the N=60 development
book:

| lag | 1 | 5 | 20 | 60 |
|---|---|---|---|---|
| rho | **+0.215** | +0.166 | +0.129 | −0.009 |

That is the overlapping-hold structure printing itself directly into the return
series. Its consequence, on the same book (single seed, Sharpe 0.948):

| measure | value |
|---|---|
| Sharpe | 0.948 |
| Mertens/Lo standard error (IID) | 0.450 → ±0.88 |
| **Lo autocorrelation-adjusted Sharpe** | **0.704** — a 26% haircut |
| stationary bootstrap, block 1 | [−0.02, +1.79] width 1.81 |
| stationary bootstrap, block 5 | [−0.51, +2.41] width 2.93 |
| stationary bootstrap, block 20 | [−1.27, +3.04] width 4.31 |
| **stationary bootstrap, block 60** | **[−1.42, +3.59]** width 5.01 |

The interval widens monotonically with the block length until the block reaches
the holding period, and then stops — which is exactly the signature of dependence
with a 60-session memory, and confirms the block bootstrap is measuring
something real rather than inflating noise.

**At the honest block length the 95% interval includes zero.** The Mertens
standard error, which assumes independent observations, is roughly a fifth of the
width the data actually supports.

The battery keeps 20 as its default block only so previously computed tables stay
comparable; `boot_block` is now an explicit parameter and a new result should
pass the holding period.

---

# 88. Round 15, Part I — what the battery says about the recommendation

Full sweep: N ∈ {15, 30, 60, 150, 400} × four arms, 15 seeds, one size-aware
round trip, \$50,000 account. The arms are a genuine factorial — the fill rule is
random in all four, so the veto is the only thing that changes between them,
which `breadth_curve.py`'s "noveto" control never managed.

## The table

**Development** (2020-01-01 … 2023-12-29, 1,004 sessions, 973 cells)

| N | base | veto only | volcap only | volcap+veto (V3) |
|---|---|---|---|---|
| 15 | 0.20 | 0.41 | 0.46 | 0.46 |
| 30 | 0.67 | 0.68 | 0.70 | 0.73 |
| 60 | 1.24 | **1.50** | 1.38 | 1.37 |
| 150 | 1.48 | **1.74** | 1.62 | 1.60 |
| 400 | 1.67 | **1.94** | 1.73 | 1.67 |

**Sealed holdout** (≥ 2024-01-01) — the fifth read, recorded as such

| N | base | veto only | volcap only | volcap+veto (V3) |
|---|---|---|---|---|
| 15 | 0.05 | 0.34 | 0.91 | **1.07** |
| 30 | 0.18 | 0.61 | **1.10** | 0.98 |
| 60 | 0.55 | 1.28 | 1.50 | **1.51** |
| 150 | 0.63 | 1.54 | 1.77 | **2.17** |
| 400 | 1.52 | 2.23 | 2.62 | **2.79** |

Three things follow immediately.

**The base population is close to worthless on the holdout** — 0.05 at N=15,
0.55 at N=60 — against 1.24 on development. This reproduces §78's central
finding on a corrected simulator: whatever the strategy earns out of sample, the
filters earn it.

**Development and the holdout disagree about which filter is best.** On
development `veto only` leads at every N ≥ 60; on the holdout it is the *worst*
of the three filtered arms and `volcap+veto` leads. Neither ordering is
stable, and the differences (0.13 to 0.34) sit inside the seed-to-seed
dispersion, which averages 0.26 on development and 0.44 on the holdout. The
honest reading is **all three filters are indistinguishable from each other and
all clearly beat the unfiltered base** — which is §80's substitutability finding,
neither confirmed nor refuted, but now measured on a real factorial.

**PBO is 0.214 on development and 0.414 on the holdout.** On the holdout,
picking the best cell of this sweep is close to uninformative about which cell
will be best next.

## Deflation: the answer depends on a choice, so both are given

The deflated Sharpe deflates against $SR^*_0$, the best Sharpe a search of $N$
trials yields **under the null**, and that needs $\operatorname{Var}(SR_n)$ —
the trial-to-trial dispersion *of noise*. Two candidates are available and they
disagree:

| period | dispersion used | sd | E[max] at 287 trials | N=60 V3 | best cell |
|---|---|---|---|---|---|
| dev | across the 20 swept cells | 0.554 | 1.60 | **−0.23** | +0.35 |
| dev | seed-to-seed at fixed config | 0.255 | 0.73 | **+0.63** | +1.21 |
| holdout | across the 20 swept cells | 0.779 | 2.25 | **−0.74** | +0.54 |
| holdout | seed-to-seed at fixed config | 0.439 | 1.26 | **+0.24** | +1.52 |

**The second is the defensible one.** The spread across the swept cells includes
the N-effect, which is a real monotone relationship rather than noise, so using
it as the null dispersion over-deflates. Seed-to-seed dispersion at a fixed
configuration is pure resampling noise and is what the statistic asks for.

On that basis **N=60 clears the 287-trial noise floor on both periods, but only
just on the holdout (+0.24)**, and the best cell clears comfortably (+1.21 dev,
+1.52 holdout). Recorded here because the harsher reading is one substitution
away, and because this is the first time in the project the deflated Sharpe has
been given a real trial count at all.

## Realised drawdowns are up to 3x worse than the Sharpe implies

`mdd_vs_expected` — realised maximum drawdown over its expectation for a random
walk with the same drift and volatility (Magdon-Ismail & Atiya 2004), on
development:

| N | 15 | 30 | 60 | 150 | 400 |
|---|---|---|---|---|---|
| ratio | 1.10 | 1.49 | **2.37** | 2.91 | 2.70 |

At N=15 the book draws down roughly what its own volatility predicts. By N=60 it
draws down **2.4 times** that. Breadth cuts volatility faster than it cuts
drawdown, because the losses arrive together — the same common-factor exposure
seen from the drawdown side. **A Sharpe comparison across N systematically
understates the risk of the higher-N configurations**, and nothing in fourteen
rounds would have shown that.

## Cost and breadth are separable, and §84 was right about which binds

The cleanest test the cost model allows: at a \$25,000 account both N=15
(\$1,667 a name) and N=60 (\$417) sit inside the ~\$1,900 inside quote, so both
pay essentially the same spread and only breadth differs. 15 seeds, development:

| account | N | \$/name | cost bps | Sharpe | seed sd | annual |
|---|---|---|---|---|---|---|
| 25k | 15 | 1,667 | 94 | 0.60 | 0.38 | +12.9% |
| 25k | 30 | 833 | 80 | 0.76 | 0.41 | +11.9% |
| **25k** | **60** | **417** | **81** | **1.37** | 0.25 | +18.4% |
| 50k | 15 | 3,333 | 174 | 0.46 | 0.39 | +9.3% |
| 50k | 30 | 1,667 | 91 | 0.73 | 0.41 | +11.4% |
| 50k | 60 | 833 | 81 | 1.37 | 0.25 | +18.4% |
| 100k | 15 | 6,667 | 346 | 0.17 | 0.41 | +1.9% |
| 100k | 30 | 3,333 | 168 | 0.54 | 0.42 | +8.0% |
| 100k | 60 | 1,667 | 93 | 1.33 | 0.25 | +17.9% |

**At matched cost, N=15 scores 0.60 against N=60's 1.37. The floor at 15 is a
breadth floor**, exactly as §84 argued, and its conclusion survives its own
correction. The cost effect is separately visible and compounds: N=15 falls
0.60 → 0.46 → 0.17 as the account grows and its orders walk further outside the
quote. N=60 is flat at 1.37 / 1.37 / 1.33, which is the mechanism behind
`N_min = max(60, account/1700)` — at \$417 to \$1,667 a name it never leaves the
inside quote.

**One row does not reproduce, at any account size.** §84 reports N=30 at
1.47 / 1.45 / 1.28 for \$25k / \$50k / \$100k; the rebuild gives
**0.76 / 0.73 / 0.54** — roughly half — while N=15 and N=60 reproduce closely at
all three. The uncommitted code cannot be recovered, so the cause is not
determinable. The consequence is: §84's claim that "N=30's dev figure was the
best of the low-N row and did not survive" rests on the single figure that does
not reproduce. In the rebuild N=30 was never best, and the curve is monotone in
N throughout.

## The tails: the heavy-tail escape hatch is closed

Per-trade Hill tail indices, development, +100% target:

| | N=15 | N=60 | N=400 |
|---|---|---|---|
| gains (below the barrier) | 3.69 | 8.45 | 11.31 |
| losses | 4.80 | 8.45 | 8.69 |

**All comfortably above 2, so variance is finite and `SE = sigma/sqrt(N)` is
legitimate arithmetic.** H1's first branch is answered before it was formally
opened: whatever explains the gap between §82's decomposition (predicting a 6%
Sharpe gain from N=15 to N=60) and the measured gain, it is not the tail index.
The remaining candidate is that §82's `sd(mu_t)` was measured on the uncapped
population, and that is what H1 should now test.

The barrier also matters less than its name suggests. **17.0% of the candidate
population touches +100%**, and only 8–10% of trades actually taken do, because
both filters select lower-volatility names that touch less often. Its value is
variance reduction, not upside capture — and that reproduces §82 exactly: mean
gross **+14.44%** with the target against **+16.08%** without, for a standard
deviation of **49.3% against 69.9%**. Giving up 1.6 points of mean to remove 20
points of sigma is the entire case for the target, and it holds.

## 89. What Part I changes, and what it does not

**Does not change:** N=60 remains the recommendation. It survives an honest
single cost charge (dev 1.46 → 1.37, holdout 1.62 → 1.51), it clears the
287-trial noise floor on the defensible deflation, and the breadth floor beneath
it is real at matched cost.

**Changes:**
1. §84's `orders/day` column is the per-session cap `ceil(N/H)`, not the realised
   rate `N/H`. "N=60 costs the same effort as N=30" is wrong — N=30 is half an
   order a session.
2. §84's N=30 row does not reproduce at any account size, and it is the row that
   carried the argument for moving from 30 to 60. The move is still right; the
   stated reason was not.
3. Drawdown risk at N ≥ 60 is 2.4–2.9x what the volatility implies. Any future
   comparison across N must report it.
4. The evidence base is far thinner than the trade counts suggest — ~17
   independent time observations behind 972 trades, and no development
   configuration whose block-bootstrap interval excludes zero.
5. Three superseded simulators are marked deprecated in place, pointing at
   `strategy/book.py`.

**Part II (H1–H5 in `PREREGISTRATION_R15.md`) has not been run.** Nothing in it
has been looked at.

---

# 90. Round 15, Part II — H1: why sixty positions is an identity, not a choice

## H1 as pre-registered, and its prediction falsified

`PREREGISTRATION_R15.md` predicted that `sd(mu_t)` measured on the **vol-capped**
population would be materially below the 24.9% §82 measured on the uncapped one —
which would have made `sigma_common` something the filter already moves, and
therefore a fourth lever. Measured on development, traded return (gross of the
+100% target, one flat round trip):

| population | n | cells | mean | **sd(mu_t)** | sd(eta) | sd total | ceiling |
|---|---|---|---|---|---|---|---|
| uncapped | 44,241 | 973 | +13.47% | **24.6%** | 41.2% | 49.3% | 0.55 |
| vol-capped at q=0.50 | 22,373 | 973 | +11.72% | **24.3%** | 36.4% | 43.9% | 0.48 |

Two things follow.

**The decomposition reproduces §82 almost exactly** — 24.6% against its 24.9%,
41.2% against its 41%. §82's arithmetic was faithful; the rebuild confirms it
rather than overturning it.

**And the prediction is falsified.** The vol cap moves `sd(mu_t)` from 24.6% to
24.3% — a 1% relative change, nothing. What it moves is `sd(eta)`, 41.2% → 36.4%.
**The volatility filter is purely an idiosyncratic-variance filter; it does not
touch the common factor at all.** Per the pre-registration this branch stops: the
§82-versus-§84 gap is not explained by which population was measured.

## What does explain the gap

The decomposition predicts portfolio `sd = sqrt(sd(mu)^2 + sd(eta)^2/N)`, so
going N=15 → N=60 should improve the Sharpe by a factor of **1.06**. Measured:

| N | 15 | 30 | 60 | 150 |
|---|---|---|---|---|
| predicted ratio vs N=15 | 1.00 | 1.03 | **1.06** | 1.07 |
| measured ratio vs N=15 | 1.00 | 1.58 | **2.94** | 3.45 |

The resolution is that **in this design `N` is not only a breadth parameter.**
The stagger admits `ceil(N/H)` entries a session, so the entry rate is

$$\lambda = N/H$$

and the number of bets placed over `T` sessions is `T*N/H` — which also scales
with `N`. The decomposition describes cross-sectional averaging **at one
instant**; it is silent about how many sequential bets the book gets to place.
Over the development period N=15 places **255** trades and N=60 places **973**.

Adding the temporal term, the expected improvement is roughly
`1.06 * sqrt(60/15) = 2.12`, against a measured 2.94. Isolating it directly by
switching the stagger off, so the book fills as one synchronised cohort:

| stagger | N=15 | N=60 | ratio |
|---|---|---|---|
| off (one cohort) | 0.36 | 0.64 | **1.78x** |
| on (60 cohorts) | 0.44 | 1.37 | **3.11x** |
| decomposition | — | — | 1.06x |

Neither figure matches the decomposition, and the stagger roughly doubles the
effect on top. **The two results were never in conflict; they measure different
objects, and comparing them was the error.** §82's decomposition is right about
the cross-section and was read as if it described the book.

## The consequence, which is the answer to the concurrency question

$$N_{\text{pos}} = H \cdot \lambda$$

is an identity, not a result. §80 closed both of its terms:

* `H >= 60` — finding 2, the excess is a threshold and its CI excludes zero only
  from 60 sessions;
* `lambda` cannot be reduced by entering on a calendar cadence — finding 1, 2 of
  84 configurations admissible, best Sharpe 0.70.

With `H = 60` and `lambda = 1` order a session — which is the operational
maximum a person is willing to run, and is where §84 landed — **`N = 60` is
forced arithmetic.** It is not a portfolio-construction choice that can be
optimised; it is what the two closed levers leave.

Cutting `N` therefore costs bets one-for-one, and the fundamental law prices
that exactly. With `IR = IC * sqrt(BR)` (Grinold 1989) and `BR` proportional to
`lambda`:

| target N | lambda | BR vs N=60 | IR cost | **IC per bet must rise by** |
|---|---|---|---|---|
| 30 | 0.50 | 0.50x | 0.71x | **+41%** |
| 20 | 0.33 | 0.33x | 0.58x | **+73%** |
| 15 | 0.25 | 0.25x | 0.50x | **+100%** |

**That is the price of a smaller book, stated in the only currency that can pay
for it.** A thirty-name version of this strategy is not a weighting question or a
risk question; it requires a selector delivering 41% more information per bet
than random selection among the survivors.

Against that bar, §80 measured the existing composite's top decile at **−1.18%
excess** where a random feature scored **+0.77%** — the current feature set
delivers *negative* incremental information as a selector. Which is precisely why
H3 tests the one untapped source against the one estimand that has never been
used.

---

# 91. Round 15, Part II — H2: the sector cap is a drawdown lever, not a Sharpe lever

## What was tested

A hard cap on **concurrent holdings per sector**, using the cached
`sic_sectors.parquet` (17 sectors). It is a concentration control, not a ranking,
so it cannot smuggle in a selection effect — which is what makes it a legitimate
test of `sigma_common` rather than a disguised selector.

Motivated by the tape: the live screen run this session (2026-08-28) fired ten
names, of which SMJF, KXIN, SUGP, MIMI, RCON and SGLY were all China-domiciled
microcaps unwinding together. Ten slots bought one position six times.

## Development, 15 seeds, one size-aware round trip

| N | cap | Sharpe | ± | annual | maxDD | vs uncapped |
|---|---|---|---|---|---|---|
| 15 | none | 0.46 | 0.39 | +9.3% | −47.6% | — |
| 15 | 1 | 0.51 | 0.34 | +9.8% | −43.0% | +0.04 |
| 15 | 2 | 0.33 | 0.24 | +5.3% | −53.1% | −0.13 |
| 15 | 5 | 0.40 | 0.31 | +7.3% | −50.1% | −0.07 |
| 30 | none | 0.73 | 0.41 | +11.4% | −38.7% | — |
| **30** | **1** | **0.82** | 0.24 | +9.5% | **−21.2%** | +0.09 |
| 30 | 5 | **0.96** | 0.24 | +15.4% | −36.4% | +0.23 |
| 60 | none | **1.37** | 0.25 | +18.4% | −35.8% | — |
| **60** | **1** | 0.90 | 0.24 | +5.3% | **−10.8%** | −0.46 |
| 60 | 2 | 0.92 | 0.30 | +7.4% | −20.2% | −0.45 |
| 60 | 3 | 1.21 | 0.20 | +12.5% | −25.2% | −0.15 |
| 60 | 5 | 1.39 | 0.27 | +17.1% | −32.2% | +0.02 |
| 150 | none | **1.60** | 0.18 | +17.7% | −33.2% | — |
| **150** | **1** | 0.94 | 0.19 | +2.4% | **−4.4%** | −0.66 |
| 150 | 2 | 1.07 | 0.16 | +3.9% | −7.4% | −0.54 |
| 150 | 3 | 1.15 | 0.20 | +5.2% | −11.3% | −0.46 |
| 150 | 5 | 1.44 | 0.19 | +8.7% | −17.4% | −0.17 |

**On its first acceptance clause — "Sharpe at fixed N rises" — H2 fails.** At
N=60, the operative size, every tight cap costs Sharpe: −0.46, −0.45, −0.15, and
+0.02 at a cap so loose it barely binds. Not one delta exceeds its own seed
dispersion.

**But look at the drawdown column.** At N=60 a cap of one takes the maximum
drawdown from **−35.8% to −10.8%** — it removes more than two thirds of the worst
loss — while return falls from +18.4% to +5.3%. Return and risk fall together and
Sharpe falls slightly, which is exactly what a *diversification* constraint does
to a book whose returns are already right-skewed: it removes the concentrated
bets, and those carry both the drawdowns and the upside.

The relationship is monotone and remarkably clean. At N=150 the cap walks the
book down a frontier — Sharpe 1.60 → 1.44 → 1.15 → 1.07 → 0.94 against drawdowns
of −33.2% → −17.4% → −11.3% → −7.4% → **−4.4%**. **This is a risk-preference
frontier, not a Sharpe improvement**, and it is the cleanest monotone
relationship the project has produced.

So `sigma_common` **is** movable — the −10.8% drawdown proves it — but moving it
does not buy Sharpe, because the common factor is carrying return as well as
risk. That is a different finding from H1's, where the volatility filter turned
out not to touch `sd(mu_t)` at all, and the two together say the common factor is
reachable only by composition, and is not free when you reach it.

## The clause that matters is the second one

H2's acceptance criterion had two limbs, and the second is the one aimed at the
actual question: *"or the N needed for a target Sharpe falls."*

With 17 sectors, a cap of one makes the book **physically incapable of holding
more than 17 names**. So "N=60, cap 1" is not a 60-name book — it is a ~17-name
book, and it scores **0.90 with a −10.8% drawdown**. The unconstrained 15-name
book scores **0.46 with a −47.6% drawdown**.

That comparison, if it survives being matched properly on the number of
positions actually held, is the answer to the concurrency question: not "hold
fewer names" but "hold fewer *correlated* names". It is measured in §92.

---

# 92. Round 15, Part II — the answer: hold fewer *correlated* names, not fewer names

## The comparison that had to be matched

There are 17 sectors, so a cap of one name per sector makes the book physically
incapable of holding more than 17 positions no matter how many slots it is given.
"N=60 with cap 1" is not a 60-name book, and §91's table compared books of
different sizes. The comparison that means something matches on **what is
actually held**, measured directly rather than assumed.

Development, 15 seeds, one size-aware round trip, $50,000:

| config | slots | cap | **live positions** | orders/day | Sharpe | ± | annual | maxDD |
|---|---|---|---|---|---|---|---|---|
| unconstrained | 15 | — | **15.1** | 0.25 | **0.46** | 0.39 | +9.3% | −47.6% |
| unconstrained | 20 | — | 20.1 | 0.34 | 0.58 | 0.28 | +10.8% | −44.2% |
| unconstrained | 30 | — | 29.9 | 0.51 | 0.73 | 0.41 | +11.4% | −38.7% |
| unconstrained | 60 | — | 57.3 | 0.97 | 1.37 | 0.25 | +18.4% | −35.8% |
| unconstrained | 100 | — | 97.2 | 1.64 | 1.46 | 0.17 | +16.7% | −33.4% |
| **sector cap 1** | 30 | 1 | **15.4** | 0.26 | **0.82** | 0.24 | +9.5% | **−21.2%** |
| **sector cap 1** | 60 | 1 | **15.4** | 0.26 | **0.90** | 0.24 | +5.3% | **−10.8%** |
| **sector cap 1** | 100 | 1 | 15.5 | 0.26 | **0.92** | 0.19 | +3.4% | **−7.0%** |
| sector cap 2 | 30 | 2 | 28.4 | 0.48 | 0.68 | 0.33 | +9.8% | −34.8% |
| **sector cap 2** | 60 | 2 | **29.3** | 0.49 | **0.92** | 0.30 | +7.4% | **−20.2%** |
| **sector cap 2** | 100 | 2 | **30.0** | 0.50 | **0.93** | 0.19 | +5.0% | **−11.7%** |

## The result

**At the same book size and the same order rate, sector diversification roughly
doubles the Sharpe and halves the drawdown.**

| | unconstrained | sector-capped |
|---|---|---|
| live positions | 15.1 | 15.4 |
| orders per session | 0.25 | 0.26 |
| **Sharpe** | **0.46** | **0.82 – 0.92** |
| **max drawdown** | **−47.6%** | **−21.2% to −7.0%** |

Nothing operational differs. The same number of line items, the same number of
orders, the same capital, the same signal, the same filters. The only change is
a refusal to hold two names from one sector at the same time.

Read against the unconstrained curve, a Sharpe of 0.82 would otherwise require
roughly **35 positions** (interpolating 30 → 0.73 and 60 → 1.37). So the
constraint delivers the performance of a ~35-name book in **15 names** — a
**57% reduction in concurrent positions at no cost in Sharpe**, and a large gain
in drawdown.

**The effect holds at 30 names as well as 15**, and the row that does not win
says why. At 30 slots a cap of two does not bind — 17 sectors x 2 = 34 > 30 — so
that book is effectively unconstrained (28.4 live, Sharpe 0.68 against 0.73) and
scores like one. Give it 60 or 100 slots so the cap actually binds and the same
~30-name book scores **0.92 and 0.93** against the unconstrained 0.73, with the
drawdown falling from −38.7% to −20.2% and −11.7%.

| live positions | unconstrained | sector-capped | delta | maxDD unc. | maxDD capped |
|---|---|---|---|---|---|
| ~15 | 0.46 | **0.90** | **+0.44** | −47.6% | −10.8% |
| ~30 | 0.73 | **0.93** | **+0.20** | −38.7% | −11.7% |

The gain shrinks as the book grows, which is exactly what the decomposition
predicts: `sigma^2_eta / N` vanishes with breadth and `sigma^2_common` does not,
so managing the common term matters most where breadth is scarcest.

**H2's second acceptance clause passes on development.** Its first clause —
"Sharpe at fixed N rises" — failed, and that failure was real: at a fixed 60
slots the cap costs Sharpe. But the clause that was aimed at the actual question,
"the N needed for a target Sharpe falls," passes clearly.

## Why this works, and why nothing before found it

§80 fixed the identity `N = (H/tau) * K * (1-delta)` and closed two of its three
terms. Every subsequent attempt to cut `N` moved a term of that identity, and
each cost bets one-for-one. **The sector cap does not appear in the identity at
all.** It leaves `H`, `tau` and `K` exactly where they were and changes the
*composition* of the names filling the slots — which changes `sigma_common`, the
term §82's decomposition treats as a constant of nature.

H1 showed that constant is untouched by the volatility filter: `sd(mu_t)` is
24.6% uncapped and 24.3% vol-capped. The filter is purely an idiosyncratic-risk
filter. **The common factor was never being managed at all**, by anything, in
fourteen rounds — and it is the term that dominates at high `N`, because
`sigma^2_eta / N` vanishes and `sigma^2_common` does not.

That is also why the effect is invisible in every per-trade statistic the project
has computed. The trades are identical. Only which ones are held *at the same
time* changes.

## Honest limits

1. **The Sharpe delta is about one combined seed standard deviation.** 0.46 ± 0.39
   against 0.82 ± 0.24 on a single comparison. The drawdown difference is far
   larger relative to its dispersion, and the monotone frontier across caps 1, 2,
   3 and 5 at every `N` (§91) is stronger evidence than the point estimate.
2. **Development only.** The holdout read is separate and is recorded when taken.
3. **The sector labels are SIC-derived and static**, not point-in-time. A
   reclassification is not a look-ahead in any strong sense — sector membership
   does not change with price — but it is not a clean point-in-time join either.
4. **17 sectors is a coarse grid.** A cap of one is therefore a hard ceiling of 17
   names, which is why the capped rows converge to the same book whatever slot
   count they are given. A finer partition — SIC industry, or a statistical
   clustering of return correlations — would give a continuous lever instead of a
   ceiling, and that is the obvious next step rather than a limit of the idea.
5. The return of the capped book is **much lower** (+3.4% to +9.5% against
   +18.4%). This is not a free improvement; it is a different point on a risk
   frontier, and a person who wants +18% a year still needs the 60-name book.

---

# 93. Round 15, Part II — H4: the inverse-volatility arm was a no-op, and once fixed it does nothing

## The defect, found because the numbers were identical to the digit

H4's first pass returned **0.46 / 0.46** at N=15 and **0.73 / 0.73** at N=30 for
equal and inverse-volatility weighting — the same to two decimals, with the same
drawdown and the same seed dispersion. Two weighting schemes do not agree that
exactly.

`risk_shaped.py:120`, which `strategy/book.py` inherited:

```python
iv = 1.0 / pool["B7_realised_vol"].to_numpy()
w  = iv / iv.mean() / cfg.n_slots
```

`pool` at that point is the set the fill rule has **already sampled down to**,
and the stagger two lines earlier admits `ceil(n_slots / H)` fills a session —
which is **one** for every `n_slots <= H`. With one name in the pool,
`iv / iv.mean()` is identically 1 and the expression returns `1 / n_slots`: the
equal-weight book, exactly.

**Every inverse-volatility result this project has reported at N ≤ 60 was an
equal-weight result.** That covers §80's weighting comparison and §82's, at every
retail configuration. The fix is to normalise against the day's **eligible** set
rather than the sampled subset, and it is pinned by a test that asserts the
weights vary and that low-volatility names carry the larger share.

## H4 re-run, with the tilt actually applied

Development, 15 seeds. `wt sd` is the cross-sectional dispersion of weights
scaled by `N` — zero for equal weight by construction, and non-zero here only
because the fix works.

| N | weight | Sharpe | ± | annual | maxDD | wt sd | delta |
|---|---|---|---|---|---|---|---|
| 15 | equal | 0.46 | 0.39 | +9.3% | −47.6% | 0.000 | — |
| 15 | invvol | 0.35 | 0.37 | +6.2% | −50.0% | 0.355 | **−0.11** |
| 30 | equal | 0.73 | 0.41 | +11.4% | −38.7% | 0.000 | — |
| 30 | invvol | 0.83 | 0.35 | +12.8% | −35.7% | 0.265 | **+0.10** |
| 60 | equal | 1.37 | 0.25 | +18.4% | −35.8% | 0.000 | — |
| 60 | invvol | 1.37 | 0.27 | +17.7% | −34.2% | 0.303 | **−0.00** |

**Nothing.** −0.11, +0.10, −0.00 — signs that do not agree, magnitudes far inside
the seed dispersion. Inverse-volatility weighting neither helps nor hurts.

That is the *expected* answer and it is worth stating why, because a null result
with a reason behind it is more useful than one without. §78 established
`E[R]` is **flat in the score** inside the surviving population, and flat expected
returns is precisely the condition under which equal weight is optimal — there is
no expected-return dispersion to tilt toward. The volatility cap has already
removed the high-volatility half, so tilting further within the low-volatility
survivors is re-weighting on a variable whose informative range has been spent.

## Kelly, since position size keeps being raised as a lever

The growth-optimal fraction solved numerically on the measured per-trade payoff
(mean +13.47%, sd 49.3%) is **f\* = 0.629**, half-Kelly 0.315.

That number is not directly comparable to the book's 1/60 = 1.67% a name, because
Kelly on a single sequential bet is not Kelly on sixty concurrent correlated
ones. The comparison that does hold is at the aggregate: the book is **long-only
and fully invested**, so there is no leverage headroom to exploit in either
direction. Position sizing is not a free lever here — **breadth is the binding
constraint, not leverage**, and that is what H1's arithmetic already said.

---

# 94. Round 15, Part II — H2 on the sealed holdout: the drawdown effect replicates, the Sharpe effect half does

This is the **sixth** read of the sealed holdout (§78, §80, §82, §84, Part I's
re-cost, and this). Recorded as such.

| config | slots | cap | live | ord/day | Sharpe | ± | annual | maxDD |
|---|---|---|---|---|---|---|---|---|
| unconstrained | 15 | — | 15.1 | 0.25 | 1.07 | 0.55 | +25.5% | −21.8% |
| unconstrained | 20 | — | 20.0 | 0.34 | 1.04 | 0.72 | +21.5% | −23.7% |
| unconstrained | 30 | — | 29.8 | 0.51 | 0.98 | 0.57 | +15.7% | −26.0% |
| unconstrained | 60 | — | 58.2 | 1.00 | **1.51** | 0.45 | +17.2% | −19.5% |
| unconstrained | 100 | — | 97.5 | 1.68 | 1.86 | 0.46 | +17.4% | −16.1% |
| sector cap 1 | 30 | 1 | 16.3 | 0.28 | 1.05 | 0.36 | +12.6% | −14.8% |
| sector cap 1 | 60 | 1 | 16.3 | 0.28 | 1.10 | 0.36 | +6.6% | −7.5% |
| sector cap 1 | 100 | 1 | 16.5 | 0.28 | 0.94 | 0.28 | +3.4% | −5.0% |
| sector cap 2 | 30 | 2 | 29.3 | 0.51 | **1.30** | 0.48 | +21.0% | −21.9% |
| sector cap 2 | 60 | 2 | 31.0 | 0.53 | 1.26 | 0.47 | +10.6% | −12.9% |
| sector cap 2 | 100 | 2 | 31.6 | 0.54 | 1.20 | 0.36 | +6.2% | −7.9% |
| **sector cap 3** | 60 | 3 | **44.2** | 0.76 | **1.59** | 0.43 | +15.6% | **−14.9%** |
| **sector cap 3** | 100 | 3 | 46.2 | 0.80 | **1.66** | 0.36 | +10.5% | −9.6% |

## Matched pairs, both periods

| live | dev unc. | dev capped | **holdout unc.** | **holdout capped** |
|---|---|---|---|---|
| ~15 | 0.46 | **0.90** (+0.44) | **1.07** | **1.10** (+0.03) |
| ~30 | 0.73 | **0.93** (+0.20) | **0.98** | **1.30** (+0.32) |
| ~45 | ~1.0 | 1.21 (+0.2) | ~1.2 | **1.59** (+0.4) |

**The N=15 result does not replicate.** Development showed +0.44 and the holdout
shows +0.03. That was the largest effect in the development table and it is the
one that vanished — the oldest pattern in this project, and the reason the
holdout exists.

**The ~30 and ~45 comparisons do replicate**, and more strongly out of sample than
in it: +0.32 at 30 names and about +0.4 at 45.

**The drawdown effect replicates everywhere and is large.** At ~16 names, −21.8%
becomes −14.8% / −7.5% / −5.0%. At ~30, −26.0% becomes −21.9% / −12.9% / −7.9%.
At ~45, −19.5% becomes −14.9%. In every pair, in both periods, the capped book
draws down less.

## The configuration this actually supports

Against the standing recommendation — 60 names, unconstrained:

| | dev Sharpe | holdout Sharpe | dev maxDD | holdout maxDD | live positions |
|---|---|---|---|---|---|
| unconstrained, 60 slots | **1.37** | **1.51** | −35.8% | −19.5% | 57–58 |
| **cap 3, 60 slots** | 1.21 | **1.59** | **−25.2%** | **−14.9%** | **41–44** |

**A per-sector cap of three cuts the book from ~58 names to ~42 — a 28%
reduction — at no reliable cost in Sharpe (−0.16 dev, +0.08 holdout, both inside
the seed dispersion of 0.2–0.45) and with a materially better drawdown in both
periods.**

That is the concurrency answer that survives a holdout: not a halving, and not
the doubling of Sharpe development advertised at 15 names, but a real ~28%
reduction with better risk, from a constraint that costs nothing to implement and
does not touch the identity `N = H * lambda` at all.

## Why the small-book result failed and the medium one held

Consistent with the mechanism rather than with luck. With 17 sectors a cap of one
is a hard ceiling of 17 names, so it does not merely diversify — it *forces* the
book to sit out days when its sectors are full, cutting the trade count and the
return (annual +3.4% to +12.6% against +25.5%). That is a different strategy, not
a diversified version of the same one. A cap of two or three binds often enough to
decorrelate and rarely enough to keep the book invested, and those are the ones
that replicate.

---

# 95. Round 15, Part II — the fifth adjusted/unadjusted failure, caught by a number that was too good

## What happened

H3 and H5 needed minute bars, and the cache covered only **12.7%** of the
round-15 population — it had been fetched for earlier rounds' screens, which
select an entirely different population. So 1,654 sessions of the 2020-2026
candidate set were fetched (98,378 symbol-days, taking the cache from 921 MB to
1.2 GB), and both hypotheses were run on it.

H5's first result was that entering at the daily open returned **+15.35%** gross
against **+1.85%** for the minute open — *the same session, the same bar*. An
execution difference cannot be eight times the return.

Pairing the two directly, on the same symbol-day:

| minute open / daily unadjusted open | |
|---|---|
| within 1% | **64.2%** |
| within 5% | 67.9% |
| 95th percentile | **20.0** |
| 99th percentile | **40.0** |

SOS on 2021-03-26: daily open **6.22**, minute open **4672.50**. Ratio
**750.000** — to six figures.

`alpaca_fetch.py:127` requests minute bars with **`adjustment="all"`**. The
entire minute cache is split- and dividend-**adjusted**, while
`panel_strategy.parquet` carries the raw `*_unadj` series that every filter and
every forward path in this project uses. They are two different price bases and
they had been paired.

## The feature this killed, which was the best one

`M_mean_trade_usd` — mean dollars per trade, a retail-participation proxy —
scored **IC −0.0528, t −6.82**: the second-strongest feature in the entire
project, behind only realised volatility, and one of three to survive BH-FDR
against all 287 prior trials.

It is a **price level times volume**, computed from adjusted prices. A name that
later reverse-splits carries an inflated adjusted dollar volume *today*, and
names reverse-split because they keep falling. **High adjusted trade size implies
a future consolidation implies worse returns.** That is §72's bias exactly,
rediscovered in a new data source, and its t-statistic of −6.82 is how strong a
look-ahead can look.

Withdrawn. Every remaining feature is **scale-invariant within the session** — a
ratio, a share, or a standard deviation of log returns — so a constant adjustment
factor cancels exactly, and a unit test now rescales a synthetic session by 750x
and asserts that no emitted feature moves.

The idea survived the implementation: `M_shares_per_trade`, the same notion
without the price, is one of the two features that beats the barrier base rate
(§96).

## H5, once the comparison is valid

The fix is to use minute prices only as **within-session ratios** and apply those
to the unadjusted daily open — the ratio is unaffected by a constant factor. All
four rules on the common subsample where every one is priced (21,234 rows):

| entry | touch % | mean gross | median | sd | IR/position | vs open |
|---|---|---|---|---|---|---|
| daily open | 18.2% | +15.35% | +4.33% | 50.7% | 0.303 | — |
| minute open | 18.2% | +15.35% | +4.33% | 50.7% | 0.303 | +0.000 |
| **10:00 print** | 18.7% | **+16.25%** | +5.24% | 51.0% | **0.319** | **+0.016** |
| 09:30–10:00 VWAP | 18.6% | +16.00% | +4.91% | 50.9% | 0.314 | +0.012 |

The second row is the check: the minute open scaled by its own ratio to itself
must reproduce the daily open exactly, and it does to five decimals.

**Waiting until 10:00 is worth about +5% of information ratio per position** —
0.303 to 0.319, mean gross +15.35% to +16.25%. Small, positive, and in the
direction the microstructure literature predicts (Heston, Korajczyk & Sadka 2010):
the open is the widest-spread minute of a microcap session and there is no
compensating drift to capture, which §71 had already established from the other
side.

It is not a large enough effect to change the recommendation on its own, and it
is development-only. It is also free — the order is placed thirty minutes later.

## The rule this adds

**A cached bar's adjustment basis is part of its schema, and must be asserted
where the two meet.** Four previous failures (§57, §64, §67, §72) were all inside
one data source. This one is *between* two sources that were each internally
correct. The daily panel is scrupulous about `_unadj`; the minute cache is
adjusted and never said so; nothing in between checked.

Cheap defences, both now in place: features from a possibly-adjusted source must
be scale-invariant, and a level from one source may only meet a level from
another through a ratio.

---

# 96. Round 15, Part II — H3: a selector from the minute tape

## Why this was the only place left to look

§80 measured the existing composite as a selector and it is dead: the top decile
returned **−1.18%** excess where a *random* feature returned **+0.77%**. Every
non-price block died in `6cb0b55`. The minute cache was the one source never
turned into cross-sectional features, and the only reason it was available at all
is that this round fetched 1,654 sessions of the round-15 population — the
existing cache covered **12.7%** of it, having been built for earlier screens.

Coverage after the fetch: **96.8%** of the development frame.

## (a) Scored the way every block has been scored

Within-cell IC on the demeaned return, `cross_section.score_feature`, cells
bootstrapped rather than observations. 973 cells.

| feature | IC | t | 95% CI |
|---|---|---|---|
| **M_close_participation** | **+0.0389** | **5.20** | [+0.0243, +0.0541] |
| **M_volume_spikiness** | **+0.0328** | **4.13** | [+0.0175, +0.0488] |
| M_open_participation | −0.0205 | −2.65 | [−0.0361, −0.0051] |
| M_minutes_traded | −0.0197 | −2.40 | [−0.0363, −0.0039] |
| M_vwap_position | +0.0159 | 2.00 | [+0.0005, +0.0316] |
| M_intraday_vol | −0.0112 | −1.47 | [−0.0261, +0.0044] |
| *B0_random (control)* | *+0.0070* | *0.92* | *[−0.0073, +0.0222]* |

**Three survive BH-FDR padded with all 287 prior trials**, and the random control
sits at t = 0.92 where it belongs. These are the first non-composite features in
the project to clear that bar.

They are also, on their own, not tradable: the rank-weighted long-short net
return is negative for every one of them (−0.72% for the best IC). An IC of 0.039
against a cost floor is a real effect that cannot pay for itself as a
dollar-neutral portfolio — which is the same wall §78 hit, and the reason the
second estimand exists.

## (b) Scored the way a barrier strategy actually pays

`P(touch the +100% barrier | top decile within the day)`, ranked within the day so
the day effect cannot masquerade as selection. Base rate **17.1%**.

| feature | top 10% | lift | 95% CI | bottom 10% |
|---|---|---|---|---|
| **M_intraday_vol** | **22.9%** | **1.34** | [21.0%, 24.8%] | **6.4%** |
| **M_shares_per_trade** | **21.8%** | **1.28** | [19.9%, 23.6%] | 12.4% |
| M_vwap_position | 18.7% | 1.10 | [16.4%, 20.9%] | 24.6% |
| *B0_random (control)* | *17.4%* | *1.02* | *[15.2%, 19.7%]* | *17.2%* |

Two features beat the base rate with intervals excluding it, and the control does
not. **Intraday realised volatility on the crash day raises the barrier-touch rate
from 17.1% to 22.9%, and its bottom decile falls to 6.4%** — a spread of nearly
four to one across the same population.

Note that `M_intraday_vol` scores **nothing** on the IC (t = −1.47, wrong sign).
It is invisible to the estimator every previous block was judged by, and it is the
strongest thing here. That is the whole argument for the second estimand: with a
barrier exit, which names *touch* is not the same question as which names have a
higher mean.

Economically it is coherent rather than mysterious. The population is already
filtered to the **low** half by *daily* realised volatility, so the top decile by
*intraday* volatility is the set of normally-quiet names that had one violent
session. A stock that just moved violently is more likely to move violently
again — and a +100% barrier needs exactly that.

## (c) As a fill rule — the only form that can reduce the book

Same population, same filters, same horizon, same stagger, same size-aware cost.
Only the rule choosing which survivor fills a free slot changes. Development,
15 seeds for the random arm (the ranked arms are deterministic, hence ± 0.00).

| N | fill rule | orders/day | Sharpe | annual | maxDD | vs random | vs control |
|---|---|---|---|---|---|---|---|
| 15 | random | 0.25 | 0.46 | +9.3% | −47.6% | — | — |
| 15 | **top M_intraday_vol** | 0.25 | **1.02** | **+23.5%** | −49.1% | +0.55 | **+1.03** |
| 15 | top M_shares_per_trade | 0.25 | 0.65 | +12.9% | −38.3% | +0.19 | +0.66 |
| 15 | *top B0_random* | 0.25 | *−0.01* | *−2.9%* | *−63.2%* | *−0.47* | — |
| 30 | random | 0.51 | 0.73 | +11.4% | −38.7% | — | — |
| 30 | **top M_intraday_vol** | 0.51 | **1.38** | **+23.3%** | −36.3% | +0.65 | **+1.02** |
| 30 | top M_shares_per_trade | 0.51 | 1.25 | +22.1% | −33.8% | +0.51 | +0.89 |
| 30 | *top B0_random* | 0.51 | *0.36* | *+4.7%* | *−58.6%* | *−0.37* | — |
| 60 | random | 0.97 | 1.37 | +18.4% | −35.8% | — | — |
| 60 | **top M_intraday_vol** | 0.97 | **1.89** | **+29.7%** | **−25.8%** | +0.53 | +0.61 |
| 60 | top M_shares_per_trade | 0.97 | **2.10** | **+31.8%** | −30.8% | +0.74 | +0.82 |
| 60 | *top B0_random* | 0.97 | *1.28* | *+16.9%* | *−40.1%* | *−0.08* | — |

The control matters and it behaves: filling deterministically on a **random**
column scores at or below the random fill at every N, so the gain is the ranking
and not the determinism.

**On development, a 30-name book filled by intraday volatility scores 1.38 — the
same as the 60-name random book's 1.37, at half the positions and +23.3% annual
against +18.4%.** That is the concurrency answer in its strongest form.

## The cost objection, tested rather than assumed

The selector prefers thinner, more volatile names — median ADV **\$0.73M** against
**\$2.48M** for a random fill. The ADV-binned cost model does charge it more,
**127 bps against 91**, a ratio of 1.40x. But the model bins on ADV and cannot see
volatility, and spread scales with volatility, so it might underprice exactly what
the selector picks.

Measured independently with the **Corwin–Schultz (2012)** high-low estimator,
which uses only daily OHLC and never touches the quote data the cost model was
fitted on:

| | median CS spread |
|---|---|
| top decile by intraday vol | **256 bps** |
| the rest | **178 bps** |
| ratio | **1.43x** |

Against the cost model's 1.40x. **The extra cost of the selector's names is
already being charged, to within about 2%** — roughly 3 bps on a 127 bps round
trip. The objection is real and it is covered.

---

# 97. Round 15, Part II — H3 on the sealed holdout: both selectors die, one of them twice

Seventh read of the holdout, and it kills the round's best-looking result.

| N | fill rule | trades | **dev Sharpe** | **holdout Sharpe** | dev vs random | holdout vs random |
|---|---|---|---|---|---|---|
| 15 | random | 165 | 0.46 | 1.07 | — | — |
| 15 | top M_intraday_vol | 165 | **1.02** | **0.14** | +0.55 | **−0.93** |
| 15 | top M_shares_per_trade | 165 | 0.65 | 2.70 | +0.19 | +1.63 |
| 15 | *top B0_random* | 165 | *−0.01* | *1.14* | *−0.47* | *+0.06* |
| 30 | random | 330 | 0.73 | 0.98 | — | — |
| 30 | top M_intraday_vol | 330 | **1.38** | **0.09** | +0.65 | **−0.89** |
| 30 | top M_shares_per_trade | 330 | 1.25 | 3.63 | +0.51 | +2.65 |
| 30 | *top B0_random* | 330 | *0.36* | *1.40* | *−0.37* | *+0.42* |
| 60 | random | 648 | 1.37 | 1.51 | — | — |
| 60 | top M_intraday_vol | 648 | **1.89** | **0.05** | +0.53 | **−1.46** |
| 60 | top M_shares_per_trade | 648 | 2.10 | 4.62 | +0.74 | +3.11 |

## M_intraday_vol: refuted

Development +0.55 / +0.65 / +0.53. Holdout **−0.93 / −0.89 / −1.46**. Not a
weakening — a sign flip of the same magnitude, at every book size, monotonically
worse as N rises. A feature chosen as the best of fourteen on one period,
behaving as the worst on the next.

Its development evidence was not weak, either: it was the only feature whose
barrier-touch interval excluded the base rate at both ends (22.9% top decile,
6.4% bottom, base 17.1%), it had a calibrated random control, and its economic
story was coherent. **None of that survived contact with 2024–2026.**

## M_shares_per_trade: a look-ahead, and it got past a guard written for it

Sharpe **4.62** at N=60 with a −7.0% maximum drawdown and +65% a year. Nothing in
this universe does that, and §95 had just finished withdrawing its sibling.

`M_mean_trade_usd` was withdrawn because a price *level* from an adjusted cache
encodes future corporate actions. `M_shares_per_trade` was its replacement with
the price removed — and **Alpaca adjusts VOLUME as well, in the opposite
direction**. A name that later reverse-splits carries a *deflated* adjusted share
count today, so ranking on shares-per-trade ranks on not-about-to-consolidate.

Measured on the holdout population:

| | share with adj/unadj > 1.5 |
|---|---|
| top decile by M_shares_per_trade | **0.0%** |
| the rest | **7.4%** |

Spearman with the adjustment ratio **−0.307**; with `1/price`, **+0.464**. The
top decile contains **no** future reverse-splitters at all. It is a survivorship
screen built from tomorrow's corporate actions — §72's retraction mechanism
exactly, this time pointing the flattering way.

**The scale-invariance test written in §95 did not catch it, because it rescaled
prices and left volume alone.** A split adjustment moves price and volume in
*opposite* directions; a guard that tests one is not a guard. Fixed: the test now
rescales price by 750x and volume by 1/750x, and no trade-size feature is emitted
in any form.

## The control is the most useful column in the table

`B0_random` — a fixed random number per row, used as a fill rule — moves the
holdout Sharpe by **+0.06 at N=15 and +0.42 at N=30**. A meaningless column
appears to improve the book by four tenths of a Sharpe.

That is the measurement error on every deterministic single-draw arm in this
table, and it is larger than most effects the project has chased. The random-fill
arm averages 15 seeds and has a seed dispersion of 0.45–0.57 on the holdout; the
ranked arms are one draw each and have no dispersion to report, which makes them
look precise and is why the control had to be there.

## Verdict

**H3 fails.** No intraday feature reduces the position count. The IC results in
§96 stand as measurements — three features clear BH-FDR against 287 prior trials
and the random control is calibrated — but an IC of 0.039 with a negative
long-short net return was never tradable, and the two features that looked
tradable are one overfit and one look-ahead.

The minute cache is now built and correct for the round-15 population (1,654
sessions fetched, 96.8% development and 99.7% holdout coverage), which is the
durable output. What it does not contain is a selector.

---

# 98. Round 15 — what it recommends, and the lesson

## The question, answered

*Can the book hold fewer than sixty positions?*

**Yes, about 28% fewer, and the reason is not the one anyone was looking for.**

`N = H * lambda` is an identity, not a result. §80 closed `H` (≥60, the excess is
a threshold) and `tau` (calendar cadence, 2 of 84 configurations admissible), so
with one order a session as the operational limit, **N=60 is forced arithmetic**.
Cutting it costs bets one-for-one, and the fundamental law prices that exactly:
a 30-name book needs **+41%** more information per bet, a 15-name book **+100%**.

Neither of the two ways to buy that information worked:

* **A selector** would raise IC per bet. H3 tested fourteen intraday features
  against the estimand that actually pays. On development the best added +0.6
  Sharpe at every N; on the holdout it **inverted** (−0.9 to −1.5), and the
  runner-up turned out to be a look-ahead. §80's verdict on selectors stands.
* **Better weighting** would raise the information transferred. H4 is a flat
  null once its no-op was fixed — and that was the expected answer, because §78
  found expected return **flat in the score** among survivors, which is exactly
  when equal weight is optimal.

What did work is not in the identity at all. **A cap on concurrent holdings per
sector changes which names occupy the slots, not how many slots there are** — so
it moves `sigma_common`, the term §82's decomposition treats as a constant and
which H1 showed the volatility filter never touched (`sd(mu_t)` 24.6% uncapped,
24.3% capped).

## The configuration

> Crashed microcaps: fell ≥15% over five sessions, \$0.50–\$10 unadjusted,
> ADV ≥ \$250k. Keep the lower half by 20-day realised volatility. **60 slots
> with at most three names per sector held at once.** Equal weight, at most one
> entry per session, exit at +100% or after 60 sessions, enter at the next open —
> or at 10:00, which is worth about +5% of information ratio and costs nothing.

| | dev Sharpe | holdout Sharpe | dev maxDD | holdout maxDD | live positions |
|---|---|---|---|---|---|
| §84's recommendation | **1.37** | **1.51** | −35.8% | −19.5% | 57–58 |
| **with a cap of 3** | 1.21 | **1.59** | **−25.2%** | **−14.9%** | **41–44** |

Sharpe is a wash (−0.16 development, +0.08 holdout, both well inside the seed
dispersion of 0.20–0.45). The drawdown is better in **both** periods. The book is
**28% smaller**. The constraint costs nothing to implement and needs no new data —
`sic_sectors.parquet` has been on disk since round 11.

For a smaller book still, a cap of two gives ~30 names at holdout Sharpe 1.30
against the unconstrained 30-name book's 0.98 — but that is a 0.2 Sharpe give-up
against the 60-name configuration, and it is a choice about how many line items a
person will tolerate, not a free lunch.

**What is not recommended:** a cap of one. It is a hard ceiling of 17 names, it
was the largest effect on development (+0.44), and it is the one that vanished on
the holdout (+0.03). The oldest pattern in this project.

## What this round actually produced

The recommendation moved 28%. The infrastructure moved further.

**Built:** `strategy/book.py`, one simulator replacing three that disagreed;
`eval/battery.py`, five tiers of metrics with 38 tests against closed forms;
`PREREGISTRATION_R15.md` with the mathematics and 33 references; the minute cache
rebuilt for the actual population, 1,654 sessions, 96.8%/99.7% coverage.

**Found, in code that had been shipping results:**

1. The round-14 table came from **uncommitted code**; `cost_model.py` was
   imported by nothing.
2. Cost was charged **1.5 round trips** on the time-stop path and **0.5** on the
   profit-target path — a threefold difference between two arms of one sweep.
3. `slots.py`'s "noveto" control still ranked on the composite, so
   `breadth_curve.py` changed two things at once.
4. Exits were scheduled on **candidate days**, not trading sessions.
5. The inverse-volatility arm was **a no-op at every N ≤ 60** in every prior
   round — `iv / iv.mean()` over a one-name pool.
6. §84's `orders/day` column is the per-session **cap**, not the realised rate;
   "N=60 costs the same effort as N=30" is wrong.
7. §84's N=30 row does **not reproduce** at any account size.
8. Two look-aheads in new features, §95 and §97, both from the same source.

**Measured for the first time:** 972 trades carry ~17 independent time
observations; daily returns are autocorrelated to lag 60 and the honest bootstrap
interval is five times the Mertens standard error; realised drawdowns run 2.4–2.9x
their own expectation at N ≥ 60; per-trade tail indices are 3.7–11.3, so the
variance arithmetic every breadth argument rests on is valid.

## 99. The lesson

**A guard written for one direction of a bias does not cover the other.**

§95 withdrew `M_mean_trade_usd` because an adjusted price *level* encodes future
corporate actions, and added a test that rescales a synthetic session's prices by
750x and asserts no feature moves. `M_shares_per_trade` — the same idea with the
price removed — passed that test and reached the holdout at Sharpe 4.62, because
**a split adjustment moves price and volume in opposite directions** and the test
only moved price.

The general form: an adjustment is a transformation of *several* fields at once.
A scale-invariance guard must apply the whole transformation, not the half of it
that motivated the guard. The corrected test rescales price by 750x **and** volume
by 1/750x.

This is the sixth adjusted/unadjusted failure in this project (§57, §64, §67,
§72, §95, §97), and the first two of those were seven rounds ago. The pattern is
not carelessness about adjustment — the codebase is scrupulous about it. It is
that each new *pairing* of data sources creates a new place for the same bias,
and the check has to live where the sources meet.

---

# 100. The consolidated specification

Rounds 1-15 are recorded in this log as a narrative, which is the right shape for
process and the wrong shape for a design. **`research/penny/STRATEGY_SPEC.md`**
is the design: the strategy as it stands, every element of it, and the catalogue
of what has been tested against what is still open.

It exists because the answer to "what is the strategy" was spread across ninety
nine sections, and because the answer to "what should be tried next" was not
written down anywhere at all. The live candidates it names, in order:

1. **A finer partition than SIC sector** -- a correlation-clustered tree instead
   of 17 buckets, turning the one lever that replicated (SS92, SS94) into a
   continuous one rather than a 17-name ceiling.
2. **Balance sheet: cash / market cap and net cash.** Specified in
   `FORMULATION.md` as block B4 and never fetched -- `etl/xbrl.py` pulls share
   counts only. B4's dilution half is dead; its cash half has never been tested.
3. **ATM program size and warrant overhang**, open since SS21.
4. **A conditional release rule** -- H=60 as a maximum rather than a fixed term.
   Since `N = H * lambda`, cutting the AVERAGE hold cuts the book. It must be
   tested as a time/state release and never as a price stop, which is dead twice.

And the one it rules out on evidence rather than taste: **breakout and
continuation patterns**. The 441-cell grid tested momentum, gap-ups, new 20-day
highs and volume shocks, and every family was negative gross. In this universe
continuation is the losing side.

---

# 101. Round 17 — R15 measured the way it would be traded: killed on dev

Pre-registered in `PREREGISTRATION_R17.md` (commit `8654541`, pushed before any
causal number existed). Dev only: market sessions 2020-01-02..2023-12-29, panel
truncated at 2023-12-29 before anything read it. **The holdout was not
touched.** Trial count 317.

## Why it was re-measured

R15 was about to drive a live account, and its numbers had four defects:

1. The frame dropped candidates using their future (a flagged split in the next
   61 bars, or a missing forward return, `cross_section.py:89-90`). It did this
   before the within-day ranking, so it changed which names were selected.
2. The equity curve was realised-only: positions carried at cost until
   signal + 60 (`book.py:329`).
3. Dev windows reached into the sealed period.
4. The holdout had already been read 7 times.

A data audit that read only prices settled what "causal" had to mean. Of the
panel's split flags inside dev holding windows, 84% have no Alpaca
corporate-action record. They are genuine gaps on near-round multiples, so
defect 1 removed real trades, not artifacts.

## Reproduction first

`r15_matched_book.run`, 60 slots, cap 3, 15 seeds, dev: **Sharpe 1.2138 ±
0.2015, +12.5%/yr, maxDD −25.2%, 41.0 live**. That is §91 exactly, and it was
unchanged after `book.py` gained its MTM mode.

## Result (15 seeds; paired stationary bootstrap, block 60, 2,000 reps)

| series | Sharpe | 95% CI | CAGR | maxDD | total |
|---|---|---|---|---|---|
| **A** causal, no stop | **0.68** (seed sd 0.16) | [−0.27, 1.73] | +15.1% | −31.7% | +76.6% |
| B causal, −50% stop | 0.56 (0.19) | [−0.44, 1.66] | +11.0% | −31.9% | +53.1% |
| SPY (total return) | 0.61 | | +11.7% | −33.8% | +55.3% |
| QQQ | 0.76 | | +18.4% | −34.7% | +95.6% |
| IWM | 0.36 | | +6.4% | −41.2% | +28.0% |

**Cell A against the benchmarks:**
- vs SPY: Sharpe +0.07, bootstrap CI [−0.62, +0.66], Ledoit–Wolf p 0.80.
- vs QQQ: −0.08, CI [−0.80, +0.67].
- Alpha vs SPY: +5.6%/yr, Newey–West CI [−11.4, +22.6], t 0.65.
- Beta: 0.84 to SPY, 0.77 to IWM, 0.61 to QQQ.

**Diagnostics:**
- Fills: the limit at prior close × 1.10 filled 99.2% of the time.
- Positions: 40.7 live on average. Cost: 81 bps mean.
- Exits: 83% time, 10% target, 5.7% open at the cutoff, 0.8% delisted.
- Closed trades: +5.9% mean net, −2.1% median.
- Deflated Sharpe at N = 317: 0.07.
- Split corrections: one vendor split was missing from the adjusted series, and no traded position carried it.

## Decisions, as pre-registered

- **Stop:** B's Sharpe is 0.12 lower and its drawdown is no shallower, so the
  stop is **rejected**. "Every stop hurts" holds a third time.
- **Kill:** cell A's Sharpe CI includes 0. **R15 is not taken live.** No holdout
  read #8 follows.

## Where the 1.21 went (labelled diagnostic, not a cell)

`scripts/r17_decompose.py` D1 books cell A's own causal trades both ways, gross
of cost, over 15 seeds:

| | Sharpe |
|---|---|
| marked to market | **0.77** |
| realised-only booking (carried at cost, P&L on release) | **1.63** |

**The same trades score more than twice as high when the open P&L is hidden.**
Most of R15's 1.21 was the accounting, not the edge. Gross against net is not
like for like, but cost runs about 2.3%/yr of equity (81 bps × ~172 trades a
year ÷ 60 slots), and that is too small to take 1.63 down to 1.21. So the
causal frame and paths do not appear to have made the trades worse. D2 was
meant to show this directly.

The run was stopped by the session's memory reaper before D2 (what the leak
dropped) and D3 (path conventions) finished. They are recorded as **not run**.

## What R15 actually is

A crashed-microcap book with beta 0.84 to SPY and 0.77 to IWM. Its point Sharpe
sits between SPY's and QQQ's, and no interval separates it from either. Its
alpha is +5.6%/yr, with a 95% interval 34 points wide. Four years of 60-session
holds are about 17 independent bets, too few to tell a 5%/yr alpha from zero.

## Standing lessons

- **Never read a Sharpe off a realised-only equity curve.** Here it inflated the
  Sharpe 2.1× on identical trades. `eval/benchmark.assert_marked` now refuses
  such a series.
- **A filter on the future does not only drop rows; it re-ranks the survivors.**
  The forward filter ran before the within-day median and the composite
  z-ranks.
- **A split flag is not a split.** Check it against the vendor's corporate-action
  records before treating it as one (`scripts/fetch_corporate_actions.py`).
