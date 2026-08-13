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
