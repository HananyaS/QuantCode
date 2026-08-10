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

*(Sections 9+ — dataset construction, results, and holdout — appended as the
run proceeds.)*
