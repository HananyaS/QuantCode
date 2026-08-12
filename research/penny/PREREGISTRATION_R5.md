# Round 5 Pre-Registration — The Cascade

**Status:** committed before any round-5 walk-forward result. The commit hash is
the timestamp. Constants below may not be edited in place; amend by appending a
dated section.

**Honest disclosure, stated first.** The pool definition in §2 was chosen *after*
looking at an in-sample table of intraday return by opening gap. It is a
hypothesis generated from data, not an independent prior. Everything that
follows is designed on the assumption that it might be a fluke, and §7 says what
would demonstrate that.

---

## 1. What rounds 1–4 got wrong about the pool

Every round used a hand-fixed screen — `gap ≥ +10%` in rounds 1–3, `gap ≥ +3%`
in round 4 — justified in round 1 as selecting "in play" names. **The screen was
never once tested against its own alternative.**

Measured across 1,455,390 liquid low-priced symbol-days (ADV ≥ $1M, prior
unadjusted close $0.50–$10, 2016–2026), intraday open→close return is
**monotonically decreasing in the opening gap**:

| gap bucket | n | open→close | hit rate |
|---|---|---|---|
| < −10% | 9,292 | −0.00% | 48.5% |
| **−10% to −3%** | **68,737** | **+0.46%** | **50.9%** |
| −3% to 0% | 702,143 | −0.05% | 46.2% |
| 0% to +3% | 586,182 | −0.14% | 44.2% |
| +3% to +10% | 77,386 | −0.43% | 41.0% |
| +10% to +25% | 8,831 | −1.07% | 37.7% |
| > +25% | 2,819 | **−3.68%** | 31.6% |

A one-sided `gap ≥ X` screen therefore selects the **losing tail**. Rounds 1–3
bought the −1.07% bucket; round 4 bought the −0.43% and −1.07% buckets. Four
rounds of negative gross are substantially explained by the screen itself.

This is short-term reversal (Lehmann 1990; Lo & MacKinlay 1990; Nagel 2012 on
its liquidity-provision interpretation), measured here on the exact population
and horizon this study cares about. Buying oversold names is **long-only** and
so respects the standing constraint.

---

## 2. The pool — fixed now

```
POOL:  gap in [-0.10, -0.03]
       prior unadjusted close in [$0.50, $10.00]
       adv_20 >= $1,000,000
       >= 30 prior bars
```

Realised: **68,030 symbol-days, 2,605 sessions, 3,050 symbols** (`candidates_rev.parquet`).

Chosen over alternatives because the effect is robust to the cut, not because
it was the maximum. Seven windows tested in-sample, all with CIs excluding zero:

| window | n | open→close | CI |
|---|---|---|---|
| [−0.15, −0.02] | 137,829 | +0.356% | [+0.146%, +0.581%] |
| [−0.12, −0.03] | 71,393 | +0.458% | [+0.153%, +0.775%] |
| **[−0.10, −0.03]** | **68,737** | **+0.461%** | **[+0.164%, +0.763%]** |
| [−0.08, −0.03] | 63,808 | +0.452% | [+0.159%, +0.748%] |
| [−0.10, −0.02] | 132,806 | +0.353% | [+0.145%, +0.573%] |
| [−0.20, −0.05] | 30,658 | +0.522% | [+0.030%, +1.008%] |
| [−0.06, −0.02] | 118,133 | +0.331% | [+0.155%, +0.529%] |

Stability, in-sample: **10 of 11 years positive** (2020 the exception at
−0.50%); excluding 2020 gives **+0.638%, CI [+0.371%, +0.913%]**; day-clustered
over 2,616 days, p = 0.0025.

`min_adv_20` is **$1M**, not round 4's $5M, because the effect concentrates in
thinner names — $1–5M: +0.683%, $5–20M: +0.392%, >$20M: +0.087% (CI includes
zero). That trades cheap execution for a larger edge, and §4 tests whether the
trade is worth making.

---

## 3. Hypothesis

> **H1″.** Within the pool of §2, a cascade selection has walk-forward
> out-of-sample **net** return positive, with the day-clustered 95% CI excluding
> zero, at ≥300 OOS trades spanning ≥150 distinct days, against fully-charged
> measured cost (`cost_realized`, round-4 definition: 09:35 ask + 15:55 bid,
> depth charged on both legs).

---

## 4. Gate R — does the edge survive to 09:35?

**The one measurement that can kill this round outright, and it runs first.**

The table in §1 is `open → close`. This study enters at **09:35**, five minutes
after the open. If reversal is a liquidity-provision effect it may be largely
captured in the opening minutes, leaving nothing by 09:35.

Measured on a **year-stratified probe of 264 sessions / 6,881 candidate-days**
before any modelling:

- **Gate R passes** if mean `09:35 → 15:55` return over the probe is **≥ +0.20%**
  with a day-clustered CI excluding zero.
- **Gate R fails** → the pool is not tradable at this entry time. Do not model
  it. Report, and test alternative entries (§6) as the surviving question.

`open → close` is never used as a result. It selects where to spend fetches.

---

## 5. The cascade

Three stages, each with its own feature set and its own population. Nothing
downstream may use a feature the stage above it could not see.

**Stage 1 — pool.** §2, fixed. Not learned this round: with the screen only just
corrected, learning it as well would confound two changes, which is exactly the
mistake round 4 made by changing population and label together.

**Stage 2 — refine.** Daily, cross-sectional, regime, filing and flow features.
Output: a ranking over the pool.

**Stage 3 — exact.** Premarket sequence and 09:34 microstructure on stage-2
survivors only. Output: the final selection.

---

## 6. Constants — fixed now

| Constant | Value |
|---|---|
| `GAP_MIN` / `GAP_MAX` | **−0.10 / −0.03** |
| `MIN_ADV` | **$1,000,000** |
| `PRICE_MIN` / `PRICE_MAX` | $0.50 / $10.00 |
| `TOP_FRACTIONS` | 0.01, 0.02, 0.05, 0.10, 0.20 |
| `ORDER_SIZES` | $2,000, $5,000, $10,000 |
| `MIN_OOS_TRADES` / `MIN_OOS_DAYS` | 300 / 150 |
| `MIN_TRAIN_ROWS` | 2,000 |
| `LABEL_COST` | **0.0437, fixed** — never per-trade (round 4, §34) |
| `MAX_SPREAD_BPS` | 200 |
| `PRIOR_TRIALS` | **89** (72 through round 3, +17 eligible cells in round 4) |
| `FDR_Q` | 0.05 |
| entry variants (§4 fallback) | 09:35 open, 09:45, 10:30, opening-range breakout |
| confirmatory arms | `{gbdt_rank, lambdarank, quantile_q90, blend_transformer}` |

**`LABEL_COST` is fixed and this is not negotiable.** Round 4 labelled
`ret > cost_hat`, and the model learned to minimise the hurdle rather than
maximise the return: ρ(score, cost) = −0.48, and the precision ladder inverted
to −3.32% at the top 0.5%. A per-trade threshold the model can also predict is
not a target.

---

## 7. What would falsify this round

1. **Gate R fails** — the edge is gone by 09:35. Most likely single outcome.
2. Costs at ADV ≥ $1M exceed the gross edge. Round 4 measured 66.6 bps at
   ADV ≥ $5M; thinner names are dearer, and the pool's edge is 46 bps. **On the
   arithmetic available before measuring, this round starts behind.**
3. Walk-forward destroys it — the in-sample stability of §2 is over the whole
   panel, and 2020 alone contributed 10,674 of 68,737 rows.
4. The nulls fail on their own terms (shuffled label, date shift), now read with
   round 4's §35 caveat: a null "failing" by *improving* indicates negative
   skill, not leakage.

---

## 8. Validity requirements

Unchanged from round 4 §10, and non-negotiable:

1. Walk-forward only, purged and embargoed; the original holdout stays retired.
2. Every fold trains on ≥ `MIN_TRAIN_ROWS`.
3. Calibrators and conditional moments fitted inside the fold, on training rows.
4. No feature may use information at or after 09:35:00.
5. `cost_realized` stays in `_forbidden_columns`.
6. Nulls must destroy performance.
7. Delisted symbols retained; survivorship re-checked on the new pool.
8. Every rung reported, including the unflattering ones.
9. **The pool was data-derived. Round 5 is therefore a confirmatory test of a
   hypothesis generated in round 4's data, and no result from it may be
   described as an out-of-sample discovery.**

---

## Amendment 1 — 2026-08-12, after the Gate R probe, before any modelling

**Gate R FAILED as specified.** On a year-stratified probe of 264 of 2,605
sessions (6,727 matched candidate-days):

| entry | n | mean | CI | hit |
|---|---|---|---|---|
| 09:31 | 5,151 | +0.50% | [−0.34%, +1.27%] | 51.7% |
| **09:35** | **5,289** | **+0.37%** | **[−0.39%, +1.06%]** | 49.9% |
| 09:40 | 5,223 | +0.23% | | 48.6% |
| 09:45 | 5,159 | +0.17% | | 47.1% |
| 10:00 | 5,146 | +0.08% | | 46.4% |
| 10:30 | 4,972 | +0.05% | | 45.3% |
| 11:00 | 4,795 | +0.13% | | 47.8% |

§4 required **both** a point estimate ≥ +0.20% **and** a CI excluding zero. The
first is met (+0.37%); the second is not.

**The failure is attributable to power, not to the effect, and the arithmetic is
checkable rather than rhetorical:**

- the probe is 8.1% of available sessions;
- its CI half-width is ±0.72pp against ±0.31pp for the full-sample open→close
  measurement — the √n widening one would predict;
- the point estimate is what the full sample implies: decay leaves 74% of the
  09:31 edge at 09:35, and 0.74 × 0.50% = **0.37%**, against +0.46% open→close.

Reversal decays monotonically through the morning and is ~zero by 10:00, which
is the predicted signature of liquidity provision (Nagel 2012) and is evidence
the effect is real rather than an artefact.

**Single pre-specified escalation.** The probe existed to decide where to spend
fetches (§4), and it says: spend them. Minute bars are fetched for all 2,605
pool sessions and **Gate R is re-evaluated exactly once** on the full sample,
against the identical threshold.

Constraints on this escalation, fixed now:

1. **One escalation only.** If Gate R fails on the full sample, the round stops.
   There is no third sample to draw and none will be sought.
2. **The threshold does not move.** ≥ +0.20% with a CI excluding zero, at 09:35.
3. **The entry time does not move.** 09:35 is the decision instant for the gate.
   The alternative entries in §6 remain a *separate, exploratory* question and
   may not be substituted for a failed gate.
4. **This escalation is a trial.** `PRIOR_TRIALS` rises 89 → **90**.

Recorded because "we ran it again on more data" is only legitimate when the
decision to do so, and its limits, are written down before the result is seen.
