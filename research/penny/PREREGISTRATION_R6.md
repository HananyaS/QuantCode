# Round 6 Pre-Registration — the cascade, on a positive-gross pool

**Status:** committed before any round-6 result. The commit hash is the
timestamp. Constants may not be edited in place; amend by appending a dated
section.

---

## 1. Why this is not a re-run of round 5

Round 5 stopped at Gate R, which asked: *is the reversal pool tradable
unconditionally?* Answer: **+0.31% gross at 09:35** (n=51,361, 2,575 days,
p=0.0295, CI [−0.03%, +0.63%]) against an extrapolated ~67 bps cost. Not
tradable as-is, and the round stopped as its amendment required.

Round 6 asks a **different question**: *can a cascade concentrate that pool
above its cost bar?*

The distinction matters, and it is the one thing that makes this legitimate
rather than a fifth bite at the same apple:

- **Rounds 1–4 hunted subsets of populations whose gross was negative.** That is
  searching for a pocket where the sign flips, and it is exactly what a
  multiple-testing framework exists to discipline. It failed four times.
- **This pool's gross is positive.** Concentrating a positive-expectancy
  population is the ordinary use of a model, not a search for an anomaly. A
  model that lifts +31 bps to +100 bps is doing something unremarkable; a model
  that lifts −19 bps to +100 bps was never plausible.

**What has changed since round 5's stop, and it is not a reinterpretation.**
Round 5 *extrapolated* the reversal pool's cost from round 4's ADV ≥ $5M
measurement. That was a real gap: the reversal pool floors at **ADV ≥ $1M**, and
a three-session probe showed a decision half-spread of **63.3 bps against 22.4**.
Round 6 measures the pool's own cost bar for the first time (§3). If the pool
costs 90+ bps round trip, no cascade result can be tradable and this round ends
on arithmetic rather than on a model.

---

## 2. Hypothesis

> **H1‴.** Within the round-5 reversal pool, a three-stage cascade selection has
> walk-forward out-of-sample **net** return positive, with the day-clustered 95%
> CI excluding zero, at ≥300 OOS trades spanning ≥150 distinct days, against the
> pool's **own measured** `cost_realized`.

---

## 3. Gate C — the pool's own cost bar. Runs first.

Measured on all 2,605 sessions, three windows (09:34 decision, 09:35 entry,
15:55 exit), both books, depth charged on both legs — the round-4 definition,
now applied to this pool.

- **Gate C passes** if the median round trip at a $2,000 order is **< 150 bps**.
  That is roughly 5× the unconditional edge, and a cascade producing 5× lift is
  a strong but not absurd ask.
- **Gate C fails** → report the measured bar and stop. No cascade is fitted.

150 bps is set *before* the measurement and is deliberately generous: the
purpose is to rule out the hopeless case, not to pre-judge the plausible one.

---

## 4. The cascade — three stages, fixed now

Each stage fits on its predecessor's **training** survivors and scores its
**test** survivors, independently. `model/cascade.py`, tested in
`tests/test_penny_cascade.py`.

| stage | features | keep | objective |
|---|---|---|---|
| **1 · context** | daily (`BASE_FEATURES`), regime, cross-sectional ranks | 0.50 | `rank` |
| **2 · fundamentals** | + EDGAR recency, insider/institutional flow, float | 0.40 | `binary` |
| **3 · execution** | + premarket aggregates, `cost_hat` microstructure | 0.25 | `quantile` α=0.9 |

Compound retention 5%. Stage 3 uses the quantile objective because this
population's payoff lives in the right tail, and stage 1 uses ranking because
"which of today's names" is the question a daily strategy actually asks.

**Ablation, required.** Each stage is also run alone, and the cascade is
compared against a single-stage GBDT over the union of all features. If the
cascade does not beat the flat model, that is the result and it is reported.

---

## 5. Constants — fixed now

| Constant | Value |
|---|---|
| pool | round-5 `candidates_rev.parquet`, unchanged |
| `TOP_FRACTIONS` (flat comparator) | 0.01, 0.02, 0.05, 0.10, 0.20 |
| `ORDER_SIZES` | $2,000, $5,000, $10,000 |
| `MIN_OOS_TRADES` / `MIN_OOS_DAYS` | 300 / 150 |
| `MIN_TRAIN_ROWS` | 2,000 |
| `LABEL_COST` | **0.0437, fixed** — never per-trade |
| `MAX_SPREAD_BPS` | 200 |
| `GATE_C_MAX_BPS` | **150** |
| `PRIOR_TRIALS` | **91** (90 through round 5, +1 for the horizon study) |
| `FDR_Q` | 0.05 |
| confirmatory arms | `{cascade, flat_gbdt}` |

`LABEL_COST` stays constant. Round 4 labelled `ret > cost_hat`, the model
learned to minimise the hurdle, ρ(score, cost) = −0.48, and the precision ladder
inverted to −3.32% at the top 0.5%.

---

## 6. What would falsify this round

1. **Gate C fails** — the pool's own cost exceeds 150 bps. On the three-session
   probe the entry half-spread alone was 66.7 bps, so this is a live risk.
2. The cascade does not beat the flat GBDT — staging adds nothing.
3. Walk-forward destroys the lift; the +0.31% is in-sample-ish over a pool
   chosen from an in-sample table.
4. Nulls fail on their own terms, read with round 4 §35's caveat.
5. Retention collapses below `MIN_OOS_TRADES` before the lift materialises.

**Prior expectation, recorded so it cannot be revised afterwards:** the pool
needs roughly **3–5× concentration** to clear its likely cost. Rounds 2–3
observed model lift of ~+1.8pp on selected populations, so this is not absurd —
but no round of this project has yet produced a *confirmed* lift that survived
correction, and the base rate for that is zero for five rounds.

---

## 7. Validity requirements

Unchanged and non-negotiable: walk-forward only; every fold ≥ `MIN_TRAIN_ROWS`;
all fitting inside the fold; no feature at or after 09:35:00; `cost_realized`
forbidden as a feature; nulls must destroy performance; delisted symbols
retained; every rung reported including the unflattering ones.

**And specific to a cascade:** no stage may be fitted on any population derived
from its own test fold. This is the failure mode a cascade uniquely permits, it
compounds silently across stages, and it is pinned by
`tests/test_penny_cascade.py::test_no_stage_is_ever_fitted_on_its_own_test_fold`.

**Standing:** the pool was derived from data in round 5. Round 6 is a
confirmatory test of that hypothesis and no result may be described as an
out-of-sample discovery.

---

## Amendment 1 — 2026-08-13, before any round-6 model result

**Adding order sizes $250, $500 and $1,000.** Recorded as an amendment rather
than slipped into §5, because it is a **new axis chosen after seeing data** — the
partial cost measurement, not a model result, but data nonetheless.

### What prompted it

On 620 of 2,605 quote sessions (3,934 joined trades), two facts:

**Cost-aware selection buys nothing.** Gross is flat across cost quintiles while
cost rises 35×:

| quintile | n | mean cost | gross | net |
|---|---|---|---|---|
| Q1 | 787 | 15.7 bps | +0.155% | **−0.00%** |
| Q2 | 787 | 30.3 | +0.171% | −0.13% |
| Q3 | 786 | 58.1 | +0.276% | −0.31% |
| Q4 | 787 | 117.1 | +0.216% | −0.95% |
| Q5 | 787 | 556.8 | +0.401% | −5.17% |

Every gross CI includes zero. The best cell is breakeven before any model.

**The edge sits in names too thin to buy at $2,000.** Requiring the entry book
to absorb the order (58% of rows) cuts gross from **+0.244% to +0.034%**.

### Why smaller sizes, and why this is not fishing

Depth shortfall is `(order/inside − 1)⁺ × half_spread`. It scales with order
size and **no round has ever tested below $2,000**. The median entry book in
this pool is $2,719, so at $500 most of the pool fits inside the quote and the
depth term goes to zero by construction — this is arithmetic, not a hoped-for
empirical regularity.

The distinction from fishing: the *mechanism* is known in advance and the
prediction is directional and falsifiable — smaller orders must reduce depth
cost, and the open question is only whether **gross survives the tradeability
filter** at $500 the way it fails at $2,000. If gross at $500 is also ~+0.03%,
the effect is untradable at any size and the round ends.

### Fixed now

| Constant | Value |
|---|---|
| `ORDER_SIZES` | **250, 500, 1,000**, 2,000, 5,000, 10,000 |
| `PRIOR_TRIALS` | 91 → **94** (one per added size) |

Everything else in §3–§7 is unchanged. Gate C's 150 bps threshold still refers
to the **$2,000** round trip, so the gate cannot be passed by shrinking the
order until it clears its own bar.

### Predicted outcome, recorded before the run

At $500 the depth term largely vanishes and cost should approach twice the
half-spread, roughly **45 bps** (30.6 entry + 14.4 exit at the median). Against
an unconditional +24 to +31 bps gross that is **still negative**, so the round
needs the cascade to roughly double gross on a population where every quintile
CI currently includes zero. **The honest prior is that this fails.**
