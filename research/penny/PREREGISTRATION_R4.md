# Round 4 Pre-Registration — Penny-Stock Long Side

**Status:** committed before any round-4 result was computed. The commit hash of
this file is the timestamp. Nothing below may be changed after the first round-4
number is produced; changes must be made by appending a dated amendment section,
never by editing the constants in place.

Rounds 1–3 pre-registered their gates as literals inside the study script, so
"pre-registered" rested on reading git history of a file that also changed for
unrelated reasons. This file exists to make the claim checkable.

---

## 1. What round 4 tests, and why it is not a re-run

Round 3 concluded that the binding constraint is microstructure. Two structural
defects made that conclusion look more final than the evidence supports.

**Defect 1 — the cost table was mis-specified.** It sized every order at *1% of
ADV* (`etl/quotes.py:196`), so more liquid names mechanically received larger
orders and appeared more expensive. Re-computed at a **fixed** order size, which
is how the strategy would actually trade, the ranking inverts:

| ADV bucket | n | RT @ $2k | RT @ $10k |
|---|---|---|---|
| ≤ $300k | 1,501 | 395.1 bps | 1034.6 |
| $300k–1M | 2,085 | 259.4 | 683.8 |
| $1–5M | 2,731 | 149.4 | 371.6 |
| $5–20M | 1,956 | 98.6 | 215.5 |
| > $20M | 1,521 | **63.1** | **124.0** |

The reported "irreducible 77 bps floor" is a whole-panel median, not a floor.
Cost is a **filter**, and `adv_20` is ex-ante and already in the panel.

**Defect 2 — narrow funnel, shallow cut.** The screen triggers on `gap ≥ +10%`
with `adv_20 ≥ $100k` (`etl/candidates.py:59-65`), giving 20,558 candidates over
a decade, and selection took the top 10%. That is a recall posture applied to a
precision problem.

Round 4 therefore **widens the funnel and cuts far deeper**, inside the liquidity
band where execution is cheap.

---

## 2. Hypothesis

> **H1′.** Within the liquidity-screened funnel defined in §3, a cost-aware
> selection has walk-forward out-of-sample **net** return positive, with the
> day-clustered 95% confidence interval excluding zero, at **≥300 out-of-sample
> trades spanning ≥150 distinct days**, against **fully-charged measured cost**
> as defined in §4.

One-sided; the alternative is `E[net] > 0`.

---

## 3. Constants — fixed now

| Constant | Value |
|---|---|
| `LIQUID_ADV_MIN` | **$5,000,000** trailing 20-day dollar volume |
| `GAP_MIN` | **+0.03** |
| `PRICE_MIN` / `PRICE_MAX` | **$0.50 / $10.00** (unadjusted prior close, unchanged) |
| `TOP_FRACTIONS` | **0.005, 0.01, 0.02, 0.05, 0.10** |
| `ORDER_SIZES` | **$2,000, $5,000, $10,000** (unchanged from round 3) |
| `MIN_OOS_TRADES` | **300** |
| `MIN_OOS_DAYS` | **150** |
| `MIN_RETAINED` | **removed** — see §6 |
| `MIN_TRAIN_ROWS` | **2,000** |
| `PRIOR_TRIALS` | **72** |
| `MAX_SPREAD_BPS` | **200** (tradeability ceiling, §5) |
| `FDR_Q` | **0.05** |

Rationale for the two that carry the round:

- **`LIQUID_ADV_MIN = $5M`** — the cost filter. Measured round-trip at a $2k
  order is 63–99 bps in this band versus 395 bps at the thin end. Note this is
  still *below* the `$1M` floor round 1's own `_veto_thin_adv` used
  (`features/veto.py:84`), which the $100k screen contradicted.
- **`GAP_MIN = +0.03`** — measured on cached 2024 daily bars, `ADV≥$5M, gap≥+3%`
  yields 3,542 symbol-days/yr (≈35k over the decade) versus 1,853/yr for the
  current screen. This is what makes a 0.5% cut measurable.

---

## 4. Cost definition — fully charged

Two distinct quantities, and conflating them is the failure this section exists
to prevent.

- **`cost_hat`** — the **decision input**. Computed from NBBO in the strictly
  pre-entry window **`[09:34:00, 09:35:00)` ET**. May inform features, labels and
  the decision rule.
- **`cost_realized`** — what is **charged** in evaluation. Measured entry book
  `[09:35:00, 09:36:00)` plus measured exit book `[15:55:00, 15:56:00)`.
  Never an input to anything.

Decide with what was knowable; get charged what happened.

Round 3 charged `depth_shortfall` **once** (`etl/quotes.py:204`), i.e. assumed
the 15:55 exit walks no book, while depth was 398 of the 728 bps. Round 4 charges
each leg from its own measured book.

Round 3's cost table covered **56.3%** of trades and dropped the remainder from
the bootstrap rather than defaulting them (`run_round3_study.py:167`). Round 4
re-measures all sessions. Any trade still lacking a firm quote is **filtered as
untradeable** (§5), not silently dropped.

---

## 5. Tradeability filter

Applied before modelling. A candidate is untradeable, and therefore outside the
strategy by definition, if any of:

- no firm inside quote in the decision window;
- `ask_notional` < the order size under test;
- `half_spread_bps` > `MAX_SPREAD_BPS`.

The count removed and its unconditional return are reported, so the filter's
effect on the population is visible rather than hidden.

---

## 6. Precision posture

Recall is expected to be low and is not a criterion. `MIN_RETAINED = 0.05`
(`run_round3_study.py:63`) is a recall floor and is **removed**. It is replaced by
`MIN_OOS_TRADES` / `MIN_OOS_DAYS`, which exist solely so a day-clustered interval
is meaningful — a power floor, not a preference about how often to trade.

Headline KPIs, reported at **every** rung of the ladder, not only the winner:
net expectancy per trade, hit rate, profit factor, trades/month, worst day,
worst month, max consecutive losers, and the trade and distinct-day counts.

---

## 7. Arms

Confirmatory family, declared here as an explicit registry rather than inferred
from a label prefix:

```
CONFIRMATORY_ARMS = {"gbdt_liquid", "ev_rule", "blend_transformer"}
```

`blend_transformer` is promoted from round 3's exploratory arm (+1.22% gross,
CI [+0.14%, …] — the only interval in the project excluding zero). Anything not
in this set is exploratory, is labelled as such in the output, and is excluded
from the H1′ family before the cell family is assembled.

Round 3 excluded exploratory rows by a string prefix on a label, and only when
Gate B had failed (`run_round3_study.py:384-388`) — so a passing gate would have
silently admitted them to the confirmatory family. The registry closes that.

---

## 8. Multiple testing

- One cell per (grid row × order size), one-sided p from `p_value_greater`.
- Benjamini–Hochberg at `FDR_Q = 0.05` over eligible confirmatory cells **plus
  `PRIOR_TRIALS = 72`**. Round 3 reported `n_trials = len(cells) + 27`
  (`run_round3_study.py:391`) but ran BH on the cells array alone, so the prior
  trials never harshened the correction. They do now.
- Deflated Sharpe against the true cumulative trial count (72 + round-4 cells).
- Every step of the adaptive ladder (§9) is a trial and is counted.

---

## 9. Adaptive ladder

Round 4 runs to completion rather than halting at a gate. When a rung fails the
filter is tightened and the round continues. **Each move is a trial**, is
recorded in `LOG.md` with the number that triggered it, and enters the count in
§8.

| If | Then |
|---|---|
| liquid-tail cost still exceeds achievable gross | raise `LIQUID_ADV_MIN` to $20,000,000 and re-cut |
| fewer than `MIN_OOS_TRADES` at the 0.5% rung | lower `GAP_MIN` toward 0.0 before loosening the cut |
| a slice has no firm quotes | filter it as untradeable; report its size and unconditional return |
| the EV rule underperforms the rank cut | report both; the rank cut remains the pre-registered comparator |
| every CI includes zero | report the negative with the full precision ladder as the deliverable |

---

## 10. Validity requirements

Non-negotiable; a violation invalidates the round regardless of the result.

1. Walk-forward only, purged and embargoed. The original sealed holdout was
   examined twice in rounds 1–2 and stays retired.
2. Every fold trains on ≥ `MIN_TRAIN_ROWS`. Round 3's fold 0 trained on 610 rows
   and fabricated the round-2 headline.
3. Isotonic calibration and all conditional moments are fitted **inside the fold,
   on training rows only**.
4. No `cost_hat` input may use a quote at or after 09:35:00.
5. `cost_realized` columns remain in `_forbidden_columns` and may never reach a
   feature list.
6. Shuffled-label and date-shift nulls must destroy performance — for the EV rule
   and calibrator, not only the GBDT.
7. The universe keeps delisted symbols; the survivorship check is re-run on the
   rebuilt candidate set.
8. Results are reported at every rung, including the unflattering ones.

---

## 11. What would falsify the round-4 thesis

The thesis is that cost is a filter rather than a wall. It is falsified if,
inside the liquid band with fully-charged cost, no rung of the precision ladder
produces a net CI excluding zero. In that case the three-round negative stands,
cost-aware selection is eliminated with it, and the remaining untested levers are
the short side (gated on borrow data) and a different candidate population
entirely.
