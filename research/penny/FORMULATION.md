# Round 11 — Formulation

Everything in this round is defined here first and implemented to match. Where
code and this document disagree, the code is wrong.

---

## 1. Why the previous rounds could not have succeeded

Rounds 1–10 all estimated the same object: the unconditional mean forward return
of a screened set,

$$\hat\mu \;=\; \frac{1}{|S|}\sum_{(i,t)\in S} R_{i,t},\qquad
S=\{(i,t): \text{screen holds}\}$$

§76 showed why that estimator cannot answer the question. Decompose the forward
return of stock $i$ entered on day $t$ into a **day** effect, a **sector-within-day**
effect, and an **idiosyncratic residual**:

$$\boxed{\;R_{i,t} \;=\; \mu_t \;+\; \lambda_{t,g(i)} \;+\; \eta_{i,t}\;}\tag{1}$$

with $g(i)$ the sector of $i$, and $\mathbb{E}[\eta \mid t,g]=0$ by construction.

A screen that fires on few days puts almost all of its variance into $\mu_t$.
Measured directly: the sector arm produced 286 trades on **48 distinct days**, and
the paired control gave

$$\hat{\mathbb{E}}[\eta] = -1.06\%,\qquad \text{CI } [-12.09\%,\, +9.77\%].$$

The interval is 22 points wide on 48 clusters. **The design had almost no power to
detect selection**, so "no effect" and "an effect we cannot see" were
indistinguishable. Every earlier round shares this defect.

---

## 2. What this round estimates instead

Only $\eta$. Define the **within-cell demeaned return** over the cell
$c=(t,g)$:

$$\tilde R_{i,t} \;=\; R_{i,t} \;-\; \frac{1}{|c|}\sum_{j\in c} R_{j,t}
\;=\; \eta_{i,t} - \bar\eta_c \tag{2}$$

$\mu_t$ and $\lambda_{t,g}$ cancel **algebraically**, not statistically. No control
group, no matching, no assumption about the market.

### 2.1 Signal quality — rank information coefficient

For a candidate feature $f$ (computed only from information available at $t$),

$$\mathrm{IC}_c \;=\; \rho_{\text{Spearman}}\!\left(f_{i}, \tilde R_{i}\right)_{i\in c},
\qquad |c| \ge n_{\min} \tag{3}$$

$$\overline{\mathrm{IC}} \;=\; \frac{1}{|C|}\sum_{c\in C}\mathrm{IC}_c,
\qquad
t_{\mathrm{IC}} \;=\; \frac{\overline{\mathrm{IC}}}{\hat\sigma(\mathrm{IC})/\sqrt{|C|}} \tag{4}$$

with $\hat\sigma$ from a **cell-clustered bootstrap** over days, since cells on the
same day are not independent.

### 2.2 Tradable form — a cell-neutral portfolio

Inside each cell rank the feature and form weights

$$w_{i} \;=\; \frac{\mathrm{rank}(f_i) - \overline{\mathrm{rank}}_c}
{\sum_{j\in c}\left|\mathrm{rank}(f_j) - \overline{\mathrm{rank}}_c\right|},
\qquad \sum_{i\in c} w_i = 0,\quad \sum_{i\in c}|w_i| = 1 \tag{5}$$

$$R^{\text{LS}}_c \;=\; \sum_{i\in c} w_i R_{i,t}
\;=\; \sum_{i\in c} w_i \eta_{i,t} \tag{6}$$

The second equality holds because $\sum_i w_i = 0$ annihilates $\mu_t+\lambda_{t,g}$.
**This is the estimand.** It is measured as a research instrument first; the borrow
constraint (§67: only 16.1% of this universe is shortable) is a separate question
applied afterwards, in §5.

### 2.3 Costs

Charged on both legs at the measured schedule by liquidity band (round 4):

$$R^{\text{LS,net}}_c \;=\; \sum_i w_i R_{i,t} \;-\; \sum_i |w_i|\,c_i \tag{7}$$

---

## 3. Improvement blocks

Each block is a set of features. Blocks are tested **one at a time** against the
same population and the same estimator, then in combination. A block is retained
only if it clears §4.

| # | block | features | prior / rationale |
|---|---|---|---|
| **B0** | control | random feature, and $-r^{(5)}$ itself | must score ≈ 0; calibrates the estimator |
| **B1** | crash shape | share of the fall in the single worst session; open-to-close vs overnight split; range expansion | a gap-down (news) and a slide (distribution) are different mechanisms |
| **B2** | idiosyncratic depth | $\varepsilon_{i,t} = r^{(5)}_{i,t} - m_{g,t}$ and its rank within cell | §76's split, now as a continuous within-cell feature |
| **B3** | liquidity & attention | $\log$ ADV, dollar-volume shock, volume/ADV on the crash day | Barber–Odean attention; also a capacity proxy |
| **B4** | balance sheet | cash / market cap, net cash, shares-outstanding growth (dilution) | a crash to below net cash is not the same event as a crash on operations |
| **B5** | filings | days since 8-K item 3.01 (deficiency), shelf S-3 count, 424B takedowns, Form 4 net insider buying | round 2 measured the largest single lift of the project here |
| **B6** | short interest | FINRA short volume ratio, trend | squeeze fuel |
| **B7** | prior behaviour | count of prior ≥50% single-day moves; realised vol; time since listing | "repeat offenders" |
| **B8** | reversal state | distance from 20-day low, consecutive down days, RSI-like | classic short-horizon reversal |

Literature anchors, to be checked rather than assumed:
Jegadeesh (1990) and Lehmann (1990) on short-horizon reversal;
De Bondt–Thaler (1985) long-horizon;
Bali–Cakici–Whitelaw (2011) MAX / lottery — predicts these names **underperform**;
Barber–Odean (2008) attention; Avramov–Chordia–Goyal (2006) on liquidity
conditioning reversal. If B8 works and B3 conditions it, that matches
Avramov et al.; if B4/B5 dominate, it is closer to distress mispricing.

---

## 4. Acceptance rules, fixed now

A block is **retained** only if all of the following hold on the development
period (signals before 2024-01-01):

1. $\overline{\mathrm{IC}} \ne 0$ with a cell-clustered bootstrap CI excluding zero;
2. the sign is **stable** across the two halves of development, and across the
   four largest sectors;
3. the cell-neutral net return (7) is positive with CI excluding zero;
4. it is not carried by one period: leave-one-year-out leaves every CI excluding
   zero;
5. the effect survives dropping the largest 10% of cells by $|c|$.

Multiple testing: every feature evaluated counts. Benjamini–Hochberg at
$q=0.05$ over the **cumulative** trial count, which stands at 195 from round 9
plus everything added here.

**Kill rule.** If B0 (the control) scores non-zero, the estimator is broken and
the round stops until it is fixed. If no block clears, the family is closed.

**The sealed holdout (2024-01-01 →) is not read in this round at all.**

---

## 5. If, and only if, a block survives

Then and only then:

* re-express as long-only, since §67 established 84% of this universe cannot be
  borrowed — the long leg alone must carry it;
* re-introduce $\mu_t$ deliberately (the day effect is real and positive; it is
  simply not evidence of skill) and check the combination compounds;
* walk-forward the parameters, never fitting on the traded year;
* then one holdout read.

---

## 6. Standing invariants inherited from earlier rounds

1. Filters touch **unadjusted** prices only (§72).
2. Panels are **continuous**; eligibility is a column, never a row filter (§74).
3. Corporate actions excluded across lookback **and** holding window (§57).
4. A truncated forward window is a **delisting**, exited at the last print (§65).
5. Ties in barrier resolution go to the **stop**; gaps through it fill at the open.
6. Read the largest individual trades before believing any aggregate — this has
   overturned four conclusions that passed every aggregate test (§57, §64, §72, §74).

---

## Amendment 1 — 2026-08-18, before any block was scored

**Cells are days, not day×sector.** §2 defined $c=(t,g)$. Measured on the
development period, that cell is unusable:

| crash threshold | candidates | $(t,g)$ cells $\ge 5$ | median $|c|$ | coverage |
|---|---|---|---|---|
| ≥15% | 58,943 | 3,262 | **2** | 62.1% |
| ≥20% | 29,054 | 1,277 | **1** | 46.9% |

A cell of one or two names cannot be ranked, and dropping the rest discards half
the population. Day cells are ample:

| crash threshold | $(t)$ cells $\ge 5$ | median $|c|$ | coverage |
|---|---|---|---|
| **≥15%** | **1,837** | **16** | **99.3%** |
| ≥20% | 1,377 | 8 | 94.8% |

So the estimator demeans within $c=(t)$, which removes $\mu_t$ — the term §76
showed dominates the variance — and the sector term $\lambda_{t,g}$ enters as
**feature block B2** instead, as $\varepsilon_{i,t}=r^{(5)}_{i,t}-m_{g,t}$.

This is a power decision, not a convenience: it takes the design from **48
clusters to 1,837**.

**Population fixed now:** eligible symbol-days with $r^{(5)}\le-15\%$, price
\$0.50–\$10, ADV ≥\$250k, no corporate action, lookback span ≤18 calendar days,
in a day-cell of at least 5 names. 58,943 candidates on the development period.

**Outcome for block scoring:** the plain $h$-session forward return from the next
open, net of the measured cost schedule — no barriers. Exit engineering is a
separate question and would confound signal quality with it. Barriers return in
§5 only if a block survives.
