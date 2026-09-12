# Round 15 Pre-Registration — cutting the position count, and the instruments to judge it by

> The consolidated strategy design — every filter, the entry rule, the
> weighting, the exits, the execution model, the risk controls and the
> catalogue of what is tested versus open — is **`STRATEGY_SPEC.md`**.
> This document is the round's hypotheses and the battery's mathematics.

**Status:** Part I (the battery and the corrected book) is BUILT and RUN.
**Part II has now been executed**; the hypotheses below are as they were fixed
before the runs, and the outcomes are recorded in the table immediately after —
the body text is left unedited so the predictions can be read against what
happened.

## Outcomes, recorded after the fact

| | outcome | where |
|---|---|---|
| **H1** variance model | **FALSIFIED as predicted.** The vol cap leaves `sd(mu_t)` at 24.3% against 24.6% uncapped — it is purely an idiosyncratic filter. The §82/§84 gap is instead explained by `N` setting the *bet rate* (`lambda = N/H`) as well as the breadth. Tail indices 3.7–11.3, all above 2, so the variance arithmetic is valid. | §90 |
| **H2** `sigma_common` | **FIRST CLAUSE FAILS, SECOND PASSES, PARTLY REPLICATES.** Sharpe at fixed N falls. At matched book size the sector cap wins on development at 15 names (+0.44) and 30 (+0.20); on the holdout the 15-name gain **vanishes** (+0.03) while 30 (+0.32) and 45 (+0.4) hold. The drawdown improvement replicates everywhere and is large. | §91, §92, §94 |
| **H3** intraday selector | **FAILS.** 3 of 14 features clear BH against 287 prior trials and the control is calibrated, but their long-short net is negative. The two that looked tradable as fill rules die on the holdout: `M_intraday_vol` inverts (dev +0.53/+0.65/+0.55, holdout −0.93/−0.89/−1.46) and `M_shares_per_trade` is a look-ahead (its top decile contains 0.0% future reverse-splitters against 7.4%). Both withdrawn. | §95, §96, §97 |
| **H4** weighting | **NULL, and a defect found.** The inverse-vol arm was a no-op in every prior round (`iv/iv.mean()` over a one-name pool). Fixed; the tilt then moves Sharpe by −0.11 / +0.10 / −0.00 / −0.01 across N. Equal weight stands. | §93 |
| **H5** entry timing | **SMALL POSITIVE.** Entering at 10:00 rather than the open lifts IR per position 0.303 → 0.319. Development only, and it is free. | §95 |

**Holdout reads consumed by this round:** Part I's re-cost (5th), H2's matched
book (6th), H3's selector (7th). Recorded rather than discovered later.

---

## 0. Why this round exists

The project has one validated effect: the **low-volatility effect inside crashed
microcaps** — Bali, Cakici & Whitelaw's (2011) MAX finding in the corner of the
market where it should be strongest. §78 validated it, §80 showed the
seven-feature composite was a wrapper around that one feature, and §84 landed on
a configuration requiring **60 concurrent positions held 60 sessions**.

Sixty line items is the objection. This round asks whether it can be fewer, and
insists on being able to tell the difference before it tries.

---

# PART I — the evaluation battery (built, and already used)

## 1.1 The problem it solves

Fourteen rounds compared configurations on a single Sharpe ratio: no interval, no
deflation for the size of the search behind it, and no test of the variance
assumption the breadth argument rests on. Three separate defects follow.

**Defect 1 — the Sharpe was read as precise.** §82 measured a seed-to-seed
standard deviation of **0.75** at N=30, larger than most differences the log
draws conclusions from.

**Defect 2 — the breadth argument assumes finite variance.** §80 finding 4 is
`SE = sigma/sqrt(N)`. For a tail index `alpha < 2` the sample mean converges at
`N^(1-1/alpha)`, not `N^(-1/2)`, and the required breadth is a different number.

**Defect 3 — nominal trade counts overstate the evidence.** At a 60-session hold
with an entry a day, labels overlap almost completely.

## 1.2 The metrics, with their mathematics

Implemented in `research/penny/eval/battery.py`, tested in
`tests/test_penny_battery.py` — each against a case whose answer is known in
closed form.

### Tier 1 — the Sharpe ratio, honestly

**Non-normal standard error** (Mertens 2002; Lo 2002):

$$\widehat{\mathrm{SE}}(\widehat{SR}) = \sqrt{\frac{1}{T}\left(1 + \frac{\widehat{SR}^2}{2} - \gamma_3\widehat{SR} + \frac{\gamma_4-3}{4}\widehat{SR}^2\right)}$$

with $\gamma_3$ skewness and $\gamma_4$ non-excess kurtosis. Under normality it
collapses to $\sqrt{(1+SR^2/2)/T}$.

**Autocorrelation adjustment** (Lo 2002, eq. 12) — overlapping 60-session holds
induce exactly the serial dependence that makes $\sqrt{q}$ scaling optimistic:

$$SR(q) = \frac{SR \cdot q}{\sqrt{q + 2\sum_{k=1}^{q-1}(q-k)\rho_k}}$$

**Probabilistic Sharpe Ratio** (Bailey & López de Prado 2012):

$$\widehat{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR}-SR^*)\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^2}}\right)$$

**Deflated Sharpe Ratio** (Bailey & López de Prado 2014) — PSR against the
Sharpe a search of size $N$ produces from pure noise:

$$SR^*_0 = \sqrt{\operatorname{Var}(SR_n)}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac1N\right) + \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right]$$

$\gamma$ is Euler–Mascheroni. `stats.deflated_sharpe_ratio` has existed since
round 4 and had **never been called with a real trial count**.

**Minimum Track Record Length** (Bailey & López de Prado 2012):

$$\mathrm{MinTRL} = 1 + \left[1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^2\right]\left(\frac{\Phi^{-1}(\alpha)}{\widehat{SR}-SR^*}\right)^2$$

### Tier 2 — the shape of the loss

Sortino & Price (1994); **Omega** (Keating & Shadwick 2002),
$\Omega(\theta)=\mathbb{E}[(r-\theta)^+]/\mathbb{E}[(\theta-r)^+]$; Calmar (Young
1991); **Ulcer index** and Martin ratio (Martin & McCann 1989); CVaR/expected
shortfall (Rockafellar & Uryasev 2000); Rachev ratio (Biglova, Ortobelli, Rachev
& Stoyanov 2004).

**Expected maximum drawdown** (Magdon-Ismail & Atiya 2004) — reported *beside*
the realised drawdown. Their closed form is a two-branch series that is easy to
mis-implement, so it is simulated; the one exactly known case anchors the test:

$$\mathbb{E}[\mathrm{MDD}] = \sqrt{\pi/2}\;\sigma\sqrt{T} \quad (\mu = 0)$$

A realised drawdown far worse than its own expectation falsifies the return model
the Sharpe assumes — something no amount of Sharpe precision would reveal.

**Hill (1975) tail index**, over the $k$ largest order statistics:

$$H_k = \frac{1}{k}\sum_{i=1}^{k}\ln\frac{X_{(i)}}{X_{(k+1)}},\qquad \alpha = 1/H_k$$

*Implementation note.* A profit target puts a **point mass at the barrier**: with
a +100% target a large share of trades return exactly +1.00 less cost. Hill on
that sample sees a wall of ties, drives $H_k \to 0$ and reports $\alpha \to
\infty$ — which reads as "no tail" when what happened is "the tail was truncated
by construction". The right tail is therefore estimated only on gains below the
barrier, with the truncated share reported alongside.

### Tier 3 — breadth, which is the actual question

$$IR = IC\sqrt{BR} \qquad\text{(Grinold 1989)}$$
$$IR = TC \cdot IC\sqrt{BR},\quad TC = \operatorname{corr}(w^{\text{ideal}}, w^{\text{actual}}) \qquad\text{(Clarke, de Silva \& Thorley 2002)}$$
$$BR_{\text{eff}} = \frac{N}{1+(N-1)\bar\rho} \qquad\text{(Buckle 2004)}$$
$$N_{\text{eff}} = 1/\textstyle\sum_i w_i^2 \qquad\text{(Herfindahl)}$$

**Average uniqueness** (López de Prado, *AFML*, ch. 4): with $c_t$ the number of
labels live at $t$ and $L_i$ the span of label $i$,

$$u_i = \frac{1}{|L_i|}\sum_{t\in L_i}\frac{1}{c_t},\qquad \text{effective sample} = \sum_i u_i$$

*Interpretation limit, stated so the number is not over-read:* this measures
**temporal** overlap. Two concurrent positions in different symbols are treated
as fully redundant, which they are not — they share $\mu_t$, not $\eta$. It is
therefore the right bound on independent **time** observations, and the
correlation-adjusted breadth above is what speaks to the cross-section.

### Tier 4 — is the backtest real

**PBO by combinatorially symmetric cross-validation** (Bailey, Borwein, López de
Prado & Zhu 2017): cut $T$ rows into $S$ blocks; over every choice of $S/2$
blocks as in-sample, find the in-sample-best configuration and record its
out-of-sample rank $w$; with $\lambda = \ln(w/(1-w))$, $\mathrm{PBO} =
\Pr[\lambda \le 0]$. A PBO near 0.5 means selecting the best cell of a sweep
tells you nothing about which will be best next.

**Stationary bootstrap** (Politis & Romano 1994) — geometric blocks, wrapped
circularly. Unlike the project's day-clustered resample this preserves
dependence *within* the return series, which is what a Sharpe or drawdown
depends on.

### Tier 5 — execution realism

Amihud (2002) ILLIQ; **Corwin & Schultz (2012)** high–low spread and Roll (1984)
— both from OHLC alone, so they are an independent cross-check on
`cost_model.py`'s quote-fitted schedule rather than a restatement of it; and the
square-root impact law $\Delta P/P = Y\sigma\sqrt{Q/V}$ (Torre & Ferrari 1997;
Almgren, Thum, Hauptmann & Li 2005).

## 1.3 The corrected book

`research/penny/strategy/book.py` replaces three divergent simulators
(`slots.run_slots`, `risk_shaped.run`, `long_reversion_portfolio.simulate`) with
one. Four defects fixed, each unit-tested:

1. **Cost charged exactly once.** `cross_section.forward_return` returns `R`
   already net of a full round trip; `risk_shaped.attach_target_returns` returns
   `R_t*` **gross**; every simulator subtracted the same `0.5 * cost` from
   whichever it was handed. The time-stop arm therefore paid **1.5 round trips**
   and the profit-target arm **0.5** — a threefold difference between two arms of
   one sweep, favouring the target. The book works in gross space and applies one
   size-aware round trip at the position size actually traded.
2. **Exits scheduled on the market calendar**, not on candidate days.
3. **A real no-selection control**: `fill` and `veto_frac` are independent, where
   `slots.py`'s "noveto" arm still filled with `nlargest(COMPOSITE)`.
4. **The daily equity curve is returned**, which is what makes every metric above
   computable at all.

---

# PART II — the concurrency problem (fixed here, NOT yet run)

## 2.1 The identity, and what is already closed

`FORMULATION.md` Amendment 2:

$$N_{\text{pos}} = \frac{H}{\tau}\,K\,(1-\delta)$$

| lever | status | evidence |
|---|---|---|
| cadence $\tau$ | **closed** | §80 finding 1 — 2 of 84 configurations admissible, best Sharpe 0.70 |
| horizon $H$ | **closed** | §80 finding 2 — the excess is a threshold, not an accrual; CI excludes zero only at $H\ge60$ |
| cohort size $K$ | open, but costly | §78 — $\mathbb{E}[R]$ is flat in the score inside the survivors |

The last row is the whole problem:

$$\mathbb{E}\!\left[\tfrac1K\textstyle\sum_{i\in S_K} R_i\right] = \mathbb{E}\!\left[\tfrac1{|S|}\textstyle\sum_{i\in S} R_i\right],\qquad \operatorname{Var} = \frac{\sigma^2_\eta}{K} + \sigma^2_{\text{common}}$$

**Only the variance moves.** Cutting $K$ without a selector is random
subsampling. §84 measured its price: Sharpe 0.52 at N=15 on a \$50k account and
**0.66 at \$25k, where cost is not binding** — so the floor at 15 is a breadth
floor, not a cost one.

## 2.2 What is already measured dead, and must not be re-proposed

- §80: the composite's **top decile scored −1.18% excess** [−3.20%, +0.97%] while
  a **random feature scored +0.77%**. The existing feature set is dead *as a
  selector*.
- `6cb0b55`: B4 dilution (`shares_growth_1y`, IC −0.0090, t −1.09), B5 filings
  (best t −1.98), B6 short interest (t +0.40) — every non-price block, dead
  against the within-cell estimator.
- §70, §82: stop losses, twice, on different populations and horizons.
- §71: entry at the signal day's close.
- §60: premarket at every threshold tested.

## 2.3 Hypotheses, fixed now

### H1 — the variance model

Estimate the tail index of per-trade returns and re-derive the required breadth
under the measured tail rather than a Gaussian assumption. Re-measure
$\mathrm{sd}(\mu_t)$ on the **vol-capped** population.

*Motivation.* §82's decomposition (+100% row: $\mathrm{sd}(\mu_t)=24.9\%$,
$\mathrm{sd}(\eta)=41\%$) predicts portfolio
$\mathrm{sd}=\sqrt{0.0620+0.1681/N}$, so N=15 → 0.2706 and N=60 → 0.2546: a **6%
Sharpe improvement**. §84 measured **0.52 → 1.36, a 161% improvement**. Both
cannot be right. Worse, §84's N=400 holdout Sharpe of 2.66 **exceeds the
decomposition's own ceiling** of $\mu/\mathrm{sd}(\mu_t)=1.11$ per 60 sessions
(≈2.27 annualised).

*Prediction, recorded before the run.* $\mathrm{sd}(\mu_t)$ measured on the
vol-capped population is materially below the 24.9% measured on the uncapped one.

*Falsified if* capped and uncapped $\mathrm{sd}(\mu_t)$ agree within their
intervals — the two results then disagree for some other reason and this branch
stops.

### H2 — $\sigma_{\text{common}}$ as a fourth lever

Cohort-composition constraints that leave the identity untouched: a cap on names
per sector using the cached `sic_sectors.parquet`, and correlation-aware slot
allocation.

*Motivation from the tape, not only theory.* The live screen run on 2026-08-28
fired ten names; SMJF, KXIN, SUGP, MIMI, RCON and SGLY were all China-domiciled
microcaps unwinding together. Ten slots bought one position six times. If
$\sigma_{\text{common}}$ is what the vol cap already moves, moving it further
lowers the $N$ needed for a given Sharpe without touching $\tau$, $H$ or $K$.

*Accepted if* Sharpe at fixed $N$ rises — or the $N$ needed for a target Sharpe
falls — with the CI excluding zero on development, confirmed once on holdout.

### H3 — a selector from minute bars, the one untapped asset

`data/research_penny/minute/dt=*` holds **2,628 sessions, 2016-02-17 →
2026-08-07, 921 MB**, used only ever for premarket aggregates and barrier
labelling — never as cross-sectional features. Coverage is
candidate-symbols-per-session and is enumerated in each partition's
`requested.json` sidecar, which **must be honoured**: a session cached under
round 3's screen is present on disk but missing round 4's symbols, and a
file-existence check silently returns stale coverage.

Two estimands, because §78 established the signal is monotone but **left-tailed**
— good at removing losers, useless at picking winners:

1. **Within-cell IC** on the demeaned return, the same bar every block has faced
   (`cross_section.score_feature`, which bootstraps cells, not observations),
   against §80's benchmark of −1.18% top-decile excess and +0.77% for a random
   feature.
2. **`P(touch the +100% barrier | f)`** with a precision ladder
   (`eval/precision.py`). With a barrier exit the portfolio return is dominated
   by *which names touch it*; a variable lifting the touch rate from 20% to 30%
   moves the economics decisively while barely registering as a Spearman IC on
   the mean. This matches the standing framing that this is a precision game
   where recall stays low.

*Candidate features* (all computable from cached minute bars, all point-in-time
at or before the signal close): intraday realised-volatility term structure; the
low-vol effect measured at minute rather than daily frequency; volume-profile
shape; closing-auction participation; time-of-day of the crash; and the intraday
path of the signal session.

*Accepted if* top-decile precision beats the base rate with the CI excluding it
**after BH against the full cumulative trial count, these trials included**.

### H4 — weighting (only if H2 or H3 survives)

Equal weight is near-optimal under a veto precisely because $\mathbb{E}[R]$ is
flat in the score; it stops being optimal the moment a selector creates
dispersion. Then: inverse-volatility (which compounds the validated low-vol
effect), fractional Kelly for gross exposure (Kelly 1956; Thorp 2006; MacLean,
Thorp & Ziemba 2011), and Hierarchical Risk Parity (López de Prado 2016) as the
covariance-free comparator, with Ledoit–Wolf (2004) shrinkage for any covariance.

### H5 — entry timing

Entry is currently the next open — the widest-spread print of the microcap day.
The same minute data answers this properly: next open vs a 10:00 entry vs a
first-30-minute VWAP. §71 already showed the overnight move is not real, which
removes the reason to pay up for the open. Anchors: Heston, Korajczyk & Sadka
(2010) on intraday periodicity; Bogousslavsky (2021) on intraday reversal.

## 2.4 Constants fixed now

| constant | value |
|---|---|
| population | fell ≥15% over 5 sessions, \$0.50–\$10 unadjusted, ADV ≥ \$250k, cell ≥ 5 |
| horizon $H$ | 60 sessions |
| profit target | +100%, no stop |
| entry | next open (H5 tests alternatives; the default does not move) |
| composite signs | frozen at the pre-2020 fit, never refitted |
| dev / holdout | signs `< 2020-01-01`; development `2020-01-01 … 2024-01-01`; sealed holdout `≥ 2024-01-01` |
| cost | one size-aware round trip, `cost_model.round_trip_bps` |
| seeds | 15, with the seed dispersion reported beside every Sharpe |
| account | \$50,000 |

## 2.5 Leakage rules

- **Filters touch unadjusted prices only** (`etl/screens.py`, §72). Two traps:
  `is_safe_for_screen` defaults to `return True`, so a new feature is **not**
  caught unless added to `ADJUSTED_LEVEL_COLUMNS`; and `dv`/`adv20` are
  whitelisted yet computed from the adjusted close
  (`build_universe.py:108-111`).
- Point-in-time joins: EDGAR on `accepted_et`, XBRL on `known_from`, FINRA lagged
  one session. Minute features may use only bars at or before the signal close.
- `eval/folds.py` documents that purging is unnecessary because labels are
  same-session. That was true for rounds 1–6 and is **false at a 60-session
  hold**. No fold-based estimator may be used in R15 until purging is added.
- The holdout is read **once per hypothesis**, and only after the development
  decision is recorded here.

## 2.6 The trial ledger

Reads of the sealed holdout: §78 (1), §80 (2), §82 (3), §84 (4), and **Part I's
re-cost (5)**. Part I is a correction of a mis-costed baseline rather than a new
selection, but it is a fifth look and is recorded as one.

Deflated Sharpe is computed against `search_trials.jsonl` (195) plus
`PRIOR_TRIALS = 72` plus this round's own cells.

---

## References

Almgren, R., Thum, C., Hauptmann, E. & Li, H. (2005). Direct estimation of equity market impact. *Risk*, 18(7).
Amihud, Y. (2002). Illiquidity and stock returns. *Journal of Financial Markets*, 5(1).
Bailey, D. & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2).
Bailey, D. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5).
Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4).
Bali, T., Cakici, N. & Whitelaw, R. (2011). Maxing out: stocks as lotteries and the cross-section of expected returns. *Journal of Financial Economics*, 99(2).
Biglova, A., Ortobelli, S., Rachev, S. & Stoyanov, S. (2004). Different approaches to risk estimation in portfolio theory. *Journal of Portfolio Management*, 31(1).
Bogousslavsky, V. (2021). The cross-section of intraday and overnight returns. *Journal of Financial Economics*, 141(1).
Buckle, D. (2004). How to calculate breadth: an evolution of the fundamental law of active portfolio management. *Journal of Asset Management*, 4(6).
Clarke, R., de Silva, H. & Thorley, S. (2002). Portfolio constraints and the fundamental law of active management. *Financial Analysts Journal*, 58(5).
Corwin, S. & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance*, 67(2).
Grinold, R. (1989). The fundamental law of active management. *Journal of Portfolio Management*, 15(3).
Heston, S., Korajczyk, R. & Sadka, R. (2010). Intraday patterns in the cross-section of stock returns. *Journal of Finance*, 65(4).
Hill, B. (1975). A simple general approach to inference about the tail of a distribution. *Annals of Statistics*, 3(5).
Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3).
Keating, C. & Shadwick, W. (2002). A universal performance measure. *Journal of Performance Measurement*, 6(3).
Kelly, J. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4).
Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2).
Lo, A. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4).
López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *Journal of Portfolio Management*, 42(4).
López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (ch. 4 uniqueness, ch. 7 purging, ch. 12 CPCV.)
MacLean, L., Thorp, E. & Ziemba, W. (2011). *The Kelly Capital Growth Investment Criterion*. World Scientific.
Magdon-Ismail, M. & Atiya, A. (2004). An analysis of the maximum drawdown. *Quantitative Finance*, 4(5).
Martin, P. & McCann, B. (1989). *The Investor's Guide to Fidelity Funds*. Wiley.
Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working paper.
Politis, D. & Romano, J. (1994). The stationary bootstrap. *JASA*, 89(428).
Rockafellar, R. & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3).
Roll, R. (1984). A simple implicit measure of the effective bid-ask spread. *Journal of Finance*, 39(4).
Sortino, F. & Price, L. (1994). Performance measurement in a downside risk framework. *Journal of Investing*, 3(3).
Thorp, E. (2006). The Kelly criterion in blackjack, sports betting and the stock market. In *Handbook of Asset and Liability Management*.
Torre, N. & Ferrari, M. (1997). *Market Impact Model Handbook*. BARRA.
Young, T. (1991). Calmar ratio: a smoother tool. *Futures*, 20(1).
