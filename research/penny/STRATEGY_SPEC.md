# Penny-stock strategy specification

> **Round 17 (2026-11-22): do not use the Sharpe figures below as evidence.**
> - They come from a realised-only equity curve and a frame that read the future
>   (`PREREGISTRATION_R17.md`, `LOG.md` §101).
> - Measured causally and marked to market on dev, the strategy scores **Sharpe
>   0.68** with a 95% CI of [−0.27, 1.73]. SPY scores 0.61 and QQQ 0.76.
> - Alpha is +5.6%/yr with a CI of [−11, +23]. Beta to SPY is 0.84.
> - It was **killed for live trading** by the pre-registered rule. Under
>   realised-only booking the same trades score 1.63.

**Status as of round 15 (2026-08-30).** This is the consolidated design: the
strategy as it now stands, every element of it, the reasoning and references
behind each choice, and the catalogue of what has been tested and what remains.

Companion documents: `FORMULATION.md` (the round-11 estimator),
`PREREGISTRATION_R15.md` (the evaluation battery's mathematics and this round's
hypotheses), `LOG.md` (the full process record, §1–99).

---

## 1. The strategy

> **Population.** Common stock on AMEX/NASDAQ/NYSE. Unadjusted close
> \$0.50–\$10.00. ADV20 ≥ \$250,000, computed strictly trailing. Fell **≥15% over
> the previous five sessions**. At least five eligible names that day.
> No corporate action over the lookback or the holding window, either direction.
>
> **Filter.** Keep the **lower half by 20-day realised volatility**, ranked
> within the day. (Optionally drop the worst quarter by composite; see §3.2 —
> it adds nothing once the volatility cap is applied.)
>
> **Book.** 60 slots, **at most 3 names per sector held concurrently**, equal
> weight, at most `ceil(N/H) = 1` new position per session.
>
> **Entry.** The next session's open, or 10:00 (§4).
>
> **Exit.** +100% profit target, or the close of session 60. **No stop.**
>
> **Cost.** One size-aware round trip, `2·h·(1 + max(order/inside − 1, 0))`,
> fitted by ADV decile.

Measured: development Sharpe 1.21, holdout **1.59**; drawdown −25.2% / −14.9%;
~41–44 concurrent positions; about one order a session.

### 1.1 What the edge is

The **low-volatility effect inside crashed microcaps** — Bali, Cakici & Whitelaw
(2011)'s MAX finding in the corner of the market where lottery preference should
be strongest. §80 established that the seven-feature composite validated in §78
was a wrapper around this one feature (IC −0.0845, t −12.0, against +0.038 for
the next best), so the strategy needs **one feature and no fitted signs**.

It is a **veto, not a selector**: §78 measured the top decile at −1.18% excess
where a random feature scored +0.77%. The composite removes losers and cannot
pick winners, and §96–97 found nothing that can.

---

## 2. Filters and patterns — the catalogue

### 2.1 In the strategy

| filter | value | why | reference |
|---|---|---|---|
| crash | ≥15% over 5 sessions | defines the population; short-horizon reversal | Jegadeesh (1990), Lehmann (1990) |
| price band | \$0.50–\$10 **unadjusted** | the microcap regime; **must be unadjusted** (§72) | — |
| ADV floor | ≥ \$250k trailing | below this the spread eats the edge | round 4, §37–38 |
| **volatility cap** | lower half, 20-day realised, ranked within the day | **the validated effect** | Bali–Cakici–Whitelaw (2011); Ang et al. (2006) |
| corporate actions | excluded over lookback **and** holding window, both directions | §57 (reverse on exit day), §72 (forward in lookback) | — |
| cell size | ≥5 names | a cell too small to rank cannot be ranked | §78 |
| **sector cap** | ≤3 concurrent per SIC sector | cuts `sigma_common`; 28% smaller book | §92, §94 |

### 2.2 Tested and dead — do not re-propose

| pattern | verdict | where |
|---|---|---|
| intraday veto sets (09:35→15:55) | negative gross before costs | rounds 1–4 |
| gap screens (`gap ≥ +3%`) | the screen selected the losing tail | §5 |
| momentum (up A% → continues) | gross negative | 441-grid |
| gap up ≥ +5% | −2.06%, gap-ups **fade** | 441-grid |
| reverse split → rise | −10.72%, hypothesis inverted | 441-grid |
| overnight (buy close, sell open) | +0.08% over 317,873 obs; no anomaly | 441-grid |
| volume shock, range compression, new highs/lows, consecutive downs | net negative | 441-grid |
| premarket volume, every threshold | ≥1× ADV **−0.769%**; ≥20× ADV **−2.928%** | §60 |
| **stop losses** | IR/pos 0.217 → 0.129 at −20%; rejected twice | §70, §82 |
| entry at the signal day's close | period split disagrees | §71 |
| dilution / share issuance (B4) | IC −0.0090, t −1.09 | `6cb0b55` |
| filings, deficiency, shelf count (B5) | best t −1.98, at the noise floor | `6cb0b55` |
| short interest (B6) | IC +0.0030, t +0.40 | `6cb0b55` |
| insider / institutional flow | ρ=+0.90 but a 22 bps spread | §41 |
| sector conditioning as a strategy | real but one event, CI 22 points wide | §76 |
| composite as a **selector** | top decile −1.18% vs random +0.77% | §80 |
| calendar entry cadence (τ) | 2 of 84 admissible, best Sharpe 0.70 | §80 |
| shorter horizon (H<60) | the excess is a threshold, not an accrual | §80 |
| intraday features as a selector | dev +0.6, **holdout −0.9 to −1.5** | §97 |
| inverse-volatility weighting | −0.11 / +0.10 / −0.00 / −0.01 | §93 |

### 2.3 Designed but never built — the live candidates

Ordered by expected value, with the reason each is still open.

**a. A finer partition than SIC sector.** §92's sector cap is the only lever that
worked, and 17 sectors is a coarse grid — a cap of one is a hard ceiling of 17
names, which is why it fails to bind smoothly. **A correlation-clustered
partition** (hierarchical clustering on trailing return correlations, or López de
Prado's (2016) HRP tree) would give a continuous concentration lever instead of a
ceiling. This is the direct continuation of the one thing that replicated.

**b. Balance sheet: cash / market cap, and net cash.** `FORMULATION.md` B4
specified them; `etl/xbrl.py:51-55` only ever fetched *share counts*. **No
balance-sheet concept has ever been fetched.** A crash to below net cash is
mechanically a different event from a crash on operations — the classic
deep-value screen (Graham's net-nets), and the one non-price idea that was
specified and never tested. Note B4's *share-count* half is dead, which is
evidence about dilution, not about cash.

**c. ATM program size and warrant overhang.** Open since §21. The strongest
mechanism story for why a microcap bounce dies: an at-the-market program sells
into every rally by construction. Requires filing-text parsing — the most new
machinery of anything here.

**d. Conditional holding period.** §80 closed `H` by shortening it *uniformly*.
It has never been tested as a **release rule**: keep H=60 as the maximum but
return capital early from positions that have already resolved. Since
`N = H × λ`, cutting the *average* hold cuts the book proportionally. The risk is
that this is a stop loss in disguise, and stops are dead twice over (§70, §82) —
so it must be tested as a *time/state* release, never a price stop.

**e. `short_exempt_volume`.** Present in `finra_short_volume.parquet`, never used
in any derived feature. Cheap to test; low prior, since B6's other columns died.

**f. Breakout / continuation patterns.** Deliberately **not** proposed. The
441-cell grid tested momentum, gap-ups, new 20-day highs and volume shocks; every
family was negative gross. In this universe the continuation trade is the losing
side, and that is one of the better-established facts the project has.

---

## 3. Position weighting and sizing

### 3.1 Weighting: equal, and this is not laziness

Equal weight is **optimal here for a specific reason**: §78 found `E[R]` **flat
in the score** among survivors, and flat expected returns is exactly the condition
under which equal weight maximises the information ratio — there is no dispersion
to tilt toward.

Inverse-volatility, the one alternative that compounds the validated effect
rather than betting against it, was measured at −0.11 / +0.10 / −0.00 / −0.01
across N (§93). Note this was **a no-op in every prior round** — `iv/iv.mean()`
over a one-name pool is identically 1 — so §80's and §82's weighting comparisons
were equal-weight against equal-weight.

Not worth testing until a selector exists: HRP (López de Prado 2016),
Ledoit–Wolf (2004) shrinkage. With 60 microcaps and short histories the
covariance is unestimable, which is what HRP is for — but with no expected-return
dispersion there is nothing for it to allocate.

### 3.2 The composite veto is vestigial

Keep it or drop it; §80, §82 and Part I's factorial all say it adds nothing once
the volatility cap is applied. Dropping it removes seven features and the entire
sign-fitting surface. **Recommended: drop it.**

### 3.3 Sizing

$$N_{\min} = \max\!\left(\underbrace{60}_{\text{breadth floor}},\ \frac{\text{account}}{\approx\$1{,}700}\right)$$

The breadth floor is real and is **not** a cost artefact: at a \$25,000 account
N=15 and N=60 pay nearly the same spread (94 vs 81 bps) and score **0.60 against
1.37** (§88). The second term is the inside-quote notional — above it, orders
walk the book.

Full-Kelly on the measured per-trade payoff is `f* = 0.629`. That is not
comparable to the book's 1/60 per name, because Kelly on one sequential bet is
not Kelly on sixty concurrent correlated ones. The comparison that holds: the
book is **long-only and fully invested**, so there is no leverage headroom in
either direction. **Breadth binds, not leverage** (Kelly 1956; Thorp 2006;
MacLean, Thorp & Ziemba 2011).

---

## 4. Entry timing

Currently the next session's open — the **widest-spread minute of a microcap
session**. Measured alternatives, barrier re-evaluated against each entry price
on the common subsample (§95):

| entry | mean gross | IR/position |
|---|---|---|
| open | +15.35% | 0.303 |
| **10:00** | **+16.25%** | **0.319** |
| 09:30–10:00 VWAP | +16.00% | 0.314 |

**Waiting until 10:00 is worth about +5% of information ratio and costs nothing.**
Consistent with intraday periodicity (Heston, Korajczyk & Sadka 2010) and with
§71, which found no overnight drift worth paying the opening spread to capture.

Not tested: a participation-capped order worked over the first hour, which is
what a real fill would be. That needs an execution simulator, not a price series.

---

## 5. Exits

**+100% profit target, no stop, 60-session maximum.**

The target is a **variance** instrument, not an upside one: it reduces the
per-trade standard deviation from 69.9% to 49.3% while giving up 1.6 points of
mean (+16.08% → +14.44%), which raises IR/position from 0.230 to 0.293. Only
**17.0%** of the population touches it, and 8–10% of trades actually taken.

**Stops are dead twice** (§70 on one population, §82 on another): IR/position
0.217 → 0.129 at −20%. The mechanism is that these names gap, so a stop is a
random exit at the worst moment rather than a floor.

The 60-session floor is not tunable: the excess is a **threshold**, with a CI
excluding zero only at H ≥ 60 (§80).

---

## 6. Execution and slippage

The binding real-world constraint, and modelled three ways so they can disagree.

**The charge.** `2·h·(1 + max(order/inside − 1, 0))` — flat below the inside
quote (~\$1,900 median notional), linear above it. Fitted from 9,794 measured
NBBO symbol-days by ADV decile. Both sides charged, exactly once per trade —
which had not been true of any prior round (§85).

**Independent cross-checks**, from OHLC alone, so they cannot restate the
quote-fitted model: Corwin & Schultz (2012) high-low spread, Roll (1984) serial
covariance, Amihud (2002) ILLIQ. Used in §96 to confirm that the extra cost of
thinner names was already being charged: the model's ratio 1.40× against
Corwin–Schultz's 1.43×.

**Impact, for capacity work:** `ΔP/P = Y·σ·√(Q/V)` (Torre & Ferrari 1997;
Almgren, Thum, Hauptmann & Li 2005). Not yet used to derive a capacity curve —
open.

**Not modelled, and it matters:** partial fills, halts (frequent in this
universe), the opening auction, and the fact that a limit at +100% may never be
lifted in size. Every result assumes the printed barrier is achievable.

---

## 7. Risk controls

| control | setting | evidence |
|---|---|---|
| concurrency | 60 slots | breadth floor, §88 |
| **sector concentration** | ≤3 per sector | the only lever that replicated, §94 |
| entry rate | ≤ `ceil(N/H)` = 1/session | prevents a synchronised cohort, §82 |
| one position per symbol | no re-entry while held | §65's overlapping-entry error |
| corporate actions | excluded both directions | §57, §72 |
| leverage | none, long-only, fully invested | §93 |

**Drawdown is worse than the Sharpe implies.** Realised maximum drawdown runs
**2.4–2.9× its own expectation** for a random walk with the same drift and
volatility (Magdon-Ismail & Atiya 2004) at N ≥ 60. Losses arrive together. Any
comparison across N must report this.

---

## 8. Evaluation protocol

Never a bare Sharpe. `research/penny/eval/battery.py`, five tiers, mathematics
and references in `PREREGISTRATION_R15.md` §1.2. The minimum reportable set:

1. **Sharpe with three uncertainties** — Mertens/Lo standard error, stationary
   bootstrap at a block equal to the holding period, and seed dispersion. They
   differ by up to 5× and the naive one is the narrowest.
2. **Deflated Sharpe** against the cumulative trial count (315 as of round 15),
   with `Var(SR_n)` from **seed-to-seed noise at a fixed configuration**, not
   from the spread across a sweep — the latter contains the real N-effect and
   over-deflates.
3. **Average uniqueness.** 972 nominal trades are ~17 independent time
   observations at H=60.
4. **Realised vs expected maximum drawdown.**
5. **A calibrated control in every comparison.** A random feature used as a fill
   rule moved holdout Sharpe by +0.42 (§97). Without the control that reads as
   an improvement.
6. **PBO** over any sweep whose best cell is being quoted.

**Standing invariants:** filters touch unadjusted prices only; EDGAR on
`accepted_et`, XBRL on `known_from`, FINRA lagged one session; features from a
possibly-adjusted source must be **scale-invariant in price *and* volume**
(§97); the holdout is read once per hypothesis and the read is recorded.

---

## 9. What is still missing

1. **A selector.** Without one the position count is fixed by `N = H × λ`. The
   bar is +41% IC per bet for 30 names, +100% for 15. Nothing has cleared it.
2. **A capacity curve** — the AUM at which square-root impact eats the alpha.
3. **Execution realism**: partial fills, halts, whether a +100% limit lifts in
   size.
4. **Purging in `eval/folds.py`**, whose docstring assumes same-session labels.
   Harmless today (only rounds 3/4/6 use folds) and blocking for any future
   fold-based work at H=60.
5. **A dates module.** `SEAL = 2024-01-01` and `DEV_START = 2020-01-01` are
   duplicated across ten files.
6. **A live screener for this configuration.** `screen_today.py` screens the
   retracted §65 rule and should be replaced or removed.

---

## References

Almgren, Thum, Hauptmann & Li (2005) · Amihud (2002) · Ang, Hodrick, Xing & Zhang
(2006) · Bailey & López de Prado (2012, 2014) · Bailey, Borwein, López de Prado &
Zhu (2017) · Bali, Cakici & Whitelaw (2011) · Buckle (2004) · Clarke, de Silva &
Thorley (2002) · Corwin & Schultz (2012) · Grinold (1989) · Heston, Korajczyk &
Sadka (2010) · Hill (1975) · Jegadeesh (1990) · Keating & Shadwick (2002) · Kelly
(1956) · Ledoit & Wolf (2004) · Lehmann (1990) · Lo (2002) · López de Prado (2016,
2018) · MacLean, Thorp & Ziemba (2011) · Magdon-Ismail & Atiya (2004) · Martin &
McCann (1989) · Mertens (2002) · Politis & Romano (1994) · Rockafellar & Uryasev
(2000) · Roll (1984) · Sortino & Price (1994) · Thorp (2006) · Torre & Ferrari
(1997) · Young (1991)

Full citations in `PREREGISTRATION_R15.md`.
