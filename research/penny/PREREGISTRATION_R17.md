# Round 17 Pre-Registration — R15 measured the way it would be traded

**Status:** FIXED 2026-11-22, before any causal number was computed. Results are
recorded in `LOG.md` §100 and in the outcome table added above §0 afterwards;
the body below is not edited once a run exists.

## Outcome, recorded after the fact (2026-11-22)

| | outcome |
|---|---|
| **Cell A** | Sharpe **0.68**, 95% CI [−0.27, 1.73]; CAGR +15.1%; maxDD −31.7%; alpha vs SPY +5.6%/yr, NW CI [−11.4, +22.6]; beta 0.84. SPY 0.61, QQQ 0.76. |
| **Cell B** | Sharpe 0.56; maxDD −31.9%. **Stop rejected** (rule 1). |
| **Decision** | **KILL** (rule 2: the adopted cell's Sharpe CI includes 0). R15 is not taken live and **no holdout read #8** follows. |
| Diagnostic D1 | The same causal trades score 0.77 marked to market and 1.63 under realised-only booking (gross of cost). |

Erratum, pointer only: the results are in `LOG.md` **§101**, not §100, which
was already taken.

**What had already run when this was written (and nothing else):**
1. The legacy reproduction: `r15_matched_book.run` at 60 slots, sector cap 3,
   15 seeds on dev → **Sharpe 1.2138 ± 0.2015, +12.5%/yr, maxDD −25.2%,
   41.0 live positions**. It matches §91 exactly.
2. Two data-integrity audits that read **prices, not returns or outcomes**: split
   flags against the adjusted series, and Alpaca corporate-action records against
   the panel (§2.3).

---

## 0. Why this round exists

R15 is the only penny strategy validated on both dev and holdout (Sharpe 1.21 /
1.59). It is about to drive a live account. Four defects stop its numbers from
serving as the go/no-go.

| # | defect | where | effect |
|---|---|---|---|
| 1 | **Selection reads the future.** The frame drops any candidate with a flagged split anywhere in its next 61 bars, and any candidate whose next-open entry or forward closes are missing. | `cross_section.forward_return:89-90`, applied in `veto_holdout.frame:31-32` | Rows go missing *before* the day's ≥5-name check (`frame`), the within-day volatility median (`book.py:232-234`) and the composite veto. The selected names, not just the dropped ones, change. A live account knows none of this at the close of day t. |
| 2 | **The equity curve is realised-only.** A position is carried at cost until `signal + 60` (`book.py:305, 329`), so open P&L never shows in the daily returns. | `book.py` | The Sharpe and drawdown leave out every mark-to-market swing. They cannot be compared with SPY/QQQ, which are marked daily. |
| 3 | **Dev reads holdout prices.** Dev signals in late 2023 have 60-session forward windows that run into 2024. | `reproduce_round14.build_frame` | The dev number uses up to ~3 months of sealed-period prices. |
| 4 | **The holdout is worn.** 7 reads so far (PREREGISTRATION_R15 Outcomes). | — | With H = 60, the only clean out-of-sample evidence left is forward paper trading. |

**The data audit changed what "causal" has to mean (§2.3).** Of the panel's
split flags inside dev holding windows, only **16%** match a split in Alpaca's
corporate-action records. The other 84% are **genuine price jumps** on
near-round multiples that the wide `is_possible_split` flag catches. So defect 1
did not remove artifacts. It removed real trades, in both directions.

---

## 1. Held fixed from R15

- Population: `cross_section.candidate_mask` (unadjusted close $0.50–$10,
  trailing ADV20 ≥ $250k, 5-session unadjusted return ≤ −15%, no split flagged
  on the signal day, lookback span limit).
- Filters within the day: volatility cap 0.50 on `B7_realised_vol`, then the
  composite veto 0.25 with signs frozen at the pre-2020 fit (`specs/VETO.json`).
- Book: 60 slots, sector cap 3 (SIC, `sic_sectors.parquet`), at most
  ceil(60/60) = 1 entry per session, equal weight 1/60, random fill, seeds 0–14,
  $50,000 account.
- Cost: `cost_model.round_trip_frac` (size-aware, ADV-decile fit) at the
  position's notional, plus §2.6's regulatory sell fees.
- Exit horizon H = 60 sessions; profit target +100%.

## 2. The causal measurement (what changes)

### 2.1 Frame
Candidates are `candidate_mask & period`. The ≥5-names-per-day check is counted
on candidates. **No row is removed for anything that happens after its signal
date.** Features come from `cross_section.build_features` (trailing only).

### 2.2 Entry: buy limit in the opening auction
- The order goes into the opening auction of the next market session `e` at
  `L = close_unadj(t) × 1.10`. **δ = +10% is fixed here.** It is a collar
  against a thin auction printing a news gap, not a signal. It is the only
  value of δ that will be run.
- It fills iff the stock has a bar on `e` and `open_unadj(e) ≤ L`, at
  `open_unadj(e)`.
- Unfilled means skipped: no position, no cost, the slot stays free, and the
  session's one entry is spent. There is no retry and no substitute name.

### 2.3 Holding path: split- and dividend-adjusted
- The path is built from the vendor's adjusted OHLC divided by the adjusted
  open on `e`. The anchor cancels inside a window, so this is a return, not a
  screen.
- The vendor misses some splits. For every split in Alpaca's corporate-action
  records (`scripts/fetch_corporate_actions.py`, fetched 2019-10-01..2023-12-31
  only) whose ex-date falls in the window:
  - If the adjusted jump on ex-date −1, 0 or +1 is within 25% of the split's
    price multiple, the adjusted series has **not** absorbed it. Multiply the
    path from that day on by `new_rate/old_rate`.
  - Otherwise leave it.
  - A split whose multiple is itself within 25% of 1 cannot be told from noise
    and is left alone.
- Panel split flags with no vendor record are treated as genuine moves.
- Audit, prices only: vendor splits fall on the unadjusted jump day 86.7% of the
  time and are absorbed by the adjusted series 86.7% of the time. 0.42% of dev
  candidate windows contain a vendor split.

### 2.4 Exits
Barriers are live from the entry bar `e` onward (a GTC exit pair placed after
the opening fill).
- **Target:** the adjusted high reaches 2 × entry → exit at exactly 2 × entry.
  It is never credited above the limit.
- **Stop (cell B only):** the adjusted low reaches 0.5 × entry → exit at
  `min(0.5 × entry, adjusted open of that bar)`, so a gap through the stop fills
  at the open.
- **Same bar touches both:** the stop.
- **Time:** the adjusted close of market session `e + 60`. If the stock has no
  bar that session, exit at the next bar's close; if there are no more bars, at
  the last print.
- **Halts:** the mark carries forward. **Delisting:** exit at the last print.
  The audit found no early path end within 10 days before a recorded rename, so
  early ends are treated as delistings.

### 2.5 Accounting: marked to market
- Position value is `size × (M_k − c)`. `M_k` is the adjusted close on session
  k over the entry price, frozen at the exit price after an exit. `c` is the
  full round-trip cost, charged at entry.
- From the decision (close of t) to the fill (open of e), the reserved cash
  counts at par.
- Sizing uses equity marked at the decision close: `size = min(equity_t / 60,
  cash)`.
- As in R15, the slot stays locked until session `e + 60` even after a target or
  stop exit; its capital then returns to cash. It is released later only if the
  time exit itself is late (a halt).
- Order of events within a session: release, decide, mark.
- The daily return is the change in marked equity. Sharpe =
  `mean / sd × √252` on raw returns, as in every prior round.

### 2.6 Regulatory fees (Alpaca charges $0 commission)
Constants are taken from the cited fact sheet
`research/intraday/docs/research/2026-11-alpaca-order-facts.md`, using the
higher of the recent and scheduled rates.
- **On every sell:**
  - SEC fee: 2.78 × 10⁻⁵ × notional ($27.80 per $1M, above today's $20.60).
  - FINRA TAF: $0.000195 per share, capped at $9.79 (the 2027 rate).
- **On both sides:** CAT fee, $0.000003 per share.

Shares = position notional / entry price. All three fees are added to `c`.

### 2.7 Period
- Dev = market sessions 2020-01-02..2023-12-29. Signals only on dev sessions.
- Marks use only bars dated ≤ 2023-12-29. Positions still open at the end are
  marked there, not closed, so **no sealed-period price is read**.
- `dates.assert_dev` enforces this on every array the run reads.

## 3. Cells — exactly two

| cell | rule |
|---|---|
| **A** | causal (§2), no stop |
| **B** | causal (§2), stop at −50% of the split-adjusted entry |

Cumulative trial count **315 → 317**. These are the only causal cells to be run.
Anything else computed is a labelled diagnostic and is never selectable.

## 4. Statistics

- Per seed: the MTM daily return series `r_s`. **Point Sharpe** is the mean over
  the 15 seeds of the per-seed Sharpe, as R15 reported. **Max drawdown** is the
  mean over seeds.
- **Benchmarks** are total-return daily series on the same sessions: SPY, QQQ,
  and IWM as a diagnostic. They come from Alpaca daily bars with
  `adjustment="all"` (dividends included), fetched to 2023-12-29 for this round.
- **CIs** use one stationary bootstrap (Politis & Romano 1994,
  `battery.stationary_bootstrap_indices`): mean block 60 sessions, 2,000
  replicates, seed 17. The same resampled index is applied to all 15 seeds and
  to every benchmark, so the differences are paired. CIs are 95% percentile
  intervals.
- **Sharpe difference vs SPY and vs QQQ:** a point estimate and the paired
  bootstrap CI. As a cross-check, the Ledoit–Wolf (2008) HAC test (Parzen kernel,
  automatic bandwidth) on the seed-averaged return series.
- **Alpha and beta vs SPY:** OLS of the seed-averaged daily excess return on
  SPY's daily excess return. Excess is over the 3-month T-bill (FRED DTB3, as a
  daily rate). Newey–West standard errors with 60 lags; annualised alpha =
  252 × intercept; the bootstrap CI is reported alongside. Alpha and beta are
  linear in returns, so the seed-averaged series has the mean of the seed alphas.
  IWM beta is reported as a diagnostic.
- **Also reported:** CAGR; limit fill rate; exit mix (target / stop / time /
  delist / open at the end); live positions; trades with a vendor-split
  correction; deflated Sharpe at N = 317.

## 5. Decision rules (fixed now)

1. **Stop adoption.** Adopt B iff `Sharpe_B ≥ Sharpe_A − 0.05` **and**
   `maxDD_B > maxDD_A` (shallower). Otherwise adopt A.
2. **Kill.** For the adopted cell: if the 95% CI of its Sharpe includes 0, **or**
   its point Sharpe is below SPY's dev Sharpe on the same sessions, R15 is
   **not** taken live. No holdout read follows, and the live system's first
   strategy moves to the liquid intraday track (plan Phase 4).
3. **Otherwise** R15 goes forward to holdout read #8, **only on the user's
   explicit go**.
4. **Holdout read #8 pass criterion (fixed now).**
   - Signals 2024-01-01 → the last session with full data, marks to that
     session, cell and every setting unchanged.
   - **Pass** iff the point Sharpe exceeds both SPY's and QQQ's on the same
     sessions **and** the 95% Newey–West lower bound of annualised alpha vs SPY
     is > 0.
   - Sharpe above both with an alpha CI that includes 0 is **inconclusive**: R15
     may go to paper trading labelled "unconfirmed".
   - Anything else is a **fail**.

## 6. What this round cannot tell us

- Fills: a daily bar has no queue, depth or auction imbalance. The fill rule is
  `open ≤ limit`; the true auction price for a $833 order is not observed.
- The cost model is fitted on 2023 NBBO snapshots and applied to 2020–2023.
- 60-session holds over 4 years give ≈ 17 independent observations, so CIs will
  be wide.
- Sector labels are static (R15 §92 limit 3).
- Which symbols exist in the daily panel comes over from R15 and is not
  re-audited this round. `in_universe` itself is causal: prior close, trailing
  ADV, the day's own return.
- Paper trading fills auction orders at the quote, not at the auction price,
  per Alpaca staff on the forum (fact sheet item 4). Forward paper therefore
  cannot test this fill rule either.
