# Stage 2 research program: minute-level entry and exit on the Stage-1s pool

**Status: APPROVED by the user 2026-10-28.** Experiments run in the order of §10; each is pre-registered here before it runs. Items marked
**[my call]** are defaults the user can overturn; items marked **[OPEN]** need the user.
Parent doc: `2026-10-27-pool-then-trade-poc.md` (§1-§13). Literature review in progress:
`../research/2026-10-27-stage2-literature.md`.

## 0. Why a new program (the user's critique, accepted)

1. **Under-specified experiments.** Stages 2/2b/2c never wrote down, in one place, the exact
   label, how each row is tagged, the model output, the decision rule, and the metric. Every
   experiment below carries all five (§4, §5, §8).
2. **Resolution mismatch.** Stage 1 now selects names for *minute-to-minute* trade potential
   (the share-beats-costs label). Stages 2b/2c then traded them with a *daily-bar* bracket plan.
   Stage 2 (the only minute model) ran on the Round-1 *volatility* pool with a 15-bar LightGBM.
   From here on, Stage 2 runs at minute resolution **on the Stage-1s pool's own rows**, so the
   training distribution matches the deployed one.
3. **Thin search.** No time-series architecture and no global context (day and multi-day levels,
   support/resistance, momentum, market, sector, events) were tried. §6-§7 cover them, and §10
   orders them so each one is isolated.

Honest prior, **corrected 2026-10-28** (§1.1): on admissible rows, the Stage-1s+pm pool's average
random trade is -0.18% before costs; the picks fall -0.94% open to close and the universe -0.74%.
The earlier -0.25% / -1.33% included outcome-selected rows. The literature review
(`../research/2026-10-27-stage2-literature.md`) found no peer-reviewed minute-level, after-cost
success on $1-$20 gappers, so the prior is a null. A long-only model has to find up-swings inside a
population that drifts down. E0 (§10) turns this into the precision the model must reach.

## 1. Population (exact)

### 1.1 Correction found by the builder check (2026-10-28)
- **Outcome-selected rows.** The Stage-1s universe (cached minute-days with >= 60 bars) never
  applied R16's provenance flag (`data/research_intraday/r16_provenance.parquet`). 13,548 of its
  106,452 rows (13%) are **not admissible**: they are in the minute cache only because of an
  outcome-selected pool (C2, e.g. R15 signal sessions, which "fell >= 15% by D's close"). Their
  mean open-to-close was -5.6%, against -0.6% on admissible rows.
  - Re-run on admissible rows only (same model, arms and bar; scratch `s1pm_admissible.py`):
    Stage 1s+pm **still PASSES** (22.7% vs universe 16.5%; vs room_model +0.4, CI +0.1..+0.7;
    vs base +0.7, CI +0.4..+1.0).
  - Stage 1s alone **still FAILS** vs room_model (-0.31, CI -0.57..-0.04).
  - Drift: universe -0.74%, picks -0.94%, picks' mean random trade -0.18%.
  - The random-trade analysis and the error analysis also used unfiltered cache rows. E0
    replaces their numbers.
- **The >= 60-session-bar rule is day-D information.** Bar count is only known at the close, so
  it is dropped. Labels are computed on whatever bars exist; a missing bar means no fill.
- **E-1 (new, before E0): universe repair.** The universe becomes admissible minute-cache rows
  inside the wide-band panel, with no bar-count filter. Stage 1s+pm is re-scored on it, as the
  §13 model unchanged (a correction, not a new trial), and the pool is that model's top 10.

- **Pool:** for each day D, the top 10 **admissible** cached gap-day names (§1.1) by the
  Stage-1s + pre-market model's out-of-fold score, re-scored in E-1 (was: commit 36d8491; days with >= 20 eligible names; scored 2017 -> 2025-08-06,
  1,361 days, about 13,600 pool stock-days). Every pool name has minute bars, because the
  Stage-1s universe is defined as cached names with >= 60 session bars.
- **Training population (ablation E3):** P1 = pool only; P2 = top 30 by the same score;
  P3 = the whole Stage-1s universe (about 96,500 stock-days) with the Stage-1 score and rank as
  features. **Evaluation is always on P1 (the top 10).**
- **Long only** (user decision, 2026-10-27).

## 2. Decision clock and execution model (exact)

- Minute index k = 0 (09:30) ... 389 (15:59), on the cache's adjusted price basis. Same-day
  ratios only. The traded (unadjusted) price is p_unadj = p_adj x f_D, with
  f_D = close_unadj_D / close_D from the daily cache (the price a live trader sees; not a
  forward-looking quantity).
- **Decision minutes:** t = 09:31 ... 15:30, after bar t closes. Training subsamples every 5th
  minute **[my call]**; evaluation uses every minute.
- **Entry:** market buy at the open of bar t+1, at O_{t+1} x (1 + h), where h is the half-spread.
  If bar t+1 is missing, there is no fill.
- **Exits:** the take-profit is a resting limit sell, filled only on a trade-through
  (High_k >= TP + 1 tick), at TP. **The stop (builder Q5, my call)** is a resting stop-market
  that triggers when Low_k <= SL. It fills at min(SL, O_k) x (1 - h): at the stop, or at bar k's
  open if the price gapped through. It never benefits from a wick that recovers. Sensitivity:
  min(SL, O_{k+1}) x (1 - h), the parent's next-open rule, capped at the stop. The time exit is a market sell at O_{T+1} x (1 - h). If TP
  and SL are both touched in one bar, it counts as SL. A run of >= 5 missing bars is treated as a
  halt: no fills, and exits move to the next bar that exists.
- **Account [as Stage 2c]:** $25,000, 10 slots x $2,500. One position per symbol, a 5-minute
  cooldown before re-entry, no entries after 15:30, flat by the 15:56 open. Shares are capped at
  10% of the entry bar's volume; an order under $500 after the cap is skipped.

## 3. Costs (exact)

- **Commission (the user's figure):** max($2.50, $0.01 x shares) per order, two orders per trade.
- **Spread:** h from Stage 2c's model (the refit book where it has inputs; otherwise
  max(55.4 bps rt, model) under $10, and a log-linear half-spread for $10+). It is charged on
  market legs only; a TP limit exit pays no spread.
- **Impact:** sigma_daily x sqrt(notional / ADV20), as in 2b/2c.
- Commission alone at $2,500, as a round trip: **$1: 2.00% | $2: 1.00% | $3: 0.67% | $5: 0.40% |
  $10+: 0.20%.** Below about $3, commission dominates every other choice. The price floor is an
  explicit lever in E6.
- **Point-in-time cost vs charged cost (builder Q3, my call):** the refit book's h comes from
  day-D quotes at 09:35 and 15:55, so it is not known at t. The feature c_rt, the barrier floor
  and the EV therefore use a **point-in-time estimate**: Stage 2c's log-linear half-spread model
  on prior-day inputs (log price at t, log ADV20, log 20-day volatility), plus commission at
  p_unadj(t), plus impact. The simulation then charges the **realized** cost: the refit book
  where it has inputs, else the same model. So "a TP hit is a net win" holds in expectation, not
  exactly.
- **Entries at 09:31-09:44 (my call):** there are no quotes before 09:35, and spreads are widest
  at the open (McInish & Wood 1992, via the literature review). These entries pay 2x the
  half-spread; E6 tests excluding the first 15 minutes.

## 4. Labels (exact tagging)

For pool name i at decision minute t, with entry e = O_{t+1}(1 + h):

- **sigma_t:** standard deviation of the last 30 one-minute log returns (pre-market bars included
  when t is early; at least 5 returns), floored at 0.5 x atr20_pct / sqrt(390).
- **Barriers:** u = max(k_u x sigma_t x sqrt(H), 1.5 x c_rt), so a TP hit is always a net win;
  d = k_d x sigma_t x sqrt(H); vertical barrier T = min(t + H, 385).
- **Primary label, triple barrier, 3 classes:** scan bars k = t+1 ... T in time order and tag
  - TP if High_k >= e(1+u) + 1 tick first,
  - SL if Low_k <= e(1-d) first (a tie inside one bar goes to SL; the fill is as in §2),
  - TIME if neither is hit by T.
- **Realized net return r_net** (always stored): TP -> u - commission/notional;
  SL -> min(e(1-d), O_k)(1-h)/e - 1 - commission/notional (the stop fill of §2);
  TIME -> O_{T+1}(1-h)/e - 1 - commission/notional.
- **Label B, multi-task partner of the triple barrier (user decision 2026-10-28):** Stage 1's
  share-beats-costs label, conditioned on the entry minute. For every later bar j that exists in
  [t+1, 385], take the net exit return r_j = C_j(1-h)/e - 1 - commission/notional. Then:
  - **B = share of those j with r_j > 0.** Costs are already inside r_j, so the threshold is 0.
    It is the minute-level version of Stage 1s's label: how many profitable exits this entry
    leaves open for the rest of the day.
  - Also stored: n_j (the number of sell minutes; it shrinks late in the day, and minute of day
    is a feature) and mean(r_j).
- **Training (multi-task):**
  - Deep models (M2-M5): a shared encoder with heads for A (3-class cross-entropy) and B (binary
    cross-entropy on the fraction). Loss = CE_A + lambda x BCE_B, with lambda = 0.5 **[my call]**.
  - LightGBM (M1) has no shared trunk, so A and B are separate models: multiclass, plus
    `cross_entropy` on the fraction. Only A's EV drives trades; B is reported (IC, calibration),
    and gating on B is an E6 ablation.
- **Secondary label, fixed-horizon regression:** y_H = O_{min(t+1+H, 386)}(1-h)/e - 1 - c_rt.
- **Auxiliary labels (for TP/SL placement heads):** MFE_H = max High / e - 1 and
  MAE_H = 1 - min Low / e over [t+1, T].
- **Default barrier set [my call]:** H = 30, (k_u, k_d) = (2, 1). E5 ablates
  H in {15, 30, 60, to-close} x (k_u, k_d) in {(2, 1), (1, 1)}.
- Labels use only bars after t, by construction. A unit test asserts that changing any bar <= t
  leaves the label unchanged, and that changing any bar > T leaves it unchanged too.

## 5. Model output and decision rule (exact)

- **Output per (i, t):** calibrated class probabilities p_TP, p_SL, p_TIME, plus mu_TIME (the
  expected r_net given TIME, from a regression head).
- **Expected value:** EV = p_TP x r_TP + p_SL x rbar_SL + p_TIME x mu_TIME, where r_TP and
  rbar_SL are the label's realized net returns (rbar_SL = the training-years mean SL loss per
  price bucket, gap-through included).
- **Calibration:** isotonic, fitted on the last training year (the inner validation year) only.
- **Rule:** enter if EV > theta while a slot is free; theta = 0 by default, and tuned only on the
  dev years (E6). When several names qualify in the same minute, the highest EV goes first.
- **Breakeven precision (reported in E0):** with no TIME outcomes,
  p* = |rbar_SL| / (r_TP + |rbar_SL|). A model is worth testing only if its top EV decile beats p*.

## 6. Features (point-in-time: bars <= t, daily rows <= D-1, pre-market < 09:30, filings accepted <= t)

| Family | Contents | Source / availability |
|---|---|---|
| F0 window (old Stage 2) | lr1, lr3, lr_15, bar range, window range, VWAP dev, volume ratio, up-fraction, missing bars, minutes since open | minute cache |
| F1 intraday levels and structure | HOD/LOD so far (distance in sigma and %), minutes since HOD/LOD, # new HODs; opening range 5/15/30 (distance, breakout state); VWAP distance and VWAP-band z; position in the day range; return since open and since prior close (red-to-green); pre-market high/low distance; nearest 5-bar fractal swing high above / low below; distance to round numbers ($0.50 / $1 grid, **unadjusted price**) | minute + pre-market caches |
| F2 multi-day levels | prior-day H/L/C distance; 5/20/60-day H/L distance; floor pivots P, R1, S1, R2, S2 from D-1; 52-week-high distance; consecutive up/down days; gap / atr20 | daily cache (5,354 symbols) |
| F3 momentum and volume | returns over 1/5/15/30/60 min; signed (up minus down) volume over 5/15/30 min; cumulative relative volume = cum volume / (ADV20 x intraday volume curve to minute k, the curve fitted on training years); volume acceleration; trade-count rate; sigma_5 / sigma_30 | minute + daily caches |
| F4 pre-market | §13's 10 features | pre-market cache (31%) + 09:25-09:29 bars (64%) |
| F5 market context | intraday breadth across all cached names at t (share above their open, median return since open, share making a new HOD in the last 5 min, dispersion); the same over the other 9 pool names; QQQ D-1 return and 5/20-day trend; VIX D-1 level and change | caches; QQQ and VIX daily exist. **SPY/QQQ/IWM minute bars (04:00-16:05) and daily bars: downloaded (user approved 2026-10-27), 2016-01-04 -> 2025-08-06** |
| F6 sector | SIC sector (2,879 symbols mapped, 13 groups); sector breadth among cached names at t | `sic_sectors.parquet`; **the 11 SPDR sector ETFs + XBI, minute and daily: downloaded**; plus stock-minus-sector-ETF return since open and over the last 15 min |
| F7 events and flows | EDGAR forms accepted before t (8-K items; S-1/S-3/424B* offerings = dilution; 10-Q/K) in the last 1/5/30 days and today before t, plus minutes since the latest; FINRA short-volume ratio D-1 and its 5-day mean | `edgar_filings.parquet` (has `accepted_et`); FINRA from 2018-08 |
| F8 meta | minute of day, weekday, Stage-1 score and rank, price bucket, c_rt | - |

- **Sequence input (for E4):** the last L in {60, 120} bars, with channels: log return; open,
  high and low relative to close; log volume / ADV-per-minute; VWAP distance; trade count; and a
  mask. Pre-market bars pad early t. Each stock is scaled by sigma_daily / sqrt(390).
- **Leakage tests (each family):** a perturbation test (random changes to bars > t must not move
  any feature at t), filing timestamps <= t, and daily rows <= D-1.

## 7. Models

| Id | Model | Size | Role |
|---|---|---|---|
| M0 | Logistic regression (L2) | - | floor |
| M1 | LightGBM multiclass + regression head (Stage 1's fixed params) | - | main baseline; the autoresearch workhorse |
| M2 | Dilated causal TCN | ~100k params | sequence |
| M3 | GRU, 2 x 64 | ~60k params | sequence |
| M4 | Transformer encoder, PatchTST-style (patch 5, d=64, 3 layers, 4 heads) | ~150k params | sequence; run after M2/M3 (literature: no evidence it beats boosting at minute frequency after costs) |
| M5 | Hybrid: the best of M2-M4's encoder + an MLP on F1-F8, with multi-task heads (3-class TB + y_H + MFE/MAE quantiles) | ~200k params | expected best |

- Deep models: AdamW, early stopping on the inner validation year, 3 seeds per configuration
  (reported as mean +/- sd), deterministic torch.
- Foundation models (Chronos etc.) are **dropped**. The literature review found that zero-shot
  financial forecasts lose after even 3 bp costs (Valeyre 2025), and that only finance-pretrained
  backbones help (Rahimikia 2025).
- **Literature prior for E4:** boosting on engineered features leads on tabular data (Grinsztajn
  2022; Qlib LightGBM IR 1.02 vs LSTM 0.56). Sequence models do better on raw windows. So the
  hybrid M5 is the deep model with the best chance.

## 8. Metrics (exact)

**Model level** (dev test years, per year and pooled, with day-clustered bootstrap CIs):
- Delta multiclass log-loss vs the class-prior forecast; Brier score; AUC (TP vs rest); ECE over
  10 bins, plus a reliability plot.
- **EV ranking:** the daily rank IC between EV and realized r_net over all of the day's (i, t);
  **top-decile lift** = mean r_net of the top 10% of EV decision points minus the mean over all
  points, and the top decile's TP-first rate vs p*.

**Trading level.** The pass bar (user decision 2026-10-28) is judged once, on the confirmation
period 2023-01-01 -> 2025-08-06. All of the following must hold:
1. The mean daily net return of the compounded $25k account has a 95% day-clustered bootstrap
   CI (2,000 resamples) whose lower bound is > 0.
2. Paired daily differences vs C1 (random timing), C2 (open-and-hold), C3 (the best
   practitioner rule, chosen on dev) and C4 (the shuffled-label model) each have a CI lower
   bound > 0, with Holm correction across the 4.
3. Deflated Sharpe >= 0.95, with N = every logged trial, autoresearch iterations included.
4. Reported, not gated: the 2x-cost run; each year; the run without the first 15 minutes; and
   C5 (SPY, QQQ) side by side: Sharpe, correlation and beta. Also Hansen's SPA p-value against
   C1 across all logged trials, and the t-statistic vs Harvey-Liu-Zhu's 3.0 hurdle.
- **Deflated Sharpe definition (builder Q4, my call):**
  - N = every trial logged in this program, plus the parent POC's 21 trials.
  - Every logged trial, autoresearch iterations included, also runs the account simulation on
    its dev predictions (cheap next to model fitting). This gives each trial a daily-return
    series.
  - V[SR] and PBO (CSCV) come from those dev series. The Sharpe tested is the confirmation-period
    Sharpe.
- **Secondary:** Sharpe (sqrt 252); Deflated Sharpe with N = every logged trial; PBO (CSCV over
  the dev daily returns of all trials); max drawdown; Calmar; hit rate; payoff ratio;
  trades/day; holding minutes; exposure; a gross -> commission -> spread -> impact
  decomposition; per year, price bucket and entry hour; capacity at 1x/2x/4x size; a 2x-cost run.

**Controls (same pool, days, slots and exits):**
- C1 random entry minute (same number of trades per day): the timing null.
- C2 buy at the 09:31 open and hold to the close: the pool's drift.
- C3 practitioner rules with the same barriers: ORB-15 breakout, VWAP reclaim, pre-market-high
  breakout, HOD breakout. Following Zarattini et al. 2024, the ORB rule is also run restricted to
  names with relative volume > 100% ("stocks in play"); that study found relative volume to be the
  feature that separates winners.
- C4 the same model on labels shuffled within each day: the model null. **It is trade-count
  matched (builder note, my call):** each day it takes its top-k by EV, with k = the real model's
  trade count that day. Otherwise "beats C4" can mean "beats no trades".
- C5 SPY (CLAUDE.md requires it; from the downloaded SPY daily bars through 2025-08-06) and
  QQQ, buy-and-hold over the same days.

## 9. Protocol (forking paths under control)

- **Development, purged block cross-validation (user decision 2026-10-28):**
  - The pool has only 4, 11 and 23 days in 2017, 2018 and 2019, so a yearly walk-forward cannot
    train its first folds.
  - 2020-2022 is split into **6 half-year blocks**. Each block is scored by a model trained on
    every other block plus all of 2017-2019, with a 5-trading-day embargo on both sides of the
    test block. Labels end by 15:56 the same day, so no label spans a block edge.
  - Inner validation (early stopping, isotonic calibration, theta) uses the training block next
    to the test block in time (the following block for 2020H1), embargoed the same way **[my
    call]**.
  - Every choice is made here: features, models, labels, theta, and every autoresearch
    iteration. About 680 days are scored, and the 6 blocks feed CSCV/PBO.
- **Confirmation:** 2023-01-01 -> 2025-08-06. At most 3 frozen finalists are pre-registered in
  this doc, then run once each, **strictly forward**: test year Y in {2023, 2024, 2025} is
  predicted by a model trained on everything before Y. **The bar is judged here.** Caveat: these
  years were already seen by Stages 1-2c for other questions (the pool and its drift), so they are
  only partly clean.
- **Holdout:** >= 2025-08-07 stays sealed. It is run once, for the one winner, and only if
  confirmation passes.
- **Trial ledger:** every dev evaluation is appended to `trials.jsonl` and feeds Deflated
  Sharpe's N and PBO.
- **Autoresearch loop (dev only):**
  - Scope: the Stage-2m feature and config modules.
  - **Metric (user decision 2026-10-28): top-tenth net lift.** The mean r_net of the top 10%
    of decision points by EV minus the mean r_net over all decision points. Pooled over the
    3 dev folds; higher is better.
  - **Guards on every kept change:** dev Delta log-loss (label A) does not worsen, and the
    full test suite passes, leakage tests included.
  - **Null calibration at the loop's end:** a sigma-correlated score can show lift with no
    signal, because SL gap-through grows with sigma. So the final kept configuration is also
    run on (a) labels shuffled within each day and (b) a random-walk price null with matched
    per-stock minute volatility. Its lift is reported net of both.
  - Guard: the full test suite, including the leakage tests.
  - Each iteration runs a 3-fold version of the purged CV (test = 2020, 2021, 2022, each a pair
    of blocks) with M1 on a cached panel.
  - Deep models are scheduled runs, not loop iterations.
- **Stop rule:** if, after E2 and E4, the best dev top-decile lift has a CI that includes 0
  **and** the best dev policy's daily-return CI upper bound is below 0, the long-only track stops
  and is reported. No open-ended search.

## 10. Experiment sequence (sequential elimination)

Order agreed with the user 2026-10-28. Days run LightGBM work (E0-E3, E5-E6); nights run E4's deep models. Each step gets its own pre-registration entry (exact label, output, metric, predicted outcome) committed before it runs.

| Exp | Question | Varies (one factor) | Fixed | Expected outcome |
|---|---|---|---|---|
| E-1 | Universe repair (§1.1) | admissible rows, no bar-count filter; re-score Stage 1s+pm (correction, not a trial) | §13 model | pool = new top 10 |
| E0 | What are we predicting, and how hard is it? | barrier set (6) | pool P1, no model | Label card: class base rates by year x price x time of day; breakeven precision p*; oracle ceiling (perfect TP-first, with slots); C1/C2 distributions. Expect the base TP rate to sit below p* |
| E1 | Floors | arm: C1, C2, C3 x 4, M0, M1-on-F0 | default label, P1 | M1-on-F0 ~ C1 (as in Round 1) |
| E2 | Which training population? (decided first, since every later step inherits it) | P1 / P2 / P3 | M1 on F0+F1+F8, default label | P3 + score helps, more for deep models |
| E3 | Which context matters? The autoresearch loop runs here | feature family, add-one then leave-one-out (F1..F7 on F0+F8) | M1, E2's population, default label | F1, F3 and F4 lift; F5/F6 small; F7 helps on offering days |
| E4 | Do sequence models add anything? (nights; M2-M4 start once E2 is decided, M5 after E3) | M2 / M3 / M4 / M5 vs M1 | E2's population; M5 uses E3's features | M5 >= M1 by a small margin; pure-sequence < M1 |
| stop check | after E3 + E4 | §9 stop rule | | |
| E5 | Label and exit design | H x (k_u, k_d); A-only (lambda = 0) vs A+B (lambda in {0.5, 1}) vs B-only (+ a separate exit model) vs y_H; fixed vs model-driven exit | best model | A+B >= A-only through a more stable encoder; longer H helps net of costs |
| E6 | Policy | theta; slot count; price floor ($2/$3); first-15-min exclusion; cooldown | best model | the price floor matters most |
| E7 | Confirmation | <= 3 pre-registered finalists | frozen | the bar |
| E8 | Holdout | 1 | frozen | once |

## 11. Deliverables per experiment

- A pre-registration entry in this doc, committed before the run.
- Code + tests (including leakage tests); the full suite passes before each commit.
- An MLflow run (params, metrics, artifacts; `MLFLOW_ALLOW_FILE_STORE=true`).
- An HTML tear sheet, a dashboard card, a ledger line, and a memory update.
- Tear-sheet figures:
  - label base-rate heatmaps (time of day x price)
  - reliability curves
  - EV-decile lift with CIs
  - net equity vs controls, with a drawdown panel
  - a monthly return heatmap
  - per-year bars with CIs
  - the cost waterfall
  - net trade-return and holding-time distributions
  - the entry time-of-day profile
  - SHAP by family
  - deep-model learning curves
  - a trial-count / Deflated-Sharpe panel

## 12. Compute and data budget (estimates, to be measured in each smoke run)

- Machine: 12 cores, 16.8 GB RAM (about 6 GB free), torch 2.9 **CPU only**, no GPU.
- **Decision panel, P1:** about 13.6k stock-days x 360 minutes = 4.9M rows. A 1-in-5 training
  subsample (~1M rows x ~200 float32 features) is about 0.8 GB. M1 takes about 2-4 min per fold,
  so a 6-block dev pass is about 15-25 min.
- **P3:** about 7x larger, stored as memmap / float16.
- **Fixed training budget (builder Q6, my call):**
  - Every LightGBM fit uses at most 2M training rows, whatever the population. P1 uses about
    1-in-2 minutes; P3 a sparser sample of more stock-days. E2 therefore compares populations at
    an equal row budget.
  - Deep models draw a fresh 1M windows per epoch from the population, so P3 costs about the
    same per epoch as P1.
  - The loop's time target is a soft ~10 minutes per iteration.
- **Sequence data:** per stock-day arrays of about 400 bars x 8 channels. P1 is about 200 MB;
  P3 is about 1.4 GB (memmap). Windows are sliced on the fly.
- **Deep training:** at an estimated 1.5-3k samples/s on CPU, one epoch of ~1M samples takes about
  6-11 min. About 10 epochs x 6 blocks x 3 seeds is 6-11 h per architecture, so E4 is about two
  nights. **CPU overnight runs (user decision, 2026-10-27).**

## 13. Decisions so far (user, 2026-10-27)

| Question | Decision |
|---|---|
| ETF downloads | Yes: SPY, QQQ, IWM, the 11 SPDR sectors and XBI, minute (04:00-16:05) and daily, 2016-01-04 -> 2025-08-06 (`stage2m/etf_fetch.py`). Holdout dates are refused by the code |
| Short side | Out of scope: long only |
| Autoresearch metric | Deferred until the whole experiment is designed |
| Compute | CPU, overnight runs |
| Label (2026-10-28) | Multi-task: A (triple barrier, which drives the EV and the trade) + B (Stage 1's share-beats-costs, conditioned on the entry minute) |

| Pass bar (2026-10-28) | As proposed in §8: CI lo > 0; beats C1-C4 (Holm); Deflated Sharpe >= 0.95; SPY/QQQ and the 2x-cost run reported, not gated |
| Experiment order (2026-10-28) | §10: E0 label card -> E1 floors -> E2 population -> E3 features (+ autoresearch) -> E4 sequence models at night -> stop check -> E5 labels and exits -> E6 policy -> E7 confirmation -> E8 holdout |
| Dev splits (2026-10-28) | Option A: purged CV over 6 half-year blocks of 2020-2022 (2017-2019 always in training), 5-day embargo; confirmation strictly walk-forward 2023, 2024, 2025 |
| Loop metric (2026-10-28) | Top-tenth net lift on the dev folds; guards: Delta log-loss does not worsen + tests and leakage tests pass; shuffled and random-walk null calibration at the end |

Design complete: the literature review and the builder check are folded in. Awaiting the
user's approval of the whole document.

## 14. Calls made by Claude (review these)

| # | Call | Where |
|---|---|---|
| 1 | Decision minutes 09:31-15:30; training subsamples 1-in-5 minutes (or the 2M-row budget), evaluation every minute | §2 |
| 2 | Account $25k, 10 x $2,500, one position per symbol, 5-min cooldown, 10%-of-bar-volume cap, skip orders under $500 | §2 |
| 3 | TP fills only on a 1-tick trade-through; the stop fills at min(SL, O_k)(1-h); sensitivity at the next-open rule | §2 |
| 4 | Point-in-time cost for features, barriers and EV; realized cost charged in the simulation; 2x half-spread for 09:31-09:44 | §3 |
| 5 | sigma_t = std of the last 30 one-minute log returns, floored at 0.5 x atr20/sqrt(390); default barriers H = 30, (k_u, k_d) = (2, 1), u >= 1.5 x c_rt | §4 |
| 6 | Label B threshold 0 net (costs inside r_j); multi-task weight lambda = 0.5 | §4 |
| 7 | Isotonic calibration on the inner validation block; theta = 0 until E6 | §5, §9 |
| 8 | C4 shuffled-label control matched on the real model's daily trade count | §8 |
| 9 | Deflated Sharpe N = program trials + the parent's 21; every trial runs the account simulation | §8 |
| 10 | Fixed 2M-row LightGBM budget; 1M windows per epoch for deep models; ~10-min loop iterations | §12 |
| 11 | Foundation models dropped; the Transformer runs after the TCN/GRU | §7 |
| 12 | E-1 universe repair: admissible rows, no bar-count filter, Stage 1s+pm re-scored as a correction (not a trial) | §1.1 |
| 13 | Inner validation = the adjacent training block, embargoed | §9 |

## 15. Pre-registrations, tranche 1 (committed 2026-10-28, before any run)

Implementation plan: `QuantCode-poolpoc/docs/superpowers/plans/2026-10-28-stage2m-tranche1.md`.
**E-1, E0 and E1 read dev data only (<= 2022-12-31)**, with one exception: E-1 re-scores the
parent §13 model over its own 2017 -> 2025-08-06 span, as a correction. The pass bar (§8)
applies at E7 only. Everything below is descriptive or a floor.

### E-1 Universe repair (a correction, not a trial)
| Item | Value |
|---|---|
| Universe | Wide-band panel rows ∩ R16-admissible minute-cache keys; no bar-count filter. Rows with 0-1 session bars: share_cost = 0, flagged `no_bars` |
| Model | §13 unchanged (Stage 1s features + 20 pre-market features, LightGBM, yearly walk-forward 2017 -> 2025-08-06), with base, room_model, naive, random and null arms |
| Output | §13 verdict and contrasts; pool days per year; `pool.parquet` (rank, in_p1 = top 10, in_p2 = top 30, all scored rows = P3) |
| Predicted | The §13 verdict is unchanged (PASS). Pool days per year are close to the admissible re-run's (4/11/23 in 2017-2019, ~200+ per year after) |

### E0 Label card (no model, descriptive)
| Item | Value |
|---|---|
| Population | P1 (top 10), dev days (<= 2022) |
| Barrier sets | H in {15, 30, 60, to-close} x (k_u, k_d) in {(2, 1), (1, 1)}: 8 sets (§4) |
| Tables | Class shares (TP/SL/TIME/NO_FILL) by year x price bucket (<2, 2-3, 3-5, 5-10, 10+) x hour; mean r_net per class; breakeven p* = \|rbar_SL\| / (rbar_TP + \|rbar_SL\|); label B (b_share) by hour x price |
| Arms through the §2 account | Oracle (every TP-first row, priority r_net: the ceiling); C1 (one random filled minute per pool name-day); C2 (09:31 open -> 15:56 open) |
| Predicted | For every set, the TP-first base rate is below p*. C1 and C2 have mean daily returns < 0. The oracle is large and positive (the ceiling is not the problem) |

### E1 Floors (dev, purged 6-block CV, default barrier H = 30, (2, 1))
| Item | Value |
|---|---|
| Arms | M0 (logistic) and M1 (LightGBM), both on F0 + F8, entering on EV > 0 with priority EV; C1 random timing matched to M1's daily trade count; C2; C3 = ORB-15, ORB-15 with RVOL > 1, VWAP reclaim, pre-market-high break, HOD break; C4 = M1 on labels shuffled within day, top-k matched to M1's count |
| Model metrics | Delta log-loss vs prior, Brier, AUC (TP vs rest), ECE10, daily rank IC (EV vs r_net), top-decile lift with a day-clustered CI, top-decile TP rate vs p* |
| Trading metrics | Mean daily net return with CI; paired differences vs C1-C4 (Holm); Sharpe; max drawdown; trades/day. Every arm is logged as a trial |
| Predicted | M1-F0's top-decile lift CI includes 0 (as in Round 1). No arm has a dev daily-return CI above 0. C3's rules lose net of costs on $1-$10 names (the literature's ORB evidence excludes penny stocks) |

## 16. Results, tranche 1 (2026-10-28; poolpoc 7220076..916b04d)

- **E-1 (correction): PASS.** On admissible rows, the top 10 have 22.7% of random trades beating
  costs vs 16.4% for the universe; vs naive +1.5 (CI +1.2..+1.8), vs room_model +0.6
  (+0.3..+0.8), vs base +1.1 (+0.8..+1.4). Null gate valid. 82,355 scored rows, 1,298 days.
- **E0: 3 of 3 predictions held.** H30 (2,1): 44.6% of decision minutes cannot fill; filled rows
  are TP 2.6% / SL 21.7% / TIME 75.8%; r_TP +6.2%, rbar_SL -4.7%, p* 42.9%. Every set has its TP
  rate below p*. C1 -1.05%/day, C2 -1.97%/day, SPY open-to-close +0.04%/day, oracle +2.8%/day.
- **E1 (run 916b04dcbb4e, 685 test days):** prediction 1 FAILED (M1's top-decile lift +0.76 pp,
  CI +0.67..+0.84; AUC 0.641), but the shuffled-label null also shows +0.33 pp (CI +0.24..+0.42,
  rank IC 0.15). Predictions 2 and 3 HELD: every arm loses. M1 (EV > 0) -0.57%/day is worse than
  C1 -0.40% and C4 -0.36% (Holm p 0.004); M1 beats C2 and all five C3 rules.
- **Finding (feeds E5/E6):** §5's EV values a TP at the row's own u but an SL at the bucket mean,
  so EV > 0 selects high-sigma minutes. M1's entries had mean u 21% (all rows 7.6%) and their
  SLs cost -9.2% (assumed -4.7%). The E6 policy factor gains a level: a row-specific SL term.

## 17. Pre-registrations, tranche 2 (committed 2026-10-28, before any run)

Implementation plan: `QuantCode-poolpoc/docs/superpowers/plans/2026-10-28-stage2m-tranche2.md`.
Dev data only (<= 2022-12-31). Everything below is a dev comparison; the pass bar applies at E7.

### F1 intraday levels and structure (exact; decision minute t, bars <= t, pre-open < 09:30)
| Feature | Definition |
|---|---|
| hod_pct, lod_pct | c_t / HOD_t - 1, c_t / LOD_t - 1 (HOD/LOD over bars 0..t) |
| hod_sig, lod_sig | log(c_t / HOD_t) / sigma_t, log(c_t / LOD_t) / sigma_t (sigma_t as §4) |
| mins_since_hod, mins_since_lod | t minus the latest bar that set the HOD / LOD |
| n_new_hod | bars 1..t whose high exceeds every earlier high of the day |
| or5_dist, or15_dist, or30_dist | c_t / ORH_k - 1, ORH_k = max high of bars 0..k-1; NaN while t < k-1 |
| or5_state, or15_state, or30_state | +1 above ORH_k, -1 below ORL_k, 0 inside; NaN while t < k-1 |
| vwap_dist, vwap_z | c_t / VWAP_t - 1; (c_t - VWAP_t) / SD_t, SD_t = the volume-weighted std of the bar price about VWAP_t |
| day_pos | (c_t - LOD_t) / (HOD_t - LOD_t) |
| ret_open, ret_prev | c_t / (first open) - 1; c_t x f / prior_close_unadj - 1 (red-to-green) |
| pmh_dist, pml_dist | c_t / pre-open high - 1, c_t / pre-open low - 1 (NaN without pre-open bars) |
| swing_hi_dist, swing_lo_dist | H / c_t - 1 for the nearest confirmed 5-bar fractal high above c_t; c_t / L - 1 for the nearest confirmed fractal low below (fractal j confirmed at t >= j + 2) |
| r50_up, r50_dn, r100_up, r100_dn | distance from px = c_t x f (unadjusted) to the next $0.50 / $1 grid level above and below, over px |

### E2 Which training population?
| Item | Value |
|---|---|
| Populations | P1 = top 10, P2 = top 30, P3 = every scored admissible row, by E-1's score (pool.parquet), dev days |
| Fixed | M1 (LightGBM, Stage 1 PARAMS) on F0 + F1 + F8; default label H30 (2,1); E1's purged 6 half-year folds; §5's EV, theta = 0 |
| Training rows | Equal budget: per fold, B = 400,000 filled rows drawn uniformly (seed 7) from the population's (name-day, minute) rows on the training dates; P2 and P3 store a seeded 1-in-3 and 1-in-8 sample of decision minutes per name-day, P1 all |
| Calibration | Isotonic on P1's inner-validation rows (t % 5 == 1) for every population |
| Evaluation | Every P1 decision minute of the test blocks (as E1) |
| Arms | M1-P1, M1-P2, M1-P3 traded on EV > 0 through the account (3 trials); each population's shuffled-label null (model metrics only) |
| Primary metric | Null-adjusted top-tenth net lift on P1 test rows: an arm's daily top-decile lift minus its own null's; populations compared by the paired daily difference vs P1 (day-clustered CI) |
| Guard | Delta log-loss vs the fold's P1 training prior |
| Decision | Carry forward the population with the best null-adjusted lift if its paired difference vs P1 has CI lo > 0 and its Delta log-loss is >= P1's; otherwise P1 |
| Predicted | (§10) P3 helps: M1-P3's null-adjusted lift beats M1-P1's (CI lo > 0), P2 in between. Every traded arm still loses (daily CI hi < 0): the EV asymmetry is unchanged until E6 |

### Amendments after the tranche-1 review (2026-10-28, committed before any E2 run)

The whole-branch review of tranche 1 (poolpoc `.superpowers/sdd/2026-10-28-stage2m-tranche1/final-review.md`)
found two defects that change §16's numbers and two spec flaws. Fixed at poolpoc 5d252af:

- **Fill floor (§2).** A fill that bar t+1's volume cap leaves under $500 is now NO_FILL, as §2
  says. It was kept before: 14.1% of filled rows, mean r_net -4.8%.
- **Matched controls (§8).** C1 and C4 now admit rows while the account can still hold them,
  up to M1's *realized* trades that day. C1 admits in random order, C4 by the null's EV.
  Before, they were matched on candidates, so the account kept only 91% (C1) and 40% (C4) of
  M1's count. Per-dollar and paired per-trade returns are now reported beside the daily means.
  The best C3 rule is now chosen per dollar.
- **§16 is pre-fix.** E0 and E1 are re-run on the fixed labels, and §16b supersedes them. The
  10 E1 trial lines from 916b04dcbb4e stay in `trials.jsonl`: they count as trials and carry a
  supersede line.

Two amendments, decided before any E2 number exists:

1. **Lift references (amends §8, §9 and E2).**
   - A cost-only score, -c_rt_pit, already gives most of E1's top-decile lift. On the pre-fix
     labels it gave +0.70 pp against M1's +0.76.
   - From E2 on, every reported lift therefore carries three things: the arm's own
     shuffled-label null lift, the cost-only score's lift, and the split of the lift into gross
     (r_net + c_rt_pit) and cost (-c_rt_pit) over the same top-decile rows. Both references are
     paired by day.
   - The user's loop metric (§9, top-tenth net lift, within-day) stays. The configuration the
     loop keeps must, at the loop's end, also beat the cost-only score's lift (paired CI lo > 0).
     Otherwise the loop has found a cost sort, and that is how it is reported.
   - E2's population comparison is unaffected: the cost-only reference is the same rows for
     every population, so it cancels in the paired difference against P1.
2. **Row-specific SL term (amends §5).**
   - **The flaw.** EV valued a TP at the row's own u but an SL at the training bucket mean.
     That favoured high-sigma minutes: M1's SLs cost -9.2% against the -4.7% EV assumed.
   - **New SL term, from E2 on:**
     r_SL = (1 - d)(1 - h_pit) - 1 - comm_pit - 2 x imp_pit + slip[bucket].
   - **Its inputs:**
     - d = k_d sigma_t sqrt(H), as in §4.
     - h_pit = the point-in-time half-spread that c_rt_pit uses: (c_rt_pit - comm_pit -
       2 x imp_pit) / 2.
     - slip[bucket] = the mean of (r_net - nominal r_SL) over the training SL rows of each price
       bucket, which is the gap-through cost.
   - **TP term:** unchanged.
   - **The bucket-mean SL term** becomes E6's comparison level.
   - **E2 "Fixed"** now reads "§5's EV with the row-specific SL term, theta = 0".
   - **E2's prediction**, amended in the same commit: every traded arm still loses (daily CI
     hi < 0); the row-specific SL term lowers EV on high-sigma minutes but does not by itself
     create an edge. The population prediction is unchanged.

## 16b. Results, tranche 1 after the review fixes (2026-10-28; poolpoc 5d252af / b08b7a0; supersedes §16's E0 and E1)

- **E0: 3 of 3 predictions held.**
  - H30 (2,1): 52.4% of decision minutes cannot fill (was 44.6%).
  - Filled rows: TP 2.75% / SL 22.4% / TIME 74.9%.
  - r_TP +6.56%, rbar_SL -4.43%, p* 40.3%; a random entry nets -1.64%.
  - C1 -1.22%/day, C2 -1.93%/day, SPY open-to-close +0.04%/day, oracle +2.9%/day.
- **E1 (run b08b7a063052, 685 test days).**
  - **Prediction 1 FAILED.** M1's top-decile lift is +0.29 pp (CI +0.20..+0.36); AUC 0.633.
  - **Against the references:**
    - The lift beats the shuffled-label null (whose lift is -0.21) by +0.50 pp.
    - It does **not** beat a cost-only sort: +0.02 pp, CI -0.08..+0.12.
    - Split: gross -0.25 pp + cost +0.54 pp. The minutes M1 likes are cheaper, not better.
  - **Predictions 2 and 3 HELD.** Every arm loses.
    - M1 -0.62%/day, M0 -0.49, C1 -0.56, C4 -1.05, C2 -1.99. The rules lose -1.24..-2.88%/day.
    - M1, C1 and C4 each made 3,070 trades.
  - **Per trade, M1 is no better than random:** -0.19 pp vs C1 (CI -0.50..+0.13). It beats the
    null by +0.93 pp.
  - **Reading.** On F0 + F8, the model's ranking adds nothing over the cost estimate. The
    tranche-2 work (F1 levels, populations, a row-specific SL term) has to produce gross skill:
    a lift above the cost-only score.

## 17b. Results, tranche 2: E2 (2026-10-28; poolpoc e43b7286efea, clean; MLflow e24811c488104291878d2a05d0b86b92)

M1 on F0 + F1 + F8, with 400,000 training rows per fold from each population. It was tested on every P1 minute of 685 days, under 6 purged folds. The determinism refit gave identical predictions. Runtime 1,904 s, peak RAM 3.8 GB.

| Trained on | Lift over its null (pp) | Raw lift: net = gross + cost (pp) | vs cost-only (pp) | AUC (TP) | dlogloss | Arm, %/day (CI hi) | Per trade |
|---|---|---|---|---|---|---|---|
| P1 | +0.225 (+0.135..+0.311) | +0.723 = +0.024 (-0.051..+0.095) + 0.699 | +0.461 (+0.368..+0.548) | 0.710 | +0.0410 | -0.201 (-0.125) | -1.06% |
| P2 | +0.300 (+0.207..+0.392) | +0.694 = -0.019 + 0.712 | +0.432 | 0.724 | +0.0442 | -0.267 (-0.180) | -1.30% |
| P3 | +0.419 (+0.331..+0.504) | +0.667 = -0.027 (-0.096..+0.037) + 0.694 | +0.406 (+0.319..+0.488) | 0.720 | +0.0447 | -0.287 (-0.193) | -1.24% |
| cost-only sort | | +0.262 = -0.592 + 0.854 | | | | | |

The predictions: P3 beats P1 (+0.194 pp, CI +0.125..+0.262) HELD; P2 falls between them HELD; every traded arm loses HELD. SPY open-to-close was +0.027%/day, and every arm is below it.
**Decision (pre-registered rule): carry P3.**

Reading:
1. **The ranking is still a cost sort.** In every population the gross part of the top-decile lift has a CI that straddles 0, and all of the lift is in the cost part. The models now beat the cost-only sort by about 0.4 pp, unlike E1. But they win by not paying the cost-only sort's adverse gross (-0.59 pp), not by finding minutes that rise. Cheap minutes are worse minutes, and M1 has learned to take the cheap ones that are not worse.
2. **P3's win is a null effect.** On raw lift P1 leads (+0.72 vs +0.67 pp). P1 also leads against the cost-only sort (+0.46 vs +0.41) and in the account (-0.20 vs -0.29 %/day). P3 wins only after null adjustment, because its shuffled-label null transfers worse to P1's minutes. A null trained on another population is not a neutral baseline for P1 test rows, so the null-adjusted metric favours the population whose null is most mismatched. The pre-registered rule is followed, and E3 trains on P3. E3's primary metric is a paired difference inside one population, so this choice moves its comparisons little.
3. **Every arm loses** about 1.1-1.3% per trade at about 2 trades per day.

## 17c. E2 amendment 3, after the tranche-2 review (2026-10-28; written after E2's first full run, committed before its re-run)

The whole-branch review of tranche 2 (poolpoc `.superpowers/sdd/2026-10-28-stage2m-tranche2/final-review.md`)
found that E2's decision metric is biased by construction, and that the cost-only score no longer
bounds what a model without skill gets. This amendment was written after §17b's numbers existed,
so it is a correction, not a pre-registration. Both calls are reported. Fixed at poolpoc e143f19.

1. **The null-adjusted metric favours the larger populations (review C1).**
   - A within-day shuffle gives each row a label drawn from its own day. The mean share of a
     filled row's day that is its own name-day is 15.1% in P1, 5.6% in P2 and 2.3% in P3. So the
     fewer names a day has, the more of the name-day's own signal (sigma, d, cost and the
     outcome) the null keeps.
   - The nulls' lift falls in that order: +0.497, +0.394 and +0.249 pp. Subtracting each
     population's own null removes more real signal from P1 than from P3. The metric would
     order P3 > P2 > P1 even with equal models, which was the predicted outcome.
   - §17b's "P3 beats P1 by +0.194 pp" is -0.055 pp from the models (P3 ranks worse) plus
     +0.249 pp from the nulls.
2. **The decision now compares raw lift.**
   - Every population is scored on the same P1 test rows. Subtracting any reference common to
     all of them therefore leaves the order unchanged.
   - The rule is otherwise the same: carry the population with the best raw lift if its paired
     raw difference vs P1 has CI lo > 0 and its Delta log-loss is >= P1's; otherwise P1.
   - The pre-registered call (the same rule on null-adjusted lift) is reported beside it.
   - Prediction 1 is still judged as pre-registered. It is recorded as "held on the
     pre-registered metric; produced by the null term", not as "P3 helps".
   - The null-adjusted difference vs P1 is reported with its three parts, each paired by day
     with a CI: raw, gross (r_net + c_rt_pit) and null.
3. **A mechanical-EV reference (review I1).**
   - Under the row-specific SL term (amendment 2), an EV with no forecast already sorts minutes
     by d (that is, by sigma) and by cost. P1's null beats the cost-only score by about 0.24 pp.
     So -c_rt_pit no longer bounds what a model without skill gets.
   - The **mechanical EV** is §5's EV with the row-specific SL term, with the class
     probabilities fixed at the fold's P1 training prior and mu_TIME fixed at the training TIME
     mean. It ranks a row by its known terms alone: target, stop, costs and the bucket's slip.
   - Every model and every null now reports three things, all paired by day: its gross/cost
     split, its lift over the cost-only score, and its lift over the mechanical EV. The
     mechanical EV reports its own split.
4. **The re-run.**
   - E2 is re-run at the fix commit. The models are unchanged, and the determinism refit
     confirms they are identical.
   - The re-run saves its prediction frame, so later readings need no further re-run.
   - The three trial lines of e43b7286efea get a supersede line, and they still count as trials.

On §17b's numbers the amended rule carries **P1**, the rule's default: raw lift is P1 +0.723 >
P2 +0.694 > P3 +0.667 pp. The re-run confirms or overturns this.

## 17d. Results, E2 re-run under amendment 3 (2026-10-28; poolpoc e143f195d90a, clean; MLflow 0325ca6550214b90a9fe74b110a9fcf7)

This is the same models on the same rows. The determinism refit gave identical predictions, and
every §17b number was reproduced to the last digit. Runtime 1,919 s, peak RAM 4.0 GB. The prediction
frame is saved beside the report (predictions.parquet).

| Scored by | Top-decile lift (pp) | Gross part | Cost part | vs cost-only | vs mechanical EV |
|---|---|---|---|---|---|
| M1-P1 | +0.723 (+0.645..+0.797) | +0.024 | +0.699 | +0.461 | **+0.081 (-0.011..+0.162)** |
| M1-P2 | +0.694 | -0.019 | +0.712 | +0.432 | +0.052 (-0.040..+0.137) |
| M1-P3 | +0.667 | -0.027 | +0.694 | +0.406 | +0.026 (-0.063..+0.106) |
| null-P1 | +0.497 | -0.109 | +0.606 | +0.236 | -0.144 (-0.179..-0.111) |
| null-P2 | +0.394 | -0.047 | +0.441 | +0.132 | -0.248 |
| null-P3 | +0.249 | -0.016 | +0.265 | -0.013 | -0.393 |
| mechanical EV | +0.642 (+0.598..+0.688) | -0.049 (-0.086..-0.007) | +0.690 | +0.380 | |
| cost-only sort | +0.262 | -0.592 | +0.854 | | |

- **The difference vs P1, by part.**
  - P3: raw -0.055 pp (CI -0.111..+0.003), gross -0.051, null -0.249 (CI -0.289..-0.208).
  - P2: raw -0.029, gross -0.043, null -0.103.
  - P3's pre-registered win is entirely the null term.
- **Decision (§17c, raw lift): carry P1.** P1 has the best raw lift. The pre-registered rule
  carries P3. Prediction 1 is recorded as held on the pre-registered metric, produced by the null
  term.

Reading:
1. **No model beats the EV with no forecast.**
   - The mechanical EV fixes p at the training prior and ranks minutes only by their own target,
     stop and cost. It already reaches +0.642 pp, almost all of it cost.
   - M1-P1 adds +0.081 pp (CI -0.011..+0.162). P2 and P3 add less.
   - Every shuffled-label null falls below the mechanical EV. The nulls add noise to the
     row-term sort. They are not a no-skill floor.
2. **What M1-P1 adds over the mechanical EV is gross, not cost, but it is not proven.**
   - The gross part of its edge is +0.073 pp (CI -0.012..+0.150). The cost part is +0.008.
   - So the model's departure from the row-term sort points at minutes that do better before
     costs, which is the shape a forecast would have. At this sample size it is not
     distinguishable from zero.
   - The model's EV rank correlates 0.58 within day with the mechanical EV, and only 28% of its
     top decile is in the mechanical EV's top decile.
3. **Every traded arm loses**, as in §17b.

## 18. Pre-registrations, tranche 3 (committed 2026-10-28, before any F2-F7 code or E3 run)

Dev data only (<= 2022-12-31). Decision minute t means after bar t closes (bar t spans
09:30+t to 09:31+t). Every feature at t reads bars <= t, pre-open bars < 09:30, daily rows
<= D-1, and filings accepted at or before 09:31+t on D. Price ratios use one basis: the minute
cache's adjusted basis, which is the daily cache's adjusted columns. A ratio of two prices on the
same basis does not depend on a later split. A feature is NaN when its inputs are missing. Each
family gets a perturbation test (changing bars after t moves nothing at t) and a date test (no
daily row >= D and no filing after the decision time).

### F2 multi-day levels (daily bars <= D-1)
| Feature | Definition |
|---|---|
| pdh_dist, pdl_dist, pdc_dist | c_t / H_{D-1} - 1, c_t / L_{D-1} - 1, c_t / C_{D-1} - 1 |
| hi5_dist, lo5_dist, hi20_dist, lo20_dist, hi60_dist, lo60_dist | c_t / max(H over the last k sessions <= D-1) - 1, c_t / min(L ...) - 1; NaN with fewer than k sessions |
| piv_p, piv_r1, piv_s1, piv_r2, piv_s2 | c_t / level - 1, with D-1's floor pivots P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H, R2 = P + (H - L), S2 = P - (H - L) |
| hi252_dist | c_t / max(H over up to 252 sessions <= D-1) - 1; NaN with fewer than 60 |
| run_days | the signed run of consecutive up (+) or down (-) closes ending at D-1 |
| gap_atr | (O_D / C_{D-1} - 1) / atr20, where O_D is the 09:30 bar's open |

### F3 momentum and volume (bars <= t)
| Feature | Definition |
|---|---|
| r5, r30, r60 | log(c_t / c_{t-k}), with c_{t-k} the last present close at or before t-k; NaN if none (F0 already has 1, 3 and 15) |
| sv5, sv15, sv30 | sum over the last k bars of v x sign(c - o), divided by the sum of v; NaN without volume |
| crvol | cumulative volume to t / (adv20 in shares x the intraday volume curve to t). The curve is E1's (fitted on 2017-2019) |
| vacc | mean v of the last 5 bars / mean v of bars t-29..t-5; NaN if the denominator is 0 |
| sig_ratio | std of the 1-minute log returns over the last 5 present bars / the same over the last 30 |

Trade count is dropped: the minute arrays the panel loads have no trade-count field.

### F4 pre-market (the pool's §13 columns, constant over the day)
pm_gap, pm_late_ret, pm_range, pm_close_pos, pm_dist_vwap, pm_log_dv, pm_dv_to_adv, pm_late_share,
pm_bars, pm_fade.

### F5 market context (ETF minute bars <= t, daily rows <= D-1, the day's pool at t)
| Feature | Definition |
|---|---|
| spy_ret_open, qqq_ret_open, iwm_ret_open | the ETF's last close at or before t / its 09:30 open - 1 |
| spy_r15, iwm_r15 | the ETF's 15-minute return to t |
| qqq_ret_d1, qqq_tr5, qqq_tr20 | QQQ C_{D-1}/C_{D-2} - 1, C_{D-1}/C_{D-6} - 1, C_{D-1}/C_{D-21} - 1 |
| vix_d1, vix_chg_d1 | VIX close on D-1; VIX C_{D-1}/C_{D-2} - 1 |
| br_above_open, br_med_ret_open, br_new_hod5, br_disp | over the day's P3 pool names with a bar at or before t: the share above their 09:30 open, the median return since the open, the share that set a new high of day in the last 5 bars, and the cross-sectional std of the return since the open |

### F6 sector (SIC sector -> sector ETF; ETF minute bars <= t)
| Feature | Definition |
|---|---|
| sec_ret_open | the sector ETF's return since its 09:30 open, to t |
| rel_sec_open, rel_sec_15 | the stock's return since the open minus the ETF's; the same over the last 15 minutes |
| sec_breadth | the share of the day's P3 names in the same sector that are above their open at t; NaN with fewer than 3 |

The sector -> ETF map:
- biotech_pharma -> XBI
- software_it, hardware_semis -> XLK
- financials -> XLF
- retail_consumer -> XLY
- transport_shipping, industrials -> XLI
- medical_devices, healthcare_services -> XLV
- mining_metals, chemicals_materials -> XLB
- energy -> XLE
- telecom_media -> XLC (IWM before XLC's first bar)
- real_estate -> XLRE
- other, shell_blank_check, unknown, unmapped -> IWM

### F7 events and flows (EDGAR accepted_et <= 09:31+t on D; FINRA <= D-1)
| Feature | Definition |
|---|---|
| ev_8k_today, ev_any_today | 8-K filings, and filings of any form except 3, 4, 5, 144 and SC 13*, accepted on D before the decision time |
| ev_8k_5d, ev_off_30d, ev_per_5d | 8-Ks over the 5 calendar days before D plus today before t; offerings (S-1, S-3, F-1, F-3, their /A, 424B*) over 30 days; 10-Q/10-K (and /A) over 5 days |
| ev_mins_since | minutes from the latest such filing (any form, as in ev_any_today) to the decision time, capped at 43,200; NaN if none in 30 days |
| short_d1, short_5d | FINRA short volume / total volume on D-1, and its mean over the last 5 sessions <= D-1; NaN before 2018-08 |

### E3 which context matters? (add-one, all, then leave-one-out)
| Item | Value |
|---|---|
| Population | E2's carried population; calibration and test on P1 exactly as E2 |
| Fixed | M1 (Stage 1 PARAMS); H30 (2,1); E1's purged 6 half-year folds; B = 400,000 filled training rows per fold (seed 7); EV with the row-specific SL term; theta = 0 |
| Configurations | base = F0 + F8; add-one: base + Fk for k = 1..7 (7); all = F0..F8; leave-one-out: all minus Fk (7; run after add-one) |
| Primary metric | per configuration, the paired daily difference of top-decile lift vs base (add-one) or vs all (leave-one-out), day-clustered CI, Holm over the 7 |
| References (§17 amendment 1) | every configuration's lift vs the cost-only score, its gross/cost split, and, for base and all, the shuffled-label null |
| Guard | Delta log-loss vs the P1 training prior is reported per configuration; a family that lowers it is flagged |
| Traded arms | base, all and the best add-one configuration on EV > 0 through the account (3 trials); the other configurations are model-level trials |
| Decision | E3's feature set is base plus every family whose add-one difference has Holm-adjusted CI lo > 0. If none qualifies, the set is base + F1 (the user's critique asked for levels) and is reported as unproven |
| Predicted | (§10) F1, F3 and F4 lift; F5/F6 small; F7 helps mainly on offering days. Against the cost-only score, no configuration's lift has CI lo > 0 (the gross part stays <= 0). Every traded arm still loses |

### Amendments after building F2-F7 (2026-10-28, committed before any E3 run)
1. **Filings that are not public at acceptance are never read.** EDGAR records CORRESP,
   UPLOAD, DRS (and DRS/A) and DRSLTR at the time they are submitted. The SEC releases staff
   correspondence weeks after its review closes, and a confidential draft becomes public only
   when the company files publicly. Counting them at accepted_et would read information no
   trader had, so they belong to no F7 group. In dev data they are about 5% of filings (47,209 of 972,311).
   "SCHEDULE 13*" (EDGAR's form name from late 2024) is the same as SC 13*.
2. **Family sizes follow the tables above:** F2 has 17 features, F3 has 9 (trade count dropped),
   F5 has 14, F6 has 4, F7 has 8. The plan's "19 / 10 / 7" were miscounts.
3. **Conventions where the tables are silent:**
   - "D-1" means the latest row dated before D for that symbol (or for QQQ, VIX and FINRA), not a
     calendar lookup. A FINRA row older than 31 days is never used.
   - Minute windows are clipped at the open. A missing minute adds no volume to sv or vacc. vacc
     is NaN for t < 5. sig_ratio needs 5 returns in both windows, as bars.sigma_t does.
   - Every 15-minute return (spy_r15, iwm_r15, rel_sec_15) is NaN for t < 15.
   - A stock with no 09:30 bar has no return since the open, so it is left out of every breadth
     feature that day (about 1% of name-days). sec_breadth is NaN when the sector has fewer than
     3 names in the day's pool.
   - br_new_hod5 is strict (a tie with the earlier high is not new) and bar 0 counts as a new
     high, so the feature is close to 1 for t <= 4 by construction.
   - A level that is not positive (a pivot S1 or S2 below zero after a wide-range day) has no
     distance.
   - XLC stands for telecom_media from 2018-06-19, its first session in the ETF cache; IWM is
     used before that.
4. **The FINRA short ratios are dropped from E3's model inputs (short_d1, short_5d); F7 enters
   E3 as its 6 EDGAR features.** The FINRA cache was fetched by today's tickers, and Alpaca's
   bars carry today's ticker for the whole history. A company that changed its ticker since
   therefore has no FINRA row under its old name. That is 17% of P1 name-days from 2018-09 (26%
   in March 2021), and every sampled gap was a renamed or reused ticker. The gap predicts the
   label: on name-days without a row the mean filled r_net is -1.95% vs -1.59% with one, a paired
   daily difference of +0.50 pp (se 0.06). A feature whose missingness encodes a later rename is
   future data. The columns stay in the panel for diagnostics.
5. **EDGAR and sector coverage has the same origin but no measurable label content.** Both are
   keyed by today's ticker-to-CIK map, so a company delisted since has no filings and no sector:
   7.8% of pool name-days (5.4% of P1), rising from 2% in 2022 to 20% in 2019. Its label
   distribution matches the covered rows (TP 2.76% vs 2.75%, SL 22.4% vs 22.4%, paired r_net
   difference +0.12 pp, se 0.07). F6 and F7 are kept, and E3 also reports their add-one difference
   on covered name-days only. A sensitivity that disagrees with the full result voids the family's
   claim.

### Amendment after the tranche-2 review (2026-10-28, committed before any E3 full run)
6. **E3's population, its no-skill reference and a restated prediction.**
   - **Population: P1.** That is E2's call under §17c's raw-lift rule.
     - The E3 smoke at poolpoc 682ef8b ran on P3 before this amendment. Smoke runs are not
       trials.
     - Its numbers were seen. On 2 folds at B = 100,000, neither +F1 nor +F3 cleared Holm.
       Base and all beat the cost-only score by +0.36 and +0.51 pp (CI lo > 0).
   - **The binding no-skill reference is the mechanical EV** (§17c item 3), fitted per fold on
     the P1 training rows.
     - Every configuration reports its lift over the mechanical EV, its gross/cost split and its
       lift over the cost-only score.
     - The base and all nulls report the same three.
   - **§17 amendment 1's loop condition now reads:**
     - The configuration the loop keeps must beat the mechanical EV's lift (paired CI lo > 0).
     - It counts as a forecast only if the gross part of its lift also has CI lo > 0.
     - Otherwise it is reported as a sort on cost and risk.
   - **The original prediction stays on the record and is judged as written:** against the
     cost-only score, no configuration's lift has CI lo > 0.
     - Under the row SL term it is expected to fail for a mechanical reason (review I1), and the
       smoke suggests it does.
   - **The prediction that now carries the hypothesis:**
     - Every configuration beats the mechanical EV (CI lo > 0), as E2's models beat their nulls.
     - No configuration's gross part has CI lo > 0.
     - Together these say the lift is a sort on cost and risk, not a forecast of minutes that
       rise.
   - **E3's design is otherwise unchanged.** Its primary metric is a paired difference inside one
     population. Its decision rule (Holm over the 7 add-one differences) does not change.

### Amendments after the tranche-3 review (2026-10-28, committed before any E3 full-run number exists)
The whole-branch review of tranche 3 (poolpoc `.superpowers/sdd/2026-10-28-stage2m-tranche3/final-review.md`)
found two gaps. The E3 add-one run was already under way at poolpoc 440b1e4, which predates
both rules. Both are applied after the run, from its saved prediction frame, with no refit.
7. **The void rule for F6 and F7 (makes amendment 5 operational).**
   - A family keyed by today's ticker map (+F6, +F7) that qualifies on all rows (mean > 0 and
     Holm p < 0.05) is re-scored on covered name-days only. Those are name-days whose symbol is in
     today's sector map.
   - It *disagrees* when its covered-only paired difference vs base has mean <= 0, or a 95% CI
     lo <= 0 (unadjusted: the covered rows are 95% of the full rows, so this checks robustness,
     not a second discovery).
   - A family that disagrees is dropped from the decision, and the reason is stated. A family
     that does not qualify on all rows has its covered-only number reported and nothing else.
   - The verdict prints the covered-only and offering-day numbers.
8. **A basis sensitivity (tranche-3 review I2).**
   - The minute and daily caches were fetched on different dates. On 0.2% of pool name-days (0.29%
     of P1, 25 symbols) the two disagree by an exact split factor, constant per symbol. That
     breaks F2's level ratios, F1's ret_prev, the traded price behind costs and price buckets, and
     the pool score's pm_gap.
   - Every E2 and E3 lift difference is also reported with those symbols' name-days left out of
     the *evaluation*. A symbol is broken when any of its dev pool name-days has |log(minute
     close / daily adjusted close)| > 0.2.
   - An add-one family whose qualification flips under the exclusion is marked basis-sensitive.
   - The exclusion is a read, not a filter for any strategy, because it selects on a future split.
   - The repair is to rescale the minute and pre-market bars to the daily basis per symbol, or
     to re-fetch them on one date. It re-selects the E-1 pool: 11 of the 21 broken P1 name-days
     have pm_gap at its 5.0 cap. So it re-runs every tranche, and is left to the user before E4
     and E6, where costs and price buckets matter.

## 18b. Results, tranche 3: E3 (2026-10-28; poolpoc 440b1e4eda76, clean; MLflow add-one 302cfe649b6945d4ad758f94faaaf9af, leave-one-out 508d19906ba74a329041f52ea6fe4b48)

The run trained on P1 (amendment 6) and scored every P1 test minute: 685 days and 2,466,000 rows.
- The add-one run took 2,510 s with a peak of 5.8 GB. The determinism refit of base in fold
  2020H1 was identical.
- The leave-one-out run took 3,457 s with a peak of 5.0 GB. Its all model reproduces the
  add-one run's numbers exactly, which is a cross-run determinism check. The in-run check
  refits base, and leave-one-out has no base.
- Both prediction frames are saved beside their reports.
- Amendments 7 and 8 are applied from the add-one frame with no refit (poolpoc e853e91,
  `rescore.py`, rescore.json beside the report).

| Configuration | Top-decile lift (pp) | Gross part | Cost part | vs cost-only | vs mechanical EV | AUC (TP) | Δ log-loss |
|---|---|---|---|---|---|---|---|
| base (F0+F8) | +0.685 (+0.633..+0.739) | -0.075 (-0.121..-0.029) | +0.760 | +0.424 | +0.043 (-0.011..+0.093) | 0.631 | +0.0093 |
| +F1 levels | +0.714 | +0.013 (-0.061..+0.083) | +0.701 | +0.453 | +0.073 (-0.017..+0.154) | 0.711 | +0.0411 |
| +F2 multi-day | +0.726 | +0.005 | +0.721 | +0.464 | +0.084 (+0.013..+0.154) | 0.661 | +0.0185 |
| +F3 momentum | +0.679 | -0.048 | +0.727 | +0.418 | +0.038 | 0.669 | +0.0190 |
| +F4 pre-market | +0.696 | -0.030 | +0.726 | +0.434 | +0.054 | 0.639 | +0.0129 |
| +F5 market | +0.755 (+0.706..+0.805) | -0.002 (-0.047..+0.047) | +0.757 | +0.494 | +0.114 (+0.065..+0.161) | 0.616 | +0.0040 |
| +F6 sector | +0.699 | -0.043 | +0.742 | +0.438 | +0.058 | 0.640 | +0.0104 |
| +F7 filings | +0.682 | -0.074 | +0.756 | +0.420 | +0.040 | 0.630 | +0.0098 |
| **all** | **+0.850 (+0.772..+0.923)** | **+0.171 (+0.097..+0.242)** | +0.678 | +0.588 | **+0.208 (+0.120..+0.293)** | 0.722 | +0.0435 |
| null-base | +0.625 | -0.016 | +0.641 | +0.363 | -0.017 (-0.046..+0.011) | 0.559 | -0.0010 |
| null-all | +0.487 | -0.070 | +0.557 | +0.226 | -0.155 (-0.188..-0.123) | 0.553 | -0.0029 |
| mechanical EV | +0.642 (+0.598..+0.688) | -0.049 (-0.086..-0.007) | +0.690 | +0.380 | | | |
| cost-only sort | +0.262 | -0.592 | +0.854 | | | | |

| Add-one vs base (pp) | All rows | Holm p | Covered only (am. 7) | Without broken symbols (am. 8) |
|---|---|---|---|---|
| +F1 | +0.029 (-0.046..+0.098) | 1.000 | | |
| +F2 | +0.040 (-0.021..+0.100) | 0.996 | | |
| +F3 | -0.006 | 1.000 | | |
| +F4 | +0.010 | 1.000 | | |
| **+F5** | **+0.070 (+0.026..+0.114)** | **0.007** | | **+0.064 (+0.019..+0.111), Holm 0.042** |
| +F6 | +0.014 | 1.000 | +0.011 (-0.052..+0.064) | |
| +F7 | -0.004 | 1.000 | +0.007 (-0.024..+0.036) | |

- **Decision: base + F5 (30 features).**
  - F5 (market context) is the only family with Holm p < 0.05.
  - The Δ log-loss guard flags it. Its Δ log-loss is 0.0040 against base's 0.0093, and its AUC
    is 0.616 against 0.631.
  - Amendment 7 voids nothing, because neither F6 nor F7 qualifies.
  - Under amendment 8, the 22 broken symbols are 2.19% of rows. Without them F5 still qualifies,
    and no family is basis-sensitive.
- **Predictions as pre-registered:**
  - "F1, F3 and F4 lift": **failed**. None qualifies. F1 raises AUC from 0.631 to 0.711, but its
    lift difference is +0.029 (Holm 1.0).
  - "F5/F6 small": **failed**. F5 is the only qualifier.
  - "F7 helps mainly on offering days": **held, weakly**. The point estimates are +0.031 on
    offering name-days and +0.013 on the rest, and the CIs overlap (minor M7).
  - "No configuration beats the cost-only score": **failed**, for the mechanical reason
    amendment 6 gave. Every configuration beats it with CI lo > 0.
  - "Every traded arm loses": **held**.
- **Predictions as amended (amendment 6):**
  - "Every configuration beats the mechanical EV": **failed**. Base, +F1, +F3, +F4, +F6 and +F7
    have CI lo <= 0.
  - "No configuration's gross part has CI lo > 0": **failed**. The all model's gross part is
    +0.171 (CI +0.097..+0.242).

| Traded arm (EV > 0) | Mean %/day (95% CI) | Trades | Net per trade | vs SPY open-to-close |
|---|---|---|---|---|
| base | -0.178 (-0.240..-0.117) | 888 | -1.37% | -0.205 |
| all | -0.156 (-0.236..-0.085) | 1,219 | -0.91% | -0.183 |
| +F5 | -0.104 (-0.161..-0.049) | 874 | -0.91% | -0.131 |
| SPY open-to-close | +0.027 (-0.054..+0.110) | | | |

| Leave-one-out vs all (pp) | All rows | Holm p | Without broken symbols | Gross part | vs mechanical EV | AUC (TP) |
|---|---|---|---|---|---|---|
| all -F1 | -0.038 (-0.082..+0.004) | 0.410 | -0.040, Holm 0.268 | +0.109 (+0.041..+0.168) | +0.170 (+0.089..+0.239) | 0.685 |
| all -F2 | -0.037 (-0.082..+0.005) | 0.410 | -0.037, Holm 0.279 | +0.133 | +0.171 | 0.716 |
| all -F3 | -0.019 (-0.058..+0.016) | 0.610 | -0.025, Holm 0.362 | +0.154 | +0.189 | 0.719 |
| all -F4 | -0.032 (-0.071..+0.005) | 0.410 | -0.042, Holm 0.196 | +0.124 | +0.175 | 0.718 |
| all -F5 | -0.042 (-0.081..-0.003) | 0.210 | -0.044, Holm 0.196 | +0.139 | +0.166 | 0.722 |
| all -F6 | -0.034 (-0.071..-0.001) | 0.264 | -0.036, Holm 0.196 | +0.140 | +0.174 | 0.718 |
| all -F7 | -0.009 (-0.049..+0.029) | 0.624 | -0.014, Holm 0.475 | +0.160 (+0.091..+0.225) | +0.199 (+0.117..+0.281) | 0.718 |

- **Leave-one-out.**
  - No single family is necessary. Every drop costs the all model between 0.009 and 0.042 pp,
    and none survives Holm, with or without the broken symbols.
  - Every reduced model keeps a gross part with CI lo > 0, and every one beats the mechanical
    EV.
  - Dropping F1 costs the most AUC (0.722 to 0.685) and the most gross (+0.171 to +0.109).
- **Predictions, leave-one-out run:**
  - "No configuration beats the cost-only score": failed.
  - "Every traded arm loses": held.
  - "Every configuration beats the mechanical EV": held, within this run.
  - "No gross skill": failed.
- **Pooled over both runs (minor M6):** "every configuration beats the mechanical EV" fails,
  because of add-one's base and others, and "no gross skill" fails.

Unregistered reads on the all model. These come from the saved add-one frame with 2,000
day-clustered draws. They were chosen after the add-one numbers were seen, so they are robustness
checks, not tests.

| All model's gross part | pp (95% CI) |
|---|---|
| every row | +0.171 (+0.097..+0.242) |
| covered name-days only | +0.174 (+0.100..+0.244) |
| without the 22 broken symbols | +0.166 (+0.092..+0.235) |
| decision minutes after 09:44 (t >= 14) | +0.193 (+0.117..+0.268) |
| 09:31-09:44 (t < 14; 3.6% of rows) | -0.213 (-0.420..-0.020) |
| fold 2020H1 / 2020H2 / 2021H1 | +0.401 (+0.214..+0.590) / +0.137 (-0.014..+0.294) / +0.276 (+0.112..+0.453) |
| fold 2021H2 / 2022H1 / 2022H2 | +0.150 (+0.031..+0.278) / +0.167 (+0.034..+0.326) / -0.080 (-0.336..+0.134) |

Unregistered diagnostic: which EV term carries the lift? It refits base, +F1, +F5 and all
exactly as E3 add-one did (every EV matches the saved frame to 0.0). Each is then scored three
ways:
- both terms, which is the model's EV;
- class probabilities only, with mu_TIME fixed at the fold's training TIME mean;
- the TIME-return regressor only, with p fixed at the fold's P1 training prior.

The script and its JSON are in data/.../440b1e4eda76/e3-diag/.

| Model | Scored with | Top-decile lift | Gross part | Cost part | vs mechanical EV |
|---|---|---|---|---|---|
| base | both | +0.685 | -0.075 (-0.121..-0.029) | +0.760 | +0.043 (-0.011..+0.093) |
| base | p only | +0.642 | +0.013 (-0.028..+0.053) | +0.629 | +0.000 |
| base | mu only | +0.707 | -0.076 (-0.119..-0.034) | +0.783 | +0.065 (+0.025..+0.103) |
| +F5 | both | +0.755 | -0.002 | +0.757 | +0.114 (+0.065..+0.161) |
| +F5 | p only | +0.672 | +0.017 (-0.019..+0.056) | +0.655 | +0.030 (-0.002..+0.061) |
| +F5 | mu only | +0.769 | -0.007 (-0.052..+0.037) | +0.777 | +0.128 (+0.086..+0.168) |
| +F1 | p only | +0.323 | -0.024 | +0.347 | -0.319 |
| +F1 | mu only | +0.702 | -0.025 | +0.727 | +0.060 (+0.006..+0.114) |
| all | both | +0.850 | +0.171 (+0.097..+0.242) | +0.678 | +0.208 (+0.120..+0.293) |
| all | p only | +0.475 | **+0.112 (+0.047..+0.182)** | +0.363 | -0.167 |
| all | mu only | +0.742 | +0.031 (-0.018..+0.081) | +0.711 | +0.101 (+0.051..+0.149) |

The model's EV ranks within day with a rank correlation of 0.80 to 0.94 to its mu-only score,
and 0.65 to 0.79 to its p-only score.

Reading:
1. **The all model is the first configuration that forecasts.**
   - Its top tenth of minutes does better before costs. The gross part is +0.171 pp with CI lo
     > 0, where base's is -0.075 and the mechanical EV's is -0.049.
   - It beats the mechanical EV by +0.208 pp (CI +0.120..+0.293), and its own null by +0.362.
   - So it meets both legs of amendment 6's loop condition.
   - The result survives the covered-only read, the basis exclusion and dropping the opening
     window.
   - It is positive in 5 of the 6 folds and fades in the last one: 2022H2 is -0.080 with a wide
     CI.
2. **The forecast is not in any single family.**
   - Each family on its own lifts base's gross part from -0.075 to between -0.074 and +0.013.
     The CI lo is <= 0 in every case.
   - Added one at a time, the families gain +0.346 pp of gross in total. Together they gain
     +0.246.
   - So the all model is many small gains accumulated, not one interaction. The add-one design
     with Holm over seven is not powered to see any one of them.
   - Leave-one-out agrees. No single family is necessary, and every reduced model still
     forecasts.
3. **The pre-registered decision carries a configuration that sorts on cost and risk.**
   - Base + F5 beats the mechanical EV (+0.114, CI +0.065..+0.161). Its gross part is -0.002
     (CI -0.047..+0.047).
   - So by amendment 6 it is a sort on cost and risk, not a forecast.
   - The decision stands as written. Whether E4 also carries the all model is the user's call.
4. **F5's gain runs through the TIME-return regressor, and it is cost.**
   - With F5 added, the mu-only score beats the mechanical EV by +0.128 (base's mu-only:
     +0.065). Its gross part is -0.007.
   - F5's p-only score adds +0.030 (n.s.).
   - So F5 helps the regressor sort minutes by cost and risk. Its ETF and pool-breadth bars
     vary with the time of day, as costs do, and that is consistent with a clock or market-state
     proxy for cost (unregistered diagnostic).
   - The all model's forecast sits in its class probabilities instead. With mu fixed, its gross
     part is +0.112 (CI +0.047..+0.182). The regressor then adds the cost sort on top.
   - The mu term alone pulls base's gross part down to -0.076. A regressor of net return
     learns cost first, which is the tranche-1 lesson again.
5. **Every traded arm loses.**
   - The best is +F5, at -0.104 %/day with CI hi -0.049. It loses about 0.9% per trade, at
     about 1.3 trades a day.
   - A relative top-tenth edge of +0.17 pp before costs is far smaller than the round trip
     these names cost, so the EV > 0 rule still trades losers.
6. **F2's pdc_dist duplicates F1's ret_prev** (correlation 0.99999, minor M4). The +F2 row is
   therefore partly a re-test of F1's previous-close level.


## 18c. Amendment 9: the basis repair and the re-run (2026-10-28, committed before any re-run)

User decision, 2026-10-28: "carry both into E4, and repair the price basis first". So amendment
8's repair is done before E4, and every tranche re-runs on the repaired caches.

9. **The repair (poolpoc 3595eef..6674ea3, `stage2m/rebase.py`).**
   - **The rule.**
     - A name-day is off-basis when its first regular minute open and its last regular minute
       close both differ from the daily cache's open and close by |log| > 0.2.
     - The two prices confirm each other when they agree within 0.05 log.
     - Per symbol, the confirming days are sorted by level and split where neighbours differ by
       more than 0.10 log. A cluster is valid when every member is within 0.10 of its median,
       and it has two or more members or one whose two prices agree.
     - A symbol is repaired only when exactly one valid cluster holds at least half of its
       off-basis days. Its cluster days are then divided by the cluster median (prices / k,
       volume x k, trade count unchanged).
     - A pre-market name-day is rescaled when it shares minutes with the rebased minute bars
       and their median close ratio is off by |log| > 0.2. With no shared minutes, it is rescaled
       when its last pre-09:30 close over the rebased 09:30 open is within 0.10 of a factor
       already confirmed for that symbol.
     - Every other name-day keeps its bars and is listed. Nothing is dropped.
   - **One rule came from reading the caches, before any re-run.** The single-cluster
     condition was added after the plan's rule was applied to the universe scan.
     - BBBY's minute bars on six days in 2020-02/03 sit near $10 while the daily cache falls
       from 7.8 to 2.9. Their levels are 1.41, 1.38, 1.47, 1.66, 1.78 and 2.14, and three of the
       six days have disagreeing prices. That is another security's bars, not a basis factor,
       but three days would otherwise have been rescaled.
     - Every true break in the scan has exactly one factor per symbol.
     - This check reads only the two caches' prices, never a label or a result.
   - **Counts, 2016-02-17..2025-08-06 (160,665 minute name-days).**
     - **Minute:** 744 name-days on 28 symbols rebased. Factors are exact split ratios (5, 7, 8,
       10, 12, 15, 20, 30, 50, 1/5, 1/6) or a split with dividend adjustment (VISN 0.325, CXAI
       49.96, KIDZ 15.06).
     - **Dev (<= 2022):** 150 name-days on 11 symbols. In the old pool these are 78 of 50,240 dev
       name-days and 20 of 7,270 P1 name-days.
     - **Pre-market:** 17 name-days on 5 symbols rescaled (3 overlap, 14 match), all in 2025. No
       dev pre-market bar changes.
     - **Residual, listed and not repaired:** 43 name-days on 37 symbols (dev: 33 on 27). These
       are single-price disagreements, BBBY's six days, and three days that open at exactly x50
       but close 5-9% off it (CXAI, GRML, SMTK, all after 2022).
     - A scan of the repaired caches finds no confirmed break, and no pre-market disagreement
       on shared minutes (24 before).
   - **The holdout (>= 2025-08-07) is not repaired.** Its 251 day folders per cache are copied
     byte for byte and never read. They are repaired with the same rule when the holdout is
     unsealed.
   - **Every reader changes.** paths.MINUTE_DIR and premarket.PM_DIR now point at the repaired
     copies, and the raw caches stay as fetched.
     - The cached_bars, year_labels and year_pm keys carry the file paths. year_pm now also
       keys the minute files it merges, and the Stage 2m panel key now carries both caches'
       day-file stats. So every downstream cache rebuilds.
     - The Stage 1w room-model score is unchanged, because its label is a ratio within one day's
       minute bars and k cancels.

**What is identical.** The code of every runner and every decision rule is unchanged:
- E-1's pool rule (§15);
- E0's card;
- E1's floors;
- E2's population rule (§17c, amendment 3);
- E3's add-one rule with amendments 6-8.

Amendment 8's exclusion is re-derived the same way, by `basis` over the new pool's dev
name-days. It now lists only the residual disagreements. The re-run executes from a detached
poolpoc worktree at the repair's last commit (6674ea3, or the commit that fixes a finding of the
repair's whole-branch review, which runs first), one heavy run at a time, in this order:
1. E-1;
2. E0;
3. E1;
4. E2;
5. E3 add-one on E2's population;
6. E3 leave-one-out;
7. `basis` and `rescore`.

§18b's unregistered robustness reads are repeated on the new frames, again as reads, not tests.

**What supersedes what.** The re-run's numbers supersede §16b, §17d and §18b. The old MLflow
trials stay, each tagged as superseded by its re-run. The trial count is the old count plus the
re-run's.

**Predictions (before any re-run):**
1. The pool changes on at most a few percent of P1 name-days.
2. E2 still carries P1.
3. E3's add-one decision is unchanged: base + F5, with Holm p < 0.05.
4. The all model keeps a gross part with CI lo > 0, and it beats the mechanical EV with CI lo > 0.
5. No population or family decision flips.
6. Every traded arm loses.

**E4 carries both** base + E3's qualifying set and the all model, whatever the re-run decides
(user decision 2026-10-28). If no family qualifies, the first of the two is base.

## 19. Results, the re-run on the repaired caches (2026-10-28/29; poolpoc 6674ea3d1f9f, clean; supersedes §16b, §17d and §18b)

Every run executed from a detached worktree at 6674ea3, one at a time, in §18c's order. The
review's one Important finding (I1: pre-market name-days with no regular minute bar were neither
repaired nor listed) was fixed at 858d5a1 and changes only 2025 pre-market rows. The re-run
therefore kept the 6674ea3 build; the check that 858d5a1 leaves the pool unchanged is at the end
of this section.

MLflow runs (data/research_poolpoc/mlruns): E-1 ba7fae6da7284f739724bae8f11eb1db, E0
e734c1fe3dc9427db9b2a78f69c8f9e6, E1 45e953f085074a66af9f132964051e13, E2
43d11e8f8a7742c9a471853cbd42f5d0, E3 add-one 437757294530475d99f3a2a9ec6e4a7b, E3
leave-one-out 9d80bf73d7c848a59af244abc00a55be. Each is tagged `supersedes` with the pre-repair
run it replaces, and each pre-repair run is tagged `superseded_by`; all twelve stay in the
trial count.

### E-1 (pool hash fc6972f7d7058274; was c596877f26c1d5d2)

| Contrast (share of random trades beating costs, top 10) | Before | After |
|---|---|---|
| model vs universe | 22.7% vs 16.4%, +6.3 (+6.0..+6.6) | 22.7% vs 16.4%, +6.3 (+6.0..+6.6) |
| vs naive | +1.5 (+1.2..+1.8) | +1.4 (+1.1..+1.8) |
| vs room model | +0.6 (+0.3..+0.8) | +0.5 (+0.2..+0.8) |
| vs base | +1.1 (+0.8..+1.4) | +1.0 (+0.7..+1.3) |
| null IC / model IC | 0.029 / 0.338 | 0.027 / 0.338 |

- **PASS, as before.** 82,355 rows on 1,298 days, unchanged.
- **The pool moved more than predicted.** 698 of 7,270 dev P1 name-days left and 698 joined
  (9.6%; 1,396 membership changes), all in 2021-22.
  - Only 12 of the 1,396 are repaired name-days.
  - The cause is the walk-forward pool model. 11 repaired name-days in 2020 are training rows
    for every later fit, so every score from 2021 on moves a little. The within-day rank
    correlation of old and new scores is 0.977, and at a top-10 cut that is enough to replace
    15.4% of the 4,530 P1 name-days of 2021-22 (16.5% in 2021, 14.4% in 2022).
  - Prediction 1 ("at most a few percent of P1 name-days") **failed**.

### E0 (label card)

| Number | Before | After |
|---|---|---|
| TP-first rate vs p* (H30 (2,1)) | 2.8% vs 40.3% | 2.7% vs 40.4% |
| mean r_net, random entry | -1.64% | -1.64% |
| C1 random, %/day | -1.216 | -1.271 |
| C2 open-hold, %/day | -1.928 | -1.948 |
| oracle, %/day | +2.88 | +2.83 |
| open off the daily basis by > 20% | 0.275% of name-days | 0.000% |
| open off by > 3% | 0.564% | 0.289% |

- All three predictions **held**, as before. SPY open-to-close is unchanged (+0.039%/day).

### E1 (floors)

| Number | Before | After |
|---|---|---|
| M1 top-decile lift (pp) | +0.285 (+0.199..+0.363) | +0.302 (+0.210..+0.383) |
| = gross + cost | -0.253 + 0.539 | -0.243 + 0.545 |
| vs null | +0.499 (+0.420..+0.574) | +0.483 (+0.403..+0.563) |
| vs cost-only score | +0.024 (-0.077..+0.121) | +0.074 (-0.015..+0.165) |
| AUC (TP) | 0.633 | 0.624 |
| M1 / C1 / C4, %/day | -0.623 / -0.563 / -1.049 | -0.675 / -0.568 / -1.095 |

- Predictions unchanged: 1 failed (the lift's CI excludes 0), 2 and 3 held.
- The in-run determinism refit (fold 2020H1) is identical.

### E2 (training population)

| Number (pp unless %/day) | Before | After |
|---|---|---|
| raw lift P1 / P2 / P3 | +0.723 / +0.694 / +0.667 | +0.744 / +0.710 / +0.688 |
| P3 vs P1, raw | -0.055 (-0.111..+0.003) | -0.056 (-0.114..+0.008) |
| null-adjusted P1 / P2 / P3 | +0.225 / +0.300 / +0.419 | +0.246 / +0.314 / +0.447 |
| M1-P1 vs mechanical EV | +0.081 (-0.011..+0.162) | +0.099 (+0.001..+0.188) |
| mechanical EV's lift = gross + cost | +0.642 = -0.049 + 0.690 | +0.645 = -0.043 + 0.688 |
| M1-P1 / P2 / P3, %/day | -0.201 / -0.267 / -0.287 | -0.197 / -0.256 / -0.310 |

- **Decision unchanged: carry P1** (best raw lift, amendment 3). The pre-registered
  null-adjusted rule would carry P3, as before. Prediction 2 **held**.
- Both pre-registered predictions held again, and every traded arm loses.
- M1-P1's lift over the mechanical EV now has CI lo just above zero (+0.001). It was -0.011.
  Its gross part is +0.048 (-0.035..+0.124; was +0.024, -0.051..+0.095), so by amendment 6 it
  is still a sort on cost and risk, not a forecast.
- 1,937 s, 4.0 GB; the in-run determinism refit is identical.

### E3 add-one (context families)

| Add-one vs base (pp) | Before | After |
|---|---|---|
| +F1 | +0.029 (-0.046..+0.098), Holm 1.000 | +0.068 (+0.011..+0.125), Holm 0.084 |
| +F2 | +0.040 (-0.021..+0.100), Holm 0.996 | +0.046 (-0.014..+0.106), Holm 0.384 |
| +F3 | -0.006 (-0.070..+0.045), Holm 1.000 | +0.037 (-0.004..+0.080), Holm 0.315 |
| +F4 | +0.010 (-0.034..+0.055), Holm 1.000 | +0.029 (-0.015..+0.074), Holm 0.384 |
| **+F5** | **+0.070 (+0.026..+0.114), Holm 0.007** | **+0.101 (+0.051..+0.150), Holm 0.003** |
| +F6 | +0.014 (-0.050..+0.067), Holm 1.000 | +0.036 (-0.002..+0.075), Holm 0.315 |
| +F7 | -0.004 (-0.033..+0.024), Holm 1.000 | +0.018 (-0.011..+0.045), Holm 0.384 |

| Reference (pp) | Before | After |
|---|---|---|
| base vs mechanical EV | +0.043 (-0.011..+0.093) | +0.016 (-0.076..+0.089) |
| base, gross part | -0.075 (-0.121..-0.029) | -0.096 (-0.172..-0.035) |
| +F5 vs mechanical EV | +0.114 (+0.065..+0.161) | +0.117 (+0.035..+0.185) |
| +F5, gross part | -0.002 (-0.047..+0.047) | +0.008 (-0.062..+0.069) |
| all vs mechanical EV | +0.208 (+0.120..+0.293) | +0.181 (+0.089..+0.268) |
| **all, gross part** | **+0.171 (+0.097..+0.242)** | **+0.153 (+0.075..+0.228)** |
| mechanical EV's lift = gross + cost | +0.642 = -0.049 + 0.690 | +0.645 = -0.043 + 0.688 |
| traded arms base / all / +F5, %/day | -0.178 / -0.156 / -0.104 | -0.150 / -0.162 / -0.121 |

- **Decision unchanged: F5 alone qualifies.** S1 is base + F5 (30 features), as E4's
  pre-registration (§20.3) assumed.
- **Every family's add-one difference rose**, mostly because base got weaker: base's gross part
  fell from -0.075 to -0.096 and its lift over the mechanical EV from +0.043 to +0.016. +F1's
  CI now excludes zero (+0.011..+0.125), but its Holm p is 0.084, so it does not qualify.
- **The all-families model is still the only forecast by amendment 6.** Its gross part is
  +0.153 pp, CI lo +0.075. +F5's gain is still a cost sort: its gross part's CI includes zero.
- **Every traded arm still loses** (each daily CI hi < 0). SPY open-to-close is +0.027%/day.
- Predictions: `every_arm_loses` held. `f1_f3_f4_lift`, `f5_f6_small` and
  `no_config_beats_cost_only` failed, as before. `f7_mainly_offering_days` **flipped to failed**:
  +F7 is +0.019 pp on offering name-days against +0.030 pp on the rest. Both amended predictions
  (amendment 6) failed, as before.
- 2,505 s, peak 5.80 GB; the in-run determinism refit (fold 2020H1) is identical. The first
  attempt was killed for memory at 10:22 on 2026-10-28; see "Run notes" below.

### E3 leave-one-out

| Drop one family from all (pp) | Before | After |
|---|---|---|
| -F1 | -0.038 (-0.082..+0.004), Holm 0.41 | -0.031 (-0.078..+0.015), Holm 1.00 |
| -F2 | -0.037 (-0.082..+0.005), Holm 0.41 | +0.003 (-0.034..+0.040), Holm 1.00 |
| -F3 | -0.019 (-0.058..+0.016), Holm 0.61 | -0.010 (-0.043..+0.027), Holm 1.00 |
| -F4 | -0.032 (-0.071..+0.005), Holm 0.41 | -0.010 (-0.046..+0.027), Holm 1.00 |
| -F5 | -0.042 (-0.081..-0.003), Holm 0.21 | -0.047 (-0.086..-0.008), Holm 0.077 |
| -F6 | -0.034 (-0.071..-0.001), Holm 0.26 | +0.009 (-0.021..+0.039), Holm 1.00 |
| -F7 | -0.009 (-0.049..+0.029), Holm 0.62 | -0.004 (-0.037..+0.030), Holm 1.00 |

- **No single removal clears Holm, as before.** Only F5's removal has a CI below zero. The
  others moved toward zero, so the all model's forecast now leans more on F5 and less on the
  rest. Read beside the add-one table, the families overlap: their seven add-one gross gains
  over base sum to +0.52 pp, about twice the +0.25 pp that all of them gain together.
- The traded arm "all" loses -0.162%/day (CI hi -0.088). ~60 min, peak 5.05 GB.

### Amendment 8 (`basis`, `rescore`)

- `basis` on the new pool finds 22 residual breaks (0.04% of 50,240 dev name-days) on 17
  symbols; 2 are P1 name-days. Amendment 8 excludes 15 symbols, 1.62% of the test rows (was
  2.19%).
- **E3 under the exclusion: unchanged.** F5 alone qualifies (+0.100 pp, +0.053..+0.147, Holm
  0.0035).
- **E2 under the exclusion:** P1 is still the best on raw lift (+0.729 against +0.696 and
  +0.672). But M1-P1's lift over the mechanical EV is +0.087 pp (-0.014..+0.175). The full
  sample's CI lo of +0.001 does not survive, so "M1-P1 beats the mechanical EV" is not robust.
  It was never claimed as a forecast (its gross part's CI includes zero).

### Robustness reads (unregistered, §18b's set, on the new frame)

| Gross part of the all model's top-decile lift (pp) | Before | After |
|---|---|---|
| every row | +0.171 (+0.097..+0.242) | +0.153 (+0.075..+0.228) |
| covered name-days only | +0.174 (+0.100..+0.244) | +0.154 (+0.072..+0.232) |
| without the broken symbols | +0.166 (+0.092..+0.235) | +0.148 (+0.070..+0.223) |
| after 09:44 (t >= 14) | +0.193 (+0.117..+0.268) | +0.174 (+0.097..+0.250) |
| 09:31-09:44 (t < 14, 3.6% of rows) | -0.213 (-0.420..-0.020) | -0.124 (-0.325..+0.071) |
| fold 2020H1 | +0.401 (+0.214..+0.590) | +0.411 (+0.243..+0.598) |
| fold 2020H2 | +0.137 (-0.014..+0.294) | +0.088 (-0.043..+0.227) |
| fold 2021H1 | +0.276 (+0.112..+0.453) | +0.142 (-0.006..+0.308) |
| fold 2021H2 | +0.150 (+0.031..+0.278) | +0.129 (-0.007..+0.260) |
| fold 2022H1 | +0.167 (+0.034..+0.326) | +0.218 (+0.056..+0.388) |
| fold 2022H2 | -0.080 (-0.336..+0.134) | -0.061 (-0.412..+0.207) |

- The forecast survives every subset. It is positive in 5 of 6 folds, as before, but only 2
  folds now have CI lo > 0 (was 4). 2021H1 fell the most (+0.276 to +0.142). The pool's churn
  was in 2021-22 (see E-1), but 2022H1 rose, so the churn alone does not explain the shift.

### EV-term diagnostic (unregistered, refit)

Each of base, +F1, +F5 and all was refit exactly as in E3 add-one (same folds, B = 400,000 rows,
seed 7, P1). The refit's full EV equals E3's saved EV on all 2,466,000 rows (largest absolute
difference 0). Each model is then scored three ways: the full EV; "cls", its class probabilities
with mu_TIME fixed at the fold's training TIME mean; and "mu", its mu_TIME regressor with the
probabilities fixed at the fold's training prior.

| Model, EV term | Gross part (pp), before | Gross part (pp), after | vs mechanical EV (pp), after |
|---|---|---|---|
| base, full | -0.075 (-0.121..-0.029) | -0.096 (-0.172..-0.035) | +0.016 (-0.076..+0.089) |
| base, cls | +0.013 (-0.028..+0.053) | +0.019 (-0.027..+0.062) | +0.000 (-0.052..+0.048) |
| base, mu | -0.076 (-0.119..-0.034) | -0.079 (-0.119..-0.035) | +0.059 (+0.016..+0.100) |
| +F1, full | +0.013 (-0.061..+0.083) | +0.034 (-0.051..+0.112) | +0.084 (-0.021..+0.171) |
| +F1, cls | -0.024 (-0.093..+0.040) | +0.021 (-0.054..+0.092) | -0.297 (-0.388..-0.209) |
| +F1, mu | -0.025 (-0.076..+0.029) | -0.008 (-0.071..+0.056) | +0.070 (-0.003..+0.136) |
| +F5, full | -0.002 (-0.047..+0.047) | +0.008 (-0.062..+0.069) | +0.117 (+0.035..+0.185) |
| **+F5, cls** | +0.017 (-0.019..+0.056) | **+0.044 (+0.004..+0.089)** | **+0.042 (+0.005..+0.082)** |
| +F5, mu | -0.007 (-0.052..+0.037) | -0.020 (-0.063..+0.023) | +0.108 (+0.063..+0.152) |
| all, full | +0.171 (+0.097..+0.242) | +0.153 (+0.075..+0.228) | +0.181 (+0.089..+0.268) |
| **all, cls** | +0.112 (+0.047..+0.182) | **+0.075 (+0.010..+0.147)** | -0.228 (-0.301..-0.150) |
| all, mu | +0.031 (-0.018..+0.081) | +0.039 (-0.018..+0.099) | +0.107 (+0.040..+0.170) |

- **F5's lift over the mechanical EV is still mostly the mu_TIME regressor, and that part is
  cost.** Scored on mu alone, +F5 beats the mechanical EV by +0.108 pp with a gross part of
  -0.020.
- **New after the repair: F5's class probabilities alone now pass both legs of amendment 6, by a
  hair.** Gross +0.044 (CI lo +0.004) and +0.042 over the mechanical EV (CI lo +0.005). Before the
  repair both CIs included zero. In the full EV the regressor's cost sort swamps it (full gross
  +0.008). This read was chosen after E3's numbers were seen, is not Holm-corrected, and is a
  lead for E5's exit and EV design, not a finding.
- **The all model's forecast still sits mostly in its class probabilities**, but less than
  before: cls alone keeps a gross part of +0.075 (CI lo +0.010; was +0.112). Neither term alone
  reaches the full EV's +0.153; the two terms together sort better than either.
- The cls-only EVs of +F1 and all fall far below the mechanical EV (-0.30, -0.23 pp): without the
  row's own TIME return they lose the cost sort.
- Rank correlation of the full EV with its cls-only / mu-only variants, after: base 0.77 / 0.92;
  +F1 0.64 / 0.79; +F5 0.79 / 0.94; all 0.66 / 0.81 (before: within 0.01 of each).
- About 18 minutes (the six folds took 1,015 s), peak 4.1 GB; output
  6674ea3d1f9f/e3-diag/ev_terms_diag.json (script: the scratchpad's ev_terms_diag_rerun.py,
  copied to the basis-repair ledger directory).

### The 858d5a1 check

858d5a1 (the review's I1 fix) changes only 2025 pre-market rows. Per the ledgered ruling it runs in
three steps: (1) build the caches with 858d5a1 into side roots and diff them against the 6674ea3
build, alongside E4's full run; (2) after E4 ends, copy only the changed 2025 pre-market files
into the canonical roots (cache keys carry file mtimes, so a copy mid-run could invalidate a
cache E4 reads); (3) re-run E-1 at 858d5a1 and require its pool hash to equal fc6972f7d7058274.
If both hold, E-1 through E3 stand for 858d5a1. Its outcome is reported with E4's results (§21).

### §18c's predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | The pool changes on at most a few percent of P1 name-days | **Failed**: 9.6% (walk-forward churn; see E-1) |
| 2 | E2 still carries P1 | Held |
| 3 | E3's add-one decision is unchanged: base + F5, Holm p < 0.05 | Held (Holm p 0.003) |
| 4 | The all model keeps a gross part with CI lo > 0 and beats the mechanical EV with CI lo > 0 | Held (+0.075; +0.089) |
| 5 | No population or family decision flips | Held (E2 P1; E3 F5; amendment 8 changes neither) |
| 6 | Every traded arm loses | Held (E1, E2, E3 add-one and leave-one-out; each daily CI hi < 0) |

### Erratum to §18c (review finding M9)

§18c says the Stage 1w room-model score is unchanged "because its label is a ratio within one
day's minute bars and k cancels". The conclusion holds, but the reason is wrong. The room label
is (high - low) / open from the **daily unadjusted** bars (stage1/panel.py, `eod_stats` and
`_attach_day_rows`), not from minute bars, so the minute-cache repair never reaches it. §18c is
committed and is not edited; this erratum corrects it.

### Run notes

- **The first E3 add-one attempt was killed for memory** (2026-10-28 10:22). It built the P1
  family panel (3.5 GB) inside the same process as its evaluate step (5.8 GB), on a 15.6 GB
  machine. Claude Code's background-shell reaper stopped it under OS memory pressure while the
  session was idle.
- **The fix changes no number.** panel3_p1 was pre-built in its own process with chunk_days 10
  (chunk invariance is tested, and chunk_days is not in the cache key). It took 429 s and peaked
  at 3.54 GB. Its PANEL_COLS part is identical to E2's chunk-40 panel_p1. E3 then read the cache.
- Memory was logged every 15 s for the rest of the re-run. The lowest available memory during E3
  was 2.3 GB.

## 20. Pre-registration, tranche 4: E4 sequence models (2026-10-29, committed before any E4 code runs on real data)

User decisions carried here:
- E4 carries both feature sets (§18c).
- Deep models train overnight on the CPU (§12).
- 2026-10-29: "Nightly run, no further questions. Continue the research, doc everything". So
  every open choice below is my call. Each is marked **[my call]** and listed in §20.9.

### 20.1 Question

Does a model that reads the raw minute window rank P1 decision minutes better than M1
(LightGBM on engineered features) does?

### 20.2 Fixed, exactly as in E3

These are fixed so that every E4 number pairs with the E3 re-run's frame on the same rows and
days:
- Population P1 (E2's call). Barrier H30 (2, 1).
- E1's purged 6 half-year folds, with the 5-session embargo.
- B = 400,000 filled training rows per fold, from `run_e2.budget_rows` with seed 7. These are
  the very rows E3's M1 trained on.
- The inner-validation rows are `run_e1._fit_rows` on the fold's inner-validation block
  (t % 5 == 1, filled, seed 7). They are used for early stopping and for the isotonic
  calibration.
- Every P1 test minute is predicted.
- EV = p_TP r_TP + p_SL x (the row's SL term + the training SL rows' slip in its price bucket) +
  p_TIME mu_TIME, with `models._ev_terms`, as E2/E3 compute it. Theta = 0.
- Dev only (<= 2022-12-31). The holdout stays sealed.

### 20.3 The two feature sets and their M1 references

- **S1 = base + the E3 re-run's qualifying families.** That is base + F5 if the re-run's add-one
  decision is as before, or base alone if no family qualifies.
- **S2 = all:** F0, F1-F7 (F7 as its 6 EDGAR features) and F8.
- Each set's M1 reference is read from the E3 re-run's add-one predictions frame, not refit. For
  S1 that is the `add_<family>_ev` column, or `base_ev` if no family qualifies; for S2 it is
  `all_ev`. The mechanical EV (`mech_ev`) comes from the same frame.

### 20.4 Sequence input [my call, within §6]

- **The window.** The last L = 60 bars ending at bar t, the decision bar (bar j is minute
  09:30 + j, and a decision at t reads bars <= t). It is clock-aligned: one row per minute,
  whether or not a bar printed.
- **Pre-market padding.** Before 09:30 the window holds the 120 minutes from 07:30 to 09:29,
  from the pre-market cache merged with the 09:25-09:29 minute bars, as `bars.py` reads them.
- **Scale.** s = atr20 / sqrt(390), with atr20 from daily rows <= D-1.
- **8 channels per minute:**
  1. log(c_k / the previous present close) / s
  2. log(o_k / c_k) / s
  3. log(h_k / c_k) / s
  4. log(l_k / c_k) / s
  5. log(1 + traded shares_k / (ADV20 shares / 390)), where traded shares = v / f_D as in
     `bars.py`
  6. log(c_k / session VWAP_k) / s, with the VWAP accumulated from 09:30 (0 before 09:30)
  7. the regular-session flag
  8. a present-bar mask
- **Missing values.** A minute without a bar is 0 in every channel, with mask 0. Every channel
  is clipped to +/-10.
- **Trade count.** §6's trade count is not in the bar caches, so it is left out. The
  regular-session flag takes its slot.
- **The only static input to M2-M4 is t / 360.** A pure-sequence model gets nothing else.
- **L = 60 only, not {60, 120}: compute.** L = 120 is an optional secondary run for the best
  pure-sequence model, and only if the nights allow. It is not part of any decision.

### 20.5 Models (§7) [sizes and layers are my call within §7's budget]

- **M2, TCN.** 5 residual blocks of 2 causal dilated convolutions each: kernel 3, dilations 1,
  2, 4, 8 and 16, 56 channels, ReLU, dropout 0.1. Its receptive field (125 bars) covers L. It
  reads the last step's state.
- **M3, GRU.** 2 layers x 64, with dropout 0.1 between them. It reads the last step's state.
- **M4, PatchTST-style.**
  - Patches of 5 minutes over all 8 channels together, so 12 patches, each projected to
    d = 64 with a learned position embedding.
  - 3 encoder layers (4 heads, feed-forward 128, dropout 0.1), then the mean over the patches.
- **The head of M2-M4.** The state, concatenated with t / 360, feeds an MLP 64 (ReLU, dropout
  0.1). That feeds three heads: 3 logits (label A), 1 logit (label B), and mu_TIME.
- **M5, hybrid: the best encoder of M2-M4 plus an MLP on the features.**
  - "Best" is the largest primary-metric mean among M2-M4; on a tie, the smaller model.
  - The features are S_k's. Each is standardised on the fold's training rows: median-imputed,
    clipped to the training 1st/99th percentiles, and z-scored. A missing indicator is added for
    each feature that has NaNs in training.
  - The MLP is 128 -> 64. Its output, concatenated with the encoder state and t / 360, feeds an
    MLP 64 and the same three heads.
  - M5 runs twice, as M5-S1 and M5-S2.
  - §7's auxiliary heads (y_H, MFE/MAE quantiles) are deferred to E5, where the label is the
    factor that varies. E4 keeps the label fixed.

### 20.6 Training [my call, within §4 and §7]

- **Loss.** CE_A + 0.5 x BCE_B (soft target b_share; lambda = 0.5 as in §4) + 1.0 x Huber
  (delta 1) on 100 x r_net, over the TIME rows only, for the mu_TIME head.
- **Optimiser.** AdamW (lr 1e-3, weight decay 1e-4), batch 1,024, gradient-norm clip 1.0.
- **Epochs.** At most 8 over the 400,000 rows, reshuffled each epoch from the seed. Early
  stopping on the inner-validation CE_A, patience 2; the best epoch's weights are restored.
- **Seeds.** 7, 8 and 9 per configuration. Torch runs deterministic, with threads fixed per run.
- **Calibration.** As M1: isotonic one-vs-rest on the inner-validation raw probabilities, then
  renormalised.
- **A configuration's primary score is its 3-seed ensemble.** The calibrated probabilities and
  mu_TIME are averaged over the seeds, giving one EV per row, which is how it would be
  deployed. The per-seed numbers are reported as mean +/- sd. The ensemble favours the deep
  models a little over a single M1 fit, so the per-seed numbers should be read beside it.

### 20.7 Metrics, decision, references

- **Primary metric.** The paired daily difference of top-decile lift against the M1 reference,
  with a day-clustered 95% CI (2,000 resamples). Five comparisons:
  - M2, M3 and M4 each vs M1-S1;
  - M5-S1 vs M1-S1;
  - M5-S2 vs M1-S2.

  Holm over the five.
- **Decision.** A deep configuration adds something when its difference is positive with Holm
  p < 0.05.
  - If an M5 qualifies, the qualifying M5 with the larger mean becomes E5's model.
  - If only a pure-sequence model qualifies, it is reported. Its encoder is already inside M5,
    so E5 still uses M1.
  - Otherwise M1 stays. S1 and S2 are both carried on, as §18c says.
- **The binding no-skill reference (amendment 6).** Each configuration gets:
  - its lift over the mechanical EV, paired;
  - its gross/cost split;
  - its lift vs the cost-only score.

  A configuration "forecasts" when its lift over the mechanical EV has CI lo > 0 and its gross
  part has CI lo > 0.
- **No shuffled-label null is trained for the deep models.** The mechanical EV is the binding
  no-skill reference, and a null would double the compute. [my call]
- **Guard.** Delta log-loss against the fold's P1 training prior, and the AUC (TP vs rest), per
  configuration. A configuration whose Delta log-loss is worse than M1-S1's is flagged.
- **Traded arms.** They are logged as trials:
  - which arms: the best of M2-M4, M5-S1 and M5-S2;
  - how: on EV > 0 through the account (fixed basis), as in E3, with SPY open-to-close beside
    them;
  - every configuration is also a model-level trial.
- **Determinism.** M2 is fitted twice on fold 2020H1 with seed 7. The predictions must hash
  identically.

### 20.8 Smoke, compute rule, order, predictions

- **Smoke (not a trial).** The 3-group CV's first 2 folds, B = 50,000, 2 epochs, 1 seed,
  n_boot 300, all five configurations. It measures throughput.
- **Compute rule (fixed now, applied after the smoke).** If the smoke's throughput puts the full
  E4 above 20 CPU-hours, M2-M4 run 1 seed (7) and M5 keeps its 3. The full E4 is M2, M3, M4 and
  M5 x 2, with 3 seeds each. Under the rule, a single seed is M2-M4's primary score.
- **Order (§7).** M2 and M3 first, then M4 (the Transformer after the TCN/GRU), then M5 (after
  E3).
- **The stop check (§9)** is read after E4. It stops the long-only track only if the best dev
  top-decile lift has a CI that includes 0.
- **Predictions (§10, made exact):**
  1. Pure sequence < M1: M2, M3 and M4 each have a mean difference vs M1-S1 below 0.
  2. M5 >= M1 by a small margin: M5-S2 vs M1-S2 has mean >= 0, and its CI includes 0.
  3. No deep configuration qualifies under Holm.
  4. No pure-sequence model forecasts by amendment 6.
  5. Every traded arm loses (daily CI hi < 0).

### 20.9 Calls made by Claude in this section (review these)

| # | Call | Where |
|---|---|---|
| 1 | L = 60 only; 07:30-09:29 pre-market padding, clock-aligned; the regular-session flag replaces the trade count | 20.4 |
| 2 | t / 360 is the only static input of the pure-sequence models | 20.4 |
| 3 | Layer sizes of M2-M4; M4's patches mix the channels (not channel-independent) | 20.5 |
| 4 | §7's auxiliary heads move to E5; M5 has the same three heads as M2-M4 | 20.5 |
| 5 | Loss weights, optimiser, 8 epochs, patience 2 | 20.6 |
| 6 | The primary score is the 3-seed ensemble | 20.6 |
| 7 | No shuffled-label null for the deep models; the mechanical EV binds | 20.7 |
| 8 | The 20 CPU-hour compute rule | 20.8 |

## 21. Results, tranche 4: E4 sequence models (2026-10-29/31; poolpoc cf7045cfa449, clean; MLflow fdbb354be48b49c0b45441930246498e)

**No sequence model beats the tabular M1, and all five §20.8 predictions held.** The three
networks read the last 60 minutes of bars. Each scored below M1-S1 (base + F5) on top-decile
lift: the TCN by −0.043 pp, the GRU by −0.156 and the PatchTST by −0.095. Joined to the tabular
features, the TCN only ties its M1 reference. M5-S1 trails M1-S1 by −0.001 pp and M5-S2 leads
M1-S2 by +0.019 pp, both with CIs wide across zero. Nothing qualifies under Holm, so E5 keeps
M1. M5-S2 forecasts by amendment 6 (gross part +0.133 pp, CI +0.076..+0.193; +0.201 over the
mechanical EV), but M1-S2 forecasts as well (+0.153 gross), so the sequence input adds nothing
measurable. It does sharpen the class probabilities: every network reaches a TP AUC of 0.69 to
0.71, against M1-S1's 0.610. That gain does not reach the EV's top decile. Every traded arm
still loses, by −0.031 to −0.055 %/day, with each CI wholly below 0. The §9 stop check does not
fire (21.5). The final review's M1 exposure changes no number (21.6), and the 858d5a1 check
passed (21.7).

### 21.1 What ran

- **Code.** poolpoc 127a392..cf7045c (seq_data, seq_models, seq_train, run_e4), run from the
  detached worktree that ran E3 (panel key 9df83a549f000ae3, so every E4 test row pairs with E3's
  saved M1 predictions on the identical panel). The final review's fixes landed afterwards at
  9d357cf; neither changes a number of this run (21.6).
- **Smoke first (§20.8).** Five configurations, seed 7, 50,000 rows, 2 epochs, 2 folds, 4
  threads. It projected 76.8 process-hours for the full run with 3 seeds each. That is above
  the 20-hour line, so the compute rule applies: M2, M3 and M4 run seed 7 only, and M5-S1 and
  M5-S2 keep seeds 7, 8 and 9. The smoke's in-run determinism refit (M2, fold 2020H1) hashed
  identically.
- **Smoke incidents, fixed before the full run (cf7045c).**
  1. The first smoke attempt ran from the main worktree. Its LF copies of the source hash to a
     different panel key than the run worktree's CRLF checkout, so three parts started 3.5 GB
     panel rebuilds and were stopped. From then on, every E4 step ran from the run worktree, and
     both drivers resolve the caches alone first.
  2. The smoke's evaluate stopped on "a test row missing from the E3 frame". The E3 frame keeps
     NO_FILL rows (cls = −1) with NaN reference EVs. The check now tests presence by key, and
     finiteness only on filled rows.
  3. M5-S2's smoke part peaked at 7.6 GB in the feature scaler. The transform now runs in
     float32 chunks (full-size probe: 4.16 GB).
- **Full run layout.** Phase A ran M2 and M3 in parallel at 6 threads each, then M4. The encoder
  for M5 was then chosen by §20.5's rule (`encoder`). Phase B ran the six M5 fits one at a time
  at 10 threads, alternating S1 and S2 by seed. Then `evaluate` ran.
- **Phase A (launched 2026-10-29 20:36) took 7 h 27 min.**
  - Per-part runtimes and peaks:

    | Part | Slot | Runtime | Peak RAM |
    |---|---|---|---|
    | M3 | second | 10,851 s | 2.87 GB |
    | M4 | second, after M3 | 8,406 s | 2.71 GB |
    | M2 | first | 26,779 s | 3.87 GB |

  - M2's runtime includes the in-run determinism refit of fold 2020H1. The refit hashed
    identically: 422331281f5e6b18 both times.
  - M2 is the slowest encoder. It trained for 2,967 to 4,575 s per fold at 6 threads. M3 took
    1,292 to 2,280 s, and M4 about 1,370 s.
  - Every part trained on identical rows: each fold's train hash is equal across all nine
    parts.
  - By §20.5's rule the encoder was M2, the largest mean difference vs M1-S1 of the three
    (encoder.json).
- **Phase B (2026-10-30 04:03 to 10-31 01:55) took 21 h 52 min.**
  - The M5-S1 parts took 13,948 to 15,599 s each, with peaks of 4.27 to 4.34 GB.
  - The M5-S2 parts took 11,262 to 11,625 s, with peaks of 6.24 to 6.33 GB. That is above the
    4.16 GB full-size probe, but at least 1.86 GB stayed free.
  - The phase took longer than the smoke projected because the smoke's M5 parts used the
    provisional M3 encoder. The first M5 fold was timed before any layout change was
    considered. The serial layout was then kept, because parts resume only whole (ledger
    ruling).
  - `evaluate` took 95 s.
- **The whole run took 29 h 20 min of wall time and 34.5 process-hours.** The smoke projected
  76.8 for three seeds everywhere. At the measured speeds, three seeds for M2 to M4 would have
  added about 25 process-hours.
- **Epochs.** The pure networks often hit the 8-epoch cap with the 8th epoch as their best:
  - M2 in 4 of its 6 folds;
  - M4 in 4 of 6;
  - M3 in 1 of 6.

  The M5 fits mostly stopped early: 16 of M5-S1's 18 folds and all 18 of M5-S2's. M5-S2's best
  epoch was the 2nd to 4th. The budget was pre-registered (§20.6) and is not changed after the
  fact (21.3, point 6).
- **Trial count.** E4 adds 8 rows to trials.jsonl: 5 model configurations and 3 traded arms.

### 21.2 Results

All 685 P1 dev test days (2020H1 to 2022H2), every P1 test minute (2,466,000 rows), H30 (2,1),
the row SL term, 2,000 day-clustered bootstrap resamples. Each score is the seed ensemble.
Lifts are in percentage points of r_net.

| Configuration vs its M1 reference | Lift difference (pp) | p | Holm p |
|---|---|---|---|
| M2 TCN vs M1-S1 | −0.043 (−0.111..+0.035) | 0.2530 | 0.7590 |
| M3 GRU vs M1-S1 | −0.156 (−0.232..−0.069) | 0.0030 | 0.0150 |
| M4 PatchTST vs M1-S1 | −0.095 (−0.166..−0.016) | 0.0240 | 0.0960 |
| M5-S1 vs M1-S1 | −0.001 (−0.066..+0.075) | 0.9760 | 1.0000 |
| M5-S2 vs M1-S2 | +0.019 (−0.054..+0.105) | 0.6390 | 1.0000 |

The next table splits each score's top-decile lift into its gross and cost parts. It also gives
the lift against the two no-skill references: the mechanical EV (binding) and the cost-only
sort. Δ log-loss is the gain over the fold's training prior.

| Scored by | Top-decile lift (pp) | Gross part | Cost part | vs mechanical EV | vs cost-only | AUC (TP) | Δ log-loss | ECE10 (TP) |
|---|---|---|---|---|---|---|---|---|
| M2 TCN | +0.719 (+0.669..+0.770) | +0.033 (−0.013..+0.080) | +0.686 (+0.661..+0.712) | +0.074 (+0.020..+0.125) | +0.490 (+0.423..+0.564) | 0.701 | +0.0353 | 0.0018 |
| M3 GRU | +0.606 (+0.548..+0.666) | −0.023 (−0.074..+0.031) | +0.629 (+0.604..+0.654) | −0.039 (−0.099..+0.022) | +0.378 (+0.299..+0.461) | 0.698 | +0.0339 | 0.0018 |
| M4 PatchTST | +0.667 (+0.613..+0.720) | +0.039 (−0.008..+0.086) | +0.628 (+0.603..+0.653) | +0.022 (−0.036..+0.078) | +0.438 (+0.367..+0.516) | 0.691 | +0.0281 | 0.0018 |
| M5-S1 | +0.762 (+0.707..+0.819) | −0.018 (−0.066..+0.034) | +0.779 (+0.754..+0.805) | +0.117 (+0.062..+0.170) | +0.533 (+0.463..+0.606) | 0.703 | +0.0404 | 0.0014 |
| M5-S2 | +0.846 (+0.782..+0.909) | +0.133 (+0.076..+0.193) | +0.713 (+0.686..+0.739) | +0.201 (+0.134..+0.269) | +0.617 (+0.538..+0.701) | 0.714 | +0.0446 | 0.0021 |
| M1-S1 | +0.762 (+0.689..+0.825) | +0.008 (−0.062..+0.069) | +0.754 (+0.729..+0.780) | +0.117 (+0.035..+0.185) | +0.534 (+0.468..+0.597) | 0.610 | +0.0053 | 0.0019 |
| M1-S2 | +0.827 (+0.746..+0.905) | +0.153 (+0.075..+0.228) | +0.674 (+0.647..+0.701) | +0.181 (+0.089..+0.268) | +0.598 (+0.519..+0.681) | 0.714 | +0.0431 | 0.0023 |
| mechanical EV | +0.645 (+0.599..+0.695) | −0.043 (−0.081..+0.001) | +0.688 (+0.664..+0.712) | | +0.417 (+0.351..+0.492) | | | |
| cost-only sort | +0.229 (+0.160..+0.299) | −0.623 (−0.688..−0.559) | +0.851 (+0.827..+0.876) | | | | | |

| Configuration | Seeds | Lift difference per seed (pp) | SD (pp) | Guard flag |
|---|---|---|---|---|
| M2 TCN | 1 | −0.043 | | no |
| M3 GRU | 1 | −0.156 | | no |
| M4 PatchTST | 1 | −0.095 | | no |
| M5-S1 | 3 | −0.036, +0.000, −0.013 | 0.018 | no |
| M5-S2 | 3 | +0.015, −0.029, −0.023 | 0.024 | no |

| Configuration | Parameters | Epochs run per fold | Best epoch | Peak GB | Process-hours |
|---|---|---|---|---|---|
| M2 TCN | 99,181 | 6, 8 | 4, 7, 8 | 3.87 | 7.4 |
| M3 GRU | 43,717 | 6, 8 | 4, 6, 7, 8 | 2.87 | 3.0 |
| M4 PatchTST | 108,357 | 8 | 7, 8 | 2.71 | 2.3 |
| M5-S1 | 116,141 | 4 to 8 | 2 to 5, 7, 8 | 4.34 | 12.3 |
| M5-S2 | 130,093 | 4, 5, 6 | 2, 3, 4 | 6.33 | 9.5 |

The traded arms trade at EV > 0 on a fixed basis, as in E2 and E3. The pure arm is M2, the best pure
configuration.

| Traded arm | Mean %/day (CI) | Trades | Per day | Days traded | Net per $ | vs SPY open-to-close |
|---|---|---|---|---|---|---|
| M2 TCN | −0.037 (−0.062..−0.012) | 246 | 0.36 | 133 | −1.08% | −0.064 (−0.149..+0.022) |
| M5-S1 | −0.031 (−0.060..−0.003) | 428 | 0.62 | 145 | −0.53% | −0.058 (−0.147..+0.024) |
| M5-S2 | −0.055 (−0.096..−0.013) | 815 | 1.19 | 257 | −0.49% | −0.082 (−0.171..+0.006) |
| SPY open-to-close | +0.027 (−0.054..+0.110) | | | 685 | | |

### 21.3 Reading

1. **Pure sequences lose to M1-S1, and capacity does not order them.**
   - The TCN comes closest, at −0.043 pp. The GRU is furthest, at −0.156.
   - Only the GRU's deficit survives Holm (p 0.015). M4's unadjusted CI lies below zero, but
     its Holm p is 0.096.
   - None of the three forecasts by amendment 6.
   - M2 beats the mechanical EV by +0.074 pp (CI +0.020..+0.125), but its gross part
     straddles 0 (+0.033, CI −0.013..+0.080). Its edge over the no-skill EV is therefore not
     shown to be a forecast of the price move.
2. **The networks get better probabilities, but not a better top decile.**
   - From the bars and the time of day alone, every network lifts TP AUC to 0.69 to 0.71, and
     the log-loss gain to +0.028 to +0.045. M1-S1 reaches 0.610 and +0.005.
   - That is as much as M1-S2 gets from all 102 tabular features (0.714, +0.043).
   - Yet in M5-S1, the sharper TP probability leaves the lift where M1-S1 has it (−0.001 pp),
     with a gross part of −0.018.
   - A reading, not a result: the 60-minute window mostly teaches the networks the scale of the
     coming move, that is, whether a 2-ATR target is in reach that day. The EV already prices
     that scale through r_TP and the cost. It ranks the minutes that are likely to hit TP
     better. It does not find minutes whose net return is higher.
3. **The sequence adds nothing measurable to the full tabular set.**
   - M5-S2 vs M1-S2 is +0.019 pp (CI −0.054..+0.105). M5-S2 forecasts, but M1-S2 does too, by
     about as much (gross +0.133 vs +0.153; over the mechanical EV +0.201 vs +0.181).
   - The seed spread (SD 0.024 pp) is as large as the difference itself.
   - The three-seed ensembles score above their average seed: by +0.015 pp for M5-S1 and +0.031
     for M5-S2. That is about the size of effect being looked for.
4. **Every traded arm loses, but less per day than §19's arms.**
   - §19's best arm lost −0.12 %/day. These arms trade on 133 to 257 of the 685 days.
   - Per dollar traded they still lose 0.49% to 1.08%.
   - None beats SPY's open-to-close drift. Each difference has a CI that includes 0, and every
     mean is negative.
5. **What E4 rules out.** A 60-minute window of the 8 bar channels, through these three
   encoders at this budget (400,000 rows a fold, 8 epochs), does not find net-return signal
   that M1 misses on P1 at H30 (2,1).
6. **What E4 leaves open.**
   - Longer windows.
   - Other targets: E5's labels and exits.
   - More training. M2 and M4 had their best epoch at the cap in 4 of 6 folds, so they may be
     under-trained. M5, which carries the M2 encoder, stopped early in 34 of 36 folds, so more
     epochs are unlikely to change the M5 comparisons. They could still move M2 and M4 against
     M1-S1.

### 21.4 Predictions (§20.8)

| # | Prediction | Result | |
|---|---|---|---|
| 1 | Pure sequence < M1: M2, M3 and M4 each have a mean difference vs M1-S1 below 0 | −0.043, −0.156, −0.095 | held |
| 2 | M5-S2 vs M1-S2 has mean ≥ 0 and a CI that includes 0 | +0.019 (−0.054..+0.105) | held |
| 3 | No deep configuration qualifies under Holm | the only positive difference (M5-S2) has Holm p 1.000 | held |
| 4 | No pure-sequence model forecasts by amendment 6 | M2 gross CI lo −0.013; M3 and M4 vs mechanical EV CI lo −0.099 and −0.036 | held |
| 5 | Every traded arm loses (daily CI hi < 0) | hi −0.012, −0.003, −0.013 | held |

Every arm traded, so the zero-trade reading ruled before the run (ledger, 2026-10-29 20:42) was
not needed.

### 21.5 The stop check (§9)

§9 stops the long-only track if, after E2 and E4, both of these hold:

- the best dev top-decile lift has a CI that includes 0;
- the best dev policy's daily-return CI upper bound is below 0.

The second condition holds for every E4 arm (hi −0.003 to −0.013). The first fails. The best
dev top-decile lift is M5-S2's +0.846 pp (CI +0.782..+0.909); M1-S2's is +0.827. **The rule
does not fire, and the track continues to E5 (labels and exits) with M1.**

The open question to the user is whether the stop rule should judge lift over the mechanical EV
instead of raw lift. It does not change this outcome:

- over the mechanical EV, the best is M5-S2's +0.201 pp (CI +0.134..+0.269);
- on the gross part, the best is M1-S2's +0.153 (CI +0.075..+0.228).

Both lie wholly above 0. The question stays open for E5 to E7, where it may bind.

### 21.6 The final review and the M1 exposure

A fresh reviewer read dc5333a..cf7045c during phase A: the plan's five Review Focus items, §20,
and the ledger's rulings. It found 0 Critical, 1 Important and 10 Minor issues. All five Review
Focus items hold:

- the window at t sees only bars ≤ t;
- the pre-market slots are exact;
- EV terms and references pair by (date, symbol, t) on identical rows;
- `fit` restores the best epoch;
- parts are meta-checked.

**I1, fixed at 9d357cf.** `evaluate --mode full` ran on whatever parts existed. A hand-run
evaluate during the run would therefore have Holm-corrected over the wrong family and logged
permanent trials. A full evaluation now requires:

- all five configurations;
- one of the two pre-registered seed layouts;
- full-mode parts only.

This run's own evaluate ran the cf7045c code, after every part had finished. The guard's
condition was checked by hand on its results.json, and it holds:

- the configurations are m2, m3, m4, m5_s1 and m5_s2;
- the seeds are M2 to M4 [7] and M5-S1 and M5-S2 [7, 8, 9], the compute rule's second layout;
- all nine parts are mode full, at cf7045cfa449, clean;
- every M5 part's encoder is m2.

**M1, re-graded Important, fixed at 9d357cf.**

- **The finding.** A name-day without a finite, positive atr20 took the median scale of its
  40-date build chunk, which includes later sessions. That breaks the rule that no future data
  reaches an input. What leaked is a cross-sectional ATR median, not a label signal.
- **The exposure.** It touched 8 of 7,270 P1 dev name-days: HUT 2021-07-14, GAME 2021-07-16,
  CPOP 2021-07-29, MITQ 2021-08-05, MYSE 2021-09-13, KSCP 2022-02-25, NA 2022-08-09 and LASE
  2022-10-28.
- **The fix.** The scale now comes from the median of the same session's P1 names.
- **The read.** This run was not rebuilt. An unregistered read (e4_m1_read.py on the run's
  saved scores; e4/m1_read.json) drops those name-days' 2,880 test rows. No mean, CI end or Holm
  p changes at four decimals.
- **Why exactly nothing changes.** It is structural: all 2,880 rows are NO_FILL, because a
  name-day with no finite atr20 never fills. They carry NaN EVs and NaN returns. E4's training
  rows (`run_e2.budget_rows`) and validation rows (`models.training_rows`) both exclude NO_FILL.
  The leaked scale therefore reached only windows that were never trained on, validated on or
  scored. A re-run at 9d357cf would differ only in those windows.

**The ten Minor findings** and the rulings on the reviewer's "declined to judge" list are in
the E4 ledger. Two of them were checked by hand on this run:

- **M2.** The determinism refit of M2, fold 2020H1, hashed identically.
- **M3.**
  - Each fold's train hash is equal across all nine parts.
  - Every M5 part used encoder m2 and its own feature set (30 or 102 columns).
  - E4's terms (fold, cls, price_bucket, r_net, c_rt_pit) equal E3's saved frame on all
    2,466,000 keys.
  - The references in E4's predictions are E3's own scores: M1-S1 is E3's +F5 and M1-S2 is
    E3's all-families model. The mechanical EV is E3's too.

### 21.7 The 858d5a1 check (§19)

**Step 1 passed (2026-10-29, during phase A).** The caches were rebuilt with 858d5a1 into side
roots (6 minutes, 0.53 GB). They differ from the 6674ea3 build only where the ruling allows:

- **Minute cache:** 0 of 5,253 files changed.
- **Pre-market cache:** 110 of 5,170 files changed, all in 2025 (February to 2025-08-06).
  - None is in the holdout (compared by byte hash, never parsed).
  - No changed symbol has a regular minute bar that day.
- **Provenance listing:** the 238 changed pre-market name-days are the new "match" repairs, and
  125 unmatched pre-market open-gaps are newly listed as "unexplained". That is a listing only:
  no file outside 2025 changed.

**Steps 2 and 3 passed (2026-10-31, after E4 ended).**

- **The copy.** The 110 changed files and the provenance listing were copied into the canonical
  roots. The old copies are kept in premarket_rebased.6674ea3, and every copy was sha-checked.
  The canonical pre-market root then equalled the 858d5a1 build byte for byte (5,170 files).
- **The E-1 re-run.** E-1 was re-run at cf7045c, whose E-1 code is byte-identical to
  858d5a1's: the later commits add only E4 files.
  - It rebuilt only the 2025 pre-market feature cache.
  - It wrote pool hash fc6972f7d7058274, equal to the re-run's.
  - 0 of 82,355 pool rows differ, and the dev rows hash identically (aaa1e6b50f1852f3).
  - The run is MLflow 5f3b33c8bf6a4bfabdaf85b356707985, tagged as a check, not a trial.
- **The fallback read was not needed.** It was ruled before the run, for a pool that would
  differ only in 2025.

E-1 through E4 therefore stand for 858d5a1, and for 9d357cf (21.6).

### 21.8 Calls made by Claude in this section (review these)

| # | Call | Where |
|---|---|---|
| 1 | Phase B kept its serial 10-thread layout after the first M5 fold was timed (parts resume only whole) | 21.1 |
| 2 | The run was not rebuilt for review finding M1; an unregistered read without the 8 name-days stands in (it changes nothing) | 21.6 |
| 3 | The run's evaluate (cf7045c, no I1 guard) was not re-run at 9d357cf; the guard's condition was checked by hand | 21.6 |
| 4 | The 858d5a1 step 3 ran E-1 at cf7045c, whose E-1 code equals 858d5a1's; the provenance listing was swapped with the pre-market files | 21.7 |
| 5 | The reading in 21.3 point 2 (the networks learn the move's scale, which the EV already prices) is a hypothesis, not a tested result | 21.3 |

## 22. Pre-registration, tranche 5: E5 label and exit design (2026-10-31, approved by the user; committed before any E5 code runs on real data)

This section replaces the draft `2026-10-31-e5-preregistration-draft.md` (intraday 70da4d0).

User decisions (2026-10-31, "approved, go ahead with your four answers. I won't be here so
don't wait for my answers"):
1. **The primary metric** is the top decile's absolute mean net return. Lift is reported beside
   it.
2. **The model-driven exit** runs as E5d, after E5a-c. Its rule is written once E5a has chosen
   the barrier set. It is committed here as an amendment, with its predictions, before E5d runs.
3. **The stop rule (§9)** reads lift over the mechanical EV, not raw lift. A second check is
   added at the end of E6 (22.6).
4. **Both feature sets** are carried: S1 (base + F5, 30 features) and S2 (all seven families,
   102 features).

Every other open choice below is my call, marked **[my call]** and listed in 22.8.

### 22.1 Fixed, exactly as in E3 and E4

These are fixed so that every E5 number pairs with the E3 re-run's frame on the same rows and
days:
- **Rows and days.** Population P1 (E2's call), the six purged half-year folds, the 685 test
  days 2020H1-2022H2, every P1 test minute. NO_FILL rows (no bar t+1, a zero notional, or a
  missing cost or sigma input) are excluded from training, validation and every metric.
- **Training.** B = 400,000 filled training rows per fold, drawn once with seed 7. NO_FILL does
  not depend on the barrier set, so the same rows are drawn for every set.
- **M1.** LightGBM with Stage 1's parameters at 4 threads: multiclass + isotonic calibration on
  the inner-validation block, the mu_TIME regressor, the B model, and EV with the row's own SL
  term (§17 amendment 2).
- **Costs and account.** The cost model of §3, the account of §2, theta = 0, fixed basis.
- **Data.** The panel is E3's (`panel3_p1`, key 9df83a549f000ae3), and the pool hash is
  fc6972f7d7058274. Dev data only (<= 2022-12-31). The holdout stays sealed.

### 22.2 E5a: which barrier set? (one factor: the set, 8 levels, within each feature set)

| Item | Exact definition |
|---|---|
| Levels | (H, k_u, k_d) in {15, 30, 60, close} x {(2, 1), (1, 1)}. The reference is the default, H30 (2, 1). |
| Label (per set X) | §4's triple barrier with X's (H, k_u, k_d). T = min(t + H, 385), and T = 385 for "close". u = max(k_u sigma_t sqrt(T − t), 1.5 c_rt) and d = k_d sigma_t sqrt(T − t). So the sqrt term is sqrt(H) until T reaches 385, and it shrinks from there; for "close" it is sqrt(385 − t). This matches labels.py; §4's "sqrt(H)" is the same rule before 15:55. Tagging is §4's: TP (High >= e(1+u) + 1 tick), SL (Low <= e(1−d); a tie in one bar is SL) or TIME at T, whichever is first. X's labels come from the E0 label cache (`labels_stage2m_p1_dev_<X>`) and are joined to the panel on (date, symbol, t). They replace cls, r_net, u, d and exit_min, and add y_h. b_share, the costs and the features do not depend on X. |
| Model | M1 refit per set and feature set, on X's labels: 8 sets x {S1, S2} = 16 configurations, 6 folds each. |
| Output | EV_X with X's terms. The mechanical EV_X has probabilities at the fold's P1 training prior under X, and mu_TIME at the training TIME mean under X. It is the no-skill reference. |
| Outcome per row | r_net_X, the row's realized net return under X's exits (§4). |
| **Primary metric ("top-decile net")** | Per day, on filled rows: the mean r_net_X of the rows whose within-day percentile rank of EV_X is > 0.9 (the same top tenth as the lift). The comparison is the paired daily difference vs the default set's top-decile net in the same feature set, on the days both have. The CI is a day-clustered bootstrap (2,000 resamples). Holm runs over the 7 alternatives within each feature set. |
| Decision | A set qualifies when its difference vs the default has mean > 0 and Holm p < 0.05. The qualifying set with the largest mean becomes E5b's set, decided separately for S1 and S2. If none qualifies, the default stays. |
| Also reported per configuration | The lift and its gross/cost split, the lift over the mechanical EV_X, and the forecast test (amendment 6: lift over the mechanical EV and gross part, both CI lo > 0). The top decile's gross mean (r_net + c_rt) and cost mean (−c_rt). The top decile's TP/SL/TIME shares. Δ log-loss and TP AUC against X's own labels. The per-fold mean of the paired difference. |
| Traded arms | Each configuration's EV_X > 0 arm through the account (fixed basis, X's exit minutes), against SPY open-to-close: 16 arms. |
| Integrity check | Two checks run before any metric. (1) The default set's labels from the E0 cache equal the panel's own label columns exactly (cls, r_net, u, d, exit_min) on every row. (2) The default refit's EVs equal E3's saved EVs on every row: S1 against E3's `add_f5_ev` and S2 against `all_ev`, float32 equality with NaN in the same rows. If either fails, E5a stops and reports. |

**Predictions (E5a):**
1. No alternative set qualifies under Holm in either feature set, so the default stays. The
   strongest gross signal so far is +0.15 pp. An exit moves the top decile mostly through cost
   and drift, and E0 shows both are nearly flat across sets.
2. The two to-close sets have the two lowest top-decile nets of the eight, in both feature sets.
   That is the intraday fade, as for random entries in E0.
3. Every traded arm loses (daily CI hi < 0).
4. M1-S2 forecasts by amendment 6 under the default, and under at most two other sets.

### 22.3 E5b: which score? (one factor: the target, 3 levels; the exit fixed at E5a's set)

| Item | Exact definition |
|---|---|
| Levels | **A:** E5a's EV at its chosen set X*, the reference. **y_H:** a LightGBM regression on y_h at X*'s T. y_h is the label's stored barrier-free time exit: a market sell at the first present open >= T + 1, net of commission and both impact legs (labels.py). The prediction is the score. **B:** M1's B model (cross-entropy on b_share, §4), with the predicted share as the score. B does not depend on the barrier set. |
| Exit and outcome | Fixed at X*'s r_net for every level. Only the ranking changes. |
| Model details | y_H uses the `huber` objective with alpha = the training rows' median absolute deviation of y_h about its median **[my call: these names have fat tails]**. It uses LGB_BASE, the same budget rows, the same features and seed 7. It is not calibrated, because only its ranking is used. B is the b_hat already fitted inside E5a's M1. |
| Primary metric | E5a's top-decile net of each score, paired by day vs A's, with Holm over the 2 within each feature set. |
| Decision | y_H or B replaces A only with mean > 0 and Holm p < 0.05. If both qualify, the one with the larger mean wins. |
| Also reported | The lift and its gross/cost split, the lift over the mechanical EV_X*, and the daily rank IC against r_net_X*. |
| Trials | 4 model-level trials, one per score and feature set, on their daily top-decile net. There are no traded arms, because y_H and B have no EV threshold **[my call]**. |

**Predictions (E5b):**
1. Neither y_H nor B beats A, in either feature set. A's EV contains each row's cost, which
   drives the top decile (the cost part was +0.67 to +0.78 pp in E4). y_H also nets out cost,
   so it comes closest.
2. B's top-decile net is below A's in both feature sets. B scores how many later minutes are
   profitable, not how much.

### 22.4 E5c: which exit? (one factor: the exit, 2 levels; the score fixed at E5b's winner)

| Item | Exact definition |
|---|---|
| Levels | **Barrier:** X*'s TP/SL/TIME exits, outcome r_net_X*, the reference. **Time only:** no TP or SL, a market sell at the first present open >= T + 1 with X*'s T. Its outcome is y_h at X*. |
| Ranking | E5b's winning score W in each feature set. Both levels rank the same filled rows by the same score. |
| Primary metric | The top-decile net with outcome y_h minus the top-decile net with outcome r_net_X*, paired by day. It is a single comparison, so there is no Holm; alpha is 0.05. |
| Decision | The time-only exit replaces the barrier exit only with mean > 0 and p < 0.05. |
| Trials | 2 model-level trials (time-only, S1 and S2). There are no traded arms: a time-only arm would need each row's time-exit minute, which the labels store only for TIME rows **[my call]**. |

**Prediction (E5c):** the time-only exit has the lower top-decile net in both feature sets. The
stop cuts the left tail of names that fade, and the TP limit saves the exit spread.

### 22.5 E5d: the model-driven exit (rule pre-registered after E5a-c)

The question is whether TP and SL placed from predicted MFE/MAE quantiles over X*'s horizon
(§4's auxiliary labels, §7's placement heads) beat E5c's exit. The rule will be committed here,
as an amendment with its predictions, after E5a-c's results and before any E5d code runs on
real data. The rule covers which quantiles, how they are fitted without the training rows
scoring themselves, how u and d follow, the relabelling, and the metric. Its primary metric will
be E5a's top-decile net, paired against E5c's chosen exit as a single comparison.

### 22.5a E5d's rule (amendment, written after E5a-c's results in §23, committed before any E5d code)

E5a-c kept E3's design in both feature sets: X* = H30 (2, 1), the score A (M1's EV) and the
barrier exit (§23). E5d has one factor, the exit. It compares TP and SL placed per minute from
predicted excursions against that configuration, within each feature set s.

- **Targets.** X*'s own auxiliary labels from the E0 cache (§4). mfe is the maximum favourable
  excursion and mae the maximum adverse excursion, both over [t+1, T] with
  T = min(t + 30, 385), on filled minutes only. They depend on the horizon alone, not on
  k_u or k_d.
- **Heads.** Two LightGBM quantile regressions on s's features: models.LGB_BASE with
  objective = quantile, seed 7. One fits the median of mfe (q_u = 0.5). The other fits the 75th
  percentile of mae (q_d = 0.75). They are fitted on each fold's budget rows: B = 400,000, seed
  7, the same draw as M1's, since relabelling does not change the fill.
- **No minute scores itself.** The fold's test and inner-validation minutes take the heads
  fitted on its training days. Inner-validation days are disjoint from training days by
  construction (splits.purged_folds). The training days take nested out-of-fold values. The
  fold's training days, sorted, are cut into 3 contiguous blocks of equal day count;
  splits.purged_folds uses fixed calendar blocks, so it cannot split them. Each block is
  predicted by heads fitted on the other two blocks' budget rows (B rows, seed 7). Labels end
  the same day, so no label crosses a block edge. It is asserted that no training day's value
  comes from a fit that saw that day.
- **Placement.** u_i = max(q̂_mfe, 1.5 c_rt), the labels' cost floor, and
  d_i = max(q̂_mae, 0.001). The share of minutes floored is reported on each side, with the
  distributions of u_i and d_i against X*'s u and d.
- **Relabelling.** labels.label_day runs on the fold's minutes with X*'s bars and costs, under
  Barrier(30, k_u = 0, k_d = 1), with sig′ = d_i / √(T − t) and c_rt′ = u_i / 1.5 at the
  decision minutes. Then d = d_i and u = u_i, to rounding (asserted to 1e-9 relative). c_rt
  enters label_day only through the TP floor, and r_net's costs come from comm and imp, which
  are unchanged. Where sig or c_rt is not finite (NO_FILL), sig′ and c_rt′ keep the original
  value. The relabel replaces cls, r_net, u, d and exit_min. y_h, the fill, the costs and the
  features are X*'s. No existing module is edited.
- **The score.** M1 is refit per fold and feature set on the relabelled minutes, with E5a's
  training draw, validation rows, calibration and EV (whose TP and SL terms read the row's
  u_i and d_i). Its EV ranks the minutes.
- **Metric and decision.** The primary metric is E5a's top-decile net: the mean relabelled
  r_net of the top tenth by the refit's EV. It is paired by day against the default
  configuration's top-decile net in the same feature set (E5a's H30 (2, 1) column, which is
  E3's M1). That is a single comparison per feature set (p_holm = p). The model-driven exit
  replaces the barrier exit in s only with mean > 0 and p < 0.05. Also read: the gross/cost
  split, the lift over the mechanical EV (amendment 6, mechanical EV under the relabelled
  terms), the top-decile TP/SL/TIME shares, TP AUC, and the traded EV > 0 arm through the
  account (fixed basis, the relabelled exit minutes, vs SPY open-to-close).
- **Integrity.** Any failure stops E5d.
  1. The relabel path with sig′ = sig, c_rt′ = c_rt and X*'s own barrier reproduces X*'s
     cached labels exactly (cls, r_net, u, d, exit_min, with NaN in the same rows), on the
     smoke's name-days and on the first fold's in the full run.
  2. The NO_FILL minutes after relabelling are X*'s.
  3. The nested day sets are disjoint, asserted per fold.
- **Predictions (Claude's, before the run).**
  1. The model-driven exit does not replace the barrier exit in either feature set.
  2. Its top-decile net is below zero in both.
  3. Its top-decile TP share is above the default's (3-4%, §23) in both. With u at the
     median MFE, the TP sits much nearer than 2σ√30.
  4. Every E5d traded arm loses (daily CI hi < 0).
- **Smoke.** The 3-group CV's first 2 folds, S1 only, B = 100,000, 300 resamples, no trials.
  It also times the heads and the relabel.
- **Trials.** 2 model-level and 2 traded trials, logged to trials.jsonl and MLflow.
- **Compute.** 6 folds × 2 feature sets. Per (fold, s): 8 quantile fits (3 inner blocks × 2
  heads, plus 2 outer), one relabel of the fold's minutes (§22.7's 35-77 s per pass), and one
  M1 fit. About 45 to 90 minutes in all, run as one job from a clean commit in the run
  worktree.

| # | Call made by Claude (review these) |
|---|---|
| 9 | One quantile pair (q_u = 0.5, q_d = 0.75), not a grid: one pre-registered comparison per feature set, no new forking path |
| 10 | The heads use the same feature set as M1 in each feature set |
| 11 | The nested split is 3 contiguous day blocks, without embargo (labels end the same day) |
| 12 | d_i is floored at 0.001; u_i keeps the labels' 1.5 c_rt floor |
| 13 | The relabel reuses label_day through sig′ and c_rt′, so labels.py and panel_build.py stay untouched (their bytes key the caches) |

### 22.6 Amendments to the stop rule (§9), from E5 on (user decision 2026-10-31)

- **The reading.** "The best dev top-decile lift" in §9 means the best lift over the mechanical
  EV: the paired daily difference, the binding no-skill reference of §18 amendment 6. E4's stop
  check is unchanged; it did not fire under any reading (§21.5).
- **The added check.** After E6, the long-only track stops and is reported, without E7, if no
  traded arm logged in E5 or E6 has a mean daily net return whose 95% CI upper bound is > 0.

### 22.7 Compute and order

- **Labels.** All eight sets are in the E0 label cache. A miss rebuilds them in 35 to 77 s
  each.
- **Fits.** E3 fit 17 M1 configurations in 42 minutes (peak 5.8 GB). E5a is 16 configurations,
  so about 45 to 60 minutes. E5b is 2 y_H fits per fold, about 10 minutes. E5c needs no fit.
- **Order.** A smoke run first: 2 folds of the 3-group CV, the default and H60 (1, 1), S1 only,
  B = 100,000, 300 resamples, no trials. Then E5a, E5b and E5c, each run once from one clean
  commit in the run worktree, one heavy job at a time.
- **Trials.** 16 model-level and 16 traded trials (E5a), 4 (E5b) and 2 (E5c), all logged to
  trials.jsonl and MLflow.

### 22.8 Calls made by Claude (review these)

| # | Call | Where |
|---|---|---|
| 1 | E5 drops lambda (A+B through a shared encoder): M1 has no shared trunk, and no deep model was carried | 22.2 |
| 2 | S1 and S2 are separate Holm families, each deciding its own set | 22.2 |
| 3 | The default set's refit must reproduce E3's EVs exactly, or E5a stops | 22.2 |
| 4 | No shuffled-label nulls in E5a; the mechanical EV per set is the no-skill reference | 22.2 |
| 5 | E5b varies only the score; the exit stays at E5a's set | 22.3 |
| 6 | Huber loss for the y_H regression, with alpha = the training MAD | 22.3 |
| 7 | No traded arms in E5b or E5c (no EV threshold; no stored time-exit minute) | 22.3, 22.4 |
| 8 | E5d's rule is written after E5a-c and committed before it runs | 22.5 |

## 23. Results, tranche 5: E5a-c label and exit design (2026-10-31; poolpoc ddc248da1892, clean; MLflow E5a 624d1e75e9034907bc3f8dbd6d09ea53, E5b 8697fc312d30403597ad39fef2489fc1, E5c 897515ab4e804359b5bffc2a6b5773aa)

### 23.0 Runs and integrity

- **Code.** poolpoc e5.py, run_e5.py and metrics.py at ddc248d. That commit carries the final
  review's two Important fixes: the default set now runs first, check 2 runs right after its
  folds, and predictions are saved before any metric. Each stage ran once, in order, from the
  run4 worktree at that clean commit, one job at a time. The smoke ran at 390e7fb and again at
  ddc248d, with identical numbers.
- **Time.** E5a took 4,957 s (fits 4,872 s, about 50 s per fit; §22.7 had projected 45 to 60
  minutes), with a peak of 5.78 GB. E5b took 165 s and E5c 17 s. Every set's labels were a
  cache hit.
- **Integrity check 1 passed.** The default set's cached labels equal the panel's on every row
  (cls, r_net, u, d, exit_min).
- **Integrity check 2 passed.** On every row, the default refit's S1 EV, S2 EV and mechanical
  EV equal E3's saved predictions (float32, 2,466,000 rows). So E5a's default column is E3's
  M1-S1 and M1-S2. S2's gross part, +0.153 pp (CI +0.075..+0.228), and its lift over the
  mechanical EV, +0.181, match §19.
- **Trials.** E5a logged 16 model-level and 16 traded trials. E5b logged 4 model-level (y_H
  and B per feature set) and E5c 2 (time-only per feature set).
- **Units.** pp means percentage points of net return per minute traded. Every top-decile
  figure is the daily mean over the 685 test days.

### 23.1 E5a: which barrier set?

**S1 (base + F5)**

| Set | Top-decile net, pp (CI) | vs default, pp (CI) | Holm p | Gross | Cost | Lift over mech. EV (CI) | Forecast | Top-decile TP / SL / TIME | TP AUC |
|---|---|---|---|---|---|---|---|---|---|
| H15 (2, 1) | -0.913 (-0.961..-0.873) | +0.013 (-0.031..+0.061) | 1.000 | -0.054 | -0.860 | +0.100 (+0.057..+0.137) | no | 3% / 29% / 68% | 0.657 |
| H15 (1, 1) | -0.908 (-0.960..-0.866) | +0.018 (-0.026..+0.065) | 1.000 | -0.061 | -0.847 | +0.135 (+0.089..+0.173) | no | 14% / 28% / 58% | 0.701 |
| H30 (2, 1) (default) | -0.926 (-1.012..-0.855) | — | — | -0.033 | -0.893 | +0.117 (+0.035..+0.185) | no | 4% / 27% / 70% | 0.610 |
| H30 (1, 1) | -0.895 (-0.960..-0.834) | +0.031 (-0.012..+0.081) | 1.000 | -0.028 | -0.866 | +0.167 (+0.101..+0.229) | no | 16% / 25% / 59% | 0.635 |
| H60 (2, 1) | -0.890 (-1.022..-0.774) | +0.036 (-0.032..+0.102) | 1.000 | +0.044 | -0.934 | +0.151 (+0.032..+0.260) | yes | 4% / 24% / 72% | 0.582 |
| H60 (1, 1) | -0.879 (-1.025..-0.764) | +0.047 (-0.034..+0.119) | 1.000 | +0.026 | -0.905 | +0.184 (+0.048..+0.299) | no | 17% / 21% / 63% | 0.591 |
| close (2, 1) | -1.060 (-1.388..-0.756) | -0.134 (-0.403..+0.129) | 1.000 | +0.132 | -1.192 | -0.043 (-0.356..+0.226) | no | 3% / 14% / 83% | 0.596 |
| close (1, 1) | -1.079 (-1.433..-0.755) | -0.153 (-0.452..+0.133) | 1.000 | +0.131 | -1.210 | +0.156 (-0.103..+0.412) | no | 13% / 10% / 77% | 0.588 |

**S2 (all)**

| Set | Top-decile net, pp (CI) | vs default, pp (CI) | Holm p | Gross | Cost | Lift over mech. EV (CI) | Forecast | Top-decile TP / SL / TIME | TP AUC |
|---|---|---|---|---|---|---|---|---|---|
| H15 (2, 1) | -0.880 (-0.944..-0.827) | -0.018 (-0.072..+0.031) | 1.000 | +0.022 | -0.902 | +0.134 (+0.073..+0.184) | no | 3% / 23% / 74% | 0.735 |
| H15 (1, 1) | -0.863 (-0.922..-0.815) | -0.002 (-0.057..+0.052) | 1.000 | +0.030 | -0.893 | +0.179 (+0.125..+0.229) | no | 14% / 23% / 63% | 0.734 |
| H30 (2, 1) (default) | -0.862 (-0.954..-0.775) | — | — | +0.111 | -0.973 | +0.181 (+0.089..+0.268) | yes | 3% / 20% / 77% | 0.714 |
| H30 (1, 1) | -0.827 (-0.921..-0.748) | +0.034 (-0.007..+0.073) | 0.612 | +0.129 | -0.956 | +0.234 (+0.143..+0.316) | yes | 15% / 19% / 65% | 0.682 |
| H60 (2, 1) | -0.832 (-0.974..-0.697) | +0.030 (-0.059..+0.113) | 1.000 | +0.238 | -1.070 | +0.209 (+0.073..+0.342) | yes | 3% / 18% / 80% | 0.682 |
| H60 (1, 1) | -0.813 (-0.950..-0.690) | +0.048 (-0.027..+0.125) | 1.000 | +0.235 | -1.048 | +0.250 (+0.120..+0.375) | yes | 16% / 16% / 68% | 0.650 |
| close (2, 1) | -0.464 (-0.838..-0.088) | +0.398 (+0.064..+0.757) | 0.154 | +0.925 | -1.388 | +0.554 (+0.192..+0.933) | yes | 3% / 11% / 86% | 0.634 |
| close (1, 1) | -0.711 (-1.070..-0.362) | +0.151 (-0.170..+0.469) | 1.000 | +0.623 | -1.334 | +0.525 (+0.210..+0.841) | yes | 13% / 8% / 80% | 0.632 |

**Traded arms (EV > 0, fixed basis, each set's exits), mean daily net return**

| Set | S1 %/day (CI) | S1 trades/day | S2 %/day (CI) | S2 trades/day |
|---|---|---|---|---|
| H15 (2, 1) | -0.051 (-0.097..-0.011) | 0.59 | -0.105 (-0.173..-0.051) | 0.99 |
| H15 (1, 1) | -0.030 (-0.073..+0.004) | 0.46 | -0.075 (-0.135..-0.028) | 1.15 |
| H30 (2, 1) | -0.121 (-0.195..-0.055) | 1.35 | -0.162 (-0.243..-0.088) | 1.84 |
| H30 (1, 1) | -0.048 (-0.099..-0.002) | 0.92 | -0.123 (-0.204..-0.047) | 1.65 |
| H60 (2, 1) | -0.206 (-0.314..-0.117) | 2.10 | -0.249 (-0.347..-0.156) | 2.59 |
| H60 (1, 1) | -0.132 (-0.216..-0.053) | 1.45 | -0.223 (-0.316..-0.135) | 2.32 |
| close (2, 1) | -0.441 (-0.610..-0.279) | 3.52 | -0.377 (-0.538..-0.231) | 3.28 |
| close (1, 1) | -0.673 (-0.861..-0.490) | 4.21 | -0.603 (-0.763..-0.450) | 3.86 |

SPY open-to-close over the same days: +0.027 (-0.054..+0.110) %/day.

- **Decision.** No set qualifies in either feature set, so X* is the default H30 (2, 1) for S1
  and for S2.
  - The largest difference is S2's close (2, 1): +0.398 pp, raw p 0.022, Holm p 0.154. Its
    per-fold differences (2020H1 to 2022H2) are +0.72, +1.06, −0.31, +1.16, +0.16 and
    −0.37 pp, and its CI is 0.7 pp wide.
  - Its top tenth still loses −0.46 pp: a gross part of +0.93 against a cost of −1.39.
- **Every top-decile net is negative**, from −0.46 to −1.08 pp. Costs run from −0.85 to
  −1.39 pp, and no gross part exceeds +0.93.
- **In S2, the horizon moves gross and cost together.**
  - The top decile's gross part rises with H: +0.02/+0.03 at 15 minutes, +0.11/+0.13 at 30,
    +0.24 at 60, and +0.93/+0.62 at the close, for (2, 1)/(1, 1).
  - Its cost rises too: −0.90/−0.89, −0.97/−0.96, −1.07/−1.05 and −1.39/−1.33.
  - So the longer horizon's top tenth is costlier names, with a larger part of the move.
  - In S1 no gross part reaches +0.14, and the two close sets have the lowest nets.
- **Forecast (amendment 6).** S2 forecasts at 6 of the 8 sets, all but the two 15-minute ones.
  S1 forecasts only at H60 (2, 1).
- **TP AUC falls with the horizon** (S2: 0.73 at 15 minutes, 0.63 at the close), and so does
  the daily rank IC (S2: +0.34/+0.33 at 15 minutes, +0.20/+0.18 at the close, for (2, 1)/(1, 1)).
- **Every traded arm's mean is negative.**
  - The 15-minute arms and H30 (1, 1) S1 lose least, because they trade least: 0.46 to 1.15
    trades a day.
  - The close arms lose most: −0.38 to −0.67 %/day at 3.3 to 4.2 trades a day.
  - One arm's CI reaches zero: H15 (1, 1) S1, −0.030 %/day (CI −0.073..+0.004), at 0.46
    trades a day.
- **Why the long-horizon arms trade more and lose more.** This was measured after the run from
  E5a's saved predictions, pooled over test minutes. It was not pre-registered, so it explains
  and decides nothing.
  - In every set measured, the minutes with EV > 0 lose on average.
  - Their share of filled minutes grows with the horizon.

| Set | Feature set | Share with EV > 0 | Rows | Mean EV | Mean realized r_net |
|---|---|---|---|---|---|
| H15 (1, 1) | S1 / S2 | 0.12% / 0.40% | 1,474 / 4,672 | +0.37% / +0.63% | −0.36% / −0.60% |
| H30 (2, 1) | S1 / S2 | 0.52% / 0.95% | 6,103 / 11,194 | +0.51% / +0.84% | −0.31% / −0.66% |
| H60 (2, 1) | S1 / S2 | 1.33% / 2.24% | 15,731 / 26,462 | +0.68% / +1.05% | −0.35% / −1.06% |
| close (2, 1) | S1 / S2 | 6.55% / 10.05% | 77,184 / 118,476 | +0.81% / +1.24% | −1.07% / −1.20% |
| close (1, 1) | S1 / S2 | 5.75% / 9.14% | 67,829 / 107,741 | +0.81% / +1.08% | −0.90% / −1.11% |

  The EV's level is optimistic, more so at long horizons. Its ranking is what the top-decile
  metric reads. This is the EV-term asymmetry seen before: a per-row TP payoff against pooled
  loss terms selects volatile rows. The level is E6's θ. E5 does not change it.

### 23.2 E5b: which score? (at H30 (2, 1) in both feature sets)

| Feature set | Level | Top-decile net, pp (CI) | vs reference, pp (CI) | p (Holm) | Gross | Cost | Lift over mech. EV (CI) | Forecast | Rank IC |
|---|---|---|---|---|---|---|---|---|---|
| S1 | A (EV), reference | -0.926 (-1.012..-0.855) | — | — | -0.033 | -0.893 | +0.117 (+0.035..+0.185) | no | +0.266 |
| S1 | y_H (Huber) | -0.893 (-0.945..-0.838) | +0.033 (-0.023..+0.103) | 0.286 | -0.049 | -0.844 | +0.150 (+0.104..+0.199) | no | +0.280 |
| S1 | B (b_hat) | -1.176 (-1.278..-1.085) | -0.250 (-0.332..-0.174) | 0.001 | -0.026 | -1.150 | -0.133 (-0.241..-0.037) | no | +0.112 |
| S2 | A (EV), reference | -0.862 (-0.954..-0.775) | — | — | +0.111 | -0.973 | +0.181 (+0.089..+0.268) | yes | +0.279 |
| S2 | y_H (Huber) | -0.817 (-0.900..-0.742) | +0.045 (-0.010..+0.100) | 0.109 | +0.085 | -0.902 | +0.226 (+0.148..+0.299) | yes | +0.295 |
| S2 | B (b_hat) | -1.022 (-1.113..-0.932) | -0.160 (-0.238..-0.075) | 0.001 | +0.231 | -1.253 | +0.021 (-0.076..+0.116) | no | +0.123 |

- **Decision.** A, the EV, stays in both feature sets.
- **y_H.** It differs from A by +0.033 pp (p 0.286) in S1 and +0.045 pp (p 0.109) in S2,
  neither significant.
  - Its rank IC is slightly higher than A's: +0.280 against +0.266, and +0.295 against +0.279.
  - Its top tenth costs less: −0.84 against −0.89, and −0.90 against −0.97.
  - In S2 its gross part is lower than A's (+0.085 against +0.111). So its small edge is
    partly a cost sort.
- **B.** It is below A in both: −0.250 and −0.160 pp, Holm p 0.001.
  - b_hat's top tenth costs much more (−1.15 and −1.25), so B leans to volatile names.
  - In S2, B has the largest gross part of the three (+0.231) but loses it to cost.

### 23.3 E5c: which exit? (under A, at H30 (2, 1))

| Feature set | Level | Top-decile net, pp (CI) | vs reference, pp (CI) | p (Holm) | Gross | Cost | Lift over mech. EV (CI) | Forecast |
|---|---|---|---|---|---|---|---|---|
| S1 | barrier exit, reference | -0.926 (-1.012..-0.855) | — | — | -0.033 | -0.893 | +0.117 (+0.035..+0.185) | no |
| S1 | time-only exit | -0.923 (-1.035..-0.838) | +0.002 (-0.032..+0.030) | 0.873 | -0.031 | -0.893 | +0.143 (+0.034..+0.224) | no |
| S2 | barrier exit, reference | -0.862 (-0.954..-0.775) | — | — | +0.111 | -0.973 | +0.181 (+0.089..+0.268) | yes |
| S2 | time-only exit | -0.838 (-0.952..-0.742) | +0.024 (-0.004..+0.050) | 0.083 | +0.135 | -0.973 | +0.229 (+0.117..+0.329) | yes |

- **Decision.** The barrier exit stays in both. The time-only exit differs by +0.002 pp
  (p 0.873) in S1 and +0.024 pp (p 0.083) in S2.
- **Why the two exits barely differ.** The rows and the costs are the same. At H30 (2, 1) the
  barrier exit already ends 70% (S1) and 77% (S2) of the top tenth's minutes at TIME, and only
  3-4% at TP. So the two exits differ on at most about 30% of those minutes, mostly at the
  stop.

### 23.4 Predictions

| # | Prediction (before the run) | Outcome | Measured |
|---|---|---|---|
| E5a-1 | No set qualifies under Holm; the default stays | held | best Holm p 0.154 (S2 close (2, 1)) |
| E5a-2 | The two close sets have the two lowest top-decile nets, in both feature sets | failed | held in S1; in S2 they are the two highest |
| E5a-3 | Every traded arm loses (daily CI hi < 0) | failed | all 16 means are negative; H15 (1, 1) S1 has hi +0.004 |
| E5a-4 | S2 forecasts at the default and at most two other sets | failed | the default and 5 others |
| E5b-1 | Neither y_H nor B replaces A | held | y_H p 0.286 / 0.109; B below A |
| E5b-2 | B's top-decile net is below A's | held | −0.250 / −0.160 pp |
| E5c-1 | The time-only exit has the lower top-decile net | failed | +0.002 / +0.024 pp above the barrier exit, neither significant |

Three of seven held. The failures are about the close sets in S2, one low-activity arm, and
S2 forecasting more widely than expected. None changes a decision.

### 23.5 What follows

- **E5's choices stand at E3's** in both S1 and S2: H30 (2, 1), the EV (A) and the barrier
  exit. E5d (§22.5a, committed with this section) tests model-placed TP and SL against
  exactly that configuration.
- **The §22.6 added check (after E6), as written, would not fire on E5's arms.** An E5 arm
  already has a CI hi > 0: H15 (1, 1) S1, +0.004. But that arm's mean is −0.030 %/day at 0.46
  trades a day. Its CI reaches zero because it trades little, not because it wins.
  - For the user: as approved, the check is passed by any arm that trades rarely enough.
  - It is not changed here, since the user approved it. E6 will report it as written, beside
    the best arm's mean.
- **S2's close (2, 1) is not carried.** The rule decides, and it failed Holm (+0.40 pp,
  Holm 0.154, top-decile net still −0.46 pp). It is recorded for E6/E7 as the largest gross
  part so far: +0.93 pp before a cost of −1.39.
- **Stop rule (§9 as amended in §22.6).** Neither condition holds, so §9 does not fire.
  - The best dev lift over the mechanical EV has CI lo > 0 (S2 close (2, 1) +0.554, lo +0.192;
    the S2 default +0.181, lo +0.089).
  - The best traded arm's CI hi is +0.004, not below 0.

### 23.6 Calls made by Claude during E5a-c (review these)

| # | Call |
|---|---|
| 1 | Integrity check 2 runs right after the default set's folds, not after all fits (review I1). The outcome is the same; a failure would have cost minutes, not the hour. |
| 2 | A check-2 stop record would carry check 1 as all True, by construction: the hook runs only after check 1 passed. |
| 3 | The EV-level table in 23.1 was computed after the results and decides nothing. |

## 23b. Results, tranche 5: E5d model-driven exit (2026-10-31; poolpoc 8ece6e9d7ee7, clean; MLflow cc0ac22653564e2d97147294d0131a96)

### 23b.0 Runs and integrity

- **Code.** poolpoc e5d.py and run_e5d.py at 8ece6e9. That is the plan's three tasks (ebc25da,
  d318201, a9e97f4) plus the final review's fixes.
  - A fresh whole-branch review (opus) found no leakage.
  - Three findings were fixed test-first before the full run. One was its Important finding
    (memory transients). Two I regraded from Minor: a fit-days check that could not fail, and
    no check that E5a's default column holds X*'s labels.
  - The suite passed, 1,916 tests.
- **Smoke.** It ran at a9e97f4 and again at 8ece6e9, with identical predictions row for row
  (229 s and 233 s, peak 4.2 and 4.1 GB).
- **Full run.** One job, from the run4 worktree at the clean commit 8ece6e9.
  - It took 2,202 s (§22.5a estimated 45 to 90 minutes), with a peak of 4.76 GB.
  - Per fold in S1: heads about 62 s, relabel about 30 s, M1 about 30 s. Per fold in S2: heads
    about 139 s, relabel about 29 s, M1 about 66 s.
- **Checks, all passed.**
  - Check 1: the panel equals X*'s cached labels.
  - mfe and mae are finite on every filled minute.
  - E5a's default column holds X*'s cls, r_net and exit_min on the same 2,466,000 test
    minutes.
  - Integrity 1: the relabel path with X*'s own inputs reproduces X*'s cache, on every P1 dev
    name-day.
  - Integrity 2: NO_FILL is unchanged after every fold's relabel, in both feature sets.
  - Integrity 3: no head was fitted on rows from a day it scores, for each nested block and the
    outer heads.
- **Trials.** 2 model-level and 2 traded, logged to trials.jsonl and MLflow.

### 23b.1 The comparison (at H30 (2, 1), score A, M1 refit)

| Feature set | Exit | Top-decile net (pp) | Gross | Cost | Lift over mech. EV | Top tenth TP / SL / TIME | TP AUC |
|---|---|---|---|---|---|---|---|
| S1 | barrier (E5a's default) | −0.926 (−1.012..−0.855) | −0.033 | −0.893 | +0.117 (+0.035..+0.185) | 4% / 27% / 70% | 0.610 |
| S1 | model-driven | −0.877 (−0.921..−0.834) | −0.020 | −0.857 | +0.354 (+0.307..+0.400) | 33% / 27% / 40% | 0.729 (fold mean) |
| S2 | barrier (E5a's default) | −0.862 (−0.954..−0.775) | +0.111 | −0.973 | +0.181 (+0.089..+0.268) | 3% / 20% / 77% | 0.714 |
| S2 | model-driven | −0.809 (−0.859..−0.762) | +0.096 | −0.905 | +0.482 (+0.429..+0.537) | 34% / 26% / 40% | 0.739 (fold mean) |

| Feature set | Model-driven minus barrier (pp) | p | Decision | Per fold, 2020H1 to 2022H2 (pp) |
|---|---|---|---|---|
| S1 | +0.049 (+0.000..+0.108) | 0.049 | **model-driven exit** | −0.05, +0.06, +0.08, +0.01, +0.08, +0.11 |
| S2 | +0.053 (−0.011..+0.118) | 0.105 | barrier exit | −0.12, +0.10, +0.12, +0.05, +0.04, +0.13 |

- **Decision, by the pre-registered rule.**
  - S1: mean > 0 and p 0.049 < 0.05, so the model-driven exit replaces the barrier exit in S1.
  - S2: p 0.105, so the barrier exit stays.
  - S1's p sits at the threshold and its CI touches zero. Holm over the two feature sets would
    give 0.098, but the rule was one comparison per feature set, so the rule's answer stands.
  - Both feature sets are positive in 5 of 6 folds; 2020H1 is negative in both.
- **Every top-decile net is still negative**, −0.88 and −0.81 pp under the model-driven exit.
- **The top tenth's exits change.** Its TP share rises from 3-4% to 33-34%, and its TIME share
  falls from 70-77% to 40%.
- **Not comparable across the two exits:**
  - TP AUC and the lift over the mechanical EV both change with the label, since u_i and d_i
    move the TP label and the mechanical EV's terms.
  - The gross/cost split. Its cost is −c_rt_pit whatever the exit, but a TP exit pays no exit
    spread and one impact leg less, so the gross column overstates the model-driven exit's
    gross. Read the net only (review minor 7).

### 23b.2 Where the exits sit

| Feature set | u_i median | X*'s u median | u_i at the cost floor | d_i median | X*'s d median | d_i at its floor |
|---|---|---|---|---|---|---|
| S1 | 2.32% | 5.96% | 82% of filled test minutes (85% of all placed minutes) | 2.80% | 2.97% | 0% |
| S2 | 2.35% | 5.96% | 81% (84%) | 2.76% | 2.97% | 0% |

Medians are on filled test minutes, as the median over folds.

- **The TP floor binds on about 4 in 5 minutes.** The median 30-minute MFE is below the
  labels' cost floor of 1.5 c_rt. The review measured this on filled 2020-21 minutes: a median
  MFE of 1.06% against a median floor of 2.14%, with P(mfe ≥ floor) = 0.31.
- **So E5d mostly tested a TP at 1.5 c_rt** (median 2.3%, against X*'s 6.0%), with a stop near
  X*'s (2.8% against 3.0%). The stop's floor never binds.
- **What a result here says.** It is about a near, cost-floor TP. It is not about model-placed
  exits in general.

### 23b.3 Where the difference comes from (measured after the run; decides nothing)

These numbers come from the saved predictions, over the same test minutes and days. They were
not pre-registered. The score is either E5a's barrier-trained EV (A) or E5d's refit EV (D).
Each row's top tenth is read under both exits.

| Feature set | Score | Barrier exit (pp) | Model-driven exit (pp) |
|---|---|---|---|
| S1 | A (E5a) | −0.926 (−1.012..−0.855) | −0.918 (−0.964..−0.873) |
| S1 | D (E5d refit) | −0.876 (−0.953..−0.808) | −0.877 (−0.921..−0.834) |
| S1 | −c_rt only | −1.460 | −1.271 |
| S2 | A (E5a) | −0.862 (−0.954..−0.775) | −0.887 (−0.937..−0.836) |
| S2 | D (E5d refit) | −0.804 (−0.899..−0.724) | −0.809 (−0.859..−0.762) |
| S2 | −c_rt only | −1.460 | −1.290 |

- **The exit itself adds nothing.** On the same minutes, the model-driven exit differs from the
  barrier exit by:
  - S1: +0.008 pp (p 0.76) on E5a's top tenth, and −0.001 (p 0.94) on the refit's.
  - S2: −0.025 pp (p 0.37) on E5a's top tenth, and −0.005 (p 0.83) on the refit's.
- **The whole difference is in which minutes the refit ranks top.** With the barrier exit held
  fixed, D's top tenth beats A's by +0.050 pp (p 0.025) in S1 and +0.058 (p 0.027) in S2.
- **That top tenth is cheaper.**
  - Its mean c_rt is lower by 0.036 pp in S1 and 0.067 in S2, about the whole difference.
  - The gain is therefore in the cost part, not the gross part: about +0.01 pp of gross in
    S1, and about −0.01 in S2.
- **It is not plain cost sorting.** Ranking by −c_rt alone loses −1.27 and −1.29 pp.
- **Reading.** Refitting M1 on cost-floor TPs moves its ranking toward cheaper minutes at about
  the same gross. It does not forecast the move better. This is the cost-sort pattern seen
  before.

### 23b.4 Traded arms (EV > 0, fixed basis, each exit's own exit minutes)

| Arm | Mean %/day (95% CI) | Trades/day | Net per $ | Sharpe |
|---|---|---|---|---|
| model-driven S1 | −0.027 (−0.057..−0.001) | 0.60 | −0.51% | −1.09 |
| barrier S1 (E5a) | −0.121 (−0.195..−0.055) | 1.35 | −1.03% | −2.06 |
| model-driven S2 | −0.039 (−0.086..−0.002) | 0.80 | −0.52% | −1.09 |
| barrier S2 (E5a) | −0.162 (−0.243..−0.088) | 1.84 | −0.99% | −2.46 |

- **Both E5d arms lose.** Their CI hi is below zero, barely.
- **They lose less per day than E5a's default arms.** They trade about half as often, and they
  lose about half as much per dollar traded.

### 23b.5 Predictions

| # | Prediction (before the run) | Outcome | Measured |
|---|---|---|---|
| E5d-1 | The model-driven exit does not replace the barrier exit in either feature set | failed | S1 replaced, p 0.049; S2 kept, p 0.105 |
| E5d-2 | Its top-decile net is below zero in both | held | −0.877 / −0.809 pp |
| E5d-3 | Its top-decile TP share is above the default's in both | held | 33.5% / 34.4%, against 3.7% / 3.0% |
| E5d-4 | Every E5d traded arm loses (daily CI hi < 0) | held | hi −0.001 / −0.002 %/day |

Three of four held. The failure is the threshold result in S1.

### 23b.6 What follows

- **E5's design going into E6, by the rules.**
  - S1: H30 (2, 1), with TP and SL placed by the quantile heads and M1 refit on those labels,
    scored by A.
  - S2: H30 (2, 1), score A, the barrier exit.
- **For the user.** By the rule, S1 now uses the model-driven exit. But 23b.3 shows the gain
  comes from a cheaper selection, not from the exit. The E6 pre-registration (drafted, not run)
  carries the rule's choice and asks the user about the alternative.
- **Stop rule (§9 as amended in §22.6).** The lift over the mechanical EV has CI lo > 0 in every
  configuration, so §9 does not fire. The end-of-E6 check stands as approved. E5d's two arms
  both have hi < 0.
- **The final review's deferred minors.**
  - Integrity 1 does not run through the placement path.
  - The floored share in the books covers all placed minutes.
  - The heads' out-of-sample coverage is not reported.
  - The gross/cost split is read across exits (see 23b.1).
  - TP AUC is reported per fold only.
  - The model-level trials lack q_u, q_d and the budget.
  - There is no end-to-end identity test.

### 23b.7 Calls made by Claude during E5d (review these)

| # | Call |
|---|---|
| 14 | Integrity 1 ran on every P1 dev name-day in both modes, a superset of §22.5a's "the smoke's / the first fold's". |
| 15 | Minutes without a placement keep X*'s labels, copied exactly. They were not relabelled under Barrier(30, 0, 1), which would have given them the cost-floor TP. |
| 16 | Two checks were added before any fit: mfe and mae are finite on filled minutes, and E5a's default column holds X*'s labels on the same test minutes. |
| 17 | Two review findings were regraded from Minor to Important and fixed before the full run: integrity 3's block check could not fail, and E5a's labels were never checked. |
| 18 | The decomposition in 23b.3 was computed after the run, and it decides nothing. |

## 24. Pre-registration, tranche 6: E6 policy (2026-10-31, approved by the user; committed before any E6 code runs on real data)

This section replaces the draft `2026-10-31-e6-preregistration-draft.md` (intraday 9ab4f37).

User decision (2026-10-31, "approved, go ahead with your five answers"). The draft's §10 asked
five questions:
1. **The guard.** A level is kept only if its daily mean rises under Holm *and* its net per
   dollar traded rises (proposed in the draft, approved).
2. **θ** uses fixed levels compared on the dev test days with Holm, as E5a chose barrier sets
   (proposed, approved). There is no per-fold inner-validation tuning.
3. **S1's exit.** S1 keeps the model-driven exit that §22.5a's rule chose (proposed, approved).
4. **The end-of-E6 check is tightened [my answer].** The draft offered no proposal here, so the
   answer is mine. An arm passes only if both its mean daily net return *and* its net per dollar
   traded have a 95% CI upper bound > 0 (24.5). Trading rarely can widen the daily CI to reach
   zero, but it cannot lift the net per dollar.
5. **The $10 price cap runs as an exploratory arm [my answer].** The draft offered no proposal
   here either.
   - Each feature set gets one arm that excludes names priced $10 or more.
   - It is logged as a trial and reported with its paired difference.
   - It can never be kept, it never enters the combined policy, and the end-of-E6 check does not
     read it.
   - It was added after §0's reads showed $10+ to be the worst bucket, so it is a forking path,
     labelled as one.

**Two levels the program promised that the draft left out.** Rereading the program found two E6
levels the draft omitted:
- **The B gate (§4: "gating on B is an E6 ablation") is restored [my call].** It is one level,
  pre-planned in §4 before any E6 read. It needs no refit: E5a saved b_hat for both feature
  sets.
- **The bucket-mean SL term (§17 amendment 2: "becomes E6's comparison level") is not run [my
  call].**
  - It needs M1's class probabilities, which E5 did not save, so every fold would have to be
    refit (about 30 minutes, one heavy job).
  - It is the term E2 replaced because it selects volatile rows (§16: those entries' SLs cost
    −9.2% against an assumed −4.7%). As a comparison level it cannot become E6's policy.
  - The cost if this is wrong: a comparison the program listed goes unreported. It can run
    later as one refit pass.

### 24.0 What E6 starts from

- **E5's design** (§23, §23b), with M1 and score A (M1's EV), at H30 (2, 1).
  - S1 (base + F5) uses the model-placed TP and SL (E5d, chosen by the rule with p 0.049; §23b.3
    shows the gain is a cheaper top tenth, not the exit).
  - S2 (all families) uses the barrier exit.
  - Data: P1, the six purged half-year folds, 685 dev test days (2020-2022).
- **Every traded arm loses so far.**
  - The best by mean is E5d S1: −0.027 %/day (CI −0.057..−0.001) at 0.60 trades a day.
  - The only E5 arm whose daily CI reaches zero is E5a's H15 (1, 1) S1: −0.030 %/day (hi
    +0.004) at 0.46 trades a day.
- **Descriptive reads were made while drafting (draft §0).** They re-simulated E5's arms on the
  same dev days E6 is judged on, so the predictions in 24.7 are not blind.
  - The first 14 decision minutes lose about −2.5% per dollar in the barrier arms.
  - $10+ is the worst price bucket in every arm, and under $2 is not the worst.
  - 10 slots bind on some days.
  - Net per dollar is not monotone in EV.

### 24.1 Fixed

- **Predictions.** The saved per-minute EV, r_net, class and exit minute. Nothing is refitted.
  - S1: E5d's `md_s1_*` columns (poolpoc 8ece6e9d7ee7/e5d).
  - S2: E5a's `H30_ku2_kd1_s2_ev` with H30 (2, 1)'s labels (ddc248da1892/e5a).
  - b_hat: E5a's `s1_b_hat` and `s2_b_hat`. Label B does not depend on the barrier or the exit,
    and E5d's refit used E5a's training rows, features and seed, so S1's b_hat is the same model.
  - The two frames must have the same rows in the same order (asserted).
- **Account.** §2's simulator, fixed basis.
  - $25,000, with $2,500 per position, the labels' size. The costs depend on size, so a
    different size would need new labels.
  - One position per symbol, no entries after 15:30, flat by 15:56. Highest EV first within a
    minute.
- **Days.** The 685 dev test days (every day in the frames). Nothing after 2022-12-31 is read.
- **The default policy is E5's arms:** θ = 0, 10 slots, no price floor, the first 15 minutes
  allowed, a 5-minute cooldown, no B gate.

### 24.2 Factors and levels

One factor at a time against the default, in each feature set.

| Factor | Default | Levels | Alternatives per feature set |
|---|---|---|---|
| θ: enter only if EV > θ | 0 | 0.25%, 0.5%, 1.0% (about the 25th, 50th and 85th percentiles of EV among EV > 0 in E5's arms) | 3 |
| Slots | 10 | 3, 5 | 2 |
| Price floor: the decision minute's traded price on the repaired basis (`price_bucket`) | none | $2 (bucket ≥ 1), $3 (bucket ≥ 2) | 2 |
| First 15 minutes | entries allowed | no entry before the 09:45 open: decision minutes t ≤ 13 excluded | 1 |
| Cooldown before a symbol re-enters | 5 min | 15, 30 min | 2 |
| B gate: enter only if b_hat > g | none | g = the median b_hat of the default arm's candidates (EV > 0) in that feature set, computed once from the frame before any arm runs [my call] | 1 |

- That is 11 alternatives per feature set: 22 traded trials, logged to trials.jsonl.
- **The exploratory arm** (answer 5): price bucket 0-3, so under $10, with every other setting
  at its default. That is 2 more logged trials, marked `exploratory`.
- **The default arm** is E5's, already logged. It is re-run as an integrity check, not as a
  trial.
- A price floor or the cap also drops minutes whose price is missing (bucket −1).

### 24.3 Metric and decision

- **The primary metric** is the account's mean daily net return over the 685 days, with zero on
  days without trades.
  - It is paired by day against the default policy in the same feature set, with 2,000
    resamples.
  - Holm is applied within each factor and feature set.
- **Decision.** A level replaces the default only if both hold:
  1. Its paired difference has mean > 0 and Holm p < 0.05.
  2. Its net per dollar traded (Σ r_net × notional over Σ notional) is higher than the
     default's. This is a point estimate; its day-clustered CI is reported.
  - A level with no trades has no net per dollar and is not kept.
- **Also read, for every arm:**
  - Trades, trades per day and days traded.
  - Net per dollar with its day-clustered 95% CI (days resampled, Σ pnl over Σ notional per
    resample).
  - Sharpe, maximum drawdown, and the paired difference vs SPY open-to-close.

### 24.4 The combined policy and its controls

- **The combined policy.**
  - It combines every kept level, taking one level per factor. If a factor keeps more than one
    level, the one with the largest paired-difference mean is used [my call].
  - It is run once per feature set and logged as one trial. It is the policy E5-E6 hands to E7.
  - If no level is kept in a feature set, that feature set's policy is the default, and nothing
    new is logged.
- **For the combined policy only** (and for the default, where nothing is kept):
  - Net per dollar by price bucket and by entry hour (the hour of the entry bar t + 1).
  - The paired difference vs each control. These are reported, not gated.
- **The controls.**
  - **C1**, random timing.
    - Per day, the same number of trades as the policy.
    - They are drawn in a uniform random order (seed 11) from the filled minutes that pass the
      policy's non-EV filters (price floor, first-15 rule) [my call: so C1 differs from the
      policy only in how it chooses].
    - They are admitted through sim.admit with the policy's slots and cooldown.
  - **C2**, open-and-hold: E1's logged series (poolpoc 6674ea3d1f9f, P1, fixed basis), on the
    same days.

### 24.5 After E6: the end-of-E6 check (§22.6, tightened by answer 4)

- **The rule.** The long-only track stops and is reported, without E7, unless at least one
  traded arm logged in E5 or E6 has both:
  - a mean daily net return whose 95% CI upper bound is > 0, and
  - a net per dollar traded whose day-clustered 95% CI upper bound is > 0.
- **The arms it reads:**
  - E5's 18 traded arms (E5a 16, E5d 2), re-simulated from the saved predictions. Each must
    reproduce its logged daily series exactly (24.6).
  - E6's 22 alternatives and the combined policies.
  - It does not read the exploratory arms or the controls.
- **E7** (confirmation, at most 3 finalists) does not run without the user in any case.

### 24.6 Integrity (any failure stops E6)

1. **The default reproduces E5.** The default policy under E6's code, with slots = 10 and
   cooldown = 5 passed explicitly, gives E5's arm:
   - S1 against E5d's arm, S2 against E5a's H30 (2, 1) S2 arm.
   - Full mode: the daily series equals the logged trial's series exactly. Both modes: the
     mean and the trade count equal the run's results.json.
   - Every one of E5's 18 arms that the end-of-E6 check re-simulates must reproduce its own
     logged series in the same way.
2. **The simulator's new parameters change nothing at their defaults.**
   - sim.py's slot count and cooldown become parameters that default to config's. Unit tests
     pin today's results.
   - Each factor changes only what it names, checked at run time against the default arm's
     candidates:
     - θ, the price floor, the first-15 rule and the B gate give a subset of the default's
       candidates.
     - Slots and cooldown give the same candidates.
3. **No late rows.** The frames' dates are asserted ≤ 2022-12-31, and the holdout guard runs on
   them.

### 24.7 Predictions (Claude's, before the run; informed by 24.0's reads)

1. No E6 policy, the combined policy included, has a daily CI lower bound > 0 in either feature
   set.
2. The first-15-minute exclusion is kept in S2. It is not kept in S1, where those minutes are
   only 7% of the model-driven arm's trades.
3. Neither price floor is kept in either feature set.
4. No slot or cooldown level is kept: they trade less without raising net per dollar.
5. No θ level is kept in either feature set.
6. The B gate is not kept in either feature set. b_hat's top tenth is the costliest of the
   three scores (§23.2), so the gate leans to volatile, expensive minutes.
7. The end-of-E6 check, as tightened, fires: no arm has both upper bounds > 0.

### 24.8 Compute and order

- Each arm is one account simulation over the saved predictions, taking seconds. With E5's 18
  re-simulated arms, E6 is minutes, with no model fit.
- **Code (poolpoc):**
  - `sim.run_account` and `sim.admit` take `slots` and `cooldown`.
  - An `e6.py` holds the policies, candidates, the net-per-dollar CI, the decision, the combined
    policy, C1 and the end-of-E6 check.
  - A `run_e6.py` loads the frames, runs the arms, writes the report and results.json, and logs
    the trials and an MLflow run.
- **Order.**
  - A smoke run first, on E5's smoke outputs: S1 only, from E5d's smoke and E5a's smoke; 300
    resamples; no trials.
  - Then the full run, once, from one clean commit in the run worktree.

### 24.9 Calls made by Claude (review these)

| # | Call |
|---|---|
| 19 | E6 reuses E5's saved predictions; nothing is refitted. θ uses fixed levels compared on the dev test days with Holm (answer 2). |
| 20 | Each position stays at $2,500, the labels' size, because the labels' costs depend on it. "Slots" changes how many positions can be open, not their size. |
| 21 | The guard in 24.3 (net per dollar must rise) keeps E6 from rewarding trading less (answer 1). |
| 22 | The θ levels come from the EV quantiles among EV > 0 in E5's arms (24.0). |
| 23 | "The first 15 minutes" means no entry before the 09:45 open: decision minutes t ≤ 13, the entries §3 charges twice the half-spread. The draft's §0 read used t ≤ 14. |
| 24 | The draft's §0 reads were made on the dev days E6 is judged on, before §24. They shaped the levels and predictions, and they are disclosed. |
| 25 | Answers 4 and 5 are mine: the draft offered no proposal for them. |
| 26 | The B gate is restored from §4 with one level, the median b_hat among the default's candidates. The bucket-mean SL term (§17) is not run. |
| 27 | When a factor keeps several levels, the combined policy takes the one with the largest paired-difference mean. |
| 28 | C1 draws from the minutes that pass the policy's non-EV filters, with the policy's slots and cooldown. C2 is E1's logged series. |
| 29 | The end-of-E6 check does not read the exploratory arms or the controls. |

### 24.10 Amendment 10: a strict reading beside the guard (2026-11-01; committed after the smoke, before the full run)

- **Why.**
  - The final code review ran 24.3's rule on synthetic worlds with no signal, where the default
    loses. The point-estimate guard kept 26-45% of those noise levels.
  - With a losing default, a level that trades less raises the daily mean. Its net per dollar is
    above the default's about half the time by chance alone, so the guard does not stop it.
  - The bootstrap p also understates the doubt when a difference rests on few days, as with a
    level that changes few trades.
  - The smoke (S1 only, on E5's smoke outputs) had already run. It kept no level. No full-data
    E6 number existed when this was written.
- **What does not change.** 24.3's decision stands as answer 1 set it. "Kept" is the binding
  reading. The predictions (24.7), the combined policy (24.4) and the end check (24.5) are all
  judged on it.
- **Reported beside every level, the exploratory cap included:**
  1. A paired, day-resampled 95% CI of the level's net per dollar minus the default's. The days
     either arm traded on are resampled whole, and both ratios are recomputed on each resample.
     There is no CI if either arm traded on fewer than 20 days.
  2. The number of days on which the level's daily return differs from the default's.
- **The strict reading.**
  - A level is kept strictly iff it is kept, the CI in item 1 has a lower bound above 0, and it
    differs from the default on at least 20 days.
  - The strict combined policy takes one strictly kept level per factor, the one with the largest
    mean.
  - When it differs from 24.4's combined policy, it runs once, with its own C1, C2 and breakdown,
    and is logged as a trial (`combined_strict_{s}`).
  - The predictions are also read on the strict reading and reported beside the pre-registered
    reading.
  - The end check does not read the strict combined arm. Whether that arm would pass is reported.
- **Which reading goes to E7 is the user's call.**

| # | Call |
|---|---|
| 30 | The reviewer asked for the user to rule before the full run. The user is away. Claude kept the binding rule and added the strict reading, committed before any full-data number, so neither reading was chosen after seeing a result. |

## 24b. Results, tranche 6: E6 policy (2026-11-01; poolpoc f8eea37adfce, clean; MLflow c8def45c6c724003b33bfa322f5c0c6e)

### 24b.0 Runs and integrity

- **Code.** poolpoc sim.py, e6.py and run_e6.py at f8eea37. That is the plan's four tasks
  (c1bf838, e212404, e19d0ab, 7f830dc) plus the final review's fix pass (f8eea37).
  - A fresh whole-branch review (opus) said "yes after fixes". It re-ran all 18 of E5's traded
    arms on the full outputs through E6's code, and every one reproduced exactly.
  - Two fixes were made test-first before the full run:
    - Its Important finding, the guard. This gave the strict reading of §24.10, committed first
      at intraday a5e6aa8.
    - The trials path, regraded from Minor. A full run given `--trials` would have written E6's
      trials into the real log anyway. The test for it also covers the full-mode tail, which
      had no test.
  - The suite passed, 1,953 tests.
- **Smoke.** It ran at 7f830dc and again at f8eea37 (S1 only, E5's smoke outputs, 300
  resamples). The numbers were identical. Each run took about 2 s, at a peak of 0.76 GB.
- **Full run.** One job, from the run4 worktree at the clean commit f8eea37, with 2,000
  resamples. It took 16.4 s, at a peak of 1.59 GB.
- **Checks, all passed.**
  - E5's 18 traded arms reproduce their results.json mean and trade count, and their logged
    daily series exactly.
  - Each feature set's default equals its E5 arm (S1 e5d/s1, S2 e5a/H30_ku2_kd1/s2).
  - Each level changes only what it names.
  - No date is after DEV_END.
- **Trials.** 27 traded trials: per feature set 11 levels, the cap and the combined policy, plus
  S2's strict combined policy. They went to trials.jsonl (133 lines to 160) and MLflow.
- **The B gate's g.** 0.585 in S1, 0.502 in S2.

### 24b.1 S1: model-placed exit (E5d), one change at a time

The default (E5d's arm) makes −0.027%/day (CI −0.057..−0.001) at 0.60 trades a day, trading on
74 of 685 days (410 trades). Its net per dollar is −0.51% (−0.93..−0.00).

Columns:
- "vs default": the paired daily difference, in %/day.
- "net per $ minus default": §24.10's paired CI, in pp.
- "days ≠": the number of days on which the level's daily return differs from the default's.

| Level | vs default (95% CI) | Holm p | Net per $ (95% CI) | Net per $ minus default | Days ≠ | Trades/day | Kept | Strict |
|---|---|---|---|---|---|---|---|---|
| EV > 0.25% | +0.020 (+0.001..+0.041) | 0.064 | −0.22% (−0.62..+0.27) | +0.29 (−0.09..+0.67) | 68 | 0.36 | no | no |
| EV > 0.5% | +0.026 (+0.004..+0.054) | 0.063 | −0.03% (−0.60..+0.56) | +0.48 (−0.11..+0.89) | 72 | 0.25 | no | no |
| EV > 1.0% | +0.027 (+0.002..+0.057) | 0.064 | +0.00% (no CI: 14 days) | +0.51 (no CI) | 73 | 0.11 | no | no |
| 3 slots | +0.009 (−0.001..+0.024) | 0.081 | −0.38% (−0.76..+0.05) | +0.13 (−0.04..+0.31) | 14 | 0.51 | no | no |
| 5 slots | +0.006 (+0.001..+0.014) | **0.024** | −0.41% (−0.80..+0.05) | +0.10 (+0.01..+0.19) | **7** | 0.58 | **kept** | no |
| price ≥ $2 | −0.001 (−0.007..+0.005) | 0.864 | −0.54% (−0.94..−0.08) | −0.03 (−0.15..+0.08) | 6 | 0.58 | no | no |
| price ≥ $3 | +0.004 (−0.005..+0.013) | 0.864 | −0.51% (−0.97..−0.00) | −0.00 (−0.20..+0.19) | 13 | 0.51 | no | no |
| no entry before 09:45 | +0.006 (−0.002..+0.019) | 0.203 | −0.40% (−0.76..+0.00) | +0.11 (−0.05..+0.27) | 14 | 0.58 | no | no |
| 15-min cooldown | +0.007 (−0.006..+0.024) | 0.327 | −0.47% (−0.96..+0.05) | +0.04 (−0.25..+0.28) | 25 | 0.48 | no | no |
| 30-min cooldown | +0.008 (−0.002..+0.019) | 0.210 | −0.52% (−1.14..+0.16) | −0.01 (−0.30..+0.28) | 28 | 0.41 | no | no |
| B gate (b_hat > 0.585) | +0.021 (−0.003..+0.045) | 0.084 | −0.30% (−0.87..+0.37) | +0.21 (−0.46..+0.81) | 70 | 0.26 | no | no |
| price < $10 (exploratory) | +0.007 (−0.001..+0.017) | p 0.094 | −0.45% (−0.93..+0.13) | +0.06 (−0.10..+0.29) | 20 | 0.50 | never | never |

- **Only 5 slots is kept**, and it rests on 7 days. The account rarely fills more than 5 slots,
  so the two arms are identical on the other 678 days. This is the few-days case the review
  flagged. The strict reading refuses it on the days rule, although its per-dollar CI
  is just above 0.
- **The three EV thresholds miss Holm narrowly** (raw p 0.021-0.037, Holm 0.063-0.064). They
  trade less: 0.11-0.36 a day against 0.60.

### 24b.2 S2: barrier exit (E5a's H30 (2, 1)), one change at a time

The default (E5a's arm) makes −0.162%/day (CI −0.243..−0.088) at 1.84 trades a day, trading on
330 days (1,260 trades). Its net per dollar is −0.99% (−1.44..−0.57) and its drawdown is −68.9%.

| Level | vs default (95% CI) | Holm p | Net per $ (95% CI) | Net per $ minus default | Days ≠ | Trades/day | Kept | Strict |
|---|---|---|---|---|---|---|---|---|
| EV > 0.25% | +0.069 (+0.034..+0.106) | 0.0015 | −0.89% (−1.50..−0.33) | +0.11 (−0.20..+0.41) | 264 | 1.19 | **kept** | no |
| EV > 0.5% | +0.107 (+0.063..+0.153) | 0.0015 | −0.72% (−1.60..+0.10) | +0.27 (−0.27..+0.82) | 305 | 0.87 | **kept** | no |
| EV > 1.0% | +0.120 (+0.069..+0.172) | 0.0015 | −0.91% (−2.20..+0.42) | +0.09 (−0.89..+1.18) | 321 | 0.53 | **kept** | no |
| 3 slots | −0.005 (−0.026..+0.013) | 0.704 | −1.11% (−1.59..−0.61) | −0.11 (−0.28..+0.02) | 26 | 1.68 | no | no |
| 5 slots | −0.002 (−0.006..+0.002) | 0.614 | −1.02% (−1.47..−0.59) | −0.03 (−0.05..−0.00) | 12 | 1.80 | no | no |
| price ≥ $2 | +0.008 (−0.015..+0.029) | 0.454 | −1.00% (−1.46..−0.55) | −0.01 (−0.16..+0.11) | 40 | 1.72 | no | no |
| price ≥ $3 | +0.017 (−0.011..+0.045) | 0.454 | −1.06% (−1.56..−0.59) | −0.06 (−0.25..+0.13) | 86 | 1.51 | no | no |
| no entry before 09:45 | +0.073 (+0.040..+0.109) | 0.0005 | −0.61% (−1.10..−0.12) | **+0.39 (+0.18..+0.61)** | 132 | 1.66 | **kept** | **kept** |
| 15-min cooldown | −0.002 (−0.035..+0.028) | 1.000 | −1.15% (−1.61..−0.69) | −0.16 (−0.38..+0.05) | 108 | 1.60 | no | no |
| 30-min cooldown | +0.003 (−0.037..+0.047) | 1.000 | −1.27% (−1.72..−0.80) | −0.27 (−0.55..−0.01) | 122 | 1.41 | no | no |
| B gate (b_hat > 0.502) | +0.133 (+0.081..+0.185) | 0.0005 | −0.39% (−1.16..+0.38) | **+0.60 (+0.08..+1.14)** | 290 | 0.85 | **kept** | **kept** |
| price < $10 (exploratory) | +0.022 (−0.005..+0.050) | p 0.109 | −0.98% (−1.41..−0.58) | +0.01 (−0.16..+0.20) | 59 | 1.62 | never | never |

p = 0.0005 is the floor at 2,000 resamples.

- **The guard problem the review predicted shows up here.**
  - The three EV thresholds raise the daily mean strongly (Holm 0.0015), and each has a point
    net per dollar above the default's.
  - Each of them trades much less (0.53-1.19 a day against 1.84).
  - Their paired per-dollar differences have CIs that straddle 0. Most of their daily gain is
    losing less by trading less. The strict reading refuses all three.
- **Two levels raise net per dollar on its own CI:**
  - Skipping the first 15 minutes: +0.39 pp on 132 differing days.
  - The B gate: +0.60 pp on 290 days.
  - These are the only strictly kept levels in E6.
- **Longer cooldowns make each dollar worse.** The 30-minute cooldown's net per dollar falls by
  0.27 pp, with a CI below 0 (−0.55..−0.01). Why was not measured.

### 24b.3 The combined policies and their controls

| Feature set | Policy | %/day (95% CI) | Trades/day | Days traded | Net per $ (95% CI) | vs C1 (%/day) | vs C2 | vs SPY o→c | End check |
|---|---|---|---|---|---|---|---|---|---|
| S1 | binding: 5 slots | −0.021 (−0.047..+0.001) | 0.58 | 74 | −0.41% (−0.80..+0.05) | +0.059 (+0.028..+0.095) | +1.99 (+1.69..+2.30) | −0.048 (−0.137..+0.037) | passes |
| S1 | strict: the default | −0.027 (−0.057..−0.001) | 0.60 | 74 | −0.51% (−0.93..−0.00) | +0.054 (+0.026..+0.089) | +1.98 | −0.054 (−0.144..+0.032) | fails |
| S2 | binding: EV > 1.0%, no entry before 09:45, B gate | **+0.015 (−0.036..+0.062)** | 0.32 | 54 | +0.54% (−1.32..+2.71) | +0.061 (+0.005..+0.123), p 0.026 | +2.02 (+1.72..+2.34) | −0.012 (−0.115..+0.081) | passes |
| S2 | strict: no entry before 09:45, B gate | −0.013 (−0.074..+0.040) | 0.82 | 147 | −0.18% (−1.04..+0.60) | +0.100 (+0.035..+0.175) | +1.99 (+1.69..+2.31) | −0.040 (−0.145..+0.058) | passes (not read) |

**The controls.**
- C1 is random entry minutes with the same count per day, under the policy's price and open
  rules. It loses −0.046 to −0.112%/day (net per $ −1.7 to −1.9%).
- C2 is E1's open-and-hold. It loses −2.007%/day (−2.316..−1.708).

**What the controls show.**
- Every policy beats both controls. The model's timing is worth +0.05 to +0.10%/day over random
  minutes from the same pool.
- That is not enough to cover costs. No policy's daily CI clears 0, and none beats SPY's
  open-to-close.

### 24b.4 Where S2's binding result comes from (measured after the run; decides nothing)

The scripts and outputs for 24b.4-24b.5 are in `data/research_poolpoc/stage2m/f8eea37adfce/e6/after_run/`.

- **Five trades.**
  - The policy made +$2,490 on 219 trades. Its top five made +$4,293. Their r_net were +0.37,
    +0.36, +0.34, +0.34 and +0.31.
  - Without those five, its net per dollar is −0.40%.
  - Four of the five were on 2021-01-28, the meme-squeeze week. The fifth was LADR on
    2020-03-18.
- **The squeeze week carries it.**
  - Its best day was 2021-01-25: +$2,031 on 9 trades, 8 of them winners. Without that day, the
    policy made +$459.
  - By half-year (%/day): 2020H1 +0.045, 2020H2 −0.025, 2021H1 +0.136, 2021H2 −0.004, 2022H1
    +0.031, 2022H2 −0.081.
- **By price bucket** (trades, net per $): <$2 16 at +4.23%; $2-3 35 at +1.88%; $3-5 50 at
  +0.95%; $5-10 88 at −0.75%; $10+ 30 at +0.58%.
- **By entry hour:** entries from 09:45 to 09:59 made +3.66% per $ on 46 trades. The 15:00 hour
  made −4.67% on 2.
- **The strict combined policy.** Its top five trades are nearly the same (four on 2021-01-28,
  plus LADR). Its other 553 trades lose $6,495.
  By half-year: −0.016, +0.026, +0.099, −0.040, +0.018, −0.156.

### 24b.5 Data integrity: one security under two tickers (found after the run)

- **What was found.** On 2021-01-28, AIFC and JAN took the same trades: the same entry minutes
  and the same exits, with r_net within 0.0005 of each other. Their minute bars that day have
  identical volume and trade count in all 309 minutes, and prices differing by a constant
  0.79%. That is one security under two symbols, rebased differently. The account held it
  twice.
- **How common it is.** I scanned every P1 dev name-day for a same-day partner with identical
  regular-session volume and trade count.
  - 6 of 6,850 name-days (0.09%) are such pairs, on 3 days.
  - AIFC/JAN on 2021-01-28 and 2021-02-25.
  - ATYR/LIFE on 2021-03-24.
- **What it changes.** I re-ran with the second symbol of each pair removed (JAN twice, LIFE
  once; 1,080 rows):
  - S1 does not change.
  - The S2 default: −0.165%/day, against −0.162 logged.
  - The S2 binding combined policy: +0.006%/day (−0.044..+0.050), net per $ +0.22%
    (−1.57..+2.26), against +0.015 and +0.54% logged.
  - The S2 strict combined policy: −0.022%/day (−0.080..+0.024), net per $ −0.32%.
  - Which levels are kept was not re-decided on the cleaned frame; that would be a second E6.
- **Earlier tranches.** These 6 name-days have been in the P1 pool since Stage 1. At 0.09% of
  name-days, their effect on E1-E5's top-decile reads should be small, but it has not been
  measured. Traded arms on those 3 days could have moved.
- **The fix, before E7, is the user's call.**
  - Dedupe the pool by security identity known before the open, such as the vendor's asset id
    or a ticker-change table.
  - Do not dedupe on the day's own bars. That would filter on day-D data, which a live system
    cannot do at 09:30.
  - The fix changes the Stage-1 pool and cache-keyed code, so the caches downstream of it
    would be rebuilt.

### 24b.6 The end-of-E6 check (§24.5, tightened by answer 4)

- **The check.** It reads 42 arms: E5's 18, E6's 22 levels, and the 2 combined policies. An arm
  passes if its daily CI hi > 0 and its net-per-dollar CI hi > 0.
- **The result: 13 pass, so the check does not fire and the long-only track continues.**

| Arm | Mean %/day | Daily hi | Net per $ hi | Trades/day |
|---|---|---|---|---|
| E5a H15 (1, 1) S1 | −0.030 | +0.004 | +0.10% | 0.46 |
| S1 EV > 0.25% / > 0.5% | −0.007 / −0.001 | +0.007 / +0.011 | +0.27% / +0.56% | 0.36 / 0.25 |
| S1 3 slots / 5 slots (= S1 combined) | −0.018 / −0.021 | +0.002 / +0.001 | +0.05% / +0.05% | 0.51 / 0.58 |
| S1 15 / 30-min cooldown | −0.020 / −0.019 | +0.003 / +0.005 | +0.05% / +0.16% | 0.48 / 0.41 |
| S1 B gate | −0.006 | +0.006 | +0.37% | 0.26 |
| S2 EV > 0.5% / > 1.0% | −0.055 / −0.042 | +0.006 / +0.015 | +0.10% / +0.42% | 0.87 / 0.53 |
| S2 B gate | −0.029 | +0.023 | +0.38% | 0.85 |
| S2 combined | +0.015 | +0.062 | +2.71% | 0.32 |

- **What a pass means.** Only one passing arm has a positive mean, and none has a lower bound
  above 0.
- **The check is weak.** Most arms pass by being uncertain: they trade rarely and their upper
  bounds sit barely above 0. A pass says these arms are not ruled out. It does not say they
  work.
- **After the dedupe,** the S2 combined policy would still pass (hi +0.050, net per $ hi
  +2.26%).

### 24b.7 Predictions (§24.7)

| # | Prediction (before the run) | Binding reading | Strict reading (§24.10) |
|---|---|---|---|
| 1 | No E6 policy has a daily CI lo > 0 | held | held |
| 2 | Skipping the first 15 minutes is kept in S2, not in S1 | held | held |
| 3 | Neither price floor is kept | held | held |
| 4 | No slot or cooldown level is kept | **failed** (S1 5 slots, on 7 days) | held |
| 5 | No EV threshold is kept | **failed** (all three in S2) | held |
| 6 | The B gate is not kept | **failed** (S2) | **failed** (S2) |
| 7 | The end-of-E6 check fires | **failed** (13 arms pass) | **failed** |

Three of seven held on the binding reading and five on the strict one.

### 24b.8 Reading

- **No E6 policy makes money with confidence.**
  - The best, S2's binding combined policy, is +0.015%/day (CI −0.036..+0.062).
  - It is three filters chosen on the same 685 days they are judged on.
  - It is positive only through five trades in one squeeze week, and two of those are one
    security counted twice (24b.4-24b.5). Without the duplicate it is +0.006%/day.
- **What E6 found that looks real is two filters in S2:**
  - Skipping the 09:30-09:44 entries.
  - The B gate.
  - Both raise net per dollar on a paired CI that clears 0, over many differing days. Together
    they still lose: −0.013%/day and −0.18% per $.
- **The EV thresholds in S2 are the case the guard missed.** They cut a losing default's
  trading by 35-71% and the daily loss with it. Per dollar they change nothing measurable.
- **S1's model-placed exit leaves nothing to tune.** Its default already trades 0.6 times a day.
  Only 5 slots passes the binding rule, and it does so on 7 days.
- **The θ wording in §24.2 is corrected here.** §24.2 put the θ levels at "about the 25th, 50th
  and 85th percentiles of EV among EV > 0". In E6's frames, the share of EV > 0 minutes at or
  below each level is:
  - S1: 38%, 62% and 85%.
  - S2: 34%, 53% and 72%.
  - The levels are fixed values and stand. Only the description was off.

### 24b.9 What follows

- **By §24.5 the long-only track continues.** E7 comes next and needs the user.
- **Three questions for the user before E7:**
  1. Which reading goes on: the binding one or the strict one?
     - Binding: S1 at 5 slots. S2 with EV > 1.0%, no entry before 09:45, and the B gate.
     - Strict: S1 at the default. S2 with no entry before 09:45 and the B gate.
  2. The duplicate-security fix (24b.5), which changes the pool and its caches.
  3. Whether the end check should stay as weak as it is. Any 1 of 42 arms keeps the track, and
     13 do so only by being uncertain.
- **Still pending before E7:**
  - The basis-repair minors (§19), deferred because they change cache keys.
  - The duplicate fix.
- **The final review's deferred minor.** The paired daily difference's bootstrap p is still the
  plain day bootstrap. It is anti-conservative when a difference rests on few days. The
  days-differing count, now reported, is the disclosure.

### 24b.10 Calls made by Claude during E6 (review these)

| # | Call |
|---|---|
| 31 | The strict combined policy got its own C1, C2 and breakdown, so both readings carry the same evidence. The end check does not read it: adding arms after pre-registration could only weaken the stop rule. |
| 32 | The reads in 24b.4 and 24b.5 were computed after the run. They decide nothing. |
| 33 | The duplicate-security fix was not applied. It changes the Stage-1 pool and cache-keyed code shared with every tranche, so it is the user's call before E7. |
| 34 | §24.2's θ percentiles were wrong (24b.8). The levels are fixed values and stand. |

## 25. Amendment 11: one security per name-day, the basis-repair minors, and the dev re-run (2026-11-01; committed before any code or re-run)

User decision, 2026-11-01: "fix the duplicate tickers first, then go strict".
- The duplicate fix (24b.5) is applied, and every dev tranche that feeds E7 re-runs on the fixed
  universe.
- The strict reading (§24.10) is the one carried to E7. E7's pre-registration (§26) is written
  after this re-run, because its finalists are the re-run's strict policies.
- The end check (§24.5) stays as pre-registered. The user did not ask to tighten it.
- The basis-repair minors (§19's review), deferred until the caches were next rebuilt, are fixed
  in the same rebuild.

### 25.1 What a scan of the daily cache found (prices only; no label or result was read)

- **The daily cache.** It holds 25,545 same-day groups of rows with identical unadjusted open,
  high, low, close, volume and trade count (volume > 0), from 2016-01-04 to 2025-08-06. They
  come from 70 symbol sets.
  - 47 sets repeat over 11 to 2,395 sessions. These are renames and SPAC mergers that the
    vendor serves under both symbols, for example IAC/PPLI, ATYR/LIFE, AIFC/JAN, AIFC/ARCI,
    ECA/OVV, KIND/NXDR and AXG/SWIN.
  - 23 sets match on a single session and never again.
- **E-1's universe.** The scan checked §19's pool, which has every scored row from 2017-01 to
  2025-08-06.
  - 178 name-day pairs (356 rows) are one security under two symbols: 124 in dev and 54 in
    confirmation.
  - Both symbols sit in P1 on 29 days: 13 in dev and 16 in confirmation. AXG/SWIN in 2023-24
    accounts for 14 of the 16.
  - A same-day match on volume and trade count alone, with prices left free, finds the same 356
    rows and no others.
- **What a doubled P1 slot did.** On 3 of the 13 dev days, both symbols have the same minute
  bars, so the account traded the security twice (24b.5). On the other 10, one symbol has no
  minute bars that day: the minute vendor files the security's bars under one symbol, usually
  its later one. That symbol filled a pool slot with a name that could not trade, and in Stage
  1 it was a training row with a zero label.
- **Every pair is visible before the open.** All 178 pairs also match on each of the two
  sessions before the day.

### 25.2 The rule (exact)

- **Identity, known before the open.** On decision day D, two symbols are one security when
  their daily-cache rows on each of the two sessions before D (the cache's own session
  calendar) are identical in unadjusted open, high, low, close, volume and trade count, with
  volume > 0 and trade count > 0.
  - Two sessions, not one, because a single-session match can be chance: 23 sets match once
    and never again.
  - Symbols linked this way form groups (connected components).
- **One row per group stays.** It is the symbol whose regular-session minute bars that day exist
  (the label's n_bars >= 2), when exactly one symbol has them. Otherwise it is the
  alphabetically first symbol.
  - This reads day D's minute files. It chooses which symbol's copy of one security's bars is
    used, and never whether the security is in the universe. Every security-day stays exactly
    once, with or without bars. §1.1's rule (no filter on day-D bar counts) is about dropping
    name-days, and none is dropped here.
  - Where both symbols have bars, the bars are the same up to a constant price basis (AIFC/JAN:
    0.79%).
- **Where it is applied.** In E-1's universe (`stage2m.universe.load`), right after the
  admissible-key join and the label join, and before the pre-market features and their
  within-day ranks.
  - So the walk-forward Stage 1s+pm model is fit and scored on one row per security-day.
  - P1, P2 and P3 hold distinct securities, in dev and in confirmation.
- **Left as it is.**
  - The parent POC's cached daily panel computes its D-1 feature ranks over the whole $1-50
    band, duplicates included. One extra row among the thousands of names of a day moves a
    percentile rank by less than 0.1 points.
  - The Stage 1w room model, a control arm in E-1, is not refit.
- **Checks. Any failure stops the re-run.**
  1. After the dedupe, no two universe rows of one day share their day-D daily volume and trade
     count. This reads day-D data as a check, never as a filter.
  2. No two P1 name-days of one day share their regular-session minute volume and trade count
     (24b.5's scan, run on the new pool over dev and confirmation).
  3. The rows removed are counted per year. In 2017-2025 the count must equal the scan's 178.

### 25.3 The basis-repair minors (§19's whole-branch review)

- **M1.** A conflict day is rebased with its symbol's cluster factor when its 09:30 open is
  within 0.01 log of that factor and its last regular close is within 0.10 log (SPREAD) of it.
  - This repairs CXAI 2023-03-15, GRML 2024-07-05 and SMTK 2025-04-07. Each opens at exactly
    x50, with the close 5-9% off. All three are in confirmation.
  - The close condition is a pre-run correction (2026-11-01, before any code ran on real data).
    Without it, a day whose two prices disagree by more than a cluster's spread would be
    rebased on its open alone. The existing test that a disagreeing day stays a conflict pins
    this.
- **M2.** A pre-market name-day whose minute name-day is listed but not rebased gets no
  pre-market factor, neither overlap nor match, because its ratios were measured against
  off-basis bars. If it is itself off-basis (an open gap or an overlap break), it is listed
  "unexplained". There are 0 such cases today. (Pre-run correction, 2026-11-01: the first
  wording named only the overlap factor.)
- **M3.** The overlap rule needs at least 2 shared minutes, or 1 shared minute whose q_open
  agrees with it within 0.10 log.
  - BNRG 2025-02-03 (6.00 vs 5.86) and KIDZ 2025-07-10 (15.0 vs 15.1) have one shared minute
    each and agree, so nothing changes today.
- **M5.** The build writes to `<root>.tmp`, then writes a manifest, then renames. The readers'
  entry points refuse a root without the manifest.
- **M6.** Name-days after `--last-day` but before the holdout are listed
  "after-last-day-unrepaired", not "holdout-unrepaired".
- **M7.** The rewrite asserts that price and volume columns are floats.
- **M8.** `basis` uses rebase's public functions and its THRESH.
- **M10.** `rescore` needs `--basis` explicitly. It has no unscoped default.
- **M12.** `day_ratios` asserts one daily row per symbol.
- **Not changed.**
  - M4 fails safe, and loosening it would reopen the BBBY case.
  - M9 is the erratum in §19.
  - M11's type hints on existing test helpers are not added, because CLAUDE.md says tests are
    never modified automatically. The M2 case gets a new test.
- **Expected effect.** Byte-comparing the old and new rebased roots, only M1's three minute
  name-days may differ. The holdout's day folders are again copied without being read.
  - A dry run of the factor decisions on the raw caches (2026-11-01; no label read) changes
    exactly 4 provenance rows. The three M1 days go from conflict to rebased. SMTK
    2025-04-07's pre-market open gap is no longer listed, because the rebased 09:30 open now
    meets the pre-market close; it gets no factor, so its bytes do not change.
  - New counts: minute rebased 747, unexplained 40; pre-market match 252, overlap 3,
    unexplained 124.

### 25.4 The re-run

- **Where it runs.** From a detached run worktree at the fix's last commit, after the fix's
  whole-branch review, one heavy run at a time.
- **Order.**
  1. The rebase build into new roots, the byte diff against the old roots, and
     `basis --universe`.
  2. E-1, with checks 1-3.
  3. E0, E1, E2, E3 add-one, E3 leave-one-out, then `basis` and `rescore` (amendment 8).
  4. E5a, E5b, E5c, E5d, E6.
- **E4 is not re-run.** It takes 29 hours of wall time, and its decision (no sequence model beats
  M1; M5 ties) does not feed E7, which uses M1. Its numbers stay as measured on §19's pool and
  are labelled that way.
- **Decision rules are unchanged.** Each decision is re-made by its own rule. Where code is
  wired to an earlier decision, the re-run passes the new run's outputs explicitly:
  - E3 reads E2's population.
  - E5 reads E3's predictions and its S1 = base + F5.
  - E5d is written for X* = H30 (2, 1).
  - E6 reads E5d's exits and E1's C2 series.
- **If a decision changes:**
  - E2, E3 or E5a: the cascade stops before the next tranche. The change is written as an
    amendment before that tranche runs.
  - E5d, in one feature set: E6 uses that feature set's chosen exit. That is the barrier exit
    if E5d does not replace it.
- **What supersedes what.** The re-run's numbers supersede §19 (E-1 to E3), §23, §23b and §24b.
  - The old MLflow runs are tagged `superseded_by`.
  - The trial count is the old count plus the re-run's.
- **E6's reading.** The strict reading is carried, per the user.
  - The binding reading is still computed and reported.
  - The end check still reads §24.5's 42 arms.

### 25.5 Predictions (before any re-run)

1. The pool churns through the walk-forward refit, as in §19: 5-15% of dev P1 name-days change.
2. E-1 passes. E2 carries P1. E3's add-one keeps base + F5, possibly with another family. The all
   model keeps a gross part with CI lo > 0.
3. E5a: no set qualifies. E5b keeps the EV (A). E5c keeps the barrier exit.
4. E5d: S2's exit is not replaced. S1's is replaced, which is a coin flip at §23b's p = 0.049.
5. E6, strict reading: S1 keeps nothing. S2 keeps no entry before 09:45 and the B gate. No policy
   has a daily CI lo > 0, and the end check does not fire.
6. Every traded arm of E1-E5 still loses.

| # | Call |
|---|---|
| 35 | The identity rule needs a match on two prior sessions, not one, because 23 symbol sets match once by chance. All 178 universe pairs pass it. |
| 36 | Within a group, the symbol with the day's minute bars is kept. This reads day D only to choose which copy of one security's bars to use; no security-day is dropped or added. Otherwise the alphabetically first symbol is kept, because the free caches carry no vendor asset id. |
| 37 | The Stage-1 panel's D-1 feature ranks over the $1-50 band are left as cached, and the Stage 1w room model is not refit. One duplicate among a day's thousands of names moves a percentile rank by less than 0.1 points. |
| 38 | E4 is not re-run. Its decision does not feed E7, and it costs 29 hours. |
| 39 | The basis-repair minors are fixed in this rebuild, except M4 (fails safe), M9 (erratum) and M11's edits to existing tests. |

### 25.6 Check 3's count, corrected after it fired (2026-11-01; committed before E-1 is re-run)

- **What happened.**
  - Step 1 of §25.4 came out exactly as §25.3 expected:
    - 3 minute files differ: CXAI 2023-03-15, GRML 2024-07-05 and SMTK 2025-04-07;
    - 0 pre-market files differ;
    - 4 provenance rows change;
    - each root now has a manifest.
  - E-1 at poolpoc c38fe6e then stopped on §25.2 check 3's count, before scoring. In 2017-2025 it
    removed 225 rows, not the scan's 178.
  - Check 1 held: no day-D twins are left.
  - Check 3's set equality held: the pre-open groups and the day-D twin groups are the same 251
    pairs in 2016-2025.
- **Why.** The scan (§25.1) counted twins among the rows of the old pool file.
  - That file holds only the days E-1 scores: days with at least 20 universe names
    (`evaluate.MIN_NAMES`).
  - E-1's universe also holds 835 days in 2017-2025 with fewer than 20 names. Scoring drops them,
    and none of their rows is in the old pool.
  - 47 of the 225 pairs fall on those days. They are the same securities the scan found:
    ANIX/ITUS, BIOS/OPCH, ECA/OVV, PGEN/XON, AIFC/ARCI, FLNA/PTIE, NEWM/TDAY, AIFC/JAN, KIND/NXDR
    and AXG/SWIN.
  - The other 178 are the scan's pairs.
  - This was found from universe keys and daily-cache fingerprints only. No label, score or
    result was read.
- **The correction.** Check 3's count is taken on the scan's own days.
  - The rows removed in 2017-2025 on days with at least 20 names before the dedupe must number
    exactly 178.
  - The rows removed on other days are reported beside it, not gated.
  - The rule, check 1, check 3's set equality and every decision rule are unchanged.
- **A consequence.** A day with exactly 20 names that holds a pair keeps 19 after the dedupe, so
  E-1 no longer scores it. This is part of the pool churn §25.5 predicts.
- **Who ruled.** The executor ruled this while running the cascade. The user can reverse it. No
  number of the re-run depends on how the check is worded, because the check only gates.

### 25.7 Amendment 12: E5b's S2 score changes in the re-run, and E5d and E6 keep score A (2026-11-01; committed before E5d is re-run)

- **What happened.**
  - The re-run reached E5d (poolpoc 11deec336929). E-1's checks passed, with 178 rows removed on
    scored days (§25.6). E2 carried P1, E3 add-one kept base + F5, and E5a kept H30 (2, 1) in
    both feature sets, so §25.4's stop rules did not fire.
  - **E5b's decision changed in S2.** y_H (the Huber regression on y_h) now beats A on E5a's
    top-decile net: +0.067 pp (CI +0.011..+0.123), Holm p 0.014. The old run had +0.045 pp
    (−0.010..+0.100), p 0.109. S1 still keeps A: y_H +0.028 pp, p 0.219. B is below A in both
    sets, as before.
  - E5c ran under E5b's winner, as §22.4 says (y_H in S2), and kept the barrier exit in both
    sets.
  - E5d then stopped on its own guard before reading anything: "E5b did not keep the EV". §22.5a
    was written on the premise that E5a-c kept score A in both feature sets.
- **What the program says.** §25.4 lists the decisions whose change stops the cascade (E2, E3,
  E5a) or is carried (E5d's exit, into E6). E5b's is in neither list. Downstream of E5b:
  - E5d's rule (§22.5a) is built on A. The model-driven arm's score is M1's EV, refit on the
    relabelled minutes, because the EV reads each minute's u_i and d_i. Its reference is "E5a's
    H30 (2, 1) column, which is E3's M1".
  - E6 (§24.1, approved by the user) takes S2's score by name: "E5a's `H30_ku2_kd1_s2_ev`". Its
    entry rule (EV > θ, θ in percent of notional) needs a calibrated expected return. y_H is
    "not calibrated, because only its ranking is used" (§22.3). It has no EV threshold, which is
    why E5b has no traded arms.
- **The ruling [Claude's; the user can reverse it].** E5d and E6 keep score A in both feature
  sets, as §22.5a, §24.1 and §25.4 wire them. E5b's S2 result is reported and not carried.
  - E5d runs as written. Its guard on E5b is lifted by an explicit flag, and the run records
    which feature set's E5b decision it did not carry.
  - E6 runs as written.
  - E7's S2 finalist ranks by A.
  - Why: carrying y_H would mean designing a new E6 for S2 (an entry threshold on an
    uncalibrated score) inside a re-run. The cascade exists to avoid that kind of forking path.
    The flip also comes with a 20.6% turnover of dev P1: 1,411 of the 6,850 test name-days in
    2020-2022 changed through the walk-forward refit. The score difference is marginal under
    either run.
  - Cost if wrong: S2's policy ranks by a score whose top decile is 0.067 pp a minute worse on
    dev than y_H's. Reversing this means building an E6 policy on y_H, about an hour of compute
    after the code. E-1 to E5c are unaffected either way.
- **§25.5's prediction 3** ("E5b keeps the EV") failed for S2. It is scored in §25b.

| # | Call |
|---|---|
| 40 | E5b's S2 change (y_H over A) is not carried into E5d or E6. Both stay on score A, because their rules and E6's entry threshold are defined on a calibrated EV. |

## 25b. Results, the dev re-run on one security per name-day (2026-11-01; poolpoc 11deec336929 for E-1 to E5c, 6a0620385fee for E5d and E6; clean)

> **Superseded by §25d.** This run's dedupe matched on split-adjusted volume and missed about
> half of the duplicates (§25c). Its numbers stand only as the record of that run.

### 25b.0 Runs and integrity

- **Code.** §25's five tasks and the final review's fix pass (poolpoc 1811aa8..c38fe6e), §25.6's
  count (11deec3), and amendment 12's flag (6a06203). The suite passed at each commit (1,998
  tests at 6a06203).
- **Where it ran.** Everything ran from the run4 worktree, one job at a time.
  - The rebase build ran at c38fe6e. rebase.py did not change after it, so its roots stand.
  - E-1 to E5c ran at 11deec3. E5d and E6 ran at 6a06203, reading E-1 to E5c from
    11deec336929 by explicit path. Only run_e5d.py differs between the two commits, and no cache
    key reads it.
- **Wall time.**

  | Step | Time | Step | Time |
  |---|---|---|---|
  | E-1 | 72 s | E3 leave-one-out | 3,551 s |
  | minute-twin scan | 13 s | basis, rescore | 34 s, 36 s |
  | E0 | 369 s | E5a | 5,067 s |
  | E1 | 286 s | E5b, E5c | 204 s, 19 s |
  | E2 | 1,826 s | E5d | 2,256 s |
  | E3 panel prebuild | 397 s | E6 | 22 s |
  | E3 add-one | 2,414 s | | |

  The peak was 5.8 GB, in E5a.
- **§25.2's checks, all passed.**
  1. No day-D twins are left among the kept rows.
  2. There are 0 minute twins among P1's 12,940 name-days (2017 to 2025-08-06), and no P1 day
     lacks a minute file.
  3. The pre-open groups equal the day-D twin groups: 251 = 251. The rows removed on scored days
     in 2017-2025 number exactly 178 (§25.6), and 225 on all days.
     - By year: 2016 26, 2017 13, 2018 18, 2019 26, 2020 11, 2021 53, 2022 47, 2023 24, 2024
       25, 2025 8.
     - 9 of the removed rows had the day's minute bars, against a partner that also had them.
- **Each tranche's own checks passed.**
  - E3's panel equals E2's.
  - E5a's check 2.
  - E5d's integrity 1-3.
  - E6: all 18 E5 arms reproduce their logged series exactly, and each default equals its E5a
    arm.
- **MLflow** (data/research_poolpoc/mlruns):

  | Run | ID |
  |---|---|
  | E-1 | be31b252a3d942c28c2eb6fadfddaac8 |
  | E0 | 4f232356cb564e61b3600e4b518ec73b |
  | E1 | 0394ef20ed674d16baea9a8dd69c1081 |
  | E2 | 841988ed4dc0412e9a06dd0ab590630c |
  | E3 add-one | 3aece214077e46f4971d1ef54cda2384 |
  | E3 leave-one-out | babe5d1aa2cb47439b2161047fa5041f |
  | E5a | 722927c696c347688f7867465c20abdb |
  | E5b | 5e728a3d7d6e4c5cb8b18afa26882725 |
  | E5c | 9545b1afb7c3428a9a54d7f38b33c6e5 |
  | E5d | 0fcee793394945b68a3a35bac8b64329 |
  | E6 | 89ace9d9464248f0ad8d3e081d44c79f |

  - Each is tagged `supersedes` with the run it replaces (§19's six, §23's three, §23b's, and
    §24b's). Each replaced run is tagged `superseded_by`.
- **Trials.** The re-run logged 103 trial lines: E1 10, E2 3, E3 21, E5a 32, E5b 4, E5c 2, E5d 4
  and E6 27.
  - The replaced runs' lines get a `supersede` marker each (10 markers in all) and still count.
  - The program now has 261 trial lines.

### 25b.1 E-1 and the pool

| Contrast (share of random trades beating costs, top 10) | §19 | Re-run |
|---|---|---|
| model vs universe | 22.7% vs 16.4%, +6.3 (+6.0..+6.6) | 22.7% vs 16.4%, +6.2 (+5.9..+6.5) |
| vs naive | +1.4 (+1.1..+1.8) | +1.3 (+1.0..+1.7) |
| vs room model | +0.5 (+0.2..+0.8) | +0.4 (+0.1..+0.7) |
| vs base | +1.0 (+0.7..+1.3) | +0.9 (+0.6..+1.1) |

- **PASS.** 82,101 rows on 1,294 days (was 82,355 on 1,298).
  - Four days fell below 20 names after the dedupe and are no longer scored (§25.6's
    consequence).
  - The no-bars share fell from 0.48% to 0.27%: most removed rows were the bar-less copy of a
    security.
- **The pool turned over more than predicted.**
  - 1,508 of 7,270 dev P1 name-days left and 1,468 joined (20.7%). In 2020-2022 it was 1,411 of
    6,850 (20.6%).
  - By year, the share that left: 2017 13%, 2018 36%, 2019 17%, 2020 23%, 2021 20%, 2022 18%.
  - In 2018, 40 of the 54 departures are the four days no longer scored: 2018-10-26, 10-31,
    11-14 and 12-07.
  - The cause is the walk-forward pool model, as in §19. The removed rows were training rows
    for every later fit, mostly with zero labels, and the dedupe also changes the pre-market
    features' within-day ranks.
  - The within-day rank correlation of old and new scores is 0.94 (median; §19's was 0.977).
    At a top-10 cut that replaces a fifth of P1.
  - §25.5 prediction 1 (5-15%) **failed**.

### 25b.2 E0 to E3

| | §19 | Re-run |
|---|---|---|
| E0: C1 / C2 / oracle (%/day) | −1.271 / −1.948 / +2.83 | −1.187 / −1.801 / +2.81 |
| E0: H30 (2,1) TP share vs p* | 2.7% vs 40.4% | 2.7% vs 40.1% |
| E1: M1 top-decile lift | +0.302 pp (+0.210..+0.383) | +0.360 pp (+0.272..+0.441) |
| E1: M1's lift minus the cost-only score's | +0.074 (−0.015..+0.165) | +0.154 (+0.065..+0.245) |
| E1: M1 / C1 / C4 (%/day) | −0.675 / −0.568 / −1.095 | −0.621 / −0.557 / −1.007 |
| E2: raw lift P1 / P2 / P3 | +0.744 / +0.710 / +0.688 | +0.739 / +0.693 / +0.713 |
| E2: decision | P1 | P1 |
| E3: +F5 vs base | +0.101 pp, Holm p 0.003 | +0.077 pp, Holm p 0.014 |
| E3: decision | base + F5 | base + F5 |
| E3: the all-families model's gross part | +0.153 pp (+0.075..+0.228) | +0.156 pp (+0.064..+0.234) |
| E3: traded base / all / +F5 (%/day) | −0.150 / −0.162 / −0.121 | −0.115 / −0.147 / −0.127 |

- **E0** holds 3 of 3 predictions, as before.
- **E1** holds 2 of 3, as before: prediction 1 fails. M1's lift now beats the cost-only score
  with CI lo > 0 (+0.154 pp). Its lift still splits into a negative gross part and a positive
  cost part: −0.194 + 0.554 pp.
- **E2** carries P1 on raw lift (amendment 3), as before.
- **E3's add-one** keeps F5 alone. Leave-one-out finds no family the all-families model needs
  (every Holm p ≥ 0.357), and the Δ log-loss guard flags none.
- **Every traded arm of E1-E3 loses**, as before.

### 25b.3 E5a to E5c

- **E5a.** No set qualifies in either feature set, so H30 (2, 1) stays.
  - The default top-decile net is −0.923 pp in S1 and −0.850 pp in S2 (was −0.926 and −0.862).
  - The only arm whose daily CI reaches 0 is now H15 (1, 1) in S2: −0.042%/day (hi +0.004).
    It was H15 (1, 1) in S1.
- **E5b.** S1 keeps A. **S2's decision changed to y_H** (+0.067 pp, Holm p 0.014; it was
  +0.045, p 0.109). Amendment 12 (§25.7) keeps A downstream.
- **E5c** keeps the barrier exit in both feature sets. It ran under E5b's winner: in S2 the
  time-only exit is +0.033 pp, p 0.069.

### 25b.4 E5d (run with `--carry-a`; S2's y_H is recorded as not carried)

| | Barrier top-decile net | Model-driven | Difference | p | Decision | Traded arm (%/day) |
|---|---|---|---|---|---|---|
| S1 | −0.923 pp | −0.887 pp | +0.036 (−0.013..+0.093) | 0.148 | **barrier** (was model exit, p 0.049) | −0.038 (hi −0.000), 0.58 trades a day |
| S2 | −0.850 pp | −0.795 pp | +0.055 (−0.013..+0.136) | 0.127 | barrier | −0.007 (hi +0.026), 0.82 trades a day |

- **S1's model-placed exit no longer replaces the barrier exit.** §23b's p = 0.049 was a coin
  flip, as §25.5 said.
- By §25.4, E6 now uses the barrier exit in both feature sets. `run_e6.sources` read E5d's
  decision and chose E5a's frames.
- The placement is as before: the median u_i is 2.3-2.4% against X*'s 5.9%, and the top-decile
  TP share is 0.33-0.35 against 0.03-0.04.

### 25b.5 E6 (both feature sets on the barrier exit)

**S1.** The default (E5a's H30 (2, 1) S1 arm, no longer E5d's) makes −0.127%/day (CI
−0.197..−0.069) at 1.46 trades a day, trading on 235 days. Its net per dollar is −1.00%
(−1.50..−0.54).

The columns are as in §24b.1:
- "vs default": the paired daily difference, in %/day.
- "Net per $ minus default": §24.10's paired CI, in pp.

| Level | vs default | Holm p | Net per $ (95% CI) | Net per $ minus default | Days ≠ | Trades/day | Kept | Strict |
|---|---|---|---|---|---|---|---|---|
| EV > 0.25% | +0.047 (+0.018..+0.075) | 0.0015 | −1.01% (−1.76..−0.37) | −0.01 (−0.36..+0.33) | 188 | 0.93 | no | no |
| EV > 0.5% | +0.081 (+0.045..+0.118) | 0.0015 | −0.89% (−2.03..+0.10) | +0.11 (−0.67..+0.80) | 212 | 0.61 | **kept** | no |
| EV > 1.0% | +0.110 (+0.069..+0.153) | 0.0015 | −0.63% (−2.52..+0.93) | +0.36 (−1.11..+1.69) | 227 | 0.32 | **kept** | no |
| 3 slots | +0.011 (−0.004..+0.027) | 0.340 | −1.01% (−1.56..−0.52) | −0.01 (−0.14..+0.12) | 26 | 1.29 | no | no |
| 5 slots | −0.000 (−0.011..+0.012) | 0.874 | −1.02% (−1.53..−0.53) | −0.03 (−0.12..+0.08) | 13 | 1.41 | no | no |
| price ≥ $2 | +0.005 (−0.014..+0.023) | 0.559 | −1.05% (−1.59..−0.61) | −0.06 (−0.21..+0.09) | 39 | 1.32 | no | no |
| price ≥ $3 | +0.031 (+0.004..+0.060) | 0.050 | −0.93% (−1.49..−0.44) | +0.07 (−0.18..+0.30) | 60 | 1.16 | no | no |
| no entry before 09:45 | +0.026 (+0.003..+0.050) | 0.028 | −0.85% (−1.35..−0.41) | +0.14 (−0.05..+0.33) | 73 | 1.36 | **kept** | no |
| 15-min cooldown | +0.017 (−0.003..+0.040) | 0.120 | −0.98% (−1.55..−0.48) | +0.01 (−0.17..+0.20) | 64 | 1.28 | no | no |
| 30-min cooldown | +0.033 (+0.009..+0.063) | 0.004 | −0.94% (−1.44..−0.44) | +0.06 (−0.14..+0.28) | 74 | 1.15 | **kept** | no |
| B gate (b_hat > 0.506) | +0.092 (+0.050..+0.140) | 0.0005 | −0.69% (−1.34..−0.09) | +0.30 (−0.13..+0.76) | 213 | 0.59 | **kept** | no |
| price < $10 (exploratory) | +0.029 (+0.009..+0.051) | p 0.006 | −0.95% (−1.45..−0.47) | +0.05 (−0.12..+0.21) | 48 | 1.22 | never | never |

**S2.** The default (E5a's H30 (2, 1) S2 arm) makes −0.147%/day (CI −0.229..−0.073) at 1.74
trades a day, trading on 325 days. Its net per dollar is −0.91% (−1.43..−0.43).

| Level | vs default | Holm p | Net per $ (95% CI) | Net per $ minus default | Days ≠ | Trades/day | Kept | Strict |
|---|---|---|---|---|---|---|---|---|
| EV > 0.25% | +0.075 (+0.043..+0.112) | 0.0015 | −0.69% (−1.36..−0.04) | +0.23 (−0.07..+0.54) | 264 | 1.14 | **kept** | no |
| EV > 0.5% | +0.086 (+0.047..+0.128) | 0.0015 | −0.85% (−1.81..+0.08) | +0.07 (−0.51..+0.64) | 294 | 0.78 | **kept** | no |
| EV > 1.0% | +0.118 (+0.073..+0.166) | 0.0015 | −0.72% (−2.17..+0.81) | +0.20 (−0.94..+1.36) | 314 | 0.44 | **kept** | no |
| 3 slots | −0.010 (−0.021..+0.000) | 0.088 | −1.01% (−1.52..−0.52) | −0.09 (−0.18..−0.02) | 21 | 1.67 | no | no |
| 5 slots | −0.002 (−0.004..−0.000) | 0.088 | −0.94% (−1.46..−0.44) | −0.02 (−0.04..−0.00) | 5 | 1.72 | no | no |
| price ≥ $2 | −0.017 (−0.043..+0.005) | 0.310 | −1.08% (−1.55..−0.66) | −0.17 (−0.35..−0.03) | 40 | 1.64 | no | no |
| price ≥ $3 | −0.013 (−0.046..+0.019) | 0.455 | −1.15% (−1.66..−0.69) | −0.24 (−0.46..−0.05) | 76 | 1.49 | no | no |
| no entry before 09:45 | +0.062 (+0.033..+0.094) | 0.0005 | −0.57% (−1.08..−0.07) | **+0.35 (+0.15..+0.57)** | 123 | 1.61 | **kept** | **kept** |
| 15-min cooldown | +0.001 (−0.037..+0.039) | 1.000 | −1.04% (−1.52..−0.55) | −0.13 (−0.37..+0.12) | 130 | 1.52 | no | no |
| 30-min cooldown | +0.009 (−0.030..+0.049) | 1.000 | −1.12% (−1.62..−0.63) | −0.20 (−0.46..+0.06) | 140 | 1.34 | no | no |
| B gate (b_hat > 0.475) | +0.075 (+0.028..+0.122) | 0.0010 | −0.92% (−1.64..−0.25) | −0.01 (−0.46..+0.42) | 274 | 0.87 | no | no |
| price < $10 (exploratory) | +0.041 (+0.015..+0.071) | p 0.002 | −0.78% (−1.24..−0.31) | +0.14 (−0.01..+0.29) | 60 | 1.49 | never | never |

- **The B gate is no longer kept in S2.** Its net per dollar is now below the default's
  (−0.92% against −0.91%). In §24b it was the larger of S2's two strictly kept levels (+0.60
  pp).
- **Skipping 09:30-09:44 is again the one level that raises S2's net per dollar on its own
  CI**: +0.35 pp (+0.15..+0.57) over 123 differing days. It is the only strictly kept level in
  E6.
- **The guard problem is now in both feature sets.** Every EV threshold raises the daily mean
  strongly by trading less, and none moves net per dollar on its own CI.

| Feature set | Policy | %/day (95% CI) | Trades/day | Days traded | Net per $ (95% CI) | vs C1 (%/day) | vs C2 | End check |
|---|---|---|---|---|---|---|---|---|
| S1 | binding: EV > 1.0%, no entry before 09:45, 30-min cooldown, B gate | +0.004 (−0.015..+0.021) | 0.16 | 33 | +0.36% (−1.24..+1.85) | +0.023 (+0.001..+0.045), p 0.04 | +1.85 (+1.54..+2.16) | passes |
| S1 | strict: the default | −0.127 (−0.197..−0.069) | 1.46 | 235 | −1.00% (−1.50..−0.54) | +0.065 (+0.004..+0.125), p 0.039 | +1.72 | fails |
| S2 | binding: EV > 1.0%, no entry before 09:45 | −0.020 (−0.084..+0.036) | 0.41 | 117 | −0.53% (−2.17..+1.10) | +0.053 (−0.008..+0.109), p 0.097 | +1.83 | passes |
| S2 | strict: no entry before 09:45 | −0.085 (−0.166..−0.014) | 1.61 | 305 | −0.57% (−1.08..−0.07) | +0.155 (+0.079..+0.233) | +1.76 | fails (not read) |

- **No E6 policy makes money with confidence.** The best, S1's binding combined policy, is
  +0.004%/day on 33 trading days. It is four filters chosen on the same days they are judged on.
- **Neither strict policy has a dev CI that reaches 0.** S1's strict policy is its default:
  −0.127%/day (hi −0.069). S2's is −0.085%/day (hi −0.014).
- Every policy beats both controls. C1 loses −0.018 to −0.240%/day and C2 −1.849%/day.

### 25b.6 The end-of-E6 check (§24.5, unchanged)

- **7 of the 42 arms pass, so the check does not fire.** It fired for none in §24b either: 13
  passed then.
- The passing arms: E5a H15 (1, 1) S2; E5d S2; S1 EV > 0.5% and > 1.0%; S1's binding combined
  policy; S2 EV > 1.0%; S2's binding combined policy.
- Every one passes by being uncertain. Only S1's binding combined policy has a mean above 0
  (+0.004%/day), and it trades on 33 days.
- **Neither strict policy would pass.** Both have a daily CI upper bound below 0.

### 25b.7 Predictions (§25.5)

| # | Prediction | Result |
|---|---|---|
| 1 | 5-15% of dev P1 name-days change | **failed** (20.7%) |
| 2 | E-1 passes; E2 carries P1; E3 keeps base + F5, maybe more; the all-families model keeps a gross part with CI lo > 0 | held |
| 3 | E5a: no set qualifies; E5b keeps A; E5c keeps the barrier exit | **failed** in part (E5b chose y_H in S2) |
| 4 | E5d: S2's exit is not replaced; S1's is | **failed** in part (S1's is not replaced) |
| 5 | E6 strict: S1 keeps nothing; S2 keeps no entry before 09:45 and the B gate; no CI lo > 0; the end check does not fire | **failed** in part (S2 keeps the 09:45 rule but not the B gate) |
| 6 | Every traded arm of E1-E5 still loses | held on the means. Two arms' CIs reach 0 (E5a H15 (1, 1) S2, E5d S2), as in §23 |

On E6's own predictions (§24.7): 2 of 7 held on the binding reading and 6 of 7 on the strict
one. The one strict failure is "the end check fires".

### 25b.8 Reading

- **The duplicate fix moved more than the duplicates.**
  - It removed 178 scored rows (0.2%). Through the walk-forward pool model it replaced a fifth
    of dev P1, and every later tranche re-ran on that pool.
  - Three decisions flipped: E5b in S2, E5d in S1, and E6's B gate in S2. Each was near its
    threshold before: p 0.109, p 0.049, and a per-dollar CI lo of +0.08 pp.
  - The program's decisions near a threshold are not stable under a 20% change of the pool.
- **What survives both runs:**
  - Skipping the 09:30-09:44 entries in S2. It raises net per dollar on a paired CI that
    clears 0 in both runs (+0.39 and +0.35 pp).
  - The ranking skill that beats the mechanical EV (E3's gross part, about +0.15 pp).
  - Every policy beats random timing and open-and-hold.
  - No policy makes money.
- **The strict reading leaves two finalists that lose on dev with their whole CI below 0:**
  - S1, the default: −0.127%/day (hi −0.069).
  - S2, no entry before 09:45: −0.085%/day (hi −0.014).
  - §26's E7 would run them once on 2023-2025.

### 25b.9 What follows

- **E7 is not started.** The user chose the strict reading and E7 ("then go strict") while
  §24b's strict S2 had a dev CI reaching 0 (−0.013%/day, hi +0.040). After the re-run, neither
  strict finalist's dev CI reaches 0.
  - E7 runs each finalist once and uses up the program's confirmation years.
  - So whether to run it on these two is put back to the user, with E7's pre-registration
    drafted but not committed.
- **The choices:**
  1. Run E7 on the two strict finalists as planned. Expected result: both fail.
  2. Stop the long-only track here, per the strict reading, and keep 2023-2025 unspent.
  3. Something else, such as the binding S1 policy as a third finalist. That would reverse
     "go strict".
- E4 stays as measured on §19's pool (§25.4).

| # | Call |
|---|---|
| 41 | E5d and E6 ran at 6a06203 (amendment 12's flag), reading E-1 to E5c from 11deec336929 by path. Only run_e5d.py differs, and no cache key reads it. |
| 42 | E7 is not started. Its premise changed: the strict finalists' dev CIs are now wholly below 0. The confirmation years can be spent only once, so the user decides. |

## 25c. Amendment 13: the identity rule reads unadjusted volume, as §25.2 says (2026-11-01; committed before the fix and the second re-run)

- **What was found.** While writing §25b's note on the final review's minor M-d, a re-count
  showed that M-d is not minor.
  - The code's rule (identity.py) matched on the daily cache's `volume`, which is
    split-adjusted. §25.2 says unadjusted. The prices it matched were unadjusted (`*_unadj`),
    and trade counts are never adjusted.
  - Each symbol's adjusted volume carries that symbol's own later split factors. A security
    served under two symbols matched only when both symbols had the same corporate actions
    after the day. Detection therefore depended on events after D.
  - Example: on 2021-06-01, XL and SPRU have the same unadjusted open (6.97) and close (7.25),
    the same unadjusted volume (7,837,026) and the same trade count (27,648). SPRU's adjusted
    volume is 979,628, which is 1/8 of that, from a reverse split after the date. On
    2020-03-18, RUBI's adjusted volume is 40 against MGNI's 2,378,082.
- **What the rule finds on unadjusted volume** (daily cache only, no label or result read):
  - The cache has 44,710 same-day groups with identical unadjusted OHLC, unadjusted volume and
    trade count, from 114 symbol sets. §25.1 found 25,545 groups from 70 sets.
  - §19's pool holds 362 pairs (724 rows): 262 in dev and 100 in confirmation. §25.1 found
    178 (124 and 54). Both symbols sit in P1 on 23 dev days and 31 confirmation days. §25.1
    found 13 and 16.
  - The pre-open rule (two prior sessions) on unadjusted volume finds 184 groups that §25b's
    universe still holds, 138 of them in dev. With the 178 already removed, that makes the
    scan's 362. Check 3 tests the match pair by pair.
  - The dev rows still held come from four sets: SPRU/XL (48 days), DGLY/KUST (40), MGNI/RUBI
    (40) and HIND/RSLS (10). On 19 of those days at least one copy is in P1, and on 9 both
    are.
  - Two rows of §25.1's scan do not match on unadjusted volume (NRXP/TBRGU on 2019-01-16, one
    trade each). Neither is in the pool.
  - Matching on prices and trade count alone, with no volume, finds the same members in the
    scored universe. It adds 80 members elsewhere in the cache, all rows with no unadjusted
    volume.
- **Why the checks passed.** Checks 1 and 3 and §25.1's scan read the same adjusted column, so
  they shared the blind spot. The final review named exactly this as M-d. I graded it minor on
  the evidence that today's numbers reproduced the scan, and that evidence was circular. Check
  2 reads minute bars and could not catch these pairs: on the doubled P1 days one copy has no
  minute bars.
- **The fix.**
  - identity's reader gives the rule the unadjusted share volume as its `volume` (the daily
    cache's `volume_unadj`). The rule's functions are unchanged, so their committed tests still
    describe them. A new test pins the reader.
  - A row with no unadjusted volume cannot be matched (volume > 0 fails). None is in the scored
    universe's pairs.
  - CODE_VERSION becomes identity-v2, so the cached groups are rebuilt.
  - Check 3's gate becomes 362.
  - Checks 1 and 3 still read the rule's volume column. The independent evidence is the
    prices-and-trade-count match above, which finds the same scored members.
- **The re-run.** §25.4's cascade runs again from the run4 worktree, with the same stop rules,
  the same E5d flag as amendment 12 (§25.7), and E7 still not started.
  - §25b is superseded. Its results stand as the record of a run that missed half the
    duplicates, and the new results go in §25d.
  - The MLflow runs and trial lines it replaces get `superseded_by` tags and `supersede`
    markers, as in §25b.0. The program gains about 103 more trial lines, which raises E7's DSR
    trial count.
- **Predictions (before the re-run):**
  1. E-1 removes exactly 362 rows on scored 2017+ days, and checks 1 to 3 pass. The Stage 1
     label still passes.
  2. 10% to 25% of dev P1 name-days change against §25b's pool.
  3. E2 keeps P1, E3 keeps base + F5, and E5a keeps H30 (2, 1) in both feature sets, so the
     cascade does not stop.
  4. Both strict E6 policies have a daily CI upper bound below 0.
  5. The end-of-E6 check (§24.5) passes on at least one arm.
- **Cost if wrong:** about 5 hours of compute and the extra trial lines, if the user then stops
  the long-only track. Leaving it means E7, and any later use of the pool, runs on a universe
  that still holds 138 dev and 46 confirmation duplicates.

| # | Call |
|---|---|
| 43 | The identity rule reads unadjusted share volume, as §25.2 says. §25b is superseded, and §25.4's cascade re-runs. |
| 44 | The fix is in the reader, which supplies `volume_unadj` as the rule's volume. The rule's functions and their committed tests stay unchanged. |

## 25d. §25c's cascade stops at E5a, and stays there (2026-11-05)

- **What ran.** §25c's re-run ran E-1 to E5a from the run4 worktree at c7878b02c333, then stopped
  at the E5a gate as §25.4 requires (exit 3, by design).
  - S1 keeps H30 (2, 1).
  - S2 now picks H30 (1, 1): +0.060 pp against the default (CI +0.017..+0.104, Holm p 0.042).
  - The best traded arm is H15 (1, 1) on S2, at −0.021 %/day (CI hi +0.015). Every E5a traded
    arm loses.
- **What follows.**
  - On 2026-11-04 the user chose to freeze the cascade here. E5b to E6 do not run on c7878, and
    §25.4's amendment before E5b is not written.
  - §25b's E6 results are not re-measured on the corrected universe. They stand as the record of
    a run that missed half the duplicates.
- **What uses it.** c7878's E5a predictions (`c7878b02c333/e5a/predictions.parquet`) are the
  input to Stage 3 (`2026-11-05-stage3-algo-program.md`), which takes its baselines from them.
- **The trial count.**
  - The ledger holds 327 trial lines and 10 supersede markers, so E7's DSR count starts Stage 3 at
    N = 327 + 21 = 348.
  - E7 has not started. Confirmation now belongs to Stage 3's finalists, and runs only when the
    user says so.
- **Interpreter.** E-1 to E5a ran on the system Python 3.13. From Stage 3 on, runs use the ML
  venv, with mlflow 3.16.0 installed on 2026-11-04 and numpy, pandas and scikit-learn unchanged.
  B0's fidelity check (Stage 3 §3) pins the venv to c7878's logged series.

| # | Call |
|---|---|
| 45 | §25c's cascade stays frozen at E5a (the user's answer, 2026-11-04). Stage 3 builds on c7878's E5a predictions. |
