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
