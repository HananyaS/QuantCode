# Pool-then-trade proof of concept: design (draft for user approval)

Status: DRAFT, 2026-10-27. Nothing is built until the user approves this document.
Calls marked **[my call]** were made by Claude, not by the user; each can be overridden.

## 1. Intent

The user's direction (2026-10-27): for a proof of concept, first build the **pooling** stage, which
each morning selects stocks with a large room for intraday trading, using labels built from
minute data aggregated to the day. Then build a model that **sets one entry and one exit per
selected stock per day**, trained on the pooled subset, so that what it learns is conditioned on
the pool, as it would be in production. It must be **much faster** than R16 (hours, not days),
and it needs a **strong evaluation system**.

Why: R16 (still running, ~2 days) ranks the pool by gap and pre-market volume, not by predicted
room; trains its model on all names, not the pool; and makes per-minute "up in h minutes?" calls
rather than a daily entry/exit plan. This POC tests the user's own decomposition directly.

## 2. Goals, non-goals, anti-goals

Goals
- G1. Stage 1: a pre-open pool whose names have measurably more intraday room than the universe
  and than a naive "yesterday was volatile" pick.
- G2. Stage 2: inside the pool, a daily entry + exit policy whose **net, compounded** account
  return beats its controls and SPY, with a day-clustered CI.
- G3. One evaluation harness that both stages and every later idea plug into, giving the same
  report and charts every time.
- G4. Full development run under **2 hours** on this machine (16 GB, ≤ 4 workers, ≤ 3 GB RAM
  while R16 runs).

Non-goals
- No live feed, no broker, no paper trading in this POC. No downloads.
- No change to R16, its prereg, its data or its running process.

Anti-goals (a "success" that would still be a failure)
- A pool that only rediscovers "volatile stocks stay volatile" and is called an edge. Stage 1 is
  judged against that naive baseline.
- A Stage-2 profit that comes from look-ahead (adjusted prices in a filter, the day's own high/low
  in a feature, a label leaking into an entry rule), or that disappears after costs.
- Reading the sealed holdout (dates ≥ 2025-08-07).

## 3. Constraints (inherited, binding)

- Data: cached only. Daily bars `data/research_penny/daily/bars_YYYY.parquet` (whole market,
  ~3,900 symbols/day, raw + unadjusted columns). Minute bars `data/research_penny/minute/dt=*/`
  (~420 symbols/day; the names earlier rounds requested, mostly gap days). Pre-market bars
  `premarket/dt=*/` (same requested set). Quote samples `quotes_r4/`, `quotes_rev/`.
- Holdout ≥ 2025-08-07 stays sealed; a guard test asserts no touched date ≥ it.
- Unadjusted prices in every filter and screen; adjusted only for returns. Features pass the
  joint rescale test (price × c, volume ÷ c).
- No future data: every feature for day D uses data available at its decision time.
- Compare against SPY (cached span ends 2024-12-30; later dev days are marked missing).
- Tests never edited to pass; full suite passes before each commit. Python 3.13 at the full path.
- Branch `research/poolpoc` cut from `research/intraday`, own worktree. Never merged to main,
  never pushed.

## 4. The data problem, and how the design handles it

Minute bars exist only for ~420 names a day, chosen by earlier rounds' gap pools. So:
- **Stage 1 runs on the whole market using daily bars**, where the day's room can be measured
  for every name. [my call]
- **Stage 2 can only trade pool names that also have minute bars.** The harness reports, per
  day, how many pool names are in the minute cache. Stage-2 results are labelled
  "pool ∩ minute cache". If that intersection is small (median < 5 names/day), Stage 2 is
  reported as conditional, and getting minute bars for the full pool becomes a user decision
  about the data budget (downloads are forbidden now).
- Stage-1 labels that need the order of moves inside the day (§5.2) are computed on the covered
  subset only and used as a check that the daily label tracks what is actually tradeable.

## 5. Stage 1: pooling

### 5.1 Universe and timing
Every day D: symbols with prior unadjusted close in [$1, $10] and 20-day median dollar volume
≥ $1M (R16's filter), excluding split days. Decision time 09:25 ET. [my call: same filter as
R16 so results are comparable]

### 5.2 Labels ("room")
- **Primary, all names (daily bars):** `room_D = (high_D − low_D) / open_D`, unadjusted.
- **Tradeable checks, covered names (minute bars, 09:35–15:55):**
  - `best_long_D` = max over t1 < t2 of close(t2)/close(t1) − 1 (best buy-low-then-sell-high);
  - `up_room_D` = max high after 09:35 / open(09:35) − 1;
  - `down_room_D` = 1 − min low after 09:35 / open(09:35).
- Reported: the correlation of `room_D` with `best_long_D` on covered days. If it is weak, the
  pool is retrained on `best_long_D` over covered names. [my call: start with the label every
  name has]

### 5.3 Features (all known at 09:25)
From daily bars up to D−1: ATR% over 5/20 days, realized volatility 5/20 days, mean `room` over
5/20 days, last day's return and range, distance from the 20-day high/low, dollar-volume ratio
(5d/20d), price level bucket, day-of-week. Pre-market gap and pre-market dollar volume only where
pre-market bars exist, with a missing-flag. Cross-sectional ranks per day.

### 5.4 Model and pick
LightGBM regression on rank(room) per day (fast, deterministic with fixed seed). Pick top N = 10
names per day. [my call: N = 10, like R16]

### 5.5 Stage-1 pass bar (fixed before any run)
Out-of-fold, on development days:
- mean `room` of the picked 10 beats the **naive baseline** (top 10 by mean `room` over the last
  20 days) with a day-clustered CI above zero, and beats the universe mean;
- on covered days, picked names' mean `best_long` also beats the naive baseline's.

## 6. Stage 2: daily entry and exit inside the pool

### 6.1 What it decides
For each pool name on day D, a **plan**: whether to trade, the entry trigger, and the exit levels
(take-profit, stop-loss), plus a 15:55 time stop. Long only. [my call: shorting low-priced names
has borrow problems that cached data cannot measure]

### 6.2 Policy family (small, fixed grid)
- Entry: at minute t ≥ 09:35, a classifier reads the last k = 15 bars and fires once per name per
  day when P(reaching TP before SL) > threshold. Variants: threshold ∈ {0.5, 0.6}.
- Exit levels from the **pool-conditional distribution of remaining room**: quantile regressions
  predict, at entry, the q-quantile of up-move and down-move until 15:55; TP = up quantile q_tp,
  SL = down quantile q_sl. Variants: (q_tp, q_sl) ∈ {(0.5, 0.5), (0.4, 0.7)}.
- Grid size 2 × 2 = 4 cells, plus the controls in §7.3. The trial count is recorded and
  Deflated Sharpe uses it.
- Training rows: pool names only (out-of-fold pool picks on training dates), so the model learns
  the pool's distribution.

### 6.3 Fills and costs (same rules as R16's engine)
One-bar latency; entry at the next bar's open; SL filled at the next bar's open, gaps included,
never at the stop level; no fills in halts or missing minutes; round-trip cost from the refit cost
model, with size capped by the quoted size. Sizing: equal weight, at most 5 open positions,
$2,000 per position.

## 7. Evaluation system (the core deliverable)

One harness, `research/poolpoc/eval/`, used by both stages and all later ideas.

7.1 **Splits.** Walk-forward by date: 6 expanding folds over 2016-02 to 2025-08-06, with a
5-day embargo between train and test. The holdout is never loaded; a guard asserts it.

7.2 **Metrics.**
- Stage 1: per-day rank IC of score vs `room`; top-10 lift over the universe and over the naive
  baseline; precision@10 for the top `room` decile; all by year.
- Stage 2: net compounded account curve; total and per-year return; Sharpe; max drawdown; hit
  rate; average win/loss; trades per day; turnover; cost share of gross; exposure.
- Uncertainty: day-clustered bootstrap CIs (2,000 resamples) on every headline number.

7.3 **Controls and baselines** (each must be beaten):
- Stage 1: universe average; naive volatility persistence; random pick of 10.
- Stage 2, same pool and same days: random entry minute with the same exit rule; buy at 09:35 and
  hold to 15:55; fixed TP/SL (+3%/−3%) with the same entries; the same policy on non-pool
  covered names (does pooling help?); SPY buy-and-hold over the same days.

7.4 **Leak and sanity tests** (run on every report):
- Shuffled labels within day → IC and account fall to the null band.
- Synthetic martingale minute paths with the real volatility → no edge.
- Joint price/volume rescale → identical outputs.
- Point-in-time check: every feature for D is recomputed from data truncated at its decision time
  and must match.
- Determinism: two runs give identical output hashes.
- Holdout guard: no date ≥ 2025-08-07 touched.

7.5 **Report.** Every run writes one HTML report with charts (account vs controls vs SPY, lift
by year, IC over time, cost breakdown, biggest trades table), a plain-words verdict line, and an
MLflow run. Results are added to the morning report artifact.

7.6 **Pre-declared bar.** The §5.5 and §7 thresholds above are frozen in the first commit, before
any result is computed. A cell passes only if its net compounded account CI excludes zero and it
beats every Stage-2 control, after Holm correction across the 4 cells.

## 8. Speed

- Vectorized pandas/numpy over a precomputed panel; no per-event engine replay in the research
  loop. Built once and cached per stage as parquet with input hashes.
- The event-bus engine replays the single best cell on a 50-day sample, as a parity check that the
  vectorized simulator and the engine agree on fills.
- Budget: Stage-1 panel + model < 20 min; Stage-2 full grid with controls and nulls < 90 min, at
  ≤ 3 GB RAM while R16 runs. A smoke mode (1 year, 1 fold) runs in < 5 min and is in the tests.

## 8b. Settled after the builder check [my calls]

- **Stage-1 scores for Stage 2 (no leakage):** Stage 1 is walk-forward by year: the model scoring
  year Y is trained only on years < Y (first scored year 2017). Stage 2 trains and tests only on
  dates that have such scores, with its own 6 expanding folds and 5-day embargo.
- **Pass thresholds:**
  - Stage 1: the day-clustered 95% CI of (picked − naive) mean `room` has its lower bound > 0,
    and likewise for (picked − universe).
  - Stage 2: for each cell and each control, the paired day-clustered CI of (cell − control) net
    daily return has its lower bound > 0, after Holm correction across all cell × control
    contrasts (4 cells × 5 controls = 20). The account's own CI must also exclude zero.
- **Entry target and TP/SL, without circularity:**
  1. The quantile models are fitted first. They give each candidate minute its own TP and SL.
  2. The classifier's target = 1 if TP is hit before SL under those levels, with the §6.3 fill
     rules.
  3. A bar that touches both counts as SL first (conservative).
  4. If neither is hit by 15:55, the trade exits at the time stop, and the target = 1 only if the
     net return > 0.
- **Costs:** reuse the R16 refit `CostModel` (`research/penny/etl/quotes.py`, median round trip
  55.4 bps). Names without quote samples use its prediction from price, ADV and volatility. The
  size cap is the shares at the best quote where a quote sample exists; otherwise 5% of the entry
  bar's dollar volume.
- **SPY:** compared on dev days that have SPY (through 2024-12-30). The 2025 dev days are reported
  separately, never filled.
- **Parity:** the event-bus engine in `research/intraday` (`run_replay`) replays the best cell on
  50 sampled days. If the adapter costs more than half a day, parity is deferred and the report
  says so.
- **Budget:** the 2-hour budget includes the bootstraps (2,000 resamples) and every null. The
  martingale null runs on a one-year subsample; the shuffle null uses one seed per cell.

## 9. Left to the builder

File layout inside `research/poolpoc/`, LightGBM hyperparameters (fixed defaults, not tuned),
chart styling, the exact cost-model call.

## 10. Open for the user

1. The Stage-1 label: day range for all names (current call), or the best long trade on covered
   names only.
2. Whether R16 keeps running alongside (current call: yes; the POC is sized to fit beside it).
3. Long only (current call), or also short.

## 11. Round 2 of the POC: $10–$50 names and real commissions (pre-declared 2026-10-27, before any result)

**The user's decision (2026-10-27):** "you can add the 10-50 to the pool. regarding the commisions -
cent per stock, 2.5$ min." Everything below is fixed before any Round-2 number exists.

### 11.1 Universe ("wide band")
Prior unadjusted close in **[$1, $50]**, 20-day median dollar volume ≥ $1M, split days excluded:
the §5.1 filter with the price cap raised from $10 to $50. Price-bucket edges become
[2, 3, 5, 10, 20]. Everything else in Stage 1 (features, label = room, LightGBM on rank(room),
yearly walk-forward, top 10) is unchanged.

### 11.2 Stage 1w (1 trial)
The `full / all / room` run on the wide band. Same pass bar as §8b: the day-clustered 95% CI
lower bounds of (picked − naive) and (picked − universe) mean room are both > 0.

### 11.3 Stage 2c: Stage 2b's daily bracket plan with commissions (8 trials)
- **Two pools**, each with Stage 2b's 4 plan cells (entry ∈ {open, dip} × q ∈ {0.3, 0.5}):
  - **P1**: top 10 of the wide universe by the Stage-1w score.
  - **P2**: top 10 by the same score among names with prior close ≥ $10 (the cheaper-to-trade side).
  The plan models are trained per pool, on that pool's earlier years only, as in Stage 2b.
- **Account [my call, A17]:** $25,000 with 10 × $2,500 positions. Below $25,000 a US margin account
  is held to the pattern-day-trader limit (3 day trades per 5 days), so a 10-name daily strategy
  needs at least that much.
- **Commission (the user's figure):** each order (entry and exit) pays max($2.50, $0.01 × shares),
  with shares = notional / unadjusted price. No cap, no exchange or regulatory fees.
- **Spread, impact, depth:**
  - **Under $10:** Stage 2b's rule, unchanged (the refit book where it has inputs, else max(55.4 bps,
    the model)).
  - **$10 and up:** the refit book where it has inputs. Otherwise **[my call, A18]** a half-spread
    predicted from a log-linear fit of the refit book's 09:35 half-spread on log price, log dollar
    ADV and log 20-day volatility (R² 0.94 in-sample), floored at half a tick. The 09:35 (wider)
    spread is charged on both legs, plus the existing impact term σ·√(notional / ADV).
- **Controls, nulls, fills, bootstrap, 2× cost run, minute-order validation:** all as in Stage 2b,
  per pool. K1 (naive top 10 by room20) and K2 (random 10) are drawn from the same price range as
  their pool.
- **Pass bar:** as Stage 2b. The account CI lower bound is > 0, and the cell beats each of the 5
  controls, with Holm correction across 8 cells × 5 controls = 40 contrasts.
- **Trials:** 12 (Round 1) + 1 (Stage 1w) + 8 (Stage 2c) = **21**, the Deflated-Sharpe N.
- **Sensitivity, not a trial:** the same 8 cells on a $100,000 account (10 × $10,000), so that the
  $2.50 minimum weighs less.

### 11.4 Not re-run
Stage 2 (the minute entry model) only ever saw gap-day minute bars. Few $10–$50 names are in that
cache, so it is not re-run (A20).

### 11.5 Amendment after the code review, before any Stage-2c result
- The refit book has **no rows above $10**, so the $10+ spread model is pure extrapolation from
  cheaper names (A18, corrected). A sensitivity run is added, and it is not a trial: the **tick
  spread**. For $10+ names without refit inputs, the half-spread is half a tick, plus impact and
  commission. This is the most optimistic spread a plain order can get. Reading rule: a cell that
  passes only under the tick spread is reported as *inconclusive, spread-dependent*, not a PASS. A
  cell that fails even under the tick spread fails regardless of the spread model.
- The report gives, per pool, the commission bps, the other cost bps and the share of trades
  priced by the spread model.
- Minute-order validation is effectively empty for P2 (few $10+ names have minute bars), so P2's
  daily-fill pessimism is unmeasured (A22). The row count per pool is printed.

## 12. Stage 1s: a "share of random trades that beat costs" label (pre-declared 2026-10-27, before any result)

The user's call, after the random-trade analysis (933M random long trades on 18,488 sampled cached
stock-days): day range picks bigger wins *and* bigger losses, and the chance a random trade makes
any profit does not rise with it. The user chose this label: "go ahead with the share-beats-costs label".

### 12.1 Label
For a stock-day with at least 60 regular-session minute bars (09:30-15:59 ET, minute closes,
cached gap-day minute data): over every pair (buy bar i, later sell bar j) of that day,
`share_cost` = share of pairs with close_j / close_i - 1 > c, where c = 2% if the prior close
(unadjusted) is under $10, else 1% (Round-2 round-trip costs: ~1.8% and ~1.0%). Same-day ratios,
so the minute cache's adjustment cancels.

### 12.2 Universe and decision time
Wide-band ($1-$50, ADV filter) panel rows that have the label: the cached gap-day names, so the
universe is chosen with the day-D opening gap. The decision is therefore at 09:30, not pre-open;
the features stay the pre-open daily ones. Only ~0.1% of rows are $10+, so in practice this is the
$1-$10 gap pool. Days with fewer than 20 names are not evaluated (MIN_NAMES, as in Stage 1);
about 1,400 of 2,372 days remain, few in 2017-2019.

### 12.3 Model and arms (1 trial)
Stage 1's LightGBM, fixed parameters, on the per-day rank of `share_cost`, yearly walk-forward,
scored 2017-2025 (to 2025-08-06). Top 10 per day. Arms: **model**; **room model** (the Stage-1w
score, the volatility pick); **naive** (room20, prior 20-day mean range); **random** (seed 11);
**universe** (mean of all names); **null** (shuffled-label model, seed 1).

### 12.4 Pass bar (all must hold)
Day-clustered bootstrap (2,000), paired by day: CI lower bound > 0 for (model - universe),
(model - naive) and (model - room model) mean `share_cost` of the picks. Validity gate as in
Stage 1: null - naive CI lower bound <= 0 and null IC < 0.2 x model IC, else INVALID. Determinism:
two fits, identical hashes. Holm p across the three contrasts reported.

### 12.5 Reported, not part of the bar
Picks' mean random-trade return (`mean_pair_ret`) and its value net of c; room; open-to-close;
per year; IC. A pass says only that the label can be ranked before the open within this pool;
whether it trades is a later Stage-2 question.

## 13. Stage 1s + pre-market features (pre-declared 2026-10-27, before any result)

The user's call after the Stage-1s error analysis (the model predicts swing, not direction; past
daily bars cannot carry direction): "add the pre-market bars as features and retest".

### 13.1 Data (no download)
Bars stamped before 09:30 ET only (the decision time, §12.2), from two caches: the pre-market
cache (04:00-09:34, fetched for an earlier round's candidates: day-D opening gap >= 10%, price
$0.50-$10, ADV >= 100k, all known at 09:30) and the regular minute cache's 09:25-09:29 bars. Where
both have a minute, the pre-market cache's bar is used. Minute prices are on the daily adjusted
basis (09:30 open matches the daily adjusted open on 98.9% of rows), so the pre-market gap uses
the prior session's adjusted close. About 70% of the Stage-1s rows have at least one bar.

### 13.2 Features (10, plus their per-day ranks; NaN when there are no pre-open bars)
pm_gap (last pre-open close / prior adjusted close - 1, clipped to [-0.9, 5]); pm_late_ret (last
close / last close before 09:00 - 1); pm_range ((high - low) / vwap); pm_close_pos ((last - low) /
(high - low)); pm_dist_vwap (last / vwap - 1); pm_log_dv (log1p of close x volume summed);
pm_dv_to_adv (that dollar volume / adv20); pm_late_share (share of it from 09:00 on); pm_bars;
pm_fade (last / high - 1). Added to Stage 1's 24 inputs; same LightGBM, parameters, label
(`share_cost`), universe, walk-forward, top 10, MIN_NAMES as §12.

### 13.3 Arms and bar (1 trial)
Arms: **pm model**; **base model** (§12's model, refitted in the same run); room model; naive;
random; universe; null (pm model on the shuffled label). Pass = CI lower bound > 0 for the pm
model minus each of universe, naive, room model **and base model** on mean `share_cost`, plus the
§12 validity gate and determinism. Holm across the four.

### 13.4 Pre-declared diagnostics (reported, not part of the bar)
The question behind the trial is direction: daily rank IC of the pm model vs the any-gain share
and vs open-to-close (with CIs), next to the base model's; the top tenth's chance of closing up.
Sensitivity (not a trial): the same features from 09:25-09:29 bars only, which nearly every row
has, to check that the result does not come from which names the pre-market cache covers.
