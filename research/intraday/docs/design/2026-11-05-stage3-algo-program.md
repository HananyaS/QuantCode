# Stage 3: the algo phase, long only, costs unchanged (program and A0 pre-registration)

- **Date:** 2026-11-05. Committed before any Stage 3 code runs on data.
- **Code:** branch `research/poolpoc`, package `research/poolpoc/stage3/`. Runs from the
  `.worktrees/poolpoc-run4` checkout of the commit that adds it (its bytes key the c7878 caches,
  §25c).
- **Parent:** the Stage 2m program, `2026-10-27-stage2-minute-research-program.md`. §25d there
  records that §25c's cascade is frozen at E5a.
- **Literature:** `../research/2026-11-04-stage3-literature.md`.

## 0. Why

- Stage 1s, the pre-open pool, passes. Stage 2m's minute-level entry and exit does not: every dev
  arm loses after costs (327 trial lines; N = 327 + 21 = 348 at the start of Stage 3).
- The user asked for a fresh attack on the algo phase with the phd-skills method: what is done,
  the restrictions, the literature, ablations, evaluation and adversarial thinking. **Costs stay
  as §3/§16 and there are no short trades.**
- **The gap.** A winner needs about +0.5 to +0.7 pp net per trade, or about 1.8% to 2.0% gross. B0's
  trades make about +0.7% gross (S2) and +0.27% (S1), so the gap is about 3x.
- **The biggest risk, from the adversarial review of §25b's S2 trades.** Gross rises one for one
  with cost:
  - by cost tercile, gross is 0.07 / 0.52 / 1.52% at costs 0.60 / 1.07 / 2.15%;
  - net is flat at −0.53 / −0.54 / −0.63%;
  - gross peaks in names that fell most since the open.
- **What that looks like.** A liquidity or bounce premium that a market-order taker pays away. If
  so, better information only slides a trade along gross = cost. The two exits are paying less
  of the spread, or finding gross in cheap names.
- **A0 tests this first.** It needs minutes of compute and no refit.

## 1. The user's decisions (2026-11-04)

1. **Passive orders.** Passive exits only, under the existing TP limit-fill rule (a print one tick
   through the limit), plus a stress where the print must clear the limit by one full modelled
   spread. Entries stay market orders.
2. **Overnight.** A daily-data check (A0.4), descriptive only. Holding overnight would be a new
   program.
3. **DSR.** Dev screens use the absolute pass rule (§6). DSR ≥ 0.95, with V[SR] = 1/(T−1) and
   N = all trial lines + 21, binds at confirmation.
4. **§25c.** It stays frozen at E5a, and Stage 3's baselines come from c7878's saved E5a
   predictions.

## 2. Ground rules

- Costs, fills and the account are as in Stage 2m:
  - $25k as 10 slots of $2,500, fixed basis;
  - 5-minute cooldown;
  - a 10% bar-volume cap;
  - no order under $500.
- Long only, paper only, no broker calls.
- **Data windows.**
  - Dev ≤ 2022-12-31 for every choice.
  - Confirmation is 2023-01-01 to 2025-08-06. It runs once, on at most 3 finalists, only when the
    user says so.
  - The holdout (≥ 2025-08-07) stays sealed.
  - No Stage 3 step reads a row after 2022-12-31. A0.4 drops the last dev session, because its
    next open falls in 2023.
- **Pre-registration.** Every experiment states its label, tagging, output, rule, metric and
  predicted outcome before it runs. Post-hoc reads decide nothing.
- **Trial lines.** Every account evaluation on dev is a trial line in
  `data/research_poolpoc/stage2m/trials.jsonl`. This includes A0's stress arms. Descriptive reads
  are logged in results.json and are not trials.
- **Code.**
  - New logic only in `research/poolpoc/stage3/`. The Stage 2m files that key the caches are not
    edited.
  - Committed tests are never edited; new tests are added.
  - The full suite passes before each commit. Nothing is pushed or merged.
- **Compute.**
  - Run in the ML venv.
  - One heavy job at a time, with RAM ≤ 6 GB per job.

## 3. Baselines (B0)

| Arm | Frame | Policy | Status |
|---|---|---|---|
| B0-S1 | `frame_e5a(pa, "H30_ku2_kd1", "s1")` | DEFAULT | c7878's E5a arm `H30_ku2_kd1_s1`, −0.069 %/day (CI −0.118..−0.019); already a trial line, not re-logged |
| B0-S2 | `frame_e5a(pa, "H30_ku1_kd1", "s2")` | `Policy(skip_open=True)` | new trial line `s3-b0` |

- `pa` is `c7878b02c333/e5a/predictions.parquet`: 685 test days and 6,850 P1 name-days, folds
  2020H1 to 2022H2.
- Each arm goes through `e6.run_arm` (fixed basis).
- **Fidelity: any failure stops A0.**
  1. B0-S1 and S2's DEFAULT arm, run through `e6.run_arm`, reproduce their logged daily series
     (`trials/e5a_H30_ku2_kd1_s1_*_c7878b02c333.parquet` and `..._H30_ku1_kd1_s2_...`). The
     largest absolute difference must be < 1e-12.
  2. Relabelling the P1 test name-days with Stage 3's label function and no passive window
     (§4.3) reproduces E5a's `X_cls` and `X_exit_min` exactly for both X. It also reproduces
     `X_r_net` to within 1e-6 on every filled row.
  3. No row after 2022-12-31 is read.

## 4. A0: kill-or-redirect tests

**Shared inputs.**
- The predictions above.
- The relabel context: `panel_build.context` over every P1 dev name-day, from the cached bars and
  the pre-2023 spread model, cut to the 6,850 test name-days.
- From that context, `gross_px` per row, for both barrier sets:
  - entry at `O_{t+1}`;
  - exit at `tp_px` for TP, `min(e(1−d), O_k)` for SL, the time-exit open for TIME, and `L` for
    PASSIVE (§4.3);
  - `gross_px` is the exit over the entry, minus 1;
  - so it is the trade-price return before any modelled cost.
- `gross_pit = r_net + c_rt_pit` is reported beside it.
- Terciles are cut on the rows of the read in question. CIs resample whole days, with 2,000
  resamples, seed 20261027.

### 4.1 A0.1: gross against cost (descriptive)

- **Rows.**
  - *Population:* every filled P1 test row (t = 1..360) at S2's X* = H30 (1,1).
  - *Selected:* the rows with `H30_ku1_kd1_s2_ev > 0`.
  - The same pair at S1's X* = H30 (2,1) with `s1_ev` is secondary.
- **Outputs.**
  - The OLS slope of `gross_px` on `c_rt_pit`, with a day-clustered CI.
  - Mean `gross_px`, `gross_pit`, `r_net` and `c_rt_pit` by `c_rt_pit` tercile.
  - Mean `gross_px` by tercile of `ret_open = C_t / O_first − 1`, the close of bar t over the
    day's first present open, which is known at t.
- **Reading "liquidity premium" (LP), on S2.** All three must hold:
  - the population slope is ≥ 0.5 with CI lo > 0;
  - the selected slope is ≥ 0.8;
  - |mean `r_net` in the top cost tercile − mean `r_net` in the bottom| < 0.3 pp on the selected
    rows.
- **Why the population slope.**
  - Among EV > 0 rows, gross rises with cost even without a premium. To clear EV > 0, a costly
    row needs a larger predicted gross, so selection alone produces the slope.
  - In the unselected population, under a martingale, barrier exits have zero mean trade-price
    return. The tick-through TP fill and the gap-through SL fill both bias toward a negative
    slope.
  - So a population slope ≥ 0.5 is evidence of a premium in the names themselves.

### 4.2 A0.2: entry one minute later

- **Why.**
  - The labels treat `O_{t+1}` as the mid and charge a half-spread on top: e = O(1 + h).
  - If the model picks minutes whose next print is at the bid, the real ask is about 2h above
    that print, and the sim's entry is too cheap by about h.
  - That bias is largest in wide-spread names, where the gross is. Waiting one minute
    re-randomises the print.
- **Paired read (descriptive).**
  - Rows: the selected rows of §4.1 at both X*.
  - For each row, the next row's outcome: entry at `O_{t+2}`, with barriers computed at t+1.
    Rows whose t+1 is unfilled are dropped from both sides.
  - Output: mean `gross_px` now and delayed, by `c_rt_pit` tercile, with the paired difference's
    day-clustered CI.
  - Reading "bounce": in the top cost tercile (S2), the delayed mean is < 0.5 x the undelayed
    mean, and the difference has CI hi < 0.
- **Account arms (2 trial lines, `s3-a0`, variant `shift1`).**
  - Within each name-day, `ev` and `b_hat` move forward one minute (row t carries t−1's score).
    Every outcome column stays on row t, and row t = 1 has no score.
  - B0-S1's frame and policy, and B0-S2's frame and policy, each run through `e6.run_arm`.
  - Reported: each arm's daily mean and CI, per-$ mean and CI, and the paired difference against
    its B0.

### 4.3 A0.3: passive-exit replay

- **The rule.** `stage3/passive.py: label_day_passive(b, sig, f, hs_real, c_rt_pit, notional,
  comm, imp, barrier, n_passive, stress)`. It takes the inputs of `labels.label_day`.
  - **Before the window.** For decision minute t (entry e, u, d and T as `labels.label_day`), let
    k0 = T − N. Bars k = t+1..k0 follow the standard rule: SL first, then TP.
  - **The limit.** If the trade is still open after bar k0 closes, the TP order is amended to a
    sell limit L = min(C_last, tp_px). C_last is the last present close in bars t+1..k0.
  - **In the window.** For bars k = k0+1..T:
    - SL is checked first, filled as now; a bar touching both is SL.
    - Otherwise the limit fills at L when High_k ≥ L + TICK/f (base), or when
      High_k ≥ L + max(TICK/f, 2·h_k·L) (stress), with h_k the modelled half-spread at bar k.
    - A passive fill is class PASSIVE (3), with exit_min = k.
  - **Otherwise** the exit is TIME, as now: a market sell at the first present open ≥ T+1.
  - **Costs.** PASSIVE pays `comm + imp`, like TP. SL and TIME pay `comm + 2·imp`.
  - **Window edge.** k0 ≥ t+1 for H30 with N ≤ 10, since T − t ≥ 25. The function asserts it.
  - **N = None** gives exactly `labels.label_day`'s cls, exit_min, exit_px, gross_ret and r_net.
- **Levels.** N ∈ {5, 10}, base and stress, at both X* (8 relabels).
- **Account arms (8 trial lines, `s3-a0`, variant `passive{N}` or `passive{N}_stress`).**
  - Each feature set's B0 frame, with `r_net`, `exit_min` and `cls` taken from the relabel. The
    scores, policy and account are unchanged.
  - Paired against its B0: the daily difference (CI, bootstrap p), the per-$ difference
    (`e6.per_dollar_diff`) and the number of days that differ.
  - Holm runs across the 4 base contrasts.
- **Per-trade reads on B0's own trades (descriptive).**
  - Mean `r_net(N) − r_net(market)` on the same rows, with a day-clustered CI. It is split into
    rows the limit filled and rows that fell back to the market, and by price bucket. The tick is
    ~1% of the price at $1, so the rule's one-tick trade-through costs most there.
  - Fill rate.
  - **Adverse selection:** `O_{T+1}(1 − h)/L − 1`. On filled rows it is what the market exit
    would have paid relative to the fill (> 0 means the fill sold too cheap). On fallback rows it
    is how far below the limit the market exit landed.
- **The weakest assumption.** There is no queue in the fill model. The stress arms stand in for
  queue position.
- **Readings.**
  - **"Gain"** for (feature set, N): the base arm's paired daily difference vs its B0 has Holm
    p < 0.05 with mean > 0, and the stress arm's paired daily mean vs B0 is > 0.
  - **"Rescue":** a gain, and the base arm's own daily mean > 0 with per-$ mean > 0.

### 4.4 A0.4: overnight, on daily data (descriptive)

- **Rows.** Every P1 dev name-day, 2017 to 2022, except the last dev session.
- **Returns (adjusted prices, used for returns only).**
  - Intraday: `close(D)/open(D) − 1`.
  - Overnight: `open(D+1)/close(D) − 1`, where D+1 is the next session. A name with no row on D+1
    is counted, not filled.
- **Approximate overnight round trip.** `comm` and `2·imp` at t = 360, plus `h` at minute 385
  for the buy and `h` at minute 1 for the sell. Day D's open spread stands in for D+1's.
- **Output.** Means and day-clustered CIs, by tercile of that cost.
- **Reading.** If any tercile's overnight net has CI lo > 0, the user hears of it as a lead
  outside this program. Otherwise it is recorded and closed.
- **Caveat.** The pool is chosen before the open on D with no future data. D+1's open needs the
  name to trade on D+1.

### 4.5 A0.5: the intraday profile (descriptive)

- **Windows.** Thirteen half-hour windows over bar opens: [max(30w, 1), 30(w+1)) for w = 0..11,
  then 15:30 to 15:56 (open 360 to open 386).
- **Window return.** The first present open at or after the end over the first present open at or
  after the start, minus 1.
- **Name-day cost tercile.** The tercile of `c_rt_pit` at t = 15 (09:45), cut over the test
  name-days.
- **Output.** Mean window return, and that return minus `c_rt_pit` at the window's first
  decision minute, by tercile and in total. The test is Bogousslavsky's last-30-minutes result on
  this pool.
- **Late-entry read.**
  - Rows: t = 330..355 (15:00 to 15:25).
  - Hold-to-close, from `Hclose_ku2_kd1_y_h` (net; it does not depend on k) and the
    `gross_px` analogue `O_386/O_{t+1} − 1`.
  - Over all rows, and over rows with `H30_ku1_kd1_s2_ev > 0`.
- **Gate.** H1(e) (late entry) runs in Phase C only if the S2 EV > 0 late rows' mean net
  hold-to-close is > 0.

### 4.6 What A0 decides

| Case | Condition | Then |
|---|---|---|
| D1 | LP holds, and no (feature set, N) shows a gain | Recommend stopping the long-only minute track. Only the report runs; the user decides. |
| D2 | LP holds, and some arm shows a gain | Phase C runs the cost-side levers only: H6 in full (with the +1-tick limit), H1 and H5. H2, H3 and H4 are not run unless the user asks. Phase A keeps only the exit-side skill curve. |
| D3 | LP does not hold | Phase A, then Phase C in the plan's order, with H6 first if it showed a gain. |

- A rescue goes to Phase D's adversarial checks before anything else.
- The bounce reading (A0.2) is reported as the mechanism. Under D3 it adds a one-minute-delay
  control to every Phase C arm.
- A0.5 gates H1(e). A0.4 only reports.

### 4.7 Trial lines

| Exp | Arms |
|---|---|
| `s3-b0` | 1 (B0-S2) |
| `s3-a0` | 2 shift1 + 4 passive base + 4 passive stress |

- That is 11 new lines, so N = 359.
- Smoke mode runs on the first fold's days with 300 resamples and logs no trial.

### 4.8 Predictions (Claude's, before the run)

1. **A0.1.** LP holds on S2: population slope 0.5 to 0.9, selected slope 1.0 to 1.3, net flat.
2. **A0.2.** The top cost tercile's gross falls by ≥ 50% when entry waits a minute, so the bounce
   reading holds. Both shift1 arms lose more than their B0.
3. **A0.3.**
   - 65% to 80% of B0's trades reach the passive window.
   - The limit fills on 50% to 70% of those at N = 10.
   - The per-trade gain on B0's trades is +0.05 to +0.25 pp at base, and negative in the < $2
     bucket (the one-tick rule).
   - The stress halves the gain or worse.
   - At least one base arm shows a gain. No arm is rescued.
4. **A0.4.** Overnight gross > 0 on average, below the approximate cost. No tercile has a net CI
   lo > 0.
5. **A0.5.** The 15:30 to 15:56 window has the highest mean gross after 10:00, below the round
   trip. H1(e) is cut.
6. **Decision.** D2.

## 5. Later phases (each pre-registered here before it runs)

- **Phase A.** Required-skill curves (ρ·rank + noise; a myopic exit oracle minus its martingale
  value); the allocation rule (ρ* ≤ 2 x attained skill, and an effect ≥ the paired MDE); and the
  label-free indicator-redundancy check.
- **Phase C (hypotheses in order).**
  - H6 passive exits;
  - H1 fewer round trips: arms (a) to (d), and (e) if A0.5 allows;
  - H5 day gate;
  - conditional: H4 dynamic exit, H3 one literature rule with meta-labeling, H2 selection-aware
    entry.
  - At most 60 new lines.
- **Phase D.** Martingale replay, cost x1.25 and x1.5, entry one minute later, drop the top 10
  name-days, the leakage audit, and a fresh-context critic.
- **Phase E.** Confirmation, only when the user says so.

## 6. Pass rule for a Phase C finalist (dev screen)

- Account daily mean > 0.
- Paired CI lo > 0 vs B0.
- Per-$ CI lo > 0.
- ≥ 150 traded days.
- ≥ 5 of 6 half-years positive.
- Holm across the hypotheses' primary contrasts.
- Beats C1 (minute-of-day matched) and C4.
- DSR and PBO (CSCV, 16 blocks, if ≥ 8 arms) are reported on dev and bind at confirmation
  (decision 3).

## 7. Calls made by Claude (review these)

| # | Call |
|---|---|
| S1 | The LP reading needs the unselected population's slope, because selection on EV > 0 makes gross rise with cost by itself. |
| S2 | `gross_px` (trade prices, no modelled cost) is primary; `gross_pit` is secondary. Mixing a point-in-time cost estimate with realized fills could create a slope. |
| S3 | A0.2 also tests a possible sim bias: an open print treated as the mid. If the bias exists, the sim is optimistic in wide-spread names. |
| S4 | The passive limit is the last trade (capped at the TP), not the last trade + 1 tick. The +1-tick arm waits for H6. |
| S5 | The stress arms count as trial lines (§2), so A0 adds 11 lines, not the plan's 3 to 5. |
| S6 | The H1(e) gate is a point estimate > 0, because it only decides whether a trial runs. The trial has its own pass rule. |
| S7 | A0.4's open spread for D+1 is day D's. A0.4 decides nothing. |

## 4b. A0 results (2026-11-05)

- **Run.** poolpoc 0062e2a6b777, from run4 in the ML venv. MLflow 178afccaf8ae43b18e4d3d772ef9170c.
- **Scope.** 685 dev test days (2020H1 to 2022H2), 277 s, 2.34 GB peak.
- **Code history.**
  - The first commit, 81fa34b, had 4 Important review findings (§4b.7). The fix commit 0062e2a ran the full A0.
  - The smoke ran on 81fa34b and again on 0062e2a.
- **Trial lines.** 11 traded lines: `s3-b0` x1 and `s3-a0` x10. The ledger now holds 338 trial lines, so N = 359.

### 4b.0 Fidelity (all pass)

- Relabelling with no passive window reproduces E5a's labels exactly at both X*: 2,466,000 rows each, with cls, exit_min and r_net all equal.
- B0-S1 and S2's DEFAULT arm reproduce their c7878 logged series exactly.
- The market relabel's arm reproduces B0 exactly.
- No row after 2022-12-31 is used.

### 4b.1 B0

| Arm | %/day (CI) | Trades/day | Net per $ |
|---|---|---|---|
| B0-S1, H30 (2,1), DEFAULT | −0.069 (−0.118..−0.019) | 1.25 | −0.63% |
| B0-S2, H30 (1,1), skip-open | −0.057 (−0.114..+0.001) | 1.54 | −0.39% |

### 4b.2 A0.1: gross against cost. LP does not hold.

- **The population has no premium.** Every filled P1 test minute at S2's X* (1,148,948 rows) has a slope of gross_px on cost of −0.015 (CI −0.042..+0.012). By cost tercile, gross_px is −0.05 / −0.11 / −0.12%.
- **The selected rows (s2_ev > 0, 9,513 rows).**
  - Slope 0.55 (CI −0.02..+1.17).
  - By cost tercile:

    | Tercile | gross_px | Cost | Net |
    |---|---|---|---|
    | Low | 0.34% | 0.61% | −0.26% |
    | Middle | 0.84% | 1.10% | −0.18% |
    | High | 1.62% | 2.08% | −0.23% |

  - S1 is the same: population slope −0.015, and net −0.39 / −0.11 / −0.11%.
- **So the gross that tracks cost is not in the names.** The model's selection adds +0.4 to +1.7 pp of trade-price gross over the population, more in costly names, and stops about 0.2 pp short of the round trip.
- **Descriptive only:** S1's selected rows in the tercile furthest down since the open net +0.10%.

### 4b.3 A0.2: entry one minute later. No bounce.

- The selected gross barely moves:
  - S2 top tercile: 1.59 → 1.57% (difference CI −0.10..+0.05);
  - S1 top tercile: 1.72 → 1.60% (CI −0.32..−0.001).
- In both feature sets, the top tercile keeps far more than half its gross.
- **Account arms.**
  - shift1-S1: −0.092 %/day, vs B0 −0.023 (CI −0.067..+0.015).
  - shift1-S2: −0.059 %/day, vs B0 −0.002 (CI −0.043..+0.034).
- **So the selected gross is not a print-at-the-bid artifact,** and the sim bias that §4.2 worried about is not visible at one minute.

### 4b.4 A0.3: passive exits. No gain, no rescue; they hurt.

| Arm | Paired vs B0, %/day | Holm p |
|---|---|---|
| S1 N=5 | +0.005 | 0.61 |
| S1 N=10 | −0.022 | 0.58 |
| S2 N=5 | −0.011 | 0.58 |
| S2 N=10 | −0.024 | 0.58 |

- **Stress arms** are significantly worse: S2 N=10 −0.051 (CI −0.083..−0.020), and S2 N=5 −0.027.
- **Per trade on B0's trades, S2 N=10.**
  - 75% reach the window, and the limit fills 89% of those.
  - Delta −0.21 pp (CI −0.39..−0.03). Filled rows −0.32; fallback rows 0.00, as they must be.
- **Adverse selection runs against the limit.**
  - On filled rows, the market time exit would have paid +0.61% over L.
  - Fallback rows' time exits landed 3.4% below L.
  - The same holds at N=5 and in S1: +0.22 to +0.94% on filled rows, −1.3 to −3.4% on fallback rows.
- **So:**
  - The rule's one-tick trade-through fills only on up-moves that keep going.
  - Selling there gives up more than the half-spread and impact it saves.
  - Missing the fill leaves the trade in a falling path.
  - H6 is dead under the existing fill rule.

### 4b.5 A0.4 overnight and A0.5 the intraday profile (descriptive)

- **A0.4.**
  - Over all 7,198 dev name-days with returns: intraday −0.64% (CI −0.99..−0.29), overnight +0.20% (CI −0.03..+0.42).
  - The cost proxy exists on 3,441 name-days, those with a 15:30 fill. On them: overnight +0.37% against an approximate round trip of 1.87%, net −1.50%.
  - The cheapest tercile's overnight is +0.67% (CI +0.04..+1.38), but its net is −0.36%.
  - No tercile has net CI lo > 0, so there is no lead.
- **A0.5.**
  - The pool loses −0.29% from 09:31 to 10:00 and −0.16% from 10:00 to 10:30.
  - After that it is about flat. The best window is 15:00 to 15:30 at +0.09% (CI +0.02..+0.16); the last, 15:30 to 15:56, is −0.07%.
  - There is no last-30-minutes premium on this pool.
  - Late entry (15:00 to 15:25, held to the close) on S2-selected rows nets −1.94% (n 422). **H1(e) is cut.**

### 4b.6 Predictions (§4.8) and the decision

- **Held:** 3a (75% of B0's trades reach the window) and 4 (overnight gross > 0, below cost, no lead).
- **3e** scores True only because the base gain is negative. It is not meaningful (deferred minor).
- **Failed:** 1 (LP), 2 (bounce), 3b (fills 89%, not 50 to 70%), 3c (the gain is negative), 3d (the < $2 bucket is +0.01), 3f, 5 and 6 (decision D2).
- **Decision: D3.** LP does not hold. Phase A runs next, then Phase C in the plan's order.
  - H6 drops: it showed no gain, and a stress arm is significantly worse.
  - H1(e) drops: the late gate is shut.
  - The bounce reading adds no one-minute-delay control; Phase D keeps its own.

### 4b.7 Review and fixes

- A fresh-context review (opus) of 81fa34b found 0 Critical, 4 Important and 9 Minor. All four Important findings were fixed test-first in 0062e2a:
  1. Float32 outcomes, so an unchanged trade is bit-identical to B0, plus the market-arm check.
  2. Adverse selection per §4.3, using `time_px`.
  3. A0.4 gross over every name-day with returns.
  4. A0.5 keeps a window's return when its start-minute cost is missing.
- The nine Minor findings are in the poolpoc ledger, `.superpowers/sdd/2026-11-05-stage3/progress.md`.

### 4b.8 What A0 changes (post-hoc; decides nothing)

- **The 3x gap was the wrong frame for this population.**
  - On the model's selected minutes, gross is real: it is not in the population, and it survives a one-minute delay.
  - Net sits 0.2 to 0.4 pp below zero at every cost level.
  - Closing that gap needs more ranking skill or better exits, not cheaper fills.
- **A0.3's fill pattern points to exit timing.**
  - Paths that tick up in a trade's last 5 to 10 minutes keep rising, by +0.2 to +0.9% to T+1.
  - Paths that don't tick up end 1.3 to 3.4% below the limit.
  - Phase A's myopic exit oracle (H4's gate) should therefore be read first.
