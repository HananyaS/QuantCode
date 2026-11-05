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

- **Phase A (pre-registered in §8; results in §8b).** Required-skill curves (ρ·rank + noise; a myopic exit oracle minus its martingale
  value); the allocation rule (ρ* ≤ 2 x attained skill, and an effect ≥ the paired MDE); and the
  label-free indicator-redundancy check.
- **Phase C (hypotheses in order; pre-registered in §9).**
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

## 8. Phase A pre-registration (2026-11-05, after A0's D3; before any Phase A code runs)

- **Why.** A0 found real gross on the model's selected minutes that stops about 0.2 pp short of
  the round trip (§4b.8). It could be closed by better ranking (entry) or better timing (exit).
  Phase A measures, without spending trial lines, how much skill each lever needs and how much
  we have, and so decides which conditional hypotheses Phase C runs.
- **Inputs.** As §4: c7878's E5a predictions (685 test days, 6,850 P1 name-days), the relabel
  context cut to them, B0-S1 and B0-S2 with their policies (§3). X* is H30 (2,1) for S1 and
  H30 (1,1) for S2.
- **Code.** `stage3/oracle.py` (targets, the oracle rule, the exit relabel), `stage3/replay.py`
  (the random-walk replay), `stage3/skill.py` (the entry reassignment, correlations, required
  skill, allocation), `stage3/indicators.py`, `stage3/run_a.py`.

### 8.1 Shared definitions

- **Candidates of s.** `C_s` = the B0 frame's rows with `ev_s > 0` that pass B0's price and open
  rules (`e6.price_mask`) and have a finite `r_net`.
- **Held pairs.** (t, k) with t in `C_s`, t+1 ≤ k ≤ T_t − 1 and k < the static label's exit_min.
  "Exit after bar k" means a market sell at the open of the first present bar j ≥ k+1, at
  O_j(1 − h_j). If no bar is present in k+1..T_t, the TIME exit applies.
- **Pass set Π for an arm.** Daily mean > 0, paired daily CI lo > 0 vs its B0, and per-$ CI
  lo > 0 (§6's first three bars; NaN fails).
- **Required skill ρ\*.** The smallest grid ρ whose oracle arm is in Π; "none" if no grid ρ is.
- **CIs.** Whole days resampled, 2,000 resamples, seed 20261027 (smoke: 300).

### 8.2 A.1 Exit skill (first)

- **Targets per held pair** (O\*(j) is the open of the first present bar ≥ j; C\*(j) the close of
  the last present bar ≤ j; σ_k is `sig` at minute k; values clipped to ±6; NaN where a price or
  σ is missing):
  - **m = 5, 15 (myopic):** x = ln(O\*(k+1+m)/O\*(k+1)) / (σ_k √m). It depends on (name-day, k)
    only.
  - **m = rest (continuation; H4's target):** D = (P_static − O\*(k+1)(1 − h))/e_t, plus imp_t when
    the static exit is TP. That is the static label's r_net minus exiting after bar k. Then
    x = D / (σ_t √(T_t − k)).
- **Prior.** μ and sd of x over the held pairs of `C_s`, per (s, m).
- **The oracle.**
  - It observes y = ρ·(x − μ)/sd + √(1 − ρ²)·ε, with ε ~ N(0, 1). For m = 5 and 15, one draw per
    (name-day, k), shared by every trade holding bar k; for rest, one per (name-day, t, k).
  - It exits after the first held bar k with μ + sd·ρ·y < 0 (the posterior mean of x is
    negative). A NaN target never exits.
  - At ρ = 1 it exits when x < 0. At ρ = 0 it never exits when μ ≥ 0, so the curve starts at B0.
  - The noise is drawn once per (s, m) with a fixed seed and reused at every ρ.
- **The relabel** (`oracle.label_day_exit`). Each filled row gets the oracle's flags. Before the
  exit bar, SL and TP run as in `labels.label_day`. In the exit bar the open comes first. The
  class is ORACLE (4) and it pays comm + 2·imp. Rows outside `C_s` get flags too, because the
  account may take them.
- **Grid.** ρ in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1}, for m in {5, 15, rest} and s in {S1, S2}.
  That is 48 relabels.
- **Arms.** Each relabel goes into the B0 frame (r_net, exit_min, cls) under B0's policy, through
  `e6.run_arm`. Reported against B0: the paired daily difference, the per-$ difference and the
  days that differ. Also reported, on B0's own trades, the mean r_net(oracle) − r_net(B0).
- **Attained exit skill.**
  - **Signals at bar k:**
    - mom5 and mom15: ln(C\*(k)/C\*(k − m'))/(σ_k √m');
    - pnl: ln(C\*(k)/e_t)/(σ_t √(k − t));
    - ev: `ev_s` at (name-day, k), defined for k ≤ 360.
    - All are clipped to ±6.
  - **IC** = the pooled Pearson correlation of a signal with x over held pairs where both are
    finite, with a day-clustered CI.
  - **Null.** The same IC on one random-walk replay (`replay.random_walk_bars`, seed 20261105).
    - Each name-day's present bars are redrawn i.i.d. with replacement from its own present
      bars. Each bar carries its gap from the previous close, its body, its high and low
      relative to its open, and its volume. Missing minutes and the first open are kept.
    - σ, spreads, costs, notional and ev stay as they are.
    - On the replay, the static labels, the held pairs, x and the path signals are recomputed.
      `C_s` keeps the same rows.
  - **Net IC** = IC − null IC, with days resampled jointly.
  - **ρ_att(s, m)** = the largest |net IC| over the four signals (point estimate).

### 8.3 A.2 Entry skill

- **Eligible rows** of day d: `E_d`, the rows passing B0's price and open rules with finite r_net.
- **The oracle score** q = ρ·ns_d(r_net) + √(1 − ρ²)·ε. Here ns_d is the within-day normal score
  ((rank − 0.5)/n through Φ⁻¹), and ε is one N(0, 1) draw per row (seed fixed per s, reused at
  every ρ).
- **Reassignment.** The day's ev values over `E_d`, sorted, are reassigned by q's order: the
  highest q gets the highest ev. So the count of ev > 0 rows per day is B0's, and only which rows
  carry them changes. Then B0's policy, through `e6.run_arm`.
- **Grid.** ρ in {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1}, for S1 and S2.
- **Attained entry skill ρ_eq(s).** The oracle skill whose account matches B0's: the smallest
  grid ρ whose daily mean ≥ B0's, linearly interpolated with the grid point below it. It is 0 if
  ρ = 0 already matches.
- **Also reported:** M1's within-day IC, the mean over days of corr(ns_d(ev), ns_d(r_net)) on
  `E_d`, with a CI.

### 8.4 Allocation (binding)

- **Per configuration c,** ρ_cap(c) = min(1, 2·ρ_att(c)). The oracle runs once more at exactly
  ρ_cap(c), with the same noise.
- **A lever enters Phase C iff at least one of its configurations is in Π at ρ_cap.**
  - The exit lever (H4) has configurations {S1, S2} x {5, 15, rest}, with ρ_att from §8.2.
  - The entry lever (H2 and H3) has configurations {S1, S2}, with ρ_att = ρ_eq.
- **Always in Phase C:** H1 (a) to (d) and H5. H6 and H1(e) are out (§4b.6).

### 8.5 A.3 Indicator redundancy (label-free)

- **Indicators at decision minute t,** on the day's bars ≤ t, with the close carried forward over
  missing minutes. A missing high or low is that carried close.
  - RSI(14): Cutler's, simple means of up and down moves over the last 14 changes (50 if flat).
  - Bollinger %b(20, 2) and bandwidth 4·SD20/SMA20 (population SD).
  - MACD histogram (12, 26, 9): EMAs from the day's first bar, divided by the close at t.
  - Stochastic %K(14) on highs and lows, and %D(3), the mean of the last three %K.
- **ORB distance and swing S/R** are F1 features already (`or5/15/30_dist`, `swing_hi/lo_dist`),
  so the check covers the six outputs above.
- **Rows.**
  - Test rows with t in [35, 360] and all six indicators finite.
  - 300,000 rows sampled (seeded) from folds 2020H1 to 2021H1 for training, and 150,000 from
    2021H2 to 2022H2 for testing.
  - The 102 S2 features come from the cached `panel3_p1` that E5a used.
- **Model.** One LightGBM regressor per indicator on the 102 features: 500 trees, learning rate
  0.05, 63 leaves, min_child_samples 200, seed 20261027, deterministic. The metric is R² on the
  test rows.
- **Reading.** If any indicator's R² < 0.5, one add-one refit enters Phase C: S2 plus every
  indicator under 0.5, with M1 as in E5a on the same folds and X*. Otherwise the family is
  closed.

### 8.6 Fidelity (any failure stops Phase A before a reading)

1. B0-S1 reproduces c7878's `e5a` line, and B0-S2 reproduces the `s3-b0` line (0062e2a6b777).
2. The exit relabel with no flags reproduces E5a's cls, exit_min and r_net at both X* (§3
   check 2), and its account reproduces B0 exactly.
3. The entry reassignment with q = ns_d(ev) reproduces B0's daily series exactly.
4. No row after 2022-12-31 is read.

### 8.7 Trial lines and smoke

- **No trial lines.** Every Phase A arm acts on realised outcomes, so none can become a finalist.
  The arms are logged in results.json and MLflow. N stays 359.
- **Smoke:** the first fold's days, 300 resamples, ρ in {0, 0.3, 1}, m in {5, rest}, an indicator
  sample of 20,000 and 10,000 rows, no MLflow.

### 8.8 Predictions (Claude's, before the run)

1. μ > 0 for every (s, m): selected trades drift up while held.
2. ρ_att ≤ 0.05 for every (s, m). For rest, pnl's raw |IC| is ≥ 0.05, and the null removes at
   least half of it.
3. At ρ = 1 every exit configuration is in Π. Where ρ\* exists for an exit configuration, it is
   between 0.2 and 0.5.
4. H4 does not enter.
5. The entry oracle at ρ = 0 loses more than B0 in both feature sets. ρ_eq is 0.02 to 0.10, ρ\* is
   0.15 to 0.30, and H2 and H3 do not enter.
6. At least one indicator has R² < 0.5, so the add-one refit enters. At least four of six have
   R² ≥ 0.5.

### 8.9 Calls made by Claude (review these)

| # | Call |
|---|---|
| A1 | The martingale subtraction sits on the attained skill, not on the oracle. An oracle's value at a given ρ measures foresight, so subtracting its replay value would measure nothing. The null removes mechanical correlations, such as P&L so far against what is left under barrier geometry. |
| A2 | The replay draws bars with replacement (a random walk). A permutation fixes the day's total, which builds in a correlation of about −m/390 between adjacent m-minute windows. |
| A3 | The entry's attained skill is ρ_eq, the oracle skill that matches B0's account, not the raw IC. Rank IC on barrier outcomes moves with σ and cost without mean information. |
| A4 | The Bayes rule with the held-trade prior makes ρ = 0 equal B0 when μ ≥ 0, so a weak oracle does not exit on noise. |
| A5 | The rest target (continuation value) is added beside the plan's 5- and 15-minute myopic oracles, because it is H4's training target. |
| A6 | The allocation's 2x uses the point estimate of attained skill, which is generous to the lever. |
| A7 | Phase A arms are not trial lines (§8.7). |

## 8b. Phase A results (2026-11-05)

- **Run.** poolpoc 28ed6cb18a9c, from run4 in the ML venv. MLflow dd3cc3703c904cc9bf12dbfd4d22d8d7.
- **Scope.** 685 dev test days, 2,000 resamples, 1,995 s, 4.13 GB peak. The smoke (first fold, 112
  days) ran first: 351 s, 2.98 GB, fidelity passed.
- **Trial lines.** None (§8.7). `trials.jsonl` is unchanged (338 trial lines + 10 supersede lines),
  so N stays 359.

### 8b.0 Fidelity (all pass)

- The exit relabel with no flags reproduces E5a's labels exactly at both X*: 2,466,000 rows each,
  with cls, exit_min and r_net equal (max |Δ r_net| 0).
- B0-S1 reproduces c7878's `e5a` line and B0-S2 the `s3-b0` line. The no-flag exit arm reproduces
  B0, and the entry reassignment with q = ns_d(ev) reproduces B0's daily series.
- No row after 2022-12-31 is read.

### 8b.1 A.1 Exit skill: only perfect foresight passes

- **Prior.** μ > 0 for every (s, m): 0.056 / 0.095 / 0.089 for S1 (5 / 15 / rest) and
  0.045 / 0.059 / 0.070 for S2. Held trades drift up.
- **Attained skill.** Five of six ρ_att come from `ev` (the model's EV at bar k). The path
  signals' net ICs are −0.002 to +0.078, and 15 of their 18 CIs cross 0.

  | Config | ev raw IC | ev null | ρ_att (net, CI) | ρ_cap |
  |---|---|---|---|---|
  | S1 5 | +0.064 | −0.007 | 0.078 from mom5 (−0.007..+0.140); ev nets 0.070 | 0.155 |
  | S1 15 | +0.105 | −0.023 | 0.128 (+0.050..+0.169) | 0.256 |
  | S1 rest | +0.116 | −0.019 | 0.135 (+0.046..+0.186) | 0.270 |
  | S2 5 | +0.038 | −0.030 | 0.068 (+0.034..+0.103) | 0.137 |
  | S2 15 | +0.068 | −0.046 | 0.113 (+0.052..+0.174) | 0.227 |
  | S2 rest | +0.058 | −0.035 | 0.093 (+0.035..+0.152) | 0.186 |

- **The curve is U-shaped.** The paired daily difference vs B0, in %/day:

  | Config | ρ 0.1 | 0.3 | 0.5 | 0.75 | 1 | at ρ_cap |
  |---|---|---|---|---|---|---|
  | S1 5 | −0.082 | −0.068 | −0.013 | +0.050 | +0.172 | −0.081 |
  | S1 15 | −0.046 | −0.085 | −0.069 | −0.006 | +0.173 | −0.086 |
  | S1 rest | −0.068 | −0.057 | −0.033 | +0.034 | +0.227 | −0.059 |
  | S2 5 | −0.164 | −0.143 | −0.070 | +0.015 | +0.230 | −0.166 |
  | S2 15 | −0.145 | −0.188 | −0.143 | −0.058 | +0.189 | −0.188 |
  | S2 rest | −0.100 | −0.133 | −0.086 | +0.054 | +0.292 | −0.157 |

  - Only ρ = 1 is in Π, so ρ\* = 1 for all six configurations.
  - At ρ = 0.75 every account still loses: daily means are −0.003 to −0.115%.
  - At ρ = 1 the accounts make +0.10 to +0.24 %/day, and B0's own trades gain +1.1 to +2.2 pp.
- **Why: the exits saturate.**
  - From ρ = 0.1 to 0.75 the oracle's exit is the label of 92 to 99% of filled minutes, i.e.
    of every trade the account could take (§8b.7).
  - Trades/day rise from 1.25 to 1.7–2.1 (S1) and from 1.54 to 2.4–3.0 (S2), because the freed
    slot buys another round trip.
  - On B0's own trades, the oracle's exits lower r_net at every ρ from 0.1 to 0.3
    (−0.12 to −0.55 pp per trade), before any re-entry.

### 8b.2 A.2 Entry skill: the gap is about 0.06 to 0.11 of oracle skill

| ρ | S1 %/day | S1 trades/day | S2 %/day | S2 trades/day |
|---|---|---|---|---|
| 0 | −0.453 | 3.36 | −0.829 | 6.56 |
| 0.1 | −0.227 | 3.32 | −0.496 | 6.51 |
| 0.15 | −0.132 | 3.29 | −0.340 | 6.45 |
| 0.2 | −0.043 | 3.25 | −0.170 | 6.38 |
| **0.3** | **+0.145** (CI +0.059..+0.227) | 3.17 | **+0.126** (CI +0.048..+0.204) | 6.15 |
| 0.5 | +0.518 | 2.88 | +0.697 | 5.52 |
| 1 | +0.849 | 1.08 | +1.047 | 1.81 |
| B0 | −0.069 | 1.25 | −0.057 | 1.54 |

- ρ\* = 0.3 for both feature sets; ρ = 0.2 fails Π in both.
- ρ_eq = 0.186 (S1) and 0.238 (S2). ρ_cap = 0.371 and 0.477, and both oracle arms there are in Π
  (+0.288 and +0.642 %/day).
- M1's within-day IC on `E_d` is 0.259 (S1, CI 0.250..0.268) and 0.263 (S2, CI 0.253..0.272).

### 8b.3 Allocation (binding)

- **Exit lever: out.** No configuration is in Π at its ρ_cap. H4 does not run.
- **Entry lever: in.** Both configurations are in Π at ρ_cap. H2 and H3 run.
- **Phase C:** H1 (a) to (d), H5, H2 and H3. H4, H6 and H1(e) are out.

### 8b.4 A.3 Indicators: all redundant; the family is closed

| RSI14 | %b | Bandwidth | MACD hist | %K | %D |
|---|---|---|---|---|---|
| 0.796 | 0.929 | 0.903 | 0.877 | 0.883 | 0.848 |

- Test R² on 150,000 rows (2021H2 to 2022H2), trained on 300,000 rows (2020H1 to 2021H1). No
  indicator is below 0.5, so there is no add-one refit.

### 8b.5 Predictions (§8.8)

- **Held:** 1 (μ > 0 everywhere) and 4 (H4 out).
- **Failed:**
  - 2: ρ_att is 0.068 to 0.135, not ≤ 0.05. For rest, pnl's raw |IC| is 0.028 / 0.023, and the null
    is negative, so it removes nothing.
  - 3: ρ = 1 passes everywhere, but ρ\* is 1, not 0.2 to 0.5.
  - 5: ρ = 0 is below B0 and ρ\* = 0.3 is inside 0.15 to 0.30, but ρ_eq is 0.19 / 0.24 (predicted
    0.02 to 0.10) and H2/H3 enter.
  - 6: all six R² are ≥ 0.5 (that half held), but none is below 0.5.

### 8b.6 What Phase A changes (post-hoc; decides nothing)

- **The exit oracle's "ρ\* = 1" belongs to its decision rule as much as to exits.**
  - The noise is independent from bar to bar, and the rule exits at the first bar whose noisy
    posterior is negative. Over about 20 held bars, a false exit is almost certain at any ρ < 1.
  - A real exit model's errors persist across adjacent bars, and its trigger can be set to exit
    rarely. So the curve is a pessimistic bound for exit timing in general.
  - It is binding as pre-registered. H4 returns only through a new pre-registered check, such as
    noise that persists across bars; that is the user's call.
  - Call A4 ("a weak oracle does not exit on noise") holds only at ρ = 0. At ρ = 0.05 the oracle
    exit is already the label of 29 to 93% of filled minutes.
- **One robust lesson for any exit rule:** cutting trades early frees the slot, the account
  re-enters, and the extra round trips cost more than weak timing gains. H1's fewer round trips
  attack the same channel from the other side.
- **The null is negative in all 24 (s, m, signal) cells (−0.001 to −0.046).** Corrected after
  the review (§8b.7):
  - Only rest x pnl is mechanically negative: barrier, target and spread geometry make it so even
    on driftless random walks (−0.010 to −0.028 in the reviewer's simulation). Our −0.019 / −0.026
    fit that, so the subtraction there does what call A1 intends.
  - The other negatives are most likely the noise of one replay draw: their CIs all cross 0, and
    the 24 cells share one or two independent draws. The drift the replay keeps pushes a null
    up, not down. Untested on real data.
  - The sign makes ρ_att generous to the lever. On raw IC (the conservative reading) ρ_cap would
    be 0.08 to 0.23 instead of 0.14 to 0.27. H4 fails even at ρ = 0.5 to 0.75, so neither reading
    changes the verdict.
- **ρ_eq depends on the match.**
  - The oracle at ρ_eq trades 2.6x (S1) to 4x (S2) as often as B0, because random reassignment
    spreads B0's ev > 0 count over more names and minutes.
  - Matched on per-$ instead of the daily mean, ρ_eq would be about 0.14 (S1) and 0.19 (S2).
  - Either way, M1's within-day IC (0.26) overstates its usable ranking skill, as call A3 expected.
- **For H2 and H3:** the target is oracle-equivalent ranking skill of about 0.3, up from
  0.19 to 0.24.

### 8b.7 Review

- A fresh-context review (opus) of 28ed6cb found 0 Critical, 0 Important and 11 Minor. None can
  change the allocation.
  - It hand-traced the exit oracle's t/k alignment, the noise reuse, the entry reassignment, Π,
    ρ_cap and ρ_eq (it reproduced 0.1854 and 0.2382), and found no leakage.
  - It checked the null with a small simulation on random walks with known answers (above).
- **Two wording corrections, made above.**
  - The exit share (`oracle_share`) is the share of filled minutes whose label is the oracle exit,
    not of B0's or the account's trades.
  - The null's mechanism (§8b.6).
- **For later nulls:** the replay keeps each name-day's drift and σ, so it is not a martingale
  null. A future one should demean each name-day's increments and average 5 to 10 seeds.
- The other minors are in the poolpoc ledger.

## 8c. A.1b pre-registration: the exit oracle with persistent noise (2026-11-05, before any A.1b code runs)

- **Why.** §8b.6: the noise was independent at every bar, and the oracle exits at the first bar
  whose noisy posterior is negative, so it cut 92 to 99% of trades at any ρ in 0.1..0.75. A real
  exit model's errors persist across bars. The user chose a re-check before Phase C
  ("Re-check first", 2026-11-05).
- **What stays as §8.2 and §8b.** The targets (5, 15, rest), the prior μ and sd per (s, m)
  (taken from §8b's results.json), the Bayes rule, the relabel, the grid, the arms, B0, Π, and
  each configuration's ρ_cap from §8b (the attained skill does not depend on the oracle's
  noise).
- **What changes: the noise.** The unit is the trade (name-day, t) for every target. For
  m = 5 and 15 this departs from §8.2's one draw per (name-day, k) shared by every trade:
  every trade decides with its own error.
  - **V1 "trade":** one ε ~ N(0, 1) per trade, the same at every held bar.
  - **V2 "ar1":** ε_1 ~ N(0, 1) and ε_δ = φ·ε_{δ−1} + √(1 − φ²)·η_δ along the trade's bars,
    with φ = 0.9 per minute (half-life about 6.6 bars). Marginally N(0, 1) at every bar.
  - Seeds: (20261107, variant, s, m, chunk start). The same noise at every ρ and at ρ_cap.
  - At every bar the oracle's view still correlates ρ with the standardised target, so ρ keeps
    the meaning that ρ_att and ρ_cap have.
- **Grid and arms.** ρ in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1} plus ρ_cap, for V1 and V2,
  s in {S1, S2} and m in {5, 15, rest}: 108 relabels. Reported per arm as §8.2 (the paired daily
  difference, per $, trades/day, the oracle's exit share, and the per-trade Δ on B0's trades).
- **Fidelity (any failure stops A.1b before a reading).**
  1. B0-S1 and B0-S2 reproduce their logged lines, and their means equal §8b's.
  2. The no-flag relabel reproduces E5a's labels (§8.6 check 2).
  3. Every ρ = 0 arm equals B0 on every day (μ > 0, so nothing is flagged).
  4. Every ρ = 1 arm's daily mean equals §8b's ρ = 1 arm (the noise has zero weight).
  5. No row after 2022-12-31 is read.
- **Allocation (binding; replaces §8.4's exit reading).** H4 enters Phase C iff, under V1 or V2,
  at least one configuration is in Π at its §8b ρ_cap. Two variants x six configurations is
  generous to the lever; entering only buys H4 a place in Phase C, where §6's pass rule and
  trial lines apply.
- **Descriptive (decides nothing).** The lag-1 autocorrelation of `ev_s` along t within a
  name-day, pooled: a rough gauge of how persistent a real model's view is.
- **Trial lines.** None. **Smoke:** the first fold, 300 resamples, ρ in {0, 0.3, 1}, m in
  {5, rest}, against Phase A's smoke results.
- **Predictions (Claude's, before the run).**
  1. At ρ_cap the oracle's exit share is 25 to 60% under V1 and 60 to 95% under V2, in every
     configuration.
  2. Under V1, ρ\* is at most 0.75 in at least four of six configurations.
  3. H4 still does not enter.
  4. ev's lag-1 autocorrelation is at least 0.8 in S1 and S2.
- **Calls made by Claude.**
  - A8: the trade is the noise unit for every target (the user was offered "noise drawn once
    per trade, plus one AR(1) variant").
  - A9: φ = 0.9 is fixed in advance, not fitted to the model's errors.
  - A10: the allocation takes the better of two variants, which is generous to the lever.

## 8d. A.1b results (2026-11-05): persistence removes the saturation, but H4 stays out

- **Run.** poolpoc 0902d3ed5ea7, from run4. MLflow eac76dcf70b74925b7d9ff9a4d91fd61. 685 days,
  3,200 s, 1.48 GB peak. The smoke ran first (224 s; anchors passed).
- **Trial lines.** None; `trials.jsonl` is unchanged, so N stays 359.
- **Fidelity (all pass).**
  - B0-S1 and B0-S2 reproduce their lines and §8b's means, and the labels reproduce E5a.
  - All 24 ρ = 0 arms equal B0 on every day, and all 12 ρ = 1 arms equal §8b's.
- **At Phase A's ρ_cap** (difference vs B0 in %/day, CI lo; the account's own mean; exit share =
  the share of filled minutes whose label is the oracle exit):

  | Config | ρ_cap | V1 vs B0 (lo) | V1 mean | V1 share | V1 ρ\* | V2 vs B0 | V2 share | V2 ρ\* |
  |---|---|---|---|---|---|---|---|---|
  | S1 5 | 0.155 | +0.011 (−0.019) | −0.058 | 0.44 | 0.75 | −0.006 | 0.86 | 0.75 |
  | S1 15 | 0.256 | +0.001 (−0.028) | −0.067 | 0.46 | 0.75 | −0.010 | 0.86 | 1 |
  | S1 rest | 0.270 | +0.040 (+0.006) | −0.029 | 0.44 | 0.75 | −0.014 | 0.85 | 1 |
  | S2 5 | 0.137 | +0.006 (−0.029) | −0.051 | 0.43 | 0.5 | −0.064 | 0.84 | 1 |
  | S2 15 | 0.227 | +0.001 (−0.028) | −0.056 | 0.48 | 0.75 | −0.046 | 0.86 | 1 |
  | S2 rest | 0.186 | −0.000 (−0.026) | −0.057 | 0.35 | 0.5 | −0.031 | 0.77 | 0.75 |

- **Allocation (binding, §8c): H4 stays out.** No (variant, configuration) is in Π at ρ_cap.
  Every account there still loses.
- **What persistence changes.**
  - The exit share at ρ_cap falls from 0.97–0.99 (§8b) to 0.35–0.48 (V1) and 0.77–0.86 (V2).
  - Under V1 the curve barely dips: it is never more than 0.032 %/day below B0 (S2 15 at
    ρ = 0.1), and from ρ = 0.3 it is at or above B0 in every configuration.
  - The required skill falls from 1 to 0.5–0.75 (V1) and 0.75–1 (V2). That is still 4 to 7 times
    the attained 0.07–0.14, or more on raw IC.
- **Descriptive.** ev's lag-1 autocorrelation is 0.968 (S1) and 0.977 (S2): the model's view
  persists more than V2's φ = 0.9, so V1 is the closer of the two.
- **Predictions (§8c): all four held.** V1's share is 0.35–0.48 and V2's 0.77–0.86. V1's ρ\* is
  ≤ 0.75 in all six configurations. H4 is out, and ev's lag-1 is ≥ 0.8.
- **What it changes (post-hoc; decides nothing).**
  - At the skill we have, exit timing is worth at most about +0.04 %/day over B0 (S1 rest, V1),
    against a −0.07 %/day account.
  - The exit side's lever is the round-trip count (H1), not timing. Phase C runs H1 (a) to (d),
    H5, H2 and H3.

## 9. Phase C pre-registration (2026-11-05, after §8d; before any Phase C code runs)

- **Why.** §8b.3 and §8d allocate H1 (a) to (d), H5, H2 and H3 to Phase C. The user said "Run
  all" (2026-11-05): pre-register them, review the design (G1), run them, and run Phase D on any
  arm that passes. Confirmation (Phase E) still needs its own go.
- **One feature set, one B0.** Every Phase C arm is B0-S2 with one set of changes:
  - the frame `F0 = frame_e5a(pa, "H30_ku1_kd1", "s2")`, i.e. S2's EV at its X* = H30 (1,1);
  - the policy `P0 = Policy(skip_open=True)`;
  - B0-S2 is −0.057 %/day (CI −0.114..+0.001), per-$ −0.39%, 1.54 trades/day, 292 traded days of
    685.
  - B0-S2 beats B0-S1 on both bars, and one B0 halves the lines (call C1).
- **A structural fact the arms rely on.** P1 has exactly 10 name-days per day (all 727 dev days),
  and the account has 10 slots with one position per symbol. So slots never bind: each
  name-day's trades are a sequence set by its own rows, the cooldown and its exits. "The freed
  slot buys another round trip" (§8b.6) means the same name re-entering.
- **Code.** `stage3/account.py` (the no-re-entry-after-stop account), `stage3/phase_c.py` (arms,
  controls, the pass rule), `stage3/daygate.py` (H5), `stage3/select.py` (H2), `stage3/events.py`
  (H3), `stage3/run_c_fit.py` (the refit, §9.7) and `stage3/run_c.py` (the evaluation).

### 9.1 Shared definitions

- **An arm** is a frame with F0's rows and columns, its own outcomes (`cls`, `exit_min`,
  `r_net`) and its own score in the `ev` column, run through an account:
  - candidates = score > 0, P0's price and open rules (`e6.price_mask`), a finite r_net;
  - a score of NaN means "not eligible" (an H5 off-day, an H3 non-event minute, an H2 row not
    selected);
  - the account is `sim.run_account` (fixed basis, 10 slots, cooldown 5) unless the arm says
    otherwise.
- **Re-entry rules.**
  - *cool5:* B0's 5-minute cooldown.
  - *once:* one entry per name-day, i.e. `sim.run_account` with a cooldown of 10,000 minutes.
  - *nosl:* `stage3.account.run_account`, which is `sim.run_account` plus one rule: after a trade
    whose class is SL, that symbol takes no new entry that day. With the rule off it returns
    `sim.run_account`'s trades and daily series exactly (a unit test on random candidates, and
    fidelity check 3).
- **Reported per arm.**
  - The daily mean with its CI, per-$ with its CI, trades/day and traded days.
  - Against B0-S2: the paired daily difference (CI, p) and the paired per-$ difference (CI and a
    one-sided p = (1 + #{resampled differences ≤ 0}) / (R + 1)).
  - The mean in each half-year (the six folds), the 2022 mean and the difference vs SPY
    open-to-close.
- **Controls per arm.** None is a trial line, because none can become a finalist.
  - *C1m* (§6's C1, matched on minute of day). For each day d, n_d rows, where n_d is the arm's
    trade count that day. They are drawn from the day's filled rows that pass P0's price and open
    rules, at the minutes t at which the arm traded that day (any of the day's names). They are
    admitted in a uniform random order (seed `config.RANDOM_SEED`) under the arm's cooldown, with
    the arm's own outcomes.
    - A nosl arm's C1m uses cooldown 5, because `sim.admit` has no stop rule.
  - *C4:* n_d rows of the arm's frame admitted by descending F0 ev (`sim.admit`), under the arm's
    cooldown. It uses the whole day's ev, so it is a hindsight-count control, not a policy.
  - *shift1* (§4.6, case D3): the arm with its score lagged one minute within the name-day
    (`run_a0.shifted`). It is reported as the arm's one-minute-delay control.
- **CIs.** Whole days resampled, 2,000 resamples, seed 20261027 (smoke: 300).

### 9.2 Pass rule (§6 made exact; every bar must hold)

1. Daily mean > 0 (point estimate).
2. Paired daily difference vs B0-S2: CI lo > 0.
3. Per-$: CI lo > 0.
4. ≥ 150 traded days.
5. ≥ 5 of the 6 half-year means > 0.
6. **Holm.** Each arm's primary contrast is its paired per-$ difference vs B0-S2, with the
   one-sided p of §9.1. The p-values of every arm that runs are Holm-adjusted together, and the
   arm's adjusted p must be < 0.05.
   - Per-$ is primary because several arms trade less than B0, and against a losing B0, trading
     less raises the daily mean by itself (bar 2 alone would reward it).
7. Paired daily difference vs C1m: CI lo > 0. Daily mean > C4's.

- **Reported, not binding.** These bind at confirmation (decision 3):
  - DSR with N = 359 + this phase's lines and V[SR] = 1/(T − 1)
    (`research.penny.eval.stats.deflated_sharpe_ratio`);
  - PBO (CSCV, 16 blocks, `research.penny.eval.battery.pbo_cscv`) over the arms that run, if
    there are ≥ 8;
  - shift1 and 2022.
- **After the rule.** A passing arm is a finalist and goes to Phase D (§9.9). If two or more
  pass, one combined arm runs (§9.8).

### 9.3 H1: fewer round trips (4 lines, no refit)

- **Entry.** F0's ev > 0 under P0 (B0's entry, unchanged).
- **Exit.** Relabelled through `run_a.relabel` with no flags, i.e. `labels.label_day`'s rules,
  on the 6,850 test name-days.

| Arm | Barrier | Re-entry |
|---|---|---|
| `h1a` | H30 (1,1), X* | once |
| `h1b` | H30 (1,2): the stop at 2σ√(T−t) | nosl |
| `h1c` | H60 (1,∞): no stop | once |
| `h1d` | H60 (1,2) | nosl |

- **Adaptations from the plan's arms.**
  - S2's X* has k_u = 1, so the plan's k_u = 2 becomes 1 (call C2).
  - k_d = ∞ is `labels.Barrier(60, 1, inf)`: the stop price is −∞, so no SL ever fires (a unit
    test).
  - The H60 arms keep X*'s ev as their score: only the exit and the re-entry rule change, not
    the entry (call C3).
- **Kill test (the plan's).** C1m with the arm's exit and count (bar 7), the half-year sign test
  (bar 5), and 2022 read separately.
- **Descriptive (decides nothing).** In B0-S2, the share of trades that are a name-day's first
  entry, and the mean r_net of first entries vs re-entries.

### 9.4 H5: the day gate (2 lines, a small model per fold)

- **Timing.** The gate is set at decision minute 14 (bars ≤ 14, 09:30 to 09:44). That is P0's
  first allowed decision, whose entry is the 09:45 open.
  - On an "on" day the arm trades exactly as B0. On an "off" day it takes no trade.
  - Because slots never bind, the gated arm's trades are B0's trades on its on-days.
- **Day features (17), all at t = 14.**
  - Twelve F5 columns from E5a's `panel3_p1`: `spy_ret_open`, `qqq_ret_open`, `iwm_ret_open`,
    `qqq_ret_d1`, `qqq_tr5`, `qqq_tr20`, `vix_d1`, `vix_chg_d1`, `br_above_open`,
    `br_med_ret_open`, `br_new_hod5` and `br_disp`. They are equal across a day's names.
    `spy_r15` and `iwm_r15` are dropped: they are NaN at t = 14.
  - The means over the day's 10 names of `ret_open`, `pm_gap`, `gap_atr`, `crvol` and `s_model`
    (NaN ignored).
  - A missing feature takes the training days' median.
- **Targets (one arm each).**
  - `h5_pnl`: 1[B0-S2's day return > 0], fitted on B0's traded days only. B0's daily returns come
    from E5a's out-of-fold predictions, so the training days are the fold's training dates that
    are test days of another fold.
  - `h5_drift`: 1[the pool's drift > 0]. The drift is the mean over the day's names of
    ln(C\*(389)/O\*(15)): from the 09:45 open to the last close. It is fitted on every training
    date of the fold, 2017 to 2019 included.
- **Model.** `LogisticRegression(C=1.0)` (L2, lbfgs, max_iter 2000) on features standardised with
  the training days' means and SDs, refitted per fold on `Fold.train`.
- **Threshold.** A day is on iff p̂ ≥ the 40th percentile of p̂ over the fold's training days, so
  about 60% of days are on.
  - B0 trades on 292 days, so about 60% keeps bar 4 (≥ 150 traded days) reachable (call C5).
- **Extra control (descriptive).** 1,000 random gates, each with the arm's on-day count in each
  fold (seed 20261108). Reported: the arm's per-$ percentile among them.
- **Leakage note (call C6).** B0's daily returns on training days come from M1 models that were
  fitted with the test fold's labels (E5a's K-fold), so `h5_pnl` has a second-order path from
  the test fold. `h5_drift` does not.

### 9.5 H2: selection-aware entry (up to 2 lines, from the refit in §9.7)

- **`h2q` (the plan's (b)).**
  - The score is a LightGBM quantile regression of r_net at X*, with α = 0.35.
    - It uses S2's 102 features and E5a's training draw (budget 400,000 filled rows, E5a's seed
      `config.SEED` = 7).
    - Parameters: `models.LGB_BASE` (300 trees, learning rate 0.05, 31 leaves,
      min_child_samples 200, deterministic) with `objective="quantile"` and `alpha=0.35`.
  - The rule: enter iff q̂ > τ_f.
    - τ_f is set per fold on its inner-validation rows that pass P0's rules and are filled. It is
      the value that leaves the same share of those rows above it as ev > 0 does.
    - So h2q trades about as often as B0 and differs in which rows it picks. The score column is
      q̂ − τ_f.
  - **Pre-check (label-free, on test).** The mean over days of the within-day Spearman
    correlation of q̂ with F0's ev, on rows passing P0's rules. h2q runs only if it is < 0.8.
    Otherwise it is close to B0 and costs a line for nothing.
- **`h2c` (the plan's (c)): conformal selection (Jin & Candès 2023, cfBH).**
  - *Calibration:* the fold's inner-validation rows that pass P0's rules and are filled, scored
    with the refit's ev (the same model as E5a's).
  - *Null* for a test row j: r_net_j ≤ 0. Its p-value is
    p_j = (1 + #{i in calibration: r_net_i ≤ 0 and ev_i ≥ ev_j}) / (n_cal + 1).
  - *Selection:* BH at α = 0.1 within each decision minute (date, t), across that minute's
    eligible rows (≤ 10 names).
    - BH within the minute is deployable: the cross-section at t is known at t.
    - BH over a fold or a whole day is not deployable, because it uses later minutes.
  - The score is 1 on selected rows and NaN elsewhere.
  - **Pre-check (label-free, on test).** The selected rows fall on ≥ 150 distinct days. h2c runs
    only if so.
  - The calibration rows also fit M1's isotonic calibration (E5a's design). So calibration is
    not independent of the model, and the FDR reading is approximate (call C7).
- **Cut:** EV recalibration and shrinkage of θ. They are E6's θ sweep, which failed.

### 9.6 H3: setups and meta-labeling (2 lines)

- **Events.** `controls.rule_signals` on the context's bars, with `run_e1.assemble`'s inputs: the
  pre-open high, and rvol from the 2017–2019 volume curve.
  - A minute t of a name-day is an event iff any of the five rules fires at t: `orb15`,
    `orb15_rvol`, `vwap_reclaim`, `pmh_break`, `hod_break`.
  - The setup id is the five rule flags.
- **`h3m1`.** F0's ev > 0, at event minutes only. No refit.
- **`h3meta`.** The meta-model is M1 (`models.fit("m1")`, E5a's `sl_term`), fitted per fold:
  - on the filled event rows of the fold's training dates (capped at 400,000 by
    `run_e2.budget_rows`), at X*'s labels;
  - with S2's 102 features plus the five rule flags;
  - with isotonic calibration on the inner-validation event rows.
  - It predicts the test event rows. The score is ev_meta at events, NaN elsewhere.
- **Descriptive controls.**
  - *Rule alone:* every event minute under P0, through the account (as Stage 2m's C3).
  - *The martingale null of the rule's base rate:* the event rate per name-day on one
    random-walk replay (`replay.random_walk_bars`, seed 20261105) vs the real bars.

### 9.7 The refit (`run_c_fit`, one heavy job)

- **E5a's pipeline.**
  - The panel and folds come from `run_e5._panel_and_folds`, and X*'s labels from
    `e5.set_labels` through `e5.swap_labels`.
  - Per fold: `va = run_e1._fit_rows(p1, fold.inner_val, seed)` and
    `tr = run_e2.budget_rows(p1, fold.train, 400,000, seed)`, both with E5a's seed `config.SEED` = 7.
- **Per fold, three fits.**
  1. M1 on S2 at X*: ev for the test rows (fidelity check 4) and the inner-validation rows (H2c).
  2. q̂ at α = 0.35: test rows and inner-validation rows (h2q's τ_f).
  3. The H3 meta-model.
- **Also built:** the day-feature table for all 727 dev days, from the panel at t = 14.
- **Outputs.** In `stage2m/<sha12>/s3-c-fit[-smoke]/`:
  - `predictions_c.parquet`: KEY, ev_refit, q35, ev_meta and the five rule flags;
  - `calib.parquet`: the inner-validation rows with fold, KEY, ev, q35, r_net and cls;
  - `days.parquet`;
  - `results.json`.

### 9.8 The combined arm (only if two or more arms pass)

- One arm applies the passing arms' changes together: an H1 exit and re-entry rule, the H5 gate,
  and an H2 or H3 entry.
  - If two passers change the same component, the one with the larger per-$ difference vs B0 is
    used.
- It is one more trial line, under the same pass rule, with the Holm family grown by one.

### 9.9 Fidelity, trial lines, smoke and Phase D

- **Fidelity (any failure stops Phase C before a reading).**
  1. B0-S2 through `e6.run_arm` reproduces the `s3-b0` line (max |Δ| < 1e-12).
  2. The no-flag relabel at X* reproduces E5a's cls, exit_min and r_net (§8.6 check 2).
  3. `stage3.account.run_account` with the stop rule off reproduces B0-S2's trades and daily
     series exactly.
  4. Full mode: the refit's test ev equals E5a's `H30_ku1_kd1_s2_ev` on every row
     (max |Δ| ≤ 1e-6).
  5. No row after 2022-12-31 is read.
- **Trial lines.** Experiment `s3-c`, arms `h1a`, `h1b`, `h1c`, `h1d`, `h5_pnl`, `h5_drift`,
  `h2q`, `h2c`, `h3m1` and `h3meta`.
  - At most 10 lines, plus 1 if the combined arm runs. An arm stopped by its pre-check runs no
    account and logs no line.
  - N = 359 + the lines logged.
- **Smoke.**
  - The refit runs on fold 2020H1 only, with budget 100,000; check 4 is skipped.
  - The evaluation runs on fold 2020H1's days with 300 resamples, no lines and no MLflow.
- **Phase D.** It runs only on finalists. Its checks (§5's list) are pre-registered after Phase
  C's results and before they run. If no arm passes, Phase C reports and the user decides what
  comes next.

### 9.10 Predictions (Claude's, before the run)

1. **No arm passes §9.2.** No arm's Holm-adjusted per-$ p is < 0.05.
2. **H1.**
   - Every H1 account loses: daily mean −0.10 to −0.01 %/day, per-$ −0.8% to −0.2%.
   - h1a trades 0.6 to 0.9 times a day.
   - At least one H1 arm has paired daily CI lo > 0 vs B0, by trading less.
   - None has per-$ CI lo > 0.
3. **H5.**
   - Both gates' per-$ is within ±0.3 pp of B0's.
   - Neither has per-$ CI lo > 0.
   - Each is below the 95th percentile of its random gates.
4. **H2.**
   - h2q's pre-check passes (Spearman 0.4 to 0.75). h2q loses, with per-$ −0.9% to −0.2%.
   - h2c's pre-check passes, and h2c loses.
5. **H3.**
   - h3m1 trades on < 150 days.
   - h3meta loses, with per-$ ≤ −0.3%.
   - The rule alone loses at least 0.5 %/day.
   - The replay's event rate is within 30% of the real one.

### 9.11 Calls made by Claude (review these)

| # | Call |
|---|---|
| C1 | One B0 (S2): it is better on both bars, and the line count halves. S1's entry is not re-tested. |
| C2 | The H1 arms take S2's k_u = 1, not the plan's 2. Each arm changes only the stop and the horizon of X*. |
| C3 | The H60 arms keep X*'s ev. A refit at each new barrier would change entry and exit together. |
| C4 | The primary (Holm) contrast is per-$, not the daily mean (§9.2 bar 6). |
| C5 | The H5 threshold keeps about 60% of days, fixed from the training days' p̂, so bar 4 stays reachable. |
| C6 | `h5_pnl` keeps a second-order path from the test fold through E5a's K-fold M1; `h5_drift` is the clean twin. |
| C7 | h2c calibrates on rows that also fit M1's isotonic layer. The p-values are approximate. |
| C8 | Controls (C1m, C4, shift1, random gates, rule alone) are not trial lines, as in E6. |
| C9 | A pre-check that stops an arm logs no line, and the arm leaves the Holm family. |
