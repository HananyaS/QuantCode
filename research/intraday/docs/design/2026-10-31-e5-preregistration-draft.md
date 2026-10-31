# E5 pre-registration: label and exit design (DRAFT for the user's review, 2026-10-31)

**Status: a draft. Nothing in it runs until the user approves it.** On approval it becomes §22
of the research program (`2026-10-27-stage2-minute-research-program.md`), committed before any
E5 code runs on real data. The rule that experiments are planned together with the user
(memory: fully specified experiments) is why this is a draft and not a pre-registration. E4
ended at 01:56, and the next step in §10 is E5.

## 0. What E5 starts from

- **The model is M1, with both feature sets carried (§20.7, §21).** S1 is base + F5 (30
  features). S2 is all seven families (102 features). No deep model beat M1, and M1-S2 is the
  only model that forecasts by amendment 6.
- **§10's row for E5:** H x (k_u, k_d); A-only (lambda = 0) vs A+B (lambda in {0.5, 1}) vs
  B-only (+ a separate exit model) vs y_H; fixed vs model-driven exit. The model is the best
  one. Expected: A+B >= A-only through a more stable encoder, and a longer H helps net of
  costs.
- **E4 removes the lambda factor [my call].** lambda weights label B inside a shared deep
  encoder. M1 has no shared trunk (§4): it already fits A (multiclass) and B (cross-entropy on
  the fraction) as separate models, and only A's EV trades. With no deep model carried, "A+B vs
  A-only" has nothing to vary. B returns below as a score in its own right.
- **Everything else is held at the E2-E4 settings:**
  - P1, the six purged folds, 685 test days (2020H1 to 2022H2), every P1 test minute;
  - B = 400,000 training rows a fold, seed config.SEED;
  - isotonic calibration on the inner validation year, the row SL term, the cost model of §3,
    the account of §2;
  - dev data only (<= 2022-12-31). The holdout stays sealed.

### What E0 (re-run, 6674ea3d1f9f) says about the eight barrier sets

All filled P1 dev rows (1,241,156; 52.6% of minutes do not fill). The mean point-in-time
round-trip cost is 1.61%.

| Set | TP | SL | TIME | mean u | mean d | p* | r_net, random entry | oracle %/day |
|---|---|---|---|---|---|---|---|---|
| H15 (1,1) | 9.3% | 25.3% | 65.4% | 3.2% | 2.7% | 59.2% | −1.61% | +5.68 |
| H15 (2,1) | 2.4% | 25.4% | 72.3% | 5.5% | 2.7% | 43.4% | −1.62% | +2.93 |
| H30 (1,1) | 12.0% | 22.1% | 65.9% | 4.1% | 3.8% | 58.2% | −1.62% | +5.75 |
| **H30 (2,1), default** | 2.7% | 22.3% | 75.0% | 7.7% | 3.8% | 40.4% | −1.64% | +2.83 |
| H60 (1,1) | 13.4% | 19.2% | 67.4% | 5.4% | 5.3% | 56.3% | −1.65% | +5.32 |
| H60 (2,1) | 2.9% | 19.4% | 77.7% | 10.7% | 5.3% | 38.3% | −1.67% | +2.58 |
| close (1,1) | 13.3% | 14.2% | 72.5% | 10.7% | 10.6% | 50.1% | −1.67% | +3.88 |
| close (2,1) | 2.9% | 14.4% | 82.7% | 21.3% | 10.6% | 32.8% | −1.71% | +1.82 |

- **Costs dominate every set.** A random entry loses about 1.6% whatever the exit.
- **It loses a little more the longer it is held**, from −1.61% at H15 to −1.71% to the close.
  These names fade intraday, so on random entries a longer H costs money rather than
  amortising the cost.
- **(1,1) sets hit TP about four times as often as (2,1) sets.** A TP exit is a limit sell
  that pays no spread (§3), but p* is also higher.

## 1. E5a: which barrier set? (one factor: the set, 8 levels)

| Item | Exact definition |
|---|---|
| Levels | (H, k_u, k_d) in {15, 30, 60, close} x {(2, 1), (1, 1)}; "close" means T = 385. The reference is the default H30 (2, 1). |
| Label (per set X) | §4's triple barrier, unchanged except for (H, k_u, k_d): u = max(k_u sigma_t sqrt(H), 1.5 c_rt), d = k_d sigma_t sqrt(H), T = min(t + H, 385). For "close", sqrt(H) uses H = 385 − t. Tagging is §4's: the first of TP (High >= e(1+u) + 1 tick), SL (Low <= e(1−d); a tie in one bar is SL), or TIME by T. |
| Rows | Every P1 test minute. NO_FILL rows (no bar t+1) are excluded from training and from every metric, as in E1-E4. |
| Model | M1 per set, refit on set X's label: LightGBM multiclass + isotonic + the mu_TIME regressor, with feature set S1 or S2. That is 16 configurations. |
| Output | EV_X = p_TP r_TP + p_SL (sl_nominal + slip) + p_TIME mu_TIME, with set X's terms. The mechanical EV_X (probabilities at the training prior) is the no-skill reference. |
| Outcome per row | r_net_X, the row's realized net return under set X's exits (§4). |
| **Primary metric** | Per day, the mean r_net_X over the top decile of EV_X (rows ranked within the day). Paired daily difference vs the default set's same quantity, with day-clustered bootstrap CIs (2,000 resamples). Holm over the 7 alternatives, within each feature set. |
| Why not lift | Lift subtracts the set's own all-row mean, so it scores ranking skill against a baseline the exit itself moves. E5 asks which (entry score, exit) pair earns more on the minutes it would pick. The top decile's absolute net return answers that. Lift, the gross/cost split, the lift over the mechanical EV and the forecast test (amendment 6) are all reported per set. |
| Decision | A set qualifies when its difference vs the default is > 0 with Holm p < 0.05. The qualifying set with the largest mean becomes E5b's set, decided separately for S1 and S2. If none qualifies, the default stays. |
| Also reported | Each set's traded arm through the account (EV > 0, fixed basis, as in E2-E4), against SPY open-to-close. Per-fold differences. Class rates of the model's top decile. |
| Integrity check | The default set's refit must equal E3's saved EVs on every row (S1 = E3's +F5, S2 = E3's all), as the EV-term diagnostic did. Otherwise E5a stops. |

**Predictions (E5a):**
1. No alternative set qualifies under Holm, in either feature set, and the default stays. The
   strongest gross signal so far is +0.15 pp; exits move the top decile by cost and drift,
   which E0 shows are nearly flat across sets.
2. The two to-close sets have the lowest top-decile mean r_net of the eight, in both feature
   sets. That is the intraday fade, as for random entries in E0.
3. Every set's traded arm loses (daily CI hi < 0).
4. M1-S2 forecasts by amendment 6 under the default, and under at most two other sets.

## 2. E5b: which score? (one factor: the target, 3 levels; the exit fixed at E5a's set)

| Item | Exact definition |
|---|---|
| Levels | **A**, E5a's EV (the reference). **y_H**: a LightGBM regression on y_H, the label's stored barrier-free time exit at E5a's T (§4; `y_h` in labels.py, net of the market-exit cost), with the prediction as the score. **B**: M1's existing B model (cross-entropy on the share of profitable later exits, §4), with the predicted share as the score. |
| Exit / outcome | Fixed: E5a's set's r_net for every level. Only the ranking changes. |
| Model details | y_H: objective `huber` (alpha = 1.0 x the training rows' MAD of y_H) **[my call; the fat tails of these names]**, with the same LGB_BASE as M1, the same rows and features. B: as fitted inside M1 today. Neither is calibrated, since only its ranking is used. |
| Primary metric | As E5a: the top-decile mean r_net, paired vs A, with Holm over 2, per feature set. |
| Decision | y_H or B replaces A only if > 0 with Holm p < 0.05. |

**Predictions (E5b):**
1. Neither y_H nor B beats A. A's EV contains the cost per row, which drives the top decile
   (the cost part was +0.67 to +0.78 pp in E4). y_H also nets out cost, so it comes closest.
2. B's top decile has a lower mean r_net than A's in both feature sets. B scores how many
   later minutes are profitable, not how much.

## 3. E5c: which exit? (one factor: the exit rule, the score fixed at E5b's winner)

| Item | Exact definition |
|---|---|
| Levels | **Barrier**: E5a's set (the reference). **Time only**: no TP or SL, a market sell at O_{T+1}(1−h), with E5a's T. The outcome is y_H. |
| Primary metric | As E5a, paired vs the barrier exit, with a single comparison (no Holm). |
| Prediction | The time-only exit has a lower top-decile mean r_net. The stop cuts the left tail of names that fade, and the TP limit saves the exit spread. |
| Deferred | The model-driven exit, where TP and SL are placed from predicted MFE/MAE quantiles (§7's auxiliary heads), needs a quantile rule that is not written yet (question 2 below). It is proposed as E5d after E5a-c. |

## 4. Compute and order

- Labels for all eight sets are already in the E0 label cache (35 to 77 s each to build).
- E3 fit 17 M1 configurations in 42 minutes (peak 5.8 GB). E5a is 16 configurations, so about
  45 to 60 minutes. E5b is 4 more models, about 10 minutes. E5c needs no fit.
- Order: E5a, then E5b, then E5c, each from one clean commit in the run worktree, one heavy job
  at a time.
- Trials: 16 + 4 + 2 rows, plus the traded arms, all logged to trials.jsonl and MLflow.

## 5. Calls made by Claude (review these)

| # | Call | Where |
|---|---|---|
| 1 | E5 drops lambda (A+B through a shared encoder): M1 has no trunk, and no deep model was carried | 0 |
| 2 | The primary metric is the top decile's absolute mean r_net, not lift | 1 |
| 3 | S1 and S2 are separate Holm families, each deciding its own set | 1 |
| 4 | The default set's refit must reproduce E3's EVs exactly, or E5a stops | 1 |
| 5 | E5b varies only the score; the exit stays at E5a's set | 2 |
| 6 | Huber loss for the y_H regression | 2 |
| 7 | The model-driven exit is deferred to an E5d, with its rule written after E5a-c | 3 |

## 6. Questions only the user can answer

1. **The primary metric.** Should it be the top decile's absolute net return (proposed), or
   lift, as in E1-E4? With lift, longer holds are not penalised for their own baseline.
2. **The model-driven exit.** Should TP and SL be placed from predicted MFE/MAE quantiles, and
   at which quantiles? Or should the model-driven exit move to E6?
3. **The stop rule (§9), still open from E4.** Raw lift, lift over the mechanical EV, or the
   gross part? E5's absolute metric makes this sharper, because a set can win on net return
   while forecasting nothing.
4. **The feature sets.** Keep both S1 and S2 in E5 (proposed), or carry S2 alone, the only
   forecaster?
