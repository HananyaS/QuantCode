# E6 pre-registration: policy (DRAFT for the user's review, 2026-10-31)

**Status: a draft. Nothing in it runs until the user approves it.** On approval it becomes
§24 of the research program (`2026-10-27-stage2-minute-research-program.md`). It will be
committed there before any E6 code runs on real data. The user approved E5a-d; E6 needs its
own approval.

## 0. What E6 starts from

- **E5's design** (§23, §23b), with M1 and score A (M1's EV), at H30 (2, 1):
  - S1 (base + F5) uses the model-placed TP and SL (E5d). The pre-registered rule chose it with
    p 0.049. §23b.3 shows the gain comes from a cheaper top tenth, not from the exit itself.
  - S2 (all families) uses the barrier exit.
  - Data: P1, the six purged half-year folds, 685 dev test days (2020-2022).
- **Every traded arm loses so far.**
  - The best by mean is E5d S1: −0.027 %/day (CI −0.057..−0.001) at 0.60 trades a day, on 74
    of 685 days.
  - The only arm whose CI reaches zero is E5a's H15 (1, 1) S1: −0.030 %/day (CI up to +0.004)
    at 0.46 trades a day.
- **The EV's level is optimistic.** §23.1 showed that minutes with EV > 0 lose on average.

### Descriptive reads made while drafting (no new trial)

**Read this first.** I re-simulated E5's logged EV > 0 arms to set the levels below. These are
the same dev days E6 will be judged on, so the predictions in §7 are not blind.

| Arm (EV > 0, default policy) | Trades | Days traded | Max concurrent | Net per $ | Decision minutes 1-14: share / net per $ | Price under $2: share / net per $ | $10+: share / net per $ |
|---|---|---|---|---|---|---|---|
| S1 model-driven (E5d) | 410 | 74 | 8 | −0.51% | 7% / +0.09% | 3% / +0.82% | 17% / −0.98% |
| S1 barrier (E5a) | 928 | 189 | 10 | −1.03% | 13% / −2.45% | 9% / −0.52% | 13% / −2.01% |
| S2 barrier (E5a) | 1,260 | 330 | 10 | −1.00% | 19% / −2.57% | 7% / −0.61% | 13% / −1.42% |

- **The first minutes are the costliest.** In the barrier arms, the first 14 decision minutes
  carry 13-19% of trades and lose about −2.5% per dollar, against −0.6 to −0.8% afterwards.
  §3 charges these entries twice the half-spread.
- **The under-$2 bucket is not the worst. $10+ is.** This cuts against §10's expectation that
  the price floor matters most. A price cap is not among §10's factors, so it is not added here
  (question 5).
- **10 slots do bind on some days** (max concurrent positions 10 in both barrier arms).
- **Per dollar, net is not monotone in EV.** Among taken trades it improves from EV > 0 to EV >
  0.5% and is worst above 1% (S2: −1.00%, −0.75%, −1.25%).

## 1. Question, and the one hazard

**The question:** does an entry and account policy turn the account's daily net return
positive, or bring it closer?

**The hazard:** every arm loses, so any policy that trades less moves the daily mean toward
zero without being better. A metric that only reads the daily mean would pick "trade least".
The decision rule in §4 guards against that.

## 2. Fixed

- **Predictions.** The saved per-minute EV, r_net and exit minute. There is no refit.
  - S1: E5d's `md_s1_*` columns (poolpoc 8ece6e9d7ee7/e5d).
  - S2: E5a's H30 (2, 1) S2 EV and labels (ddc248da1892/e5a).
- **Account.** §2's simulator, fixed basis.
  - $25,000, with $2,500 per position. That is the labels' size; the costs depend on size, so a
    different size would need new labels.
  - One position per symbol, no entries after 15:30, flat by 15:56. Highest EV first within a
    minute.
- **Days.** The 685 dev test days. Nothing after 2022-12-31 is read.
- **The default policy is E5's arms:** θ = 0, 10 slots, no price floor, the first 15 minutes
  allowed, a 5-minute cooldown.

## 3. Factors and levels

One factor at a time against the default, in each feature set.

| Factor | Default | Levels | Alternatives per feature set |
|---|---|---|---|
| θ: enter only if EV > θ | 0 | 0.25%, 0.5%, 1.0% (about the 25th, 50th and 85th percentiles of EV among EV > 0) | 3 |
| Slots | 10 | 3, 5 | 2 |
| Price floor: the decision minute's traded (unadjusted) price | none | $2, $3 | 2 |
| First 15 minutes | entries allowed | no entry before the 09:45 open (decision minutes t ≤ 13 excluded) | 1 |
| Cooldown before a symbol re-enters | 5 min | 15, 30 min | 2 |

- That is 10 alternatives per feature set: 20 traded trials, logged to trials.jsonl.
- The default arm is E5's, already logged. It is re-run as an integrity check, not as a trial.

## 4. Metric and decision

- **Primary metric:** the account's mean daily net return over the 685 days, with zero on days
  without trades. It is the quantity §8's bar judges.
  - It is paired by day against the default policy in the same feature set, with 2,000
    resamples.
  - Holm is applied within each factor and feature set.
- **Decision.** A level replaces the default only if both hold:
  1. Its paired difference has mean > 0 and Holm p < 0.05.
  2. Its net per dollar traded (the sum of r_net × notional over the sum of notional) is
     higher than the default's (point estimate; its day-clustered CI is reported).

  Condition 2 is the guard. A level that only trades less raises the daily mean, but not the
  net per dollar.
- **The combined policy.** Every kept level, combined, is run once and logged as one trial.
  It is the policy E5-E6 hands to E7. If no level is kept, the default is the policy.
- **Also read, for every arm:**
  - Trades per day and days traded.
  - Net per dollar with its CI.
  - Sharpe, maximum drawdown, and the difference vs SPY open-to-close.
- **For the combined policy only:**
  - Net per dollar by price bucket and by entry hour.
  - Controls, reported but not gated: C1, random timing matched to the trade count on each day
    (the existing sim.admit), and C2, open-and-hold (E1's).

## 5. After E6 (§22.6)

- **The end-of-E6 check.** As approved, it reads every traded arm logged in E5 or E6. If none
  has a mean daily net return with CI upper bound > 0, the long-only track stops, without E7.
  - As worded, the check is already passed. E5a's H15 (1, 1) S1 has hi +0.004 at 0.46 trades
    a day; its CI reaches zero because it trades little (question 4).
- **E7** (confirmation, at most 3 finalists) does not run without the user in any case.

## 6. Integrity (any failure stops E6)

1. **The default reproduces E5.** The default policy under E6's code reproduces E5's logged
   daily series exactly: S1 against E5d's arm, S2 against E5a's H30 (2, 1) S2 arm.
2. **The simulator's new parameters change nothing at their defaults.** sim.py's slot count
   and cooldown become parameters whose defaults are config's, and it is tested that they give
   today's result.
   - Each factor changes only what it names: the floor and the first-15 rule drop candidates,
     θ drops candidates, and slots and cooldown change only the account.
3. **No late rows.** The predictions' dates are asserted ≤ 2022-12-31.

## 7. Predictions (Claude's, before the run; informed by §0's reads)

1. No E6 policy, the combined policy included, has a daily CI lower bound > 0 in either
   feature set.
2. The first-15-minute exclusion is kept in S2. It is not kept in S1, where those minutes are
   only 7% of the model-driven arm's trades.
3. Neither price floor is kept in either feature set.
4. No slot or cooldown level is kept: they trade less without raising net per dollar.
5. No θ level is kept in either feature set.

## 8. Compute and code

- **Compute.** Each arm is one account simulation over saved predictions, taking seconds. The
  whole of E6 is minutes, with no model fit.
- **Code (poolpoc).**
  - `sim.run_account` and `sim.admit` take `slots` and `cooldown` arguments, defaulting to
    config's. Tests pin today's results.
  - A `run_e6.py` loads the two prediction frames, builds the candidates per level, runs the
    arms, applies §4's rule, writes the report and results.json, and logs the trials and an
    MLflow run.
  - A smoke on E5's smoke outputs runs first.

## 9. Calls made by Claude (review these)

| # | Call |
|---|---|
| 19 | E6 reuses E5's saved predictions; nothing is refitted. θ is chosen like E5a's barrier sets: fixed levels compared on the dev test days with Holm. §9's text puts θ on inner validation, which would need per-fold refits that also predict the inner-validation minutes (question 2). |
| 20 | Each position stays at $2,500, the labels' size, because the labels' costs depend on it. "Slots" changes how many positions can be open, not their size. |
| 21 | The guard in §4 (net per dollar must rise) keeps E6 from rewarding trading less. |
| 22 | The θ levels come from the EV quantiles among EV > 0 in E5's arms (§0). |
| 23 | "The first 15 minutes" means no entry before the 09:45 open: decision minutes t ≤ 13, the entries §3 charges twice the half-spread. §0's descriptive read used t ≤ 14. |
| 24 | §0's reads were made on the dev days E6 is judged on, before this draft. They shaped the levels and predictions, and they are disclosed here. |

## 10. Questions only the user can answer

1. **The guard.** Is "the daily mean rises under Holm *and* the net per dollar rises" the
   right way to stop E6 from rewarding trading less? Or should the net per dollar traded be the
   primary metric?
2. **θ.** Should θ use fixed levels on the dev test days with Holm (proposed: cheap, and how
   E5a chose sets)? Or should it be tuned per fold on inner validation, as §9 words it? That
   means refitting each fold to predict its inner-validation minutes, about 25 minutes, with
   E5d's heads for S1.
3. **S1's exit.** Should S1 keep the rule's model-driven exit (proposed, since the rule was
   pre-registered)? Or should it revert to the barrier exit, since §23b.3 shows the exit itself
   adds nothing and the gain is a cheaper top tenth?
4. **The end-of-E6 check.** As worded, E5a's H15 (1, 1) S1 already passes it (hi +0.004 at
   0.46 trades a day). Keep it as is, or tighten it? For example, it could also require at
   least one trade a day, or a net per dollar whose CI lower bound is > 0.
5. **A price cap.** Should a $10 price cap be added as a level? $10+ is the worst bucket in
   every arm (§0). It is not in §10's list, and adding it after seeing §0 is a forking path.
