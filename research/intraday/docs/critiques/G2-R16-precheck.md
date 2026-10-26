---
gate: G2 (pre-check, plumbing only)
subject: R16 smoke outputs at 05a29ad (r16/05a29adaefd4/smoke), read against g2-brief.md checks 1-8 and the code at 05a29ad
verdict: block
---

# G2-R16 pre-check: plumbing on the smoke artifacts

Scope: whether the FULL run's results.json and charts will be readable, correctly attributed and
correctly judged. Smoke numbers are not evidence and none are quoted. The strategy is not judged.
Inputs: `smoke/PASSED.json`, all 14 stage `DONE.json` files, `report/results.json`,
`evaluate/evaluation.json`, the 7 charts (PNG + HTML present), and the code at
05a29adaefd4002ee6097eaad47a87c3d1880f14 (`git log` confirms HEAD).

Overall: provenance, the dev/holdout guard, the AUC-null multiplicity wiring (m and z*), the
section 11.1.1-5 condition code, Holm, the DSR at N = 4,248, the SPY reader and the parity and
engine-replay plumbing are in place. One block: the null (a) account test can void the run, and it
sits outside the multiplicity family. In this smoke it already returned `false`, and nothing
stopped the smoke from passing.

---

## Findings

**[block] The null (a) grid-account test voids the full run, but it is an unadjusted 95% test on
each of the 54 main cells, combined with AND. It is not in P3-12's family-wise control.**
- Verified: `null_a_accounts` runs the test per main cell and sets `ok &= nr.passed`
  (research/intraday/research/r16_pipeline.py:1205-1218). Each cell passes only if a two-sided 95%
  percentile CI contains 0 (research/intraday/research/r16_nulls.py:302-312). A `False` exits the
  full run with code 5 before engine_replay (r16_pipeline.py:1523-1525).
- Verified: m counts only null (b) and the martingale AUC (r16_nulls.py:230-236, P3-12/P3-15). The
  account test has no z* (prereg P3-4, PREREGISTRATION_R16.md:1283-1285). The code therefore
  matches the prereg; the gap is in the design.
- Verified: in this smoke the account test already failed. `smoke/evaluate/DONE.json` has
  `"a_account_passed": false`, and `evaluation.json` `null_a_account.cells.*.passed` is false.
  PASSED.json was still written, because the smoke pass rule checks parity, determinism and nulls
  (c) and (d) only (r16_pipeline.py:1534-1535). It never reads `a_account_passed` or
  `nulls.passed`.
- Suspected: about 18 null models (one per (k, h, model)) feed 54 correlated cells. Even if the
  null held exactly, the chance that at least one cell's CI excludes 0 is well above 5%. That is
  an estimate from the design, not a run number. Also, P3-2 itself says a within-day-permuted
  model can learn "volatile window => volatile day". Its account can then differ from random-score
  C_b without any leakage, so a two-sided "CI contains 0" test against C_b can fire for reasons
  other than leakage.
- Effect on the full run: a likely spurious VOID at evaluate. Engine replay never runs, and H1
  cannot be judged. The stages resume, so a rerun at the same SHA reads evaluate's DONE record
  again and exits 5 again. Fixing it means new code, which means a new SHA.
- Change requested: a controller ruling before the full run reaches evaluate. Either put the
  account test inside a family-wise rule (for example, Bonferroni or Holm over the null-model
  keys, one-sided in the direction that signals leakage), or make it report-only. The ruling must
  be recorded as an amendment before any full-run evaluate output exists.

**[warn] results.json cannot show a void, and on void paths it gives the wrong reason.**
- Verified: `build_results` has no void flag and no null (a) account field
  (research/intraday/research/r16_report.py:46-56). The only pointer is the string
  `nulls.a_account = "simulated in the evaluate stage (evaluation.json: null_a_account)"`
  (r16_pipeline.py:792).
- Verified: on a nulls void (exit 5 after nulls, r16_pipeline.py:1519-1522), evaluate never ran,
  so `ev = {}`. Selection then reads `"no main cell meets §11.1.1-5"` and `failing_conditions` is
  empty (r16_pipeline.py:1365-1372). The run would read as an H1 failure when it is actually void.
- Verified: on an a_account void, `selection.cell_id` can name a cell, with the reason "highest
  DSR among cells meeting §11.1.1-5", while `gate_I_development.passed` is false and no field says
  why. The same gap applies when engine replay is not identical: `failing_conditions` lists only
  conditions 1-5 (r16_pipeline.py:1369-1371), so condition 6 is never named.
- Change requested (G2 procedure, since the code is frozen while the run is live): G2 must read the
  process exit code, `GRID_VOID.json`, `nulls/nulls.json` `passed` and `evaluate/evaluation.json`
  `null_a_account`, and must not rely on `results.json.selection` or `gate_I_development`. A later
  fix (for a rerun) should add `void: {stage, reason}` and a condition-6 entry.

**[warn] A void-capable null test with no data passes.**
- Verified: `fwer_verdict` treats a non-finite z as not exceeding z*, so `passed = True`
  (r16_nulls.py:248-258). In this smoke, null (b) kept 0 of its out-of-fold rows
  (`results.json nulls.b_dropped.gbdt-k15-h15.oof.n_kept = 0`, every row `beyond_last_session`).
  It still reported `passed: true` with `z: null` (`nulls.multiplicity.tests[1]`), and
  `extra.n_pairs = 0`.
- The same applies to a martingale result whose strata hold no pairs: AUC NaN gives z NaN, which
  passes (r16_nulls.py:206-213). Only `mg is None` is treated as a void (r16_pipeline.py:839-842).
- Suspected impact on the full run: low. Full sessions are contiguous, so d+20 targets exist for
  all but the last 20 out-of-sample days. Still, a vacuous pass would be indistinguishable from a
  real one in results.json.
- Change requested: at G2, check `n_pairs > 0` and a finite z for every one of the 36 voiding
  entries in `nulls.multiplicity.tests` and `nulls.retrain.*[].extra`. Treat any NaN or empty
  entry as a void. A later code fix should do this inside `fwer_verdict`.

**[warn] A void run writes no trial-ledger lines and no MLflow runs, and the rerun guard then
lets a later full run start without an amendment.**
- Verified: `log_runs`, which writes the 54 ledger lines and asserts their count, runs only after
  every stage finishes (r16_pipeline.py:1530-1531, 1490-1492). Each void exit returns before it
  (r16_pipeline.py:1522, 1525, 1529). The preflight refusal checks only `trial_lines(ledger) > 0`
  (research/intraday/research/run_r16.py:98-99).
- Effect: after a void, nothing in the ledger shows that the run happened (the prereg's trial
  count rule, section 12). A full run at a new SHA would also not be refused by the ledger rule.
- Change requested: if the full run voids, record it in the ledger or progress.md and in an
  amendment by hand, before any rerun.

**[warn] The per-bracket (S3) results that P3-8 promises are entry counts only, and they are not
in results.json.**
- Verified: `stage_sensitivities` writes `S3_attribution.parquet` with only
  `cell_id, brackets, n_entries` (r16_pipeline.py:1060-1069). P3-8 says "results are also
  reported per bracket ... a bracket whose result differs from the others"
  (PREREGISTRATION_R16.md:1306-1308, 1325-1326). Section 10.3 also asks for "per pool (S3)". No
  per-bracket return exists anywhere. `results.json.sensitivities` covers only S1, S2 and
  cost_x1_5 (r16_pipeline.py:1246-1249).
- Change requested: the G2 critique must state that the P3-8 bias check cannot be performed from
  the run's artifacts, or the controller must rule that the entry counts are enough.

**[note] Provenance (check 1): passes on smoke.** `results.json` carries `prereg_commit` 0322eb5…
(the Amendment-0-final commit, progress.md:87), `git_sha` 05a29ad…, `amendment0_sha` 9ae0d051…
and `provenance_sha`. All 14 stage DONE.json files exist. A DONE record's `inputs_hash` is the
sha256 of {git_sha, audit_sha, mode, sessions, stage} (r16_pipeline.py:1509-1515,
run_r16.py:106-107). No DONE record states the SHA in plain text, so SHA consistency rests on the
`<sha12>` directory name, or on recomputing the hash, which was not done here (no Python). The
trial ledger is full-mode only (r16_pipeline.py:1490), so smoke cannot test it.

**[note] Holdout guard (check 2): verified at the source.** `dev_sessions` asserts
max < 2025-08-07 (research/intraday/research/r16_audit.py:74-76). `make_folds` asserts the same
(research/intraday/research/r16_train.py:84). `TouchedDates.add` raises immediately on any holdout
date (run_r16.py:155-158), so the skipped `assert_dev` on void paths (r16_pipeline.py:1532) loses
nothing. SPY readers filter rows below the seal (research/intraday/research/r16_eval.py:332-347).
The null (b) target read calls `check_dev_date` (r16_nulls.py:101, r16_pipeline.py:750). The
largest touched date is not written to any artifact, so G2 can only check the fold `test` ranges
in results.json.

**[note] Multiplicity values (check 3): wired correctly.** m = 2 × number of (k, h, model) keys
(r16_nulls.py:234-236), and full mode has 18 keys (54 main cells / 3 exits, r16_config.py:153-155),
which gives m = 36. z* = Φ⁻¹(1 − 0.05/m) (r16_nulls.py:239-242). The rule is one-sided, and
z < −z* is labelled negative skill (r16_nulls.py:249-251). Smoke logged m = 2 and z* = 1.96, as
P3-15 requires. The C_b band uses its own |z| ≤ 3.66 (A0.4 item 9) and also voids
(r16_pipeline.py:836).

**[note] Selection and gate (checks 4, 5).** `dev_conditions` codes section 11.1.1-5 exactly:
geo CI lower bound > 0; the C_a/C_b/C_c paired CI lower bounds > 0 plus the C_b 95th percentile;
Holm < 0.05 and DSR ≥ 0.95; ex-top-1% > 0; and ≥ 300 trades on ≥ 150 days
(r16_eval.py:250-261). Holm runs over the main cells only (r16_pipeline.py:1153). N_DSR = 4248
(r16_config.py:90), and the variance is across the 54 cells with ddof 1 (r16_pipeline.py:1222-1230).
The bootstrap floor, 1/(B+1) with B = 10,000, allows Holm to pass across 54 cells. In smoke the DSR
is null because there is only one main cell (`sr_variance` reports "fewer than 2 main cells"); that
is expected. `dsr_failure` lives only in `evaluation.json`. The brief's "beats C_0" is not a
section 11.1.2 condition and has no main-vs-C_0 statistic. C_0 appears only in the full-mode H2
contrast `C_c − C_0`, so the brief's item 4 cannot be checked as written.

**[note] Parity and replay (check 6).** In smoke, parity determinism hashes match pairwise. Engine
replay `identical: true` over 180 cell-days, and the halt sensitivity is present. In full, replay
and the halt sensitivity exist only if a cell is selected (P3-4, ambiguity 13;
r16_pipeline.py:1286-1296). Two parity strata (eod_forced, gap_through_stop) were short in smoke;
the shortfall is recorded, never padded (r16_parity.py:107-114). Check it in full.

**[note] SPY (check 7): the brief is stale.** Ruling R8 (PREREGISTRATION_R16.md:645-646, 1088,
1547) fills 2025 dev days from `research_breakout/daily/bars_2025.parquet` (r16_config.py:28,
r16_eval.py:341-365). So in full, `missing_days` should be empty or near it, not "2025 missing".

**[note] Costs (check 8).** `c_label` is not in results.json or the costs DONE record (only
`refit_digest`). It is in `amendment0_inputs.json` and in every MLflow run's params
(r16_pipeline.py:1414). In smoke every fill was `refit` and bar-capped. That is plausible: cached
quotes are the 09:34/35 and 15:55 summaries only (section 9.3), and the 5-minute freshness rule
(section 8.3) keeps them out of a time-stop cell's fills. In full, check that the bracket and EOD
cells show some `quote` fills at 15:55.

**[note] Charts.** All 7 charts exist as PNG and HTML. Under full-run shapes the heatmaps (3 × 3 per
model × exit), the 54-line equity chart and the C_b 5-95% band (20 seeds) fill normally. On a
nulls-void path there are no grid series, so the charts render empty. results.json is written
before any chart (r16_report.py:330), so it survives even if a chart crashes. The full-only code in
evaluate and sensitivities (contrasts, S1/S2/cost × 1.5, S3) was never run by smoke. A code read
found no shape or dtype mismatch: trade and bracket dates are both datetime64 (r16_sim.py:472,
r16_pipeline.py:1061). It remains untested at scale.

**[note] Joint scale-invariance (section 10.7)** exists only as unit tests
(tests/test_intraday_model_features.py:24, tests/test_intraday_model_r16_universe.py:178). No run
artifact records it, so G2 cites the test suite at 05a29ad.

**[block — process, against the critic] Bash use outside the allowed commands.** This session ran
`git status --short`, `git ls-files | grep …` and `sed -n '101p' progress.md` in Bash. The rule
allows git log and git diff only. All three were read-only: nothing was written, no Python ran, no
process was touched, and no number was produced that is not already in an artifact. The required
`scripts/check_ownership.py` was not run, because the task forbids Python. Evidence: this
session's tool log.

---

## Verdict

**BLOCK** on the null (a) account finding: under the current rule, a spurious VOID in the full run
is likely, and only a controller ruling made before evaluate runs can prevent it. The four warns do
not stop the run; G2 must read past results.json as described. The process finding against the
critic is recorded per the role and does not concern the subject.
