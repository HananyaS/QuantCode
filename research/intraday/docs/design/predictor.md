# PredictorAgent design

Owner: `intraday-model`. Package: `research/intraday/model/`. Spec §2 (agent table), §4 (factors
h and model, the controls). Read with `research/intraday/contracts/` and ADR-001. Builds after the
R16 pre-registration is committed (spec §3).

## Responsibility

Map each `Window` to one `Signal` using a frozen scorer whose identity is logged as `model_hash`.

## Consumes

- `Window`: the only input. Features are computed from `Window.bars` and nothing else.

## Emits

- `Signal`, exactly one per `Window`: `ts` = `Window.ts`, `symbol` = `Window.symbol`,
  `score` = the scorer's output (finite; larger means a better long entry), `horizon_min` = `h`,
  `model_hash` = the scorer's identity (below).

## State

- The loaded scorer, immutable after construction. No per-symbol or per-event state: each
  `Window` is scored independently. Size: the model artifact (a GBDT or a small CNN/transformer,
  megabytes at most).

## Configuration

| Parameter | Default | Source |
|---|---|---|
| `scorer` | `"model"`; controls `"random"` and `"constant"` | spec §4 (main arm; "a random feature used as the score"; "the screen alone with no model") |
| `artifact_path` | none (required when `scorer="model"`) | builder's call |
| `expected_model_hash` | none (required when `scorer="model"`) | builder's call |
| `h` (`horizon_min`) | `15` (tests and smoke runs only) | R16 (h ∈ {5, 15, 30}, spec §4) |
| `k` (window length expected) | `15` (tests and smoke runs only) | R16 (k ∈ {5, 15, 30}); must equal the StateAgent's |
| model family | logistic/GBDT on window aggregates, or a small 1-D CNN/transformer on the raw window | R16 (spec §4) |
| `random_seed` (for `scorer="random"`) | `0` | builder's call |

The scorers:
- `"model"`: loads the frozen artifact at construction. `model_hash` = sha256 hex of the
  artifact file's bytes; it must equal `expected_model_hash` or construction raises. Inference
  only: training happens in research scripts (`research/intraday/research/`), never in the agent.
  CPU, single thread, deterministic mode for torch models.
- `"random"`: `score` = a uniform in [0, 1) derived from
  `sha256(f"{random_seed}|{symbol}|{ts.isoformat()}")`, so it is deterministic and independent of
  the window's content. `model_hash = f"random-v1-seed{random_seed}"`.
- `"constant"`: `score = 1.0` for every window. `model_hash = "constant-v1"`. With the RiskAgent's
  one-entry-per-symbol rule this enters each candidate at its first window: the screen-only arm.

Features: ratios within the window only (returns relative to the newest close, high/low range over
close, volume relative to the window's median volume, close relative to vwap). **Never a price,
share-volume or dollar-volume level** (`research/penny/features/intraday_cs.py` docstring and LOG
94: a level feature was the project's second-strongest fake signal). Every feature must be
invariant when all prices in a window are multiplied by a constant `c` and all volumes divided by
`c` (penny §99: the leak guards rescale price and volume together). The served bars are unadjusted
(`feed.md`), while the cached history is adjusted by a factor constant within each day; invariance
is what makes a model trained on either basis serve the same scores. Research scripts build training
windows from bars converted the same way as ReplayFeed, so training and serving see one basis.

Reuse (under `research/penny/`): `model/sequence.py` (`SequenceConfig`, `build_sequences`,
`train_sequence_model`) and `model/train.py` (`ModelConfig`, `train_one_fold`,
`walk_forward_scores`) for training in research scripts; `eval/folds.py` (`walk_forward_folds`,
`assert_folds_are_clean`) for the folds; `eval/nulls.py` (`run_all`) for the null battery.

## Invariants

- One `Signal` per `Window`, same `ts` and `symbol`.
- `score` depends only on `Window.bars` (or, for `"random"`, on seed, symbol and ts).
- `model_hash` is the same for every `Signal` in a run and identifies the scorer exactly.
- The agent never reads a file after construction, never trains, never uses the wall clock.
- No feature is a price or dollar-volume level.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| Artifact hash differs from `expected_model_hash` | sha256 at construction | raise |
| `len(Window.bars) != k` | length check | raise (the StateAgent and the predictor disagree on k) |
| The model returns NaN or inf | finiteness check before building the `Signal` (the contract would also raise) | raise (a model that cannot score a valid window is broken) |
| The artifact cannot be loaded | exception at construction | raise |

## Replay vs live

None. The same artifact, same code, no adapter.

## Test plan

`tests/test_intraday_model_predictor.py`:
- `test_one_signal_per_window_same_ts_and_symbol`
- `test_signal_horizon_is_h`
- `test_model_hash_is_sha256_of_artifact`
- `test_hash_mismatch_raises_at_construction`
- `test_window_length_not_k_raises`
- `test_non_finite_score_raises`
- `test_random_scorer_is_deterministic_and_content_free`
- `test_constant_scorer_scores_one`
- `test_features_are_scale_free` (prices × 10 and volumes ÷ 10 in a window leave the score unchanged)

## Open questions

- **[research]** k and h are R16 factors (each in {5, 15, 30}), and the model family is an R16 factor.
  The agent takes all three from configuration; R16 runs the grid and records the trial count.
- **[research]** Whether float scores are bit-identical across machines (torch kernels, BLAS).
  Phase 2's hash test runs on one machine; the nightly live-versus-replay diff (Phase 5) must run
  on the same host as live, or compare scores with a tolerance instead of by hash.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/6
