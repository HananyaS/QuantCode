# Intraday Phase 3 — StateAgent, PredictorAgent and the Round-16 Research Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the StateAgent and PredictorAgent, then the R16 pipeline that runs the pre-registration on development dates only: the provenance audit, the gated universe G(D), labels, features, walk-forward fits, a vectorised grid simulator proven equal to the engine, the controls, the account evaluation, MLflow tracking, a results writer and a run script that refuses to break the protocol.

**Architecture:** Part A has two pure agents on the Phase-1 bus. Each is tested alone against contract fixtures, and the arithmetic they share with research (window features, the CNN, the scorers, the artifact format) lives in `research/intraday/model/`. Part B lives in `research/intraday/research/`. It composes the merged agents and adapters (data, risk, exec, ops), which research may import. Agents never import one another. One fast simulator mirrors the engine's event semantics rule by rule and has to equal the engine to the cent on the pre-registered parity set before any grid number is read. Part C is one run script (`audit` / `smoke` / `full`). Part D writes the results.

**Tech Stack:** Python 3.13, pandas 2.3, numpy 2.3, pyarrow, LightGBM 4.6, torch 2.9.1+cpu, scipy, mlflow 3.16 (local file store), matplotlib 3.10 (PNG), plotly 6.3 (HTML; no kaleido is installed), pytest.

**Spec:** `research/penny/PREREGISTRATION_R16.md` at commit **4fa3cee** (final after the G1 re-check; binding; cited by §). Design docs: `research/intraday/docs/design/{state,predictor,screener,feed,risk,execution,recorder,engine}.md`. System spec: `docs/superpowers/specs/2026-10-26-intraday-system-design.md`. Interfaces built against: `docs/superpowers/plans/2026-10-26-intraday-p2-{data,risk,exec,ops}.md`.

## Global Constraints

- Execution worktree: branch `research/intraday-model`, cut from `research/intraday` **after** the four Phase-2 branches are merged into it. Never push. Never merge to main.
- Write **only** `research/intraday/state/`, `research/intraday/model/`, `research/intraday/research/`, `tests/test_intraday_state*`, `tests/test_intraday_model*`. Every test for Part B goes in `tests/test_intraday_model_r16_*.py`. Before every commit, `"$PY" scripts/check_ownership.py --agent intraday-model --base research/intraday` prints nothing.
- `PY="C:/Users/Shachar/AppData/Local/Programs/Python/Python313/python.exe"` (it is not on PATH). Commands run from the worktree root in Git Bash.
- TDD: write the failing test first. Never edit a test to make it pass. The full suite (`"$PY" -m pytest tests/ -q`) passes before every commit.
- Contracts are read-only. Event fields are strict: int fields get `int(...)` (never `np.int64`), float fields get `float` (`0.0`, never `0`), tuple fields get `tuple`, and datetimes are `datetime` with `tzinfo is timezone.utc` (`.astimezone(timezone.utc)`; never a `pd.Timestamp`). `Bar.ts == bar_start + 1 min`.
- Never emit while iterating a `set`/`frozenset`/dict built from one; iterate a sorted tuple. No wall clock and no unseeded randomness anywhere under `research/intraday/`, research included (`tests/test_intraday_guards.py` enforces both). Run durations come from the MLflow run start/end times, which MLflow stamps, not our code. No `alpaca` import at all.
- Import rules: `state/` and `model/` import only `research.intraday.contracts`, their own package, stdlib, numpy, lightgbm and torch (model only). `research/` may import every intraday package, the engine and `research.penny`.
- Data (read-only, from the main checkout, never copied): `DATA_ROOT = C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_penny`, `VOLUME_ROOT = C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_breakout/daily`, `SPY_CACHE = C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/cache/SPY.parquet`. **No downloads.** Prices in the minute cache are split- and dividend-adjusted. Volume is split-adjusted only: raw shares = adjusted ÷ `s`, with `s = volume / volume_unadj` (§4.4 V1).
- Outputs go outside the worktree (gitignored there): `OUT_ROOT = C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_intraday`. Under it: `r16/<git_sha12>/<stage>/…`, `mlruns/` (the MLflow store), `logs/` (ops `LOG_ROOT`), `r16_trials.jsonl`. Storage under 5 GB (§6.4).
- Holdout: `HOLDOUT_START = date(2025, 8, 7)`. This round has **no unseal path at all**: `UNSEAL: Final = False` is a module constant, no CLI flag exists, and nothing accepts an `unseal` argument other than `False` (prereg header box C1: no unseal before the user confirms U1). Every date that enters any stage passes `check_dev_date`.
- **Sealed rows are never loaded.** Every read of a dated file that spans the boundary passes `filters=[("date", "<", pd.Timestamp(HOLDOUT_START))]` to `pd.read_parquet`. That covers penny `daily/bars_2025.parquet`, `bars_2026.parquet`, `VOLUME_ROOT/bars_2025.parquet` (SPY and V1) and the pool files, and a post-read `assert df.date.max() < HOLDOUT_START` follows. Per-day caches (`minute/dt=…`, `premarket/dt=…`) are opened only for dates that passed `check_dev_date`.
- Every MLflow run carries the tags `conditional_on=U1`, `universe=G(D)-gated`, `period=dev` (§12, header box).
- Torch: CPU only. `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(1)`, and Python/NumPy/torch seeds fixed (§5.4). One process per fit. Parallelism comes from a process pool, never from threads.
- Tests use `fixtures.T_DEV` (2024-03-04 14:31 UTC = 09:31 EST) or explicit dev dates, never the fixtures' default `T0`. Real-data tests skip cleanly when `DATA_ROOT/minute` is absent.
- Commit trailers (two lines, last in the message): `Refs: HananyaS/QuantCode-intraday-board#22` (StateAgent tasks), `#6` (PredictorAgent / model tasks), `#30` (every R16 pipeline task), then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Upstream prerequisites (requests to the controller; verified in Task 16 Step 1)

The prereg binds the engine, so the engine has to implement these. The Phase-2 plans as written do not:

- **UP-1 (intraday-risk):** `RiskConfig.atr_floor: float | None = None`. When set, ATR = `max(mean_true_range, atr_floor)` (§5.5 `atr_floor` = $0.01; ATR skips must be 0 at G2).
- **UP-2 (intraday-exec):** a refit cost book in place of the constant `default_half_spread_bps` (§9.2–9.3). `CostInputs` gains six optional floats: `hs_open_bps, hs_close_bps, ask_notional_open, ask_notional_close, bid_notional_open, bid_notional_close`. When there is no fresh quote, `select_cost_book` returns `CostBook(interp(hs), interp(side notional), "refit")`, where `interp(x) = x_open` for ET minute `t ≤ 09:35`, `x_close` for `t ≥ 15:55`, and linear in clock minutes between them, with `t` = the fill instant's ET clock time in minutes. Every cost book's half-spread (quote or refit) is floored at `0.5 × 0.01 / fill_price × 1e4` bps. A symbol-day with no refit book raises. Research passes `decision_summary=None`, so the `"summary"` source is never used. `research/r16_costs.py::refit_book_at` (Task 10) is the reference implementation, and Task 16 tests that exec agrees with it on a grid.
- **Not required for R16:** the ScreenerAgent's R16 config (G(D), `pm_rvol`, 09:30 stamping). Research injects the `Candidates` event itself (Task 16). G3 builds the live screener later (§13).

If UP-1 or UP-2 is missing when Task 16 starts, that task stops and reports **BLOCKED (UP-n)**. Tasks 1–15 do not depend on either.

## Prereg ambiguities this plan resolves (each pinned by a test; the controller routes them into Amendment 0)

1. **Training subsample (compute).** A fixed, nested, outcome-free subsample: the `GBDT_TRAIN_CAP = 1_000_000` / `CNN_TRAIN_CAP = 200_000` windows with the smallest `subsample_key(symbol, ts)` (splitmix64, seed 16) among each fit's training windows. The same rule picks `CNN_VAL_CAP = 50_000` early-stopping windows. Inference and thresholds use every window.
2. **Tradable windows only.** Training, validation, threshold and AUC use windows with `ts < 15:30` ET (`last_entry_et`). Later windows can never trade.
3. **k-bar return.** `ret_n = log(close[-1] / open[-n])`, so the k-bar return stays inside the window. `vol_rel` uses the window median volume, or the mean when the median is 0, or 0.0 when every volume is 0.
4. **`c_label` round trip.** The entry leg uses the open-window refit prediction (ask side) and the exit leg the close-window prediction (bid side), at $2,000, as in `etl/quotes.py::round_trip_cost_bps`. The median is over dev no-selection symbol-days.
5. **Interpolation.** Linear in level (bps, $ notional) in ET clock minutes, between 09:35 and 15:55 (UP-2).
6. **C_b, C_c and C_0 do not depend on the model factor.** 27 × 20 = 540 unique C_b runs serve all 54 main cells (the gbdt and cnn twins share them). They are reported as the prereg's 1,080 pairings.
7. **Null (a) "falls to the null":** permuted-label OOF AUC within 0.5 ± 3 × SE, with SE from a day-clustered bootstrap (B = 200, seed 16). Its grid account's paired difference against C_b (seed mean) has a 95% CI containing 0. **Null (b) "same symbol slot":** the label of the same symbol and the same ET minute on the dev session 20 sessions later, dropped if absent. A session ≥ `HOLDOUT_START` is never read.
8. **C_b null band (§7 N5):** for each seed, |z| ≤ Φ⁻¹(1 − 0.01/80) = 3.66 for both AUC and IC (Spearman of score against `y`). z uses a day-clustered bootstrap SE (B = 200), and 0.01/80 is Bonferroni over 20 seeds × 2 statistics. One-sided p (§10.4) = `(1 + #{b: m*_b − m ≥ m}) / (B + 1)` (centred bootstrap).
9. **DSR variance:** `Var(SR_n)` across the 54 main cells is on the annualised scale. It is passed to `eval/stats.py::deflated_sharpe_ratio` as `var / 252` (that function works per period).
10. **Selection ties after the prereg's list:** gbdt > cnn, smaller k, time_stop > fixed_pct > atr, then smaller h. The C_b 95th percentile uses `np.percentile(…, 95)` (linear).
11. **Gross account:** the net run's fills with `cost_bps` set to 0, over the same `E_{d−1}`. Trades are identical by construction.
12. **Smoke runs:** pipeline tests only. Experiment `intraday-smoke`, their own fold scheme, 60 dev sessions, no trial-ledger line, never cited as R16 evidence.
13. **Engine full-period replay (§6.4, §11.1.6)** runs only if some cell meets §11.1.1–5. If none does, H1 fails on development and the parity test is the validity evidence. The halt-inference sensitivity (§5.6) and the engine-only parts of §10.7 run on that replay too, or not at all.
14. **S1/S2/cost × 1.5** are re-simulated grids. **S3** attributes the primary trades to G brackets (no re-simulation).
15. **Pool rules for G(D):** the prereg's stated rules are frozen constants in code (`POOL_RULES`, Task 8). If Amendment 0 records a different true rule (§4.4 1b, E7), changing it is a new reviewed task, not a config edit.

## Review Focus

1. **Daylight saving in features and labels.** The minute-of-day channels and the 15:30/15:55 cuts must follow America/New_York per date. In July 09:30 ET = 13:30 UTC. Pinned: Task 2 `test_tod_follows_dst`, Task 12 `test_labels_clip_at_1555_in_summer`.
2. **Equal-timestamp ordering.** All bars and fills at ts T reach RiskAgent before any Signal at T, so a slot freed by an exit filled on the bar ending at T can be reused by a signal at T, and a loss-limit trip at T blocks signals at T. Pinned: Task 15 `test_slot_freed_by_exit_fill_reused_same_minute`, `test_loss_limit_trip_blocks_same_minute_signal`. Also the parity test (Task 17).
3. **Zero-volume bars.** The contract allows `volume == 0`. Features and CNN channels must stay finite, and the bar cap must give 0 shares (a `zero_qty` skip, not an entry). Pinned: Task 2 `test_zero_volume_window_features_finite`, Task 15 `test_zero_qty_skip_does_not_count_as_entry`.
4. **End of session.** No entry at ts ≥ 15:30. The time stop clips to 15:55. With no bar after the time stop, the position fills at the last close at 16:01 (uncapped). The label applies the same rule. Pinned: Task 12 `test_label_falls_back_to_last_close`, Task 15 `test_eod_fill_at_last_close_uncapped`.
5. **The holdout boundary everywhere a date enters.** 2025-08-06 is allowed and 2025-08-07 is refused. That covers the null-(b) label shift, the SPY 2025 rows and the panel builder. Pinned: Task 6 `test_check_dev_date_boundary`, Task 18 `test_shifted_label_never_reads_holdout`, Task 19 `test_spy_never_reads_holdout`.

---

## File Structure

```
research/intraday/state/agent.py          StateAgent
research/intraday/state/__init__.py       re-export StateAgent
research/intraday/model/features.py       WindowArrays, arrays_from_bars, gbdt_features, cnn_input (shared with research)
research/intraday/model/cnn.py            make_cnn, cnn_scores_float64 (deterministic inference)
research/intraday/model/artifact.py       ArtifactMeta, write_artifact, read_artifact, file_sha256
research/intraday/model/scorers.py        WindowScorer, GbdtScorer, CnnScorer, random_uniform, load_scorer
research/intraday/model/agent.py          PredictorAgent
research/intraday/model/__init__.py       re-exports
research/intraday/research/r16_config.py      constants, Cell grid, configs, date/git guards, paths
research/intraday/research/r16_provenance.py  §4.3 audit (symbol/date/daily only)
research/intraday/research/r16_universe.py    §4.4 steps 1–3, G(D), screen score, top_n
research/intraday/research/r16_panel.py       per-day traded-basis bars/quotes via ReplayFeed, cached
research/intraday/research/r16_costs.py       §9 refit, refit_book_at, cost-input table, c_label, CS/Roll check
research/intraday/research/r16_amendment0.py  Amendment-0 inputs (counts only) and infeasibility check
research/intraday/research/r16_windows.py     window tensors (StateAgent semantics), labels, subsample_key
research/intraday/research/r16_train.py       folds, inner split, GBDT/CNN fits, thresholds, artifacts
research/intraday/research/r16_scores.py      batch scoring, random/constant scores, eligible-window tables
research/intraday/research/r16_sim.py         engine-mirror day/cell simulator
research/intraday/research/r16_engine.py      engine session harness (composes the merged agents)
research/intraday/research/r16_parity.py      §6.4 parity configurations, stratified days, comparison
research/intraday/research/r16_nulls.py       §10.7 nulls, martingale null, C_b band, scale test
research/intraday/research/r16_eval.py        §10 metrics, bootstraps, Holm, DSR, §11.1 conditions, §10.5 selection, SPY
research/intraday/research/r16_tracking.py    MLflow tracker, tags, trial ledger
research/intraday/research/r16_report.py      results JSON + PNG/HTML charts
research/intraday/research/run_r16.py         audit | smoke | full, with refusals and stage checkpoints
tests/test_intraday_state.py                  Task 1 (state.md test names)
tests/test_intraday_model_features.py         Task 2
tests/test_intraday_model_scorers.py          Task 3
tests/test_intraday_model_predictor.py        Task 4 (predictor.md test names)
tests/test_intraday_model_bus.py              Task 5
tests/test_intraday_model_r16_<module>.py     Tasks 6–22, one file per module
```

## Compute budget (12-core Windows laptop, CPU only, `--workers 8` by default)

| Stage | Work | Estimate |
|---|---|---|
| audit (Task 11) | pools ⋈ requested.json, daily rules, refit | 15 min |
| panel (Task 9) | ReplayFeed for about 2,370 dev days | 10 min |
| windows / labels / subsample (Task 12) | about 19M windows per k, 3 k | 20 min |
| GBDT fits (Task 13) | 108 fits ≤ 1M rows, `n_jobs=1`, 8 parallel | 15 min |
| CNN fits (Task 13) | 108 fits ≤ 200k windows, ≤ 40 epochs, 1 thread each, 8 parallel | 75 min |
| scoring (Task 14) | all OOS + inner-validation windows, 18 (k,h,model); C_b sha256 × 20 seeds | 35 min |
| nulls (Task 18) | 18 one-fold refits × 2 + martingale + C_b band | 40 min |
| parity (Task 17) | 74 configs × 50 days engine sessions + 5 days × 2 determinism | 20 min |
| grid (Task 15) | 54 + 54 + 540 + 27 + 27 unique cell-runs × about 1,870 days; then S1, S2, cost × 1.5 | 30 + 60 min |
| evaluation (Task 19) | shared bootstrap index matrices, B = 10,000 | 15 min |
| engine replay of the selected cell (conditional) | 24 configs × about 1,870 sessions, 8 parallel | 60 min |
| report (Task 21) | JSON + charts | 5 min |
| **Full run** | | **≈ 6.5 h with the conditional engine replay, ≈ 5.5 h without; hard budget 10 h** |
| **Smoke** | 60 sessions, one cell + its controls, 3 parity days | **≈ 15 min** |

The smoke run's MLflow stage durations are multiplied by the full-run scale factors that `run_r16 smoke` prints (days × cells), and that projection is checked against the 10 h budget before a full run starts.

---

## Part A — Agents (fixture-driven, no data)

### Task 1: StateAgent

**Files:**
- Create: `research/intraday/state/agent.py`; modify `research/intraday/state/__init__.py`
- Test: `tests/test_intraday_state.py`

**Interfaces:**
- Consumes: contracts `Bar`, `Gap`, `Candidates`, `Window`, `Event`; fixtures `make_bar`, `make_gap`, `make_candidates`, `T_DEV`.
- Produces: `class StateAgent` with `__init__(self, k: int = 15, max_span_min: int | None = None, require_contiguous: bool = False, name: str = "state")` (`max_span_min=None` means `2 * k`), `subscribes = (Candidates, Bar, Gap)` and `on_event(self, event: Event) -> tuple[Event, ...]`. `ValueError` at construction unless `type(k) is int and k >= 1` and `max_span_min >= k`.
- Behaviour (state.md): a second `Candidates` raises `ValueError`. `Bar`/`Gap` before `Candidates`, or for a non-candidate, is skipped. A `Bar` whose `bar_start <=` the symbol's newest raises `ValueError`. A `Gap` is stored in a per-symbol deque of `missing_bar_start`s, pruned below the oldest buffered bar's `bar_start`. On a `Bar` the agent appends, and when `len(buf) == k`, `span = (buf[-1].bar_start − buf[0].bar_start) / 1 min + 1 <= max_span_min` and (if `require_contiguous`) no stored gap inside the span, it emits `Window(ts=bar.ts, symbol=bar.symbol, bars=tuple(buf))`.

- [ ] **Step 1: Write the failing tests.** These are the eleven state.md names: `test_no_window_until_k_bars`, `test_window_has_last_k_bars_newest_last`, `test_window_ts_equals_triggering_bar_ts`, `test_gap_never_becomes_a_bar`, `test_gap_does_not_emit_window`, `test_span_over_max_suppresses_window`, `test_non_candidate_bar_ignored`, `test_out_of_order_bar_raises`, `test_one_window_per_symbol_per_minute`, `test_state_bounded_by_k_per_symbol`, `test_window_bars_is_tuple_snapshot_of_deque`. Add these three:

```python
def test_span_rule_counts_gap_minutes() -> None:      # k=3, max_span 6
    a = StateAgent(k=3); a.on_event(make_candidates(ts=T_DEV, symbols=("A",)))
    for m in (0, 1, 5):                                # span = 5 - 0 + 1 = 6 -> emitted
        out = a.on_event(_bar("A", m))
    assert len(out) == 1
    assert a.on_event(_bar("A", 9)) == ()              # buffer 1,5,9: span 9 -> suppressed
    (w,) = a.on_event(_bar("A", 10))                   # 5,9,10: span 6
    assert [b.bar_start for b in w.bars] == [_bs(5), _bs(9), _bs(10)]

def test_require_contiguous_rejects_gap_inside_span() -> None:
    a = StateAgent(k=2, require_contiguous=True); a.on_event(make_candidates(ts=T_DEV, symbols=("A",)))
    a.on_event(_bar("A", 0)); a.on_event(make_gap(ts=_bs(1) + MIN, symbol="A", missing_bar_start=_bs(1)))
    assert a.on_event(_bar("A", 2)) == ()
    assert len(a.on_event(_bar("A", 3))) == 1

def test_second_candidates_raises() -> None:
    a = StateAgent(); a.on_event(make_candidates(ts=T_DEV))
    with pytest.raises(ValueError):
        a.on_event(make_candidates(ts=T_DEV))
```
(`_bs(m) = T_DEV - MIN + m*MIN`; `_bar(sym, m)` = `make_bar(symbol=sym, bar_start=_bs(m), ts=_bs(m)+MIN)`.)

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_state.py -q`. Expected: FAIL (`ImportError: cannot import name 'StateAgent'`).
- [ ] **Step 3: Implement** `StateAgent` per the Interfaces block (`collections.deque(maxlen=k)` per symbol; `symbols` is a frozenset used only for membership).
- [ ] **Step 4: Run** the same command. Expected: 14 passed. Then the full suite and the ownership check.
- [ ] **Step 5: Commit** `feat(intraday-model): StateAgent rolling k-bar windows with span rule` with `Refs: HananyaS/QuantCode-intraday-board#22`.

---

### Task 2: Window feature functions (one implementation for the agent and for research)

**Files:**
- Create: `research/intraday/model/features.py`
- Test: `tests/test_intraday_model_features.py`

**Interfaces:**
- Produces:
  - `FEATURE_VERSION: int = 1`; `GBDT_FEATURES = ("ret_1", "ret_3", "ret_k", "range_1", "range_k", "vwap_dev", "vol_rel", "up_frac", "n_missing", "tod")`; `CNN_CHANNELS = ("log_open", "log_high", "log_low", "log_close", "log_vwap", "log1p_vol_rel", "tod")`.
  - `@dataclass(frozen=True) class WindowArrays`: `open, high, low, close, vwap, volume, minute` (each a C-contiguous `float64` array of shape `(n, k)`; `minute` = minutes from 09:30 ET to each bar's `ts`) and `span_min` (`(n,)` float64). `__post_init__` asserts the shapes agree and `k >= 3`.
  - `minutes_since_open(ts: datetime) -> float`, with ET via `zoneinfo.ZoneInfo("America/New_York")` on `ts`'s ET date.
  - `arrays_from_bars(bars: Sequence[Bar]) -> WindowArrays` (n = 1).
  - `gbdt_features(w: WindowArrays) -> np.ndarray` of shape `(n, 10)`, in the `GBDT_FEATURES` order. `ret_1 = log(c[-1]/o[-1])`, `ret_3 = log(c[-1]/o[-3])`, `ret_k = log(c[-1]/o[0])`, `range_1 = (h[-1]-l[-1])/c[-1]`, `range_k = (max h - min l)/c[-1]`, `vwap_dev = c[-1]/(Σ vwap·v / Σ v) - 1` (with mean of `vwap` when `Σ v == 0`), `vol_rel = v[-1]/denom` (denom = median, else mean, else ratio 0.0), `up_frac = mean(c > o)`, `n_missing = span_min - k`, `tod = minute[-1]/390`.
  - `cnn_input(w: WindowArrays) -> np.ndarray` of shape `(n, 7, k)` float64: `log(x / c[:, -1:])` for open/high/low/close/vwap, `log1p(v / denom)`, `minute / 390`.
  - Every reduction runs along axis 1 of a C-contiguous array, so row i's result does not depend on n.

- [ ] **Step 1: Write the failing tests**

```python
def test_feature_names_and_shapes() -> None:
    w = arrays_from_bars(make_window(5, end_ts=T_DEV + 4 * MIN).bars)
    assert gbdt_features(w).shape == (1, 10) and cnn_input(w).shape == (1, 7, 5)

def test_features_are_scale_free_x750() -> None:        # §10.7 joint rescale
    w = _random_arrays(seed=3, n=64, k=15)
    s = replace(w, **{f: getattr(w, f) * 750 for f in ("open", "high", "low", "close", "vwap")},
                volume=w.volume / 750)
    np.testing.assert_allclose(gbdt_features(s), gbdt_features(w), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cnn_input(s), cnn_input(w), rtol=1e-12, atol=1e-12)

def test_batch_and_single_rows_bit_identical() -> None:
    w = _random_arrays(seed=4, n=200, k=30)
    full = gbdt_features(w)
    for i in (0, 57, 199):
        assert np.array_equal(gbdt_features(_row(w, i)), full[i:i + 1])
        assert np.array_equal(cnn_input(_row(w, i)), cnn_input(w)[i:i + 1])

def test_tod_follows_dst() -> None:
    winter = datetime(2024, 3, 4, 14, 35, tzinfo=timezone.utc)   # 09:35 EST
    summer = datetime(2024, 7, 1, 13, 35, tzinfo=timezone.utc)   # 09:35 EDT
    assert minutes_since_open(winter) == minutes_since_open(summer) == 5.0

def test_zero_volume_window_features_finite() -> None:
    w = _random_arrays(seed=5, n=3, k=5); w = replace(w, volume=np.zeros_like(w.volume))
    assert np.isfinite(gbdt_features(w)).all() and np.isfinite(cnn_input(w)).all()

def test_n_missing_is_span_minus_k() -> None:
    w = _random_arrays(seed=6, n=1, k=5); w = replace(w, span_min=np.array([8.0]))
    assert gbdt_features(w)[0, GBDT_FEATURES.index("n_missing")] == 3.0

def test_no_level_feature() -> None:                      # predictor.md: never a price/volume level
    w = _random_arrays(seed=7, n=8, k=5)
    shifted = replace(w, volume=w.volume * 3)
    assert np.array_equal(gbdt_features(shifted), gbdt_features(w))
```
(`_random_arrays(seed, n, k)`: a seeded random walk with positive OHLC consistent with `low <= min(o,c)`, integer volumes in 0..5000, minutes `5..5+k-1`, `span_min = k`. `_row(w, i)` slices every field to `[i:i+1]` with `np.ascontiguousarray`.)

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_features.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `features.py` per the Interfaces block (numpy only, no pandas).
- [ ] **Step 4: Run.** Expected: 7 passed. Full suite; ownership check.
- [ ] **Step 5: Commit** `feat(intraday-model): scale-free window features shared by agent and research` with `Refs: HananyaS/QuantCode-intraday-board#6`.

---

### Task 3: CNN, artifact format and scorers

**Files:**
- Create: `research/intraday/model/cnn.py`, `research/intraday/model/artifact.py`, `research/intraday/model/scorers.py`
- Test: `tests/test_intraday_model_scorers.py`

**Interfaces:**
- Consumes: Task 2 `WindowArrays`, `gbdt_features`, `cnn_input`, `FEATURE_VERSION`; `research.intraday.contracts.codec.canonical_json`.
- Produces:
  - `cnn.py`: `make_cnn(n_channels: int = 7, hidden: int = 32, dropout: float = 0.3) -> torch.nn.Sequential`, with the exact layer sequence of `research/penny/model/sequence.py::_make_model("cnn")` (Conv1d k5 p2 → ReLU → BatchNorm1d → Dropout → Conv1d k5 p2 → ReLU → AdaptiveAvgPool1d(1) → Flatten → Dropout → Linear(h, 1)), with `n_channels` replacing `len(CHANNELS)`. Also `set_torch_determinism(seed: int) -> None` (the §5.4 flags plus `random.seed`, `np.random.seed`, `torch.manual_seed`). And `cnn_scores_float64(state: Mapping[str, np.ndarray], hidden: int, dropout: float, x: np.ndarray) -> np.ndarray`: builds the net in float64, loads `state`, `eval()`, runs every layer but the last under `torch.backends.mkldnn.flags(enabled=False)` with 1 thread, then `logit = (z * w).sum(-1) + b` in float64 and returns `scipy.special.expit(logit)`.
  - `artifact.py`: `ARTIFACT_VERSION = 1`; `@dataclass(frozen=True) class ArtifactMeta`: `family: Literal["gbdt", "cnn"]`, `k: int`, `h: int`, `fold: int`, `feature_version: int`, `c_label: float`, `threshold: float`, `train_digest: str`, `best_epoch: int | None`. `write_artifact(path: Path, meta: ArtifactMeta, payload: Mapping[str, object]) -> str` writes `canonical_json({"v", "meta", "payload"}) + "\n"` in mode `"x"` and returns its sha256 hex. `read_artifact(path: Path, expected_hash: str) -> tuple[ArtifactMeta, dict]` raises `ValueError` on a hash mismatch. `file_sha256(path: Path) -> str`. Payloads: gbdt `{"booster": <LightGBM model string>}`; cnn `{"hidden", "dropout", "state": {name: {"shape", "dtype": "float32", "b64"}}}`, with names sorted.
  - `scorers.py`: `class WindowScorer(Protocol)` with `model_hash: str`, `k: int`, `h: int` and `score(self, w: WindowArrays) -> np.ndarray`. `GbdtScorer` uses `lightgbm.Booster(model_str=...)`, `predict(gbdt_features(w), num_threads=1)`. `CnnScorer` uses `cnn_scores_float64`. `load_scorer(path: Path, expected_hash: str) -> WindowScorer`. `random_uniform(seed: int, symbol: str, ts: datetime) -> float` = `int.from_bytes(sha256(f"{seed}|{symbol}|{ts.isoformat()}".encode()).digest()[:8], "big") / 2**64`. `RANDOM_HASH_FMT = "random-v1-seed{seed}"`; `CONSTANT_HASH = "constant-v1"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_cnn_layers_match_penny_head() -> None:
    from research.penny.model.sequence import SequenceConfig, _make_model
    ours, theirs = make_cnn(n_channels=6), _make_model("cnn", SequenceConfig())
    assert [type(m).__name__ for m in ours] == [type(m).__name__ for m in theirs]

def test_cnn_batch_and_single_scores_bit_identical() -> None:
    set_torch_determinism(42)
    state = {k: v.detach().numpy() for k, v in make_cnn().state_dict().items()}
    x = cnn_input(_random_arrays(seed=1, n=300, k=15))
    full = cnn_scores_float64(state, 32, 0.3, x)
    assert all(np.array_equal(cnn_scores_float64(state, 32, 0.3, x[i:i + 1]), full[i:i + 1]) for i in range(0, 300, 37))

def test_gbdt_batch_and_single_scores_bit_identical(tmp_path) -> None:
    path, h = _tiny_gbdt_artifact(tmp_path)              # 400 synthetic rows, ModelConfig(n_estimators=20)
    s = load_scorer(path, h); w = _random_arrays(seed=2, n=50, k=15)
    assert all(np.array_equal(s.score(_row(w, i)), s.score(w)[i:i + 1]) for i in range(50))

def test_artifact_round_trip_and_hash(tmp_path) -> None:
    h = write_artifact(tmp_path / "a.json", META, {"booster": "x"})
    assert h == file_sha256(tmp_path / "a.json")
    assert read_artifact(tmp_path / "a.json", h)[0] == META
    with pytest.raises(ValueError):
        read_artifact(tmp_path / "a.json", "0" * 64)
    with pytest.raises(FileExistsError):
        write_artifact(tmp_path / "a.json", META, {"booster": "x"})

def test_artifact_bytes_deterministic(tmp_path) -> None:
    assert write_artifact(tmp_path / "1.json", META, PAY) == write_artifact(tmp_path / "2.json", META, PAY)

def test_random_uniform_known_value_and_range() -> None:
    u = random_uniform(0, "AAA", T_DEV)
    assert 0.0 <= u < 1.0 and u == random_uniform(0, "AAA", T_DEV) != random_uniform(1, "AAA", T_DEV)
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_scorers.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules. Import torch lazily inside the cnn functions, so a gbdt-only process never loads it.
- [ ] **Step 4: Run.** Expected: 6 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(intraday-model): deterministic CNN inference, hashed artifacts, scorers` with `Refs: HananyaS/QuantCode-intraday-board#6`.

---

### Task 4: PredictorAgent

**Files:**
- Create: `research/intraday/model/agent.py`; modify `research/intraday/model/__init__.py` (re-export `PredictorAgent`, `load_scorer`, `random_uniform`)
- Test: `tests/test_intraday_model_predictor.py`

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces: `class PredictorAgent` with `__init__(self, scorer: Literal["model", "random", "constant"], *, h: int = 15, k: int = 15, artifact_path: Path | None = None, expected_model_hash: str | None = None, random_seed: int = 0, name: str = "predictor")`, `subscribes = (Window,)`, property `model_hash -> str`, and `on_event(event) -> tuple[Signal]`. Construction raises `ValueError` on: an unknown scorer; `scorer == "model"` without the path or hash; a hash mismatch (from `read_artifact`); artifact `meta.k != k` or `meta.h != h`. `on_event`: `len(window.bars) != k` raises `ValueError`. `score = float(...)`. A non-finite score raises `ValueError` before the `Signal` is built. Emits `Signal(ts=window.ts, symbol=window.symbol, score=score, horizon_min=h, model_hash=self.model_hash)`. The `"constant"` scorer scores `1.0`; `"random"` uses `random_uniform(random_seed, symbol, window.ts)`.

- [ ] **Step 1: Write the failing tests.** These are the nine predictor.md names: `test_one_signal_per_window_same_ts_and_symbol`, `test_signal_horizon_is_h`, `test_model_hash_is_sha256_of_artifact`, `test_hash_mismatch_raises_at_construction`, `test_window_length_not_k_raises`, `test_non_finite_score_raises` (an artifact whose booster yields NaN is not constructible, so test it with a monkeypatched scorer returning `np.nan`), `test_random_scorer_is_deterministic_and_content_free` (two windows with the same symbol and ts but different bars give the same score), `test_constant_scorer_scores_one`, `test_features_are_scale_free` (prices × 10 and volumes ÷ 10 on volumes divisible by 10, same score from a tiny gbdt artifact). Add:

```python
def test_artifact_k_h_must_match_config(tmp_path) -> None:
    path, h = _tiny_gbdt_artifact(tmp_path, k=15, h=5)
    with pytest.raises(ValueError, match="h"):
        PredictorAgent("model", k=15, h=15, artifact_path=path, expected_model_hash=h)

def test_score_is_builtin_float() -> None:
    (s,) = PredictorAgent("random", k=3).on_event(make_window(3, end_ts=T_DEV))
    assert type(s.score) is float and type(s.horizon_min) is int
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_predictor.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `agent.py`.
- [ ] **Step 4: Run.** Expected: 11 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(intraday-model): PredictorAgent with model, random and constant scorers` with `Refs: HananyaS/QuantCode-intraday-board#6`.

---

### Task 5: State + Predictor on the bus; determinism across hash seeds

**Files:**
- Test: `tests/test_intraday_model_bus.py`

**Interfaces:**
- Consumes: `EventBus`, `run_replay`, `clock_ticks` (engine); `read_log` (ops); Tasks 1 and 4. A test-only `ScriptedFeed` agent (`name="feed"`, subscribes `ClockTick`) releases fixture bars for three symbols, one with gaps, at each tick.

- [ ] **Step 1: Write the tests**

```python
def test_cascade_bar_window_signal_carry_cause_ts(tmp_path) -> None:
    envs = list(_run(tmp_path / "a.jsonl"))
    sig = [e for e in envs if isinstance(e.event, Signal)]
    by_seq = {e.seq: e for e in envs}
    assert sig and all(isinstance(by_seq[e.cause_seq].event, Window) for e in sig)
    assert all(e.event.ts == by_seq[e.cause_seq].event.ts for e in sig)

def test_two_runs_identical_hash(tmp_path) -> None:
    assert _hash(tmp_path / "1.jsonl") == _hash(tmp_path / "2.jsonl")

def test_hash_identical_across_pythonhashseed(tmp_path) -> None:   # the tests/test_intraday_replay.py subprocess pattern
    assert _hash_in_subprocess(tmp_path / "a.jsonl", "0") == _hash_in_subprocess(tmp_path / "b.jsonl", "12345")
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_bus.py -q`. Expected: PASS. This task adds no production code, so a FAIL is a bug in Tasks 1–4: reproduce it as a unit test there, fix, rerun.
- [ ] **Step 3: Guards and ownership.** `"$PY" -m pytest tests/test_intraday_guards.py -q` passes. Ownership prints nothing.
- [ ] **Step 4: Commit** `test(intraday-model): state and predictor on the bus, deterministic across hash seeds` with `Refs: HananyaS/QuantCode-intraday-board#22` and `Refs: HananyaS/QuantCode-intraday-board#6`.

---

## Part B — the R16 pipeline (development dates only)

Every Part-B commit carries `Refs: HananyaS/QuantCode-intraday-board#30`.

### Task 6: Constants, the cell grid, configs and guards

**Files:**
- Create: `research/intraday/research/__init__.py` (empty), `research/intraday/research/r16_config.py`
- Test: `tests/test_intraday_model_r16_config.py`

**Interfaces:**
- Consumes: `RiskConfig` (risk), `ExecConfig` (exec), `SealedHoldoutError` (data.errors).
- Produces (all values verbatim from the prereg section cited):
  - Paths: `REPO_ROOT`, `PREREG_PATH = REPO_ROOT/"research/penny/PREREGISTRATION_R16.md"`, plus `DATA_ROOT`, `VOLUME_ROOT`, `SPY_CACHE`, `SPY_2025 = VOLUME_ROOT/"bars_2025.parquet"` (§7 R8), `OUT_ROOT`, `MLRUNS_DIR = OUT_ROOT/"mlruns"`, `TRIAL_LEDGER = OUT_ROOT/"r16_trials.jsonl"` (§12), `BREAKOUT_LEDGER = C:/Users/Shachar/Desktop/PythonProjects/QuantCode/research/breakout/ledger.jsonl`.
  - Dates: `HOLDOUT_START = date(2025, 8, 7)`, `DEV_FIRST = date(2016, 2, 17)`, `UNSEAL: Final = False`. `check_dev_date(d: date) -> None` raises `SealedHoldoutError` for `d >= HOLDOUT_START`; it has no unseal parameter.
  - Factors (§6.1): `K_LEVELS = (5, 15, 30)`, `H_LEVELS = (5, 15, 30)`, `MODELS = ("gbdt", "cnn")`, `EXITS = ("time_stop", "fixed_pct", "atr")`. TP/SL (§5.5): `TP_SL_PCT = {5: (0.0375, 0.025), 15: (0.06375, 0.0425), 30: (0.09, 0.06)}`, `TP_SL_ATR = {5: (2.10, 1.40), 15: (3.63, 2.42), 30: (5.13, 3.42)}`, `ATR_PERIOD = 5`, `ATR_FLOOR = 0.01`.
  - §5.1/§4.4: `MIN_PRICE = 1.0`, `MAX_PRICE = 10.0`, `MIN_ADV_USD = 1_000_000.0`, `ADV_WINDOW = 20`, `TOP_N = 10`, `SPLIT_LOG_STEP = math.log(1.25)`, `UNADJUST_TOL = 0.02`, `BASIS_RATIO_MAX = 1.5`.
  - §5.5/§5.6: `THRESHOLD_QUANTILE = 0.90`, `RANDOM_THRESHOLD = 0.90`, `CONSTANT_THRESHOLD = 1.0`, `C_B_SEEDS = tuple(range(20))`.
  - §6.3/§10: `N_PRIOR_TRIALS = 4194`, `N_R16 = 54`, `N_DSR = 4248`, `BOOT_B = 10_000`, `BOOT_SEED = 16`, `STATIONARY_BLOCK = 5.0`, `PARITY_N_DAYS = 50`, `PARITY_PER_STRATUM = 10`, `PARITY_SEED = 16`, `DETERMINISM_DAYS = 5`, `FOLDS = dict(n_splits=6, embargo_days=5, expanding=True, min_train_days=504)`, `INNER_FIT_FRACTION = 0.8`.
  - Ambiguity 1: `GBDT_TRAIN_CAP = 1_000_000`, `CNN_TRAIN_CAP = 200_000`, `CNN_VAL_CAP = 50_000`, `SUBSAMPLE_SEED = 16`.
  - `Arm = Literal["main", "C_a", "C_b", "C_c", "C_0"]`; `@dataclass(frozen=True, order=True) class Cell`: `arm`, `k`, `h`, `model: str | None`, `exit`, `seed: int | None`, with properties `cell_id` (`M-k15-h5-gbdt-atr`, `Ca-k15-h5-gbdt-atr`, `Cb-k15-h5-atr-s07`, `Cc-k15-h5-atr`, `C0-k15-h5-atr`), `universe_arm` (`"screen"` for main/C_b/C_c, `"no_selection"` for C_a/C_0) and `scorer` (`"model"`, `"random"`, `"constant"`).
  - `main_cells() -> tuple[Cell, ...]` (54, sorted); `unique_runs() -> tuple[Cell, ...]` (54 main + 54 C_a + 540 C_b + 27 C_c + 27 C_0 = 702); `controls_of(main: Cell) -> dict[str, tuple[Cell, ...]]` (keys `C_a`, `C_b` (20), `C_c`, `C_0`).
  - `risk_config(cell: Cell, threshold: float, start_equity: float) -> RiskConfig` (§5.5 keys; `atr_floor=ATR_FLOOR` for atr, per UP-1) and `exec_config() -> ExecConfig` (`quote_size_frac=0.5`, `quote_max_age_min=5`, `bar_participation_cap=0.02`, `entry_max_wait_min=2`, `eod_et=time(16, 1)`, `impact_coefficient=1.0`, `levels_penalty=1.0`).
  - `git_state(repo: Path) -> tuple[str, bool]`: `(HEAD sha, clean)`, where clean ⇔ `git status --porcelain --untracked-files=all -- research/` is empty. `prereg_commit(repo: Path) -> str` = `git log -1 --format=%H -- research/penny/PREREGISTRATION_R16.md`.

- [ ] **Step 1: Write the failing tests**

```python
def test_grid_sizes() -> None:
    assert len(main_cells()) == 54 == N_R16
    runs = unique_runs()
    assert Counter(c.arm for c in runs) == {"main": 54, "C_a": 54, "C_b": 540, "C_c": 27, "C_0": 27}
    assert len({c.cell_id for c in runs}) == 702
    assert len(controls_of(main_cells()[0])["C_b"]) == 20

def test_cell_id_format() -> None:
    assert Cell("main", 15, 5, "gbdt", "atr", None).cell_id == "M-k15-h5-gbdt-atr"
    assert Cell("C_b", 15, 5, None, "atr", 7).cell_id == "Cb-k15-h5-atr-s07"

def test_check_dev_date_boundary() -> None:
    check_dev_date(date(2025, 8, 6))
    with pytest.raises(SealedHoldoutError):
        check_dev_date(date(2025, 8, 7))
    assert UNSEAL is False and "unseal" not in inspect.signature(check_dev_date).parameters

def test_risk_config_per_exit() -> None:
    m = Cell("main", 15, 30, "gbdt", "fixed_pct", None)
    rc = risk_config(m, 0.61, 25_000.0)
    assert (rc.tp_pct, rc.sl_pct, rc.atr_period, rc.quote_size_frac) == (0.09, 0.06, 5, 0.5)
    assert risk_config(replace(m, exit="atr"), 0.61, 25_000.0).sl_atr == 3.42

def test_dsr_trial_count() -> None:
    assert N_DSR == N_PRIOR_TRIALS + N_R16 == 4248
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_config.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `r16_config.py`. `test_risk_config_per_exit` needs UP-1 for `atr_floor`: until that merges, the atr branch of `risk_config` raises `NotImplementedError("UP-1")` and the test is collected with `pytest.importorskip`-style gating on `"atr_floor" in RiskConfig.__dataclass_fields__` (skip reason "UP-1 pending").
- [ ] **Step 4: Run.** Expected: PASS (one skip possible). Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): frozen constants, cell grid and dev-date guard`.

---

### Task 7: Provenance and admissibility audit (§4.3)

**Files:**
- Create: `research/intraday/research/r16_provenance.py`
- Test: `tests/test_intraday_model_r16_provenance.py`

**Interfaces:**
- Consumes: `data.cache.minute_cache_membership`, `data.paths.requested_file`/`premarket_file`; `r16_config`.
- Produces:
  - `POOL_FILES: dict[str, PoolSpec]` for the §4.2 table. `PoolSpec(name, path, rule_doc, membership_time, cls: Literal["C0", "C1", "C2"])`, where `candidates_r15` and `hyp_pool` are split by row rule: R15 signal session C2 / entry session C0; `hyp_pool` C2 iff `open_unadj / prev_close_unadj >= 1.8`, else C0.
  - `read_pool_keys(spec: PoolSpec) -> pd.DataFrame` with only `symbol, date` (plus the two columns the hyp_pool/R15 split needs), read via `pd.read_parquet(columns=...)`.
  - `reproduce_pool(spec, daily: pd.DataFrame) -> PoolReproduction(n_file, n_rule, n_sym_diff, frac_diff, differing: pd.DataFrame)` (step 1a; >1% ⇒ differing rows are U).
  - `classify(cache_keys: pd.DataFrame, pools: Mapping[str, pd.DataFrame], repro: Mapping[str, PoolReproduction]) -> pd.DataFrame` with columns `symbol, date, classes` (sorted tuple of str), `admissible` and `rescue_pool`.
  - `completeness(pools, cache_keys) -> pd.DataFrame` (`pool, year, share`) (step 4a; a rescue needs `share >= 0.99`).
  - `run_audit(cache: Literal["minute", "premarket"], dev_dates: Sequence[date], daily: pd.DataFrame) -> AuditResult(table: pd.DataFrame, counts: dict, completeness: pd.DataFrame, repro: dict)`.
  - `write_audit(result: AuditResult, out: Path) -> str` writes `r16_provenance_<cache>.parquet` and returns its sha256 (§4.3 output path `OUT_ROOT/r16_provenance.parquet` = both caches concatenated with a `cache` column).
- Guard: the module never reads a minute or pre-market parquet. Cache membership comes from `requested.json`, or `minute_cache_membership`'s fallback. `assert_no_price_reads` wraps `pd.read_parquet` during `run_audit` and raises if a requested column set includes `open/high/low/close/vwap` of a minute or pre-market file.

- [ ] **Step 1: Write the failing tests** (synthetic pools and daily rows under `tmp_path` via `data.synthetic`):

```python
def test_c1_and_c2_member_is_admissible() -> None:
    t = classify(_keys(("A", D)), {"candidates_r4": _keys(("A", D)), "forensic_fetch": _keys(("A", D))}, REPRO_OK)
    assert t.loc[0, "classes"] == ("C1", "C2") and bool(t.loc[0, "admissible"])

def test_only_c2_or_u_excluded() -> None:
    t = classify(_keys(("A", D), ("B", D)), {"forensic_fetch": _keys(("A", D))}, REPRO_OK)
    assert t.set_index("symbol")["admissible"].to_dict() == {"A": False, "B": False}
    assert t.set_index("symbol").loc["B", "classes"] == ("U",)

def test_hyp_pool_gap_ge_1_8_is_c2() -> None: ...
def test_r15_signal_vs_entry_session_classes() -> None: ...
def test_reproduction_over_1pct_marks_rows_u() -> None: ...
def test_rescue_needs_99pct_completeness() -> None: ...
def test_audit_reads_no_minute_prices(tmp_path, monkeypatch) -> None:
    run_audit("minute", (D,), _daily(tmp_path), root=tmp_path)      # raises if a price column is read
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_provenance.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Rules for step 1a come from the pool builders cited in §4.2: `etl/candidates.py::CandidateSpec` for R1–3 (and R4 with `min_gap=0.03`, `min_adv_20=5e6`), the R5 rule as stated in §4.2, and the R15 rule over 5 sessions. Dollar ADV follows each builder's own definition (`etl/bars.py::add_cross_day_features` `adv_20`). Vectorised groupby/rolling only.
- [ ] **Step 4: Run.** Expected: 7 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): provenance and admissibility audit, symbol/date only`.

---

### Task 8: Universe G(D) and the screen (§4.4, §5.1)

**Files:**
- Create: `research/intraday/research/r16_universe.py`
- Test: `tests/test_intraday_model_r16_universe.py`

**Interfaces:**
- Consumes: `data.universe.causal_universe`/`UniverseResult`, `data.ranking.convert_premarket`, `data.cache.read_premarket_day`, `data.basis.*`; Task 7's tables; `r16_config`.
- Produces:
  - `POOL_RULES: tuple[GateBracket, ...]`, where `GateBracket(name, gap_lo, gap_hi, adv_def, adv_min, crash: bool)` encodes the four §4.4 1b brackets. Year spans come from the audit's pool spans (`spans: Mapping[str, tuple[date, date]]`).
  - `gate(daily_before: pd.DataFrame, open_d: pd.Series, d: date, spans) -> pd.DataFrame` (index symbol; bool column per bracket and `in_g`). `open_d` is D's adjusted official open, the only D price read. The gap is `open_D / close_{D−1} − 1` on the adjusted basis.
  - `@dataclass(frozen=True) class UniverseDay`: `session_date`, `no_selection: tuple[str, ...]`, `screen: tuple[str, ...]` (≤ 10, sorted), `screen_score: pd.Series`, `brackets: pd.DataFrame`, `pm_status: pd.Series` (`"bars" | "requested_no_bars"`), `prior: pd.DataFrame` (from `UniverseResult`), `counts: dict[str, int]`. The counts keys are `step1, gate, cache, provenance, pm_not_requested, pm_c2u, basis_check, feed_reject, no_volume_factor, top_n_binds`.
  - `build_universe_day(d: date, daily_before: pd.DataFrame, open_d: pd.Series, f_d: pd.Series, provenance: pd.DataFrame, pm_provenance: pd.DataFrame, pm: pd.DataFrame, minute_first_open: pd.Series, feed_rejects: Mapping[str, str], spans, variant: Literal["primary", "S1", "S2"] = "primary") -> UniverseDay`. The order is exactly §4.4: step 1 (`causal_universe`, `MIN/MAX_PRICE`, `MIN_ADV_USD`, `ADV_WINDOW`, split flags) → 1b `gate` → 2 cache ∩ admissible minute provenance (S1 keeps C2/U) → pm not requested ⇒ removed; pm only C2/U ⇒ removed (C3) → 2b basis check `pm_last_close × f_{D−1}` vs `minute_first_open × f_D` within factor 1.5 → feed rejects (all kinds) ⇒ removed. S2 drops the crash bracket. Every removal is counted.
  - `screen_scores(pm_traded: pd.DataFrame, prior: pd.DataFrame, members: tuple[str, ...]) -> pd.Series`: `pm_gap = pm_close_last / close_unadj_{D−1} − 1` (prices already × f_{D−1} by `convert_premarket`), `pm_rvol = Σ_{04:00 ≤ bar_start < 09:25} close·volume / adv20_raw`; `score = mean(rank(pct=True, method="average"))` over members with pm bars; descending, ties by symbol; `top_n = 10`.

- [ ] **Step 1: Write the failing tests**

```python
def test_gate_brackets() -> None:      # gaps +12% (ADV 2M), +5% (ADV 2M), +5% (ADV 6M), -5%, -15%, +1% crash-entry, +1%
    g = gate(...)
    assert g["in_g"].to_dict() == {"G12": True, "G5lo": False, "G5hi": True, "N5": True,
                                   "N15": False, "CRASH": True, "FLAT": False}

def test_gate_reads_only_open_of_d() -> None:   # daily_before with a D row raises LookAheadError
def test_pm_not_requested_removed_from_every_arm() -> None:
def test_requested_no_bars_stays_in_no_selection_skipped_by_screen() -> None:
def test_pm_only_c2_or_u_removed_and_counted() -> None:           # C3
def test_basis_check_factor_1_5() -> None:
def test_feed_reject_removed_before_ranking() -> None:
def test_screen_ranks_within_g_top10_ties_by_symbol() -> None:
    u = build_universe_day(...)          # 12 members with pm bars, two tied
    assert len(u.screen) == 10 and u.counts["top_n_binds"] == 1 and u.screen == tuple(sorted(u.screen))
def test_screen_score_invariant_to_rescale() -> None:              # prices x750, volume /750
def test_s2_drops_crash_bracket() -> None:
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_universe.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** with vectorised pandas. `f_D` is used only for step 2b (prereg §10.7 C2). Assert `daily_before.date.max() < d`.
- [ ] **Step 4: Run.** Expected: 10 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): gated universe G(D) and the 09:30 in-gate screen`.

---

### Task 9: Panel builder (traded-basis bars through the merged ReplayFeed)

**Files:**
- Create: `research/intraday/research/r16_panel.py`
- Test: `tests/test_intraday_model_r16_panel.py`

**Interfaces:**
- Consumes: `data.replay_feed.ReplayFeed` (`open_session(d, symbols, screen_id) -> SessionData`), `data.rejects.Reject`; `r16_config.check_dev_date`.
- Produces:
  - `PANEL_COLUMNS = ("symbol", "bar_start_ns", "open", "high", "low", "close", "vwap", "volume")` (volume int64, prices float64, `bar_start_ns` int64 UTC ns); `QUOTE_COLUMNS = ("symbol", "ts_ns", "bid", "ask", "bid_size", "ask_size")`.
  - `session_to_frames(data: SessionData) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Reject, ...]]`, sorted by `(symbol, bar_start_ns)`. The values are the `Bar` fields as they are, with no recomputation.
  - `build_panel_day(feed: ReplayFeed, d: date, symbols: tuple[str, ...], out_dir: Path) -> PanelDay(path_bars, path_quotes, rejects)`. Writes `dt=D/bars.parquet`, `quotes.parquet`, `rejects.jsonl`. Idempotent (skips an existing complete day). `check_dev_date(d)` runs first.
  - `load_panel_day(out_dir: Path, d: date) -> tuple[pd.DataFrame, pd.DataFrame]`.
  - `ReplayFeed` is constructed with `unseal=False` literally.

- [ ] **Step 1: Write the failing tests**

```python
def test_frames_equal_session_data(tmp_path) -> None:     # synthetic day incl. a dividend symbol (f=1.345, s=1)
    data = ReplayFeed(root, root / "v").open_session(D, ("A", "B"), "screen:x:2024-03-04")
    bars, quotes, rej = session_to_frames(data)
    for row in bars.itertuples():
        b = data.bars[(row.symbol, _utc(row.bar_start_ns))]
        assert (row.open, row.close, row.volume) == (b.open, b.close, b.volume)

def test_holdout_day_refused(tmp_path) -> None:
    with pytest.raises(SealedHoldoutError):
        build_panel_day(ReplayFeed(root, root / "v"), date(2025, 8, 7), ("A",), tmp_path)

def test_rejected_symbol_has_no_bars(tmp_path) -> None: ...
def test_build_is_idempotent(tmp_path) -> None: ...
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_panel.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 4 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): traded-basis panel built through ReplayFeed`.

---

### Task 10: Cost refit, refit book, cost-input table, `c_label` (§9)

**Files:**
- Create: `research/intraday/research/r16_costs.py`
- Test: `tests/test_intraday_model_r16_costs.py`

**Interfaces:**
- Consumes: `data.replay_quotes.read_quote_summaries`; `research.penny.etl.quotes.leg_cost_bps`, `round_trip_cost_bps`; `research.penny.eval.battery.corwin_schultz_spread`, `roll_spread`; `exec.costs.CostInputs`, `StaticCostInputs`.
- Produces:
  - `fit_refit(fit_set: pd.DataFrame) -> RefitModel` (frozen dataclass: `coef: dict[str, dict[str, float]]` for the targets `log_hs`, `log_ask_notional`, `log_bid_notional`; `r2: dict[str, float]`; `years: tuple[int, ...]`; `digest: str` = sha256 of canonical JSON). The OLS regressors are fixed by §9.3: `log(close_unadj_{D−1})`, `log(adv20_usd)` (raw × raw), `log(σ_20d)`, `window_close` (0 = 09:34/35, 1 = 15:55) and year dummies, via `numpy.linalg.lstsq`.
  - `predict_books(model: RefitModel, rows: pd.DataFrame) -> pd.DataFrame` with the six UP-2 fields per `(symbol, date)`. A year after the last fitted year uses the last dev year's dummy (§9.3, E15).
  - `refit_book_at(minute_et: float, open_: float, close_: float) -> float`: the UP-2 interpolation (reference implementation). Plus `half_tick_floor_bps(price: float) -> float` = `0.5 * 0.01 / price * 1e4`.
  - `cost_inputs_table(daily: pd.DataFrame, books: pd.DataFrame, sessions: Sequence[date]) -> StaticCostInputs`. `sigma_daily` = sample std (ddof 1) of adjusted `close.pct_change()` over the last 20 returns before D. `adv_shares` = median of the last 20 sessions' split-adjusted `volume_v` before D ÷ `s_{D−1}` (§9.1 V1). A missing value means no entry, and the universe step then counts the day as `no_volume_factor`.
  - `c_label(books, inputs, prices) -> float` (ambiguity 4). Its unit is a **return fraction** (median round-trip bps ÷ 1e4), compared directly with `fwd_ret` in Task 12. and `spread_crosscheck(panel_days, books) -> pd.DataFrame` (per time-of-day bucket: median CS, Roll and refit, plus a flag when CS or Roll exceeds the refit by > 50%).

- [ ] **Step 1: Write the failing tests**

```python
def test_refit_recovers_known_coefficients() -> None:      # synthetic data from known betas + noise(seed 16)
def test_interpolation_endpoints_and_midpoint() -> None:
    assert refit_book_at(5.0, 80.0, 40.0) == 80.0 and refit_book_at(385.0, 80.0, 40.0) == 40.0
    assert refit_book_at(195.0, 80.0, 40.0) == pytest.approx(60.0)
    assert refit_book_at(0.0, 80.0, 40.0) == 80.0 and refit_book_at(391.0, 80.0, 40.0) == 40.0
def test_adv_shares_median_over_split_only_factor() -> None:   # dividend f=1.345, s=1 -> volume unchanged
def test_adv_shares_reverse_split_inside_window() -> None:     # one basis via s_{D-1}
def test_sigma_uses_prior_sessions_only() -> None:
def test_c_label_is_a_single_constant() -> None:
def test_holdout_year_uses_last_dev_dummy() -> None:
```
(minute_et here = minutes after 09:30 ET; 09:35 = 5.0, 15:55 = 385.0.)
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_costs.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 7 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): cost refit on the no-selection population, refit book, c_label`.

---

### Task 11: Amendment-0 inputs and the infeasibility check (§6.5 step 2, §11.4)

**Files:**
- Create: `research/intraday/research/r16_amendment0.py`
- Test: `tests/test_intraday_model_r16_amendment0.py`

**Interfaces:**
- Consumes: Tasks 7, 8, 10; `pyarrow.parquet.read_schema` and min/max of the index for SPY.
- Produces:
  - `amendment0_inputs(...) -> dict` holding the class counts by year/pool/cache, pool reproducibility and completeness, pool spans, `|G ∩ cache| / |G|` by year and bracket, basis-check and `no_volume_factor` counts per year, the `top_n` binding share, the median admissible names per day after step 1, `c_label`, the refit digest, coefficients and R², the SPY spans (`SPY_CACHE` schema plus index min/max; `SPY_2025` rows with `symbol == "SPY"` and `date < HOLDOUT_START`, min/max only), the `BREAKOUT_LEDGER` line count, the `search_trials.jsonl` line count, and the provenance sha256s.
  - `audit_feed_rejects(d: date, symbols: tuple[str, ...]) -> dict[str, str]` (symbol → reject kind). It applies ReplayFeed's `open_session` rules 3–6 (data plan Task 9) but reads only the minute columns `symbol, minute_et, open` and daily rows, so the audit's universe and refit fit set exist before any minute return is read. The panel stage later asserts that its ReplayFeed rejects equal these, day by day. A difference stops the run.
  - `half_days(dates) -> tuple[date, ...]`: sessions whose minute file has `max(minute_et) < "13:00"`, reading the `minute_et` column only (§4.4 step 3).
  - `infeasible(inputs: dict) -> bool` (§11.4: C_0 cannot reach ≥ 300 trades on ≥ 150 days by counts, or the median admissible after step 1 is < 3).
  - `write_amendment0(inputs, out: Path) -> str` (canonical JSON; returns its sha256).
  - `amendment0_recorded(prereg_text: str, audit_sha: str) -> bool`: true iff the prereg contains a heading starting `### Amendment 0` **and** the literal `audit_sha`.
- The only data this module reads: `symbol`, `date`, daily rows, `requested.json`, quote summaries, D's official open, D's first regular minute open (step 2b), the SPY schema and index span, and ledger line counts. No return of any kind is computed (§4.3, §6.5).

- [ ] **Step 1: Write the failing tests**: `test_audit_feed_rejects_equal_replayfeed_rejects` (synthetic day with one of each reject kind, compared with `ReplayFeed.open_session(...).rejects`), `test_audit_feed_rejects_read_only_open_column`, `test_half_days_from_minute_et_only`, `test_infeasible_when_median_below_3`, `test_amendment0_recorded_requires_heading_and_sha`, `test_spy_span_reads_no_prices` (monkeypatched `read_parquet` columns check), `test_inputs_are_canonical_json`.
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_amendment0.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 7 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): Amendment-0 inputs from counts only, infeasibility check`.

---

### Task 12: Windows and labels (§5.3, §5.4)

**Files:**
- Create: `research/intraday/research/r16_windows.py`
- Test: `tests/test_intraday_model_r16_windows.py`

**Interfaces:**
- Consumes: `model.features.WindowArrays`, `minutes_since_open`; Task 9 frames; `state.StateAgent` (tests only).
- Produces:
  - `@dataclass(frozen=True) class DayWindows`: `symbol: np.ndarray[str]`, `ts_ns: np.ndarray[int64]`, `end_idx: np.ndarray[int64]` (row index of the newest bar in the day's bars frame), `arrays: WindowArrays` (n rows).
  - `day_windows(bars: pd.DataFrame, k: int, max_span_min: int | None = None) -> DayWindows` uses `numpy.lib.stride_tricks.sliding_window_view` per symbol. A window ending at bar i is valid iff `i >= k−1` within its symbol and `(bar_start[i] − bar_start[i−k+1]) / 60e9 + 1 <= 2k`. It asserts every window's last `bar_ts <= ts` (§10.7 c). Output order is `(ts, symbol)`.
  - `labels(bars: pd.DataFrame, windows: DayWindows, h: int, c_label: float, session_date: date) -> pd.DataFrame` with columns `p_in, p_out, fwd_ret, y, entry_bar_idx, exit_bar_idx, valid`. Rules (§5.4): `p_in` = open of the symbol's first bar with `T <= bar_start < T + 2 min`, else `valid=False`. `p_out` = open of the first bar with `bar_start >= min(T + h, 15:55 ET)`, else the last close of the session (`exit_bar_idx = -1`). `y = fwd_ret > c_label`. `tradable = ts < 15:30 ET`.
  - `subsample_key(symbols: np.ndarray, ts_ns: np.ndarray, seed: int = SUBSAMPLE_SEED) -> np.ndarray[uint64]`: splitmix64 over `(sha256(symbol) first 8 bytes as uint64) ^ ts_ns ^ seed`, vectorised.

- [ ] **Step 1: Write the failing tests**

```python
def test_windows_identical_to_state_agent_on_synthetic_day() -> None:   # 3 symbols, random gaps, k in (5, 15, 30)
    for k in (5, 15, 30):
        ours = day_windows(BARS, k)
        agent = _agent_windows(BARS, k)                 # StateAgent fed Candidates + bars in feed order
        assert [(w.symbol, w.ts) for w in agent] == list(zip(ours.symbol, _utc(ours.ts_ns)))
        assert all(np.array_equal(gbdt_features(arrays_from_bars(w.bars)), gbdt_features(_row(ours.arrays, i)))
                   for i, w in enumerate(agent))

@pytest.mark.skipif(not (DATA_ROOT / "minute").is_dir(), reason="caches absent")
def test_windows_identical_to_state_agent_on_20_fixed_dev_days() -> None:   # §5.3 parity, k=15
    ...                                               # the 20 days: every 100th dev session from DEV_FIRST

def test_label_entry_wait_two_minutes() -> None:
def test_label_falls_back_to_last_close() -> None:
def test_labels_clip_at_1555_in_summer() -> None:      # 2024-07-01, h=30 window at 15:29 EDT
def test_tradable_mask_excludes_1530_and_later() -> None:
def test_window_never_contains_future_bar() -> None:   # corrupted ts order raises AssertionError
def test_subsample_key_deterministic_and_order_free() -> None:
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_windows.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** `minute` per bar = `(bar_ts − 09:30 ET of that date)` in minutes, computed once per day via `minutes_since_open` on the 390 distinct minute stamps.
- [ ] **Step 4: Run.** Expected: 8 passed (or 7 + 1 skip). Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): StateAgent-identical training windows and engine-rule labels`.

---

### Task 13: Folds and fits (§5.4, §5.5, §10.1)

**Files:**
- Create: `research/intraday/research/r16_train.py`
- Test: `tests/test_intraday_model_r16_train.py`

**Interfaces:**
- Consumes: `research.penny.eval.folds.walk_forward_folds`, `assert_folds_are_clean`; `research.penny.model.train.ModelConfig`; `research.penny.model.sequence.SequenceConfig`; `model.cnn.make_cnn`, `set_torch_determinism`, `cnn_scores_float64`; `model.artifact.write_artifact`; Task 12.
- Produces:
  - `make_folds(dev_dates: Sequence[date]) -> list[Fold]`: `walk_forward_folds(dev_dates, **FOLDS)` then `assert_folds_are_clean(folds, 5)`, and asserts `max(dev_dates) < HOLDOUT_START`.
  - `inner_split(train_dates: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]`: the first `floor(0.8 × n)` dates and the rest.
  - `TrainingSet(x: np.ndarray, y: np.ndarray, dates: np.ndarray, keys: np.ndarray)`. `select_capped(ts: TrainingSet, dates: pd.DatetimeIndex, cap: int) -> TrainingSet` keeps the `cap` smallest keys among rows in `dates`, then orders the result by `(date, symbol, ts)`.
  - `fit_gbdt(train: TrainingSet) -> lightgbm.LGBMClassifier` with `ModelConfig().to_lgb()` plus `n_jobs=1, deterministic=True, force_row_wise=True`.
  - `fit_cnn(train: TrainingSet, val: TrainingSet | None, epochs: int | None) -> tuple[dict[str, np.ndarray], int]`. With `val`, it early-stops under the `SequenceConfig` defaults (40 epochs, patience 6, batch 256, AdamW lr 1e-3 wd 1e-4, dropout 0.3, `pos_weight` and `clip_grad_norm_(1.0)` as `train_sequence_model`, permutation from `torch.Generator().manual_seed(42 + epoch)`) and returns `(best_state, best_epoch)`. With `val=None` it trains exactly `epochs` epochs (C4) and returns the final state.
  - `fit_fold(k, h, family, fold, windows_store, c_label) -> FoldArtifacts(inner_path, inner_hash, threshold, path, model_hash, best_epoch)`. Steps: fit on the inner 80% (capped); `threshold = np.quantile(scores of every tradable window on the inner-validation dates, 0.90)`; refit on 100% (capped) with the threshold held fixed (CNN: `best_epoch` epochs); write `r16/<sha>/models/{family}-k{k}-h{h}-f{fold}.json`.
  - `train_all(...)` runs `fit_fold` for the 108 `(k, h, family, fold)` combinations in a `ProcessPoolExecutor(max_workers)`. Each worker calls `set_torch_determinism(42)`. Results are collected in sorted key order.

- [ ] **Step 1: Write the failing tests**

```python
def test_folds_are_clean_and_dev_only() -> None:
    folds = make_folds(_dev_dates(1100))
    assert len(folds) == 6 and all(f.test_end < pd.Timestamp(HOLDOUT_START) for f in folds)

def test_inner_split_is_date_blocked() -> None:
    fit, val = inner_split(pd.DatetimeIndex(_dev_dates(100)))
    assert len(fit) == 80 and fit.max() < val.min()

def test_select_capped_nested_and_deterministic() -> None:
def test_gbdt_fit_deterministic(tmp_path) -> None:           # two fits -> identical artifact sha256
def test_cnn_fit_deterministic(tmp_path) -> None:            # 2,000 synthetic windows, 3 epochs, twice -> same sha
def test_cnn_refit_runs_best_epoch_count() -> None:          # C4
def test_threshold_from_inner_validation_only() -> None:     # test-date scores changed -> threshold unchanged
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_train.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Training windows come from Task 12 per day, kept only where `tradable & valid`, under the no-selection universe of the training dates (§5.4). Construct the CNN in float32 for training and score in float64 through `cnn_scores_float64`.
- [ ] **Step 4: Run.** Expected: 7 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): walk-forward GBDT/CNN fits, inner-block thresholds, hashed artifacts`.

---

### Task 14: Scores for the grid (main, C_a, C_b, C_c, C_0)

**Files:**
- Create: `research/intraday/research/r16_scores.py`
- Test: `tests/test_intraday_model_r16_scores.py`

**Interfaces:**
- Consumes: `model.scorers.load_scorer`, `random_uniform`; Tasks 12–13.
- Produces:
  - `score_oos(k, h, family, fold_artifacts, days) -> None` writes `r16/<sha>/scores/{family}-k{k}-h{h}.parquet` with the columns `date, symbol, ts_ns, score` (float64) for OOS windows of the **S1 superset** of the no-selection universe (primary plus C2/U, so S1 needs no rescoring) with `score >= threshold` and `tradable`. **Label validity is never a filter for scoring or eligibility**, because the engine scores every window it sees. `valid` filters only training, early-stopping validation and AUC/IC. The threshold quantile is over every tradable inner-validation window, valid or not, because that is the population the engine scores. It also writes `diag-{family}-k{k}-h{h}.parquet`: the 2,000,000 smallest-`subsample_key` OOS tradable valid windows (`date, score` float32, `y` int8, `fwd_ret` float32) for AUC/IC.
  - `random_scores(seed: int, symbols, ts_ns) -> np.ndarray` (the loop over `random_uniform`; ISO strings precomputed per distinct minute).
  - `eligible_signals(cell: Cell, d: date, universe: UniverseDay, store) -> pd.DataFrame` with `ts_ns, symbol, score`, sorted by `(ts_ns, symbol)`. Model: stored rows ∩ the cell's universe arm. Random: windows of the screen universe with `u >= 0.90`. Constant: every tradable window of the arm, score 1.0.

- [ ] **Step 1: Write the failing tests**

```python
def test_batch_scores_equal_predictor_agent(tmp_path) -> None:    # 300 sampled windows, gbdt and cnn artifacts
    for w, s in zip(agent_windows, stored_scores):
        (sig,) = PredictorAgent("model", k=15, h=15, artifact_path=p, expected_model_hash=hh).on_event(w)
        assert sig.score == s                                       # bit-identical

def test_random_scores_equal_predictor_agent() -> None:
def test_constant_cell_every_tradable_window_eligible() -> None:
def test_window_with_invalid_label_still_eligible() -> None:      # no entry bar within 2 min: still a Signal
def test_main_and_c_a_share_one_artifact() -> None:
def test_only_oos_days_scored_by_their_fold() -> None:
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_scores.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 6 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): grid scores bit-identical to PredictorAgent`.

---

### Task 15: The engine-mirror simulator (§8, §9, §10.2)

**Files:**
- Create: `research/intraday/research/r16_sim.py`
- Test: `tests/test_intraday_model_r16_sim.py`

**Interfaces:**
- Consumes: Task 9 frames, Task 10 `StaticCostInputs` / `refit_book_at` / `half_tick_floor_bps`, `research.penny.etl.quotes.leg_cost_bps` (scalar via `exec.costs.leg_cost_bps_scalar`), `risk.brackets.true_range`, `make_bracket`, `risk.sizing.size_order`, `floor_shares`, `exec.sizing.fill_cap`, `fresh_quote`. Importing the agents' pure helpers keeps the formulas literally shared.
- Produces:
  - `@dataclass(frozen=True) class FillRec`: `ts_ns: int, symbol: str, side: str, qty: int, price: float, cost_bps: float, cap: str, cost_source: str, kind: Literal["entry", "stop", "target", "time", "eod"], gap_through: bool`.
  - `@dataclass(frozen=True) class DayResult`: `fills: tuple[FillRec, ...]`, `pnl_net: float`, `pnl_gross: float`, `end_equity: float`, `skips: dict[str, int]`, `flags: dict[str, bool]` (`gap_through_stop`, `partial_fill`, `eod_forced`).
  - `simulate_day(d: date, bars: pd.DataFrame, quotes: pd.DataFrame, signals: pd.DataFrame, cost_inputs: StaticCostInputs, rcfg: RiskConfig, xcfg: ExecConfig, cost_multiplier: float = 1.0) -> DayResult`.
  - `simulate_cell(cell: Cell, days: Sequence[date], ...) -> CellResult(daily: pd.DataFrame[date, pnl_net, pnl_gross, equity_start, equity_end, n_trades], trades: pd.DataFrame)`. Days run in order and `E_d = E_{d−1} + pnl_net`, starting at 25,000.
- **Engine semantics the simulator mirrors** (the sources are the Phase-2 Interfaces blocks). Each rule has a named test.
  - R1. Order at minute T: tick, then the bars released at T in sorted-symbol order (each followed by that symbol's quote stamped T), then the fills caused by those bars, then the Signals at T in sorted-symbol order. (`test_slot_freed_by_exit_fill_reused_same_minute`)
  - R2. Risk skip order: killed/loss_limit → halted → `ts >= 15:30` → `score < threshold` → busy / entries ≥ 1 → occupied ≥ 5 → no bar → qty < 1 → bracket str. Only an emitted intent counts toward entries. (`test_zero_qty_skip_does_not_count_as_entry`, `test_expired_entry_blocks_reentry`)
  - R3. Qty = `size_order(ref = decision bar close, available = cash − reserved, bar_volume = decision bar volume, fresh quote age 0..5 min at the signal ts, quote_size_frac 0.5)`. (`test_quote_cap_at_entry`)
  - R4. Entry fill on the first bar with `T <= bar_start < T + 2 min` at its open, `q = min(qty, fill_cap(buy, bar_start, bar, fresh_quote))`. A remainder is cancelled. No such bar: expiry at the first event ≥ T + 3 min, and the slot frees then. (`test_partial_entry_remainder_cancelled`, `test_entry_expires_after_two_minutes`)
  - R5. Bracket from the decision-bar close: fixed `ref·(1±pct)`; ATR `ref ± m·max(mean TR of the last 5 session bars, 0.01)` (UP-1). Triggers run from the fill bar on, stop first, fill at the open of the next bar with `bar_start >= trigger bar ts`. (`test_gap_through_stop_fills_below_stop`, `test_both_hit_is_stop`)
  - R6. Time stop at `min(T + h, 15:55)`: at the first tick ≥ it, a pending exit if none exists. Fill at the first bar with `bar_start >= tick`. (`test_time_stop_clipped_1555`)
  - R7. Exit fills are capped per bar `fill_cap(sell, …)`; a remainder retries next bar. EOD at the 16:01 tick: the remaining qty at the last close, uncapped, with the quote fresh at 16:01 if one exists. (`test_eod_fill_at_last_close_uncapped`)
  - R8. Leg cost = `leg_cost_bps_scalar(qty·price, side_notional, max(hs, half_tick_floor), σ, qty / adv_shares, 1.0, 1.0)` × `cost_multiplier`. The hs source is a fresh quote, else `refit_book_at` (UP-2). Cash: buy `−p·q·(1+c/1e4)`, sell `+p·q·(1−c/1e4)`. (`test_cost_matches_exec_formula`)
  - R9. Loss limit: after every bar (progressively, in R1 order) and every fill, `cash + Σ qty × last close ≤ 0.98 × E_{d−1}` ⇒ no entry for the rest of the day, same-minute signals included. (`test_loss_limit_trip_blocks_same_minute_signal`)
  - R10. Slots count distinct symbols with a position or a working intent. (`test_partial_position_counts_one_slot`)
- Performance: per day, process only the eligible-signal timestamps and the bar indices of open trades. Per-trade exit paths are found with numpy on that symbol's arrays. The target is ≤ 1 ms per cell-day on the median day (a `pytest-benchmark`-free test asserts that the 60-day smoke grid of one cell runs in < 5 s of CPU, measured by the test's own `time.process_time` — tests may use the clock, `research/` may not).

- [ ] **Step 1: Write the failing tests** (the named tests above, on hand-built bar frames for `D = 2024-03-04`, each asserting the exact `FillRec` tuple and `pnl_net` to 1e-9) plus:

```python
def test_constant_scorer_enters_sorted_symbols_as_slots_free() -> None:   # §7 N4: 7 symbols, 5 slots
def test_flat_day_pnl_zero_and_equity_carries() -> None:
def test_gross_equals_net_with_zero_costs() -> None:
def test_cost_multiplier_scales_only_costs() -> None:
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_sim.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Until UP-1/UP-2 merge, the ATR floor and the refit book are implemented locally behind the same formula. Task 16's cross-check is what proves equality.
- [ ] **Step 4: Run.** Expected: all pass. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): engine-mirror day simulator with compounded account`.

---

### Task 16: Engine session harness

**Files:**
- Create: `research/intraday/research/r16_engine.py`
- Test: `tests/test_intraday_model_r16_engine.py`

**Interfaces:**
- Consumes: `engine.replay.run_replay`, `clock_ticks`; `data.feed.FeedAgent`, `data.replay_feed.ReplayFeed`, `data.clock.et_to_utc`; `state.StateAgent`; `model.PredictorAgent`; `risk.RiskAgent`; `exec.ExecutionAgent`, `exec.SimBroker`; `ops.monitor.MonitorAgent`, `ops.halts.RecordedHaltLog`; `ops.recorder.read_log`; `ops.manifest.write_manifest`.
- Produces:
  - `upstream_ready() -> dict[str, bool]` (`"UP-1": "atr_floor" in RiskConfig.__dataclass_fields__`, `"UP-2"`: the six refit fields in `CostInputs`).
  - `run_engine_session(d: date, cell: Cell, candidates: tuple[str, ...], artifact: tuple[Path, str] | None, threshold: float, start_equity: float, cost_inputs: StaticCostInputs, log_path: Path, infer_halts: bool = False) -> EngineDayResult(log_hash, fills: tuple[FillRec, ...], pnl_net: float, skips: dict, kills: int, halts: int)`. Agents in engine.md order without the screener: `[FeedAgent(ReplayFeed(unseal=False)), StateAgent(k), PredictorAgent(...), RiskAgent(d, risk_config(...)), ExecutionAgent(SimBroker(exec_config(), cost_inputs)), MonitorAgent(RecordedHaltLog(()), infer_halts_from_gaps=infer_halts, gap_halt_min=5)]`. Sources: `[clock_ticks(et_to_utc(d, "09:25"), et_to_utc(d, "16:02")), (Candidates(ts=et_to_utc(d, "09:30"), screen_id=f"{cell.universe_arm}:{config_hash(cell_params)[:12]}:{d:%Y-%m-%d}", symbols=candidates),)]`, with `ops.manifest.config_hash` over the cell's §5 parameters. `candidates` = the arm's universe **plus that day's feed-rejected symbols**. The feed turns those into `Gap`s only, so the result must not change, and the parity stratum "un-adjustment reject" actually exercises the reject path. `pnl_net = risk.books.cash − start_equity`. It asserts no open position remains. Fills are rebuilt from the log's `Fill` envelopes (`kind` from the cause: Bar ⇒ by the exit log; tick ⇒ `eod`).
  - `check_dev_date(d)` runs first. `kills > 0` or `halts > 0` with `infer_halts=False` is recorded and makes parity fail (the simulator models neither).

- [ ] **Step 1: Check upstream.** Run `"$PY" -c "from research.intraday.research.r16_engine import upstream_ready; print(upstream_ready())"`. Expected: both `True`. If either is `False`, stop and report **BLOCKED (UP-n)** to the controller.
- [ ] **Step 2: Write the failing tests** (a synthetic cache under `tmp_path` via `data.synthetic`, 3 symbols, one day):

```python
def test_engine_and_simulator_agree_on_synthetic_day(tmp_path) -> None:   # gbdt tiny artifact, all 3 exits
    for exit_ in EXITS:
        e = run_engine_session(D, _cell(exit_), CANDS, ART, THR, 25_000.0, INPUTS, tmp_path / f"{exit_}.jsonl")
        s = simulate_day(D, BARS, QUOTES, SIGNALS, INPUTS, risk_config(_cell(exit_), THR, 25_000.0), exec_config())
        assert e.fills == s.fills and abs(e.pnl_net - s.pnl_net) < 0.005

def test_exec_refit_book_matches_reference() -> None:     # UP-2 vs refit_book_at on a 0..391 minute grid
def test_engine_session_twice_identical_hash(tmp_path) -> None:
def test_engine_refuses_holdout_date(tmp_path) -> None:
def test_no_kill_no_halt_on_clean_day(tmp_path) -> None:
```
- [ ] **Step 3: Run** `"$PY" -m pytest tests/test_intraday_model_r16_engine.py -q`. Expected: FAIL.
- [ ] **Step 4: Implement.** A disagreement in test 1 is a simulator bug unless an engine rule contradicts §8. In that case stop and report the engine defect to the controller with the event log. Never adapt the simulator to a prereg violation.
- [ ] **Step 5: Run.** Expected: 5 passed. Full suite; ownership.
- [ ] **Step 6: Commit** `feat(r16): engine session harness composing the merged agents`.

---

### Task 17: The parity test (§6.4, §11.1.6, §10.7 determinism)

**Files:**
- Create: `research/intraday/research/r16_parity.py`
- Test: `tests/test_intraday_model_r16_parity.py`

**Interfaces:**
- Consumes: Tasks 14–16.
- Produces:
  - `parity_configurations() -> tuple[ParityConfig, ...]`, where `ParityConfig(k, h, scorer: Literal["gbdt", "cnn", "random0", "constant"], exit, universe_arm)`. It holds every (exit × h × scorer × arm) at k = 15 (72) plus (k ∈ {5, 30}, h = 15, time_stop, gbdt, screen), 74 in total.
  - `day_flags(days, configs, sim_fn, panel) -> pd.DataFrame` (`date, gap_through_stop, partial_fill, eod_forced, unadjust_reject`, booleans only; the function returns no P&L field and the test asserts its columns).
  - `stratified_parity_days(flags: pd.DataFrame, n: int = 50, per_stratum: int = 10, seed: int = 16) -> tuple[date, ...]`. Strata in the order gap_through_stop, partial_fill, eod_forced, unadjust_reject: from days carrying the flag and not yet chosen, sample `min(10, available)` with `np.random.default_rng(16)`, then fill uniformly to 50, sorted. A shortfall is recorded in the report, never padded.
  - `compare_day(sim: DayResult, eng: EngineDayResult) -> list[str]`: (ts, symbol, side, qty, price) equal exactly, `|Δcost_bps| <= 1e-9`, `|Δpnl| < 0.005`.
  - `run_parity(...) -> ParityReport(passed: bool, n_config_days: int, mismatches: list[dict], strata_counts: dict, determinism_hashes: dict)`. Determinism: the first 5 parity days, main gbdt config, replayed twice with identical log hashes.
- Ordering guard: `run_parity` is called before any grid result is computed, and `run_r16` refuses the grid stage unless `parity/DONE.json` has `passed: true` (Task 22).

- [ ] **Step 1: Write the failing tests**: `test_74_configurations`, `test_strata_sampling_deterministic_and_min_10`, `test_strata_shortfall_recorded`, `test_compare_day_detects_one_cent_and_qty_diffs`, `test_day_flags_has_no_pnl_columns`.
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_parity.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Engine sessions run in a process pool, one session per task, and are reassembled in sorted `(config, date)` order. Parity logs are kept under `OUT_ROOT/logs/` (session_log_path).
- [ ] **Step 4: Run.** Expected: 5 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): pre-registered engine parity test with stratified days`.

---

### Task 18: Nulls and guards (§10.7, §7 N5)

**Files:**
- Create: `research/intraday/research/r16_nulls.py`
- Test: `tests/test_intraday_model_r16_nulls.py`

**Interfaces:**
- Consumes: Tasks 12–15; `sklearn.metrics.roc_auc_score`; `scipy.stats.spearmanr`.
- Produces:
  - `permuted_within_day(y, dates, seed=16) -> np.ndarray`; `shifted_labels(labels, dev_sessions, lag=20) -> pd.Series` (same symbol and same ET minute 20 sessions later; NaN if absent or the target session is ≥ `HOLDOUT_START`).
  - `null_a(k, h, family, fold) -> NullResult` and `null_b(...) -> NullResult` (retrain on null labels, OOF AUC, z vs 0.5 with the clustered SE, `passed`). Null (a) also returns the paired account test against C_b on that fold (ambiguity 7).
  - `null_c_tripwire(day_windows) -> None` (every window's max bar ts ≤ its ts; no score uses a later bar). `null_d_constant(...) -> bool`: constant-price panel and zero costs give exactly zero daily returns in every cell.
  - `martingale_paths(bars: pd.DataFrame, seed: int) -> pd.DataFrame`: per symbol-day, log-close random walk with the day's realised minute σ and zero drift. Volumes and timestamps are kept; `open = prev close`, `high/low = max/min(open, close)`, `vwap = (open + close)/2`. `martingale_auc(...)` = the fold model's AUC on synthetic windows and synthetic labels, reported with `AUC_real − AUC_synth` and its clustered CI.
  - `cb_band(seed_scores, y, dates) -> dict[int, tuple[float, float, bool]]` (AUC z, IC z, inside) at |z| ≤ 3.66 (ambiguity 8). Any `False` voids the run.
  - `NullResult(name, stat, z, passed, detail)`; `run_nulls(...) -> list[NullResult]` on one fold (the last) per (k, h, model).

- [ ] **Step 1: Write the failing tests**

```python
def test_permutation_keeps_daily_base_rate() -> None:
def test_shifted_label_never_reads_holdout() -> None:
    s = shifted_labels(LABELS, SESSIONS_TO_2025_08_06, lag=20)
    assert s[LABELS.date >= SESSIONS_TO_2025_08_06[-20]].isna().all()
def test_martingale_paths_zero_drift_same_sigma() -> None:      # 2,000 paths: mean log-ret ~0, sigma within 5%
def test_null_d_constant_prices_zero_dispersion() -> None:
def test_cb_band_flags_a_leaky_seed() -> None:                    # score = y + noise -> outside band
def test_tripwire_fires_on_future_bar() -> None:
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_nulls.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** A null "failing by improving" (AUC significantly < 0.5) is reported as negative skill, not leakage (LOG §35): `NullResult.detail = "negative skill"` with `passed=True`.
- [ ] **Step 4: Run.** Expected: 6 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): R16 null battery, martingale AUC null and C_b band`.

---

### Task 19: Evaluation (§10.2–10.5, §11.1, §7 SPY)

**Files:**
- Create: `research/intraday/research/r16_eval.py`
- Test: `tests/test_intraday_model_r16_eval.py`

**Interfaces:**
- Consumes: `research.penny.eval.stats.{sharpe_ratio, probabilistic_sharpe_ratio, deflated_sharpe_ratio}`, `research.penny.eval.battery.{sharpe_se, min_track_record_length, max_drawdown, expected_max_drawdown}`; Task 15 `CellResult`.
- Produces:
  - `iid_day_indices(n: int, B: int = 10_000, seed: int = 16) -> np.ndarray[int32]` (B, n) and `stationary_indices(n, B, mean_block=5.0, seed=16)`, vectorised (geometric run starts via `rng.random(...) < 1/mean_block`, then `np.maximum.accumulate` on start positions). Both are built once per run and shared by every series, so all comparisons are paired.
  - `geo_mean(r) -> float` = `exp(mean(log1p r)) − 1`; `boot_geo_ci(r, idx) -> tuple[float, float]` (2.5/97.5 percentiles of the resampled geometric means); `boot_mean_ci(d, idx)`; `one_sided_p(r, idx) -> float` (ambiguity 8).
  - `holm(p: Mapping[str, float], alpha=0.05) -> dict[str, tuple[float, bool]]`.
  - `dsr(r, n_trials: int, sr_var_annual: float) -> float` = `deflated_sharpe_ratio(r, n_trials, sr_var_annual / 252)`.
  - `cell_metrics(result: CellResult, idx_iid, idx_stat) -> dict`: every §10.3 item (trade stats, top-20 trades, top-1%/top-10 removed, activity, Sharpe/SE/PSR/MinTRL, MDD vs expectation, cap counts, cost decomposition, quote-source usage, rejects, `top_n` binding share, entry-eligible rate, EOD-forced counts and ages, invalid-bracket skips per exit, per year and per pool, gross alongside net).
  - `dev_conditions(main: dict, controls: dict, holm_ok: bool, dsr_val: float) -> dict[str, bool]` for §11.1.1–5: (1) geometric CI lower > 0; (2) paired-difference CI lower > 0 vs C_a, the C_b seed mean and C_c, **and** compounded return > `np.percentile(C_b compounded, 95)`; (3) Holm p < 0.05 and DSR ≥ 0.95 at N = 4,248; (4) compounded return after removing the top 1% of trades > 0 (median trade reported, negative-median flag); (5) ≥ 300 trades on ≥ 150 days.
  - `select_cell(table: pd.DataFrame) -> str | None` (§10.5 plus ambiguity 10).
  - `contrasts(daily: Mapping[str, pd.DataFrame], idx_iid) -> pd.DataFrame` (`family, contrast, mean_diff, ci_lo, ci_hi, p, q_bh, rejects`). All are paired daily differences with the shared indices, one-sided p, and BH-FDR (q = 0.05, `research.penny.eval.stats.benjamini_hochberg`) within each family (§10.4). Families: **H2** = {C_c − C_0 per (k, h, exit)} ∪ {Main − C_a per main cell}, net. **H3** = {Main(gross) − C_b seed-mean(gross) per main cell}, plus the per-(k, h, model) OOF AUC vs the martingale AUC from Task 18. **H4** = {Main(fixed_pct) − Main(time_stop), Main(atr) − Main(time_stop) at the same (k, h, model)}, net. H2–H4 never enter the gate.
  - `spy_open_close(dates: Sequence[date]) -> pd.Series`: 2016–2024 from `SPY_CACHE`, 2025 from `SPY_2025` rows `symbol == "SPY"` read with `filters=[("symbol", "==", "SPY"), ("date", "<", HOLDOUT_START)]` (R8), NaN where missing. It calls `check_dev_date` on every date. `spy_stats(r, spy) -> dict` (correlation, OLS alpha and beta on overlapping days, missing-day list).

- [ ] **Step 1: Write the failing tests**

```python
def test_geo_ci_on_constant_series_is_point() -> None:
def test_paired_indices_shared() -> None:
def test_holm_known_example() -> None:            # p = [0.01, 0.04, 0.03] -> adjusted [0.03, 0.06, 0.06]
def test_dsr_converts_annual_variance() -> None:
    r = _series(seed=1); v = 0.25
    assert dsr(r, 4248, v) == deflated_sharpe_ratio(r, 4248, v / 252)
def test_conditions_require_all_controls() -> None:
def test_selection_tie_rules() -> None:
def test_spy_never_reads_holdout() -> None:
    with pytest.raises(SealedHoldoutError):
        spy_open_close([date(2025, 8, 6), date(2025, 8, 7)])
def test_power_floor() -> None:
def test_contrast_families_and_bh() -> None:      # H2 has 27 + 54 rows, H3 54, H4 36; q from benjamini_hochberg
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_eval.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 9 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): account evaluation, paired day-clustered CIs, Holm, DSR, selection`.

---

### Task 20: MLflow tracking and the trial ledger (§12)

**Files:**
- Create: `research/intraday/research/r16_tracking.py`
- Test: `tests/test_intraday_model_r16_tracking.py`

**Interfaces:**
- Produces:
  - `class R16Tracker`: `__init__(self, store_dir: Path = MLRUNS_DIR, experiment: str = "intraday")` sets `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` **inside `__init__`**, then builds `mlflow.MlflowClient(tracking_uri=store_dir.as_uri())`. Nothing happens at import. The experiment is created lazily on the first `log_run`.
  - `log_run(self, name: str, params: Mapping[str, object], metrics: Mapping[str, float], tags: Mapping[str, str], artifacts: Sequence[Path] = ()) -> str` (non-finite metrics are dropped and listed in the tag `dropped_metrics`; params are stringified; batches go through `log_batch`).
  - `base_tags(git_sha: str, clean: bool, provenance_sha: str, event_log_hash: str, prereg_commit: str) -> dict[str, str]`, which always includes `conditional_on="U1"`, `universe="G(D)-gated"`, `period="dev"`, `git_clean`.
  - `append_trial(path: Path, cell_id: str, run_id: str, metrics: Mapping[str, float]) -> None` (canonical JSON line). `trial_lines(path) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
def test_env_set_in_constructor_not_import(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    import research.intraday.research.r16_tracking as t; importlib.reload(t)
    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ
    t.R16Tracker(tmp_path / "mlruns")
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"

def test_run_carries_u1_and_universe_tags(tmp_path) -> None:
def test_experiment_created_on_first_run_only(tmp_path) -> None:
def test_non_finite_metric_dropped_and_listed(tmp_path) -> None:
def test_trial_ledger_lines(tmp_path) -> None:
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_tracking.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 5 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): MLflow file-store tracker with U1 tags and trial ledger`.

---

### Task 21: Results writer (JSON + PNG/HTML)

**Files:**
- Create: `research/intraday/research/r16_report.py`
- Test: `tests/test_intraday_model_r16_report.py`

Before writing chart code, load the `dataviz` and `plotly` skills (agent definition, Plots row).

**Interfaces:**
- Consumes: Task 19 outputs.
- Produces:
  - `RESULTS_SCHEMA_VERSION = 1`; `build_results(...) -> dict` with top-level keys `schema_version, prereg_commit, git_sha, conditional_on ("U1"), universe, period ("dev"), amendment0_sha, provenance_sha, n_trials {r16: 54, binding: 4248}, folds, parity, nulls, cells {cell_id: metrics}, controls {cell_id: {C_a, C_b_seed_mean, C_b_seeds, C_c, C_0}}, contrasts {H2, H3, H4 with BH-FDR q}, holm, selection {cell_id | null, reason}, gate_I_development {passed, failing_conditions}, sensitivities {S1, S2, S3, S4, cost_x1_5, halts}, spy, engine_replay {ran, identical, log_hashes_sha}`.
  - `write_results(out_dir: Path, results: dict, series: Mapping[str, pd.Series]) -> dict[str, Path]`. It writes `results.json` (canonical JSON, NaN → null) and, each as `.png` (matplotlib, Agg, `dpi=150`, `metadata={"Software": None}`) and `.html` (plotly, `include_plotlyjs=True`):
    - `equity_by_arm`: the selected cell (or the highest-DSR main cell if none is selected, labelled so) against C_a, C_c, C_0, with the C_b seeds as a 5–95% band, log-scale equity, and the title stating "conditional on U1".
    - `equity_all_main`: all 54 main cells, faint lines.
    - `heatmap_<metric>` for `geo_mean_net`, `dsr`, `ci_lower`: facets model × exit, axes k × h.
    - `control_contrasts`: a forest plot of the paired-difference CIs vs C_a, C_b and C_c per main cell.
    - `nulls`: AUC vs martingale AUC, and C_b seed AUCs inside the band.
  - The writer computes nothing statistical. It only lays out `results` and `series`.

- [ ] **Step 1: Write the failing tests**: `test_results_json_schema_keys`, `test_results_json_deterministic` (written twice, same bytes), `test_all_charts_written_png_and_html`, `test_titles_state_u1`.
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_report.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 4 passed. Full suite; ownership.
- [ ] **Step 5: Commit** `feat(r16): machine-readable results and report charts`.

---

## Part C — the run script

### Task 22: `run_r16` — audit | smoke | full, with refusals

**Files:**
- Create: `research/intraday/research/run_r16.py` (`"$PY" -m research.intraday.research.run_r16 <mode> [--workers 8]`)
- Test: `tests/test_intraday_model_r16_run.py`

**Interfaces:**
- Consumes: every Part-B module.
- Produces:
  - `main(argv: list[str] | None = None) -> int`; `Mode = Literal["audit", "smoke", "full"]`; `STAGES = ("universe_pre", "panel", "universe", "costs", "windows", "train", "scores", "nulls", "parity", "grid", "sensitivities", "evaluate", "engine_replay", "report")`. `universe_pre` = `build_universe_day` with the audit's `audit_feed_rejects`, which gives the symbol set for `ReplayFeed`. `panel` builds the bars and asserts its rejects equal the audit's. `universe` is the final per-day `UniverseDay` (primary, S1, S2). `costs` builds the cost-input table. `c_label` and the refit model are read from the Amendment-0 inputs, never refit.
  - MLflow params per run (§12): every §5 key from `r16_config` (flattened), `cell_id`, `arm`, `scorer`, `seed`, `fold`, `config_hash`, `screen_id` format, `model_hash`, `c_label`, the refit digest and the prereg commit. Each grid run writes its trade file `trades/<cell_id>.parquet` and tags `event_log_hash` = sha256 over (the parity log-hash list ‖ that trade file's sha256). Engine-replayed cells get the sha256 over their ordered per-session log hashes.
  - `preflight(mode: Mode, repo: Path) -> Preflight(git_sha, prereg_commit, audit_sha | None)`. It refuses (raises `RefusedError`, exit code 3, message naming the rule) when any of these holds:
    1. the tree under `research/` is dirty (`git_state`);
    2. mode ≠ audit and `amendment0_recorded(prereg text at HEAD, audit_sha)` is false;
    3. mode = full and `OUT_ROOT/r16/<sha12>/smoke/PASSED.json` is missing;
    4. mode = full and `TRIAL_LEDGER` already has lines (a rerun needs a recorded amendment and the controller moving the file);
    5. `infeasible(amendment0)` is true (§11.4: exit 4, "infeasible on cached data", no return read).
  - There is no `--unseal` and no `--holdout` flag. The dev-session list is `sessions(DATA_ROOT) ∩ [DEV_FIRST, HOLDOUT_START)` minus `half_days(...)` (Task 11; the list is recorded in the Amendment-0 inputs). `check_dev_date` runs on every session that any stage touches. At exit, `assert max(touched) < HOLDOUT_START`.
  - Stage checkpoints: `OUT_ROOT/r16/<sha12>/<mode>/<stage>/DONE.json` holds the input hashes. Resume skips a DONE stage only under the same SHA and inputs.
  - `audit` runs Tasks 7, 10, 11 (counts only) and prints the audit sha256 and the Amendment-0 inputs path. Then it **stops**. Nothing else can run until the prereg records Amendment 0.
  - `smoke`: 60 dev sessions (every 30th session from 2017-01-03), folds `walk_forward_folds(n_splits=2, embargo_days=5, min_train_days=30)`, cell `M-k15-h15-gbdt-time_stop` and its controls (C_b seeds 0–1), parity on 3 stratified days of that one config, engine replay of the cell on its OOS days, evaluate, report. MLflow experiment `intraday-smoke`. No trial-ledger line. It writes `smoke/PASSED.json` only if every stage finishes, parity passes, determinism passes and the nulls (c)/(d) pass. It prints the projected full-run time from the MLflow stage durations.
  - `full`: every stage in order. The grid stage refuses unless `parity/DONE.json.passed`. The nulls stage fails ⇒ the run is void (exit 5) with the report written. `engine_replay` runs only if `select_cell` returns a cell (ambiguity 13): 24 configs over all OOS days in parallel, with start equity from the simulator's path; any day differing from the simulator ⇒ `grid_void=True` and exit 6 (§6.4 void-and-rerun is an amendment, not automatic). MLflow: one run per (unique cell-run, fold) plus one aggregate run per cell (54 main aggregate runs carry DSR and the conditions). 54 trial-ledger lines are appended, and `trial_lines == 54` is asserted.

- [ ] **Step 1: Write the failing tests** (preflight with a temp git repo; stages monkeypatched):

```python
def test_refuses_dirty_tree(tmp_repo) -> None:
def test_refuses_without_amendment0(tmp_repo) -> None:
def test_full_refuses_without_smoke_pass(tmp_repo) -> None:
def test_full_refuses_nonempty_trial_ledger(tmp_repo) -> None:
def test_no_unseal_or_holdout_flag() -> None:
    with pytest.raises(SystemExit):
        main(["full", "--unseal"])
    src = inspect.getsource(run_r16)
    assert "unseal=True" not in src and "UNSEAL = True" not in src
def test_session_list_stops_before_holdout() -> None:
def test_grid_refuses_before_parity_pass(tmp_path) -> None:
def test_infeasible_exits_4_without_reading_returns(tmp_repo) -> None:
def test_resume_skips_done_stage_same_sha_only(tmp_path) -> None:
```
- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_model_r16_run.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 9 passed. Full suite; guards; ownership.
- [ ] **Step 5: Commit** `feat(r16): run script with audit/smoke/full modes and hard refusals`.

---

## Part D — operating the round (no code; the stop points are binding)

### Task 23: Audit → Amendment 0 → smoke → overnight full run → G2 handoff

**Files:** none in the repo. Outputs go under `OUT_ROOT`.

- [ ] **Step 1: Clean tree, full suite.** `git status --porcelain -- research/` is empty. `"$PY" -m pytest tests/ -q` is all green.
- [ ] **Step 2: Audit.** Run `"$PY" -m research.intraday.research.run_r16 audit --workers 8`. Expected: exit 0, and it prints the audit sha256 and `…/amendment0_inputs.json`. **STOP.** Report to the controller: the two paths, the sha, the `infeasible` flag and the ambiguities list of this plan. The hypothesis designer writes Amendment 0 into the prereg (counts, `c_label`, refit coefficients, SPY span, breakout ledger lines, half-day list, ambiguities 1–15) and commits it. No minute return has been read up to here.
- [ ] **Step 3: Smoke.** After Amendment 0 is committed (and the tree is clean), run `"$PY" -m research.intraday.research.run_r16 smoke --workers 8`. Expected: exit 0, `smoke/PASSED.json` written, and a projected full-run time ≤ 10 h printed. If it fails, debug with `superpowers:systematic-debugging` / `phd-skills:debug`, fix test-first in the owning task, commit, and rerun the smoke on the new SHA.
- [ ] **Step 4: Full run (overnight).** Run `"$PY" -m research.intraday.research.run_r16 full --workers 8` with `run_in_background`. Expected: exit 0 (or 5/6 with the report written). Check `mlflow` through `mlflow-mcp` (`search_experiments` "intraday", `list_runs`), with 54 aggregate main runs and `r16_trials.jsonl` at 54 lines.
- [ ] **Step 5: Hand to G2.** Report to the controller: `results.json` path and sha256, the chart paths, parity/nulls/C_b-band verdicts, `gate_I_development`, the selected cell (or none), the engine replay outcome, the actual stage durations, and every result labelled "conditional on U1". Do not interpret beyond the prereg's rules. The holdout stays sealed: this plan contains no path that reads it.
