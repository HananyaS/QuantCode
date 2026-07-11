# QuantCode Roadmap

**Goal:** turn the existing cross-sectional backtesting pipeline into a fully
automated live trading system, trading US equities, that produces
meaningful supplemental income.

**Current state (as of this roadmap):** a solid, tested *research* pipeline
exists — `UniverseAgent → CrossSectionalFeatureAgent → CrossSectionalLabelingAgent
→ RankingModelAgent → PortfolioAgent → MultiAssetBacktestAgent →
MultiAssetEvaluationAgent` (see `agents/multi_orchestrator.py`) — but nothing
in the repo can place a trade. This roadmap sequences the work needed to
close that gap safely, without skipping the validation steps that separate a
strategy that *looks* good in one backtest from one that actually holds up
with real money on the line.

## Implementation status

Phases 1-3 and the Phase 4 gate *checker* are implemented, all behind opt-in
config flags (defaults preserve prior behavior exactly) and covered by
tests (`python -m pytest tests/ -q`). **Everything that touches a broker is
paper-trading only** — `ExecutionAgent` hard-asserts `paper=True` — no code
in this repo can place a real-money order. Actually flipping to real capital
(the second half of Phase 4) remains a deliberate, manual, un-implemented
step gated by `utils/go_live_gate.py`, not something to automate.

| Phase | Status |
|---|---|
| 0 — Reality check | n/a (documentation only) |
| 1 — Prove the edge is real | ✅ implemented |
| 2 — Harden risk management | ✅ implemented |
| 3 — Paper-trading execution infra | ✅ implemented |
| 4 — Go-live gate | ✅ gate *checker* implemented; real-capital go-live itself intentionally not automated |
| 5 — Scale | manual/business process, not code |

---

## 0. Reality check / operating assumptions

**Capital math.** On a sub-$25k account, a target of $1-3k/month implies
48-144% annualized returns. That is not a realistic target for a systematic
strategy from day one — pursuing it directly all but guarantees blowing up
the account through excess leverage or risk-taking. This roadmap instead
targets a **validated, positive-Sharpe, automated system first**; income
grows with assets under management (reinvested profits, and possibly added
contributions) over a realistic 12-24 month horizon, not immediately.

**PDT rule.** Accounts under $25k are limited to 3 day-trades per rolling
5 business days (a "day-trade" = opening and closing the same position on
the same day). The current strategy is naturally swing-style — labels use a
5-day forward return (`configs/universe.yaml: labeling.forward_period: 5`)
and `PortfolioAgent` holds positions across multiple days via rank-based
entry/exit. But nothing today explicitly *prevents* a same-day round trip
(e.g. a trailing-stop exit right after a same-day entry). The execution
layer built in Phase 3 must guard against this explicitly.

**Go/no-go philosophy.** No phase advances to more real capital without an
explicit, numeric gate. Each phase below ends with an exit criterion —
treat it as a hard gate, not a suggestion.

---

## Phase 1 — Prove the edge is real ✅

*No live infra yet. Everything in this phase is backtest-only.*

`agents/ranking_model_agent.py` still does one purged temporal train/test
split per run for the *production* pipeline (unchanged, still the fastest
path for a normal run) — but validation and tuning no longer rely on that
single split:

- ✅ **Walk-forward validation** — `utils/walk_forward.py`. Multiple
  expanding-window folds across time, reporting per-fold train/test IC and
  the `ic_gap` (train_ic − test_ic) as a direct overfitting signal, instead
  of one point estimate. Tests: `tests/test_walk_forward.py`.
- ✅ **Optuna HPO** — `utils/hpo.py`, wired against the search space in
  `configs/hpo_params.yaml` (previously an unread stub). The objective is
  mean walk-forward test IC penalized by the mean IC gap, so tuning can't
  just fit to one lucky period. Purge days always track the trial's sampled
  `forward_period`. Tests: `tests/test_hpo.py`.
- ✅ **Experiment tracking** — `utils/experiment_tracker.py`. Appends one
  JSON line per run (config + metrics) to a log file and reads it back as a
  flat, comparable DataFrame. Tests: `tests/test_experiment_tracker.py`.
- ✅ Train-vs-test IC gap is logged per fold directly by `walk_forward_validate`
  (the `ic_gap` column), on top of the pre-existing heuristic thresholds in
  `RankingModelAgent` (`|IC| > 0.15`) and `MultiAssetEvaluationAgent`
  (`Sharpe > 2.0`), which only catch obviously-too-good-to-be-true results.

**Exit criterion:** run `utils.hpo.run_hpo` against real universe data and
confirm the best trial's `mean_test_ic` (Optuna trial `user_attrs`) is
consistently positive across folds — not just one split — before trusting a
config enough to feed it into Phase 3.

---

## Phase 2 — Harden risk management ✅

*Still backtest-only.*

All additions are opt-in constructor/config params — default behavior is
byte-for-byte unchanged from before this phase.

- ✅ **Portfolio-level max-drawdown circuit breaker** —
  `PortfolioAgent(max_drawdown_limit=..., de_risk_on_breach=...)`. Tracks
  its own realised equity from the weight path each day; once drawdown from
  the running peak breaches the limit, new entries halt for the day, and
  `de_risk_on_breach=True` additionally force-exits all open positions
  (logged as a `circuit_breaker_exit` trade).
- ✅ **Volatility-targeted sizing** — `PortfolioAgent(vol_target_sizing=True,
  vol_window=...)`. Inverse-volatility position sizing as a third option
  alongside equal-weight / softmax score-weighting.
- ✅ **Correlation exposure caps** — `PortfolioAgent(max_correlation=...,
  corr_window=...)`. A candidate entry is skipped if its trailing return
  correlation with any held position exceeds the threshold.
- ✅ **Slippage / market-impact cost model** —
  `MultiAssetBacktestAgent(slippage_coef=..., dollar_volume_window=...)`.
  Adds a per-asset, per-day cost term scaled by `sqrt(participation rate)`
  (notional traded ÷ trailing avg dollar volume) on top of the flat bps
  `transaction_cost`; `slippage_coef=0` (default) reduces exactly to the old
  flat-cost formula.

All four are exposed as commented, off-by-default knobs in
`configs/universe.yaml`. Tests: `tests/test_portfolio_risk.py`,
`tests/test_slippage_cost.py`.

**Exit criterion:** before relying on a config in Phase 3, re-run Phase 1's
walk-forward validation with these risk params turned on and realistic
`slippage_coef` and confirm results still hold — this repo does not do that
comparison automatically, it's a manual step per config.

---

## Phase 3 — Build automated paper-trading execution infra ✅

- ✅ **Execution agent** — `agents/execution_agent.py`. Uses
  `alpaca.trading.TradingClient` to reconcile the broker account to today's
  target weights (last row of `portfolio_weights`). **Hard-asserts
  `paper=True`** in `__init__` — there is no code path in this repo that can
  submit a real-money order; that assertion is the actual enforcement
  mechanism, not just documentation.
- ✅ **Persistent state ledger** — `utils/state_store.py`. SQLite-backed
  (`data/live_state.db`): every submitted order and a daily account
  snapshot (equity/cash/positions), independent of `PortfolioAgent`'s
  in-memory backtest simulation, which still resets every run.
- ✅ **Live orchestrator** — `agents/live_orchestrator.py`. Shares the exact
  same research pipeline as `MultiAssetOrchestrator` (via the extracted
  `_build_research_pipeline()`) through `PortfolioAgent`, then feeds weights
  to `ExecutionAgent` instead of `MultiAssetBacktestAgent` — live and
  backtest decisions are made identically by construction.
- ✅ **Daily scheduler** — `utils/scheduler.py` (APScheduler
  `BlockingScheduler`, cron trigger) plus the CLI entry point `run_live.py`,
  runnable either once (`python run_live.py`, for an OS
  scheduler/Task-Scheduler/cron to invoke) or as a standing process
  (`python run_live.py --schedule --hour 16 --minute 30`).
- ✅ **Same-day round-trip (PDT) guard** — `ExecutionAgent` skips any sell
  for a ticker the ledger shows was already bought today, logged as
  `skipped_pdt_guard` (a working control, not a failure — see Phase 4's gate).
- ✅ **Monitoring/alerting** — `utils/logger.add_file_handler()` (rotating
  file handler on the root logger, wired in `run_live.py`) plus
  `utils/alerting.send_alert()` (Slack-compatible webhook via
  `ALERT_WEBHOOK_URL`), fired automatically on any `run_live.py` failure.

Tests: `tests/test_execution_agent.py`, `tests/test_state_store.py`,
`tests/test_live_orchestrator.py`, `tests/test_scheduler.py`,
`tests/test_monitoring.py`.

**Exit criterion:** the system runs unattended, fully automated, against the
Alpaca **paper** account for a meaningful live-forward window — target
3-6 months — with no critical failures. *(Not yet started — the code exists
and is tested, but no paper-trading track record has been accumulated yet.
`utils/go_live_gate.py`'s `min_days` default of 63 encodes this window.)*

---

## Phase 4 — Go-live gate & capital scaling

- ✅ **Gate checker implemented** — `utils/go_live_gate.py:evaluate_go_live()`.
  Evaluates a `StateStore`'s recorded paper-trading history against numeric
  thresholds (`min_days` ≥ 63 trading days, `min_sharpe`, `max_drawdown`,
  `max_problem_orders`) and returns `{passed, reasons, ...}`. **Decision
  support only** — it reads the ledger and returns a verdict; it does not
  place, authorize, or gate any trade itself, and nothing in the codebase
  calls it automatically before anything happens.
- ⛔ **Real-capital execution — deliberately not implemented.** Per current
  project scope, live execution is paper-trading only
  (`ExecutionAgent`'s `paper=True` assertion is the enforcement). Flipping
  to real capital means: consciously relaxing that assertion, swapping
  credentials to a funded Alpaca account, and only after
  `evaluate_go_live()` passes on an actual accumulated paper-trading
  history — a deliberate future decision, not a config flag to toggle today.
- Track live-vs-backtest performance decay continuously once live — this is
  one of the most common real-world failure modes for systematic
  strategies. Material decay should trigger a pause and re-diagnosis, not
  be pushed through. (Not yet built — there's no accumulated live history
  to compare against yet.)
- Out of scope for now, flagged for later: wash-sale rules and 1099
  tax tracking become relevant once real money is trading.

---

## Phase 5 — Scale

- Reinvest profits and/or add contributions as the live track record
  accrues; revisit the income target against realistic AUM growth rather
  than the original capital base.
- Additional asset classes (crypto, futures, options) are explicitly
  deferred — not part of this roadmap unless revisited later.

---

## Survivorship bias note

`UniverseAgent`'s `tickers: "sp500"` path uses *current* S&P 500
constituents (fetched from Wikipedia), not point-in-time historical
membership. This is a known, documented limitation (see the docstring in
`agents/universe_agent.py`) that inflates backtest performance somewhat.
It is not addressed in this roadmap's phases above but should be kept in
mind when interpreting Phase 1/2 backtest results — treat reported
Sharpe/IC as an upper bound, not a guarantee.
