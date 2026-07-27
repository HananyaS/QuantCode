# Research Log: Live-Pipeline Skeptical Review (pre-first-scheduled-run)

Deep review of the full live paper-trading flow — scheduled workflow →
`scripts/live_run.py` → data fetch → decision agents → `ExecutionAgent` →
state persistence — performed the evening before the first unattended
scheduled run. Four findings fixed (autoresearch-discipline, one atomic
commit each, full pytest guard on every keep); the rest documented here so
they're known limitations rather than surprises.

## Fixed

### 1. Live Kelly decision was one trading day stale (conceptual, critical)

`KellyPositionAgent`'s estimators follow the forecast-for-t convention —
the value at date t uses only data through t-1, pre-shifted at the source
(`utils/conditional_vol.py`, `utils/regime_classifier.py`). So the position
row at the last completed bar t was decided from data through **t-1**, and
`live_run.py` executed that row during session **t+1**: bar t's close never
influenced the trade. The validated backtest (`KellyBacktestAgent`, no
shift) holds the row-t position *during* t — live was systematically one
full trading day behind it.

Fix: `utils/live_decision.py::append_next_session_bar` extends each tracked
ticker's frame with an all-NaN row at the next NYSE session (Alpaca
calendar). The pre-shifted EWM estimators carry forward through the NaN
input, so `shift(1)` lands the through-latest-close estimate exactly on the
appended row; the (already-lagged) drawdown/watermark price lookups resolve
to the last real close. Regression test: a -25% final bar flattens the
next session's decision, where the stale path stayed leveraged.

**The asymmetry that must not be "fixed":** the linear strategy
(`TimingSignalAgent`/`LeveragedPositionAgent`) uses *unshifted* same-day
rolling windows — its execution lag lives downstream in
`TimingBacktestAgent.shift(1)` — so its last-bar row already *is* the next
session's position. Appending a NaN bar there would make every rolling
window NaN → signal 0 → a silent false de-risk. Documented at both call
sites. Likewise `gjr_garch_variance` drops NaN rows before fitting, so the
appended row would silently produce a flat decision — the live path asserts
`vol_method == "ewma"`.

Related: the execution weights row is now indexed by the **fill session**
for both strategies, so the StateStore ledger's `run_date` — and the PDT
same-day-round-trip guard keyed on it — matches the session orders actually
fill in.

### 2. Linear run silently skipped when the Kelly step failed (architectural)

Workflow steps are sequential by default; the two strategies are fully
independent (separate accounts, separate ledgers). Linear now runs under
`if: ${{ !cancelled() }}`.

### 3. Live-state push lost on origin/main races (architectural)

The persist step's plain `git push` failed whenever origin/main advanced
after checkout (code pushed mid-run, concurrent manual run) — and the
runner being ephemeral, the day's order ledger died with it, taking the PDT
guard's memory along. Now rebases onto the updated remote and retries; the
state commit only touches `data/live_state_*.db`, which nothing else
writes, so the rebase cannot conflict.

### 4. Buys could precede the sells that fund them (architectural)

`ExecutionAgent` iterated the weights dict in order (QQQ, QLD, TQQQ), so
rotations could submit the buy first, relying on Reg-T margin buying power
to cover both legs simultaneously — exactly-zero slack when fully invested
at 1x and rotating everything. Sells are now submitted first.

### 5. Fetch window was timezone-dependent; partial intraday bars could
### enter the signal math (conceptual — found by the final CI verification)

`_fetch_live_universe` used `end = date.today()` — whose meaning depends on
the machine's timezone. Observed live, same logical request, same evening:
the UTC CI runner's fetch stopped at Friday's bar (Monday's completed close
silently excluded — a real staleness had that run submitted orders), while
the UTC+9 laptop included Monday. Worse, mid-session runs got the current
day's **in-progress partial bar** back from both Alpaca IEX and yfinance,
feeding an 11am price into math validated exclusively on completed closes.

Fix: fetch with a 2-day-padded end (timezone-proof), then
`utils/live_decision.py::drop_incomplete_last_bar` removes any final row
whose session hasn't officially closed, judged against the **broker's own
clock and calendar** (`_session_closed_checker`) rather than local time.
Net behavior, everywhere, regardless of machine timezone: the signal sees
exactly the completed sessions and nothing else.

## Known limitations, deliberately deferred (not silently ignored)

- **Execution-time drift vs. the backtest convention.** Backtests assume
  close-to-close returns (enter at the prior close). Live orders fill at
  ~10:00–11:00am ET (15:00 UTC cron; DST swings the local time by an hour).
  The overnight gap between yesterday's close and this morning's fill is
  untracked slippage relative to the backtest. Standard practical
  compromise for daily-bar strategies; not fixable with a fixed-time cron,
  only reducible by trading closer to the open (which trades against
  opening-range spread noise) or modeling open-execution in the backtest.
- **IEX feed vs. SIP official closes.** Live decisions price off Alpaca's
  free-tier IEX feed; the backtests were built on yfinance (SIP official)
  closes. IEX last-trade can differ from the official close by small
  amounts. Immaterial for ~55-90bp daily-move signals on QQQ-family
  liquidity, but it is a (tiny) regime difference between validated and
  live inputs.
- **`^VIX` still rides yfinance** (with retry) — it's an index, not on
  Alpaca's stock API. Isolated to one signal component of the linear
  strategy; a FRED `VIXCLS` fallback is the natural next hardening step if
  yfinance flakes on it in CI.
- **No explicit failure alerting** beyond GitHub's default
  failed-workflow notification emails. Acceptable while supervised daily;
  revisit if runs go genuinely unattended for long stretches.
- **Simulated-vs-broker hysteresis state.** `KellyPositionAgent`'s
  entry/exit hysteresis tracks the *simulated* exposure path from history,
  not the actual broker position. The two agree as long as every scheduled
  run executes and fills; a missed run or partial fill makes the guard's
  `current_exposure` briefly diverge from reality until the delta-based
  execution reconverges them. Self-correcting, but worth knowing when
  reading the ledger.
- **Market holidays**: handled (Alpaca calendar gate); early-close days
  (e.g. day after Thanksgiving, 1pm ET close) still run at 15:00 UTC —
  fine in winter (10:00am ET), but the *summer* 11:00am ET firing is also
  before any early close, so no action needed.

## Verification trail

- Full suite: 301 passed after all four fixes (was 296 at baseline; 5 new
  regression tests added under TDD, RED confirmed before each fix).
- Local dry-runs of both strategies after each behavioral change: coherent
  decisions, correct session labeling
  ("Position for session X, decided from data through Y").
- CI verification: manual `workflow_dispatch` dry-run green after push.
