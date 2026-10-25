# Intraday microcap system: plan (edge first, paper only, agentic design)

## Context

The user's design has five parts: a pre-open screen for candidates with a large intraday rise
potential; real-time per-symbol streams starting around 09:25–09:30; a lightweight model that
reads the last few bars and returns a prediction; a portfolio/risk manager that places orders
with TP + SL and checks liquidity; and a broker API.

User decisions (2026-10-25/26):
- **Prove the edge first.** Build the replay and research path before any live feed or broker.
- **The screen runs just before the open**, at 09:00–09:25 ET.
- **The data budget is decided after the research.**
- **Paper trading only.**
- **Earlier research results are priors, not conclusions.** Re-check each one unless it is
  certain to apply here.
- **Add an agentic design** that keeps each module to a single responsibility and lets the
  parts be built in parallel.

Intended outcome: a modular system whose replay mode and live mode run the same agents. Its
first deliverable is a pre-registered test of **the user's own design** as the main hypothesis.

## 1. Re-checking the priors

I read the source of each claim I relied on in the first draft. Most were measured on a
different question and do not carry over.

| Claim in draft 1 | What it was actually measured on | Applies here? | Re-test as |
|---|---|---|---|
| "Every stop hurts" (§70) | The **retracted** 10-session daily long (a look-ahead sample, §72), with stops from −15% to −50% on daily closes | **No.** Different horizon, no model, poisoned sample | A factor in the exit grid |
| "Exit policy is not the lever" (§16) | 9 policies on the round-2 gap-up population, which had **no predictive signal** | **Only without a signal.** With no edge, no exit rule can create one (optional stopping on a martingale). With a model, exits are a real lever | A factor in the exit grid |
| "Gap-up stocks are the losing tail" (round 5) | The unconditional open→close mean, held to the close, with no model | **Partly.** It is the baseline the model has to beat, not a verdict on the user's screen plus model | The user's screen is the main arm; controls run beside it |
| "No minute-bar model succeeded" | Round 3: the pre-open path only, entry at 09:35, held to the close. Round 15 H3: minute features used to pick names held **60 days** | **No.** A model that reads **live bars after the open** and predicts **minutes ahead** was never tested | The main question of the round |
| Round trip of about 66.6 bps at $2k | Measured on candidates with ADV ≥ $5M (round 4) | **The method transfers; the number does not.** It has to be re-measured on this round's population | The cost model refit on this round's population |
| Free real-time data is IEX only | `alpaca_fetch.py` docstring plus the round-0 probe | Probably yes, but it is a vendor fact that can change | Check against current Alpaca docs before Phase 3 |
| Adjusted prices leak the future | Six separate incidents (§57–§97) | **Yes.** This is about how the data is built, not about markets | Hard guard tests |
| A GitHub Actions job is capped at about 6h | Vendor documentation, not checked this session | Needs checking | Check before choosing where the system runs |

What the research does give this round is its **method**, and that carries over in full:
pre-register before running, run a no-selection control and a random-feature control, judge
the compounded account rather than the per-trade mean, use unadjusted prices in every
filter, read the biggest trades, and use day-clustered CIs.

## 2. Runtime architecture: deterministic agents on an event bus

This follows the project's existing rule that agents are deterministic modules. Each agent has
exactly one responsibility, consumes and emits **typed events only**, and holds no reference
to any other agent. The bus is the only coupling.

```
             ┌──────────── EventBus (in-process; transport swappable) ────────────┐
ScreenerAgent ─Candidates→ FeedAgent ─Bar/Quote→ StateAgent ─Window→ PredictorAgent ─Signal→
  RiskAgent ─OrderIntent→ ExecutionAgent ─Fill/OrderUpdate→ (RiskAgent, RecorderAgent)
MonitorAgent ─Halt/Kill→ (RiskAgent, ExecutionAgent)      RecorderAgent ← every event
```

| Agent | Single responsibility | Consumes → emits | Replay version / live version |
|---|---|---|---|
| ScreenerAgent | Build the candidate list at 09:00–09:25 | daily + pre-market data → `Candidates` | same code on both |
| FeedAgent | Turn market data into events | → `Bar`, `Quote`, `Gap` | `ReplayFeed` (cached SIP) / `AlpacaStreamFeed` |
| StateAgent | Keep a rolling window of the last k bars per symbol, with point-in-time features | `Bar` → `Window` | same |
| PredictorAgent | Map a window to a prediction. A frozen artifact, its hash logged | `Window` → `Signal` | same |
| RiskAgent | Sizing, position limits, quote-size liquidity cap, TP/SL selection, daily loss limit. **The only agent allowed to emit an `OrderIntent`** | `Signal`, `Fill`, `Halt` → `OrderIntent` | same |
| ExecutionAgent | Translate an `OrderIntent` into broker calls, manage order state, idempotent ids, reconcile | `OrderIntent` → `Fill`, `OrderUpdate` | `SimBroker` / `AlpacaPaperBroker` (`paper=True` asserted) |
| MonitorAgent | Halts, stale feed, reconciliation mismatch, kill switch | all events → `Halt`, `Kill` | replay of the halt log / Nasdaq halt RSS |
| RecorderAgent | Append-only event log, which makes any day replayable | all events → disk | same |

Design rules:
- **Contracts are the only shared code.** `intraday/contracts/` holds the frozen event
  dataclasses (UTC timestamps, schema version) and the agent protocols. No agent imports
  another agent.
- **Time comes from a `Clock` event**, never `datetime.now()`. Replay mode is therefore
  deterministic, and two runs of one day give event logs with the same hash.
- **Single process now, services later.** The bus transport sits behind an interface. If
  the design ever needs real services, swap in-memory for ZeroMQ or Redis and no agent changes.
  This keeps your service split as a deployment option without paying for it before there is
  an edge.
- **Tests per agent:** each agent is tested alone by feeding it recorded event fixtures and
  asserting what it emits. Integration tests replay whole recorded days.

## 3. Development workflow: parallel agents, clear ownership

Agent definitions go in `.claude/agents/intraday-*.md`, following the existing
`breakout-*` pattern.

| Role | Owns (writes only here) | Notes |
|---|---|---|
| `intraday-architect` | `intraday/contracts/`, the ADRs, engine (bus, replay) (amended 2026-10-26, Ruling 9: engine stays with the architect) | Runs first, alone. Contract changes later need an ADR and the architect |
| `intraday-data` | ScreenerAgent, FeedAgent, cached data ETL | Reuses `research/penny/etl/*` |
| `intraday-model` | StateAgent, PredictorAgent, research scripts | Works only after the pre-registration is committed |
| `intraday-risk` | RiskAgent | Built against fixture `Signal`s; needs no model |
| `intraday-exec` | ExecutionAgent, SimBroker, later AlpacaPaperBroker | Built against `FakeTradingClient` (`tests/test_execution_agent.py`) |
| `intraday-ops` | MonitorAgent, Recorder sink | |
| `intraday-hypothesis-designer` | `PREREGISTRATION_R16.md` | Writes only; runs nothing |
| `intraday-critic` | `docs/critiques/` | Reviews at gates G1 (design), G2 (result), G3 (before paper). Never fixes code |
| integrator (main session) | merges, end-to-end replay tests | Resolves cross-module problems |

Mechanics:
- **Each builder works in its own git worktree** on its own branch, cut from
  `research/intraday`.
- **File ownership is enforced in review.** A change outside an agent's owned paths fails
  review.
- **Contract tests guard the seams.** Each consumer tests against the contract fixtures, so the
  builders cannot drift apart.
- **TDD everywhere.** Tests are never edited to make them pass.
- **Merge order:** contracts → (data ∥ risk ∥ exec ∥ ops) → integration replay → model
  research.

```
Week-ish 1:  [architect: contracts + ADR + bus/clock]   [designer: pre-registration R16]
Then:        data ─┐   risk ─┐   exec(Sim) ─┐   ops ─┐        critic G1 on pre-registration
                   └───────── integration: replay one recorded day end-to-end ──┘
Then:        model research (R16) inside the engine → critic G2 → Gate I
If Gate I:   data(AlpacaStreamFeed) ∥ exec(AlpacaPaperBroker) ∥ ops(halts, kill) → critic G3 → paper
```

## 4. Research: round 16 tests the user's design

`research/penny/PREREGISTRATION_R16.md`, committed before any result exists.
- **Main arm (the user's design):** the pre-open screen for intraday-rise potential, ranked
  by pre-market features. Then the model reads the last k bars after the open and predicts
  the move over the next h minutes. Then RiskAgent enters with TP/SL.
- **Controls:** no selection (all liquid low-priced names on the same days), a random feature
  used as the score, and the screen alone with no model.
- **Factors (pre-declared, kept small):**
  - k ∈ {5, 15, 30} bars, the window the model sees
  - h ∈ {5, 15, 30} minutes, the prediction horizon
  - model ∈ {logistic/GBDT on window aggregates, a small 1-D CNN or transformer on the raw
    window}
  - exit ∈ {time stop only, TP+SL fixed in percent, TP+SL scaled by ATR}
  - The trial count is recorded and Deflated Sharpe is reported.
- **Simulated fills:** one-bar latency. An SL is filled at the next bar's price, gaps
  included, never at the stop level. Nothing fills during a halt. Costs come from the refit
  cost model, with size capped by the shares at the best quote.
- **Bar to pass:** the compounded-account net return has a day-clustered CI that excludes
  zero, and it beats every control.
- **Holdout:** the 2016–2026 minute history was read in rounds 1–5, but only for
  hold-to-close questions. The pre-registration sets aside the **last 12 months as a sealed
  historical holdout for this question**, and it also treats **forward paper trading as the
  final holdout**.

**Gate I:** the main arm passes → build the live parts. It fails → record the result, stop
the intraday track, and do no re-tuning.

## 5. Phases

1. **Contracts and engine skeleton.** Contracts, bus, clock, RecorderAgent, ADR-001
   (event-driven single process). Critic G1 reviews R16 at the same time.
2. **Parallel build of the replay stack.** Data, risk, exec(Sim) and ops, each in its own
   worktree. **Exit criterion:** one recorded day replays end to end, twice, with identical log
   hashes.
3. **Round 16 research** inside the engine → critic G2 → Gate I.
4. **Data-feed decision:** replay the passing strategy on IEX bars versus SIP bars over the
   same days, with the current Alpaca terms checked first. The cost of the feed is then
   decided by that number.
5. **Live paper:** build `AlpacaStreamFeed`, `AlpacaPaperBroker` and the halt/kill monitor
   in parallel. Choose an always-on host after checking runner limits. Critic G3. Every night,
   replay the same day and diff it against the live log. End with `utils/go_live_gate.py`. The
   system stays on paper.

## Reuse (on `research/penny-veto`)
- `etl/alpaca_fetch.py`: `fetch_minute_for_day`, `make_data_client`
- `scripts/fetch_premarket.py`, `features/premarket.py` (`assert_no_post_decision_bars`)
- `etl/quotes.py`: `CostModel`, `round_trip_cost_bps`, `tradeable_mask` (refit on the new
  population)
- `etl/corporate_actions.py`: the split detectors
- `model/sequence.py`, `model/train.py`
- `eval/nulls.py`, `eval/folds.py`, `eval/portfolio.py`
- From main: `tests/test_execution_agent.py::FakeTradingClient`, `utils/state_store.py`,
  `utils/alerting.py`, `utils/go_live_gate.py`

## Constraints
- New branch `research/intraday`, cut from `research/penny-veto`. **Never merged to main.**
- Only ExecutionAgent's live adapter may import `alpaca.trading`. The existing guard in
  `tests/test_breakout_etl.py` gets extended to cover this.
- Unadjusted prices in every filter. The leak guards rescale price and volume together (§99).
- The full test suite passes before every commit. Python is
  `C:\Users\Shachar\AppData\Local\Programs\Python\Python313\python.exe`.

## Verification
- Per agent: `tests/test_intraday_<agent>.py` written test first, using fixtures from the
  contracts. Contract tests run on every branch.
- Engine: replaying a recorded day twice gives the same event-log hash. Fake broker tests
  check idempotency, reconciliation and the kill switch.
- Research: the `eval/nulls.py` battery passes, both controls are reported, and the critic
  verdicts are filed.
- Paper: a dry-run day first, then a nightly live-versus-replay diff report.
