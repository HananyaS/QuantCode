# Intraday Phase 2 — ExecutionAgent and SimBroker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ExecutionAgent and its SimBroker: every `OrderIntent` becomes `OrderUpdate`s and `Fill`s under the spec §4 fill rules (one-bar latency, stops at the next open with gaps, no fills in a halt, size cap, refit leg cost), deterministically, on the Phase 1 bus.

**Architecture:** `ExecutionAgent` is thin: it owns idempotency (seen `intent_id`s), the kill flag and the oversized-sell check, emits `accepted`/`rejected`, and routes every other event to a `Broker`. `SimBroker` is the Phase 2 broker: a per-session book of working orders and position lots that fills on `Bar`s, triggers brackets, time stops, halts, kill exits and the end-of-session flatten, and returns contract `Fill`/`OrderUpdate` events stamped with the cause's `ts`. Pure helpers (`sizing.py`, `costs.py`, `cost_inputs.py`) hold the cap, the cost model and the cost-input table so each is tested alone. The Phase 5 `AlpacaPaperBroker` implements the same `Broker` protocol and is out of scope.

**Tech Stack:** Python 3.13, stdlib (`dataclasses`, `zoneinfo`, `math`), pandas 2.3 / numpy 2.3 only inside `costs.py` (the penny scalar wrapper) and `cost_inputs.py`, pytest.

**Spec:** `research/intraday/docs/design/execution.md` (binding) and `docs/superpowers/specs/2026-10-26-intraday-system-design.md` §2, §4. Neighbours: `research/intraday/docs/design/risk.md` (shared cap parameters, `adv_shares` definition), `monitor.md` (order-flow checks on the events this agent emits), `engine.md` (tick-first, breadth-first ordering), ADR-001.

## Global Constraints

- Worktree `C:/Users/Shachar/Desktop/PythonProjects/QuantCode-intraday-exec`, branch `research/intraday-exec` cut from `research/intraday`. Never push, never merge to main.
- Write only `research/intraday/exec/` and `tests/test_intraday_exec*`. Before each commit, `"$PY" scripts/check_ownership.py --agent intraday-exec --base research/intraday` prints nothing.
- `PY="C:/Users/Shachar/AppData/Local/Programs/Python/Python313/python.exe"` (not on PATH). Commands run from the worktree root.
- Contracts are read-only (`research/intraday/contracts/`). Event fields are strict: int fields get `int` (not bool, not numpy), float fields get `float` (`0.0`, never `0`; cast numpy results with `float(...)`), ids/symbols non-empty `str`, tuples are `tuple`, datetimes are `datetime` with `tzinfo is timezone.utc` (never `pd.Timestamp`).
- Every emitted event's `ts` equals its cause's `ts` (the bus raises otherwise). `Fill.ts` = the filling `Bar.ts`, or the end-of-session tick's `ts`.
- Never emit while iterating a `set`/`frozenset`. Cross-symbol passes (Kill, end of session, time stops) iterate `sorted(..., key=lambda x: (x.symbol, x.seq))`.
- No wall clock, no `uuid`/`random`/`secrets`/`os.urandom`, no `alpaca.trading` or `*TradingClient` import anywhere under `research/intraday/exec/` in Phase 2 (`tests/test_intraday_guards.py` enforces all three). `research/intraday/exec/alpaca_paper.py` is not created.
- `exec/` imports only `research.intraday.contracts`, its own package, stdlib, pandas/numpy, and `research.penny` (never another agent package).
- ET times use `zoneinfo.ZoneInfo("America/New_York")`. Tests stamp with `fixtures.T_DEV` (2024-03-04 14:31 UTC = 09:31 ET, EST); never the fixtures' default `T0` (2026, inside the sealed holdout), so every `make_*` call in these tests passes `ts=`.
- Type hints required; assertions for sanity checks are raised explicitly (`raise AssertionError(...)`), never bare `assert` in `exec/` (stripped under `-O`).
- TDD: the failing test first; tests are never edited to make them pass. The full suite (`"$PY" -m pytest tests/ -q`) passes before each commit.
- Every commit message ends with two lines: `Refs: HananyaS/QuantCode-intraday-board#24` and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A quote stamped inside the fill bar's minute** (a `Quote` with `ts > bar.bar_start` reaching the broker before that bar's fill). Expected: the fill raises `AssertionError` instead of sizing or pricing an open with information from after it. Pinned in Task 3, `test_quote_after_fill_instant_raises`.
2. **The 09:34 summary and a fill at the 09:34 open.** The design states the guard as `Fill.ts >= 09:35 ET`, but a fill at the open of the bar starting 09:34 has `Fill.ts = 09:35` and happens at 09:34:00, before the `[09:34:00, 09:35:00)` summary window closes. Expected: the guard is on the fill instant (`bar_start`, or the tick for the end-of-session fill), so that fill uses `default_half_spread_bps`. Pinned in Task 6, `test_fill_at_0934_open_does_not_use_summary`.
3. **Overlapping exit triggers on one lot** (a stop trigger, then the time-stop tick, then a `Kill`, before any bar fills the exit). Expected: one pending exit per lot, total sell qty equals the position, never more. Pinned in Task 7, `test_one_exit_per_lot_when_triggers_overlap`.
4. **A stop hit inside the bar that filled the entry** (fill at the open, the low then crosses the stop). Expected: the fill bar is checked (the design's "from the fill bar onward"), and the exit fills at the *next* bar's open, never on the fill bar itself. Pinned in Task 6, `test_stop_triggered_on_entry_fill_bar`.
5. **No cost inputs for a symbol-day** (no `sigma_daily`/`adv_shares`). Expected: the fill raises `ValueError` naming the symbol and date, rather than charging zero impact. Pinned in Task 5, `test_missing_cost_inputs_raises`.

## Design-doc ambiguities resolved here

Each is a decision the executor must not re-open; the owning task pins it with a test.

- **09:35 guard on the fill instant, not `Fill.ts`** (Review Focus 2): a bar fill happens at `bar_start`; the summary is allowed iff `to_et(bar_start) >= 09:35` (so `Fill.ts >= 09:36` for bar fills); for the end-of-session fill the instant is the tick. Stricter than the doc's wording; same intent.
- **Entry wait window**: a buy may fill only on a bar with `intent.ts <= bar_start < intent.ts + entry_max_wait_min`; it expires on the first bar at or past the deadline, or the first tick with `ts > deadline`.
- **Quote freshness**: age is measured at the fill instant (`bar_start`), fresh iff `0 <= age <= quote_max_age_min`; a quote stamped after the fill instant raises.
- **One cap budget per bar and side**: all fills in one symbol on one bar draw on one `fill_cap` budget per side (an exit and a sell cannot each take the full quote).
- **End-of-session fill is uncapped**: the full open qty fills at the last close; the depth term of the cost model charges for the size.
- **Sell intents** carry their own `intent_id` on the `Fill`, reduce lots oldest first, and are rejected above `sellable_qty` = open qty not already committed to a pending exit or another working sell.
- **Entries into a halted symbol and after the end-of-session pass** are broker refusals (`rejected`, `"broker refusal: symbol halted"` / `"broker refusal: session closed"`).
- **Triggers during a halt** are still evaluated on any bar that arrives; the fill waits for the resume rule.
- **Missing cost inputs** raise `ValueError` (fail loud), rather than defaulting sigma or ADV.
- **Client order ids**: `intent_id` for an entry or a sell, `f"{intent_id}-exit"` for a bracket/time/kill exit, at most 128 characters.

---

## File Structure

```
research/intraday/exec/__init__.py      re-exports ExecutionAgent, SimBroker, ExecConfig, CostInputs, StaticCostInputs
research/intraday/exec/config.py        ExecConfig, ET, SUMMARY_READY_ET, HOLDOUT_START, to_et()
research/intraday/exec/ids.py           client_order_id()
research/intraday/exec/costs.py         CostInputs, CostInputsSource, StaticCostInputs, CostBook, cost functions
research/intraday/exec/cost_inputs.py   build_cost_inputs(): daily cache + 09:34 summary frames -> StaticCostInputs
research/intraday/exec/sizing.py        fresh_quote(), fill_cap()
research/intraday/exec/broker.py        Broker protocol, ExecEvent
research/intraday/exec/book.py          WorkingOrder, Lot, PendingExit (mutable internal state)
research/intraday/exec/sim_broker.py    SimBroker
research/intraday/exec/agent.py         ExecutionAgent
tests/test_intraday_exec_config.py      Task 1
tests/test_intraday_exec_costs.py       Task 2
tests/test_intraday_exec_sizing.py      Task 3
tests/test_intraday_exec_cost_inputs.py Task 4
tests/test_intraday_exec.py             Tasks 5-7: the design's named tests, through ExecutionAgent + SimBroker
tests/test_intraday_exec_replay.py      Task 8: bus cascade order, determinism and invariant sweep
```

Shared test helpers live at the top of `tests/test_intraday_exec.py` (Task 5) and are imported by `tests/test_intraday_exec_replay.py`:

```python
M = timedelta(minutes=1)
SESSION = date(2024, 3, 4)
def at(k: int) -> datetime: return T_DEV + k * M          # at(0) = 09:31 ET, at(3) = 09:34, at(390) = 16:01
def bar(k: int, symbol: str = "TEST", **ov) -> Bar:       # bar_start = at(k), ts = at(k + 1)
    return make_bar(symbol=symbol, ts=at(k + 1), bar_start=at(k), **ov)
INPUTS = CostInputs(sigma_daily=0.05, adv_shares=1_000_000.0)
def make_exec(inputs: dict[str, CostInputs] | None = None, **cfg) -> tuple[ExecutionAgent, SimBroker]:
    table = {(s, SESSION): ci for s, ci in (inputs or {"TEST": INPUTS, "AAA": INPUTS}).items()}
    broker = SimBroker(ExecConfig(**cfg), StaticCostInputs(table))
    return ExecutionAgent(broker), broker
```

Default `make_bar` volume is 10_000, so the no-quote cap is `floor(0.02 × 10_000) = 200` shares, above the default intent qty of 100.

---

### Task 1: Configuration and client order ids

**Files:**
- Create: `research/intraday/exec/config.py`, `research/intraday/exec/ids.py`
- Modify: `research/intraday/exec/__init__.py` (re-exports grow per task)
- Test: `tests/test_intraday_exec_config.py`

**Interfaces:**
- Produces:
  - `ET: ZoneInfo = ZoneInfo("America/New_York")`, `SUMMARY_READY_ET: time = time(9, 35)`, `HOLDOUT_START: date = date(2025, 8, 7)`.
  - `to_et(ts: datetime) -> datetime` (UTC-aware in, ET-aware out; `ValueError` on a non-UTC input).
  - `@dataclass(frozen=True, slots=True, kw_only=True) class ExecConfig` with `quote_size_frac: float = 1.0`, `quote_max_age_min: int = 5`, `bar_participation_cap: float = 0.02`, `entry_max_wait_min: int = 2`, `eod_et: time = time(16, 1)`, `impact_coefficient: float = 1.0`, `levels_penalty: float = 1.0`, `default_half_spread_bps: float = 100.0`. `__post_init__` raises `ValueError` on: a non-float float field or non-int int field (same strictness as the contracts), `quote_size_frac` or `bar_participation_cap` outside `(0, 1]`, `quote_max_age_min < 0`, `entry_max_wait_min < 1`, `impact_coefficient`/`levels_penalty` < 0, `default_half_spread_bps <= 0`, `eod_et` not a naive `time`.
  - `Leg = Literal["entry", "sell", "exit"]`; `client_order_id(intent_id: str, leg: Leg) -> str`: `intent_id` for `"entry"` and `"sell"` (so Phase 5's `client_order_id = intent_id` holds), `f"{intent_id}-exit"` for `"exit"`; `ValueError` if the result exceeds 128 characters (Alpaca's `client_order_id` limit) or `intent_id` is empty.

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_match_design_table() -> None:
    c = ExecConfig()
    assert (c.quote_size_frac, c.quote_max_age_min, c.bar_participation_cap) == (1.0, 5, 0.02)
    assert (c.entry_max_wait_min, c.eod_et, c.default_half_spread_bps) == (2, time(16, 1), 100.0)
    assert (c.impact_coefficient, c.levels_penalty) == (1.0, 1.0)

@pytest.mark.parametrize("kw", [dict(quote_size_frac=1), dict(quote_size_frac=0.0),
    dict(bar_participation_cap=1.5), dict(quote_max_age_min=5.0), dict(entry_max_wait_min=0),
    dict(default_half_spread_bps=0.0), dict(levels_penalty=-1.0), dict(quote_max_age_min=True)])
def test_invalid_config_rejected(kw) -> None:
    with pytest.raises(ValueError):
        ExecConfig(**kw)

def test_to_et_converts_t_dev() -> None:
    assert to_et(T_DEV).time() == time(9, 31) and to_et(T_DEV).date() == date(2024, 3, 4)

def test_client_order_id_deterministic_and_distinct_per_leg() -> None:
    assert client_order_id("20240304-TEST-1", "entry") == "20240304-TEST-1"
    assert client_order_id("20240304-TEST-1", "exit") == "20240304-TEST-1-exit"
    assert client_order_id("x", "exit") == client_order_id("x", "exit")
    with pytest.raises(ValueError):
        client_order_id("a" * 124, "exit")
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_exec_config.py -q` — Expected: FAIL (`ModuleNotFoundError: research.intraday.exec.config`).
- [ ] **Step 3: Implement** `config.py` and `ids.py` per the Interfaces block.
- [ ] **Step 4: Run** the same command — Expected: PASS. Then the full suite and the ownership check.
- [ ] **Step 5: Commit** `feat(intraday-exec): ExecConfig defaults and deterministic client order ids` (with the two trailer lines).

---

### Task 2: Leg cost model

**Files:**
- Create: `research/intraday/exec/costs.py`
- Test: `tests/test_intraday_exec_costs.py`

**Interfaces:**
- Consumes: `ExecConfig`, `to_et`, `SUMMARY_READY_ET` (Task 1); `research.penny.etl.quotes.leg_cost_bps(order_notional: pd.Series, side_notional: pd.Series, half_spread_bps: pd.Series, volatility_daily: pd.Series, participation: pd.Series, impact_coefficient: float = 1.0, levels_penalty: float = 1.0) -> pd.Series`.
- Produces:
  - `@dataclass(frozen=True, slots=True, kw_only=True) class CostInputs`: `sigma_daily: float`, `adv_shares: float`, `half_spread_bps: float | None = None`, `ask_notional: float | None = None`, `bid_notional: float | None = None`. `ValueError` unless `sigma_daily` finite `>= 0`, `adv_shares` finite `> 0`, and each optional value is `None` or finite `> 0`.
  - `class CostInputsSource(Protocol): def get(self, symbol: str, session_date: date) -> CostInputs | None`.
  - `class StaticCostInputs` with `__init__(self, table: Mapping[tuple[str, date], CostInputs])` and `get(...)`.
  - `CostSource = Literal["quote", "summary", "default"]`; `@dataclass(frozen=True, slots=True) class CostBook: half_spread_bps: float; side_notional: float | None; source: CostSource`.
  - `assert_summary_causal(fill_instant: datetime) -> None`: `AssertionError` when `to_et(fill_instant).time() < SUMMARY_READY_ET`.
  - `quote_half_spread_bps(quote: Quote) -> float` = `(ask − bid) / (2 × mid) × 10_000`, `mid = (bid + ask) / 2` (the `summarise_quotes` formula).
  - `select_cost_book(side: Literal["buy", "sell"], fill_instant: datetime, quote: Quote | None, inputs: CostInputs, config: ExecConfig) -> CostBook`. `quote` is already the fresh quote or `None` (Task 3 decides freshness). Order: fresh quote → `CostBook(quote_half_spread_bps(q), ask × ask_size if buy else bid × bid_size, "quote")` (sizes are shares); else if `inputs.half_spread_bps is not None` and `to_et(fill_instant).time() >= SUMMARY_READY_ET` → `CostBook(inputs.half_spread_bps, inputs.ask_notional if buy else inputs.bid_notional, "summary")` after calling `assert_summary_causal(fill_instant)`; else `CostBook(config.default_half_spread_bps, None, "default")`.
  - `leg_cost_bps_scalar(order_notional: float, side_notional: float | None, half_spread_bps: float, sigma_daily: float, participation: float, impact_coefficient: float, levels_penalty: float) -> float`: one-element Series into penny `leg_cost_bps`, `side_notional=None` passed as `np.nan` (the frame function then charges zero depth shortfall), result `float(...)`.
  - `fill_cost_bps(side, qty: int, price: float, fill_instant: datetime, quote: Quote | None, inputs: CostInputs, config: ExecConfig) -> tuple[float, CostSource]`: `order_notional = qty × price`, `participation = qty / inputs.adv_shares`.

- [ ] **Step 1: Write the failing tests**

```python
def test_scalar_matches_frame_function_on_grid() -> None:
    for on, sn, hs, sig, p in itertools.product([100.0, 5_000.0], [None, 50.0, 10_000.0],
                                                [5.0, 80.0], [0.0, 0.07], [0.0, 1e-4, 2.0]):
        frame = leg_cost_bps(pd.Series([on]), pd.Series([np.nan if sn is None else sn]),
                             pd.Series([hs]), pd.Series([sig]), pd.Series([p]), 1.3, 0.7)
        assert leg_cost_bps_scalar(on, sn, hs, sig, p, 1.3, 0.7) == float(frame.iloc[0])

def test_scalar_matches_closed_form() -> None:
    # half-spread + coef*sigma*sqrt(min(p,1))*1e4 + max(on/sn - 1, 0)*penalty*half-spread
    assert leg_cost_bps_scalar(2_000.0, 1_000.0, 50.0, 0.04, 0.01, 1.0, 1.0) == pytest.approx(
        50.0 + 0.04 * 0.1 * 1e4 + 1.0 * 50.0)

def test_quote_half_spread_bps() -> None:
    q = make_quote(ts=T_DEV, bid=1.00, ask=1.02)
    assert quote_half_spread_bps(q) == pytest.approx(0.02 / (2 * 1.01) * 1e4)

def test_summary_not_selected_before_0935() -> None:
    ci = CostInputs(sigma_daily=0.05, adv_shares=1e6, half_spread_bps=50.0,
                    ask_notional=5_000.0, bid_notional=4_000.0)
    t0934 = T_DEV + timedelta(minutes=3)
    assert select_cost_book("buy", t0934, None, ci, ExecConfig()).source == "default"
    book = select_cost_book("sell", t0934 + timedelta(minutes=1), None, ci, ExecConfig())
    assert book == CostBook(50.0, 4_000.0, "summary")

def test_assert_summary_causal_raises_before_0935() -> None:
    with pytest.raises(AssertionError):
        assert_summary_causal(T_DEV + timedelta(minutes=3, seconds=59))
    assert_summary_causal(T_DEV + timedelta(minutes=4))

def test_fresh_quote_beats_summary_and_uses_side_size() -> None:
    q = make_quote(ts=T_DEV, bid=1.00, ask=1.01, bid_size=300, ask_size=200)
    ci = CostInputs(sigma_daily=0.05, adv_shares=1e6, half_spread_bps=50.0)
    assert select_cost_book("buy", T_DEV + timedelta(minutes=10), q, ci, ExecConfig()).side_notional == 1.01 * 200
    assert select_cost_book("sell", T_DEV + timedelta(minutes=10), q, ci, ExecConfig()).side_notional == 1.00 * 300

@pytest.mark.parametrize("kw", [dict(sigma_daily=float("nan")), dict(adv_shares=0.0),
                                dict(half_spread_bps=0.0), dict(sigma_daily=1)])
def test_invalid_cost_inputs_rejected(kw) -> None:
    with pytest.raises(ValueError):
        CostInputs(**{**dict(sigma_daily=0.05, adv_shares=1e6), **kw})
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_exec_costs.py -q` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** `costs.py` per the Interfaces block.
- [ ] **Step 4: Run** — Expected: PASS; full suite; ownership check.
- [ ] **Step 5: Commit** `feat(intraday-exec): one-leg cost model with the 09:35 summary guard`.

---

### Task 3: Size cap and quote freshness

**Files:**
- Create: `research/intraday/exec/sizing.py`
- Test: `tests/test_intraday_exec_sizing.py`

**Interfaces:**
- Consumes: `ExecConfig` (Task 1).
- Produces:
  - `fresh_quote(quote: Quote | None, fill_instant: datetime, config: ExecConfig) -> Quote | None`: `AssertionError` if `quote.ts > fill_instant` (look-ahead into the fill bar); `None` if `fill_instant − quote.ts > timedelta(minutes=config.quote_max_age_min)`; else the quote.
  - `CapSource = Literal["quote", "bar"]`; `fill_cap(side: Literal["buy", "sell"], fill_instant: datetime, bar: Bar, quote: Quote | None, config: ExecConfig) -> tuple[int, CapSource]`: with a fresh quote `math.floor(config.quote_size_frac * (quote.ask_size if buy else quote.bid_size))`, `"quote"`; else `math.floor(config.bar_participation_cap * bar.volume)`, `"bar"`. Same expression as `risk.md`, so the two agents agree on equal inputs. Returns `int`.

- [ ] **Step 1: Write the failing tests**

```python
def test_fresh_quote_caps_by_side_size() -> None:
    b = make_bar(ts=T_DEV + 2 * M, bar_start=T_DEV + M, volume=10_000)
    q = make_quote(ts=T_DEV, ask_size=40, bid_size=70)
    assert fill_cap("buy", b.bar_start, b, q, ExecConfig()) == (40, "quote")
    assert fill_cap("sell", b.bar_start, b, q, ExecConfig()) == (70, "quote")

def test_stale_quote_falls_back_to_bar_volume() -> None:
    b = make_bar(ts=T_DEV + 7 * M, bar_start=T_DEV + 6 * M, volume=10_000)
    q = make_quote(ts=T_DEV, ask_size=40)            # age 6 min > 5
    assert fill_cap("buy", b.bar_start, b, q, ExecConfig()) == (200, "bar")
    q5 = make_quote(ts=T_DEV + M, ask_size=40)        # age exactly 5 min: fresh
    assert fill_cap("buy", b.bar_start, b, q5, ExecConfig()) == (40, "quote")

def test_quote_after_fill_instant_raises() -> None:
    b = make_bar(ts=T_DEV + 2 * M, bar_start=T_DEV + M)
    with pytest.raises(AssertionError):
        fill_cap("buy", b.bar_start, b, make_quote(ts=b.ts), ExecConfig())

def test_cap_is_int_and_floors() -> None:
    b = make_bar(ts=T_DEV + M, bar_start=T_DEV, volume=149)
    cap, _ = fill_cap("buy", b.bar_start, b, None, ExecConfig())
    assert cap == 2 and type(cap) is int
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_exec_sizing.py -q` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** `sizing.py`.
- [ ] **Step 4: Run** — Expected: PASS; full suite; ownership check.
- [ ] **Step 5: Commit** `feat(intraday-exec): fill size cap from the fresh quote or bar volume`.

---

### Task 4: Cost inputs from the daily cache and the 09:34 summary

**Files:**
- Create: `research/intraday/exec/cost_inputs.py`
- Test: `tests/test_intraday_exec_cost_inputs.py`

**Interfaces:**
- Consumes: `CostInputs`, `StaticCostInputs` (Task 2), `HOLDOUT_START` (Task 1).
- Produces: `build_cost_inputs(daily: pd.DataFrame, decision_summary: pd.DataFrame | None, session_date: date, *, holdout_start: date = HOLDOUT_START, unseal: bool = False, window: int = 20, min_sessions: int = 10) -> StaticCostInputs`.
  - `daily`: the daily cache panel (`research/penny/etl/bars.py::load_years` output) with at least `symbol, date, close, volume, close_unadj` (`close`, `volume` adjusted). `decision_summary`: `summarise_quotes` output for the 09:34 window, columns `symbol, date, half_spread_bps, ask_notional, bid_notional`; notionals are taken as-is (the round-lot unit question is `feed.md`'s, not re-decided here). File loading is the session runner's job.
  - `ValueError` if `session_date >= holdout_start and not unseal` (engine.md: every data adapter refuses holdout dates).
  - Per symbol, only rows with `date < session_date` (the day's own row never enters):
    - `sigma_daily` = sample std (`ddof=1`) of `close.pct_change()` over the last `window` returns, i.e. the last `window + 1` closes (the `bars.py:202` `vol_20d` definition, adjusted series);
    - `f_prev = close_unadj / close` of the last prior row; `adv_shares = mean(volume of the last window rows) / f_prev` (risk.md: one basis, the traded basis of `D` via `f_{D-1}`);
    - a symbol with fewer than `min_sessions` prior rows, or a non-finite result, gets no entry.
  - Summary fields filled from the `decision_summary` row with `date == session_date` (missing → `None`); values cast with `float(...)`.
  - Vectorised (groupby/tail), no per-row Python loop.

- [ ] **Step 1: Write the failing tests.** Fixture: `DATES = pd.bdate_range(end="2024-03-04", periods=26)` (`D = date(2024, 3, 4)` is `DATES[-1]`; the `date` column holds `pd.Timestamp`s, the function compares against `pd.Timestamp(session_date)`). One symbol `AAA`; adjusted `close = 1.0 + 0.01 * (i % 7)`; a 2:1 split takes effect at `SPLIT = DATES[20]`, so `close_unadj = 2 × close` for rows 0–19 and `= close` for rows 20–25 (`f = 2` then `1`); adjusted `volume = 1_000.0` on every prior row and `1e9` on `D`'s own row.

```python
def test_only_prior_sessions_enter() -> None:
    ci = build_cost_inputs(panel, None, D).get("AAA", D)
    assert ci.adv_shares == pytest.approx(1_000.0)          # f_{D-1} = 1; D's 1e9 ignored
    # mixing per-day factors would give (15*500 + 5*1000)/20 = 625: the test tells them apart

def test_adv_shares_uses_f_prev_for_every_session() -> None:
    d = SPLIT.date()                                         # last prior row (19) has f = 2
    ci = build_cost_inputs(panel, None, d).get("AAA", d)
    assert ci.adv_shares == pytest.approx(1_000.0 / 2.0)

def test_sigma_daily_matches_vol_20d_definition() -> None:
    closes = panel.loc[panel["date"] < D, "close"].tail(21)
    assert build_cost_inputs(panel, None, D).get("AAA", D).sigma_daily == pytest.approx(
        closes.pct_change().dropna().std())

def test_summary_fields_attached_for_session_date_only() -> None:
    summ = pd.DataFrame({"symbol": ["AAA", "AAA"], "date": [DATES[-1], DATES[-2]],
                         "half_spread_bps": [40.0, 99.0], "ask_notional": [5e3, 1.0], "bid_notional": [4e3, 1.0]})
    ci = build_cost_inputs(panel, summ, D).get("AAA", D)
    assert (ci.half_spread_bps, ci.ask_notional, ci.bid_notional) == (40.0, 5e3, 4e3)

def test_short_history_symbol_omitted() -> None:
    assert build_cost_inputs(panel.tail(9), None, D).get("AAA", D) is None

def test_holdout_date_refused_without_unseal() -> None:
    with pytest.raises(ValueError):
        build_cost_inputs(panel, None, date(2025, 8, 7))
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_exec_cost_inputs.py -q` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** `build_cost_inputs`.
- [ ] **Step 4: Run** — Expected: PASS; full suite; ownership check.
- [ ] **Step 5: Commit** `feat(intraday-exec): per symbol-day cost inputs from prior sessions only`.

---

### Task 5: ExecutionAgent, Broker protocol and SimBroker entries

**Files:**
- Create: `research/intraday/exec/broker.py`, `research/intraday/exec/book.py`, `research/intraday/exec/sim_broker.py`, `research/intraday/exec/agent.py`
- Test: `tests/test_intraday_exec.py` (helpers from File Structure at the top)

**Interfaces:**
- Consumes: Tasks 1–3 (`ExecConfig`, `to_et`, `client_order_id`, `CostInputsSource`, `fill_cost_bps`, `fresh_quote`, `fill_cap`).
- Produces:
  - `broker.py`: `ExecEvent = Fill | OrderUpdate`; `class Broker(Protocol)` with `refusal(self, intent: OrderIntent) -> str | None`, `submit(self, intent: OrderIntent) -> None`, `sellable_qty(self, symbol: str) -> int`, `on_bar(self, bar: Bar) -> tuple[ExecEvent, ...]`, `on_quote(self, quote: Quote) -> None`, `on_tick(self, tick: ClockTick) -> tuple[ExecEvent, ...]`, `on_halt(self, halt: Halt) -> tuple[ExecEvent, ...]`, `on_kill(self, kill: Kill) -> tuple[ExecEvent, ...]`.
  - `book.py` (`@dataclass(slots=True)`, mutable, internal): `WorkingOrder(intent: OrderIntent, client_order_id: str, remaining: int, not_before: datetime, deadline: datetime | None, seq: int)` (`deadline` = `intent.ts + entry_max_wait_min` for a buy, `None` for a sell); `ExitReason = Literal["stop", "target", "time", "kill"]`; `PendingExit(reason: ExitReason, not_before: datetime, client_order_id: str)`; `Lot(entry_id: str, symbol: str, open_qty: int, take_profit: float | None, stop_loss: float | None, time_stop: datetime | None, seq: int, exit: PendingExit | None = None, stranded: bool = False)`.
  - `SimBroker(config: ExecConfig, cost_inputs: CostInputsSource)` implementing `Broker`, plus `position(symbol: str) -> int` (sum of open lot qty), `exit_log() -> tuple[tuple[str, ExitReason], ...]` (entry id and reason, in decision order), `source_counts() -> dict[str, int]` (keys `cap:quote`, `cap:bar`, `cost:quote`, `cost:summary`, `cost:default`, one count per fill; R16 reports how often each source was used).
  - `ExecutionAgent(broker: Broker, name: str = "execution")`, `subscribes = (OrderIntent, Bar, Quote, ClockTick, Halt, Kill)`, `on_event(event: Event) -> tuple[Event, ...]`.
- Agent rules for an `OrderIntent`, first match wins, every outcome stamped `ts = intent.ts`, the id added to the seen set in every case:
  1. id already seen → `OrderUpdate(rejected, "duplicate intent_id")`;
  2. killed → `rejected, "after kill"`;
  3. `side == "sell"` and `qty > broker.sellable_qty(symbol)` → `rejected, "sell exceeds position"`;
  4. `broker.refusal(intent)` returns `r` → `rejected, f"broker refusal: {r}"` (SimBroker refuses `"symbol halted"` and, after the end-of-session pass, `"session closed"`);
  5. else `OrderUpdate(accepted, "")` and `broker.submit(intent)`.
- SimBroker `on_bar` for symbol S, in this order:
  1. `bar.bar_start <=` the last seen `bar_start` for S → `ValueError` (out of order).
  2. Halted S, or after the end-of-session pass → no fills (steps 3–5 skipped; step 6–7 still run).
  3. Per-bar, per-side cap budget: `fill_cap(side, bar.bar_start, bar, fresh_quote(latest_quote[S], bar.bar_start, cfg), cfg)` computed once per side; every fill on this bar draws it down.
  4. Exits (Task 6) and working sells (Task 6), then working buys, each in `seq` order: a buy with `not_before <= bar.bar_start` (the `not_before` of a new intent is `intent.ts`): if `bar.bar_start >= deadline` → `OrderUpdate(expired, "entry wait exceeded")`; else `q = min(remaining, budget)`; `q == 0` → `OrderUpdate(cancelled, "size cap zero")`; else `Fill(ts=bar.ts, intent_id, symbol, "buy", q, price=bar.open, cost_bps)` and a `Lot` with the intent's bracket; if `q < intent.qty` the `Fill` is followed, in the same returned tuple, by `OrderUpdate(cancelled, f"partial fill {q}/{intent.qty}: remainder {intent.qty - q} cancelled")`.
  5. Cost: `inputs = cost_inputs.get(S, to_et(bar.bar_start).date())`; `None` → `ValueError(f"no cost inputs for {S} on {date}")`; `fill_cost_bps(side, q, bar.open, bar.bar_start, fresh quote, inputs, cfg)`.
  6. Bracket triggers (Task 6).
  7. Record `bar.close` and `bar.bar_start` for S.
- `on_tick`: a working buy with `tick.ts > deadline` → `OrderUpdate(expired, "entry wait exceeded")` stamped `tick.ts` (the last eligible bar has `bar_start = deadline − 1 min`, `ts = deadline`, and is delivered before the tick at `deadline + 1 min`).

- [ ] **Step 1: Write the failing tests** (the design's names, plus two from Review Focus)

```python
def test_new_intent_accepted_with_intent_ts() -> None:
    ex, _ = make_exec()
    assert ex.on_event(make_order_intent(ts=at(1))) == (
        OrderUpdate(ts=at(1), intent_id="intent-1", status="accepted", detail=""),)

def test_entry_fills_at_next_bar_open_not_causing_bar() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))
    assert ex.on_event(bar(0)) == ()                      # bar_start 09:31 < intent ts: never fills it
    (f,) = ex.on_event(bar(1, open=1.10, high=1.12, low=1.08, close=1.11))
    assert (f.ts, f.side, f.qty, f.price) == (at(2), "buy", 100, 1.10)

def test_gap_minute_moves_fill_to_next_bar() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))              # minute at(1) is a Gap: no bar
    (f,) = ex.on_event(bar(2, open=1.03, high=1.05, close=1.04))
    assert (f.ts, f.price) == (at(3), 1.03)

def test_entry_expires_after_max_wait() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))              # eligible bar_starts: at(1), at(2)
    assert ex.on_event(make_clock_tick(ts=at(3))) == ()
    assert ex.on_event(make_clock_tick(ts=at(4))) == (
        OrderUpdate(ts=at(4), intent_id="intent-1", status="expired", detail="entry wait exceeded"),)
    ex2, _ = make_exec()
    ex2.on_event(make_order_intent(ts=at(1)))
    assert [e.status for e in ex2.on_event(bar(3))] == ["expired"]   # bar path, no fill

def test_fill_capped_by_shares_at_best_quote() -> None:
    ex, _ = make_exec()
    ex.on_event(make_quote(ts=at(1), ask_size=40))
    ex.on_event(make_order_intent(ts=at(1)))
    f, u = ex.on_event(bar(1))
    assert (f.qty, u.status) == (40, "cancelled")

def test_partial_entry_remainder_cancelled() -> None:
    ex, broker = make_exec()
    ex.on_event(make_order_intent(ts=at(1), qty=300))     # bar cap 200
    f, u = ex.on_event(bar(1))
    assert f.qty == 200 and u.detail == "partial fill 200/300: remainder 100 cancelled"
    assert broker.position("TEST") == 200 and ex.on_event(bar(2)) == ()

def test_partial_fill_precedes_remainder_cancel() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1), qty=300))
    out = ex.on_event(bar(1))
    assert [type(e) for e in out] == [Fill, OrderUpdate] and all(e.ts == at(2) for e in out)

def test_zero_cap_entry_cancelled() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))
    assert ex.on_event(bar(1, volume=49)) == (
        OrderUpdate(ts=at(2), intent_id="intent-1", status="cancelled", detail="size cap zero"),)

def test_cost_bps_matches_leg_cost_bps() -> None:
    ex, broker = make_exec()
    ex.on_event(make_quote(ts=at(1), bid=1.00, ask=1.01, ask_size=100))
    ex.on_event(make_order_intent(ts=at(1)))
    f, = ex.on_event(bar(1, open=1.00))
    hs = 0.01 / (2 * 1.005) * 1e4
    want = leg_cost_bps(pd.Series([100 * 1.00]), pd.Series([1.01 * 100]), pd.Series([hs]),
                        pd.Series([0.05]), pd.Series([100 / 1e6])).iloc[0]   # penny frame function
    assert f.cost_bps == pytest.approx(want, rel=1e-12)
    assert broker.source_counts()["cost:quote"] == 1 and broker.source_counts()["cap:quote"] == 1

def test_duplicate_intent_rejected_no_second_fill() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))
    assert ex.on_event(make_order_intent(ts=at(1))) == (
        OrderUpdate(ts=at(1), intent_id="intent-1", status="rejected", detail="duplicate intent_id"),)
    assert [e.qty for e in ex.on_event(bar(1))] == [100]

def test_out_of_order_bar_raises() -> None:
    ex, _ = make_exec()
    ex.on_event(bar(2))
    with pytest.raises(ValueError):
        ex.on_event(bar(2))

def test_missing_cost_inputs_raises() -> None:
    ex, _ = make_exec(inputs={"AAA": INPUTS})
    ex.on_event(make_order_intent(ts=at(1)))              # symbol TEST has no inputs
    with pytest.raises(ValueError, match="no cost inputs for TEST"):
        ex.on_event(bar(1))
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_exec.py -q` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** `broker.py`, `book.py`, the entry path of `sim_broker.py`, and `agent.py` per the Interfaces block. Lots and working orders live in insertion-ordered dicts keyed by id with a monotone `seq`; per-symbol passes filter in `seq` order.
- [ ] **Step 4: Run** — Expected: PASS; full suite; ownership check.
- [ ] **Step 5: Commit** `feat(intraday-exec): ExecutionAgent and SimBroker entries with one-bar latency and size cap`.

---

### Task 6: Exits — brackets, time stop, sell intents, and the 09:34 guard end to end

**Files:**
- Modify: `research/intraday/exec/sim_broker.py`
- Test: `tests/test_intraday_exec.py`

**Interfaces:**
- Consumes: Task 5's `SimBroker`, `Lot`, `PendingExit`, `WorkingOrder`.
- Rules added:
  - **Triggers** (on_bar step 6), for each open lot of S with `exit is None`, the fill bar included: `stop_loss is not None and bar.low <= stop_loss` → `PendingExit("stop", not_before=bar.ts, client_order_id(entry_id, "exit"))`; else `take_profit is not None and bar.high >= take_profit` → `"target"`. Stop is checked first, so a bar hitting both is a stop. Append `(entry_id, reason)` to `exit_log`. Triggers run even while halted (the fill waits).
  - **Exit fills** (on_bar step 4, before working sells and buys): a lot whose `exit.not_before <= bar.bar_start` sells `q = min(open_qty, sell budget)` at `bar.open`, `Fill(intent_id=entry_id, side="sell")`; `q == 0` → nothing, retry next bar; remainder stays pending; a lot at `open_qty == 0` is removed.
  - **Time stop** (on_tick): each open lot with `time_stop <= tick.ts` and `exit is None` gets `PendingExit("time", not_before=tick.ts, ...)`; it fills at the open of the bar with `bar_start >= tick.ts`.
  - **Sell intents**: `submit` of a `sell` creates a `WorkingOrder(deadline=None)`; on a bar with `not_before <= bar_start` it fills `q = min(remaining, sell budget, position(S))`, reducing lots in `seq` order (a lot reaching 0 is removed with its pending exit), `Fill(intent_id=<sell intent id>, side="sell")`; if `position(S) == 0` while `remaining > 0` → `OrderUpdate(cancelled, "position closed")`. `sellable_qty(S)` = open lot qty of lots with `exit is None` − `remaining` of working sells in S.

- [ ] **Step 1: Write the failing tests.** Helper, added to the shared helpers: `enter(ex: ExecutionAgent, symbol: str = "TEST", intent_id: str = "intent-1", **bracket) -> Fill` publishes `make_order_intent(ts=at(1), symbol=symbol, intent_id=intent_id, **bracket)` then `bar(1, symbol=symbol, open=1.00, high=1.01, low=0.99, close=1.00)` and returns the single buy `Fill` (qty 100).

```python
def test_stop_triggered_on_entry_fill_bar() -> None:
    ex, broker = make_exec()
    ex.on_event(make_order_intent(ts=at(1), stop_loss=0.95, take_profit=1.20))
    (f,) = ex.on_event(bar(1, open=1.00, high=1.01, low=0.93, close=0.96))   # buy only, no same-bar sell
    assert f.side == "buy" and broker.exit_log() == (("intent-1", "stop"),)
    (s,) = ex.on_event(bar(2, open=0.96, high=0.97, low=0.95, close=0.96))
    assert (s.side, s.price, s.ts) == ("sell", 0.96, at(3))

def test_zero_cap_exit_retries_next_bar() -> None:
    ex, _ = make_exec(); enter(ex, time_stop=at(5))
    ex.on_event(make_clock_tick(ts=at(5)))
    assert ex.on_event(bar(5, volume=0)) == ()               # cap 0: no fill, no update
    (f,) = ex.on_event(bar(6))
    assert (f.side, f.qty) == ("sell", 100)

def test_stop_fills_at_next_bar_open_including_gap_down() -> None:
    ex, _ = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20)
    assert ex.on_event(bar(2, open=0.99, high=1.00, low=0.94, close=0.95)) == ()   # trigger only
    (f,) = ex.on_event(bar(3, open=0.80, high=0.82, low=0.78, close=0.81))
    assert (f.ts, f.side, f.qty, f.price, f.intent_id) == (at(4), "sell", 100, 0.80, "intent-1")

def test_stop_never_fills_at_stop_level() -> None:
    ex, _ = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20)
    ex.on_event(bar(2, open=0.99, high=1.00, low=0.90, close=0.96))
    (f,) = ex.on_event(bar(3, open=0.97, high=0.98, low=0.94, close=0.95))
    assert f.price == 0.97 and f.price != 0.95

def test_take_profit_fills_at_next_bar_open() -> None:
    ex, broker = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20)
    ex.on_event(bar(2, open=1.10, high=1.21, low=1.09, close=1.19))
    (f,) = ex.on_event(bar(3, open=1.18, high=1.19, low=1.15, close=1.16))
    assert f.price == 1.18 and broker.exit_log() == (("intent-1", "target"),)

def test_stop_wins_when_bar_hits_both() -> None:
    ex, broker = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20)
    ex.on_event(bar(2, open=1.00, high=1.25, low=0.90, close=1.00))
    assert broker.exit_log() == (("intent-1", "stop"),)

def test_time_stop_exit_at_bar_starting_at_tick() -> None:
    ex, _ = make_exec(); enter(ex, time_stop=at(10))
    assert ex.on_event(make_clock_tick(ts=at(10))) == ()
    assert ex.on_event(bar(9)) == ()                       # bar_start at(9) < decision at(10)
    (f,) = ex.on_event(bar(10, open=1.04, high=1.05, close=1.04))
    assert (f.ts, f.price, f.side) == (at(11), 1.04, "sell")

def test_exit_remainder_stays_pending() -> None:
    ex, broker = make_exec(); enter(ex, time_stop=at(5))
    ex.on_event(make_clock_tick(ts=at(5)))
    (f1,) = ex.on_event(bar(5, volume=2_500))              # sell cap 50
    (f2,) = ex.on_event(bar(6))
    assert (f1.qty, f2.qty, broker.position("TEST")) == (50, 50, 0)

def test_oversized_sell_rejected() -> None:
    ex, _ = make_exec(); enter(ex)
    assert ex.on_event(make_order_intent(ts=at(3), intent_id="s-1", side="sell", qty=150))[0].detail == "sell exceeds position"
    assert ex.on_event(make_order_intent(ts=at(3), intent_id="s-2", side="sell", qty=60))[0].status == "accepted"
    (f,) = ex.on_event(bar(3))
    assert (f.intent_id, f.side, f.qty) == ("s-2", "sell", 60)

def test_fill_at_0934_open_does_not_use_summary() -> None:
    ci = CostInputs(sigma_daily=0.05, adv_shares=1e6, half_spread_bps=50.0, ask_notional=5e3, bid_notional=5e3)
    ex, broker = make_exec(inputs={"TEST": ci})
    ex.on_event(make_order_intent(ts=at(3)))
    (f,) = ex.on_event(bar(3, open=1.00))                   # bar_start 09:34, Fill.ts 09:35
    assert f.cost_bps == pytest.approx(leg_cost_bps_scalar(100.0, None, 100.0, 0.05, 1e-4, 1.0, 1.0))
    assert broker.source_counts()["cost:default"] == 1

def test_0934_summary_not_used_before_0935() -> None:
    ci = CostInputs(sigma_daily=0.05, adv_shares=1e6, half_spread_bps=50.0, ask_notional=5e3, bid_notional=5e3)
    ex, broker = make_exec(inputs={"TEST": ci})
    ex.on_event(make_order_intent(ts=at(2)))
    ex.on_event(bar(2))                                     # bar_start 09:33: default
    ex.on_event(make_order_intent(ts=at(4), intent_id="intent-2"))
    (f,) = ex.on_event(bar(4, open=1.00))                   # bar_start 09:35: summary
    assert f.cost_bps == pytest.approx(leg_cost_bps_scalar(100.0, 5e3, 50.0, 0.05, 1e-4, 1.0, 1.0))
    assert (broker.source_counts()["cost:default"], broker.source_counts()["cost:summary"]) == (1, 1)
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_exec.py -q` — Expected: the new tests FAIL (no exit fills, `exit_log` empty).
- [ ] **Step 3: Implement** triggers, exit fills, time stops, sell intents and `sellable_qty` in `sim_broker.py`.
- [ ] **Step 4: Run** — Expected: PASS; full suite; ownership check.
- [ ] **Step 5: Commit** `feat(intraday-exec): bracket, time-stop and sell exits at the next bar's open`.

---

### Task 7: Halts, Kill and end of session

**Files:**
- Modify: `research/intraday/exec/sim_broker.py`, `research/intraday/exec/agent.py`
- Test: `tests/test_intraday_exec.py`

**Interfaces:**
- Consumes: Tasks 5–6.
- Rules added:
  - **Halt(halted=True)** for S: set halted; each working buy in S (seq order) → `OrderUpdate(cancelled, "halt")` stamped `halt.ts`. `refusal` returns `"symbol halted"` for a new buy in S. Working sells and pending exits stay.
  - **Halt(halted=False)** for S: clear halted; every pending exit and working sell in S gets `not_before = max(not_before, halt.ts)`. Emits nothing.
  - **Kill**: the agent sets `killed` (rule 2 in Task 5). SimBroker, once: every working buy, all symbols, `sorted` by `(symbol, seq)` → `OrderUpdate(cancelled, "kill")` stamped `kill.ts`; every open lot with `exit is None`, same order → `PendingExit("kill", not_before=kill.ts, ...)` (logged in `exit_log`). Lots with a pending exit keep it. A second `Kill` emits nothing.
  - **End of session** (on_tick, after time stops and expiry, once): the first tick with `to_et(tick.ts).time() >= eod_et`: working buys → `OrderUpdate(expired, "session end")`; working sells → `OrderUpdate(expired, "session end")`; then each open lot sorted by `(symbol, seq)`: halted S → `OrderUpdate(intent_id=entry_id, status="expired", detail="exit unfilled: halted at session end")` and `stranded = True` (the lot stays in the record); else `Fill(ts=tick.ts, intent_id=entry_id, side="sell", qty=open_qty, price=last close of S, cost_bps)` with the **full** open qty (no size cap: the session ends; the cost model's depth term charges the size), cost from `fill_cost_bps(..., fill_instant=tick.ts, quote=fresh_quote(latest, tick.ts))`. Afterwards `refusal` returns `"session closed"` and `on_bar` fills nothing.

- [ ] **Step 1: Write the failing tests**

```python
def test_no_fill_while_halted() -> None:
    ex, _ = make_exec(); enter(ex, time_stop=at(5))
    ex.on_event(make_halt(ts=at(4), halted=True))
    ex.on_event(make_clock_tick(ts=at(5)))
    assert ex.on_event(bar(5)) == () and ex.on_event(bar(6)) == ()

def test_entry_cancelled_on_halt() -> None:
    ex, _ = make_exec()
    ex.on_event(make_order_intent(ts=at(1)))
    assert ex.on_event(make_halt(ts=at(1), halted=True)) == (
        OrderUpdate(ts=at(1), intent_id="intent-1", status="cancelled", detail="halt"),)
    assert ex.on_event(make_order_intent(ts=at(2), intent_id="intent-2"))[0].detail == "broker refusal: symbol halted"

def test_exit_fills_after_resume_at_first_bar_open() -> None:
    ex, _ = make_exec(); enter(ex, time_stop=at(3))
    ex.on_event(make_clock_tick(ts=at(3)))                   # exit decided at(3)
    ex.on_event(make_halt(ts=at(3), halted=True))
    ex.on_event(make_halt(ts=at(9), halted=False, reason=""))
    assert ex.on_event(bar(8)) == ()                          # bar_start at(8) < resume at(9)
    (f,) = ex.on_event(bar(9, open=0.70, high=0.75, low=0.69, close=0.72))
    assert (f.ts, f.price) == (at(10), 0.70)

def test_kill_cancels_entries_and_exits_positions() -> None:
    ex, _ = make_exec()
    enter(ex, symbol="AAA")                                   # open lot in AAA
    ex.on_event(make_order_intent(ts=at(2), intent_id="intent-2"))   # working buy in TEST
    assert ex.on_event(make_kill(ts=at(2))) == (
        OrderUpdate(ts=at(2), intent_id="intent-2", status="cancelled", detail="kill"),)
    (f,) = ex.on_event(bar(2, symbol="AAA", open=0.98, high=0.99, low=0.97, close=0.98))
    assert (f.symbol, f.side, f.qty, f.price) == ("AAA", "sell", 100, 0.98)

def test_intent_after_kill_rejected() -> None:
    ex, _ = make_exec()
    ex.on_event(make_kill(ts=at(1)))
    assert ex.on_event(make_order_intent(ts=at(1))) == (
        OrderUpdate(ts=at(1), intent_id="intent-1", status="rejected", detail="after kill"),)

def test_end_of_session_closes_open_positions() -> None:
    ex, _ = make_exec()
    enter(ex, symbol="AAA"); enter(ex, symbol="TEST", intent_id="intent-2")
    ex.on_event(bar(2, symbol="TEST", close=1.03, high=1.05))
    out = ex.on_event(make_clock_tick(ts=at(390)))            # 16:01 ET
    assert [(e.symbol, e.qty, e.price, e.ts) for e in out] == [("AAA", 100, 1.00, at(390)), ("TEST", 100, 1.03, at(390))]
    assert ex.on_event(make_order_intent(ts=at(390), intent_id="late"))[0].detail == "broker refusal: session closed"

def test_halted_position_expires_at_session_end() -> None:
    ex, broker = make_exec(); enter(ex)
    ex.on_event(make_halt(ts=at(300), halted=True))
    assert ex.on_event(make_clock_tick(ts=at(390))) == (OrderUpdate(
        ts=at(390), intent_id="intent-1", status="expired", detail="exit unfilled: halted at session end"),)
    assert broker.position("TEST") == 100

def test_one_exit_per_lot_when_triggers_overlap() -> None:
    ex, _ = make_exec(); enter(ex, stop_loss=0.95, take_profit=1.20, time_stop=at(3))
    ex.on_event(bar(2, open=0.99, high=1.00, low=0.90, close=0.96))   # stop decided at(3)
    ex.on_event(make_clock_tick(ts=at(3))); ex.on_event(make_kill(ts=at(3)))
    sells = [e for k in (3, 4, 5) for e in ex.on_event(bar(k)) if isinstance(e, Fill)]
    assert sum(f.qty for f in sells) == 100 and len(sells) == 1
```

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_exec.py -q` — Expected: the new tests FAIL.
- [ ] **Step 3: Implement** halt, kill and end-of-session handling.
- [ ] **Step 4: Run** — Expected: PASS; full suite; ownership check.
- [ ] **Step 5: Commit** `feat(intraday-exec): halt cancels and blocks fills, kill flattens, end-of-session flatten`.

---

### Task 8: The agent on the bus — cascade order, determinism, invariant sweep

**Files:**
- Test: `tests/test_intraday_exec_replay.py`
- Modify (only when a test here exposes a bug, fixed test-first with a named unit test in `tests/test_intraday_exec.py`): `research/intraday/exec/*.py`

**Interfaces:**
- Consumes: `EventBus(agents, sink=None, max_cascade=10_000)` and `publish(event) -> list[Envelope]` (`research/intraday/engine/bus.py`); `run_replay(agents, sources, log_path) -> str`, `clock_ticks(start, end, step=1 min)` (`research/intraday/engine/replay.py`); `read_log(path) -> Iterator[Envelope]` (`research/intraday/ops/recorder.py`); the Task 5 helpers.
- Produces (test-only, in this file):
  - `class ScriptedAgent` (`name: str`, `subscribes: tuple[type[Event], ...]`, `script: dict[tuple[str, datetime], tuple[Event, ...]]`), whose `on_event` returns `script.get((type(event).__name__, event.ts), ())`. It stands in for the feed (on `ClockTick`), risk (on `Bar`) and monitor.
  - `new_broker() -> SimBroker` with `INPUTS` for `AAA`, `BBB`, `TEST`.
  - `session_sources(seed: int) -> list[Iterable[Event]]`: source 0 = `clock_ticks(at(-1), at(391))`; source 1 = events from `random.Random(seed)` for `AAA`, `BBB`, `TEST`: per minute `k` in `0..389`, for each symbol in sorted order a `Bar` (`ts = at(k + 1)`, open a ±2% random walk from 1.00, `high/low` bracketing open and close, volume `randint(0, 20_000)`) or, with probability 0.1, nothing (a gap); then with probability 0.2 a `Quote` stamped `at(k + 1)` (sizes `randint(0, 500)`); then the minute's `Halt`s (one halt/resume pair on `BBB`, 6 minutes apart); then `OrderIntent`s (probability 0.05 per symbol-minute before `at(359)`, qty `randint(1, 400)`, bracket kind chosen among none / fixed `±5%` of the last close / `time_stop = ts + randint(2, 30) min`; one in ten repeats a prior id; one in ten is a `sell` with qty `randint(1, 400)`); a `Kill` at `at(200)` when `seed % 3 == 0`. Every event is built with contract-valid values (`float` prices, `int` sizes).

- [ ] **Step 1: Write the tests**

```python
def test_bus_partial_fill_then_cancel_in_one_cascade() -> None:
    feed = ScriptedAgent("feed", (ClockTick,), {("ClockTick", at(1)): (bar(0),), ("ClockTick", at(2)): (bar(1),)})
    risk = ScriptedAgent("risk", (Bar,), {("Bar", at(1)): (make_order_intent(ts=at(1), qty=300),)})
    ex, _ = make_exec()
    bus = EventBus([feed, risk, ex])
    first = bus.publish(make_clock_tick(ts=at(1)))
    second = bus.publish(make_clock_tick(ts=at(2)))
    assert [e.event.status for e in first if e.source == "execution"] == ["accepted"]   # no fill on bar(0)
    out = [e for e in second if e.source == "execution"]
    assert [type(e.event).__name__ for e in out] == ["Fill", "OrderUpdate"]
    assert out[0].event.qty == 200 and out[1].event.status == "cancelled"
    assert out[0].cause_seq == out[1].cause_seq          # same cause: the bar(1) envelope

def test_same_session_twice_identical_hash(tmp_path: Path) -> None:
    h = [run_replay([ExecutionAgent(new_broker())], session_sources(7), tmp_path / f"{i}.jsonl") for i in (0, 1)]
    assert len(h[0]) == 64 and h[0] == h[1]

def test_exec_hash_identical_across_hash_seeds(tmp_path: Path) -> None:
    # the tests/test_intraday_replay.py::_replay_in_subprocess pattern; the child imports
    # session_sources and new_broker from tests.test_intraday_exec_replay
    assert _hash_in_subprocess(tmp_path / "a.jsonl", "0") == _hash_in_subprocess(tmp_path / "b.jsonl", "12345")

@pytest.mark.parametrize("seed", range(20))
def test_invariants_hold(seed: int, tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    run_replay([ExecutionAgent(new_broker())], session_sources(seed), path)
    envs = list(read_log(path))
    for check in (fills_on_bars_at_or_after_decision, no_fill_while_halted, buy_qty_within_intent,
                  net_position_never_negative, one_ack_per_intent_and_no_fill_for_duplicates,
                  no_buy_after_kill, partial_buy_followed_by_cancel, exit_prices_are_opens_or_last_close):
        check(envs)                                       # each raises AssertionError with the offending seq
```

Each `check` is a test-file function `(envs: list[Envelope]) -> None` that asserts one design invariant over the log:
- `fills_on_bars_at_or_after_decision`: every buy `Fill` has a cause envelope that is a `Bar` with `bar_start >=` its intent's `ts`; every sell `Fill`'s cause is a `Bar` or the end-of-session `ClockTick`.
- `no_fill_while_halted`: no `Fill` for a symbol between its `Halt(True)` and `Halt(False)` envelopes.
- `buy_qty_within_intent`: total buy qty per `intent_id` `<=` the intent's `qty`.
- `net_position_never_negative`: running buys minus sells per symbol never drops below 0.
- `one_ack_per_intent_and_no_fill_for_duplicates`: each `OrderIntent` envelope gets exactly one `accepted` or `rejected` `OrderUpdate` caused by it; a rejected id's later `Fill`s are only those of its first, accepted occurrence.
- `no_buy_after_kill`: no buy `Fill` after the `Kill` envelope.
- `partial_buy_followed_by_cancel`: a buy `Fill` with `qty <` intent qty is immediately followed by `OrderUpdate(cancelled)` for the same id with the same `cause_seq`.
- `exit_prices_are_opens_or_last_close`: a sell `Fill` caused by a `Bar` has `price == bar.open`; one caused by a tick has the symbol's last logged bar close.

- [ ] **Step 2: Run** `"$PY" -m pytest tests/test_intraday_exec_replay.py -q` — Expected: PASS if Tasks 5–7 are right. A failure is a bug in them: reproduce it as a named unit test in `tests/test_intraday_exec.py`, see it fail, fix, rerun.
- [ ] **Step 3: Run** the full suite; `"$PY" scripts/check_ownership.py --agent intraday-exec --base research/intraday` (prints nothing); `grep -rnE "alpaca\.trading|TradingClient|uuid|random|datetime\.now|time\.time" research/intraday/exec/` (no output).
- [ ] **Step 4: Commit** `test(intraday-exec): bus cascade order, replay determinism and invariant sweep`.

---

## Out of scope

`AlpacaPaperBroker`, live reconciliation (blocked on a contract change, `execution.md` Open questions), `utils/state_store.py` persistence, and the session runner that loads the parquet caches into `build_cost_inputs`. `FakeTradingClient` (`tests/test_execution_agent.py:36`, `submit_order(order_data)`, `get_all_positions()`, `get_account()`) is the Phase 5 test double and is not used here.
