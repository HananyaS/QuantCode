# Intraday Phase 2 — RiskAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build RiskAgent, the only agent that turns a `Signal` into an `OrderIntent`: entry rules, sizing under a notional and a liquidity cap, a TP/SL/time-stop bracket chosen by config, position and slot books kept exactly from `Fill`/`OrderUpdate`, halts, kill and a daily loss limit.

**Architecture:** Pure modules under `research/intraday/risk/` (config, ET session bounds, brackets, sizing, books, adv) composed by one `RiskAgent` whose `on_event` dispatches by event type. The exit grid of R16 (time stop only; TP+SL fixed %; TP+SL ATR-scaled) is a `RiskConfig.exit_mode` value looked up in a dispatch table, never an `if` chain in the agent. The agent holds no reference to any other agent; it is tested by feeding it fixture events and asserting what it returns.

**Tech Stack:** Python 3.13, stdlib (`dataclasses`, `zoneinfo`, `math`, `collections`), pandas only for `adv_shares`, pytest.

**Spec:** `research/intraday/docs/design/risk.md` (binding), with `docs/superpowers/specs/2026-10-26-intraday-system-design.md` §2 and §4. Neighbours: `research/intraday/docs/design/execution.md` (partial fill, cancel order, fill prices), `research/intraday/docs/design/monitor.md` (`Halt`, `Kill`).

## Global Constraints

- Execution worktree: a separate worktree on branch `research/intraday-risk`, cut from `research/intraday`. Never push, never merge to main.
- Writes only `research/intraday/risk/` and `tests/test_intraday_risk*`. Contracts (`research/intraday/contracts/`) are read, never edited; a needed contract change is a request to `intraday-architect`.
- `PY="C:/Users/Shachar/AppData/Local/Programs/Python/Python313/python.exe"` (not on PATH). Commands run from the worktree root in Git Bash.
- Imports allowed in `risk/`: `research.intraday.contracts`, `research.intraday.risk`, stdlib, pandas. No other agent package, no `alpaca.trading`, no `*TradingClient` (`tests/test_intraday_guards.py`).
- No wall clock: no `datetime.now/utcnow/today`, `date.today`, `time.time/monotonic`, `pd.Timestamp('now')` (guard test). Time comes only from event `ts`.
- Contract field types are strict (`research/intraday/contracts/events.py`): `qty` is an exact `int`; prices are `float` (`0.0`, never `0`); symbols and ids are non-empty `str`; every datetime has `tzinfo is timezone.utc` (use `.astimezone(timezone.utc)`, never a `ZoneInfo("UTC")` or a `pd.Timestamp`).
- `OrderIntent` requires `time_stop > ts` and, for a buy, `stop_loss < take_profit` (a sell would need `take_profit < stop_loss`; Phase 2 emits buys only). The agent checks its own bracket before constructing, so the contract never raises inside a cascade (that would poison the bus).
- `RiskConfig` construction raises `ValueError` unless `last_entry_et < flatten_et`.
- Partial fill order (execution.md): `Fill(qty=k)` precedes `OrderUpdate(cancelled)` for the remainder in the same cascade. Risk keeps the filled `k` as a position and drops only the working remainder.
- Exit grid factors are config: `exit_mode ∈ {"time_stop", "fixed_pct", "atr"}` with `tp_pct`, `sl_pct`, `tp_atr`, `sl_atr`; the agent calls `make_bracket(...)`, which dispatches through `BRACKETS[cfg.exit_mode]`.
- Never emit while iterating a `set`/`frozenset`; sort first. (The agent emits at most one event per `Signal`; sets are used for membership only.)
- Tests use `research.intraday.contracts.fixtures` factories with `T_DEV` (2024-03-04 14:31 UTC = 09:31 ET, pre-holdout). Fixture defaults stamp `T0` (2026, inside the holdout): every test passes `ts` explicitly. No data files, no model.
- `zoneinfo.ZoneInfo("America/New_York")` needs the `tzdata` package on Windows; it is installed for `$PY`.
- Type hints required; small functions; `assert` for internal sanity, `ValueError` for bad inputs/config.
- TDD: failing test first; never edit a test to make it pass. The full `tests/` suite passes before every commit.
- Every commit message ends with:
  ```
  Refs: HananyaS/QuantCode-intraday-board#23
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
- Before reporting done: `$PY scripts/check_ownership.py --agent intraday-risk --base research/intraday` prints nothing.

## Review Focus

1. **ET boundaries across DST.** `last_entry_et`/`flatten_et` are wall-clock ET; a hard-coded UTC offset is right in March 2024 (EST) and an hour off in July (EDT). Expected: bounds are built with `zoneinfo` per session date. Pinned in Task 1, `test_session_bounds_follow_dst`, and Task 5, `test_last_entry_boundary_in_summer_session`.
2. **A degenerate bracket** (flat bars give ATR = 0 so `tp == sl == ref`; a wide ATR gives `stop_loss <= 0`). Expected: skip counted as `bad_bracket`, never a `ValueError` from the `OrderIntent` constructor inside a cascade. Pinned in Task 2, `test_zero_atr_bracket_is_bad`, and Task 5, `test_flat_bars_atr_bracket_skipped_not_raised`.
3. **Float under-floor in a liquidity cap** (`0.29 * 100 == 28.999999999999996`). Expected: the cap is 29 shares. Pinned in Task 3, `test_liquidity_cap_floor_is_robust_to_float_error`.
4. **Cash over-commit by concurrent working intents** when cash, not notional, binds (small `start_equity_usd`). Expected: sizing uses cash minus the notional reserved by working intents. Pinned in Task 3, `test_size_uses_available_cash`, and Task 6, `test_working_intents_reserve_cash`.
5. **Events from the wrong session or out of order** (a `Signal` stamped on another ET date, a symbol's `Bar` not after its last). Expected: raise `ValueError` (the agent's books and ATR would silently corrupt). Pinned in Task 5, `test_signal_from_other_session_raises` and `test_out_of_order_bar_raises`.

## Rulings on design-doc gaps (made here; flag to the controller if wrong)

- `session_date` is a constructor argument (one agent per session); a `Signal` or `Bar` on another ET date raises.
- Sizing uses `cash − reserved_cash()` (uncommitted cash), not raw `cash`.
- `cap` in `reason`: a liquidity cap is named only when strictly below the notional/cash qty; ties and cash-bound sizes read `notional`.
- Slots count distinct busy symbols (`position > 0` or a working intent), so a partly filled, still-working intent counts once.
- An entry counts toward `max_entries_per_symbol` when emitted, filled or not.
- Quote freshness is inclusive: age exactly `quote_max_age_min` is fresh.
- True range of a symbol's first bar of the session is `high − low`; ATR needs `atr_period` true ranges from this session.
- Liquidity caps floor with a `1e-9` tolerance; the notional qty uses a plain floor.
- Loss limit trips at `marked_equity <= start_equity_usd × (1 − daily_loss_limit_pct)`.
- `RiskConfig` rejects exit parameters that do not belong to `exit_mode` (a stray `tp_pct` under `time_stop` is an error).
- An `OrderUpdate` or `Fill` for an id never opened raises; also raised: a buy fill for an id no longer working, an overfill, and a fill whose symbol does not match the intent's. An update for a known, closed id is a no-op.
- `adv_shares` (risk.md defines it, but no Phase-2 sizing rule uses it) is a pure helper in `risk/adv.py` that raises on fewer than 20 prior sessions.

---

## File Structure

```
research/intraday/risk/__init__.py   re-exports RiskAgent, RiskConfig, adv_shares
research/intraday/risk/config.py     ExitMode, EXIT_MODES, SkipReason, RiskConfig (validated)
research/intraday/risk/session.py    ET, et_date, session_bounds
research/intraday/risk/brackets.py   Bracket, true_range, mean_true_range, BRACKETS, make_bracket
research/intraday/risk/sizing.py     CapName, floor_shares, quote_is_fresh, size_order
research/intraday/risk/books.py      Books (positions, working intents, cash, realised P&L)
research/intraday/risk/adv.py        adv_shares
research/intraday/risk/agent.py      RiskAgent
tests/test_intraday_risk_config.py   config + session units
tests/test_intraday_risk_brackets.py bracket/ATR units
tests/test_intraday_risk_sizing.py   sizing units
tests/test_intraday_risk_books.py    books units
tests/test_intraday_risk.py          the design-doc test plan (agent level) + adv
tests/test_intraday_risk_bus.py      RiskAgent on a real EventBus; determinism
```

`research/intraday/risk/__init__.py` exists and is empty; Task 8 fills it.

---

### Task 1: Config and ET session bounds

**Files:**
- Create: `research/intraday/risk/config.py`, `research/intraday/risk/session.py`
- Test: `tests/test_intraday_risk_config.py`, `tests/test_intraday_risk.py` (first test only)

**Interfaces:**
- Produces (`config.py`):
  - `ExitMode = Literal["time_stop", "fixed_pct", "atr"]`; `EXIT_MODES: tuple[str, ...] = typing.get_args(ExitMode)`.
  - `SkipReason = Literal["killed", "loss_limit", "halted", "after_last_entry", "below_threshold", "symbol_busy", "symbol_entries", "max_open", "no_bar", "zero_qty", "atr_warmup", "bad_bracket"]`.
  - `@dataclass(frozen=True, slots=True, kw_only=True) class RiskConfig` with fields and defaults: `start_equity_usd: float = 25_000.0`, `position_notional_usd: float = 2_000.0`, `max_open_positions: int = 5`, `max_entries_per_symbol: int = 1`, `entry_threshold: float = 0.5`, `exit_mode: ExitMode = "time_stop"`, `tp_pct: float | None = None`, `sl_pct: float | None = None`, `tp_atr: float | None = None`, `sl_atr: float | None = None`, `atr_period: int = 14`, `hold_min: int | None = None` (None means `Signal.horizon_min`), `quote_size_frac: float = 1.0`, `quote_max_age_min: int = 5`, `bar_participation_cap: float = 0.02`, `daily_loss_limit_pct: float = 0.02`, `last_entry_et: time = time(15, 30)`, `flatten_et: time = time(15, 55)`.
- Produces (`session.py`): `ET: ZoneInfo = ZoneInfo("America/New_York")`; `et_date(ts: datetime) -> date`; `session_bounds(session_date: date, cfg: RiskConfig) -> tuple[datetime, datetime]` returning `(last_entry_utc, flatten_utc)`, each `datetime.combine(session_date, t, ET).astimezone(timezone.utc)`.

`RiskConfig.__post_init__` raises `ValueError` for: `last_entry_et >= flatten_et`; either time not naive; `exit_mode not in EXIT_MODES`; `start_equity_usd` or `position_notional_usd` `<= 0`; `max_open_positions`, `max_entries_per_symbol`, `atr_period`, `quote_max_age_min` `< 1`; `hold_min` not None and `< 1`; `quote_size_frac`, `bar_participation_cap` not in `(0, 1]`; `daily_loss_limit_pct` not in `(0, 1)`; and the exit-parameter rule: the parameters of the chosen mode must be set and the others must be None (`time_stop`: all four None; `fixed_pct`: `tp_pct > 0`, `0 < sl_pct < 1`, ATR pair None; `atr`: `tp_atr > 0`, `sl_atr > 0`, percent pair None). A grid cell is thus unambiguous: a stray `tp_pct` under `time_stop` is an error, not silently ignored.

- [ ] **Step 1: Write the failing tests**

`tests/test_intraday_risk.py` (create; later tasks append):
```python
def test_last_entry_must_precede_flatten() -> None:
    with pytest.raises(ValueError, match="last_entry_et"):
        RiskConfig(last_entry_et=time(15, 55), flatten_et=time(15, 55))
    with pytest.raises(ValueError, match="last_entry_et"):
        RiskConfig(last_entry_et=time(15, 56), flatten_et=time(15, 55))
    RiskConfig(last_entry_et=time(15, 54), flatten_et=time(15, 55))  # ok
```

`tests/test_intraday_risk_config.py`:
```python
def test_defaults_match_design_doc() -> None:
    c = RiskConfig()
    assert (c.start_equity_usd, c.position_notional_usd, c.max_open_positions) == (25_000.0, 2_000.0, 5)
    assert (c.exit_mode, c.atr_period, c.quote_size_frac, c.quote_max_age_min) == ("time_stop", 14, 1.0, 5)
    assert (c.bar_participation_cap, c.daily_loss_limit_pct) == (0.02, 0.02)
    assert (c.last_entry_et, c.flatten_et, c.hold_min) == (time(15, 30), time(15, 55), None)

@pytest.mark.parametrize("kw", [
    dict(exit_mode="time_stop", tp_pct=0.05),
    dict(exit_mode="fixed_pct", tp_pct=0.05),                 # sl_pct missing
    dict(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=1.0),     # stop at zero
    dict(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=0.03, tp_atr=2.0),
    dict(exit_mode="atr", tp_atr=2.0),
    dict(exit_mode="atr", tp_atr=0.0, sl_atr=1.0),
    dict(exit_mode="trailing"),
])
def test_exit_params_must_match_mode(kw) -> None:
    with pytest.raises(ValueError):
        RiskConfig(**kw)

def test_each_exit_mode_constructs() -> None:
    RiskConfig(exit_mode="time_stop")
    RiskConfig(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=0.03)
    RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0)

@pytest.mark.parametrize("kw", [dict(max_open_positions=0), dict(quote_size_frac=1.5),
    dict(bar_participation_cap=0.0), dict(daily_loss_limit_pct=1.0), dict(hold_min=0),
    dict(atr_period=0), dict(position_notional_usd=0.0)])
def test_bad_scalars_rejected(kw) -> None:
    with pytest.raises(ValueError):
        RiskConfig(**kw)

def test_session_bounds_are_utc_and_match_t_dev() -> None:
    last, flat = session_bounds(date(2024, 3, 4), RiskConfig())
    assert last == datetime(2024, 3, 4, 20, 30, tzinfo=timezone.utc)
    assert flat == datetime(2024, 3, 4, 20, 55, tzinfo=timezone.utc)
    assert last.tzinfo is timezone.utc and flat.tzinfo is timezone.utc
    assert et_date(T_DEV) == date(2024, 3, 4)

def test_session_bounds_follow_dst() -> None:
    last, flat = session_bounds(date(2024, 7, 1), RiskConfig())   # EDT, UTC-4
    assert last == datetime(2024, 7, 1, 19, 30, tzinfo=timezone.utc)
    assert flat == datetime(2024, 7, 1, 19, 55, tzinfo=timezone.utc)
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_intraday_risk_config.py tests/test_intraday_risk.py -v`
Expected: FAIL / collection error, `ModuleNotFoundError: research.intraday.risk.config`.

- [ ] **Step 3: Implement `RiskConfig` in `config.py` and `et_date`, `session_bounds` in `session.py`** per the Interfaces block. Error messages name the offending field (the first test matches `"last_entry_et"`).

- [ ] **Step 4: Run to verify they pass**

Run: `$PY -m pytest tests/test_intraday_risk_config.py tests/test_intraday_risk.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite and commit**

Run: `$PY -m pytest tests -q` → all pass.
```bash
git add research/intraday/risk/config.py research/intraday/risk/session.py tests/test_intraday_risk_config.py tests/test_intraday_risk.py
git commit -m "feat(intraday-risk): RiskConfig with exit-grid validation and ET session bounds" -m "Refs: HananyaS/QuantCode-intraday-board#23" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Brackets and ATR

**Files:**
- Create: `research/intraday/risk/brackets.py`
- Test: `tests/test_intraday_risk_brackets.py`

**Interfaces:**
- Consumes: `RiskConfig`, `ExitMode`, `EXIT_MODES`, `SkipReason` (Task 1); `Bar` (contracts).
- Produces:
  - `@dataclass(frozen=True, slots=True) class Bracket: take_profit: float | None; stop_loss: float | None`.
  - `true_range(bar: Bar, prev_close: float | None) -> float`: `bar.high - bar.low` when `prev_close is None` (the session's first bar for the symbol: agents are fresh per session, so no prior close exists); else `max(bar.high, prev_close) - min(bar.low, prev_close)`.
  - `mean_true_range(trs: Sequence[float], period: int) -> float | None`: mean of the last `period` values, `None` if `len(trs) < period`.
  - `BracketFn = Callable[[float, float | None, RiskConfig], Bracket | SkipReason]`.
  - `BRACKETS: Mapping[str, BracketFn]` (a `types.MappingProxyType`) with exactly the keys of `EXIT_MODES`: `time_stop -> Bracket(None, None)`; `fixed_pct -> Bracket(ref*(1+tp_pct), ref*(1-sl_pct))`; `atr -> "atr_warmup"` if `atr is None`, else `Bracket(ref + tp_atr*atr, ref - sl_atr*atr)`.
  - `make_bracket(ref_price: float, atr: float | None, cfg: RiskConfig) -> Bracket | SkipReason`: calls `BRACKETS[cfg.exit_mode]`; a returned `Bracket` with levels set must satisfy `math.isfinite` and `0 < stop_loss < ref_price < take_profit`, else `"bad_bracket"`.

- [ ] **Step 1: Write the failing tests**

```python
FIX = RiskConfig(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=0.03)
ATR = RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0, atr_period=3)

def test_dispatch_table_covers_exactly_the_exit_modes() -> None:
    assert set(BRACKETS) == set(EXIT_MODES)

def test_time_stop_has_no_levels() -> None:
    assert make_bracket(1.02, None, RiskConfig()) == Bracket(None, None)

def test_fixed_pct_levels() -> None:
    b = make_bracket(1.02, None, FIX)
    assert b.take_profit == pytest.approx(1.071) and b.stop_loss == pytest.approx(0.9894)

def test_true_range_first_bar_and_gap() -> None:
    b = make_bar(ts=T_DEV, bar_start=T_DEV - timedelta(minutes=1), open=1.05, high=1.06, low=1.02, close=1.05, vwap=1.05)
    assert true_range(b, None) == pytest.approx(0.04)
    assert true_range(b, 1.01) == pytest.approx(0.05)   # prior close below the bar's low
    assert true_range(b, 1.10) == pytest.approx(0.08)   # prior close above the bar's high

def test_mean_true_range_needs_period() -> None:
    assert mean_true_range([0.04, 0.03], 3) is None
    assert mean_true_range([0.04, 0.03, 0.05], 3) == pytest.approx(0.04)

def test_atr_levels_and_warmup() -> None:
    assert make_bracket(1.05, None, ATR) == "atr_warmup"
    b = make_bracket(1.05, 0.04, ATR)
    assert b.take_profit == pytest.approx(1.13) and b.stop_loss == pytest.approx(1.01)

def test_zero_atr_bracket_is_bad() -> None:
    assert make_bracket(1.05, 0.0, ATR) == "bad_bracket"

def test_atr_stop_at_or_below_zero_is_bad() -> None:
    wide = RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=30.0, atr_period=3)
    assert make_bracket(1.05, 0.04, wide) == "bad_bracket"
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_intraday_risk_brackets.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `brackets.py`** per the Interfaces block; one small function per mode registered in `BRACKETS`.

- [ ] **Step 4: Run to verify they pass** — same command, all PASS.

- [ ] **Step 5: Full suite and commit**

`$PY -m pytest tests -q`, then commit `brackets.py` and its test: `feat(intraday-risk): exit-grid brackets by dispatch table, ATR from true ranges` with the two trailer lines.

---

### Task 3: Sizing

**Files:**
- Create: `research/intraday/risk/sizing.py`
- Test: `tests/test_intraday_risk_sizing.py`

**Interfaces:**
- Consumes: `RiskConfig` (Task 1); `Quote` (contracts).
- Produces:
  - `CapName = Literal["notional", "quote", "bar"]`.
  - `floor_shares(x: float) -> int`: `math.floor(x + 1e-9)` for `x >= 0` (assert), so a fraction × an integer share count that lands a hair under an integer is not cut by one.
  - `quote_is_fresh(quote: Quote | None, signal_ts: datetime, max_age_min: int) -> bool`: `quote is not None and timedelta(0) <= signal_ts - quote.ts <= timedelta(minutes=max_age_min)` (age exactly `max_age_min` is fresh).
  - `size_order(*, ref_price: float, available_cash: float, signal_ts: datetime, bar_volume: int, quote: Quote | None, cfg: RiskConfig) -> tuple[int, CapName]`:
    - `notional_qty = math.floor(max(0.0, min(cfg.position_notional_usd, available_cash)) / ref_price)` (plain floor: rounding down here keeps `qty × ref_price <= position_notional_usd` exact);
    - liquidity: fresh quote → `("quote", floor_shares(cfg.quote_size_frac * quote.ask_size))`, else `("bar", floor_shares(cfg.bar_participation_cap * bar_volume))`;
    - returns `(liq_qty, liq_name)` if `liq_qty < notional_qty`, else `(notional_qty, "notional")` (a tie is reported as `notional`). The caller skips on `qty < 1`.

- [ ] **Step 1: Write the failing tests**

```python
TS = T_DEV
def q(age_min: int, ask_size: int) -> Quote:
    return make_quote(ts=TS - timedelta(minutes=age_min), symbol="AAA", ask_size=ask_size)
def size(**kw):
    base = dict(ref_price=1.02, available_cash=25_000.0, signal_ts=TS, bar_volume=10_000, quote=None, cfg=RiskConfig())
    base.update(kw); return size_order(**base)

def test_notional_binds() -> None:
    qty, cap = size(bar_volume=1_000_000)
    assert (qty, cap) == (1960, "notional") and qty * 1.02 <= 2_000.0

def test_bar_cap_binds_without_quote() -> None:
    assert size() == (200, "bar")

def test_fresh_quote_cap_binds() -> None:
    assert size(bar_volume=1_000_000, quote=q(1, 150)) == (150, "quote")

def test_quote_age_boundary() -> None:
    assert size(quote=q(5, 150)) == (150, "quote")   # exactly max age: fresh
    assert size(quote=q(6, 150)) == (200, "bar")     # stale: bar fallback

def test_zero_liquidity_gives_zero() -> None:
    assert size(quote=q(0, 0))[0] == 0
    assert size(bar_volume=0)[0] == 0

def test_liquidity_cap_floor_is_robust_to_float_error() -> None:
    assert size(bar_volume=1_000_000, quote=q(1, 100), cfg=RiskConfig(quote_size_frac=0.29)) == (29, "quote")

def test_size_uses_available_cash() -> None:
    assert size(bar_volume=1_000_000, available_cash=1_000.0) == (980, "notional")
    assert size(bar_volume=1_000_000, available_cash=-5.0)[0] == 0
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_intraday_risk_sizing.py -v` → FAIL.
- [ ] **Step 3: Implement `sizing.py`** per the Interfaces block.
- [ ] **Step 4: Run to verify they pass** — all PASS.
- [ ] **Step 5: Full suite and commit** — `feat(intraday-risk): sizing under notional, cash and quote/bar liquidity caps` + trailers.

---

### Task 4: Books

**Files:**
- Create: `research/intraday/risk/books.py`
- Test: `tests/test_intraday_risk_books.py`

**Interfaces:**
- Consumes: `Fill` (contracts).
- Produces `class Books`:
  - `__init__(self, start_equity_usd: float) -> None`; public attributes `cash: float` (starts at `start_equity_usd`) and `realised_pnl: float` (starts `0.0`).
  - `open_intent(self, intent_id: str, symbol: str, qty: int, ref_price: float) -> None`: registers a working intent (remaining = `qty`) and the id→symbol map; a repeated id raises `ValueError`.
  - `apply_fill(self, fill: Fill) -> None`:
    - unknown `intent_id` → `ValueError("unknown intent_id ...")`; `fill.symbol` ≠ the intent's symbol → `ValueError`;
    - `buy`: the intent must be working with `remaining >= fill.qty` (else `ValueError`, "overfill" or "not working"); `cash -= price*qty*(1 + cost_bps/10_000)`; the position's `qty += fill.qty`, `cost_basis +=` the same cash amount; `remaining -= qty`, and the intent leaves the working set at 0;
    - `sell`: `fill.qty > position qty` → `ValueError`; `proceeds = price*qty*(1 - cost_bps/10_000)`; `cash += proceeds`; `released = cost_basis * qty / position_qty`; `realised_pnl += proceeds - released`; `cost_basis -= released`; a position at 0 is removed.
  - `close_working(self, intent_id: str) -> None`: unknown id → `ValueError`; a working intent is dropped (its filled part stays a position); an id that is known but no longer working is a no-op (e.g. an exit `expired` at session end).
  - `require_known(self, intent_id: str) -> None`: `ValueError` if never opened.
  - `position_qty(self, symbol: str) -> int`; `is_busy(self, symbol: str) -> bool` (position > 0 or a working intent); `occupied(self) -> int` (number of distinct busy symbols, so a partly filled, still-working intent counts once); `reserved_cash(self) -> float` (Σ remaining × ref_price over working intents); `marked_equity(self, closes: Mapping[str, float]) -> float` (`cash + Σ qty × closes[symbol]`; a position without a close raises `ValueError`); `positions(self) -> dict[str, tuple[int, float]]` (symbol → (qty, cost_basis), keys sorted).

Cost basis includes the buy leg's cost, so `realised_pnl` is net of both legs' `cost_bps`.

- [ ] **Step 1: Write the failing tests**

```python
def fill(side: str, qty: int, price: float, bps: float, iid: str = "i1", sym: str = "AAA") -> Fill:
    return make_fill(ts=T_DEV, intent_id=iid, symbol=sym, side=side, qty=qty, price=price, cost_bps=bps)

def test_partial_buys_then_sell_match_cash_formula() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02)
    b.apply_fill(fill("buy", 120, 1.03, 10.0)); b.apply_fill(fill("buy", 80, 1.04, 10.0))
    paid = 120 * 1.03 * 1.001 + 80 * 1.04 * 1.001
    b.apply_fill(fill("sell", 150, 1.10, 20.0))
    got = 150 * 1.10 * 0.998
    assert b.cash == pytest.approx(25_000.0 - paid + got, rel=1e-12)
    assert b.realised_pnl == pytest.approx(got - paid * 150 / 200, rel=1e-12)
    assert b.positions() == {"AAA": (50, pytest.approx(paid * 50 / 200, rel=1e-12))}
    assert not b.reserved_cash() and b.occupied() == 1

def test_unknown_or_mismatched_fill_raises() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02)
    with pytest.raises(ValueError, match="unknown intent_id"):
        b.apply_fill(fill("buy", 1, 1.0, 0.0, iid="nope"))
    with pytest.raises(ValueError):
        b.apply_fill(fill("buy", 1, 1.0, 0.0, sym="BBB"))

def test_overfill_and_fill_after_cancel_raise() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02)
    with pytest.raises(ValueError):
        b.apply_fill(fill("buy", 201, 1.0, 0.0))
    b.apply_fill(fill("buy", 120, 1.0, 0.0)); b.close_working("i1")
    with pytest.raises(ValueError):
        b.apply_fill(fill("buy", 1, 1.0, 0.0))

def test_oversell_raises() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02); b.apply_fill(fill("buy", 50, 1.0, 0.0))
    with pytest.raises(ValueError):
        b.apply_fill(fill("sell", 51, 1.0, 0.0))

def test_partial_then_close_working_keeps_position() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 200, 1.02)
    b.apply_fill(fill("buy", 120, 1.0, 0.0))
    assert b.occupied() == 1 and b.reserved_cash() == pytest.approx(80 * 1.02)
    b.close_working("i1")
    assert b.position_qty("AAA") == 120 and b.is_busy("AAA") and b.reserved_cash() == 0.0
    b.close_working("i1")   # known, no longer working: no-op
    with pytest.raises(ValueError):
        b.close_working("nope")

def test_marked_equity() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 100, 1.0); b.apply_fill(fill("buy", 100, 1.0, 0.0))
    assert b.marked_equity({"AAA": 0.8}) == pytest.approx(25_000.0 - 20.0)
    with pytest.raises(ValueError):
        b.marked_equity({})

def test_duplicate_open_intent_raises() -> None:
    b = Books(25_000.0); b.open_intent("i1", "AAA", 100, 1.0)
    with pytest.raises(ValueError):
        b.open_intent("i1", "BBB", 100, 1.0)
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_intraday_risk_books.py -v` → FAIL.
- [ ] **Step 3: Implement `Books`** per the Interfaces block (private `_Position`/`_Working` mutable slot dataclasses; dicts keyed by id/symbol).
- [ ] **Step 4: Run to verify they pass** — all PASS.
- [ ] **Step 5: Full suite and commit** — `feat(intraday-risk): position, working-intent and cash books kept from fills` + trailers.

---

### Task 5: RiskAgent entry path (rules, sizing, brackets, id, reason, time stop)

**Files:**
- Create: `research/intraday/risk/agent.py`
- Test: `tests/test_intraday_risk.py` (append)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces `class RiskAgent`:
  - `__init__(self, session_date: date, config: RiskConfig | None = None) -> None`; `self.name = "risk"`; `self.subscribes = (Bar, Quote, Signal, Fill, OrderUpdate, Halt, Kill)`; computes `session_bounds` once.
  - `on_event(self, event: Event) -> tuple[Event, ...]`: dispatch through a `dict[type[Event], handler]` built in `__init__`; only `Signal` can return a non-empty tuple (one `OrderIntent`).
  - `skip_counts -> dict[str, int]` (property; keys sorted, zero counts omitted); `books -> Books` (property); `killed -> bool`, `loss_limit_tripped -> bool` (properties).
- State: `_last_bar: dict[str, Bar]`, `_trs: dict[str, deque[float]]` (`maxlen=cfg.atr_period`), `_quotes: dict[str, Quote]`, `_entries: dict[str, int]`, `_halted: set[str]`, `_n: int` (intent counter), `_books`, `_killed`, `_tripped`, `_skips: Counter[str]`.
- `Bar` handler: `et_date(bar.ts) != session_date` → `ValueError`; `bar.ts <=` the symbol's last `Bar.ts` → `ValueError`; append `true_range(bar, prev.close if prev else None)`; store the bar; then the loss-limit check (Task 7 fills it in; here a no-op method `_check_loss_limit()`).
- `Quote` handler: store as the symbol's latest.
- `Signal` handler, in this order (first failure: `self._skips[reason] += 1`, return `()`):
  0. `et_date(signal.ts) != session_date` → raise `ValueError` (not a skip);
  1. `killed` → `"killed"`; `loss_limit_tripped` → `"loss_limit"`;
  2. symbol in `_halted` → `"halted"`;
  3. `signal.ts >= last_entry_utc` → `"after_last_entry"`;
  4. `signal.score < entry_threshold` → `"below_threshold"`;
  5. `books.is_busy(symbol)` → `"symbol_busy"`; `_entries[symbol] >= max_entries_per_symbol` → `"symbol_entries"`;
  6. `books.occupied() >= max_open_positions` → `"max_open"`;
  7. no `_last_bar[symbol]` → `"no_bar"`; `ref_price = last_bar.close`;
  8. `size_order(ref_price=ref, available_cash=books.cash - books.reserved_cash(), signal_ts=signal.ts, bar_volume=last_bar.volume, quote=_quotes.get(symbol), cfg=cfg)`; `qty < 1` → `"zero_qty"`; `make_bracket(ref, mean_true_range(_trs[symbol], atr_period), cfg)` returning a `str` → that reason.
  Then: `_n += 1`; `intent_id = f"{session_date:%Y%m%d}-{symbol}-{_n}"`; `hold = signal.horizon_min if cfg.hold_min is None else cfg.hold_min`; `time_stop = min(signal.ts + timedelta(minutes=hold), flatten_utc)` (always `> signal.ts`, since `signal.ts < last_entry_utc < flatten_utc` and `hold >= 1`); `reason = f"entry score={signal.score:.6g} thr={cfg.entry_threshold:.6g} cap={cap}"`; `books.open_intent(intent_id, symbol, qty, ref)`; `_entries[symbol] += 1`; return `(OrderIntent(ts=signal.ts, intent_id=..., symbol=..., side="buy", qty=qty, limit_price=None, take_profit=b.take_profit, stop_loss=b.stop_loss, time_stop=time_stop, reason=reason),)`.
  An entry counts toward `max_entries_per_symbol` when its intent is emitted, whether or not it fills.
  `available_cash = cash - reserved_cash()` is risk.md's `cash` read as uncommitted cash, so concurrent working intents cannot spend the same dollars; the reservation is at `ref_price` without the leg cost (a fill can overshoot by `cost_bps`, a few dollars on 2 000; accepted). When cash binds, `cap` reads `notional`.
- `Fill`, `OrderUpdate`, `Halt`, `Kill` handlers: stubs returning `()` in this task; Tasks 6–7 fill them.

- [ ] **Step 1: Write the failing tests** (append; module-level helpers first)

```python
D = date(2024, 3, 4)
def et(h: int, m: int, d: date = D) -> datetime:
    return datetime.combine(d, time(h, m), ET).astimezone(timezone.utc)
assert et(9, 31) == T_DEV
def bar(sym: str, ts: datetime, close: float = 1.02, volume: int = 10_000,
        high: float | None = None, low: float | None = None, open_: float | None = None) -> Bar:
    o = close if open_ is None else open_
    return make_bar(symbol=sym, ts=ts, bar_start=ts - timedelta(minutes=1), open=o,
                    high=max(o, close) if high is None else high, low=min(o, close) if low is None else low,
                    close=close, vwap=close, volume=volume)
def sig(sym: str, ts: datetime, score: float = 0.7, horizon: int = 15) -> Signal:
    return make_signal(symbol=sym, ts=ts, score=score, horizon_min=horizon)
def run(agent: RiskAgent, *events: Event) -> list[Event]:
    return [e for ev in events for e in agent.on_event(ev)]

def test_only_buy_intents_with_signal_ts() -> None:
    out = run(RiskAgent(D), bar("AAA", T_DEV), sig("AAA", T_DEV))
    assert len(out) == 1 and isinstance(out[0], OrderIntent)
    oi = out[0]
    assert (oi.ts, oi.side, oi.qty, oi.limit_price, oi.take_profit, oi.stop_loss) == (T_DEV, "buy", 200, None, None, None)
    assert oi.time_stop == et(9, 46)
    assert oi.reason == "entry score=0.7 thr=0.5 cap=bar"

def test_intent_ids_unique_and_deterministic() -> None:
    evs = (bar("AAA", T_DEV), bar("BBB", T_DEV), sig("AAA", T_DEV), sig("BBB", T_DEV))
    a, b = run(RiskAgent(D), *evs), run(RiskAgent(D), *evs)
    assert [o.intent_id for o in a] == ["20240304-AAA-1", "20240304-BBB-2"]
    assert [to_record(o) for o in a] == [to_record(o) for o in b]

def test_below_threshold_signal_skipped() -> None:
    agent = RiskAgent(D)
    assert run(agent, bar("AAA", T_DEV), sig("AAA", T_DEV, score=0.49)) == []
    assert agent.skip_counts == {"below_threshold": 1}
    assert len(run(agent, sig("AAA", T_DEV, score=0.5))) == 1     # >= threshold enters

def test_time_stop_capped_at_flatten_time() -> None:
    cfg = RiskConfig(last_entry_et=time(15, 50), flatten_et=time(15, 55))
    (oi,) = run(RiskAgent(D, cfg), bar("AAA", et(15, 45)), sig("AAA", et(15, 45), horizon=15))
    assert oi.time_stop == et(15, 55)
    (oi,) = run(RiskAgent(D, RiskConfig(hold_min=5)), bar("AAA", T_DEV), sig("AAA", T_DEV, horizon=15))
    assert oi.time_stop == et(9, 36)

def test_no_entry_at_or_after_last_entry_time() -> None:
    agent = RiskAgent(D)
    out = run(agent, bar("AAA", et(15, 29)), sig("AAA", et(15, 29)),
              bar("BBB", et(15, 30)), sig("BBB", et(15, 30)))
    assert [o.symbol for o in out] == ["AAA"] and out[0].time_stop == et(15, 44)
    assert agent.skip_counts == {"after_last_entry": 1}

def test_last_entry_boundary_in_summer_session() -> None:
    s = date(2024, 7, 1)
    agent = RiskAgent(s)
    out = run(agent, bar("AAA", et(15, 29, s)), sig("AAA", et(15, 29, s)),
              bar("BBB", et(15, 30, s)), sig("BBB", et(15, 30, s)))
    assert [o.symbol for o in out] == ["AAA"]
    assert et(15, 30, s) == datetime(2024, 7, 1, 19, 30, tzinfo=timezone.utc)

def test_signal_without_bar_skipped() -> None:
    agent = RiskAgent(D)
    assert run(agent, sig("AAA", T_DEV)) == [] and agent.skip_counts == {"no_bar": 1}

def test_signal_from_other_session_raises() -> None:
    with pytest.raises(ValueError):
        run(RiskAgent(D), sig("AAA", T_DEV + timedelta(days=1)))

def test_out_of_order_bar_raises() -> None:
    agent = RiskAgent(D); run(agent, bar("AAA", et(9, 32)))
    with pytest.raises(ValueError):
        run(agent, bar("AAA", et(9, 32)))
    run(agent, bar("BBB", et(9, 31)))   # another symbol's clock is independent

def test_size_capped_by_notional() -> None:
    (oi,) = run(RiskAgent(D), bar("AAA", T_DEV, volume=1_000_000), sig("AAA", T_DEV))
    assert oi.qty == 1960 and oi.qty * 1.02 <= 2_000.0 and oi.reason.endswith("cap=notional")

def test_size_capped_by_shares_at_best_ask() -> None:
    quote = make_quote(ts=et(9, 30), symbol="AAA", ask_size=150)
    (oi,) = run(RiskAgent(D), quote, bar("AAA", T_DEV, volume=1_000_000), sig("AAA", T_DEV))
    assert oi.qty == 150 and oi.reason.endswith("cap=quote")

def test_stale_quote_falls_back_to_bar_volume_cap() -> None:
    stale = make_quote(ts=et(9, 25), symbol="AAA", ask_size=150)
    (oi,) = run(RiskAgent(D), stale, bar("AAA", T_DEV), sig("AAA", T_DEV))
    assert oi.qty == 200 and oi.reason.endswith("cap=bar")

def test_zero_qty_after_cap_skipped() -> None:
    agent = RiskAgent(D)
    out = run(agent, make_quote(ts=et(9, 30), symbol="AAA", ask_size=0), bar("AAA", T_DEV), sig("AAA", T_DEV),
              bar("BBB", T_DEV, volume=0), sig("BBB", T_DEV))
    assert out == [] and agent.skip_counts == {"zero_qty": 2}

def test_fixed_pct_bracket_levels() -> None:
    cfg = RiskConfig(exit_mode="fixed_pct", tp_pct=0.05, sl_pct=0.03)
    (oi,) = run(RiskAgent(D, cfg), bar("AAA", T_DEV), sig("AAA", T_DEV))
    assert oi.take_profit == pytest.approx(1.071) and oi.stop_loss == pytest.approx(0.9894)
    assert oi.time_stop == et(9, 46)

def test_atr_bracket_needs_atr_period_bars() -> None:
    cfg = RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0, atr_period=3)
    agent = RiskAgent(D, cfg)
    b1 = bar("AAA", et(9, 31), close=1.00, high=1.02, low=0.98)
    b2 = bar("AAA", et(9, 32), close=1.01, high=1.03, low=1.00)
    b3 = bar("AAA", et(9, 33), close=1.05, high=1.06, low=1.02)
    assert run(agent, b1, b2, sig("AAA", et(9, 32))) == []
    assert agent.skip_counts == {"atr_warmup": 1}
    (oi,) = run(agent, b3, sig("AAA", et(9, 33)))          # ATR = (0.04 + 0.03 + 0.05) / 3
    assert oi.take_profit == pytest.approx(1.13) and oi.stop_loss == pytest.approx(1.01)

def test_flat_bars_atr_bracket_skipped_not_raised() -> None:
    cfg = RiskConfig(exit_mode="atr", tp_atr=2.0, sl_atr=1.0, atr_period=2)
    agent = RiskAgent(D, cfg)
    out = run(agent, bar("AAA", et(9, 31)), bar("AAA", et(9, 32)), sig("AAA", et(9, 32)))
    assert out == [] and agent.skip_counts == {"bad_bracket": 1}
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_intraday_risk.py -v` → the new tests FAIL (`ImportError: RiskAgent`).
- [ ] **Step 3: Implement `RiskAgent`** in `agent.py` per the Interfaces block. One private method per rule group keeps `_on_signal` short; it returns the first `SkipReason` or `None`.
- [ ] **Step 4: Run to verify they pass** — all PASS.
- [ ] **Step 5: Full suite and commit** — `feat(intraday-risk): RiskAgent entry path -- rules, caps, exit-grid brackets, ids, capped time stop` + trailers.

---

### Task 6: Fills, order updates, slots

**Files:**
- Modify: `research/intraday/risk/agent.py` (`Fill`, `OrderUpdate` handlers)
- Test: `tests/test_intraday_risk.py` (append)

**Interfaces:**
- Consumes: `Books.apply_fill`, `Books.close_working`, `Books.require_known` (Task 4).
- Produces: `Fill` handler → `books.apply_fill(fill)` then `_check_loss_limit()`, returns `()`. `OrderUpdate` handler → `books.require_known(intent_id)`; status `rejected`/`cancelled`/`expired` → `books.close_working(intent_id)`; `accepted` → nothing; returns `()`.

- [ ] **Step 1: Write the failing tests** (append; `IID = "20240304-AAA-1"`)

```python
def buy(qty: int, price: float = 1.02, ts: datetime = et(9, 32), iid: str = "20240304-AAA-1", sym: str = "AAA", bps: float = 0.0) -> Fill:
    return make_fill(ts=ts, intent_id=iid, symbol=sym, side="buy", qty=qty, price=price, cost_bps=bps)
def sell(qty: int, price: float = 1.03, ts: datetime = et(9, 33), iid: str = "20240304-AAA-1", sym: str = "AAA", bps: float = 0.0) -> Fill:
    return make_fill(ts=ts, intent_id=iid, symbol=sym, side="sell", qty=qty, price=price, cost_bps=bps)
def upd(status: str, iid: str = "20240304-AAA-1", ts: datetime = et(9, 32)) -> OrderUpdate:
    return make_order_update(ts=ts, intent_id=iid, status=status, detail="")

def test_one_entry_per_symbol_per_session() -> None:
    agent = RiskAgent(D)
    run(agent, bar("AAA", T_DEV), sig("AAA", T_DEV), buy(200))
    assert run(agent, sig("AAA", et(9, 32))) == []
    run(agent, sell(200))
    assert run(agent, sig("AAA", et(9, 33))) == []
    assert agent.skip_counts == {"symbol_busy": 1, "symbol_entries": 1}

def test_max_open_positions_respected() -> None:
    agent = RiskAgent(D, RiskConfig(max_open_positions=2))
    out = run(agent, *(e for s in ("AAA", "BBB", "CCC") for e in (bar(s, T_DEV), sig(s, T_DEV))))
    assert [o.symbol for o in out] == ["AAA", "BBB"] and agent.skip_counts == {"max_open": 1}

@pytest.mark.parametrize("status", ["rejected", "expired", "cancelled"])
def test_rejected_or_expired_update_frees_slot(status: str) -> None:
    agent = RiskAgent(D, RiskConfig(max_open_positions=1))
    run(agent, bar("AAA", T_DEV), sig("AAA", T_DEV), bar("BBB", T_DEV))
    assert run(agent, sig("BBB", T_DEV)) == []
    run(agent, upd(status), bar("BBB", et(9, 32)))
    (oi,) = run(agent, sig("BBB", et(9, 32)))
    assert oi.symbol == "BBB"
    assert run(agent, bar("AAA", et(9, 32)), sig("AAA", et(9, 32))) == []   # AAA's one entry is spent

def test_unknown_fill_raises() -> None:
    with pytest.raises(ValueError, match="unknown intent_id"):
        run(RiskAgent(D), buy(1, iid="20240304-ZZZ-9", sym="ZZZ"))

def test_unknown_order_update_raises() -> None:
    with pytest.raises(ValueError):
        run(RiskAgent(D), upd("rejected", iid="20240304-ZZZ-9"))

def test_books_match_fills() -> None:
    agent = RiskAgent(D)
    run(agent, bar("AAA", T_DEV), sig("AAA", T_DEV), buy(120, 1.03, bps=10.0), buy(80, 1.04, bps=10.0),
        sell(150, 1.10, bps=20.0))
    paid = 120 * 1.03 * 1.001 + 80 * 1.04 * 1.001
    got = 150 * 1.10 * 0.998
    assert agent.books.cash == pytest.approx(25_000.0 - paid + got, rel=1e-12)
    assert agent.books.realised_pnl == pytest.approx(got - paid * 0.75, rel=1e-12)
    assert agent.books.position_qty("AAA") == 50
    with pytest.raises(ValueError):
        run(agent, sell(51))

def test_partial_fill_then_cancel_keeps_filled_position() -> None:
    agent = RiskAgent(D, RiskConfig(max_open_positions=2))
    run(agent, bar("AAA", T_DEV), sig("AAA", T_DEV))
    run(agent, buy(120), upd("cancelled"))            # execution.md order: Fill, then cancel, one cascade
    assert agent.books.position_qty("AAA") == 120 and agent.books.reserved_cash() == 0.0
    out = run(agent, bar("AAA", et(9, 32)), sig("AAA", et(9, 32)),
              bar("BBB", et(9, 32)), sig("BBB", et(9, 32)), bar("CCC", et(9, 32)), sig("CCC", et(9, 32)))
    assert [o.symbol for o in out] == ["BBB"]
    assert agent.skip_counts == {"symbol_busy": 1, "max_open": 1}
    run(agent, sell(120))
    assert agent.books.position_qty("AAA") == 0

def test_working_intents_reserve_cash() -> None:
    agent = RiskAgent(D, RiskConfig(start_equity_usd=3_000.0))
    out = run(agent, bar("AAA", T_DEV, close=1.0, volume=1_000_000), sig("AAA", T_DEV),
              bar("BBB", T_DEV, close=1.0, volume=1_000_000), sig("BBB", T_DEV))
    assert [o.qty for o in out] == [2000, 1000]
```

- [ ] **Step 2: Run to verify the new tests fail** — `$PY -m pytest tests/test_intraday_risk.py -v` → FAIL (stub handlers ignore fills/updates).
- [ ] **Step 3: Implement the `Fill` and `OrderUpdate` handlers** per the Interfaces block.
- [ ] **Step 4: Run to verify all pass.**
- [ ] **Step 5: Full suite and commit** — `feat(intraday-risk): books from fills and order updates; partial fill keeps its position` + trailers.

---

### Task 7: Halt, Kill, daily loss limit

**Files:**
- Modify: `research/intraday/risk/agent.py` (`Halt`, `Kill` handlers, `_check_loss_limit`)
- Test: `tests/test_intraday_risk.py` (append)

**Interfaces:**
- Consumes: `Books.marked_equity` (Task 4); `_last_bar` closes (Task 5).
- Produces: `Halt` → `halted=True` adds the symbol to `_halted`, `False` discards it. `Kill` → `_killed = True` (permanent for the agent's session). `_check_loss_limit()` (after every `Bar` and `Fill`): once `books.marked_equity({s: b.close for s, b in _last_bar.items()}) <= start_equity_usd * (1 - daily_loss_limit_pct)`, set `_tripped = True` for the session. Neither emits; the loss limit stops entries only (positions keep their brackets; only the MonitorAgent emits `Kill`).

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_halted_symbol_blocked_until_resume() -> None:
    agent = RiskAgent(D)
    h = make_halt(ts=T_DEV, symbol="AAA", halted=True, reason="LULD")
    assert run(agent, bar("AAA", T_DEV), h, sig("AAA", T_DEV)) == []
    r = make_halt(ts=et(9, 32), symbol="AAA", halted=False, reason="resume")
    (oi,) = run(agent, r, bar("AAA", et(9, 32)), sig("AAA", et(9, 32)))
    assert oi.symbol == "AAA" and agent.skip_counts == {"halted": 1}

def test_no_intent_after_kill() -> None:
    agent = RiskAgent(D)
    out = run(agent, bar("AAA", T_DEV), bar("BBB", T_DEV), make_kill(ts=T_DEV, reason="stale"),
              sig("AAA", T_DEV), sig("BBB", T_DEV))
    assert out == [] and agent.killed and agent.skip_counts == {"killed": 2}

def test_loss_limit_stops_new_entries() -> None:
    agent = RiskAgent(D)          # limit: marked equity <= 24_500
    (oi,) = run(agent, bar("AAA", T_DEV, volume=1_000_000), sig("AAA", T_DEV))
    run(agent, buy(1960, 1.02, ts=et(9, 32)), bar("AAA", et(9, 32), close=0.80))   # equity 24_568.8
    assert not agent.loss_limit_tripped
    assert len(run(agent, bar("BBB", et(9, 32)), sig("BBB", et(9, 32)))) == 1
    run(agent, bar("AAA", et(9, 33), close=0.70))                                  # equity 24_372.8
    assert agent.loss_limit_tripped
    run(agent, bar("AAA", et(9, 34), close=1.20))                                  # recovery does not reset
    assert run(agent, bar("CCC", et(9, 34)), sig("CCC", et(9, 34))) == []
    assert agent.loss_limit_tripped and agent.skip_counts == {"loss_limit": 1}

def test_loss_limit_checked_after_fill() -> None:
    agent = RiskAgent(D)
    run(agent, bar("AAA", T_DEV, volume=1_000_000), sig("AAA", T_DEV), buy(1960, 1.02, ts=et(9, 32)))
    run(agent, sell(1960, 0.70, ts=et(9, 33)))     # realised loss 627.2, no bar needed
    assert agent.loss_limit_tripped
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_intraday_risk.py -v` → FAIL.
- [ ] **Step 3: Implement the handlers and `_check_loss_limit`.**
- [ ] **Step 4: Run to verify all pass.**
- [ ] **Step 5: Full suite and commit** — `feat(intraday-risk): halts, kill and the session loss limit block entries` + trailers.

---

### Task 8: `adv_shares` and package exports

**Files:**
- Create: `research/intraday/risk/adv.py`
- Modify: `research/intraday/risk/__init__.py`
- Test: `tests/test_intraday_risk.py` (append)

**Interfaces:**
- Produces: `adv_shares(daily: pd.DataFrame, session_date: date, window: int = 20) -> float`. `daily` is one symbol's daily cache rows with columns `date`, `close` (adjusted), `close_unadj`, `volume` (adjusted share volume). Rows with `date < session_date` only; the last `window` of them sorted by date; `f_prev = close_unadj / close` of the last of them; returns `float((vol / f_prev).mean())`. Raises `ValueError` if fewer than `window` prior rows, if `f_prev` is not finite and `> 0`, or if a `symbol` column holds more than one value. Vectorised; no loop.
- `__init__.py` re-exports `RiskAgent`, `RiskConfig`, `adv_shares` with `__all__`.

Why one factor (risk.md): dividing each prior day by its own factor puts a split inside the window on two share bases; dividing all by `f_{D-1}` puts the window on `D`'s traded basis. `D`'s own row never enters.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_adv_shares_uses_prior_sessions_on_one_basis() -> None:
    # 2:1 forward split inside the window (raw 1000 sh @ $2 -> 2000 sh @ $1), and a 1:10 reverse
    # split after D that the fetch-date-anchored adjustment bakes into every row (adjusted close
    # 10.0, adjusted volume 200 throughout). On D's traded basis the ADV is 2000 shares.
    days = list(pd.bdate_range(end=pd.Timestamp(D) - pd.Timedelta(days=1), periods=25))
    rows = [dict(date=d, close=10.0, close_unadj=2.0 if i < 15 else 1.0, volume=200) for i, d in enumerate(days)]
    rows += [dict(date=pd.Timestamp(D), close=10.0, close_unadj=1.0, volume=999_999),
             dict(date=pd.Timestamp(D) + pd.Timedelta(days=1), close=10.0, close_unadj=1.0, volume=999_999)]
    daily = pd.DataFrame(rows)
    assert adv_shares(daily, D) == pytest.approx(2000.0)
    assert adv_shares(daily.sample(frac=1.0, random_state=0), D) == pytest.approx(2000.0)   # order-free
    with pytest.raises(ValueError):
        adv_shares(daily.iloc[-8:], D)                      # fewer than 20 prior sessions
```

- [ ] **Step 2: Run to verify it fails** — `$PY -m pytest tests/test_intraday_risk.py::test_adv_shares_uses_prior_sessions_on_one_basis -v` → FAIL.
- [ ] **Step 3: Implement `adv_shares`; fill `__init__.py`.** Compare dates with `pd.to_datetime(daily["date"]) < pd.Timestamp(session_date)`.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Full suite and commit** — `feat(intraday-risk): adv_shares on the session's traded basis; package exports` + trailers.

---

### Task 9: On the bus, determinism, ownership

**Files:**
- Test: `tests/test_intraday_risk_bus.py`

**Interfaces:**
- Consumes: `EventBus(agents, sink=None, max_cascade=10_000)` and `EventBus.publish(event) -> list[Envelope]` (`research/intraday/engine/bus.py`); `to_record`, `canonical_json` (`research/intraday/contracts/codec.py`); `RiskAgent` from `research.intraday.risk`. Tests may import the engine; `risk/` code may not.

- [ ] **Step 1: Write the tests**

```python
D = date(2024, 3, 4)
def _bar(sym: str) -> Bar:
    return make_bar(symbol=sym, ts=T_DEV, bar_start=T_DEV - timedelta(minutes=1), open=1.02,
                    high=1.02, low=1.02, close=1.02, vwap=1.02, volume=1_000_000)
EVENTS: list[Event] = [
    make_quote(ts=T_DEV - timedelta(minutes=1), symbol="AAA", ask_size=150),
    _bar("AAA"), _bar("BBB"), _bar("CCC"),
    make_signal(ts=T_DEV, symbol="AAA", score=0.7), make_signal(ts=T_DEV, symbol="BBB", score=0.7),
    make_signal(ts=T_DEV, symbol="CCC", score=0.1),            # below threshold: no intent
]

def _log(events: list[Event]) -> list[str]:
    bus = EventBus([RiskAgent(D)])
    return [canonical_json(dict(to_record(env.event), source=env.source, seq=env.seq, cause=env.cause_seq))
            for ev in events for env in bus.publish(ev)]

def test_on_bus_risk_emits_only_order_intents_with_cause_ts() -> None:
    bus = EventBus([RiskAgent(D)])
    envs = [env for ev in EVENTS for env in bus.publish(ev)]
    mine = [e for e in envs if e.source == "risk"]
    assert [(type(e.event), e.event.symbol, e.event.qty) for e in mine] == [(OrderIntent, "AAA", 150), (OrderIntent, "BBB", 1960)]
    by_seq = {e.seq: e for e in envs}
    assert all(e.event.ts == by_seq[e.cause_seq].event.ts for e in mine)

def test_two_runs_give_identical_logs() -> None:
    assert _log(EVENTS) == _log(EVENTS)
```


- [ ] **Step 2: Run** — `$PY -m pytest tests/test_intraday_risk_bus.py -v` → PASS. This task adds no production code; it verifies the finished agent on the real bus, so a FAIL is a bug in `risk/` (fix the code, never the test).
- [ ] **Step 3: Guards and ownership**

Run: `$PY -m pytest tests/test_intraday_guards.py tests/test_intraday_design_docs.py -v` → PASS (no wall clock, no sibling imports, no trading imports).
Run: `$PY scripts/check_ownership.py --agent intraday-risk --base research/intraday` → prints nothing, exit 0.

- [ ] **Step 4: Full suite** — `$PY -m pytest tests -q` → all pass.
- [ ] **Step 5: Commit** — `test(intraday-risk): RiskAgent on the bus, deterministic log` + trailers.
