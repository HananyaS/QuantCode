# Intraday Phase 2 — Data (ScreenerAgent, ReplayFeed, FeedAgent) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `intraday-data` slice of the replay stack: a ScreenerAgent that emits one causal `Candidates` per session at 09:25 ET, and a FeedAgent + ReplayFeed that turn the cached (adjusted) SIP minute bars into traded-basis `Bar` / `Gap` / `Quote` events released on the clock.

**Architecture:** Two pure agents (`on_event(event) -> tuple[Event, ...]`) with injected read-only adapters. All I/O lives in adapters (`ReplayScreenInputs`, `ReplayFeed`); all corporate-action arithmetic lives in one pure module (`basis.py`); the agents only filter, rank, look up and emit. Nothing here imports another agent package; the contracts are the only shared code.

**Tech Stack:** Python 3.13, pandas + pyarrow (parquet with `filters=`), stdlib `zoneinfo` (`tzdata` installed), pytest. Reuse from `research/penny/` (read-only).

**Spec:** `research/intraday/docs/design/screener.md` and `research/intraday/docs/design/feed.md` (binding), with `engine.md`, `TEMPLATE.md`, the contracts (`research/intraday/contracts/`), and `docs/superpowers/specs/2026-10-26-intraday-system-design.md` §2, §4, §5 phase 2.

## Global Constraints

- Worktree for execution: a new worktree on branch `research/intraday-data`, cut from `research/intraday`. Never push, never merge to main.
- Write **only** `research/intraday/data/` and `tests/test_intraday_data*`. Before every commit: `$PY scripts/check_ownership.py --agent intraday-data --base research/intraday` prints nothing.
- `PY=C:/Users/Shachar/AppData/Local/Programs/Python/Python313/python.exe` (not on PATH). All commands run from the worktree root.
- The full suite passes before every commit: `$PY -m pytest tests/ -q`. Tests are never edited to make them pass.
- Data is read from the **main checkout** (gitignored), never copied into the worktree: `DEFAULT_DATA_ROOT = C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_penny` (`minute/dt=YYYY-MM-DD/{minute.parquet,requested.json}`, `premarket/dt=YYYY-MM-DD/premarket.parquet`, `daily/bars_YYYY.parquet`, `quote_summary_r4.parquet`, `quote_summary_rev.parquet`) and `DEFAULT_VOLUME_DAILY_ROOT = C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_breakout/daily` (`bars_YYYY.parquet`, which carries `volume_unadj`). **No downloads** (user decision 2026-10-26): nothing imports `alpaca` at all.
- Do **not** call `research.penny.etl.bars.load_years`: its `DATA_ROOT = Path("data/research_penny")` is cwd-relative and resolves to nothing in a worktree. Read the same `bars_<year>.parquet` layout through `data_root` (Task 3).
- Holdout: `HOLDOUT_START = date(2025, 8, 7)`. Every adapter method that takes a session date raises `SealedHoldoutError` for `session_date >= HOLDOUT_START` unless `unseal=True`. Tests use `fixtures.T_DEV` (2024-03-04 14:31 UTC = 09:31 EST), never `T0`.
- Contracts are strict: int fields get `int(...)` (never `np.int64`), float fields get `float` (`0.0`, never `0`), tuples not lists, datetimes are `datetime` with `tzinfo is timezone.utc` (never `pd.Timestamp`; `et_to_utc` in Task 1 is the one constructor). `Bar.ts == bar_start + 1 min`; `Gap.ts == missing_bar_start + 1 min`.
- Never emit while iterating a `set`/`frozenset`/dict built from one: iterate the sorted `Candidates.symbols` tuple. No wall clock (`datetime.now`, `pd.Timestamp('now')`, `time.time`, …), no unseeded randomness, no `alpaca.trading` — `tests/test_intraday_guards.py` enforces all three on `research/intraday/`.
- Type hints on every function; small functions; vectorised pandas for frame arithmetic (row loops only where a contract object is built per row, ≤ ~8k rows per session).
- Tests on real data skip cleanly when `DEFAULT_DATA_ROOT/minute` or `DEFAULT_VOLUME_DAILY_ROOT` is absent. Unit tests build tiny synthetic frames (Task 3's `synthetic.py`) under `tmp_path`.
- Commit messages end with two trailer lines: `Refs: HananyaS/QuantCode-intraday-board#20` (screener) and/or `#21` (feed), as each task states, then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Design decisions this plan makes (the design docs are silent or wrong)

The controller should confirm D1 with `intraday-architect` (it amends `feed.md`/`screener.md` and the R16 draft §4.4, which repeats the rule). The plan is built on D1 because the documented rule is measurably wrong.

- **D1 — share volume is un-adjusted with the split-only factor, not with `f`.** Both docs say "share volume ÷ `f`" with `f = close_unadj / close`. Alpaca's `adjustment="all"` adjusts *prices* for splits and dividends but *volume only for splits*. Measured on the caches (penny daily joined to the breakout daily, which has `volume_unadj`): BRSP 2024-03-04 has `f = 1.345`, adjusted volume = raw volume = 1,034,853; ENVB has `f = volume / volume_unadj = 0.00556` (a 1:180 split). Dividing by `f` misstates raw volume by more than 2 % on 58 % (2017), 44 % (2021) and 35 % (2024) of daily symbol-days. The minute cache follows the same convention (its regular-session volume tracks the daily adjusted volume, not volume ÷ `f`). So the volume factor is `s = volume_v / volume_unadj_v` from the breakout daily row of the same `(symbol, date)`, used only where that file's price factor agrees with the penny daily's `f` within `unadjust_tolerance` (same adjustment anchor). The screener's ADV uses `close_unadj × volume_unadj_v`, raw × raw, with no factor at all.
- **D2 — cache membership is `requested.json`, not "has rows".** A symbol that was requested for `D` but has zero rows (halted, delisted) is outcome information about `D`; dropping it at screen time would select on the outcome. Membership = the day's `minute/dt=D/requested.json` (verified ⊇ symbols present); fall back to the symbols present in `minute.parquet` on the 3 days with no `requested.json`. A requested symbol with zero rows then reaches the feed and becomes a coverage reject (`reason="requested_no_rows"`); a symbol outside membership (possible only with `restrict_to_minute_cache=False`) is a coverage reject with `reason="not_requested"`.
- **D3 — quotes are raw and need no conversion.** Verified: the 2024-03-04 decision-window `mid` equals the 09:34 minute close × `f` (BRSP 6.705 vs 6.700; ENVB 1.52 vs 1.53). Synthetic quotes come from `quote_summary_r4.parquet` ∪ `quote_summary_rev.parquet` (disjoint on `(symbol, date, window)`, verified), windows `decision`/`entry`/`exit` = 60 s from `research.penny.etl.quotes.DECISION_MINUTE`/`ENTRY_MINUTE`/`EXIT_MINUTE` (`09:34`, `09:35`, `15:55`), stamped at window end. Sizes in shares = `notional / price`, because `summarise_quotes` already multiplied by 100 (`quotes.py:139-143`); a `quote_size_scale` (default `1.0`) exists for when the round-lot assumption is checked against current Alpaca docs (Task 7 step). Quotes are suppressed for rejected symbol-days.
- **D4 — prior close needs a prior-session row.** The market's prior session is `prior_date = max(daily.date < D)`. If it is older than `D − max_prior_gap_days` (default 5 calendar days) the daily cache is missing → `MissingDataError`. A symbol whose last row before `D` is not on `prior_date` has no prior close and is out of the universe. ADV is the median of the last `adv_window = 20` rows (all 20 must have finite raw dollar volume), read from `daily_lookback_days = 45` calendar days.
- **D5 — a candidate without regular-session rows but with other rows** (only 09:25–09:29 or ≥ 16:00) is not a reject: it simply gets 390 `Gap`s.

## Review Focus

1. **Dividend-paying symbols' share volume.** A name with a large cumulative dividend factor (BRSP, `f = 1.345`) must come out with its cached volume unchanged and prices × `f`. Expected: bar volume = cached volume. Pinned in Task 2 `test_dividend_factor_leaves_share_volume_unchanged`, Task 9 `test_dividend_day_volume_unchanged`, Task 10 `test_real_dividend_and_split_volume_conversion`.
2. **Daylight saving.** 09:25 ET is 14:25 UTC in March before the switch and 13:25 UTC in July. Expected: the screen fires at the ET wall-clock time on both. Pinned in Task 1 `test_et_to_utc_handles_dst` and Task 6 `test_fires_at_screen_time_in_summer`.
3. **numpy / pandas scalars leaking into events.** `volume` read as `float64` or `np.int64`, `bar_start` as `pd.Timestamp`. Expected: emitted fields are builtin `int` / `float` / `datetime` and construction never raises on clean data. Pinned in Task 9 `test_emitted_bar_fields_are_builtin_types`.
4. **First session of a year.** On 2024-01-02 the prior close and most of the 20-day ADV window are in `bars_2023.parquet`. Expected: both years read, universe non-empty. Pinned in Task 3 `test_daily_reads_across_year_boundary`.
5. **Input row order and hash seed.** The same data in a different row order must give the same `Candidates` and the same feed output. Pinned in Task 6 `test_candidates_independent_of_input_row_order` and Task 10 `test_real_day_replay_twice_identical_hash`.

---

## File Structure

```
research/intraday/data/__init__.py       (exists, empty; unchanged)
research/intraday/data/errors.py         SealedHoldoutError, LookAheadError, MissingDataError
research/intraday/data/clock.py          ET <-> UTC helpers (the only datetime constructor)
research/intraday/data/holdout.py        HOLDOUT_START, check_holdout
research/intraday/data/paths.py          default roots and cache file paths
research/intraday/data/basis.py          adjusted -> traded basis: factors, conversion, split flags
research/intraday/data/cache.py          parquet readers: daily (penny + volume source), minute, premarket, membership
research/intraday/data/synthetic.py      tiny synthetic cache writers for tests
research/intraday/data/screen_inputs.py  ScreenInputs protocol, ReplayScreenInputs
research/intraday/data/universe.py       causal universe filter, cache intersection
research/intraday/data/ranking.py        pre-market conversion, features, top-n rank
research/intraday/data/screener.py       ScreenerConfig, config_hash, ScreenReport, ScreenerAgent
research/intraday/data/rejects.py        Reject, counts_by_arm, write_rejects_jsonl
research/intraday/data/replay_quotes.py  synthetic Quote objects from cached summaries
research/intraday/data/feed.py           SessionData, BarSource protocol, FeedAgent
research/intraday/data/replay_feed.py    ReplayFeed (BarSource over the caches)
tests/test_intraday_data_common.py       Tasks 1-3
tests/test_intraday_data_screener.py     Tasks 4-6 (every screener.md test name)
tests/test_intraday_data_feed.py         Tasks 7-9 (every feed.md test name)
tests/test_intraday_data_real.py         Task 10 (skips without data)
```

---

### Task 1: Errors, ET clock, holdout guard, paths

**Refs:** #20, #21

**Files:**
- Create: `research/intraday/data/errors.py`, `clock.py`, `holdout.py`, `paths.py`
- Test: `tests/test_intraday_data_common.py`

**Interfaces:**
- Produces:
  - `errors.py`: `class SealedHoldoutError(ValueError)`, `class LookAheadError(AssertionError)`, `class MissingDataError(ValueError)`.
  - `clock.py`: `ET: ZoneInfo = ZoneInfo("America/New_York")`; `et_to_utc(d: date, hhmm: str) -> datetime` (result `tzinfo is timezone.utc`); `et_date(ts: datetime) -> date`; `et_hhmm(ts: datetime) -> str` (`"HH:MM"`); `minute_index_to_utc(d: date, minutes: Iterable[str]) -> dict[str, datetime]` (one `et_to_utc` per distinct `"HH:MM"`, for vectorised mapping).
  - `holdout.py`: `HOLDOUT_START: date = date(2025, 8, 7)`; `check_holdout(session_date: date, holdout_start: date = HOLDOUT_START, unseal: bool = False) -> None`.
  - `paths.py`: `DEFAULT_DATA_ROOT: Path`, `DEFAULT_VOLUME_DAILY_ROOT: Path` (values in Global Constraints); `minute_file(root: Path, d: date) -> Path`, `requested_file(root: Path, d: date) -> Path`, `premarket_file(root: Path, d: date) -> Path`, `daily_file(root: Path, year: int) -> Path` (`root/"daily"/f"bars_{year}.parquet"`), `volume_daily_file(volume_root: Path, year: int) -> Path` (`volume_root/f"bars_{year}.parquet"`), `quote_summary_files(root: Path) -> tuple[Path, Path]` (`quote_summary_r4.parquet`, `quote_summary_rev.parquet`).

- [ ] **Step 1: Write the failing tests**

```python
def test_et_to_utc_handles_dst() -> None:
    assert et_to_utc(date(2024, 3, 4), "09:25") == datetime(2024, 3, 4, 14, 25, tzinfo=timezone.utc)
    assert et_to_utc(date(2024, 7, 1), "09:25") == datetime(2024, 7, 1, 13, 25, tzinfo=timezone.utc)
    assert et_to_utc(date(2024, 3, 4), "09:25").tzinfo is timezone.utc

def test_et_date_and_hhmm_round_trip() -> None:
    assert et_date(T_DEV) == date(2024, 3, 4) and et_hhmm(T_DEV) == "09:31"
    assert et_date(datetime(2024, 3, 5, 3, 0, tzinfo=timezone.utc)) == date(2024, 3, 4)

def test_minute_index_to_utc_maps_each_distinct_minute() -> None:
    m = minute_index_to_utc(date(2024, 3, 4), ["09:30", "09:31", "09:30"])
    assert list(m) == ["09:30", "09:31"] and m["09:31"] == datetime(2024, 3, 4, 14, 31, tzinfo=timezone.utc)

def test_check_holdout_boundary() -> None:
    check_holdout(date(2025, 8, 6))
    with pytest.raises(SealedHoldoutError):
        check_holdout(date(2025, 8, 7))
    check_holdout(date(2025, 8, 7), unseal=True)

def test_paths_layout() -> None:
    root = Path("R")
    assert minute_file(root, date(2024, 3, 4)) == Path("R/minute/dt=2024-03-04/minute.parquet")
    assert requested_file(root, date(2024, 3, 4)) == Path("R/minute/dt=2024-03-04/requested.json")
    assert premarket_file(root, date(2024, 3, 4)) == Path("R/premarket/dt=2024-03-04/premarket.parquet")
    assert daily_file(root, 2024) == Path("R/daily/bars_2024.parquet")
    assert volume_daily_file(Path("V"), 2024) == Path("V/bars_2024.parquet")
```

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest tests/test_intraday_data_common.py -v` → FAIL, `ModuleNotFoundError: research.intraday.data.clock`.
- [ ] **Step 3: Implement the four modules.** `et_to_utc` = `datetime.combine(d, time.fromisoformat(hhmm), tzinfo=ET).astimezone(timezone.utc)`. `SealedHoldoutError` message names the date, `holdout_start` and "pass unseal=True only by a recorded decision".
- [ ] **Step 4: Run to verify pass** — same command → 5 passed. Then `$PY -m pytest tests/test_intraday_guards.py -q` → passed.
- [ ] **Step 5: Commit** — `git add research/intraday/data/{errors,clock,holdout,paths}.py tests/test_intraday_data_common.py`; message `feat(intraday-data): ET clock, holdout guard and cache paths` + trailers (#20, #21).

---

### Task 2: Traded-basis arithmetic (`basis.py`)

**Refs:** #20, #21

**Files:**
- Create: `research/intraday/data/basis.py`
- Test: `tests/test_intraday_data_common.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (all vectorised; Series are indexed by symbol unless stated):
  - `price_factor(close_unadj: pd.Series, close: pd.Series) -> pd.Series` — `close_unadj / close`; NaN where either is non-finite or ≤ 0. Name `"f"`.
  - `volume_factor(volume: pd.Series, volume_unadj: pd.Series) -> pd.Series` — `volume / volume_unadj` (adjusted ÷ raw = the split-only factor, decision D1); NaN where either is non-finite or ≤ 0. Name `"s"`.
  - `factor_consistent(a: pd.Series, b: pd.Series, tol: float) -> pd.Series` (bool) — `abs(a / b - 1) <= tol`, aligned on index; NaN or missing → `False`.
  - `split_flags(f_prev: pd.Series, f_session: pd.Series, threshold: float) -> pd.Series` (bool, index of `f_prev`) — `abs(log(f_prev / f_session)) > threshold`; a symbol missing from `f_session` or with NaN → `False` (kept: "if the D row is missing, no split is known").
  - `to_traded_basis(bars: pd.DataFrame, f: pd.Series, s: pd.Series) -> pd.DataFrame` — copy of `bars` (needs `symbol, open, high, low, close, vwap, volume`) with the five prices × `f[symbol]` and `volume` ÷ `s[symbol]` (float, unrounded); raises `ValueError` naming the symbols if any bar's symbol lacks a finite `f` or `s` (callers filter first).

- [ ] **Step 1: Write the failing tests** (values are the measured BRSP/ENVB rows of 2024-03-04)

```python
def test_price_factor_is_raw_over_adjusted() -> None:
    f = price_factor(pd.Series({"BRSP": 6.60, "Z": 1.0}), pd.Series({"BRSP": 4.907, "Z": 0.0}))
    assert f["BRSP"] == pytest.approx(1.345, rel=1e-3) and np.isnan(f["Z"])

def test_dividend_factor_leaves_share_volume_unchanged() -> None:
    bars = pd.DataFrame({"symbol": ["BRSP"], "open": [5.0], "high": [5.1], "low": [4.9],
                         "close": [5.0], "vwap": [5.0], "volume": [1_034_853.0]})
    s = volume_factor(pd.Series({"BRSP": 1_034_853.0}), pd.Series({"BRSP": 1_034_853.0}))
    out = to_traded_basis(bars, pd.Series({"BRSP": 1.345}), s)
    assert out["volume"].iloc[0] == 1_034_853.0
    assert out["close"].iloc[0] == pytest.approx(5.0 * 1.345)

def test_reverse_split_scales_volume_by_split_factor() -> None:
    s = volume_factor(pd.Series({"ENVB": 7798.0}), pd.Series({"ENVB": 1_403_699.0}))
    bars = pd.DataFrame({"symbol": ["ENVB"], "open": [279.0], "high": [279.0], "low": [279.0],
                         "close": [279.0], "vwap": [279.0], "volume": [7798.0]})
    out = to_traded_basis(bars, pd.Series({"ENVB": 1.55 / 279.0}), s)
    assert out["volume"].iloc[0] == pytest.approx(1_403_699.0)
    assert out["close"].iloc[0] == pytest.approx(1.55)

def test_split_flags_threshold_and_missing_session_row() -> None:
    f_prev = pd.Series({"SPLIT": 0.1, "DIV": 0.97, "NOROW": 0.1, "FWD": 1.0})
    f_d = pd.Series({"SPLIT": 1.0, "DIV": 1.0, "FWD": 0.5})
    flags = split_flags(f_prev, f_d, math.log(1.25))
    assert flags.to_dict() == {"SPLIT": True, "DIV": False, "NOROW": False, "FWD": True}

def test_factor_consistent_nan_and_missing_are_false() -> None:
    ok = factor_consistent(pd.Series({"A": 1.01, "B": np.nan, "C": 1.0}), pd.Series({"A": 1.0, "B": 1.0}), 0.02)
    assert ok.to_dict() == {"A": True, "B": False, "C": False}

def test_to_traded_basis_missing_factor_raises() -> None:
    bars = pd.DataFrame({"symbol": ["X"], "open": [1.0], "high": [1.0], "low": [1.0],
                         "close": [1.0], "vwap": [1.0], "volume": [1.0]})
    with pytest.raises(ValueError, match="X"):
        to_traded_basis(bars, pd.Series(dtype=float), pd.Series({"X": 1.0}))
```

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest tests/test_intraday_data_common.py -v` → the 6 new tests FAIL (import error).
- [ ] **Step 3: Implement `basis.py`.** Map factors onto rows with `bars["symbol"].map(f)`; module docstring states D1 with the BRSP/ENVB evidence and cites `research/penny/etl/bars.py:226-238` as the event-factor precedent (that file uses the reciprocal `close / close_unadj`; `abs(log)` of the step is the same).
- [ ] **Step 4: Run to verify pass** — same command → 11 passed.
- [ ] **Step 5: Commit** — `feat(intraday-data): traded-basis factors, split-only volume factor, split flags` + trailers (#20, #21).

---

### Task 3: Cache readers, synthetic writers, ScreenInputs adapter

**Refs:** #20, #21

**Files:**
- Create: `research/intraday/data/cache.py`, `synthetic.py`, `screen_inputs.py`
- Test: `tests/test_intraday_data_common.py` (append)

**Interfaces:**
- Consumes: Task 1 paths, `check_holdout`, `minute_index_to_utc`, `MissingDataError`.
- Produces:
  - `cache.py`:
    - `DAILY_COLUMNS = ("symbol", "date", "open", "close", "open_unadj", "close_unadj", "volume")` read from the penny daily; `VOLUME_COLUMNS = ("symbol", "date", "close", "close_unadj", "volume", "volume_unadj")` read from the volume source and renamed with suffix `_v` except `volume_unadj` → `volume_unadj_v`.
    - `read_daily(data_root: Path, volume_root: Path, start: date, end: date) -> pd.DataFrame` — rows `start <= date < end` from every year in `[start.year, (end - 1 day).year]`, via `pd.read_parquet(path, columns=..., filters=[("date", ">=", ts_start), ("date", "<", ts_end)])`; penny left-joined to volume source on `(symbol, date)`; a missing year file contributes nothing; a duplicate `(symbol, date)` in either source raises `ValueError`; result sorted by `(symbol, date)`, `date` naive `datetime64[ns]`. Columns: `DAILY_COLUMNS + ("close_v", "close_unadj_v", "volume_v", "volume_unadj_v")`.
    - `minute_cache_membership(data_root: Path, d: date) -> tuple[str, ...]` — sorted symbols of `requested.json` if it exists, else sorted unique symbols of `minute.parquet`, else `()` (decision D2).
    - `read_minute_day(data_root: Path, d: date, symbols: tuple[str, ...]) -> pd.DataFrame` — rows of `minute.parquet` for `symbols` (empty frame with the cache's columns if the file is missing).
    - `read_premarket_day(data_root: Path, d: date) -> pd.DataFrame` — `premarket.parquet` plus a `bar_start` column (`datetime64[ns, UTC]`, from `minute_et` via `minute_index_to_utc`); empty frame with those columns if missing.
  - `synthetic.py` (test support, deterministic, no randomness): `write_daily(root: Path, rows: pd.DataFrame) -> None` (splits by year into `daily_file`), `write_volume_daily(volume_root: Path, rows: pd.DataFrame) -> None`, `write_minute(root: Path, d: date, rows: pd.DataFrame, requested: tuple[str, ...] | None) -> None`, `write_premarket(root: Path, d: date, rows: pd.DataFrame) -> None`, `write_quote_summaries(root: Path, r4: pd.DataFrame, rev: pd.DataFrame) -> None`, and row builders `daily_row(symbol: str, d: date, close: float, f: float = 1.0, s: float | None = None, volume: float = 1e6, open_: float | None = None) -> dict` (raw = adjusted × `f`; `volume_unadj = volume / s`, `s` defaults to `f`) and `minute_rows(symbol: str, d: date, minutes: Sequence[str], price: float = 2.0, volume: float = 1000.0) -> pd.DataFrame` (cache dtypes: `volume`, `trade_count` float64).
  - `screen_inputs.py`:
    - `class ScreenInputs(Protocol)`: `daily_before(self, session_date: date, lookback_days: int) -> pd.DataFrame` (rows in `[D - lookback_days, D)`, `read_daily` columns); `session_factor(self, session_date: date) -> pd.Series` (`f_D` per symbol, name `"f"`, **no price columns** — the only `D` daily read the screener gets); `premarket_before(self, session_date: date, cutoff: datetime) -> pd.DataFrame` (`read_premarket_day` rows with `bar_start + 1 min <= cutoff`); `minute_cache_symbols(self, session_date: date) -> tuple[str, ...]`.
    - `class ReplayScreenInputs` implementing it: `__init__(self, data_root: Path = DEFAULT_DATA_ROOT, volume_root: Path = DEFAULT_VOLUME_DAILY_ROOT, holdout_start: date = HOLDOUT_START, unseal: bool = False)`; every method calls `check_holdout` first.

- [ ] **Step 1: Write the failing tests**

```python
def test_daily_before_excludes_session_date(tmp_path) -> None:  # rows on D-1 and D written
    df = ReplayScreenInputs(tmp_path, tmp_path / "v").daily_before(date(2024, 3, 4), 45)
    assert df["date"].max() == pd.Timestamp("2024-03-01") and set(df.columns) >= {"volume_unadj_v", "close_v"}

def test_daily_reads_across_year_boundary(tmp_path) -> None:  # one symbol: 2023-12-27, -28, -29 and 2024-01-02
    df = ReplayScreenInputs(tmp_path, tmp_path / "v").daily_before(date(2024, 1, 2), 45)
    assert list(df["date"].dt.day) == [27, 28, 29] and df["date"].min().year == 2023

def test_duplicate_daily_row_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        read_daily(tmp_path, tmp_path / "v", date(2024, 3, 1), date(2024, 3, 5))

def test_minute_cache_symbols_prefers_requested_json(tmp_path) -> None:  # rows for A; requested A, B
    assert ReplayScreenInputs(tmp_path, tmp_path / "v").minute_cache_symbols(D) == ("A", "B")

def test_minute_cache_symbols_falls_back_to_present(tmp_path) -> None:  # rows for B, A; no requested.json
    assert minute_cache_membership(tmp_path, D) == ("A", "B")

def test_premarket_before_drops_bars_ending_after_cutoff(tmp_path) -> None:  # 09:23, 09:24, 09:25 bars
    pm = ReplayScreenInputs(tmp_path, tmp_path / "v").premarket_before(D, et_to_utc(D, "09:25"))
    assert list(pm["minute_et"]) == ["09:23", "09:24"]

def test_session_factor_returns_factor_only(tmp_path) -> None:
    f = ReplayScreenInputs(tmp_path, tmp_path / "v").session_factor(D)
    assert isinstance(f, pd.Series) and f.name == "f" and f["A"] == pytest.approx(0.5)

@pytest.mark.parametrize("call", ["daily_before", "session_factor", "premarket_before", "minute_cache_symbols"])
def test_replay_screen_inputs_refuse_holdout_without_unseal(tmp_path, call) -> None:
    with pytest.raises(SealedHoldoutError):
        _invoke(ReplayScreenInputs(tmp_path, tmp_path / "v"), call, date(2025, 8, 7))
```

(`D = date(2024, 3, 4)`; `_invoke` passes the extra argument each method needs. Each test writes its own frames with `synthetic.py`.)

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest tests/test_intraday_data_common.py -v` → new tests FAIL (import error).
- [ ] **Step 3: Implement `cache.py`, `synthetic.py`, `screen_inputs.py`.** `session_factor` reads `read_daily(D, D + 1 day)` and returns `price_factor(...)` indexed by symbol.
- [ ] **Step 4: Run to verify pass** — same command → all pass; `$PY -m pytest tests/test_intraday_guards.py -q` → passed.
- [ ] **Step 5: Commit** — `feat(intraday-data): cache readers and the replay ScreenInputs adapter` + trailers (#20, #21).

---

### Task 4: Causal universe filter and cache intersection (`universe.py`)

**Refs:** #20

**Files:**
- Create: `research/intraday/data/universe.py`
- Test: `tests/test_intraday_data_screener.py` (create)

**Interfaces:**
- Consumes: `basis.price_factor`, `volume_factor`, `split_flags`; `MissingDataError`, `LookAheadError`.
- Produces:
  - `@dataclass(frozen=True) class UniverseResult`: `symbols: tuple[str, ...]` (sorted), `prior: pd.DataFrame` (index symbol, for `symbols` only; columns `close_unadj`, `f`, `f_v`, `s` of the `prior_date` row), `prior_date: date`, `n_split_excluded: int`.
  - `causal_universe(daily: pd.DataFrame, session_date: date, f_session: pd.Series, *, min_price: float, max_price: float, min_adv_usd: float, adv_window: int, exclude_split_days: bool, split_log_step_threshold: float, max_prior_gap_days: int) -> UniverseResult`.
    Order: raise `LookAheadError` if any `daily.date >= session_date`; raise `MissingDataError` if `daily` is empty or `prior_date < session_date - max_prior_gap_days`; prior row = each symbol's last row, kept only if dated `prior_date` (D4); `min_price <= close_unadj <= max_price`; ADV = median of the last `adv_window` values of `close_unadj × volume_unadj_v` per symbol, requiring `adv_window` finite values, `>= min_adv_usd`; `f = price_factor(close_unadj, close)`, `f_v = price_factor(close_unadj_v, close_v)`, `s = volume_factor(volume_v, volume_unadj_v)` on the prior row; split exclusion with `split_flags(f, f_session, threshold)` when `exclude_split_days`.
  - `intersect_cache(symbols: tuple[str, ...], cache_symbols: tuple[str, ...]) -> tuple[tuple[str, ...], int]` — sorted intersection and the count removed.

- [ ] **Step 1: Write the failing tests** (a module helper `_daily(rows)` builds a frame with `synthetic.daily_row` over 20 sessions before `D = date(2024, 3, 4)`; `_universe(daily, f_session=..., **overrides)` calls `causal_universe` with the screener defaults)

```python
def test_universe_is_causal_filter_intersected_with_cache() -> None:
    u = _universe(_daily(["A", "B", "C"]))          # all pass
    kept, removed = intersect_cache(u.symbols, ("A", "C", "Z"))
    assert kept == ("A", "C") and removed == 1

def test_cache_symbol_outside_causal_filter_never_selected() -> None:
    u = _universe(_daily(["A"], cheap=["PENNY"]))   # PENNY raw close 0.40
    assert intersect_cache(u.symbols, ("A", "PENNY"))[0] == ("A",)

def test_price_filter_uses_unadjusted_prior_close() -> None:
    # RS: adjusted 40.0, raw 2.0 (f=0.05) -> in;  UP: adjusted 2.0, raw 40.0 (f=20) -> out
    assert _universe(_daily_with(RS=(40.0, 0.05), UP=(2.0, 20.0))).symbols == ("RS",)

def test_split_day_symbol_excluded() -> None:
    u = _universe(_daily(["A", "SPL"], f={"SPL": 0.1}), f_session=pd.Series({"A": 1.0, "SPL": 1.0}))
    assert u.symbols == ("A",) and u.n_split_excluded == 1

def test_ex_dividend_factor_step_is_not_a_split() -> None:
    u = _universe(_daily(["DIV"], f={"DIV": 0.97}), f_session=pd.Series({"DIV": 1.0}))
    assert u.symbols == ("DIV",) and u.n_split_excluded == 0

def test_missing_session_daily_row_keeps_symbol() -> None:
    assert _universe(_daily(["A"], f={"A": 0.1}), f_session=pd.Series(dtype=float)).symbols == ("A",)

def test_dollar_volume_uses_unadjusted_price_and_shares() -> None:
    # ADJ: adjusted close x adjusted volume = $2M but raw close x volume_unadj = $0.5M -> out
    # RAW: adjusted $0.5M, raw $2M -> in
    assert _universe(_daily_dv(ADJ=(2e6, 5e5), RAW=(5e5, 2e6))).symbols == ("RAW",)

def test_missing_daily_history_raises() -> None:
    with pytest.raises(MissingDataError):
        _universe(_daily(["A"], last_date=date(2024, 2, 20)))

def test_symbol_without_prior_session_row_excluded() -> None:
    assert "STALE" not in _universe(_daily(["A"], stale=["STALE"])).symbols

def test_short_adv_history_excluded() -> None:
    assert _universe(_daily(["A"], young={"NEW": 10})).symbols == ("A",)
```

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest tests/test_intraday_data_screener.py -v` → FAIL (import error).
- [ ] **Step 3: Implement `universe.py`** with `groupby("symbol").tail(adv_window)` then `groupby.agg(median, count)`; no Python loop over symbols.
- [ ] **Step 4: Run to verify pass** — same command → 10 passed.
- [ ] **Step 5: Commit** — `feat(intraday-data): causal universe filter on unadjusted prior-day data` + trailer (#20).

---

### Task 5: Pre-market conversion and ranking (`ranking.py`)

**Refs:** #20

**Files:**
- Create: `research/intraday/data/ranking.py`
- Test: `tests/test_intraday_data_screener.py` (append)

**Interfaces:**
- Consumes: `UniverseResult.prior` (Task 4), `basis.factor_consistent`, `basis.to_traded_basis`; reuse `research.penny.features.premarket.premarket_features(panel: pd.DataFrame, prev_close: Optional[pd.Series] = None) -> pd.DataFrame` (one row per symbol; needs `symbol, minute_et, open, high, low, close, volume`; computes `pm_dollar_volume = Σ close × volume`, `pm_close` = last close, and `pm_gap = pm_close / prev_close − 1` when `prev_close` is given; calls `assert_no_post_decision_bars(panel, minute_col="minute_et")`, the 09:35 second guard).
- Produces:
  - `convert_premarket(pm: pd.DataFrame, prior: pd.DataFrame, tol: float) -> tuple[pd.DataFrame, tuple[str, ...]]` — keep pm rows whose symbol is in `prior.index` with finite `s` and `factor_consistent(f_v, f, tol)`; return `to_traded_basis(rows, prior.f, prior.s)` and the sorted symbols dropped as unconvertible (they had pm bars but no trustworthy factor).
  - `RANKABLE_FEATURES: tuple[str, ...]` — exactly the columns `premarket_features(..., prev_close=...)` returns: `pm_dollar_volume, pm_volume, pm_bars, pm_vwap, pm_close, pm_high, pm_low, pm_range, pm_close_position, pm_dist_from_vwap, pm_late_dv_share, pm_early_dv_share, pm_vol_centroid, pm_late_drift, pm_first_minute, pm_minutes_traded, pm_gap, pm_high_gap, pm_fade_from_high`. (Not `PREMARKET_FEATURES`: its `pm_dv_to_adv`/`pm_rvol` come from `add_relative_features`, which needs `adv_20`/`prev_close` columns this pipeline does not build; R16's `pm_rvol` would be a new feature added here when R16 is final.)
  - `rank_symbols(universe: tuple[str, ...], pm_traded: pd.DataFrame, prev_close_unadj: pd.Series, rank_by: tuple[str, ...], top_n: int) -> tuple[tuple[str, ...], int]` — features from `premarket_features(pm_traded[symbol in universe], prev_close=prev_close_unadj)`; returns the selected symbols **sorted** and the count of universe symbols with no pm bars. If no universe symbol has pm bars, return `((), len(universe))` **before** calling `premarket_features` (it returns a column-less `pd.DataFrame()` on empty input, and sorting that raises `KeyError`). Order: `sort_values([*rank_by, "symbol"], ascending=[False] * len(rank_by) + [True], na_position="last", kind="mergesort")`, `head(top_n)`, then `tuple(sorted(...))`.

- [ ] **Step 1: Write the failing tests**

```python
def test_conversion_uses_prior_day_factor() -> None:
    prior = _prior(A=dict(close_unadj=2.0, f=0.5, f_v=0.5, s=0.5))  # D's own factor is 1.0, never passed
    out, dropped = convert_premarket(_pm("A", close=4.0, volume=100.0), prior, 0.02)
    assert out["close"].iloc[0] == 2.0 and out["volume"].iloc[0] == 200.0 and dropped == ()

def test_premarket_bars_converted_to_traded_basis() -> None:
    # dividend-only factor (f=0.9, s=1.0): adjusted 4.0 x 100 = $400, traded 3.6 x 100 = $360;
    # the design's "volume / f" would give 3.6 x 111.1 = $400, the adjusted number again
    prior = _prior(A=dict(close_unadj=3.0, f=0.9, f_v=0.9, s=1.0))
    out, _ = convert_premarket(_pm("A", close=4.0, volume=100.0), prior, 0.02)
    feats = premarket_features(out, prev_close=pd.Series({"A": 3.0}))
    assert feats.loc[0, "pm_dollar_volume"] == pytest.approx(360.0)
    assert feats.loc[0, "pm_gap"] == pytest.approx(3.6 / 3.0 - 1)

def test_unconvertible_premarket_symbol_skipped() -> None:
    prior = _prior(A=dict(close_unadj=2.0, f=0.5, f_v=0.25, s=0.25))   # sources disagree
    out, dropped = convert_premarket(_pm("A"), prior, 0.02)
    assert out.empty and dropped == ("A",)

def test_symbols_sorted_unique_and_cut_to_top_n_before_sorting() -> None:
    pm = _pm_gaps(Z=0.30, A=0.20, M=0.10)              # rank order Z, A, M
    sel, _ = rank_symbols(("A", "M", "Z"), pm, _prev(1.0, "A", "M", "Z"), ("pm_gap", "pm_dollar_volume"), 2)
    assert sel == ("A", "Z")

def test_symbol_without_premarket_bars_skipped() -> None:
    sel, n_missing = rank_symbols(("A", "B"), _pm_gaps(A=0.1), _prev(1.0, "A", "B"), ("pm_gap",), 5)
    assert sel == ("A",) and n_missing == 1

def test_rank_with_no_premarket_bars_returns_empty() -> None:
    assert rank_symbols(("A", "B"), _pm_gaps(), _prev(1.0, "A", "B"), ("pm_gap",), 5) == ((), 2)

def test_rank_ties_broken_by_symbol() -> None:
    sel, _ = rank_symbols(("B", "A", "C"), _pm_gaps(A=0.1, B=0.1, C=0.1), _prev(1.0, "A", "B", "C"), ("pm_gap",), 2)
    assert sel == ("A", "B")
```

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest tests/test_intraday_data_screener.py -v` → the new tests FAIL.
- [ ] **Step 3: Implement `ranking.py`.**
- [ ] **Step 4: Run to verify pass** — same command → 17 passed.
- [ ] **Step 5: Commit** — `feat(intraday-data): pre-market features on the traded basis and top-n rank` + trailer (#20).

---

### Task 6: ScreenerAgent

**Refs:** #20

**Files:**
- Create: `research/intraday/data/screener.py`
- Test: `tests/test_intraday_data_screener.py` (append)

**Interfaces:**
- Consumes: `ScreenInputs` (Task 3), `causal_universe`, `intersect_cache` (Task 4), `convert_premarket`, `rank_symbols`, `RANKABLE_FEATURES` (Task 5), `check_holdout`, `et_to_utc`, `et_date`, `et_hhmm`, `LookAheadError`; contracts `ClockTick`, `Candidates`, `Event`.
- Produces:
  - `@dataclass(frozen=True) class ScreenerConfig` with exactly these fields and defaults: `screen_time_et: str = "09:25"`, `premarket_start_et: str = "04:00"`, `arm: str = "screen"` (`"screen" | "no_selection"`), `min_price: float = 1.0`, `max_price: float = 20.0`, `min_adv_usd: float = 1_000_000.0`, `adv_window: int = 20`, `rank_by: tuple[str, ...] = ("pm_gap", "pm_dollar_volume")`, `top_n: int = 20`, `exclude_split_days: bool = True`, `split_log_step_threshold: float = math.log(1.25)`, `unadjust_tolerance: float = 0.02`, `holdout_start: date = HOLDOUT_START`, `unseal: bool = False`, `data_root: str = str(DEFAULT_DATA_ROOT)`, `volume_daily_root: str = str(DEFAULT_VOLUME_DAILY_ROOT)`, `restrict_to_minute_cache: bool = True`, `daily_lookback_days: int = 45`, `max_prior_gap_days: int = 5`, `max_late_minutes: int = 5`. `__post_init__` raises `ValueError` for an unknown `arm`, `top_n <= 0`, empty `rank_by`, or `rank_by` names outside `ranking.RANKABLE_FEATURES` (Task 5).
  - `config_hash(config: ScreenerConfig) -> str` — sha256 hex of `json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), default=date.isoformat)`.
  - `@dataclass(frozen=True) class ScreenReport`: `session_date: date`, `arm: str`, `screen_id: str`, `n_universe: int`, `n_removed_by_cache: int`, `n_split_excluded: int`, `n_unconvertible: int`, `n_no_premarket: int`, `n_selected: int`.
  - `class ScreenerAgent`: `__init__(self, inputs: ScreenInputs, config: ScreenerConfig = ScreenerConfig(), name: str = "screener")`; `subscribes = (ClockTick,)`; `on_event(self, event: Event) -> tuple[Event, ...]`; `report: ScreenReport | None` (set on firing, logged with `logging.getLogger(__name__).info`). The integrator builds `ReplayScreenInputs(Path(cfg.data_root), Path(cfg.volume_daily_root), cfg.holdout_start, cfg.unseal)` from the same config.
- Behaviour on a `ClockTick` at `ts`:
  1. First tick sets `session_date = et_date(ts)`; a later tick with another ET date raises `ValueError`. If the first tick's ET time is later than `screen_time_et + max_late_minutes` → raise `ValueError` ("run started late").
  2. Fired, or `ts < et_to_utc(D, screen_time_et)` → `()`.
  3. Fire: `check_holdout(D, holdout_start, unseal)`; `daily = inputs.daily_before(D, daily_lookback_days)`; any `daily.date >= D` → `LookAheadError`; `universe = causal_universe(daily, D, inputs.session_factor(D), ...)`; if `restrict_to_minute_cache` → `intersect_cache(universe.symbols, inputs.minute_cache_symbols(D))` else removed = 0.
  4. `arm == "no_selection"` → all intersected symbols. `arm == "screen"` → `pm = inputs.premarket_before(D, ts)`; any `bar_start + 1 min > ts` → `LookAheadError`; drop rows with `minute_et < premarket_start_et`; `convert_premarket`; `rank_symbols(..., prev_close_unadj=universe.prior.close_unadj, ...)`.
  5. Return `(Candidates(ts=ts, screen_id=f"{arm}:{config_hash(cfg)[:12]}:{D:%Y-%m-%d}", symbols=selected),)`.

- [ ] **Step 1: Write the failing tests.** A `FakeScreenInputs` dataclass in the test file holds `daily`, `f_session`, `premarket`, `cache_symbols` frames, returns them unfiltered, and records each call. `D = et_date(T_DEV)`; `tick(hhmm) = ClockTick(ts=et_to_utc(D, hhmm))`; `_good_inputs()` gives three symbols A, B, C that pass every filter, all in the cache, with pre-market gaps C > A > B.

```python
def test_emits_one_candidates_at_first_tick_at_or_after_screen_time() -> None:
    agent = ScreenerAgent(_good_inputs())
    outs = [agent.on_event(tick(t)) for t in ("09:20", "09:24", "09:25", "09:26")]
    assert [len(o) for o in outs] == [0, 0, 1, 0] and isinstance(outs[2][0], Candidates)

def test_ignores_ticks_before_screen_time_and_after_firing() -> None:
    inputs = _good_inputs(); agent = ScreenerAgent(inputs)
    for t in ("09:21", "09:25", "09:40", "15:59"):
        agent.on_event(tick(t))
    assert inputs.calls.count("daily_before") == 1

def test_candidates_ts_equals_firing_tick_ts() -> None:
    agent = ScreenerAgent(_good_inputs()); agent.on_event(tick("09:24"))
    (c,) = agent.on_event(tick("09:26"))
    assert c.ts == et_to_utc(D, "09:26")

def test_fires_at_screen_time_in_summer() -> None:
    d = date(2024, 7, 1)
    (c,) = ScreenerAgent(_good_inputs(session=d)).on_event(ClockTick(ts=datetime(2024, 7, 1, 13, 25, tzinfo=timezone.utc)))
    assert c.screen_id.endswith(":2024-07-01")

def test_no_selection_arm_returns_whole_filtered_universe() -> None:
    (c,) = ScreenerAgent(_good_inputs(), ScreenerConfig(arm="no_selection", top_n=1)).on_event(tick("09:25"))
    assert c.symbols == ("A", "B", "C") and c.screen_id.startswith("no_selection:")

def test_split_flag_reads_factor_step_not_session_prices() -> None:
    inputs = _good_inputs(prior_f={"B": 0.1}, f_session={"A": 1.0, "B": 1.0, "C": 1.0})
    (c,) = ScreenerAgent(inputs, ScreenerConfig(arm="no_selection")).on_event(tick("09:25"))
    assert c.symbols == ("A", "C")
    assert all(d < D for d in inputs.daily_dates_seen())          # no D price row ever requested

def test_holdout_date_refused_without_unseal() -> None:
    sealed = date(2025, 8, 7)
    with pytest.raises(SealedHoldoutError):
        ScreenerAgent(_good_inputs(session=sealed)).on_event(ClockTick(ts=et_to_utc(sealed, "09:25")))

def test_input_row_at_cutoff_raises() -> None:
    inputs = _good_inputs(extra_pm_minute="09:25")                 # bar ends 09:26 > tick 09:25
    with pytest.raises(LookAheadError):
        ScreenerAgent(inputs).on_event(tick("09:25"))

def test_daily_row_on_session_date_raises() -> None:
    with pytest.raises(LookAheadError):
        ScreenerAgent(_good_inputs(daily_through=D)).on_event(tick("09:25"))

def test_late_first_tick_raises() -> None:
    with pytest.raises(ValueError, match="late"):
        ScreenerAgent(_good_inputs()).on_event(tick("09:31"))

def test_second_session_date_raises() -> None:
    agent = ScreenerAgent(_good_inputs()); agent.on_event(tick("09:25"))
    with pytest.raises(ValueError):
        agent.on_event(ClockTick(ts=et_to_utc(date(2024, 3, 5), "09:25")))

def test_config_rejects_unknown_rank_feature() -> None:
    with pytest.raises(ValueError):
        ScreenerConfig(rank_by=("pm_rvol",))

def test_screen_id_changes_with_config() -> None:
    base = ScreenerConfig()
    ids = {config_hash(base), config_hash(replace(base, top_n=19)), config_hash(replace(base, unseal=True)),
           config_hash(replace(base, min_price=1.5))}
    assert len(ids) == 4

def test_same_inputs_same_candidates() -> None:
    a = ScreenerAgent(_good_inputs()).on_event(tick("09:25"))
    b = ScreenerAgent(_good_inputs()).on_event(tick("09:25"))
    assert a == b

def test_candidates_independent_of_input_row_order() -> None:
    a = ScreenerAgent(_good_inputs()).on_event(tick("09:25"))
    b = ScreenerAgent(_good_inputs(reverse_rows=True)).on_event(tick("09:25"))
    assert a == b

def test_empty_universe_emits_empty_candidates() -> None:
    (c,) = ScreenerAgent(_good_inputs(), ScreenerConfig(min_price=500.0)).on_event(tick("09:25"))
    assert c.symbols == ()

def test_report_records_cache_removals_per_arm() -> None:
    agent = ScreenerAgent(_good_inputs(cache_symbols=("A", "B")), ScreenerConfig(arm="no_selection"))
    agent.on_event(tick("09:25"))
    assert agent.report.n_removed_by_cache == 1 and agent.report.arm == "no_selection"
```

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest tests/test_intraday_data_screener.py -v` → the new tests FAIL (import error).
- [ ] **Step 3: Implement `screener.py`** following the five behaviour steps; the agent holds only `session_date`, `first_tick_seen`, `fired`, `report`.
- [ ] **Step 4: Run to verify pass** — same command → all pass. Confirm every name in `screener.md`'s Test plan appears: `grep -c "def test_" tests/test_intraday_data_screener.py` ≥ 21 and each of the 21 names greps once.
- [ ] **Step 5: Commit** — `feat(intraday-data): ScreenerAgent emits one causal Candidates per session` + trailer (#20). Run the ownership check first.

---

### Task 7: Rejects and synthetic quotes

**Refs:** #21

**Files:**
- Create: `research/intraday/data/rejects.py`, `replay_quotes.py`
- Test: `tests/test_intraday_data_feed.py` (create)

**Interfaces:**
- Consumes: `et_to_utc`, `paths.quote_summary_files`; contracts `Quote`; reuse the window constants `research.penny.etl.quotes.DECISION_MINUTE` (`"09:34"`), `ENTRY_MINUTE` (`"09:35"`), `EXIT_MINUTE` (`"15:55"`) and the 60 s window of `fetch_quotes_for_minute(..., window_seconds: int = 60)`.
- Produces:
  - `rejects.py`: `RejectKind = Literal["coverage", "unadjust", "no_daily", "no_volume_factor"]`; `@dataclass(frozen=True) class Reject`: `kind: RejectKind`, `symbol: str`, `session_date: date`, `reason: str`, `screen_id: str`; property `arm -> str` (= `screen_id.split(":")[0]`). `counts_by_arm(rejects: Iterable[Reject], kind: RejectKind) -> dict[str, int]` (keys sorted). `write_rejects_jsonl(path: Path, rejects: Iterable[Reject]) -> None` (one canonical JSON object per line, sorted by `(session_date, kind, symbol)`, `date` ISO).
  - `replay_quotes.py`: `QUOTE_WINDOWS_ET: dict[str, str] = {"decision": DECISION_MINUTE, "entry": ENTRY_MINUTE, "exit": EXIT_MINUTE}`; `read_quote_summaries(data_root: Path, d: date, symbols: tuple[str, ...]) -> pd.DataFrame` (both files, `filters` on `date`, a duplicate `(symbol, window)` raises `ValueError`, a missing file contributes nothing); `synthetic_quotes(summary: pd.DataFrame, d: date, size_scale: float = 1.0) -> tuple[dict[tuple[str, datetime], Quote], int]` — key `(symbol, stamp)` with `stamp = et_to_utc(d, QUOTE_WINDOWS_ET[window]) + 1 min`; `bid = mid × (1 − hs/1e4)`, `ask = mid × (1 + hs/1e4)` with `hs = half_spread_bps`; `bid_size = int(round(bid_notional / bid × size_scale))`, `ask_size` likewise; a row whose `Quote` construction raises, or with a zero size, or a non-finite input, is skipped and counted (the int).

- [ ] **Step 1: Write the failing tests**

```python
def test_quote_sizes_are_shares() -> None:   # measured BRSP decision row, 2024-03-04
    q, skipped = synthetic_quotes(_summary(BRSP=dict(window="decision", mid=6.705, half_spread_bps=22.371365,
                                                    bid_notional=3345.0, ask_notional=2016.0)), D)
    quote = q[("BRSP", et_to_utc(D, "09:35"))]
    assert type(quote.bid_size) is int and quote.bid_size == round(3345.0 / quote.bid)
    assert quote.ask_size == round(2016.0 / quote.ask) and skipped == 0

def test_quote_stamped_at_window_end() -> None:
    q, _ = synthetic_quotes(_summary(A=dict(window="exit")), D)
    assert list(q) == [("A", et_to_utc(D, "15:56"))]

def test_crossed_or_zero_quote_skipped() -> None:
    q, skipped = synthetic_quotes(_summary(A=dict(half_spread_bps=-5.0), B=dict(bid_notional=0.0)), D)
    assert q == {} and skipped == 2

def test_duplicate_quote_summary_raises(tmp_path) -> None:   # same (symbol, date, window) in r4 and rev
    with pytest.raises(ValueError, match="duplicate"):
        read_quote_summaries(tmp_path, D, ("A",))

def test_counts_by_arm() -> None:
    rs = [Reject("coverage", "A", D, "requested_no_rows", "screen:abc:2024-03-04"),
          Reject("coverage", "B", D, "not_requested", "no_selection:def:2024-03-04"),
          Reject("unadjust", "C", D, "open_mismatch", "screen:abc:2024-03-04")]
    assert counts_by_arm(rs, "coverage") == {"no_selection": 1, "screen": 1}
```

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest tests/test_intraday_data_feed.py -v` → FAIL (import error).
- [ ] **Step 3: Implement both modules.** Build each `Quote` with builtin types (`float(...)`, `int(...)`). Then check the quote size unit: query Alpaca's current market-data docs for historical stock quote `bid_size`/`ask_size` units via context7 (`plugin_context7`) if it is authenticated; if it is not, report NEEDS_CONTEXT in the task report and keep `size_scale = 1.0` (the cache's ×100 round-lot convention, `quotes.py:139-143`). Record the outcome in the commit body.
- [ ] **Step 4: Run to verify pass** — same command → 5 passed.
- [ ] **Step 5: Commit** — `feat(intraday-data): reject records and synthetic quotes from cached NBBO summaries` + trailer (#21).

---

### Task 8: FeedAgent (against a fake source)

**Refs:** #21

**Files:**
- Create: `research/intraday/data/feed.py`
- Test: `tests/test_intraday_data_feed.py` (append)

**Interfaces:**
- Consumes: `Reject` (Task 7), `et_to_utc`, `et_date`; contracts `Bar`, `Gap`, `Quote`, `Candidates`, `ClockTick`, `Event`.
- Produces:
  - `@dataclass(frozen=True) class SessionData`: `bars: Mapping[tuple[str, datetime], Bar]` (key `(symbol, bar_start)`), `quotes: Mapping[tuple[str, datetime], Quote]` (key `(symbol, ts)`), `rejects: tuple[Reject, ...]`, `n_quotes_skipped: int`.
  - `class BarSource(Protocol)`: `feed: str`; `open_session(self, session_date: date, symbols: tuple[str, ...], screen_id: str) -> SessionData`.
  - `class FeedAgent`: `__init__(self, source: BarSource, session_open_et: str = "09:30", session_close_et: str = "16:00", name: str = "feed")`; `subscribes = (Candidates, ClockTick)`; `on_event(self, event: Event) -> tuple[Event, ...]`; property `rejects -> tuple[Reject, ...]` (empty before `Candidates`).
- Behaviour:
  - `Candidates`: a second one raises `ValueError`. Otherwise `D = et_date(c.ts)`, `data = source.open_session(D, c.symbols, c.screen_id)`, keep `symbols = c.symbols` and the UTC bounds `et_to_utc(D, open)`, `et_to_utc(D, close)`.
  - `ClockTick` at `T`: no session → `()`. `bs = T − 1 min`; unless `open_utc <= bs < close_utc` → `()`. Else for `sym` in `symbols` (the sorted tuple): `data.bars.get((sym, bs))` or `Gap(ts=T, symbol=sym, missing_bar_start=bs)`; then `data.quotes.get((sym, T))` if present. Return the tuple.

- [ ] **Step 1: Write the failing tests.** `FakeSource` builds `SessionData` from a dict of `(symbol, "HH:MM") -> price` using `make_bar(ts=..., bar_start=..., symbol=..., feed="sip")` and records `open_session` calls. `cands(*syms)` = `Candidates(ts=et_to_utc(D, "09:25"), screen_id="screen:abc:2024-03-04", symbols=syms)`.

```python
def test_tick_releases_bar_for_minute_just_closed() -> None:
    agent = _agent({("A", "09:30"): 2.0}); agent.on_event(cands("A"))
    (bar,) = agent.on_event(tick("09:31"))
    assert isinstance(bar, Bar) and bar.bar_start == et_to_utc(D, "09:30")

def test_bar_ts_is_bar_start_plus_one_minute() -> None:
    agent = _agent({("A", "09:30"): 2.0}); agent.on_event(cands("A"))
    (bar,) = agent.on_event(tick("09:31"))
    assert bar.ts == bar.bar_start + timedelta(minutes=1) == et_to_utc(D, "09:31")

def test_missing_minute_emits_gap_not_filled_bar() -> None:
    agent = _agent({("A", "09:30"): 2.0}); agent.on_event(cands("A"))
    agent.on_event(tick("09:31"))
    (gap,) = agent.on_event(tick("09:32"))
    assert isinstance(gap, Gap) and gap.missing_bar_start == et_to_utc(D, "09:31") and gap.ts == et_to_utc(D, "09:32")

def test_every_candidate_minute_gets_exactly_one_bar_or_gap() -> None:
    agent = _agent({("A", "09:30"): 2.0, ("B", "15:59"): 3.0}); agent.on_event(cands("A", "B"))
    out = [e for t in clock_ticks(et_to_utc(D, "09:25"), et_to_utc(D, "16:02")) for e in agent.on_event(t)]
    for sym in ("A", "B"):
        starts = [e.bar_start if isinstance(e, Bar) else e.missing_bar_start for e in out if e.symbol == sym]
        assert len(starts) == 390 == len(set(starts))

def test_nothing_emitted_before_candidates() -> None:
    assert _agent({("A", "09:30"): 2.0}).on_event(tick("09:31")) == ()

def test_nothing_emitted_outside_session() -> None:
    agent = _agent({("A", "09:29"): 2.0, ("A", "16:00"): 2.0}); agent.on_event(cands("A"))
    assert agent.on_event(tick("09:30")) == () and agent.on_event(tick("16:01")) == ()

def test_output_order_sorted_symbol_then_bar_then_quote() -> None:
    agent = _agent({("A", "09:34"): 2.0, ("B", "09:34"): 3.0}, quotes={("A", "09:35"), ("B", "09:35")})
    agent.on_event(cands("A", "B"))
    assert [(type(e).__name__, e.symbol) for e in agent.on_event(tick("09:35"))] == [
        ("Bar", "A"), ("Quote", "A"), ("Bar", "B"), ("Quote", "B")]

def test_second_candidates_raises() -> None:
    agent = _agent({}); agent.on_event(cands("A"))
    with pytest.raises(ValueError):
        agent.on_event(cands("A"))
```

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest tests/test_intraday_data_feed.py -v` → new tests FAIL (import error).
- [ ] **Step 3: Implement `feed.py`.**
- [ ] **Step 4: Run to verify pass** — same command → 13 passed.
- [ ] **Step 5: Commit** — `feat(intraday-data): FeedAgent releases one Bar or Gap per candidate minute` + trailer (#21).

---

### Task 9: ReplayFeed (cached SIP minutes to the traded basis)

**Refs:** #21

**Files:**
- Create: `research/intraday/data/replay_feed.py`
- Test: `tests/test_intraday_data_feed.py` (append)

**Interfaces:**
- Consumes: `cache.read_daily`, `cache.read_minute_day`, `cache.minute_cache_membership` (Task 3); `basis.price_factor`, `volume_factor`, `factor_consistent` (Task 2); `read_quote_summaries`, `synthetic_quotes`, `Reject` (Task 7); `SessionData`, `BarSource` (Task 8); `check_holdout`, `minute_index_to_utc`.
- Produces: `class ReplayFeed` (a `BarSource`): `feed = "sip"`; `__init__(self, data_root: Path = DEFAULT_DATA_ROOT, volume_root: Path = DEFAULT_VOLUME_DAILY_ROOT, unadjust_tolerance: float = 0.02, holdout_start: date = HOLDOUT_START, unseal: bool = False, session_open_et: str = "09:30", session_close_et: str = "16:00", quote_size_scale: float = 1.0)`; `open_session(self, session_date: date, symbols: tuple[str, ...], screen_id: str) -> SessionData`.
- `open_session` algorithm, per symbol in sorted order, first matching rule wins:
  1. `check_holdout` (raises before any read).
  2. Load `minute = read_minute_day(D, symbols)`; a duplicate `(symbol, minute_et)` raises `ValueError`. Load `daily = read_daily(D, D + 1 day)` indexed by symbol; `f = price_factor(close_unadj, close)`, `f_open = price_factor(open_unadj, open)`, `f_v = price_factor(close_unadj_v, close_v)`, `s = volume_factor(volume_v, volume_unadj_v)`.
  3. Zero minute rows → `Reject("coverage")` with `reason="requested_no_rows"` if the symbol is in `minute_cache_membership(D)`, else `reason="not_requested"`. A symbol with rows always proceeds, whatever the membership file says.
  4. No daily row or non-finite `f` → `Reject("no_daily")`.
  5. `not factor_consistent(f_open, f, tol)` → `Reject("unadjust", reason="factor_mismatch")`. First session row (smallest `minute_et >= session_open_et`) has `abs(open × f / open_unadj − 1) > tol` → `Reject("unadjust", reason="open_mismatch")` (no session row → no check, decision D5).
  6. Non-finite `s` or `not factor_consistent(f_v, f, tol)` → `Reject("no_volume_factor")` (decision D1).
  7. Otherwise: rows with `session_open_et <= minute_et < session_close_et` become `Bar(ts=bs + 1 min, symbol, bar_start=bs, open/high/low/close/vwap = float(x × f), volume=int(round(volume / s)), trade_count=int(trade_count), feed="sip")`, `bs` from `minute_index_to_utc`. A contract violation propagates (bad cache voids the run).
  Quotes: `synthetic_quotes(read_quote_summaries(D, kept_symbols), D, quote_size_scale)` for symbols with no reject only.

- [ ] **Step 1: Write the failing tests.** `_root(tmp_path, **symbols)` writes a synthetic day with `synthetic.py` and returns `tmp_path` (volume source under `tmp_path / "v"`). Per symbol, keyword defaults: minute rows 09:25–16:05 at `price` (default 2.0, or the `closes` list from 09:30), `volume` 1000.0; the `D` daily row has adjusted `open` = the 09:30 minute open and `close` = the 15:59 minute close, raw = adjusted × `f` (default 1.0) for close and × `f_open` (default `f`) for open; the volume-source row has `close_v = close`, `close_unadj_v = close × f_v` (default `f`), `volume_unadj_v = volume_v / s` (default `s = f`). `no_daily=True` omits the daily row; `bad_low` / `duplicate_minute` corrupt one minute row; `requested` defaults to the symbols written; `quotes={sym: window}` writes one summary row. `_run(feed, symbols)` drives `FeedAgent(feed)` with `Candidates` then `clock_ticks(09:25, 16:02)` and returns the emitted events.

```python
def test_unadjust_scales_prices_by_factor_and_volume_inversely(tmp_path) -> None:
    root = _root(tmp_path, A=dict(f=0.1, s=0.1, price=20.0, volume=500.0))      # a 1:10 reverse split
    bars = ReplayFeed(root, root / "v").open_session(D, ("A",), SID).bars
    b = bars[("A", et_to_utc(D, "09:30"))]
    assert b.close == pytest.approx(2.0) and b.volume == 5000

def test_dividend_day_volume_unchanged(tmp_path) -> None:
    root = _root(tmp_path, A=dict(f=1.345, s=1.0, price=5.0, volume=1234.0))
    b = ReplayFeed(root, root / "v").open_session(D, ("A",), SID).bars[("A", et_to_utc(D, "09:30"))]
    assert b.volume == 1234 and b.close == pytest.approx(5.0 * 1.345)

def test_unadjusted_first_open_matches_daily_open_unadj(tmp_path) -> None:
    root = _root(tmp_path, A=dict(f=0.5, price=4.0))
    data = ReplayFeed(root, root / "v").open_session(D, ("A",), SID)
    assert data.bars[("A", et_to_utc(D, "09:30"))].open == pytest.approx(_daily_open_unadj(root, "A"))
    assert data.rejects == ()

def test_factor_mismatch_symbol_day_becomes_gaps(tmp_path) -> None:
    root = _root(tmp_path, A=dict(f=0.5, f_open=0.4))                          # open and close factors disagree
    events, agent = _run(ReplayFeed(root, root / "v"), ("A",))
    assert all(isinstance(e, Gap) for e in events if e.symbol == "A") and len(events) == 390
    assert [(r.kind, r.reason) for r in agent.rejects] == [("unadjust", "factor_mismatch")]

def test_missing_daily_row_symbol_day_becomes_gaps(tmp_path) -> None:
    root = _root(tmp_path, A=dict(no_daily=True))
    events, agent = _run(ReplayFeed(root, root / "v"), ("A",))
    assert {type(e) for e in events} == {Gap} and agent.rejects[0].kind == "no_daily"

def test_volume_factor_source_disagreement_becomes_gaps(tmp_path) -> None:
    root = _root(tmp_path, A=dict(f=0.5, f_v=0.25))
    events, agent = _run(ReplayFeed(root, root / "v"), ("A",))
    assert {type(e) for e in events} == {Gap} and agent.rejects[0].kind == "no_volume_factor"

def test_intraday_returns_unchanged_by_unadjust(tmp_path) -> None:
    root = _root(tmp_path, A=dict(f=0.05, s=0.05, closes=[10.0, 11.0, 9.9]))
    bars = sorted(ReplayFeed(root, root / "v").open_session(D, ("A",), SID).bars.values(), key=lambda b: b.bar_start)
    assert [b.close / a.close for a, b in zip(bars, bars[1:])] == pytest.approx([1.1, 0.9])

def test_zero_row_candidate_is_coverage_reject_not_only_gaps(tmp_path) -> None:
    root = _root(tmp_path, A=dict(), requested=("A", "B"))                     # B requested, no rows
    events, agent = _run(ReplayFeed(root, root / "v"), ("A", "B"))
    assert sum(1 for e in events if e.symbol == "B" and isinstance(e, Gap)) == 390
    assert [(r.kind, r.symbol, r.reason) for r in agent.rejects] == [("coverage", "B", "requested_no_rows")]

def test_coverage_rejects_counted_per_arm(tmp_path) -> None:
    root = _root(tmp_path, A=dict(), requested=("A", "B"))
    a = ReplayFeed(root, root / "v").open_session(D, ("A", "B"), "screen:abc:2024-03-04").rejects
    b = ReplayFeed(root, root / "v").open_session(D, ("A", "B", "C"), "no_selection:def:2024-03-04").rejects
    assert counts_by_arm(a + b, "coverage") == {"no_selection": 2, "screen": 1}

def test_quotes_suppressed_for_rejected_symbol_day(tmp_path) -> None:
    root = _root(tmp_path, A=dict(no_daily=True), quotes={"A": "decision"})
    assert ReplayFeed(root, root / "v").open_session(D, ("A",), SID).quotes == {}

def test_holdout_date_refused_without_unseal(tmp_path) -> None:
    with pytest.raises(SealedHoldoutError):
        ReplayFeed(tmp_path, tmp_path / "v").open_session(date(2025, 8, 7), ("A",), SID)

def test_invalid_cached_bar_raises(tmp_path) -> None:
    root = _root(tmp_path, A=dict(bad_low=True))                               # low > min(open, close)
    with pytest.raises(ValueError, match="Bar.low"):
        ReplayFeed(root, root / "v").open_session(D, ("A",), SID)

def test_duplicate_cached_bar_raises(tmp_path) -> None:
    root = _root(tmp_path, A=dict(duplicate_minute="09:31"))
    with pytest.raises(ValueError, match="duplicate"):
        ReplayFeed(root, root / "v").open_session(D, ("A",), SID)

def test_emitted_bar_fields_are_builtin_types(tmp_path) -> None:
    root = _root(tmp_path, A=dict(f=0.5))
    b = next(iter(ReplayFeed(root, root / "v").open_session(D, ("A",), SID).bars.values()))
    assert type(b.volume) is int and type(b.trade_count) is int and type(b.bar_start) is datetime
    assert type(b.open) is float

def test_replay_feed_is_deterministic(tmp_path) -> None:
    root = _root(tmp_path, A=dict(), B=dict(f=0.5), quotes={"A": "entry"})
    h = [run_replay([ScreenlessCandidates(("A", "B")), FeedAgent(ReplayFeed(root, root / "v"))],
                    [clock_ticks(et_to_utc(D, "09:25"), et_to_utc(D, "16:02"))], tmp_path / f"log{i}.jsonl")
         for i in range(2)]
    assert h[0] == h[1]
```

(`ScreenlessCandidates` is a 5-line test agent emitting a fixed `Candidates` on the 09:25 tick; `SID = "screen:abc:2024-03-04"`.)

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest tests/test_intraday_data_feed.py -v` → new tests FAIL (import error).
- [ ] **Step 3: Implement `replay_feed.py`.** Compute the factor columns and the per-symbol verdict with vectorised frame ops; the only row loop is the `Bar` construction over kept rows (`itertuples`).
- [ ] **Step 4: Run to verify pass** — same command → all pass. Confirm each of the 20 names in `feed.md`'s Test plan greps once in `tests/test_intraday_data_feed.py`.
- [ ] **Step 5: Commit** — `feat(intraday-data): ReplayFeed un-adjusts cached SIP minutes with split-only volume` + trailer (#21). Run the ownership check first.

---

### Task 10: Real-data replay of a development day

**Refs:** #20, #21

**Files:**
- Test: `tests/test_intraday_data_real.py` (create)

**Interfaces:**
- Consumes: `ScreenerAgent`, `ScreenerConfig`, `ReplayScreenInputs`, `FeedAgent`, `ReplayFeed`, `counts_by_arm`, `DEFAULT_DATA_ROOT`, `DEFAULT_VOLUME_DAILY_ROOT`; engine `run_replay`, `clock_ticks`; `research.intraday.ops.recorder.read_log` (tests may import it; the data package never does).
- Produces: nothing new in the package.

- [ ] **Step 1: Write the tests.** Module-level `pytestmark = pytest.mark.skipif(not (DEFAULT_DATA_ROOT / "minute").is_dir() or not DEFAULT_VOLUME_DAILY_ROOT.is_dir(), reason="intraday caches not present")`. `D = et_date(T_DEV)` (2024-03-04; measured on it: 105 cached symbols, 5 symbol-days fail the checks — 2 `open_mismatch`, 3 `no_volume_factor` — and BRSP and ENVB pass with a 10:00 bar). `read_log(path)` yields `Envelope`s; `_summarise` reads `env.event`. `_agents(arm)` builds a fresh screener (`ScreenerConfig(arm=arm)`) and feed per run. Ticks `clock_ticks(et_to_utc(D, "09:25"), et_to_utc(D, "16:02"))`.

```python
def test_real_day_replay_twice_identical_hash(tmp_path) -> None:
    h = [run_replay(_agents("no_selection"), [_ticks()], tmp_path / f"r{i}.jsonl") for i in range(2)]
    assert h[0] == h[1]

def test_real_day_every_candidate_minute_has_one_bar_or_gap(tmp_path) -> None:
    run_replay(_agents("no_selection"), [_ticks()], tmp_path / "r.jsonl")
    cands, per_symbol = _summarise(read_log(tmp_path / "r.jsonl"))
    assert len(cands.symbols) > 0
    assert all(per_symbol[s] == 390 for s in cands.symbols)

def test_real_dividend_and_split_volume_conversion() -> None:
    data = ReplayFeed().open_session(D, ("BRSP", "ENVB"), "t:x:2024-03-04")
    raw = _cached_minutes(D, ("BRSP", "ENVB"))           # straight from minute.parquet
    brsp = data.bars[("BRSP", et_to_utc(D, "10:00"))]
    assert brsp.volume == int(raw["BRSP"]["10:00"].volume)               # s = 1, no rescale
    assert brsp.close == pytest.approx(raw["BRSP"]["10:00"].close * 6.60 / 4.907, rel=1e-6)
    envb = data.bars[("ENVB", et_to_utc(D, "10:00"))]
    assert envb.volume == pytest.approx(raw["ENVB"]["10:00"].volume * 1_403_699 / 7798, abs=1)

def test_real_bars_agree_with_raw_quotes(tmp_path) -> None:
    data = ReplayFeed().open_session(D, _requested(D), "t:x:2024-03-04")
    pairs = _close_0934_vs_decision_mid(data)                            # non-rejected symbols with both
    assert len(pairs) >= 5 and all(abs(c / m - 1) < 0.05 for c, m in pairs)

def test_real_holdout_date_refused() -> None:
    with pytest.raises(SealedHoldoutError):
        ReplayFeed().open_session(date(2025, 8, 7), ("BRSP",), "t:x:2025-08-07")
```

- [ ] **Step 2: Run** — `$PY -m pytest tests/test_intraday_data_real.py -v -s` → 5 passed (or 5 skipped where the caches are absent). The bodies exercise only code from Tasks 1–9, so a failure here is a defect in those tasks: fix it with `superpowers:systematic-debugging`, never by editing the test. Print, and copy into the commit body: candidates per arm (`screen`, `no_selection`), `n_removed_by_cache`, reject counts by kind, and the unadjust-reject rate (rejects ÷ candidate symbol-days) on `D` — the rate `feed.md` asks R16 to report.
- [ ] **Step 3: Full verification** — `$PY -m pytest tests/ -q` → all pass; `$PY scripts/check_ownership.py --agent intraday-data --base research/intraday` → no output; `git status --short` shows only `research/intraday/data/` and `tests/test_intraday_data*`.
- [ ] **Step 4: Commit** — `test(intraday-data): one development day replays twice with identical hashes` + trailers (#20, #21), body = the Step 2 numbers.
