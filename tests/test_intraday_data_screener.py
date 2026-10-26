"""ScreenerAgent slice: causal universe, pre-market ranking, the agent (screener.md test plan)."""
from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from research.intraday.data import synthetic
from research.intraday.data.errors import MissingDataError
from research.intraday.data.universe import UniverseResult, causal_universe, intersect_cache

D = date(2024, 3, 4)
_SESSIONS = list(pd.bdate_range(end="2024-03-01", periods=20).date)
_RAW_CLOSE = 5.0
_SCREEN_DEFAULTS = dict(min_price=1.0, max_price=20.0, min_adv_usd=1_000_000.0, adv_window=20,
                        exclude_split_days=True, split_log_step_threshold=math.log(1.25),
                        max_prior_gap_days=5)


def _rows(symbol: str, sessions: list[date], **kw: float) -> list[dict]:
    return [synthetic.daily_row(symbol, d, **kw) for d in sessions]


def _daily(symbols: list[str], cheap: list[str] = (), f: dict[str, float] | None = None,
           last_date: date | None = None, stale: list[str] = (),
           young: dict[str, int] | None = None) -> pd.DataFrame:
    """20 sessions before D; every symbol's raw close is 5.0 (adjusted = 5.0 / f)."""
    f = f or {}
    sessions = _SESSIONS if last_date is None else list(
        pd.bdate_range(end=pd.Timestamp(last_date), periods=20).date)
    rows: list[dict] = []
    for sym in symbols:
        fs = f.get(sym, 1.0)
        rows += _rows(sym, sessions, close=_RAW_CLOSE / fs, f=fs)
    for sym in cheap:
        rows += _rows(sym, sessions, close=0.40)
    for sym in stale:
        older = list(pd.bdate_range(end=pd.Timestamp(sessions[-2]), periods=20).date)
        rows += _rows(sym, older, close=_RAW_CLOSE)
    for sym, n in (young or {}).items():
        rows += _rows(sym, sessions[-n:], close=_RAW_CLOSE)
    return _as_read(pd.DataFrame(rows))


def _as_read(rows: pd.DataFrame) -> pd.DataFrame:
    """The ``read_daily`` columns of synthetic rows (volume source == penny source)."""
    out = rows[["symbol", "date", "open", "close", "open_unadj", "close_unadj", "volume"]].copy()
    out["close_v"] = rows["close"]
    out["close_unadj_v"] = rows["close_unadj"]
    out["volume_v"] = rows["volume"]
    out["volume_unadj_v"] = rows["volume_unadj"]
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _daily_with(**spec: tuple[float, float]) -> pd.DataFrame:
    """symbol=(adjusted close, f)."""
    rows: list[dict] = []
    for sym, (close, f) in spec.items():
        rows += _rows(sym, _SESSIONS, close=close, f=f)
    return _as_read(pd.DataFrame(rows))


def _daily_dv(**spec: tuple[float, float]) -> pd.DataFrame:
    """symbol=(adjusted close x adjusted volume, raw close x volume_unadj); close 5.0, f 1."""
    rows: list[dict] = []
    for sym, (adj_dv, raw_dv) in spec.items():
        volume = adj_dv / 5.0
        rows += _rows(sym, _SESSIONS, close=5.0, volume=volume, s=volume / (raw_dv / 5.0))
    return _as_read(pd.DataFrame(rows))


def _universe(daily: pd.DataFrame, f_session: pd.Series | None = None,
              **overrides: object) -> UniverseResult:
    kw = {**_SCREEN_DEFAULTS, **overrides}
    fs = pd.Series(dtype=float) if f_session is None else f_session
    return causal_universe(daily, D, fs, **kw)


# ---- Task 4: causal universe ------------------------------------------------
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
