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


# ---- Task 5: pre-market conversion and ranking ------------------------------
from research.penny.features.premarket import premarket_features  # noqa: E402

from research.intraday.data.ranking import convert_premarket, rank_symbols  # noqa: E402


def _prior(**spec: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(spec, orient="index")[["close_unadj", "f", "f_v", "s"]]


def _pm(symbol: str, close: float = 2.0, volume: float = 100.0, minute: str = "09:00") -> pd.DataFrame:
    return pd.DataFrame({"symbol": [symbol], "minute_et": [minute], "open": [close], "high": [close],
                         "low": [close], "close": [close], "vwap": [close], "volume": [volume]})


def _pm_gaps(**gaps: float) -> pd.DataFrame:
    """Traded-basis pre-market bars whose close is 1 + gap (prev close 1.0)."""
    frames = [_pm(sym, close=1.0 + g) for sym, g in gaps.items()]
    if not frames:
        return _pm("X").iloc[0:0]
    return pd.concat(frames, ignore_index=True)


def _prev(close: float, *symbols: str) -> pd.Series:
    return pd.Series({s: close for s in symbols}, dtype=float)


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
