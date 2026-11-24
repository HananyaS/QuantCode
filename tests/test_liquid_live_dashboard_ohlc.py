"""dashboard.day_arrays carries the minute OHLCV a candlestick chart needs; stale day caches are refetched."""
import numpy as np
import pandas as pd
import pytest

from research.liquid.calendar import FULL_BARS
from research.liquid.data import FIELDS, Minutes
from research.liquid.live import dashboard as D


def minutes(n_days: int = 20, seed: int = 7) -> Minutes:
    rng = np.random.default_rng(seed)
    close = 400 * np.exp(np.cumsum(rng.normal(0, 3e-4, (n_days, FULL_BARS)), axis=1))
    opn = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    arrays = {f: close.copy() for f in FIELDS}
    arrays.update(open=opn, high=np.maximum(opn, close) * 1.0004, low=np.minimum(opn, close) * 0.9996,
                  volume=rng.integers(1_000, 50_000, (n_days, FULL_BARS)).astype(float))
    prev = np.concatenate([[np.nan], close[:-1, -1]])
    days = pd.bdate_range("2026-10-01", periods=n_days)
    return Minutes("SPY", days, np.full(n_days, FULL_BARS), prev_close=prev, valid=np.ones(n_days, bool), **arrays)


def test_day_arrays_has_ohlcv_matching_the_bars() -> None:
    m = minutes()
    i = len(m.dates) - 1
    a = D.day_arrays(m, i)
    for k in ("open", "high", "low", "close", "volume", "ub", "lb", "vwap"):
        assert len(a[k]) == a["n_bars"] == FULL_BARS, k
    assert a["open"] == pytest.approx(np.round(m.open[i], 2).tolist())
    assert a["volume"] == pytest.approx(m.volume[i].tolist())
    assert all(h >= max(o, c) and lo <= min(o, c)
               for o, h, lo, c in zip(a["open"], a["high"], a["low"], a["close"]))
    assert set(D.OHLCV) <= set(D.INTRADAY_COLS)
    assert D.table([{"day": "2026-10-28", **a}], D.INTRADAY_COLS)["columns"] == D.INTRADAY_COLS


def test_cache_is_current_only_with_every_ohlcv_array() -> None:
    a = D.day_arrays(minutes(), 19)
    assert D.cache_is_current({"spy_ret": 0.001, "arrays": a})
    old = {k: v for k, v in a.items() if k not in D.OHLCV}        # a day cached before OHLCV was added
    assert not D.cache_is_current({"spy_ret": 0.001, "arrays": old})
    assert not D.cache_is_current({})
