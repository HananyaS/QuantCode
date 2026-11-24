"""dashboard.market_day: one paper session's returns and chart arrays, from a fake feed (no network)."""
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from research.liquid.calendar import FULL_BARS
from research.liquid.data import FIELDS, Minutes
from research.liquid.live import dashboard as D

Session = namedtuple("Session", "day n_bars")


def synthetic(symbol: str, days: list[pd.Timestamp], seed: int) -> Minutes:
    rng = np.random.default_rng(seed)
    n = len(days)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 4e-4, (n, FULL_BARS)), axis=1)) * (1 + 0.01 * np.arange(n))[:, None]
    arrays = {f: close.copy() for f in FIELDS}
    arrays["volume"] = np.full((n, FULL_BARS), 1000.0)
    arrays["open"] = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    prev = np.concatenate([[np.nan], close[:-1, -1]])
    return Minutes(symbol, pd.DatetimeIndex(days), np.full(n, FULL_BARS), prev_close=prev, valid=np.ones(n, bool), **arrays)


class FakeFeed:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def history(self, symbol: str, sessions: list) -> Minutes:
        self.calls.append((symbol, len(sessions)))
        return synthetic(symbol, [pd.Timestamp(d) for d, _ in sessions], seed=len(sessions))


def calendar(n: int) -> list[Session]:
    return [Session(d, FULL_BARS) for d in pd.bdate_range("2026-10-01", periods=n)]


def test_market_day_reads_the_window_and_returns_its_own_session() -> None:
    cal = calendar(40)
    day = f"{cal[30].day:%Y-%m-%d}"
    feed = FakeFeed()
    out = D.market_day(feed, cal, day, history_sessions=20, symbol="SPY")
    assert feed.calls == [("SPY", 21), ("QQQ", 2)]                 # never a session after `day`
    m = synthetic("SPY", [s.day for s in cal[10:31]], seed=21)
    assert out["spy_ret"] == pytest.approx(m.close[-1, -1] / m.prev_close[-1] - 1, abs=1e-7)
    a = out["arrays"]
    assert a["n_bars"] == FULL_BARS and len(a["close"]) == len(a["ub"]) == len(a["vwap"]) == FULL_BARS
    assert all(u is None or lo is None or u > lo for u, lo in zip(a["ub"], a["lb"]))
    assert D.wrap({"x": out})                                       # JSON-safe


def test_market_day_refuses_a_day_missing_from_the_calendar() -> None:
    cal = calendar(40)
    with pytest.raises(AssertionError):
        D.market_day(FakeFeed(), cal, "2026-10-04", history_sessions=20, symbol="SPY")   # a Sunday
