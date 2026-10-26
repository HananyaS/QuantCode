"""Pure-math leg cost == penny leg_cost_bps on a seeded grid, and much faster (R16 perf)."""
from __future__ import annotations

import itertools
import math
import random
import time

import pandas as pd

from research.intraday.exec.costs import leg_cost_bps_fast, leg_cost_bps_scalar
from research.penny.etl.quotes import leg_cost_bps

NAN, INF = float("nan"), float("inf")


def _pandas(on, sn, hs, sig, p, coef, pen) -> float:
    return float(leg_cost_bps(pd.Series([on]), pd.Series([sn]), pd.Series([hs]),
                              pd.Series([sig]), pd.Series([p]), coef, pen).iloc[0])


def _grid() -> list[tuple[float, ...]]:
    rng = random.Random(16)
    rows = []
    for _ in range(2_000):
        rows.append((rng.uniform(1.0, 50_000.0), rng.choice([rng.uniform(1.0, 50_000.0), 0.0, NAN]),
                     rng.uniform(0.5, 2_000.0), rng.uniform(0.0, 0.3),
                     rng.choice([rng.uniform(0.0, 0.05), rng.uniform(0.5, 3.0), 0.0]),
                     rng.uniform(0.0, 2.0), rng.uniform(0.0, 2.0)))
    edges = itertools.product([0.0, 100.0, 1e308], [0.0, -0.0, NAN, 1e-308, 50.0, 100.0, INF, -5.0],
                              [5.0, NAN], [0.0, -0.1, 0.05, NAN], [0.0, -1.0, 1.0, 2.0, 1e-12, NAN],
                              [0.0, 1.0], [0.0, 1.0])
    return rows + list(edges)


def test_fast_matches_pandas_on_seeded_grid() -> None:
    grid = _grid()
    assert len(grid) >= 2_000
    worst = 0.0
    for args in grid:
        want, got = _pandas(*args), leg_cost_bps_fast(*args)
        if math.isnan(want) or math.isnan(got):
            assert math.isnan(want) and math.isnan(got), args
        elif math.isinf(want) or math.isinf(got):
            assert want == got, args
        else:
            worst = max(worst, abs(got - want))
    assert worst == 0.0                                    # bit-identical, not just within 1e-9


def test_scalar_wrapper_uses_fast_path() -> None:
    assert leg_cost_bps_scalar(2_000.0, None, 50.0, 0.04, 0.01, 1.0, 1.0) == leg_cost_bps_fast(
        2_000.0, NAN, 50.0, 0.04, 0.01, 1.0, 1.0)
    t0 = time.perf_counter()
    for _ in range(1_000):
        leg_cost_bps_scalar(2_000.0, 1_000.0, 50.0, 0.04, 0.01, 1.0, 1.0)
    assert (time.perf_counter() - t0) / 1_000 < 50e-6       # pandas path is ~1 ms per call


def test_fast_is_at_least_20x_quicker() -> None:
    args = (2_000.0, 1_000.0, 50.0, 0.04, 0.01, 1.0, 1.0)
    n_slow, n_fast = 200, 20_000
    t0 = time.perf_counter()
    for _ in range(n_slow):
        _pandas(*args)
    slow = (time.perf_counter() - t0) / n_slow
    t0 = time.perf_counter()
    for _ in range(n_fast):
        leg_cost_bps_fast(*args)
    fast = (time.perf_counter() - t0) / n_fast
    assert slow / fast >= 20.0, (slow, fast)
