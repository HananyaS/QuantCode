"""Tests for research/penny/eval/benchmark.py.

The bar a strategy must clear (PREREGISTRATION_R17 §5): a Sharpe above SPY's
and QQQ's on the same sessions and alpha vs SPY with a 95% lower bound above
zero. These tests pin the arithmetic that bar rests on: a benchmark compared
with itself has zero alpha, unit beta and zero Sharpe difference; a planted
alpha is recovered inside its interval; a realised-only curve is refused.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.penny.eval.benchmark import (
    alpha_beta, assert_marked, cagr, compare, ledoit_wolf_sharpe_diff, max_drawdown, sharpe,
)

N = 1000
DAYS = pd.bdate_range("2020-01-02", periods=N)


def _market(seed=0):
    rng = np.random.default_rng(seed)
    spy = rng.normal(0.0004, 0.012, N)
    qqq = 1.2 * spy + rng.normal(0.0001, 0.006, N)
    return pd.DataFrame({"SPY": spy, "QQQ": qqq}, index=DAYS)


def test_point_metrics():
    r = np.r_[0.10, -0.50, 0.20]
    assert max_drawdown(r) == pytest.approx(-0.50)
    assert cagr(np.full(252, 0.001)) == pytest.approx(1.001 ** 252 - 1)
    x = np.random.default_rng(1).normal(0.001, 0.01, 500)
    assert sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(252))


def test_realised_only_curve_is_refused():
    r = np.zeros(300)
    r[::20] = 0.02                                     # P&L only on release days
    with pytest.raises(ValueError, match="realised-only"):
        assert_marked(r)
    assert_marked(np.random.default_rng(0).normal(0, 0.01, 300))


def test_benchmark_against_itself():
    m = _market()
    ab = alpha_beta(m["SPY"].to_numpy(), m["SPY"].to_numpy())
    assert ab["alpha_ann"] == pytest.approx(0.0, abs=1e-12)
    assert ab["beta"] == pytest.approx(1.0)
    lw = ledoit_wolf_sharpe_diff(m["SPY"].to_numpy(), m["SPY"].to_numpy())
    assert lw["diff"] == pytest.approx(0.0, abs=1e-12)


def test_planted_alpha_is_recovered():
    m = _market()
    rng = np.random.default_rng(5)
    a_daily = 0.10 / 252
    y = a_daily + 0.5 * m["SPY"].to_numpy() + rng.normal(0, 0.004, N)
    ab = alpha_beta(y, m["SPY"].to_numpy(), lags=10)
    assert ab["alpha_lo"] < 0.10 < ab["alpha_hi"]
    assert ab["alpha_lo"] > 0                          # 10%/yr at 6.3%/yr idio vol over 4 years
    assert ab["beta"] == pytest.approx(0.5, abs=0.05)


def test_compare_strategy_equal_to_spy():
    m = _market()
    strat = pd.DataFrame({0: m["SPY"], 1: m["SPY"]})
    rf = pd.Series(0.0, index=DAYS)
    c = compare(strat, m, rf, n_boot=200)
    t = c.tests
    assert t["sharpe_minus_SPY"]["point"] == pytest.approx(0.0, abs=1e-12)
    assert t["sharpe_minus_SPY"]["boot_ci"][0] == pytest.approx(0.0, abs=1e-12)
    assert t["alpha_vs_SPY"]["beta"] == pytest.approx(1.0)
    assert t["alpha_vs_SPY"]["alpha_ann"] == pytest.approx(0.0, abs=1e-10)
    assert c.n_days == N - 1


def test_compare_detects_a_better_strategy_and_pairs_the_bootstrap():
    m = _market()
    rng = np.random.default_rng(9)
    strat = pd.DataFrame({s: 0.0008 + 0.3 * m["SPY"] + rng.normal(0, 0.004, N) for s in range(3)},
                         index=DAYS)
    c = compare(strat, m, pd.Series(0.0, index=DAYS), n_boot=300)
    assert c.table.loc["strategy", "sharpe"] > c.table.loc["SPY", "sharpe"]
    assert c.tests["sharpe_minus_SPY"]["boot_ci"][0] > 0
    assert c.tests["alpha_vs_SPY"]["alpha_lo"] > 0


def test_compare_refuses_missing_benchmark_days():
    m = _market()
    strat = pd.DataFrame({0: m["SPY"] + 0.0001})
    with pytest.raises(ValueError, match="benchmark lacks"):
        compare(strat, m.iloc[:-5], pd.Series(0.0, index=DAYS), n_boot=50)
