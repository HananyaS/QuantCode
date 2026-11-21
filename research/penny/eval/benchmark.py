"""benchmark.py -- a strategy against SPY / QQQ buy-and-hold, on the same sessions.

The success bar (PREREGISTRATION_R17 §4-§5, the user's decision of 2026-11-22):
a Sharpe above both SPY's and QQQ's over the same window, and alpha vs SPY whose
95% lower bound is above zero. Total return and max drawdown are reported
alongside.

Everything here works on DAILY, MARKED-TO-MARKET returns. A realised-only curve
(positions held at cost until they close) has mostly zero days and an
artificially small variance; comparing its Sharpe with a marked index is
meaningless, so `assert_marked` refuses one.

Uncertainty
-----------
* Paired stationary bootstrap (Politis & Romano 1994): one index matrix,
  applied to every strategy seed and every benchmark, so differences are
  paired in time.
* Ledoit & Wolf (2008) HAC test for a Sharpe difference: delta method on the
  four moments, Parzen kernel, Andrews (1991) AR(1) plug-in bandwidth.
* Alpha/beta: OLS on daily excess returns with Newey-West (Bartlett) errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from research.penny.eval.battery import stationary_bootstrap_indices

TRADING_DAYS = 252
ZERO_DAY_LIMIT = 0.25     # a marked book is almost never exactly flat


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------
def sharpe(r: np.ndarray, axis: int = 0) -> np.ndarray:
    """Annualised Sharpe on raw daily returns (no risk-free), as every prior round."""
    r = np.asarray(r, dtype=float)
    sd = r.std(axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, r.mean(axis=axis) / sd * np.sqrt(TRADING_DAYS), np.nan)


def cagr(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    growth = float(np.prod(1.0 + r))
    years = len(r) / TRADING_DAYS
    return growth ** (1.0 / years) - 1.0 if growth > 0 and years > 0 else -1.0


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return float((eq / peak - 1.0).min())


def assert_marked(r: np.ndarray, name: str = "strategy") -> None:
    """Refuse a series that looks realised-only (too many exactly-flat days)."""
    r = np.asarray(r, dtype=float)
    share = float(np.mean(np.abs(r) < 1e-12))
    if share > ZERO_DAY_LIMIT:
        raise ValueError(f"{name}: {share:.0%} of days are exactly flat -- this looks like a "
                         "realised-only equity curve; mark positions to market first")


# --------------------------------------------------------------------------
# HAC machinery
# --------------------------------------------------------------------------
def _andrews_parzen_bandwidth(v: np.ndarray) -> float:
    """Andrews (1991) automatic bandwidth for the Parzen kernel, AR(1) plug-in."""
    num = den = 0.0
    for a in range(v.shape[1]):
        x = v[:, a]
        x0, x1 = x[:-1], x[1:]
        rho = float(np.dot(x0, x1) / np.dot(x0, x0)) if np.dot(x0, x0) > 0 else 0.0
        rho = float(np.clip(rho, -0.97, 0.97))
        s2 = float(np.var(x1 - rho * x0))
        num += 4 * rho ** 2 * s2 ** 2 / (1 - rho) ** 8
        den += s2 ** 2 / (1 - rho) ** 4
    alpha2 = num / den if den > 0 else 0.0
    return 2.6614 * (alpha2 * len(v)) ** 0.2


def _parzen(x: float) -> float:
    x = abs(x)
    if x <= 0.5:
        return 1 - 6 * x ** 2 + 6 * x ** 3
    if x <= 1.0:
        return 2 * (1 - x) ** 3
    return 0.0


def _hac(v: np.ndarray, kernel: str = "parzen", lags: Optional[int] = None) -> np.ndarray:
    """Long-run covariance of the (T, k) demeaned matrix `v`, scaled like Var * T."""
    t = len(v)
    v = v - v.mean(axis=0)
    out = v.T @ v / t
    if kernel == "parzen":
        bw = _andrews_parzen_bandwidth(v)
        for j in range(1, t):
            w = _parzen(j / bw) if bw > 0 else 0.0
            if w == 0.0:
                break
            g = v[j:].T @ v[:-j] / t
            out += w * (g + g.T)
    else:                                        # Bartlett / Newey-West
        lags = int(lags or 0)
        for j in range(1, lags + 1):
            w = 1.0 - j / (lags + 1.0)
            g = v[j:].T @ v[:-j] / t
            out += w * (g + g.T)
    return out


def ledoit_wolf_sharpe_diff(r1: np.ndarray, r2: np.ndarray) -> Dict[str, float]:
    """Ledoit & Wolf (2008) HAC inference on SR(r1) - SR(r2), annualised."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    t = len(r1)
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    s1, s2 = np.sqrt(g1 - mu1 ** 2), np.sqrt(g2 - mu2 ** 2)
    diff = mu1 / s1 - mu2 / s2
    grad = np.array([g1 / s1 ** 3, -g2 / s2 ** 3, -mu1 / (2 * s1 ** 3), mu2 / (2 * s2 ** 3)])
    v = np.column_stack([r1, r2, r1 ** 2, r2 ** 2])
    psi = _hac(v, "parzen")
    se = float(np.sqrt(max(grad @ psi @ grad, 0.0) / t))
    ann = np.sqrt(TRADING_DAYS)
    z = diff / se if se > 0 else np.nan
    return {"diff": float(diff * ann), "se": float(se * ann),
            "lo": float((diff - 1.96 * se) * ann), "hi": float((diff + 1.96 * se) * ann),
            "p": float(2 * sps.norm.sf(abs(z))) if np.isfinite(z) else np.nan}


def alpha_beta(y: np.ndarray, x: np.ndarray, lags: int = 60) -> Dict[str, float]:
    """OLS y = a + b x with Newey-West errors; alpha annualised by 252."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    u = y - X @ coef
    t = len(y)
    s = _hac(X * u[:, None], "bartlett", lags) * t
    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.diag(cov))
    a, b = coef
    return {"alpha_ann": float(a * TRADING_DAYS), "alpha_se_ann": float(se[0] * TRADING_DAYS),
            "alpha_lo": float((a - 1.96 * se[0]) * TRADING_DAYS),
            "alpha_hi": float((a + 1.96 * se[0]) * TRADING_DAYS),
            "alpha_t": float(a / se[0]) if se[0] > 0 else np.nan,
            "beta": float(b), "beta_se": float(se[1])}


def _ols_alpha_beta_rows(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised OLS over bootstrap rows: y, x are (B, T); returns (B, 2) [alpha, beta]."""
    xm, ym = x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True)
    b = ((x - xm) * (y - ym)).sum(axis=1) / ((x - xm) ** 2).sum(axis=1)
    a = ym[:, 0] - b * xm[:, 0]
    return np.column_stack([a, b])


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------
@dataclass
class Comparison:
    table: pd.DataFrame          # one row per series: Sharpe, CAGR, maxDD, total return
    tests: Dict[str, Dict]       # Sharpe differences, alpha/beta, bootstrap CIs
    n_days: int

    def to_dict(self) -> Dict:
        return {"n_days": self.n_days, "table": self.table.reset_index().to_dict("records"),
                "tests": self.tests}


def compare(strategy: pd.DataFrame, bench: pd.DataFrame, rf: pd.Series,
            mean_block: float = 60.0, n_boot: int = 2000, seed: int = 17,
            alpha_lags: int = 60) -> Comparison:
    """Strategy seeds (columns) against benchmark columns (SPY, QQQ, ...).

    All inputs are daily returns indexed by session; the first session (a
    strategy's zero day-0 return) is dropped. Point Sharpe for the strategy is
    the mean of per-seed Sharpes; drawdown and CAGR are means over seeds.
    """
    idx = strategy.index.intersection(bench.index)
    idx = idx[1:]
    missing = strategy.index[1:].difference(idx)
    if len(missing):
        raise ValueError(f"benchmark lacks {len(missing)} strategy sessions, e.g. {missing[:3].tolist()}")
    S = strategy.loc[idx].to_numpy(float)
    B = bench.loc[idx].to_numpy(float)
    f = rf.reindex(idx).ffill().fillna(0.0).to_numpy(float)
    for j in range(S.shape[1]):
        assert_marked(S[:, j], f"seed {strategy.columns[j]}")
    if not (np.isfinite(S).all() and np.isfinite(B).all()):
        raise ValueError("non-finite daily return in strategy or benchmark")
    names = list(bench.columns)
    spy = names.index("SPY")

    rows = {"strategy": {"sharpe": float(np.mean(sharpe(S))),
                         "sharpe_seed_sd": float(np.std(sharpe(S), ddof=1)) if S.shape[1] > 1 else 0.0,
                         "cagr": float(np.mean([cagr(S[:, j]) for j in range(S.shape[1])])),
                         "max_dd": float(np.mean([max_drawdown(S[:, j]) for j in range(S.shape[1])])),
                         "total_return": float(np.mean(np.prod(1 + S, axis=0) - 1))}}
    for j, n in enumerate(names):
        rows[n] = {"sharpe": float(sharpe(B[:, j])), "sharpe_seed_sd": 0.0,
                   "cagr": cagr(B[:, j]), "max_dd": max_drawdown(B[:, j]),
                   "total_return": float(np.prod(1 + B[:, j]) - 1)}
    table = pd.DataFrame(rows).T

    # paired stationary bootstrap
    ix = stationary_bootstrap_indices(len(idx), mean_block, n_boot, seed)
    sh_s = np.stack([sharpe(S[i], axis=0).mean() for i in ix])            # (n_boot,)
    sh_b = np.stack([sharpe(B[i], axis=0) for i in ix])                    # (n_boot, nb)
    pct = lambda x: [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]  # noqa: E731
    tests: Dict[str, Dict] = {"strategy_sharpe_ci": pct(sh_s)}
    savg = S.mean(axis=1)                                                  # seed-averaged return
    for j, n in enumerate(names):
        d = sh_s - sh_b[:, j]
        tests[f"sharpe_minus_{n}"] = {
            "point": float(rows["strategy"]["sharpe"] - rows[n]["sharpe"]),
            "boot_ci": pct(d), "boot_p_le0": float(np.mean(d <= 0)),
            "ledoit_wolf_seed_avg": ledoit_wolf_sharpe_diff(savg, B[:, j])}
    ex, mx = savg - f, B[:, spy] - f
    ab = alpha_beta(ex, mx, alpha_lags)
    boot_ab = _ols_alpha_beta_rows(ex[ix], mx[ix]) * np.array([TRADING_DAYS, 1.0])
    ab["alpha_boot_ci"] = pct(boot_ab[:, 0])
    ab["beta_boot_ci"] = pct(boot_ab[:, 1])
    tests["alpha_vs_SPY"] = ab
    for n in names:
        if n != "SPY":
            tests[f"beta_vs_{n}"] = alpha_beta(ex, B[:, names.index(n)] - f, alpha_lags)
    return Comparison(table, tests, len(idx))
