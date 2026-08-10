"""stats.py — inference that survives a search over many candidate filters.

The central hazard
------------------
Searching veto sets is a data-mining machine. With ~15 candidate vetoes there
are thousands of subsets, and under a pure null the best of them will look
excellent. Every statistic here exists to price that in.

Three corrections are applied, and they answer different questions:

* **Bootstrap CI, clustered by day.** Trades on the same day share a market
  factor; treating them as independent understates the standard error badly.
  The bootstrap resamples *days*, not trades.
* **Benjamini-Hochberg FDR.** Controls the expected proportion of false
  discoveries across the whole search. Reported alongside the number of
  hypotheses actually tested, because an FDR quoted without the search size is
  meaningless.
* **Deflated Sharpe Ratio.** Bailey & Lopez de Prado (2014), "The Deflated
  Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
  Non-Normality", *Journal of Portfolio Management* 40(5). Adjusts an observed
  Sharpe for the number of trials, and for the skew and kurtosis of the return
  distribution -- both extreme here, since penny-stock returns are violently
  right-skewed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class Estimate:
    """A point estimate with an uncertainty interval and its sample size."""

    mean: float
    lo: float
    hi: float
    n: int
    n_days: int
    p_value: float

    @property
    def excludes_zero(self) -> bool:
        return (self.lo > 0) or (self.hi < 0)


def cluster_bootstrap_mean(
    values: Sequence[float],
    groups: Sequence,
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Estimate:
    """Bootstrap the mean, resampling whole clusters (days).

    Trades taken on the same session are driven by a common market factor, so
    an i.i.d. bootstrap over trades would understate the standard error and
    manufacture significance. Resampling days preserves that dependence.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    mask = np.isfinite(v)
    v, g = v[mask], g[mask]

    if len(v) == 0:
        return Estimate(np.nan, np.nan, np.nan, 0, 0, np.nan)

    unique_days = np.unique(g)
    n_days = len(unique_days)
    by_day = {d: v[g == d] for d in unique_days}

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(unique_days, size=n_days, replace=True)
        sample = np.concatenate([by_day[d] for d in picked])
        means[b] = sample.mean() if sample.size else np.nan

    observed = float(v.mean())
    lo, hi = np.nanpercentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided bootstrap p-value for H0: mean = 0.
    centred = means - np.nanmean(means)
    p = float((np.abs(centred) >= abs(observed)).mean())

    return Estimate(observed, float(lo), float(hi), int(len(v)), int(n_days), p)


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses passing BH-FDR control at level ``q``."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * np.arange(1, n + 1) / n
    passing = ranked <= thresholds

    reject = np.zeros(n, dtype=bool)
    if passing.any():
        cutoff = np.max(np.where(passing)[0])
        reject[order[: cutoff + 1]] = True
    return reject


def sharpe_ratio(daily_returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualised Sharpe of a daily return series."""
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    daily_returns: Sequence[float], benchmark_sr: float = 0.0, periods_per_year: int = 252
) -> float:
    """P(true Sharpe > benchmark), correcting for skew and kurtosis.

    Bailey & Lopez de Prado (2012). Matters here because penny-stock returns are
    extremely right-skewed: a handful of enormous winners inflate the sample
    Sharpe in a way the Gaussian formula does not penalise.
    """
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return np.nan

    sr = r.mean() / r.std(ddof=1)                     # per-period
    sr_bench = benchmark_sr / np.sqrt(periods_per_year)
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))

    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    z = (sr - sr_bench) * np.sqrt(n - 1) / denom
    return float(sps.norm.cdf(z))


def deflated_sharpe_ratio(
    daily_returns: Sequence[float],
    n_trials: int,
    trial_sr_variance: Optional[float] = None,
    periods_per_year: int = 252,
) -> float:
    """PSR against the Sharpe a *search* of ``n_trials`` would produce by luck.

    The expected maximum Sharpe under the null grows with the number of
    strategies tried. Comparing an observed Sharpe against zero, after
    searching thousands of veto sets, is the classic backtest-overfitting
    error; this compares it against that expected maximum instead.
    """
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or n_trials < 1:
        return np.nan

    if trial_sr_variance is None or not np.isfinite(trial_sr_variance) or trial_sr_variance <= 0:
        # Fall back to the sample Sharpe's own sampling variance.
        trial_sr_variance = 1.0 / max(len(r) - 1, 1)

    euler = 0.5772156649015329
    if n_trials == 1:
        expected_max = 0.0
    else:
        z1 = sps.norm.ppf(1 - 1.0 / n_trials)
        z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
        expected_max = np.sqrt(trial_sr_variance) * ((1 - euler) * z1 + euler * z2)

    # expected_max is per-period; PSR expects an annualised benchmark.
    return probabilistic_sharpe_ratio(
        r, benchmark_sr=expected_max * np.sqrt(periods_per_year),
        periods_per_year=periods_per_year,
    )


def max_drawdown(equity: Sequence[float]) -> float:
    """Maximum peak-to-trough drawdown of an equity curve."""
    e = np.asarray(equity, dtype=float)
    if len(e) == 0:
        return np.nan
    peak = np.maximum.accumulate(e)
    return float(np.min(e / peak - 1.0))


def daily_equity_curve(
    trades: pd.DataFrame,
    return_col: str = "net_return",
    date_col: str = "date",
    max_positions_per_day: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate trades into an equal-weight daily portfolio.

    Capital is split evenly across that day's signals, so a day with forty
    signals does not silently take forty times the risk of a day with one.
    Without this, "mean trade return" and "portfolio return" diverge and the
    Sharpe becomes uninterpretable.
    """
    df = trades.dropna(subset=[return_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "n_trades", "ret", "equity"])

    if max_positions_per_day is not None:
        df = (
            df.sort_values([date_col, "symbol"])
            .groupby(date_col, group_keys=False)
            .head(max_positions_per_day)
        )

    daily = (
        df.groupby(date_col)[return_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "ret", "count": "n_trades"})
        .reset_index()
        .sort_values(date_col)
    )
    daily["equity"] = (1.0 + daily["ret"]).cumprod()
    return daily
