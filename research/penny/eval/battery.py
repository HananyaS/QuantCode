"""battery.py — the metrics Sharpe alone cannot provide.

Why this module exists
-----------------------
Fourteen rounds of this project have compared strategies on one number. That is
untenable here for three specific reasons, each of which a metric below answers.

**1. The Sharpe has been read as if it were precise.** It never had an interval.
§82 measured a seed-to-seed standard deviation of 0.75 at N=30 -- larger than
most of the differences the log draws conclusions from. `sharpe_se` gives the
non-normal standard error (Mertens 2002; Lo 2002) and `deflated_sharpe_ratio`
(already in `stats.py`, never once called with a real trial count) discounts it
for the size of the search that produced it.

**2. The variance argument may not apply at all.** §80 finding 4 -- "breadth is
irreducible" -- rests entirely on `SE = sigma/sqrt(N)`, which assumes a finite,
dominant second moment. These are microcap crash returns exited on a +100%
barrier at a 13-27% hit rate. `hill_tail_index` measures whether that assumption
survives contact with the data; if `alpha < 2` the sample mean does not converge
at `1/sqrt(N)` and the required breadth is a different number entirely.

**3. Nominal trade counts overstate the evidence.** At a 60-session hold with
daily entries, labels overlap almost completely. `average_uniqueness` (Lopez de
Prado, AFML ch. 4) converts n=2,000 overlapping trades into the number of
genuinely independent observations behind them.

Everything takes a **daily return series**, which is why `strategy/book.py` had
to start returning one.

References are given per function. Where a closed form is fragile to implement
(the expected-maximum-drawdown series), a Monte Carlo is used instead and the
one analytic case is pinned as a unit test.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


# ==========================================================================
# Tier 1 -- Sharpe, done honestly
# ==========================================================================
def sharpe_ratio(r: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_se(r: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Standard error of the Sharpe ratio without assuming normality.

        SE(SR) = sqrt( (1 + SR^2/2 - g3*SR + (g4-3)/4 * SR^2) / T )

    with `g3` the skewness and `g4` the (non-excess) kurtosis of the returns.
    Under normality (g3=0, g4=3) this collapses to the familiar
    `sqrt((1 + SR^2/2)/T)`.

    Mertens (2002), "Comments on variance of the IID estimator in Lo (2002)";
    Lo (2002), "The Statistics of Sharpe Ratios", *Financial Analysts Journal*.

    Returned on the SAME annualised scale as `sharpe_ratio`.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)                     # per-period, NOT annualised
    g3 = float(sps.skew(r, bias=False))
    g4 = float(sps.kurtosis(r, fisher=False, bias=False))
    var = (1.0 + 0.5 * sr**2 - g3 * sr + 0.25 * (g4 - 3.0) * sr**2) / T
    if var <= 0:
        return float("nan")
    return float(np.sqrt(var) * np.sqrt(periods_per_year))


def sharpe_autocorr_adjusted(r: np.ndarray, q: int = 5,
                             periods_per_year: int = TRADING_DAYS) -> float:
    """Sharpe corrected for serial correlation over a q-period aggregation.

        SR(q) = SR * q / sqrt( q + 2 * sum_{k=1}^{q-1} (q-k) * rho_k )

    Lo (2002), eq. (12). With positive autocorrelation the naive
    `sqrt(q)` scaling overstates the ratio -- which matters here because
    overlapping 60-session holds induce exactly that.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < q + 2 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    rho = [float(pd.Series(r).autocorr(lag=k)) for k in range(1, q)]
    rho = [0.0 if not np.isfinite(x) else x for x in rho]
    denom = q + 2.0 * sum((q - k) * rho[k - 1] for k in range(1, q))
    if denom <= 0:
        return float("nan")
    per_period = sr * q / np.sqrt(denom)
    # rescale from a q-period ratio back to the annualised convention
    return float(per_period * np.sqrt(periods_per_year / q))


def expected_max_sharpe(n_trials: int, trial_sr_sd: float) -> float:
    """Expected maximum Sharpe from `n_trials` independent draws.

        SR*_0 = sd(SR) * [ (1-g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e)) ]

    with `g` the Euler-Mascheroni constant. This is the benchmark the deflated
    Sharpe deflates against: the Sharpe a search of this size produces from pure
    noise. Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
    """
    if n_trials < 2 or trial_sr_sd <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(trial_sr_sd * ((1.0 - g) * z1 + g * z2))


def min_track_record_length(r: np.ndarray, benchmark_sr: float = 0.0,
                            alpha: float = 0.95,
                            periods_per_year: int = TRADING_DAYS) -> float:
    """Periods needed before the Sharpe is significantly above `benchmark_sr`.

        MinTRL = 1 + [1 - g3*SR + (g4-1)/4 * SR^2] * (Z_alpha / (SR - SR*))^2

    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    Returned in PERIODS (days here), not years. `inf` if the observed Sharpe
    does not exceed the benchmark at all.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    sr_star = benchmark_sr / np.sqrt(periods_per_year)
    if sr <= sr_star:
        return float("inf")
    g3 = float(sps.skew(r, bias=False))
    g4 = float(sps.kurtosis(r, fisher=False, bias=False))
    z = sps.norm.ppf(alpha)
    return float(1.0 + (1.0 - g3 * sr + 0.25 * (g4 - 1.0) * sr**2)
                 * (z / (sr - sr_star))**2)


# ==========================================================================
# Tier 2 -- the shape of the loss
# ==========================================================================
def sortino_ratio(r: np.ndarray, mar: float = 0.0,
                  periods_per_year: int = TRADING_DAYS) -> float:
    """Excess return over downside deviation. Sortino & Price (1994).

    The denominator uses the full sample in its root, not only the losing
    observations -- halving the count for the same numerator would flatter a
    strategy that rarely loses.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    downside = np.minimum(r - mar, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return float("nan")
    return float((r.mean() - mar) / dd * np.sqrt(periods_per_year))


def omega_ratio(r: np.ndarray, threshold: float = 0.0) -> float:
    """Probability-weighted gains over losses about a threshold.

        Omega(t) = integral_t^inf (1 - F(x)) dx / integral_-inf^t F(x) dx
                 = E[(r - t)+] / E[(t - r)+]

    Keating & Shadwick (2002), "A Universal Performance Measure". Unlike the
    Sharpe it uses the whole distribution rather than its first two moments,
    which is the point for a right-tailed barrier strategy.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    gain = np.maximum(r - threshold, 0.0).mean()
    loss = np.maximum(threshold - r, 0.0).mean()
    if loss == 0:
        return float("inf") if gain > 0 else float("nan")
    return float(gain / loss)


def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=float)
    if len(e) == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(np.min(e / peak - 1.0))


def calmar_ratio(r: np.ndarray, equity: Optional[np.ndarray] = None,
                 periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised return over the worst drawdown. Young (1991)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    eq = np.cumprod(1.0 + r) if equity is None else np.asarray(equity, float)
    years = len(r) / periods_per_year
    if eq[-1] <= 0 or years <= 0:
        return float("nan")
    ann = eq[-1] ** (1.0 / years) - 1.0
    mdd = abs(max_drawdown(eq))
    return float(ann / mdd) if mdd > 0 else float("nan")


def ulcer_index(equity: np.ndarray) -> float:
    """Root-mean-square drawdown. Martin & McCann (1989).

    Penalises depth AND duration, unlike max drawdown, which is a single worst
    moment and can be produced by one bad week in a decade.
    """
    e = np.asarray(equity, dtype=float)
    if len(e) == 0:
        return float("nan")
    dd = e / np.maximum.accumulate(e) - 1.0
    return float(np.sqrt(np.mean(dd**2)))


def martin_ratio(r: np.ndarray, equity: Optional[np.ndarray] = None,
                 periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    eq = np.cumprod(1.0 + r) if equity is None else np.asarray(equity, float)
    years = len(r) / periods_per_year
    if eq[-1] <= 0 or years <= 0:
        return float("nan")
    ann = eq[-1] ** (1.0 / years) - 1.0
    ui = ulcer_index(eq)
    return float(ann / ui) if ui > 0 else float("nan")


def value_at_risk(r: np.ndarray, alpha: float = 0.05) -> float:
    """Historical VaR: the alpha-quantile of the return distribution."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.quantile(r, alpha)) if len(r) else float("nan")


def conditional_var(r: np.ndarray, alpha: float = 0.05) -> float:
    """Expected shortfall: the mean of the worst `alpha` tail.

    Rockafellar & Uryasev (2000). Coherent where VaR is not, and the relevant
    object when the loss distribution has a tail rather than a cliff.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    cut = np.quantile(r, alpha)
    tail = r[r <= cut]
    return float(tail.mean()) if len(tail) else float(cut)


def rachev_ratio(r: np.ndarray, alpha: float = 0.05) -> float:
    """Expected gain in the best tail over expected loss in the worst.

    Biglova, Ortobelli, Rachev & Stoyanov (2004). A ratio built entirely from
    the tails, which is where a convex barrier strategy lives.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    upper = r[r >= np.quantile(r, 1.0 - alpha)]
    lower = r[r <= np.quantile(r, alpha)]
    if len(lower) == 0 or lower.mean() == 0:
        return float("nan")
    return float(upper.mean() / abs(lower.mean()))


def expected_max_drawdown(mu: float, sigma: float, n_periods: int,
                          n_sims: int = 20_000, seed: int = 0) -> float:
    """E[max drawdown] for a random walk with the same drift and volatility.

    Magdon-Ismail & Atiya (2004), "An Analysis of the Maximum Drawdown". Their
    closed form is a series whose two branches are easy to mis-implement, so
    this simulates instead; the one exactly known case -- driftless, where
    E[MDD] = 1.2533 * sigma * sqrt(T) -- is pinned as a unit test.

    Reported BESIDE the realised drawdown. A realised drawdown far worse than
    its own expectation falsifies the return model the Sharpe assumes, which no
    amount of Sharpe precision would reveal.
    """
    if sigma <= 0 or n_periods < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    steps = rng.normal(mu, sigma, size=(n_sims, n_periods))
    paths = np.cumsum(steps, axis=1)
    running_max = np.maximum.accumulate(paths, axis=1)
    return float(np.mean(np.max(running_max - paths, axis=1)))


def drawdown_durations(equity: np.ndarray) -> Dict[str, float]:
    """How long the book spends below its own high-water mark.

    For a 60-session hold, time under water is the number a person actually
    experiences; depth is what a backtest reports.
    """
    e = np.asarray(equity, dtype=float)
    if len(e) == 0:
        return {"max_underwater": float("nan"), "mean_underwater": float("nan"),
                "pct_underwater": float("nan")}
    under = e < np.maximum.accumulate(e) - 1e-15
    runs, cur = [], 0
    for u in under:
        if u:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return {
        "max_underwater": float(max(runs)) if runs else 0.0,
        "mean_underwater": float(np.mean(runs)) if runs else 0.0,
        "pct_underwater": float(under.mean()),
    }


def hill_tail_index(x: np.ndarray, k: Optional[int] = None,
                    tail: str = "right") -> Dict[str, float]:
    """Hill (1975) estimator of the tail index `alpha`.

        H_k = (1/k) * sum_{i=1..k} ln(X_(i) / X_(k+1)),   alpha = 1 / H_k

    over the `k` largest order statistics. `alpha < 2` means infinite variance:
    the sample mean then converges at `N^(1-1/alpha)` rather than `N^(-1/2)`,
    and every `sigma/sqrt(N)` breadth argument in this project is measuring the
    wrong thing.

    `k` defaults to `sqrt(n)`, a standard compromise between bias (too many
    order statistics, contaminated by the body) and variance (too few).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if tail == "left":
        x = -x
    x = x[x > 0]
    n = len(x)
    if n < 20:
        return {"alpha": float("nan"), "k": 0, "n_positive": n}
    if k is None:
        k = max(5, int(np.sqrt(n)))
    k = min(k, n - 1)
    xs = np.sort(x)[::-1]
    h = float(np.mean(np.log(xs[:k]) - np.log(xs[k])))
    return {"alpha": float(1.0 / h) if h > 0 else float("inf"),
            "k": int(k), "n_positive": int(n)}


# ==========================================================================
# Tier 3 -- breadth
# ==========================================================================
def effective_n(weights: np.ndarray) -> float:
    """Inverse Herfindahl: how many positions the book really has.

        N_eff = 1 / sum_i w_i^2

    Equal weights give exactly N; concentration reduces it.
    """
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w)]
    s = np.sum(w**2)
    return float(1.0 / s) if s > 0 else float("nan")


def correlation_adjusted_breadth(n: int, rho_bar: float) -> float:
    """Independent-bet equivalent of `n` correlated bets.

        BR_eff = N / (1 + (N-1) * rho_bar)

    Buckle (2004). With `rho_bar = 0.2` and `N = 60` the effective breadth is
    under 5 -- which is the quantitative version of "ten slots bought one
    position six times".
    """
    if n < 1:
        return float("nan")
    denom = 1.0 + (n - 1) * rho_bar
    return float(n / denom) if denom > 0 else float("inf")


def transfer_coefficient(ideal_w: np.ndarray, actual_w: np.ndarray) -> float:
    """Correlation between unconstrained and implemented active weights.

        IR = TC * IC * sqrt(BR)

    Clarke, de Silva & Thorley (2002), "Portfolio Constraints and the
    Fundamental Law of Active Management". `TC` is the fraction of the signal
    that survives long-only, equal-weight and slot-cap constraints. A veto book
    with no dispersion in expected return has no ideal tilt to transfer, so a
    near-zero TC here is a finding, not a bug.
    """
    a = np.asarray(ideal_w, dtype=float)
    b = np.asarray(actual_w, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def average_uniqueness(trades: pd.DataFrame, calendar: Sequence[pd.Timestamp],
                       start_col: str = "date",
                       end_col: str = "exit_date") -> Dict[str, float]:
    """Effective sample size once overlapping labels are accounted for.

        c_t = number of trades live at t
        u_i = mean over t in [start_i, end_i] of 1 / c_t
        effective sample = sum_i u_i

    Lopez de Prado, *Advances in Financial Machine Learning*, ch. 4. At a
    60-session hold with an entry a day, roughly 60 positions are live at once,
    so the effective sample is of order n/60 -- and every confidence interval
    computed on the nominal `n` is that much too narrow.
    """
    if trades is None or len(trades) == 0:
        return {"avg_uniqueness": float("nan"), "effective_n_trades": 0.0,
                "nominal_n_trades": 0.0, "mean_concurrency": float("nan")}
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(calendar)).unique()))
    idx = {d: i for i, d in enumerate(cal)}
    starts = pd.to_datetime(trades[start_col]).map(idx).to_numpy()
    ends = pd.to_datetime(trades[end_col]).map(idx).to_numpy()
    ok = np.isfinite(starts.astype(float)) & np.isfinite(ends.astype(float))
    starts, ends = starts[ok].astype(int), ends[ok].astype(int)
    if len(starts) == 0:
        return {"avg_uniqueness": float("nan"), "effective_n_trades": 0.0,
                "nominal_n_trades": 0.0, "mean_concurrency": float("nan")}

    conc = np.zeros(len(cal))
    np.add.at(conc, starts, 1.0)
    dec = np.zeros(len(cal) + 1)
    np.add.at(dec, ends + 1, 1.0)
    conc = np.cumsum(conc - dec[:len(cal)])
    conc = np.maximum(conc, 1e-12)

    inv_cum = np.concatenate([[0.0], np.cumsum(1.0 / conc)])
    span = (ends - starts + 1).astype(float)
    u = (inv_cum[ends + 1] - inv_cum[starts]) / np.maximum(span, 1.0)
    return {
        "avg_uniqueness": float(np.mean(u)),
        "effective_n_trades": float(np.sum(u)),
        "nominal_n_trades": float(len(u)),
        "mean_concurrency": float(np.mean(conc[conc > 1e-9])),
    }


# ==========================================================================
# Tier 4 -- is the backtest real
# ==========================================================================
def pbo_cscv(returns: pd.DataFrame, n_splits: int = 8,
             max_combinations: int = 1000) -> Dict[str, float]:
    """Probability of Backtest Overfitting by combinatorially symmetric CV.

    `returns` is (T periods x N configurations). Rows are cut into `S` disjoint
    blocks; over every way of choosing S/2 blocks as in-sample, the config with
    the best in-sample Sharpe is found and its OUT-of-sample rank recorded. With

        w = (rank of the IS winner among N, out of sample), in (0, 1]
        logit = ln( w / (1 - w) )

    PBO is the fraction of splits where the in-sample winner lands below the
    out-of-sample median. Bailey, Borwein, Lopez de Prado & Zhu (2017), "The
    Probability of Backtest Overfitting", *Journal of Computational Finance*.

    A PBO near 0.5 means the selection procedure has no skill -- picking the
    best cell of a sweep tells you nothing about which cell will be best next.
    """
    R = returns.dropna(axis=0, how="any")
    T, N = R.shape
    if N < 2 or T < 2 * n_splits:
        return {"pbo": float("nan"), "n_combinations": 0, "n_configs": int(N)}
    if n_splits % 2:
        n_splits -= 1
    blocks = np.array_split(np.arange(T), n_splits)
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(0)
        combos = [combos[i] for i in
                  rng.choice(len(combos), max_combinations, replace=False)]

    def _sr(block: np.ndarray) -> np.ndarray:
        sd = block.std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, block.mean(axis=0) / sd, -np.inf)

    vals = R.to_numpy(dtype=float)
    logits: List[float] = []
    for c in combos:
        is_rows = np.concatenate([blocks[i] for i in c])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits)
                                   if i not in c])
        best = int(np.argmax(_sr(vals[is_rows])))
        oos = _sr(vals[oos_rows])
        finite = np.isfinite(oos)
        if finite.sum() < 2 or not finite[best]:
            continue
        rank = float(sps.rankdata(oos[finite])[np.flatnonzero(finite) == best][0])
        w = rank / (finite.sum() + 1.0)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(w / (1.0 - w))))
    if not logits:
        return {"pbo": float("nan"), "n_combinations": 0, "n_configs": int(N)}
    lg = np.array(logits)
    return {"pbo": float((lg <= 0).mean()), "n_combinations": int(len(lg)),
            "n_configs": int(N), "median_logit": float(np.median(lg))}


def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int,
                                 seed: int = 0) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap index matrix.

    Geometric block lengths with mean `mean_block`, wrapped circularly. Unlike
    a day-clustered resample this preserves serial dependence WITHIN the return
    series, which is what a Sharpe or drawdown statistic depends on.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    starts = rng.integers(0, n, size=(n_boot, n))
    cont = rng.random((n_boot, n)) > p
    for b in range(n_boot):
        cur = starts[b, 0]
        for t in range(n):
            if t and cont[b, t]:
                cur = (cur + 1) % n
            else:
                cur = starts[b, t]
            idx[b, t] = cur
    return idx


def bootstrap_metric_ci(r: np.ndarray, fn, mean_block: float = 10.0,
                        n_boot: int = 1000, alpha: float = 0.05,
                        seed: int = 0) -> Tuple[float, float, float]:
    """(point, lo, hi) for any metric, via the stationary bootstrap."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return float("nan"), float("nan"), float("nan")
    idx = stationary_bootstrap_indices(len(r), mean_block, n_boot, seed)
    draws = np.array([fn(r[i]) for i in idx], dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 10:
        return float(fn(r)), float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(r)), float(lo), float(hi)


# ==========================================================================
# Tier 5 -- execution realism
# ==========================================================================
def amihud_illiquidity(returns: np.ndarray, dollar_volume: np.ndarray) -> float:
    """Mean |return| per dollar traded, scaled to price impact per $1M.

    Amihud (2002), "Illiquidity and stock returns". A direct, data-cheap proxy
    for how far a given order moves the print.
    """
    r = np.abs(np.asarray(returns, dtype=float))
    dv = np.asarray(dollar_volume, dtype=float)
    m = np.isfinite(r) & np.isfinite(dv) & (dv > 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(r[m] / dv[m]) * 1e6)


def corwin_schultz_spread(high: np.ndarray, low: np.ndarray) -> float:
    """Effective spread estimated from two consecutive daily high-low ranges.

    Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads from
    Daily High and Low Prices", *Journal of Finance*. Independent of the quote
    data `cost_model.py` is fitted on, so it is a genuine cross-check rather
    than a restatement.

    Returns the mean spread as a FRACTION of price; negative daily estimates
    are set to zero, as the paper prescribes.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if len(h) < 2 or len(h) != len(l):
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = (np.log(h[:-1] / l[:-1])**2) + (np.log(h[1:] / l[1:])**2)
        h2 = np.maximum(h[:-1], h[1:])
        l2 = np.minimum(l[:-1], l[1:])
        gamma = np.log(h2 / l2)**2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float("nan")
    return float(np.mean(np.maximum(s, 0.0)))


def roll_spread(prices: np.ndarray) -> float:
    """Spread from the negative serial covariance of price changes.

        S = 2 * sqrt(-cov(dP_t, dP_{t-1}))

    Roll (1984). Undefined when the covariance is positive, which happens
    whenever genuine autocorrelation swamps bid-ask bounce -- reported as NaN
    rather than clamped, because a clamped Roll estimate is a silent lie.
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) < 4:
        return float("nan")
    d = np.diff(p)
    cov = float(np.cov(d[:-1], d[1:])[0, 1])
    if cov >= 0:
        return float("nan")
    return float(2.0 * np.sqrt(-cov) / np.mean(p))


def square_root_impact(sigma_daily: float, participation: float,
                       y: float = 1.0) -> float:
    """Temporary impact as a fraction of price.

        dP/P = Y * sigma * sqrt(Q / V)

    Almgren, Thum, Hauptmann & Li (2005), "Direct Estimation of Equity Market
    Impact"; Torre & Ferrari (1997). `participation` is Q/V, the order as a
    fraction of the day's volume.
    """
    if participation <= 0 or sigma_daily <= 0:
        return 0.0
    return float(y * sigma_daily * np.sqrt(participation))


# ==========================================================================
# assembly
# ==========================================================================
def tearsheet(daily: pd.Series, equity: Optional[pd.Series] = None,
              trades: Optional[pd.DataFrame] = None,
              calendar: Optional[Sequence[pd.Timestamp]] = None,
              n_trials: int = 1, trial_sr_sd: float = 0.0,
              periods_per_year: int = TRADING_DAYS,
              bootstrap: bool = True,
              boot_block: float = 20.0) -> Dict[str, float]:
    """Every metric above, for one strategy configuration.

    `boot_block` is the stationary bootstrap's mean block length and it is NOT a
    cosmetic choice: the daily returns of an H-session book are autocorrelated
    out to roughly lag H, so the interval widens monotonically with the block
    until it reaches H and then stops. Measured on the N=60 development book
    (rho_1 = +0.215, rho_20 = +0.129, rho_60 = -0.009):

        block   1 -> [-0.02, +1.79]   width 1.81
        block   5 -> [-0.51, +2.41]   width 2.93
        block  20 -> [-1.27, +3.04]   width 4.31
        block  60 -> [-1.42, +3.59]   width 5.01

    A block far below H understates the uncertainty; the honest choice is the
    holding period itself. 20 is kept as the default only so previously computed
    tables remain comparable -- pass `boot_block=hold` for a new result.
    """
    r = np.asarray(daily, dtype=float)
    r = r[np.isfinite(r)]
    eq = (np.cumprod(1.0 + r) if equity is None
          else np.asarray(equity, dtype=float))

    sr = sharpe_ratio(r, periods_per_year)
    se = sharpe_se(r, periods_per_year)
    sr_star = expected_max_sharpe(n_trials, trial_sr_sd)

    out: Dict[str, float] = {
        # tier 1
        "sharpe": sr,
        "sharpe_se": se,
        "sharpe_lo": sr - 1.96 * se if np.isfinite(se) else float("nan"),
        "sharpe_hi": sr + 1.96 * se if np.isfinite(se) else float("nan"),
        "sharpe_autocorr_adj": sharpe_autocorr_adjusted(r, 5, periods_per_year),
        "expected_max_sharpe": sr_star,
        "sharpe_minus_expected_max": sr - sr_star,
        "min_track_record_days": min_track_record_length(
            r, benchmark_sr=0.0, periods_per_year=periods_per_year),
        # tier 2
        "sortino": sortino_ratio(r, 0.0, periods_per_year),
        "omega": omega_ratio(r, 0.0),
        "calmar": calmar_ratio(r, eq, periods_per_year),
        "ulcer_index": ulcer_index(eq),
        "martin": martin_ratio(r, eq, periods_per_year),
        "var_95": value_at_risk(r, 0.05),
        "cvar_95": conditional_var(r, 0.05),
        "rachev_95": rachev_ratio(r, 0.05),
        "max_drawdown": max_drawdown(eq),
        "expected_max_drawdown": -expected_max_drawdown(
            float(r.mean()), float(r.std(ddof=1)), len(r)),
        "skew": float(sps.skew(r, bias=False)) if len(r) > 2 else float("nan"),
        "kurtosis": (float(sps.kurtosis(r, fisher=False, bias=False))
                     if len(r) > 3 else float("nan")),
    }
    out.update(drawdown_durations(eq))
    out["mdd_vs_expected"] = (out["max_drawdown"] / out["expected_max_drawdown"]
                              if out["expected_max_drawdown"] else float("nan"))

    if trades is not None and len(trades):
        net = trades["net"].to_numpy(dtype=float) if "net" in trades else None
        if net is not None:
            # A profit target puts a POINT MASS at the barrier: with a +100%
            # target a large share of trades return exactly +1.00 minus cost.
            # Hill on that sample sees a wall of ties, drives its mean log
            # excess to zero and reports alpha -> inf, which reads as "no tail"
            # when what actually happened is "the tail was truncated by
            # construction". So the right tail is estimated only on gains BELOW
            # the barrier, and the truncated share is reported beside it.
            # The barrier must be detected on the GROSS return, which is
            # exactly the target for every name that touched it. Net returns
            # are the target minus a per-name cost, so they are NOT tied and a
            # tie test on `net` undercounts the point mass badly -- it finds
            # only the cheapest touchers.
            gross = (trades["gross"].to_numpy(dtype=float)
                     if "gross" in trades else net)
            top = np.max(gross) if len(gross) else np.nan
            at_barrier = (np.isclose(gross, top, rtol=1e-6)
                          if np.isfinite(top) else np.zeros(len(gross), bool))
            out["barrier_share"] = float(at_barrier.mean())
            free = net[(~at_barrier) & (net > 0)]
            out["tail_alpha_gains"] = (hill_tail_index(free)["alpha"]
                                       if len(free) >= 20 else float("nan"))
            out["tail_alpha_losses"] = hill_tail_index(net, tail="left")["alpha"]
            out["trade_mean"] = float(np.nanmean(net))
            out["trade_median"] = float(np.nanmedian(net))
            out["trade_sd"] = float(np.nanstd(net, ddof=1))
            out["trade_ir"] = (out["trade_mean"] / out["trade_sd"]
                               if out["trade_sd"] > 0 else float("nan"))
        if "weight" in trades:
            out["effective_n"] = effective_n(
                trades["weight"].to_numpy(dtype=float)
                / max(trades["weight"].sum(), 1e-12))
        if calendar is not None:
            out.update(average_uniqueness(trades, calendar))

    if bootstrap and len(r) >= 60:
        _, lo, hi = bootstrap_metric_ci(
            r, lambda x: sharpe_ratio(x, periods_per_year),
            mean_block=boot_block, n_boot=500)
        out["boot_block"] = float(boot_block)
        out["sharpe_boot_lo"] = lo
        out["sharpe_boot_hi"] = hi
    return out
