"""conditional_vol.py — causal, forward-looking one-step-ahead conditional
variance forecasts (EWMA baseline + GJR-GARCH option).

Every returned value at date t is a FORECAST for date t's return, using
only data through date t-1 — never a trailing/contemporaneous statistic.
This is the estimate the Kelly sizing layer needs (sigma_hat_t^2 in
L*_t = mu_hat_t / sigma_hat_t^2), not a backward-looking realized-vol
descriptive statistic like utils used elsewhere in agents/timing/.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

_SCALE = 100.0  # arch's GARCH fitting is numerically better-conditioned on
                # returns scaled to O(1) rather than O(0.01) daily fractions.


def ewma_variance(returns: pd.Series, decay: float = 0.94) -> pd.Series:
    """RiskMetrics-style EWMA one-step-ahead variance forecast.

    sigma_hat_t^2 = decay * sigma_hat_{t-1}^2 + (1 - decay) * r_{t-1}^2

    Implemented as a contemporaneous EWM of squared returns, shifted by one
    period so the value at t depends only on returns through t-1.
    """
    r2 = returns.astype(float) ** 2
    contemporaneous = r2.ewm(alpha=1 - decay, adjust=False).mean()
    return contemporaneous.shift(1)


def gjr_garch_variance(
    returns: pd.Series,
    refit_every: int = 20,
    min_train_obs: int = 250,
    dist: str = "t",
) -> pd.Series:
    """GJR-GARCH(1,1,1) one-step-ahead variance forecast, refit periodically
    on an expanding window using only data available at each refit point —
    NEVER fit once on the full sample (that would leak future volatility
    information into early-period forecasts during walk-forward evaluation).

    Between refits, the fitted (omega, alpha, gamma, beta) parameters are
    held fixed and the variance recursion is advanced day-by-day using the
    REALIZED (not simulated) return at each step:

        sigma_hat_t^2 = omega + alpha*r_{t-1}^2 + gamma*I(r_{t-1}<0)*r_{t-1}^2
                        + beta*sigma_hat_{t-1}^2

    The gamma term is the GJR asymmetry: a negative r_{t-1} raises the
    forecast more than an equal-magnitude positive r_{t-1} (leverage
    effect).

    Args:
        returns: Daily simple returns.
        refit_every: Re-estimate GARCH parameters every N observations.
        min_train_obs: Minimum history required before the first fit;
            returns all-NaN if `returns` is shorter than this + 1.
        dist: Error distribution passed to arch_model ("t" or "normal").
    """
    from arch import arch_model

    result = pd.Series(np.nan, index=returns.index, dtype=float)

    # Series.pct_change() (the realistic caller pattern — see
    # agents/timing/kelly_position_agent.py) always has a leading NaN, and
    # arch_model() rejects ANY NaN in its input outright. Drop them before
    # fitting; `clean.index` stays a valid (date-aligned) subset of
    # `returns.index` so results can be written back positionally within it.
    clean = returns.dropna().astype(float)
    n = len(clean)
    if n < min_train_obs + 1:
        return result

    r = clean * _SCALE

    params: Optional[pd.Series] = None
    sigma_sq_state: Optional[float] = None
    last_return: Optional[float] = None

    for t in range(min_train_obs, n):
        if params is None or (t - min_train_obs) % refit_every == 0:
            train = r.iloc[:t]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = arch_model(train, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=dist)
                    fit_res = model.fit(disp="off")
                params = fit_res.params
                sigma_sq_state = float(fit_res.conditional_volatility.iloc[-1] ** 2)
                last_return = float(train.iloc[-1])
            except Exception:
                params = None
                continue

        omega = params["omega"]
        alpha = params["alpha[1]"]
        gamma = params["gamma[1]"] if "gamma[1]" in params.index else 0.0
        beta = params["beta[1]"]

        indicator = 1.0 if last_return < 0 else 0.0
        forecast = omega + alpha * last_return**2 + gamma * indicator * last_return**2 + beta * sigma_sq_state
        result.loc[r.index[t]] = forecast / (_SCALE**2)

        sigma_sq_state = forecast
        last_return = float(r.iloc[t])

    return result
