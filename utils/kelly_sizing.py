"""kelly_sizing.py — pure Kelly-criterion leverage math.

Formula derivation (see docs/research-log/kelly-leverage-sizing.md for the
full write-up)
------------------------------------------------------------------------
A fund tracking L times a daily simple return r_t (mean mu, variance
sigma_sq) has daily log-return log(1 + L*r_t). Second-order Taylor
expansion of E[log(1 + L*r)] around 0, dropping the second-order-small
mu^2 term, gives:

    g(L) = L*mu - 0.5*L^2*sigma_sq

This is algebraically identical to:

    g(L) = L*(mu - 0.5*sigma_sq) - 0.5*L*(L-1)*sigma_sq
         = [L times the underlying's own Jensen-adjusted log growth]
           - [excess leverage decay, Cheng & Madhavan 2009]

At L=1 the excess-decay term is exactly zero and g(1) = mu - 0.5*sigma_sq
(the ordinary volatility drag every buy-and-hold position has) — NOT mu
exactly. A commonly-seen one-line formula "g(L) = L*mu - 0.5*L*(L-1)*sigma_sq"
used AS g(L) directly is missing this baseline Jensen term and would
incorrectly show zero total decay (not just zero *excess* decay) at L=1.
Both forms agree on the growth-optimal leverage L* = mu/sigma_sq, since the
Jensen term is linear in L and drops out of the first-order condition.
"""
from __future__ import annotations

import math
from typing import List, Tuple

_EPS = 1e-12


def kelly_optimal_leverage(mu: float, sigma_sq: float) -> float:
    """Growth-optimal leverage L* = mu / sigma_sq.

    Returns 0.0 for a degenerate (zero or near-zero) variance estimate
    rather than +/-inf — a conditional vol estimator can legitimately
    output ~0 during a dead-flat stretch, and this must degrade to "no
    leveraged opinion" rather than propagate an infinity downstream.
    """
    if abs(sigma_sq) < _EPS:
        return 0.0
    return mu / sigma_sq


def excess_leverage_decay(leverage: float, sigma_sq: float) -> float:
    """L(L-1)/2 * sigma_sq — the Cheng & Madhavan (2009) excess decay of a
    leveraged fund's growth rate relative to L times the underlying's own
    Jensen-adjusted growth. Zero at L=0 and L=1, positive (quadratic) for
    L>1 or L<0.
    """
    return leverage * (leverage - 1) / 2 * sigma_sq


def expected_log_growth(leverage: float, mu: float, sigma_sq: float) -> float:
    """g(L) = L*mu - 0.5*L^2*sigma_sq, the fund's expected daily log-growth
    rate. Equivalently L*(mu - 0.5*sigma_sq) - excess_leverage_decay(L, sigma_sq)."""
    return leverage * mu - 0.5 * leverage**2 * sigma_sq


def fractional_kelly(
    l_star: float,
    fraction: float = 0.5,
    max_leverage: float = 3.0,
) -> float:
    """Apply a fractional-Kelly safety multiplier to the raw optimum,
    clip negative results to 0.0 (no shorting support), and clip to
    max_leverage.

    Half-Kelly (fraction=0.5) sacrifices exactly 25% of the achievable
    expected growth rate for a much narrower variance of outcomes — a
    well-known closed-form property of the quadratic g(L) above, verified
    numerically in tests/test_kelly_sizing.py.
    """
    if not math.isfinite(l_star):
        return 0.0
    target = fraction * l_star
    return float(min(max(target, 0.0), max_leverage))


def ruin_floor_cap(leverage: float, worst_case_daily_move: float, buffer: float) -> float:
    """Cap leverage so a single-day move of `worst_case_daily_move` stays at
    least `buffer` fraction away from the ruin threshold -1/L for the
    instrument in play.

    Ruin occurs when L * daily_move <= -1 (NAV recursion max(1+L*r, 0) hits
    zero). We require the assumed worst-case move to be no closer than
    `buffer` to that threshold: worst_case_daily_move <= (1/L) * (1 - buffer),
    i.e. L <= (1 - buffer) / worst_case_daily_move.
    """
    if leverage <= 0:
        return 0.0
    max_safe_leverage = (1 - buffer) / worst_case_daily_move
    return float(min(leverage, max_safe_leverage))


def map_to_instrument_blend(
    target_leverage: float,
    tiers: Tuple[Tuple[float, str], ...] = ((1.0, "QQQ"), (2.0, "QLD"), (3.0, "TQQQ")),
) -> Tuple[str, float]:
    """Map a continuous target leverage onto the smallest tier whose own
    leverage is >= target, held at a partial-cash fraction to hit the
    target exactly (e.g. 1.4x -> QLD at 70% invested, 30% cash).

    Returns ("QQQ", 0.0) — flat — for target_leverage <= 0.
    """
    if target_leverage <= 0:
        return tiers[0][1], 0.0

    ordered = sorted(tiers, key=lambda pair: pair[0])
    for tier_leverage, ticker in ordered:
        if target_leverage <= tier_leverage + _EPS:
            fraction = target_leverage / tier_leverage
            return ticker, float(min(max(fraction, 0.0), 1.0))

    top_leverage, top_ticker = ordered[-1]
    return top_ticker, 1.0
