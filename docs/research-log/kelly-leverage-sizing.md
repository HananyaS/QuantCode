# Research Log: Kelly-Criterion Leverage Sizing (agents/timing/kelly_*)

## Purpose

Replace the ad hoc `target_vol / realized_vol` linear leverage sizing
(`agents/timing/leveraged_position_agent.py`, autoresearch-tuned to
`target_vol=0.65, vol_window=12`) with a theoretically-grounded sizing rule
derived from the Kelly criterion: leverage driven by the ratio of
conditional drift to conditional variance, not a heuristic vol threshold.

The prior module was found (via leakage-free holdout checks,
`scripts/leveraged_sizing_holdout_check.py`) to have a thin-to-negative
practical edge over the simpler naive TQQQ-or-cash overlay in genuinely
out-of-sample data. This module is a from-scratch, more rigorous attempt at
the same problem.

## Flags — deviations from the original request, and why

The build request assumed several things about this repo that don't hold;
resolved as follows rather than silently deviating:

1. **No `EXP-001`/`EXP-002`** exist here. The system that was actually built
   and killed was the cross-sectional S&P 500 ranking pipeline (see
   `docs/poc-plans/breakout-and-qqq-timing.md`'s context section) — no
   numbered-experiment convention exists. Walk-forward validation follows
   `utils/timing_walk_forward.py` + the `scripts/*_holdout_check.py` pattern
   established this session; reused here.
2. **No `PLAN.md`/`DECISIONS.md`** — only `CLAUDE.md`, `ROADMAP.md`,
   `README.md`, `docs/poc-plans/`. This document lives in `docs/research-log/`,
   a new but consistent convention.
3. **Data sources**: Alpaca credentials are stale/unused this session;
   yfinance + the local parquet cache (`utils/data_cache.py`) is what's
   actually wired and verified reliable. Stooq was evaluated and explicitly
   **rejected** (bot-walled) earlier this session; FMP was never used.
4. **No GARCH/HMM libraries were installed.** Added `arch>=6.0.0` (the
   standard Python GJR-GARCH/EGARCH implementation) as a new dependency —
   well-justified rather than hand-rolling MLE fitting. For the regime
   classifier, used a deterministic MA-EWMA/Wilder-ADX classifier instead of
   an HMM — avoids a second new ML dependency, keeps regime detection
   exactly reproducible for walk-forward review (consistent with
   `CLAUDE.md`'s "deterministic results" principle), and matches this
   repo's existing signal (`agents/timing/timing_signal_agent.py`), which is
   also rule-based rather than a fitted statistical model.
5. **Existing infra reused**: `agents/timing/leveraged_backtest_agent.py`
   already backtests any per-date `(ticker, fraction)` position across
   QQQ/QLD/TQQQ + cash. The Kelly module emits the same schema
   (`context['leverage_position']`) — but see the causality-convention note
   below for why a **new** backtest agent (`kelly_backtest_agent.py`) was
   still required rather than reusing `LeveragedBacktestAgent` verbatim.

## Theoretical foundation — the corrected g(L) formula

The requested formula was `g(L) = L*mu - 0.5*L*(L-1)*sigma^2`, alongside an
explicit constraint: at `L=1` (holding the underlying itself, no leverage),
the formula "must show zero *excess* decay ... only the baseline
single-asset Jensen term." These two statements are inconsistent as
literally written — `g(1) = mu - 0 = mu` under the literal formula, which
is **not** the Jensen term (`mu - 0.5*sigma^2`); it drops the ordinary
volatility drag every buy-and-hold position has.

**Derivation.** A fund tracking `L` times a daily simple return `r_t`
(mean `mu`, variance `sigma^2`) has daily log-return `log(1 + L*r_t)`.
Second-order Taylor expansion of `E[log(1+L*r)]` around 0 (dropping the
second-order-small `mu^2` term) gives:

```
g(L) = L*mu - 0.5*L^2*sigma^2
```

This is algebraically identical to:

```
g(L) = L*(mu - 0.5*sigma^2) - 0.5*L*(L-1)*sigma^2
     = [L times the underlying's own Jensen-adjusted growth]
       - [excess leverage decay, Cheng & Madhavan 2009]
```

At `L=1` the excess-decay term is exactly zero and `g(1) = mu - 0.5*sigma^2`
— correctly matching the stated constraint. **Both forms agree on the
growth-optimal leverage** `L* = mu/sigma^2` (the Jensen term is linear in
`L` and drops out of the first-order condition), so the requested Kelly
optimum is unaffected by this correction — only the absolute `g(L)` value
away from the optimum (in particular at `L=1`) differs.

Implemented as `g(L) = L*mu - 0.5*L^2*sigma^2` in `utils/kelly_sizing.py`
(`expected_log_growth`), with `excess_leverage_decay` exposing the
`L(L-1)/2*sigma^2` term separately so both pieces remain individually
visible, per the "explainable, not a black box" requirement.

**Half-Kelly verification.** `expected_log_growth(0.5*L*, mu, sigma^2) /
expected_log_growth(L*, mu, sigma^2) == 0.75` exactly, for this quadratic
`g(L)` — verified numerically in
`tests/test_kelly_sizing.py::test_half_kelly_gives_75_percent_of_growth`,
confirming the requested closed-form property holds in this
implementation's numerics.

## Causality convention (load-bearing design decision)

`utils/conditional_vol.py` and `utils/regime_classifier.py` both use a
**forecast-for-t** convention: the value at date `t` is a genuine forecast
using only data through `t-1` (pre-shifted at the source — e.g.
`ewma_variance` explicitly `.shift(1)`s). This is the standard GARCH-
literature convention (`sigma_t^2 | Info_{t-1}`) and differs from this
repo's existing `agents/timing/timing_signal_agent.py`, whose rolling
features are **not** pre-shifted — that agent instead relies on
`TimingBacktestAgent` applying a single downstream execution-lag shift.

Consequence: `KellyPositionAgent`'s output position for date `t` is
**already** the correct position to hold *during* `t` (the lag is baked
into `mu_hat_t`/`sigma_hat_t^2`/`regime_t`'s own semantics). Reusing
`LeveragedBacktestAgent` unchanged — which expects a non-shifted,
same-day-decided position and applies its own `.shift(1)` — would
**double-lag** the Kelly strategy relative to the naive/vol-sized layers,
making any comparison between them apples-to-oranges. `kelly_backtest_agent.py`
is therefore a small, deliberate fork of `LeveragedBacktestAgent`'s cost/
turnover logic **without** the internal shift — verified explicitly in
`tests/test_kelly_backtest_agent.py::test_same_day_position_applies_to_same_day_return_no_extra_lag`.

## Empirical finding: real leveraged-ETF drag is far above the prospectus expense ratio

Cross-validating the exact-recursion synthetic construction
(`utils/synthetic_leveraged_series.py`) against real cached TQQQ (2010+)
and QLD (2006+) prices: daily-return correlation is very high (0.999 /
0.996 respectively — the mechanic is right), but cumulative divergence over
15+ years is enormous if only the published 0.95% expense ratio is
modeled. Backing out the true daily drag empirically
(`calibrate_total_annual_drag`):

| Ticker | Stated expense ratio | Calibrated full-history drag |
|---|---|---|
| TQQQ (2010-2026) | 0.95%/yr | **5.42%/yr** |
| QLD (2006-2026) | 0.95%/yr | **3.26%/yr** |

Real leveraged ETFs finance their exposure via swaps/futures whose implicit
borrowing cost tracks short-term interest rates — a cost the prospectus
expense ratio doesn't capture. **This drag is strongly rate-regime-
dependent, not a stable constant**:

| Period | Rate environment | TQQQ implied annual drag |
|---|---|---|
| 2010-2015 | near-zero rates | 1.87%/yr |
| 2016-2021 | low rates | 4.59%/yr |
| 2022-2024 | high rates | 11.08%/yr |
| 2025-2026 | high rates | 11.36%/yr |

`configs/kelly_timing.yaml`'s `synthetic_series.tiers` uses the full-history
average (5.42%/3.26%) as a defensible default for a "typical" backtest, but
this **understates current costs in the present high-rate regime by more
than 2x**. A rate-linked dynamic drag model (financing cost ≈ short-term
rate + spread) would be materially more accurate but was judged out of
scope for this build — flagged here as a known limitation and a natural
next refinement, not silently ignored.

## Bug found and fixed during validation

`gjr_garch_variance` crashed silently on every real-data call:
`Series.pct_change()` (the realistic input — see `KellyPositionAgent.run()`)
always has a leading NaN, and `arch_model()` rejects any NaN outright. Every
unit test had used clean synthetic returns (`rng.normal` directly, never
via `pct_change()`), so this was invisible until a real-data smoke test
(`scripts/kelly_backtest_run.py`) showed the GJR-GARCH path returning a
suspiciously flat `avg_leverage=0.00x` across an entire backtest window —
the broad `except Exception` silently caught the crash every single
iteration, leaving the whole conditional-variance output all-NaN and the
position agent's validity gate permanently false, with no error surfaced.
Fixed by dropping NaN before fitting; a regression test using
`Series.pct_change()`-derived returns now guards this
(`tests/test_conditional_vol.py::test_gjr_garch_variance_handles_leading_nan_from_pct_change`).
**Lesson reinforced**: synthetic-fixture unit tests, however thorough,
don't substitute for at least one real-data smoke test before trusting a
new numerical component — this is the second time this session a real-data
check surfaced a bug invisible to synthetic fixtures (the first being
`agents/universe_agent.py`'s yfinance thread-safety bug).

## Backtest results (EWMA vol_method, default config, `configs/kelly_timing.yaml`)

Synthetic QLD/TQQQ built from QQQ's real 1999-2024 return history
(`utils/synthetic_leveraged_series.py`), so the dot-com crash is included —
real TQQQ/QLD history starts too late (2010/2006) to cover it.

| Window | Strategy Sharpe | Strategy CAGR | Strategy MaxDD | Buy&hold Sharpe | Buy&hold CAGR | Buy&hold MaxDD |
|---|---|---|---|---|---|---|
| Full history (1999-2024) | **0.77** | **16.9%** | **-44.2%** | 0.42 | 0.9% | **-100.0%** |
| Dev-like (2000-2019) | 0.52 | 8.8% | -44.2% | 0.25 | -12.8% | -100.0% |
| Recent (2020-2024) | 1.08 | 30.2% | -33.9% | 0.79 | 34.1% | -81.4% |
| Dot-com stress (2000-2002) | 0.58 | -0.0% | -0.0% | -0.63 | -89.2% | -99.9% |
| GFC stress (2007-2009) | -0.63 | -15.0% | -34.2% | -0.79 | -78.7% | -94.7% |
| Blind 2025-2026 (real, frozen config) | 1.49 | 56.4% | -18.1% | 1.07 | 66.4% | -56.5% |

**The single most important number here**: constant 3x-leveraged
buy-and-hold TQQQ, held from 1999, is **wiped to a literal zero NAV during
the dot-com crash** (`max(1+3r_t, 0)` hits its absorbing floor) and — by
construction of the recursion — never recovers, even though QQQ itself went
on to one of the greatest bull runs in market history over the following
two decades. This is why real TQQQ didn't launch until 2010: a 3x QQQ
product literally could not have survived 2000-2002. The Kelly strategy
avoids this entirely (dot-com window: `avg_leverage=0.00x`, essentially
fully in cash) — the whole point of conditional, time-varying sizing over a
static one, demonstrated in the most extreme case available in the data.

In the calmer, better-behaved 2020-2024 window, the familiar CAGR-for-
Sharpe/drawdown tradeoff reappears (Sharpe 1.08 vs 0.79, but CAGR 30.2% vs
34.1%) — consistent with every other timing/sizing layer built this
session and with the literature's honest framing (timing mostly buys
downside protection, not free alpha) — except here the *tail-risk*
protection is categorically larger (avoiding total ruin vs. just a smaller
drawdown), which the earlier linear-sizing module could never demonstrate
since it was never tested against a regime severe enough to matter.

### g(L) decomposition and naive-vs-actual (full history)

```
theoretical_g = 0.000397/day   (constant-avg-leverage approximation)
realized_g    = 0.000620/day   (strategy's actual realized log-growth)
gap           = +0.000223/day  (realized > theoretical)

naive_cum_return   = +1147.3%  (avg_leverage x underlying cumulative return)
actual_cum_return  = +5466.4%  (real compounded result)
divergence         = +4319.1%
```

Both diagnostics point the same direction: the strategy's real, exactly-
compounded result substantially **beats** what a constant-average-leverage
approximation predicts. This is coherent, not a red flag — `avg_leverage`
is a poor summary statistic when leverage is systematically *higher* during
calm/trending periods and *lower* during volatile/choppy ones (exactly what
the entry/exit hysteresis is designed to do); a naive linear model applied
to the average misses this timing correlation entirely. `naive_cum_return`
vs `actual_cum_return` is the same demonstration required for the constant-
leverage case (already covered by `tests/test_synthetic_leveraged_series.py`'s
`test_diverges_from_naive_multiplicative_approximation`), extended here to
the dynamically-sized case.

### Robustness check: fractional-Kelly sensitivity

```
fractional_kelly=0.25: Sharpe=0.806  CAGR=+17.9%  MaxDD=-44.2%  avg_lev=1.10x
fractional_kelly=0.50: Sharpe=0.765  CAGR=+16.9%  MaxDD=-44.2%  avg_lev=1.09x
fractional_kelly=1.00: Sharpe=0.756  CAGR=+16.8%  MaxDD=-44.2%  avg_lev=1.10x
```

Results are nearly insensitive to the fractional-Kelly multiplier — the
**entry/exit hysteresis thresholds** (`adx_threshold`, `vol_spike_threshold`,
`drawdown_limit`, `entry_margin`) are what actually bind realized exposure
most of the time, not the raw Kelly fraction. Reassuring for robustness
(the ruin-avoidance result doesn't hinge on getting the Kelly fraction
exactly right), but it also means those hysteresis constants are the
genuinely load-bearing, still-hand-picked parameters — the natural target
for a future autoresearch-style search, not the Kelly fraction itself.

## Known limitations (flagged, not silently ignored)

1. **No config search was run.** Every threshold in `configs/kelly_timing.yaml`
   is a reasoned default (half-Kelly, ADX 25 as the conventional trending
   threshold, etc.), not tuned against data — this means there's no
   autoresearch-style selection-leakage risk of the kind found in the prior
   linear-sizing module, but it also means these numbers haven't been
   empirically optimized. A future search (mirroring
   `scripts/measure_leveraged_strategy.py`'s pattern) is a natural next step
   if requested.
2. **Synthetic drag is a static full-history average**, not rate-linked —
   understates current costs by >2x in the present high-rate regime (see
   above).
3. **Single underlying, single (long) market history.** Same caveat that
   applies to every timing/sizing module built this session.
4. **GJR-GARCH refit is periodic, not per-day** (`garch_refit_every`,
   default 20 trading days) — a deliberate cost/accuracy tradeoff (daily
   refitting would be far slower); parameters are held fixed between
   refits and the variance recursion is advanced using realized (not
   simulated) returns, which is standard practice but is itself an
   approximation of a "true" daily-refit walk-forward GARCH.

## How to reproduce

```bash
python scripts/kelly_backtest_run.py   # full historical + blind-forward report
pytest tests/test_kelly_sizing.py tests/test_conditional_vol.py \
       tests/test_regime_classifier.py tests/test_synthetic_leveraged_series.py \
       tests/test_kelly_position_agent.py tests/test_kelly_backtest_agent.py \
       tests/test_kelly_evaluation_agent.py -q
```
