"""live_run.py — manually-triggered daily paper-trading run for one strategy.

Usage:
    python scripts/live_run.py --strategy kelly   --dry-run
    python scripts/live_run.py --strategy linear  --dry-run
    python scripts/live_run.py --strategy kelly            # submits real (paper) orders

Two independent strategies, two independent Alpaca PAPER accounts, two
independent local state ledgers (data/live_state_<strategy>.db) — credentials
are read from .env as ALPACA_API_KEY_KELLY/ALPACA_SECRET_KEY_KELLY and
ALPACA_API_KEY_LINEAR/ALPACA_SECRET_KEY_LINEAR.

Decision data is fetched fresh (real QQQ/^VIX/QLD/TQQQ market data via
yfinance) every run — NOT the synthetic QLD/TQQQ series used for
backtesting (utils/synthetic_leveraged_series.py). Synthetic construction
exists to extend backtests before TQQQ/QLD's real inception; live execution
must price and size orders against real current market data.

--dry-run prints the computed decision and a preview of what orders WOULD
be placed (today's target weights vs. the broker's actual current
positions) without calling ExecutionAgent — no orders submitted, no state
written. Review a dry run before the first live run for a strategy.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

from agents.execution_agent import ExecutionAgent
from agents.timing.kelly_position_agent import KellyPositionAgent
from agents.timing.leveraged_position_agent import LeveragedPositionAgent
from agents.timing.timing_signal_agent import TimingSignalAgent
from agents.universe_agent import UniverseAgent
from utils.live_decision import (
    append_next_session_bar,
    drop_incomplete_last_bar,
    position_row_to_weights,
)
from utils.logger import get_logger
from utils.state_store import StateStore

logger = get_logger(__name__)

_TRACKED_TICKERS = ("QQQ", "QLD", "TQQQ")
_LOOKBACK_DAYS = 1500  # ~4 years -- comfortably warms up SMA-190/ADX-14/EWMA
_FETCH_MAX_ATTEMPTS = 3
_FETCH_RETRY_SECONDS = 15.0


def _run_with_retry(build_agent) -> dict:
    """Retry a UniverseAgent().run({}) call a few times with a short
    backoff. `build_agent` is a zero-arg callable returning a fresh
    UniverseAgent -- a fresh instance each attempt, no shared state to
    worry about between retries.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _FETCH_MAX_ATTEMPTS + 1):
        try:
            return build_agent().run({})["universe_data"]
        except Exception as exc:
            last_exc = exc
            if attempt < _FETCH_MAX_ATTEMPTS:
                logger.warning(
                    "Universe fetch failed (attempt %d/%d): %s -- retrying in %.0fs",
                    attempt, _FETCH_MAX_ATTEMPTS, exc, _FETCH_RETRY_SECONDS,
                )
                time.sleep(_FETCH_RETRY_SECONDS)
    raise last_exc


def _fetch_live_universe(alpaca_key: str, alpaca_secret: str, extra_tickers: tuple) -> dict:
    """QQQ/QLD/TQQQ via Alpaca's own data API (feed='iex', the free-tier
    feed -- the default SIP feed 403s without a paid subscription).
    Observed in production: yfinance returned a batch of NaN Close values
    for all three tickers on 2 of 3 real GitHub Actions runs -- reproducible,
    not a one-off blip, consistent with the rate-limiting yfinance is known
    for from datacenter IP ranges (see agents/universe_agent.py's own
    docstring). Alpaca is authenticated and account-scoped, not subject to
    that failure mode, and we already hold live trading credentials here.

    '^VIX' isn't a tradeable equity/ETF, so Alpaca's stock data API doesn't
    carry it -- it's the one ticker still fetched via yfinance (with the
    same retry-on-transient-failure treatment), only needed by the Linear
    strategy's vix_regime signal component.

    The end date is deliberately padded 2 days into the future: whether
    "today" falls inside an end=date.today() request depends on the
    MACHINE's timezone (observed live: a UTC CI runner excluded a bar the
    UTC+9 laptop included, for the same logical request). Fetch wide, then
    let the caller trim any not-yet-closed session's partial bar against
    the broker's own clock (drop_incomplete_last_bar) -- explicit and
    timezone-independent, instead of implicit and machine-dependent.
    """
    end = date.today() + timedelta(days=2)
    start = end - timedelta(days=_LOOKBACK_DAYS)
    start_s, end_s = str(start), str(end)

    universe_data = _run_with_retry(lambda: UniverseAgent(
        tickers=list(_TRACKED_TICKERS), start_date=start_s, end_date=end_s,
        benchmark="QQQ", min_assets=len(_TRACKED_TICKERS), min_history_days=500,
        data_source="alpaca", alpaca_key=alpaca_key, alpaca_secret=alpaca_secret, alpaca_feed="iex",
    ))

    if "^VIX" in extra_tickers:
        vix_data = _run_with_retry(lambda: UniverseAgent(
            tickers=["^VIX"], start_date=start_s, end_date=end_s,
            benchmark="^VIX", min_assets=1, min_history_days=500,
            data_source="yfinance",
        ))
        universe_data["^VIX"] = vix_data["^VIX"]

    return universe_data


def _decide_kelly(universe_data: dict, next_session: pd.Timestamp) -> tuple:
    """Decision FOR `next_session` (the session actually being traded),
    computed from all data through the latest completed close.

    KellyPositionAgent's estimators are forecast-for-t (value at date t uses
    data through t-1, pre-shifted at source), so the row at the last
    completed bar was decided from data through the bar BEFORE it —
    executing that row the next morning would be one full trading day
    stale. Extending the index with a placeholder bar at `next_session`
    yields the genuine decision for the session being traded. See
    utils/live_decision.py::append_next_session_bar for the mechanics.
    """
    with open("configs/kelly_timing.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cv, rg, ks = cfg["conditional_vol"], cfg["regime"], cfg["kelly_sizing"]
    assert cv["method"] == "ewma", (
        "live Kelly decisions require vol_method='ewma': gjr_garch_variance "
        "drops NaN rows before fitting, so the appended next-session bar "
        "would get a NaN forecast and silently produce a flat decision. "
        "Wire GARCH into the live path deliberately before enabling it here."
    )
    extended = append_next_session_bar(universe_data, next_session, tickers=_TRACKED_TICKERS)
    ctx = {"universe_data": extended}
    ctx = KellyPositionAgent(
        signal_ticker=cfg["universe"]["underlying_ticker"],
        vol_method=cv["method"], vol_decay=cv["ewma_decay"],
        garch_refit_every=cv["garch_refit_every"], garch_min_train_obs=cv["garch_min_train_obs"],
        mu_decay=rg["mu_decay"], adx_period=rg["adx_period"], adx_threshold=rg["adx_threshold"],
        fractional_kelly=ks["fractional_kelly"], max_leverage=ks["max_leverage"],
        worst_case_daily_move=ks["worst_case_daily_move"], ruin_buffer=ks["ruin_buffer"],
        vol_spike_threshold=ks["vol_spike_threshold"], drawdown_limit=ks["drawdown_limit"],
        entry_margin=ks["entry_margin"], min_observations=ks["min_observations"],
    ).run(ctx)
    position = ctx["leverage_position"]
    log = ctx["kelly_decision_log"]
    last = position.iloc[-1]
    last_log = log.iloc[-1]
    detail = (
        f"mu_hat={last_log['mu_hat']:.6f}  sigma_sq_hat={last_log['sigma_sq_hat']:.6f}  "
        f"l_star={last_log['l_star']:.3f}  regime={last_log['regime']}"
    )
    return position.index[-1], last["ticker"], float(last["fraction"]), detail


def _decide_linear(universe_data: dict) -> tuple:
    """Decision from the last completed bar, NO index extension — the
    deliberate asymmetry with _decide_kelly. TimingSignalAgent and
    LeveragedPositionAgent use unshifted same-day rolling windows (the
    execution lag lives downstream in TimingBacktestAgent's shift(1)), so
    the last-bar row already IS the position to hold during the next
    session. Appending a NaN bar here would make every rolling window NaN
    at that row -> signal 0 -> a silent false de-risk.
    """
    with open("configs/timing.yaml") as fh:
        cfg = yaml.safe_load(fh)
    s, ls = cfg["signal"], cfg["leverage_sizing"]
    signal_ctx = {"universe_data": universe_data}
    signal_ctx = TimingSignalAgent(
        ticker=s["ticker"], sma_window=s["sma_window"], mom_window=s.get("mom_window"),
        vol_window=s.get("vol_window"), vol_threshold=s.get("vol_threshold"),
        vix_threshold=s.get("vix_threshold"), combine=s["combine"],
    ).run(signal_ctx)
    signal = signal_ctx["timing_signal"]

    pos_ctx = {"universe_data": universe_data, "timing_signal": signal}
    pos_ctx = LeveragedPositionAgent(
        signal_ticker=ls["signal_ticker"], vol_window=ls["vol_window"],
        target_vol=ls["target_vol"], max_leverage=ls["max_leverage"],
    ).run(pos_ctx)
    position = pos_ctx["leverage_position"]
    last = position.iloc[-1]
    detail = f"signal={int(signal.iloc[-1])}"
    return position.index[-1], last["ticker"], float(last["fraction"]), detail


def _print_dry_run_preview(client, weights: dict, universe_data: dict, equity_hint: str) -> None:
    account = client.get_account()
    equity = float(account.equity)
    broker_positions = {p.symbol: float(p.market_value) for p in client.get_all_positions()}
    print(f"\n  Broker account ({equity_hint}): equity=${equity:,.2f}  "
          f"positions={broker_positions or '{}'}")
    print("\n  Preview (no orders submitted):")
    print(f"  {'ticker':<8}{'target_w':>10}{'target_$':>14}{'current_$':>14}{'delta_$':>14}  action")
    for ticker, w in weights.items():
        target_dollars = w * equity
        current_dollars = broker_positions.get(ticker, 0.0)
        delta = target_dollars - current_dollars
        price = float(universe_data[ticker]["Close"].iloc[-1])
        action = "hold / no-op"
        if abs(delta) >= 50.0:
            side = "BUY" if delta > 0 else "SELL"
            qty = int(abs(delta) // price)
            action = f"{side} ~{qty} sh @ ${price:,.2f}" if qty > 0 else "below 1 share"
        print(f"  {ticker:<8}{w:>10.3f}{target_dollars:>14,.2f}{current_dollars:>14,.2f}{delta:>14,.2f}  {action}")


def _is_market_day(client, day: date) -> bool:
    """True iff NYSE is open for regular trading on `day` (Alpaca's own
    calendar -- correctly excludes weekends AND holidays like Thanksgiving,
    Christmas, Good Friday, etc., which a plain weekday check would miss).
    """
    from alpaca.trading.requests import GetCalendarRequest
    return len(client.get_calendar(GetCalendarRequest(start=day, end=day))) > 0


def _session_closed_checker(client):
    """Build is_session_closed(bar_date) for drop_incomplete_last_bar from
    the broker's own clock and calendar: a daily bar is complete iff its
    session's official close is in the past. Calendar lookups are memoized
    -- every tracked frame shares at most a couple of distinct last-bar
    dates per run.
    """
    from alpaca.trading.requests import GetCalendarRequest
    now = pd.Timestamp(client.get_clock().timestamp)  # tz-aware
    cache: dict = {}

    def is_session_closed(bar_date) -> bool:
        day = pd.Timestamp(bar_date).date()
        if day not in cache:
            cal = client.get_calendar(GetCalendarRequest(start=day, end=day))
            if not cal:
                # Not a trading session at all (shouldn't happen for a real
                # bar) -- nothing can still be in progress, treat as closed.
                cache[day] = True
            else:
                close_dt = pd.Timestamp(cal[0].close)
                if close_dt.tzinfo is None:
                    close_dt = close_dt.tz_localize("America/New_York")
                cache[day] = now >= close_dt
        return cache[day]

    return is_session_closed


def _next_trading_session(client, after: date) -> pd.Timestamp:
    """First NYSE trading session STRICTLY after `after` (Alpaca calendar).

    This is the session the current run's orders will actually fill in:
    during regular hours it's today (today > last completed bar); submitted
    after close, a DAY market order queues for the next open. Used both as
    the Kelly decision's target session and as the run_date recorded in the
    StateStore ledger, so the PDT same-day-round-trip guard keys on the
    real fill session rather than a stale bar date.
    """
    from alpaca.trading.requests import GetCalendarRequest
    cal = client.get_calendar(GetCalendarRequest(
        start=after + timedelta(days=1), end=after + timedelta(days=10),
    ))
    assert cal, f"no NYSE session found within 10 days after {after}"
    return pd.Timestamp(cal[0].date)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["kelly", "linear"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="Preview only, submit nothing.")
    args = parser.parse_args()

    if args.strategy == "kelly":
        key_env, secret_env = "ALPACA_API_KEY_KELLY", "ALPACA_SECRET_KEY_KELLY"
        db_path = "data/live_state_kelly.db"
    else:
        key_env, secret_env = "ALPACA_API_KEY_LINEAR", "ALPACA_SECRET_KEY_LINEAR"
        db_path = "data/live_state_linear.db"

    api_key = os.environ.get(key_env)
    secret_key = os.environ.get(secret_env)
    assert api_key and secret_key, f"{key_env}/{secret_env} not set in .env"

    from alpaca.trading.client import TradingClient
    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    # Real runs only: a scheduled trigger (e.g. GitHub Actions cron) fires on
    # every weekday regardless of market holidays. Without this check, a
    # holiday run would recompute the same decision as the last real trading
    # day (no new bar exists) and could attempt to resubmit toward that same
    # target. Gate on the broker's own calendar, before any data fetch or
    # order logic runs at all -- not on --dry-run, so manual preview/testing
    # still works on weekends/holidays.
    if not args.dry_run and not _is_market_day(client, date.today()):
        print(f"{date.today()} is not an NYSE trading day -- skipping ({args.strategy}). "
              "No data fetched, no orders submitted.")
        return

    extra = ("^VIX",) if args.strategy == "linear" else ()
    print(f"Fetching live market data ({', '.join(_TRACKED_TICKERS + extra)}) ...")
    universe_data = _fetch_live_universe(api_key, secret_key, extra)
    # Trim any in-progress partial bar (fetch is deliberately end-padded;
    # completeness is decided against the broker clock, not the machine's
    # timezone) so the signal math only ever sees completed sessions --
    # matching what every backtest was validated on.
    universe_data = drop_incomplete_last_bar(universe_data, _session_closed_checker(client))
    for t, df in universe_data.items():
        print(f"  {t}: {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} bars)")

    last_bar = max(universe_data[t].index[-1] for t in _TRACKED_TICKERS)
    next_session = _next_trading_session(client, last_bar.date())

    if args.strategy == "kelly":
        decision_date, ticker, fraction, detail = _decide_kelly(universe_data, next_session)
    else:
        decision_date, ticker, fraction, detail = _decide_linear(universe_data)

    print(f"\nPosition for session {next_session.date()} ({args.strategy}), "
          f"decided from data through {last_bar.date()}:")
    print(f"  -> {ticker}  fraction={fraction:.3f}  ({detail})")

    weights = position_row_to_weights(ticker, fraction, tracked_tickers=_TRACKED_TICKERS)
    # Index the weights row by the FILL session, not the signal's bar date:
    # ExecutionAgent takes its ledger run_date from this index, and the PDT
    # same-day-round-trip guard must key on the session the order actually
    # fills in. (For Kelly, decision_date == next_session already; for
    # Linear, decision_date is the last bar whose signal, by that
    # pipeline's shift-downstream convention, is held during next_session.)
    weights_df = pd.DataFrame([weights], index=[next_session])

    if args.dry_run:
        _print_dry_run_preview(client, weights, universe_data, equity_hint=f"{args.strategy} paper account")
        print("\n  Dry run only -- no orders submitted, no state written.")
        return

    exec_universe = {t: universe_data[t] for t in _TRACKED_TICKERS}
    exec_ctx = {"portfolio_weights": weights_df, "universe_data": exec_universe}
    exec_ctx = ExecutionAgent(
        api_key=api_key, secret_key=secret_key, paper=True,
        state_store=StateStore(db_path=db_path),
    ).run(exec_ctx)

    print(f"\n  Orders: {exec_ctx['execution_orders']}")
    print(f"  Account after run: {exec_ctx['execution_account']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
