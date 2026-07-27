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
from utils.live_decision import position_row_to_weights
from utils.logger import get_logger
from utils.state_store import StateStore

logger = get_logger(__name__)

_TRACKED_TICKERS = ("QQQ", "QLD", "TQQQ")
_LOOKBACK_DAYS = 1500  # ~4 years -- comfortably warms up SMA-190/ADX-14/EWMA


def _fetch_live_universe(extra_tickers: tuple) -> dict:
    tickers = list(dict.fromkeys(list(_TRACKED_TICKERS) + list(extra_tickers)))
    end = date.today()
    start = end - timedelta(days=_LOOKBACK_DAYS)
    agent = UniverseAgent(
        tickers=tickers, start_date=str(start), end_date=str(end),
        benchmark="QQQ", min_assets=len(tickers), min_history_days=500,
        data_source="yfinance",
    )
    ctx = agent.run({})
    return ctx["universe_data"]


def _decide_kelly(universe_data: dict) -> tuple:
    with open("configs/kelly_timing.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cv, rg, ks = cfg["conditional_vol"], cfg["regime"], cfg["kelly_sizing"]
    ctx = {"universe_data": universe_data}
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
    universe_data = _fetch_live_universe(extra)
    for t, df in universe_data.items():
        print(f"  {t}: {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} bars)")

    if args.strategy == "kelly":
        decision_date, ticker, fraction, detail = _decide_kelly(universe_data)
    else:
        decision_date, ticker, fraction, detail = _decide_linear(universe_data)

    print(f"\nDecision for {decision_date.date()} ({args.strategy}):")
    print(f"  -> {ticker}  fraction={fraction:.3f}  ({detail})")

    weights = position_row_to_weights(ticker, fraction, tracked_tickers=_TRACKED_TICKERS)
    weights_df = pd.DataFrame([weights], index=[decision_date])

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
