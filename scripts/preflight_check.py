"""preflight_check.py -- decide whether this workflow firing should trade and speak.

Why this exists
---------------
GitHub Actions cron fires the live-trading workflow twice every weekday
(primary 14:45 UTC + backup 15:45 UTC) and knows nothing about the NYSE
calendar. `live_run.py` already refuses to do anything wrong on those extra
firings -- it exits early on a market holiday, and its once-per-session
guard stops the backup from re-trading a session the primary already
executed. Both of those exit 0, which is correct.

The notification step, however, ran `if: always()` and had no idea whether
any of that work actually happened. So every no-op firing still sent a
push. In practice:

  * a normal trading day -> 2 pushes, the second one meaningless
    ("succeeded" for a run that did nothing);
  * a market holiday (Thanksgiving, July 4th, ...) -> 2 pushes, BOTH
    meaningless, for a day the market never opened.

This module answers, before any market data is fetched or any order logic
runs:

  trade  -- should this firing execute the strategies at all?
  notify -- should this firing send a push notification?

so the workflow does real work, and speaks, exactly once per NYSE trading
session -- and stays completely silent the rest of the time.

The decision itself is a pure function (`decide`) so it can be tested
without a broker connection; `main()` only supplies the two facts it needs
from Alpaca and the state ledgers.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from utils.state_store import StateStore

_KELLY_DB = "data/live_state_kelly.db"
_LINEAR_DB = "data/live_state_linear.db"


@dataclass(frozen=True)
class Preflight:
    """What this firing should do, and why."""
    trade: bool
    notify: bool
    reason: str


def decide(
    *,
    is_trading_day: bool,
    already_ran: bool,
    market_open: bool,
    manual: bool = False,
) -> Preflight:
    """Decide whether this firing runs the strategies and whether it notifies.

    Args:
        is_trading_day: Whether today is an NYSE trading session, per the
            broker's own calendar (excludes weekends AND holidays).
        already_ran: Whether a completed run has already recorded today's
            session for BOTH strategies. False if either is missing, so a
            half-failed primary still gets its backup retry.
        market_open: Whether the market is open *right now*, per the
            broker's clock.
        manual: True for a `workflow_dispatch` run. A human pressed the
            button and is waiting on the answer, so a manual run always
            executes and always reports back -- the once-a-day quiet rule
            governs the *schedule*, not deliberate hands-on runs.

    Returns:
        A `Preflight` with `trade`, `notify` and a human-readable `reason`.

    The order of these checks matters: `already_ran` is tested before
    `market_open` so that a session which completed normally stays silent
    even when the backup firing lands after the close.
    """
    if manual:
        return Preflight(True, True, "manual run -- executing and reporting as requested")

    if not is_trading_day:
        return Preflight(
            False, False,
            f"{date.today()} is not an NYSE trading day -- nothing to do, staying silent",
        )

    if already_ran:
        return Preflight(
            False, False,
            "today's session already completed on an earlier firing -- "
            "backup cron standing down silently",
        )

    if not market_open:
        # Reachable only if GitHub delayed BOTH firings past the close --
        # cron is best-effort and multi-hour queue delays are documented.
        # Trading here would submit against a signal computed from a bar
        # that has since closed, and the order would queue overnight
        # instead of filling. Skipping loses the day; filling at the wrong
        # time loses money. This one IS worth a push: the day was missed.
        return Preflight(
            False, True,
            "market is closed right now and today's session never ran -- skipped "
            "rather than queue an after-hours order on a stale signal",
        )

    return Preflight(True, True, "first firing of an NYSE trading session -- executing")


def _is_trading_day(client, day: date) -> bool:
    """True iff `day` is an NYSE session, per Alpaca's calendar."""
    from alpaca.trading.requests import GetCalendarRequest

    return len(client.get_calendar(GetCalendarRequest(start=day, end=day))) > 0


def _session_date(clock) -> date:
    """Today's date in exchange-local time, from the broker's own clock.

    Taken from the broker rather than the runner's clock so the answer
    doesn't depend on the machine's timezone -- GitHub runners are UTC, and
    a UTC date can differ from the ET session date outside the cron window.
    """
    return pd.Timestamp(clock.timestamp).tz_convert("America/New_York").date()


def _already_ran(session_key: str) -> bool:
    """True iff BOTH strategies already recorded this session.

    `and`, deliberately: if Kelly recorded today but Linear died before it
    could, the day is NOT done and the backup firing must still run (and
    still report). Only a fully successful primary earns the silence.
    """
    return all(
        StateStore(db_path=db).has_snapshot_on(session_key)
        for db in (_KELLY_DB, _LINEAR_DB)
    )


def _emit(pf: Preflight, github_output: Optional[str]) -> None:
    """Print the decision, and expose it as workflow step outputs."""
    print(f"Preflight: trade={pf.trade} notify={pf.notify} -- {pf.reason}")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as fh:
        fh.write(f"trade={str(pf.trade).lower()}\n")
        fh.write(f"notify={str(pf.notify).lower()}\n")
        fh.write(f"reason={pf.reason}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual", action="store_true",
        help="This is a workflow_dispatch run: always trade, always notify.",
    )
    args = parser.parse_args()

    if args.manual:
        _emit(decide(is_trading_day=True, already_ran=False, market_open=True, manual=True),
              os.environ.get("GITHUB_OUTPUT"))
        return

    api_key = os.environ.get("ALPACA_API_KEY_KELLY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY_KELLY")
    assert api_key and secret_key, "ALPACA_API_KEY_KELLY/ALPACA_SECRET_KEY_KELLY not set"

    from alpaca.trading.client import TradingClient
    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    clock = client.get_clock()
    session = _session_date(clock)
    trading_day = _is_trading_day(client, session)
    # Only ask the ledgers once we know it's a session at all -- on a
    # holiday the answer is irrelevant and the DBs may not even exist yet.
    already = _already_ran(str(session)) if trading_day else False

    _emit(
        decide(
            is_trading_day=trading_day,
            already_ran=already,
            market_open=bool(clock.is_open),
        ),
        os.environ.get("GITHUB_OUTPUT"),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
