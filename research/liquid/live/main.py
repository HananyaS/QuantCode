"""C3 paper trading entry point (research/liquid/PAPER_C3.md).

    python -m research.liquid.live.main --check    # read-only: account, clock, next session's plan
    python -m research.liquid.live.main --run      # trade today's session, then write the report
    python -m research.liquid.live.main --report   # evening report for the latest finished session

--run trades one session and exits, so a scheduler starts it once each
trading day before 09:25 ET. On a closed day it says so and exits.
Kill switch: create data/research_liquid/live/KILL; the runner flattens and stops.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.liquid import cells
from research.liquid.live import report, signal
from research.liquid.live.broker import AlpacaPaperBroker, load_keys
from research.liquid.live.feed import AlpacaFeed
from research.liquid.live.ledger import Ledger
from research.liquid.live.runner import SYMBOL, Runner, Session, WallClock

ET = ZoneInfo("America/New_York")
LIVE_DIR = Path("data/research_liquid/live")
LEDGER = LIVE_DIR / "c3_ledger.sqlite"
KILL = LIVE_DIR / "KILL"
HISTORY_SESSIONS = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("c3")


def sessions(broker: AlpacaPaperBroker, start: date, end: date) -> list[Session]:
    out = []
    for c in broker.calendar(start, end):
        o = pd.Timestamp(c.open).tz_localize(ET) if pd.Timestamp(c.open).tzinfo is None else pd.Timestamp(c.open).tz_convert(ET)
        k = pd.Timestamp(c.close).tz_localize(ET) if pd.Timestamp(c.close).tzinfo is None else pd.Timestamp(c.close).tz_convert(ET)
        out.append(Session(pd.Timestamp(c.date), o.to_pydatetime(), int((k - o).total_seconds() // 60)))
    return out


def plan_for(feed: AlpacaFeed, cal: list[Session], day: pd.Timestamp) -> signal.DayPlan:
    past = [s for s in cal if s.day < day][-HISTORY_SESSIONS:]
    hist = feed.history(SYMBOL, [(s.day, s.n_bars) for s in past])
    today = next(s for s in cal if s.day == day)
    return signal.day_plan(hist, day, today.n_bars)


def _clients():
    key, secret = load_keys()
    return AlpacaPaperBroker(key, secret), AlpacaFeed(key, secret)


def cmd_check() -> int:
    broker, feed = _clients()
    acct = broker.account()
    clk = broker.clock()
    today = datetime.now(ET).date()
    cal = sessions(broker, today - timedelta(days=45), today + timedelta(days=10))
    nxt = next(s for s in cal if s.day >= pd.Timestamp(today))
    plan = plan_for(feed, cal, nxt.day)
    print(json.dumps({
        "account": acct.__dict__, "clock_open": clk.is_open, "next_open": str(clk.next_open),
        "position_SPY": broker.position_qty(SYMBOL), "open_orders_SPY": len(broker.open_orders(SYMBOL)),
        "next_session": {"day": str(nxt.day.date()), "open": str(nxt.open), "n_bars": nxt.n_bars},
        "plan": {"prev_close": plan.prev_close, "vol_14d": plan.vol, "weight": plan.weight,
                 "checks": len(plan.checks), "sigma_at_10:00": float(plan.sigma[29]),
                 "notional_at_weight": acct.last_equity * plan.weight}}, indent=1, default=str))
    return 0


def cmd_run() -> int:
    broker, feed = _clients()
    today = pd.Timestamp(datetime.now(ET).date())
    cal = sessions(broker, (today - timedelta(days=45)).date(), today.date())
    if not cal or cal[-1].day != today:
        logger.info("%s is not a trading session; nothing to do", today.date())
        return 0
    sess = cal[-1]
    if datetime.now(ET) >= sess.bar_start(sess.n_bars - 1):
        logger.info("session already at its last minute; not starting")
        return 0
    plan = plan_for(feed, cal, today)
    logger.info("plan %s: weight %.2f, vol %.4f, %d checks", today.date(), plan.weight, plan.vol, len(plan.checks))
    runner = Runner(broker, feed, WallClock(), Ledger(LEDGER), kill_file=KILL)
    runner.run_session(sess, plan)
    logger.info("session done; the evening report runs from 16:20 ET (--report)")
    return 0


def cmd_report() -> int:
    broker, feed = _clients()
    now = datetime.now(ET)
    cal = sessions(broker, (now - timedelta(days=45)).date(), now.date())
    done = [s for s in cal if s.bar_end(s.n_bars - 1) + timedelta(minutes=16) <= now]
    day = done[-1]
    window = done[-(HISTORY_SESSIONS + 1):]
    m = feed.history(SYMBOL, [(s.day, s.n_bars) for s in window])
    i = len(m.dates) - 1
    research = [t for t in cells.c3(m) if t.date == day.day]
    led = Ledger(LEDGER)
    live = led.trades(str(day.day.date()))
    rows = [report.trade_costs(t, m, i) for t in live if t["exit_price"] is not None]
    fid = report.fidelity(live, research)
    out = {"day": str(day.day.date()), "live_trades": live, "research_trades": [
        {"entry_check": t.entry_bar - 1, "entry": t.entry, "exit": t.exit, "kind": t.kind} for t in research],
        "costs": rows, "fidelity": fid}
    rep_dir = LIVE_DIR / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{out['day']}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    all_rows = []
    for f in sorted(rep_dir.glob("20*.json")):
        all_rows += json.loads(f.read_text(encoding="utf-8"))["costs"]
    all_live = led.trades()
    summary = report.gates(all_rows, {"live_matched_share": np.mean(
        [json.loads(f.read_text(encoding="utf-8"))["fidelity"]["live_matched_share"]
         for f in sorted(rep_dir.glob("20*.json"))
         if json.loads(f.read_text(encoding="utf-8"))["fidelity"]["live"]] or [np.nan])})
    summary["days"] = len(list(rep_dir.glob("20*.json")))
    summary["live_round_trips_total"] = len(all_live)
    (LIVE_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"day": out["day"], "fidelity": fid, "summary": summary}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if a.check:
        return cmd_check()
    if a.run:
        return cmd_run()
    return cmd_report()


if __name__ == "__main__":
    sys.exit(main())
