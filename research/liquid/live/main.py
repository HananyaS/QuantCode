"""C3 paper trading entry point (research/liquid/PAPER_C3.md).

    python -m research.liquid.live.main --check    # read-only: account, clock, safety state, next plan
    python -m research.liquid.live.main --drill    # one-time: no-shorting setting + duplicate-id test
    python -m research.liquid.live.main --run      # trade today's session (run_c3.ps1 starts it)
    python -m research.liquid.live.main --guard    # watchdog: flattens if the runner dies in a position
    python -m research.liquid.live.main --report   # evening report for every unreported session

--run trades one session and exits; on a closed day it says so and exits. It
refuses to start while data/research_liquid/live/KILL exists: every automatic
kill leaves that file, and it is removed only once the cause is logged in LOG.md.
Kill switch: create the KILL file; the runner flattens within seconds and stops.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.liquid.live import report, signal
from research.liquid.live.broker import AlpacaPaperBroker, load_keys
from research.liquid.live.feed import SIP_DELAY_MIN, AlpacaFeed
from research.liquid.live.ledger import Ledger
from research.liquid.live.runner import SYMBOL, Runner, Session, WallClock
from research.liquid.live.safety import (guard_action, read_heartbeat, single_instance, write_kill)
from research.liquid.provenance import git_state

ET = ZoneInfo("America/New_York")
LIVE_DIR = Path("data/research_liquid/live")
LEDGER = LIVE_DIR / "c3_ledger.sqlite"
KILL = LIVE_DIR / "KILL"
HEARTBEAT = LIVE_DIR / "heartbeat.json"
REPORTS = LIVE_DIR / "reports"
HISTORY_SESSIONS = 20
GUARD_POLL_S = 15.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("c3")


def sessions(broker, start: date, end: date) -> list[Session]:
    out = []
    for c in broker.calendar(start, end):
        o, k = pd.Timestamp(c.open), pd.Timestamp(c.close)
        o = o.tz_localize(ET) if o.tzinfo is None else o.tz_convert(ET)
        k = k.tz_localize(ET) if k.tzinfo is None else k.tz_convert(ET)
        out.append(Session(pd.Timestamp(c.date), o.to_pydatetime(), int((k - o).total_seconds() // 60)))
    return out


def plan_for(feed: AlpacaFeed, cal: list[Session], day: pd.Timestamp) -> tuple[signal.DayPlan, float]:
    """The day's plan from adjusted SIP history, and the raw previous close (logged for ex-dates)."""
    past = [s for s in cal if s.day < day][-HISTORY_SESSIONS:]
    hist = feed.history(SYMBOL, [(s.day, s.n_bars) for s in past])
    raw = feed.history(SYMBOL, [(past[-1].day, past[-1].n_bars)], adjustment="raw")
    today = next(s for s in cal if s.day == day)
    return signal.day_plan(hist, day, today.n_bars), float(raw.close[0, raw.n_bars[0] - 1])


def _clients():
    key, secret = load_keys()
    return AlpacaPaperBroker(key, secret), AlpacaFeed(key, secret)


def _today_session(broker) -> tuple[list[Session], Session | None]:
    today = pd.Timestamp(datetime.now(ET).date())
    cal = sessions(broker, (today - timedelta(days=45)).date(), today.date())
    return cal, (cal[-1] if cal and cal[-1].day == today else None)


# ------------------------------------------------------------------ commands
def cmd_check() -> int:
    broker, feed = _clients()
    acct, clk = broker.account(), broker.clock()
    today = datetime.now(ET).date()
    cal = sessions(broker, today - timedelta(days=45), today + timedelta(days=10))
    nxt = next(s for s in cal if s.day >= pd.Timestamp(today))
    plan, raw_prev = plan_for(feed, cal, nxt.day)
    beat, state = read_heartbeat(HEARTBEAT)
    print(json.dumps({
        "account": acct.__dict__, "shorting_enabled": broker.shorting_enabled(),
        "clock_open": clk.is_open, "next_open": str(clk.next_open),
        "position_SPY": broker.position_qty(SYMBOL), "open_orders_SPY": len(broker.open_orders(SYMBOL)),
        "kill_file": KILL.read_text(encoding="utf-8") if KILL.exists() else None,
        "heartbeat": {"ts": str(beat), "state": state},
        "next_session": {"day": str(nxt.day.date()), "open": str(nxt.open), "n_bars": nxt.n_bars},
        "plan": {"prev_close_adjusted": plan.prev_close, "prev_close_raw": raw_prev, "vol_14d": plan.vol,
                 "weight": plan.weight, "checks": len(plan.checks), "sigma_at_10:00": float(plan.sigma[29]),
                 "notional_at_weight": acct.last_equity * plan.weight},
        "code": git_state()}, indent=1, default=str))
    return 0


def cmd_drill() -> int:
    """Paper-account checks that need orders: shorting off, and a duplicate client id is rejected.
    Sends one 1-share buy limit at $1.00 (it cannot fill) and cancels it."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest
    broker, _ = _clients()
    out = {"shorting_enabled_before": broker.shorting_enabled()}
    if out["shorting_enabled_before"]:
        broker.disable_shorting()
    out["shorting_enabled_after"] = broker.shorting_enabled()
    coid = f"{report.DRILL_PREFIX}-{datetime.now(ET):%Y%m%d-%H%M%S}"
    req = LimitOrderRequest(symbol=SYMBOL, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                            limit_price=1.00, client_order_id=coid)
    first = broker._c.submit_order(req)
    try:
        dup = broker._c.submit_order(req)
        out["duplicate_client_id_rejected"] = False
        broker.cancel(str(dup.id))
    except Exception as e:                                            # noqa: BLE001
        out["duplicate_client_id_rejected"] = True
        out["duplicate_error"] = str(e)[:300]
    broker.cancel(str(first.id))
    time.sleep(2)
    out["drill_order_final"] = broker.get(str(first.id)).status
    out["open_orders_SPY"] = len(broker.open_orders(SYMBOL))
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / f"drill_{datetime.now(ET):%Y%m%d}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0 if out["duplicate_client_id_rejected"] and not out["shorting_enabled_after"] else 1


def cmd_run() -> int:
    if KILL.exists():
        logger.error("KILL file present, not trading:\n%s", KILL.read_text(encoding="utf-8"))
        return 2
    with single_instance(LIVE_DIR / "run.lock"):
        broker, feed = _clients()
        cal, sess = _today_session(broker)
        if sess is None:
            logger.info("today is not a trading session; nothing to do")
            return 0
        if datetime.now(ET) >= sess.bar_start(sess.n_bars - 1):
            logger.info("session already at its last minute; not starting")
            return 0
        if broker.shorting_enabled():
            broker.disable_shorting()
            logger.info("shorting disabled on the C3 paper account")
        plan, raw_prev = plan_for(feed, cal, sess.day)
        logger.info("plan %s: weight %.2f, vol %.4f, %d checks", sess.day.date(), plan.weight, plan.vol,
                    len(plan.checks))
        meta = {"code": git_state(), "prev_close_raw": raw_prev, "feed": f"sip>={SIP_DELAY_MIN}min+iex"}
        Runner(broker, feed, WallClock(), Ledger(LEDGER), kill_file=KILL, heartbeat=HEARTBEAT,
               meta=meta).run_session(sess, plan)
        logger.info("session done; the evening report runs from 16:20 ET (--report)")
    return 0


def cmd_guard() -> int:
    """Independent of the runner: flattens and leaves KILL if the runner dies in a position."""
    with single_instance(LIVE_DIR / "guard.lock"):
        broker = AlpacaPaperBroker(*load_keys())
        _, sess = _today_session(broker)
        if sess is None:
            return 0
        led = Ledger(LEDGER)
        day = str(sess.day.date())
        closeout = sess.bar_start(sess.n_bars - 1)
        until = closeout + timedelta(minutes=2)
        while (now := datetime.now(ET)) < until:
            if now < sess.open:
                time.sleep(min(60.0, (sess.open - now).total_seconds()))
                continue
            beat, _ = read_heartbeat(HEARTBEAT)
            try:
                pos, n_open = broker.position_qty(SYMBOL), len(broker.open_orders(SYMBOL))
            except Exception as e:                                    # noqa: BLE001  keep watching
                logger.warning("guard read failed: %s", e)
                time.sleep(GUARD_POLL_S)
                continue
            why = guard_action(now, beat, pos, n_open, sess.open, closeout)
            if why:
                led.log(now, day, "guard_flatten", reason=why, position=pos, open_orders=n_open)
                write_kill(KILL, f"guard: {why}", now)
                fills = broker.flatten(SYMBOL)
                led.log(datetime.now(ET), day, "guard_flattened", orders=[o.id for o in fills],
                        fills=[(o.filled_qty, o.filled_avg_price) for o in fills])
                logger.error("guard flattened: %s", why)
                return 1
            time.sleep(GUARD_POLL_S)
    return 0


def cmd_report() -> int:
    broker, feed = _clients()
    led = Ledger(LEDGER)
    first = min(led.days(), default=None)
    if first is None:
        logger.info("ledger is empty; nothing to report")
        return 0
    now = datetime.now(ET)
    cal = sessions(broker, (pd.Timestamp(first) - timedelta(days=45)).date(), now.date())
    REPORTS.mkdir(parents=True, exist_ok=True)
    done = [s for s in cal if str(s.day.date()) >= first
            and s.bar_end(s.n_bars - 1) + timedelta(minutes=SIP_DELAY_MIN + 4) <= now
            and not (REPORTS / f"{s.day.date()}.json").exists()]
    for s in done:                                    # every unreported session (G3 W11)
        d = str(s.day.date())
        window = [x for x in cal if x.day <= s.day][-(HISTORY_SESSIONS + 1):]
        m = feed.history(SYMBOL, [(x.day, x.n_bars) for x in window])
        live = led.trades(d)
        mids = {}
        for t in live:
            me = feed.mid_at(SYMBOL, report._ts(t["entry_filled_at"])) if t["entry_filled_at"] else np.nan
            mx = feed.mid_at(SYMBOL, report._ts(t["exit_filled_at"])) if t["exit_filled_at"] else np.nan
            mids[t["id"]] = (me, mx)
        closed = broker.closed_orders(SYMBOL, s.open - timedelta(hours=1), s.open + timedelta(hours=8))
        rep = report.day_report(s.day, m, live, led.events(d), closed, mids)
        rep["code"] = git_state()
        if not rep["safety"]["ok"]:
            write_kill(KILL, f"G-safety failed on {d}: {json.dumps(rep['safety'], default=str)[:400]}", now)
        (REPORTS / f"{d}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        logger.info("report %s: %d live, %d research trades, safety %s", d, len(live),
                    len(rep["research_trades"]), "ok" if rep["safety"]["ok"] else "FAIL")
    reps = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(REPORTS.glob("20*.json"))]
    summ = report.summary(reps) | {"code": git_state(), "written": now.isoformat()}
    (LIVE_DIR / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("days", "round_trips", "pnl_total")} |
                     {g: summ[g]["status"] for g in ("g_cost", "g_fidelity", "g_safety")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    for c in ("check", "drill", "run", "guard", "report"):
        g.add_argument(f"--{c}", action="store_true")
    a = ap.parse_args(argv)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = next(c for c in ("check", "drill", "run", "guard", "report") if getattr(a, c))
    return {"check": cmd_check, "drill": cmd_drill, "run": cmd_run, "guard": cmd_guard,
            "report": cmd_report}[cmd]()


if __name__ == "__main__":
    sys.exit(main())
