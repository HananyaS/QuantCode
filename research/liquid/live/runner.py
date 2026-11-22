"""The C3 session runner (PAPER_C3 §2): a wall-clock loop over one trading session.

Timeline for decide bar d (the bar ending at a check time HH:00/HH:30):
- bar_end(d) + 5 s : read today's bars, apply `signal.check`;
- in a position    : re-price the stop (PATCH), or sell at market if the new
                     level is at or above the price;
- flat, rule true  : buy limit at round(close x 1.001, 2); cancelled at
                     bar_end(d + 1) if unfilled; stop placed as soon as it fills;
- between checks   : the stop is polled; a fill closes the trade;
- bar_start(last)  : cancel the stop, confirm, sell at market.
A kill file, an unexpected broker state or any exception flattens the account.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from research.liquid import cells
from research.liquid.live import signal
from research.liquid.live.broker import Broker, OrderState
from research.liquid.live.feed import Feed
from research.liquid.live.ledger import Ledger

ET = ZoneInfo("America/New_York")
SYMBOL = "SPY"
MISSED_AFTER_S = 30.0          # a check more than this late is skipped, not acted on
ENTRY_POLL_S = 1.0             # keeps the entry loop near 60 req/min (Alpaca allows 200)


class Clock(Protocol):
    def now(self) -> datetime: ...
    def sleep(self, seconds: float) -> None: ...


class WallClock:
    def now(self) -> datetime:
        return datetime.now(ET)

    def sleep(self, seconds: float) -> None:
        time.sleep(max(seconds, 0.0))


@dataclass(frozen=True)
class Session:
    day: pd.Timestamp
    open: datetime                # tz-aware ET
    n_bars: int

    def bar_start(self, d: int) -> datetime:
        return self.open + timedelta(minutes=d)

    def bar_end(self, d: int) -> datetime:
        return self.open + timedelta(minutes=d + 1)


@dataclass
class Position:
    trade_id: int
    qty: int
    entry_price: float
    entry_check: int
    stop: OrderState | None
    stop_level: float
    exited_qty: int = 0            # shares already sold by a partially filled stop
    exited_value: float = 0.0


class Killed(RuntimeError):
    pass


class Runner:
    def __init__(self, broker: Broker, feed: Feed, clock: Clock, ledger: Ledger,
                 kill_file: Path | None = None, check_delay_s: float = 5.0, poll_s: float = 2.0,
                 confirm_timeout_s: float = 20.0, symbol: str = SYMBOL) -> None:
        self.b, self.f, self.clock, self.led = broker, feed, clock, ledger
        self.kill_file, self.delay, self.poll, self.timeout = kill_file, check_delay_s, poll_s, confirm_timeout_s
        self.sym = symbol
        self.pos: Position | None = None
        self.sess: Session | None = None
        self.plan: signal.DayPlan | None = None

    # -------------------------------------------------------------- utilities
    @property
    def day(self) -> str:
        return str(self.sess.day.date())

    def _log(self, event: str, **kw) -> None:
        self.led.log(self.clock.now(), self.day, event, **kw)

    def _check_kill(self) -> None:
        if self.kill_file is not None and self.kill_file.exists():
            raise Killed(f"kill file {self.kill_file}")

    def _wait_until(self, t: datetime) -> None:
        while True:
            self._check_kill()
            if self.pos is not None:
                self._poll_stop()
            now = self.clock.now()
            if now >= t:
                return
            self.clock.sleep(min(self.poll, (t - now).total_seconds()))

    def _settle(self, order_id: str) -> OrderState:
        """Poll until the order is no longer open."""
        end = self.clock.now() + timedelta(seconds=self.timeout)
        while True:
            o = self.b.get(order_id)
            if not o.open:
                return o
            if self.clock.now() >= end:
                raise Killed(f"order {order_id} still {o.status} after {self.timeout}s")
            self.clock.sleep(0.5)

    def _close_trade(self, price: float | None, kind: str, exit_check: int | None) -> None:
        p = self.pos
        if price is not None and p.exited_qty:
            price = (p.exited_value + price * p.qty) / (p.exited_qty + p.qty)   # VWAP of all exit fills
        self.led.close_trade(p.trade_id, exit_price=price, exit_ts=self.clock.now().isoformat(),
                             exit_kind=kind, exit_check=exit_check, stop_at_exit=p.stop_level)
        self._log("exit", trade_id=p.trade_id, exit_kind=kind, price=price, qty=p.qty, exit_check=exit_check)
        self.pos = None

    # --------------------------------------------------------------- session
    def run_session(self, sess: Session, plan: signal.DayPlan) -> None:
        self.sess, self.plan, self.pos = sess, plan, None
        try:
            self._reconcile("start")
            self._log("plan", weight=plan.weight, vol=plan.vol, prev_close=plan.prev_close,
                      checks=list(plan.checks), n_bars=plan.n_bars)
            if not plan.tradable:
                self._log("skip_day", reason="plan not tradable")
                return
            for d in plan.checks:
                due = sess.bar_end(d) + timedelta(seconds=self.delay)
                if self.clock.now() > due + timedelta(seconds=MISSED_AFTER_S):
                    self._log("check_missed", d=d, due=due.isoformat())     # started late: never act on a stale bar
                    continue
                self._wait_until(due)
                self._on_check(d)
            self._wait_until(sess.bar_start(plan.last))
            self._close_out()
            self._reconcile("end")
        except Killed as k:
            self._kill(str(k))
        except Exception as e:                                        # noqa: BLE001
            self._kill(f"{type(e).__name__}: {e}")
            raise

    def _on_check(self, d: int) -> None:
        bars = self.f.today(self.sym, self.sess.day, d)
        if not bars.complete:
            self._log("check_skipped", d=d, reason="no 09:30 bar in the feed")
            return
        r = signal.check(self.plan, bars.open0, bars.close, bars.volume, bars.vwap, d)
        self._log("check", d=d, price=r.price, ub=r.ub, vwap=r.vwap, enter=r.enter, stop=r.stop,
                  in_position=self.pos is not None)
        forced = False
        if self.pos is not None:
            forced = self._restop(r)
        if self.pos is None and not forced and r.enter:
            self._enter(r)

    # ----------------------------------------------------------------- entry
    def _enter(self, r: signal.CheckResult) -> None:
        acct = self.b.account()
        qty = signal.shares(acct.last_equity, self.plan.weight, r.price, acct.buying_power)
        if qty < 1:
            self._log("entry_skipped", d=r.d, reason="zero shares")
            return
        limit = signal.tick(r.price * (1 + cells.COLLAR))
        o = self.b.limit_buy(self.sym, qty, limit, f"c3-{self.day}-{r.d}-entry")
        self._log("entry_sent", d=r.d, qty=qty, limit=limit, order_id=o.id, equity=acct.last_equity)
        deadline = self.sess.bar_end(r.d + 1)
        while True:
            o = self.b.get(o.id)
            if o.status == "filled" or self.clock.now() >= deadline:
                break
            self._check_kill()
            self.clock.sleep(ENTRY_POLL_S)
        if o.open:
            self.b.cancel(o.id)
            o = self._settle(o.id)
        if o.filled_qty < 1:
            self._log("entry_unfilled", d=r.d, status=o.status)
            return
        tid = self.led.open_trade(day=self.day, entry_check=r.d, qty=o.filled_qty, decision_price=r.price,
                                  limit_price=limit, entry_price=o.filled_avg_price,
                                  entry_ts=self.clock.now().isoformat(), stop_at_entry=r.stop,
                                  weight=self.plan.weight, equity=acct.last_equity)
        self.pos = Position(tid, o.filled_qty, o.filled_avg_price, r.d, None, r.stop)
        self._log("entry_filled", d=r.d, qty=o.filled_qty, price=o.filled_avg_price, trade_id=tid)
        self._place_stop(r.stop, r.d, reference=o.filled_avg_price)

    # ------------------------------------------------------------------ stops
    def _place_stop(self, level: float, d: int, reference: float) -> None:
        lvl = signal.tick(level)
        self.pos.stop_level = level
        if lvl >= reference:                              # the research stop triggers on the next bar
            self._market_exit("sl", d, reason="stop at or above price")
            return
        try:
            self.pos.stop = self.b.stop_sell(self.sym, self.pos.qty, lvl, f"c3-{self.day}-{self.pos.entry_check}-stop-{d}")
            self._log("stop_placed", d=d, level=lvl, order_id=self.pos.stop.id)
        except Exception as e:                                        # noqa: BLE001
            self._log("stop_rejected", d=d, level=lvl, error=str(e))
            self._market_exit("sl", d, reason="stop rejected")

    def _poll_stop(self) -> None:
        p = self.pos
        if p is None or p.stop is None:
            return
        o = self.b.get(p.stop.id)
        p.stop = o
        if o.status == "filled":
            self._close_trade(o.filled_avg_price, "sl", None)
        elif not o.open and o.status != "replaced":
            self._log("stop_lost", status=o.status)
            raise Killed(f"stop order {o.id} became {o.status} while in a position")

    def _restop(self, r: signal.CheckResult) -> bool:
        """At a check while in a position. True if the position was closed at this check."""
        self._poll_stop()
        if self.pos is None:
            return False                                   # stopped between checks: re-entry allowed
        lvl = signal.tick(r.stop)
        if lvl >= r.price:
            self._cancel_stop()
            if self.pos is not None:
                self._market_exit("sl", r.d, reason="new stop at or above price")
            return True
        if self.pos.stop is not None and self.pos.stop.stop_price == lvl:
            self.pos.stop_level = r.stop
            return False
        try:
            new = self.b.replace_stop(self.pos.stop.id, lvl, f"c3-{self.day}-{self.pos.entry_check}-stop-{r.d}")
        except Exception as e:                                        # noqa: BLE001
            self._log("replace_failed", d=r.d, error=str(e))
            self._poll_stop()
            if self.pos is None:
                return True
            raise Killed(f"stop replace failed and the stop is not filled: {e}")
        got = self.b.get(new.id)
        if not (got.open and got.stop_price == lvl):
            raise Killed(f"replaced stop reads back {got.status} @ {got.stop_price}, wanted {lvl}")
        self.pos.stop, self.pos.stop_level = got, r.stop
        self._log("stop_replaced", d=r.d, level=lvl, order_id=got.id)
        return False

    def _cancel_stop(self) -> None:
        p = self.pos
        if p.stop is None:
            return
        self.b.cancel(p.stop.id)
        o = self._settle(p.stop.id)
        p.stop = o
        if o.status == "filled":
            self._close_trade(o.filled_avg_price, "sl", None)
        elif o.filled_qty:                                 # partly filled, rest cancelled
            p.exited_qty += o.filled_qty
            p.exited_value += o.filled_qty * o.filled_avg_price
            p.qty -= o.filled_qty
            self._log("stop_partial", filled=o.filled_qty, price=o.filled_avg_price, remaining=p.qty)
            p.stop = None

    def _market_exit(self, kind: str, d: int | None, reason: str) -> None:
        o = self.b.market_sell(self.sym, self.pos.qty, f"c3-{self.day}-{self.pos.entry_check}-{kind}-mkt-{d}")
        self._log("market_exit_sent", d=d, reason=reason, order_id=o.id)
        o = self._settle(o.id)
        if o.status != "filled":
            raise Killed(f"market exit {o.id} ended {o.status}")
        self._close_trade(o.filled_avg_price, kind, d)

    # ------------------------------------------------------------------ close
    def _close_out(self) -> None:
        if self.pos is None:
            return
        self._cancel_stop()
        if self.pos is not None:
            self._market_exit("time", self.plan.last, reason="session close")

    def _reconcile(self, when: str) -> None:
        q = self.b.position_qty(self.sym)
        oo = self.b.open_orders(self.sym)
        self._log("reconcile", when=when, position=q, open_orders=len(oo))
        if q != 0 or oo:
            raise Killed(f"reconcile {when}: position {q}, {len(oo)} open orders")

    def _kill(self, reason: str) -> None:
        self._log("kill", reason=reason)
        try:
            self.b.flatten(self.sym)
        finally:
            if self.pos is not None:
                self._close_trade(None, "kill", None)
