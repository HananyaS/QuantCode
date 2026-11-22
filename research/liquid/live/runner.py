"""The C3 session runner (PAPER_C3 §2): a wall-clock loop over one trading session.

Timeline for decide bar d (the bar ending at a check time HH:00/HH:30):
- bar_end(d) + 5 s : read today's bars (waiting up to 25 s for bar d itself), apply `signal.check`;
- in a position    : re-price the stop (PATCH), or sell at market if the new
                     level is at or above the price;
- flat, rule true  : buy limit at floor(close x 1.001); on the first fill the rest is
                     cancelled and the stop placed; unfilled by bar_end(d + 1) -> cancelled;
- between checks   : the stop is polled; a fill closes the trade;
- bar_start(last)  : cancel the stop, confirm, sell at market.
Prices sent are floored to the cent (G3 W1). The session's first act is a reconcile at
the open: the broker must be flat and the ledger must hold no unclosed trade.
A kill file, an unexpected broker state or any exception writes KILL, flattens, and stops;
the next session will not start until KILL is removed (G3 B4).
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
from research.liquid.live.safety import write_heartbeat, write_kill

ET = ZoneInfo("America/New_York")
SYMBOL = "SPY"
MISSED_AFTER_S = 30.0          # a check more than this late is skipped, not acted on
ENTRY_POLL_S = 1.0             # keeps the entry loop near 60 req/min (Alpaca allows 200)
FEED_WAIT_S = 25.0             # wait this long past the bar's end for the decision bar (G3 W3)
FEED_RETRY_S = 2.0
RECONCILE_AFTER_S = 5.0        # the start reconcile runs at the open + 5 s
BP_MARGIN = 0.99               # resize to this share of buying power after a rejection (G3 W8)
MISSING_BAR = "skip"           # decision bar still missing after FEED_WAIT_S: "skip" the check or "carry"


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
    stop_level: float              # research level max(UB, VWAP), unrounded
    exited_qty: int = 0            # shares already sold by a partially filled stop
    exited_value: float = 0.0
    stop_sent: float | None = None # the level actually sent (floored)
    n_stops: int = 0               # stop orders placed or replaced, for unique client ids


class Killed(RuntimeError):
    pass


class Runner:
    def __init__(self, broker: Broker, feed: Feed, clock: Clock, ledger: Ledger,
                 kill_file: Path | None = None, check_delay_s: float = 5.0, poll_s: float = 2.0,
                 confirm_timeout_s: float = 20.0, symbol: str = SYMBOL,
                 heartbeat: Path | None = None, meta: dict | None = None,
                 missing_bar: str = MISSING_BAR) -> None:
        assert missing_bar in ("skip", "carry"), missing_bar
        self.b, self.f, self.clock, self.led = broker, feed, clock, ledger
        self.kill_file, self.delay, self.poll, self.timeout = kill_file, check_delay_s, poll_s, confirm_timeout_s
        self.sym, self.heartbeat, self.meta, self.missing_bar = symbol, heartbeat, meta or {}, missing_bar
        self.pos: Position | None = None
        self.sess: Session | None = None
        self.plan: signal.DayPlan | None = None

    # -------------------------------------------------------------- utilities
    @property
    def day(self) -> str:
        return str(self.sess.day.date())

    def _log(self, event: str, **kw) -> None:
        self.led.log(self.clock.now(), self.day, event, **kw)

    def _beat(self, state: str | None = None) -> None:
        write_heartbeat(self.heartbeat, self.clock.now(), state or ("in_position" if self.pos else "flat"))

    def _check_kill(self) -> None:
        if self.kill_file is not None and self.kill_file.exists():
            raise Killed(f"kill file {self.kill_file}")

    def _wait_until(self, t: datetime) -> None:
        while True:
            self._check_kill()
            if self.pos is not None:
                self._poll_stop()
            self._beat()
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
            self._beat()
            self.clock.sleep(0.5)

    def _cancel_settle(self, order_id: str) -> OrderState:
        """Cancel, tolerating the order having just filled or closed (G3 W5); return its final state."""
        try:
            self.b.cancel(order_id)
        except Exception as e:                                        # noqa: BLE001
            o = self.b.get(order_id)
            self._log("cancel_raced", order_id=order_id, status=o.status, error=str(e))
            if o.open and o.status != "pending_cancel":
                raise Killed(f"cannot cancel {order_id} ({o.status}): {e}") from e
        return self._settle(order_id)

    def _stop_coid(self, d: int) -> str:
        p = self.pos
        p.n_stops += 1
        return f"c3-{self.day}-{p.entry_check}-s{p.n_stops}-{d}"

    def _close_trade(self, price: float | None, kind: str, exit_check: int | None,
                     order: OrderState | None = None) -> None:
        p = self.pos
        if price is not None and p.exited_qty:
            price = (p.exited_value + price * p.qty) / (p.exited_qty + p.qty)   # VWAP of all exit fills
        self.led.close_trade(p.trade_id, exit_price=price, exit_ts=self.clock.now().isoformat(),
                             exit_kind=kind, exit_check=exit_check, stop_at_exit=p.stop_sent,
                             stop_level=p.stop_level, exit_order_id=order.id if order else None,
                             exit_filled_at=order.filled_at if order else None)
        self._log("exit", trade_id=p.trade_id, exit_kind=kind, price=price, qty=p.qty, exit_check=exit_check,
                  order_id=order.id if order else None)
        self.pos = None

    # --------------------------------------------------------------- session
    def run_session(self, sess: Session, plan: signal.DayPlan) -> None:
        self.sess, self.plan, self.pos = sess, plan, None
        try:
            self._log("plan", weight=plan.weight, vol=plan.vol, prev_close=plan.prev_close,
                      checks=list(plan.checks), n_bars=plan.n_bars, **self.meta)
            stale = self.led.unclosed_before(self.day)
            if stale:
                raise Killed(f"ledger holds {len(stale)} unclosed trade(s) from {stale[0]['day']}: a run died")
            self._wait_until(sess.open + timedelta(seconds=RECONCILE_AFTER_S))
            self._reconcile("start")
            if not plan.tradable:
                self._log("skip_day", reason="plan not tradable")
                self._beat("done")
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
            self._beat("done")
        except Killed as k:
            self._kill(str(k))
        except Exception as e:                                        # noqa: BLE001
            self._kill(f"{type(e).__name__}: {e}")
            raise

    def _on_check(self, d: int) -> None:
        end = self.sess.bar_end(d) + timedelta(seconds=FEED_WAIT_S)
        while True:
            bars = self.f.today(self.sym, self.sess.day, d)
            if bars.complete and bars.has_upto:
                break
            if self.clock.now() >= end:
                if bars.complete and self.missing_bar == "carry":
                    self._log("decision_bar_carried", d=d, n_carried=bars.n_carried)   # no trade that minute
                    break
                why = "no 09:30 bar in the feed" if not bars.complete else f"bar {d} not published"
                self._log("check_skipped", d=d, reason=why, in_position=self.pos is not None)
                return
            self._check_kill()
            if self.pos is not None:
                self._poll_stop()
            self._beat()
            self.clock.sleep(FEED_RETRY_S)
        r = signal.check(self.plan, bars.open0, bars.close, bars.volume, bars.vwap, d)
        self._log("check", d=d, price=r.price, ub=r.ub, vwap=r.vwap, enter=r.enter, stop=r.stop,
                  in_position=self.pos is not None, n_carried=bars.n_carried)
        forced = False
        if self.pos is not None:
            forced = self._restop(r)
        if self.pos is None and not forced and r.enter:
            self._enter(r)

    # ----------------------------------------------------------------- entry
    def _send_entry(self, r: signal.CheckResult, limit: float, acct) -> tuple[OrderState | None, int]:
        coid = f"c3-{self.day}-{r.d}-entry"
        qty = signal.shares(acct.last_equity, self.plan.weight, r.price, acct.buying_power)
        if qty < 1:
            self._log("entry_skipped", d=r.d, reason="zero shares")
            return None, 0
        try:
            return self.b.limit_buy(self.sym, qty, limit, coid), qty
        except Exception as e:                                        # noqa: BLE001
            if "buying power" not in str(e).lower():
                raise
            qty2 = signal.shares(acct.last_equity, self.plan.weight, limit, BP_MARGIN * acct.buying_power)
            self._log("entry_resized", d=r.d, qty=qty, qty_new=qty2, error=str(e))
            if qty2 < 1:
                return None, 0
            try:
                return self.b.limit_buy(self.sym, qty2, limit, coid + "-r"), qty2
            except Exception as e2:                                   # noqa: BLE001
                if "buying power" not in str(e2).lower():
                    raise
                self._log("entry_skipped", d=r.d, reason="buying power", error=str(e2))
                return None, 0

    def _enter(self, r: signal.CheckResult) -> None:
        acct = self.b.account()
        limit = signal.floor_tick(r.price * (1 + cells.COLLAR))
        o, qty = self._send_entry(r, limit, acct)
        if o is None:
            return
        self._log("entry_sent", d=r.d, qty=qty, limit=limit, order_id=o.id, equity=acct.last_equity)
        deadline = self.sess.bar_end(r.d + 1)
        while True:
            o = self.b.get(o.id)
            if o.filled_qty > 0 or not o.open or self.clock.now() >= deadline:
                break
            self._check_kill()
            self._beat("entry")
            self.clock.sleep(ENTRY_POLL_S)
        if o.open:                                   # first fill: protect it now, drop the rest (G3 W4)
            o = self._cancel_settle(o.id)
        if o.filled_qty < 1:
            self._log("entry_unfilled", d=r.d, status=o.status)
            return
        tid = self.led.open_trade(day=self.day, entry_check=r.d, qty=o.filled_qty, decision_price=r.price,
                                  limit_price=limit, entry_price=o.filled_avg_price,
                                  entry_ts=self.clock.now().isoformat(), stop_at_entry=r.stop,
                                  weight=self.plan.weight, equity=acct.last_equity,
                                  entry_order_id=o.id, entry_filled_at=o.filled_at)
        self.pos = Position(tid, o.filled_qty, o.filled_avg_price, r.d, None, r.stop)
        self._log("entry_filled", d=r.d, qty=o.filled_qty, price=o.filled_avg_price, trade_id=tid,
                  order_id=o.id, filled_at=o.filled_at, partial=o.filled_qty < qty)
        self._place_stop(r.stop, r.d, reference=o.filled_avg_price)

    # ------------------------------------------------------------------ stops
    def _place_stop(self, level: float, d: int, reference: float) -> None:
        p = self.pos
        lvl = signal.floor_tick(level)
        p.stop_level, p.stop_sent = level, lvl
        if lvl >= reference:                              # the research stop triggers on the next bar
            self._market_exit("sl", d, reason="stop at or above price")
            return
        try:
            p.stop = self.b.stop_sell(self.sym, p.qty, lvl, self._stop_coid(d))
        except Exception as e:                                        # noqa: BLE001
            self._log("stop_rejected", d=d, level=lvl, error=str(e))
            self._market_exit("sl", d, reason="stop rejected")
            return
        self._log("stop_placed", d=d, level=lvl, order_id=p.stop.id, trade_id=p.trade_id, qty=p.qty)

    def _poll_stop(self) -> None:
        p = self.pos
        if p is None or p.stop is None:
            return
        o = self.b.get(p.stop.id)
        p.stop = o
        if o.status == "filled":
            self._close_trade(o.filled_avg_price, "sl", None, o)
        elif not o.open and o.status != "replaced":
            self._log("stop_lost", status=o.status, order_id=o.id, filled=o.filled_qty)
            raise Killed(f"stop order {o.id} became {o.status} while in a position")

    def _restop(self, r: signal.CheckResult) -> bool:
        """At a check while in a position. True if the position was closed at this check."""
        self._poll_stop()
        if self.pos is None:
            return False                                   # stopped between checks: re-entry allowed
        p = self.pos
        lvl = signal.floor_tick(r.stop)
        if lvl >= r.price:
            self._cancel_stop()
            if self.pos is not None:
                self._market_exit("sl", r.d, reason="new stop at or above price")
            return True
        if p.stop is not None and p.stop.status == "partially_filled":
            self._cancel_stop()                            # never PATCH a partly filled stop (G3 W5)
            if self.pos is None:
                return True
        if p.stop is None:
            self._place_stop(r.stop, r.d, reference=r.price)
            return self.pos is None
        if p.stop.stop_price == lvl:
            p.stop_level = r.stop
            return False
        try:
            new = self.b.replace_stop(p.stop.id, lvl, self._stop_coid(r.d))
        except Exception as e:                                        # noqa: BLE001
            self._log("replace_failed", d=r.d, level=lvl, error=str(e))
            old = self.b.get(p.stop.id)
            p.stop = old
            if old.status == "filled":
                self._close_trade(old.filled_avg_price, "sl", None, old)
                return True
            if old.status == "replaced":
                raise Killed(f"replace of {old.id} failed but the order reads 'replaced'") from e
            if old.open:
                self._cancel_stop()
            else:
                p.stop = None
            if self.pos is not None:                       # replace refused: exit as the new stop would
                self._market_exit("sl", r.d, reason="stop replace refused")
            return True
        got = self.b.get(new.id)
        if got.status == "filled":
            p.stop = got
            self._close_trade(got.filled_avg_price, "sl", None, got)
            return True
        if not (got.open and got.stop_price == lvl and got.qty - got.filled_qty == p.qty):
            raise Killed(f"replaced stop reads back {got.status} {got.qty - got.filled_qty} @ {got.stop_price}, "
                         f"wanted {p.qty} @ {lvl}")
        p.stop, p.stop_level, p.stop_sent = got, r.stop, lvl
        self._log("stop_replaced", d=r.d, level=lvl, order_id=got.id)
        return False

    def _cancel_stop(self) -> None:
        p = self.pos
        if p is None or p.stop is None:
            return
        self._log("stop_cancel_sent", order_id=p.stop.id, trade_id=p.trade_id)
        o = self._cancel_settle(p.stop.id)
        if o.status == "filled":
            p.stop = o
            self._close_trade(o.filled_avg_price, "sl", None, o)
            return
        if o.filled_qty:                                   # partly filled, rest cancelled
            p.exited_qty += o.filled_qty
            p.exited_value += o.filled_qty * o.filled_avg_price
            p.qty -= o.filled_qty
            self._log("stop_partial", order_id=o.id, filled=o.filled_qty, price=o.filled_avg_price,
                      remaining=p.qty)
        p.stop = None

    def _market_exit(self, kind: str, d: int | None, reason: str) -> None:
        p = self.pos
        o = self.b.market_sell(self.sym, p.qty, f"c3-{self.day}-{p.entry_check}-{kind}-mkt-{d}")
        self._log("market_exit_sent", d=d, reason=reason, order_id=o.id, qty=p.qty)
        o = self._settle(o.id)
        if o.status != "filled":
            if o.filled_qty:
                p.exited_qty += o.filled_qty
                p.exited_value += o.filled_qty * o.filled_avg_price
                p.qty -= o.filled_qty
            raise Killed(f"market exit {o.id} ended {o.status} with {o.filled_qty} filled")
        self._close_trade(o.filled_avg_price, kind, d, o)

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
        write_kill(self.kill_file, reason, self.clock.now())
        fills: list[OrderState] = []
        try:
            fills = self.b.flatten(self.sym) or []
            self._log("flattened", orders=[o.id for o in fills], position=self.b.position_qty(self.sym))
        finally:
            if self.pos is not None:
                done = [o for o in fills if o.filled_qty and o.filled_avg_price is not None]
                px = None
                if done:
                    px = sum(o.filled_qty * o.filled_avg_price for o in done) / sum(o.filled_qty for o in done)
                self._close_trade(px, "kill", None, done[-1] if done else None)
            self._beat("killed")
