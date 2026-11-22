"""The broker seam for the C3 runner: a Protocol, Alpaca paper, and a fake for tests.

AlpacaPaperBroker is the only module that imports alpaca.trading. It
- asserts paper=True and a paper endpoint;
- uses only the dedicated C3 key pair, refusing the KELLY and LINEAR pairs that
  the live CI system trades;
- gives every order a deterministic client_order_id, never lets alpaca-py re-POST
  an order after a 429/504, and after any submit error looks the order up by its
  client_order_id before deciding it failed (G3 W6);
- sets an HTTP timeout on every request (G3 B3);
- treats a PATCH replace as done only once the new order reads back
  (fact sheet item 5: a success code does not guarantee the replace);
- flattens with `safety.flatten`: cancel, wait until no order is open, close,
  wait until flat (G3 B3).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

KEY_VAR, SECRET_VAR = "ALPACA_API_KEY_C3", "ALPACA_SECRET_KEY_C3"
FORBIDDEN_VARS = ("ALPACA_API_KEY", "ALPACA_API_KEY_KELLY", "ALPACA_API_KEY_LINEAR")
OPEN_STATUSES = frozenset({"new", "accepted", "pending_new", "partially_filled", "pending_replace",
                           "pending_cancel", "accepted_for_bidding", "held", "calculated"})
HTTP_TIMEOUT_S = 10.0
READ_RETRIES = 3


@dataclass(frozen=True)
class OrderState:
    id: str
    client_order_id: str
    side: str                  # buy | sell
    kind: str                  # limit | stop | market
    qty: int
    status: str
    filled_qty: int = 0
    filled_avg_price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    filled_at: str | None = None   # ISO timestamp from the broker

    @property
    def open(self) -> bool:
        return self.status in OPEN_STATUSES


@dataclass(frozen=True)
class AccountState:
    equity: float
    last_equity: float
    buying_power: float


class Broker(Protocol):
    def account(self) -> AccountState: ...
    def position_qty(self, symbol: str) -> int: ...
    def open_orders(self, symbol: str) -> list[OrderState]: ...
    def limit_buy(self, symbol: str, qty: int, limit: float, coid: str) -> OrderState: ...
    def stop_sell(self, symbol: str, qty: int, stop: float, coid: str) -> OrderState: ...
    def market_sell(self, symbol: str, qty: int, coid: str) -> OrderState: ...
    def replace_stop(self, order_id: str, stop: float, coid: str) -> OrderState: ...
    def cancel(self, order_id: str) -> None: ...
    def get(self, order_id: str) -> OrderState: ...
    def close_position(self, symbol: str) -> OrderState | None: ...
    def flatten(self, symbol: str) -> list[OrderState]: ...


# ----------------------------------------------------------------- Alpaca ---
def load_keys(env_file: Path | None = None) -> tuple[str, str]:
    from dotenv import dotenv_values
    if env_file is None:
        env_file = next(p / ".env" for p in Path(__file__).resolve().parents if (p / ".env").is_file())
    env = dotenv_values(env_file)
    key, secret = env.get(KEY_VAR), env.get(SECRET_VAR)
    if not key or not secret:
        raise SystemExit(f"{KEY_VAR}/{SECRET_VAR} missing from {env_file}: C3 needs its own paper account")
    for v in FORBIDDEN_VARS:
        if env.get(v) and env.get(v) == key:
            raise SystemExit(f"{KEY_VAR} equals {v}: that account belongs to another system")
    return key, secret


def with_timeout(client, seconds: float = HTTP_TIMEOUT_S) -> None:
    """alpaca-py sends requests with no timeout (common/rest.py); give every call one."""
    session = client._session
    send = session.request

    def request(method, url, **kw):
        kw.setdefault("timeout", seconds)
        return send(method, url, **kw)

    session.request = request


def _iso(x) -> str | None:
    return x.isoformat() if isinstance(x, datetime) else (str(x) if x is not None else None)


class AlpacaPaperBroker:
    def __init__(self, key: str, secret: str) -> None:
        from alpaca.trading.client import TradingClient
        self._c = TradingClient(key, secret, paper=True)
        base = str(getattr(self._c, "_base_url", ""))
        assert "paper" in base, f"refusing a non-paper endpoint: {base!r}"
        self._c._retry = 0                  # never re-send an order POST; _submit looks it up instead
        with_timeout(self._c)

    @staticmethod
    def _state(o) -> OrderState:
        def f(x):
            return float(x) if x is not None else None
        kind = str(getattr(o.order_type, "value", o.order_type))
        return OrderState(str(o.id), o.client_order_id, str(getattr(o.side, "value", o.side)), kind,
                          int(float(o.qty or 0)), str(getattr(o.status, "value", o.status)),
                          int(float(o.filled_qty or 0)), f(o.filled_avg_price), f(o.limit_price), f(o.stop_price),
                          _iso(o.filled_at))

    @staticmethod
    def _read(fn: Callable):
        """Idempotent reads retry; order submissions never do."""
        for k in range(READ_RETRIES):
            try:
                return fn()
            except Exception:                                         # noqa: BLE001
                if k == READ_RETRIES - 1:
                    raise
                time.sleep(1.0)

    def _submit(self, req) -> OrderState:
        try:
            return self._state(self._c.submit_order(req))
        except Exception:
            found = self.by_coid(req.client_order_id)   # accepted despite the error (504, timeout)
            if found is not None:
                return found
            raise

    def clock(self):
        return self._read(self._c.get_clock)

    def calendar(self, start, end):
        from alpaca.trading.requests import GetCalendarRequest
        return self._read(lambda: self._c.get_calendar(GetCalendarRequest(start=start, end=end)))

    def account(self) -> AccountState:
        a = self._read(self._c.get_account)
        return AccountState(float(a.equity), float(a.last_equity), float(a.buying_power))

    def shorting_enabled(self) -> bool:
        return bool(self._read(self._c.get_account_configurations).shorting_enabled)

    def disable_shorting(self) -> None:
        """An oversell can then only be rejected, never open a short."""
        cfg = self._read(self._c.get_account_configurations)
        cfg.shorting_enabled = False
        self._c.set_account_configurations(cfg)

    def position_qty(self, symbol: str) -> int:
        def read():
            try:
                return int(float(self._c.get_open_position(symbol).qty))
            except Exception as e:                       # alpaca raises APIError 404 when flat
                if "position does not exist" in str(e).lower() or "404" in str(e):
                    return 0
                raise
        return self._read(read)

    def open_orders(self, symbol: str) -> list[OrderState]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], limit=500)
        return [self._state(o) for o in self._read(lambda: self._c.get_orders(req))]

    def closed_orders(self, symbol: str, after: datetime, until: datetime) -> list[OrderState]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], after=after, until=until,
                               limit=500)
        return [self._state(o) for o in self._read(lambda: self._c.get_orders(req))]

    def by_coid(self, coid: str) -> OrderState | None:
        try:
            return self._state(self._c.get_order_by_client_id(coid))
        except Exception:                                             # noqa: BLE001
            return None

    def limit_buy(self, symbol: str, qty: int, limit: float, coid: str) -> OrderState:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        return self._submit(LimitOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY,
                                              time_in_force=TimeInForce.DAY, limit_price=limit,
                                              client_order_id=coid))

    def stop_sell(self, symbol: str, qty: int, stop: float, coid: str) -> OrderState:
        """GTC, so a runner that dies leaves the position protected overnight (G3 B3)."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest
        return self._submit(StopOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL,
                                             time_in_force=TimeInForce.GTC, stop_price=stop,
                                             client_order_id=coid))

    def market_sell(self, symbol: str, qty: int, coid: str) -> OrderState:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        return self._submit(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL,
                                               time_in_force=TimeInForce.DAY, client_order_id=coid))

    def replace_stop(self, order_id: str, stop: float, coid: str) -> OrderState:
        from alpaca.trading.requests import ReplaceOrderRequest
        try:
            return self._state(self._c.replace_order_by_id(order_id, ReplaceOrderRequest(
                stop_price=stop, client_order_id=coid)))
        except Exception:
            found = self.by_coid(coid)
            if found is not None:
                return found
            raise

    def cancel(self, order_id: str) -> None:
        self._c.cancel_order_by_id(order_id)

    def get(self, order_id: str) -> OrderState:
        return self._state(self._read(lambda: self._c.get_order_by_id(order_id)))

    def close_position(self, symbol: str) -> OrderState | None:
        try:
            return self._state(self._c.close_position(symbol))
        except Exception as e:
            if "position does not exist" in str(e).lower() or "404" in str(e):
                return None
            raise

    def flatten(self, symbol: str) -> list[OrderState]:
        from research.liquid.live.safety import flatten
        return flatten(self, symbol, now=time.monotonic, sleep=time.sleep)


# ------------------------------------------------------------------- fake ---
@dataclass
class FakeBroker:
    """In-memory broker for tests. Orders fill only when the test calls fill(), or by autofill.

    Optional realism, all off by default:
    - cancel_delay: a cancel leaves the order `pending_cancel` for this many reads;
    - reject_oversell: a sell for more than the shares not already held by open
      sells raises (Alpaca: "insufficient qty available", fact sheet item 5);
    - enforce_buying_power: a limit buy whose qty x limit exceeds buying power raises.
    """
    equity: float = 100_000.0
    buying_power: float = 400_000.0
    orders: dict[str, OrderState] = field(default_factory=dict)
    pos: int = 0
    calls: list[tuple] = field(default_factory=list)
    autofill_limit: float | None = None        # fill limit buys at once at this price
    autofill_limit_qty: int | None = None      # ...for this many shares (partial), else all
    autofill_market: float | None = None       # fill market sells at once at this price
    cancel_delay: int = 0
    reject_oversell: bool = False
    enforce_buying_power: bool = False
    clock: object | None = None                # stamps filled_at when set
    _pending: dict[str, int] = field(default_factory=dict)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def _new(self, coid, side, kind, qty, limit=None, stop=None, status="new") -> OrderState:
        if side == "sell" and self.reject_oversell and qty > self.pos - self._held():
            raise RuntimeError("insufficient qty available for order")
        o = OrderState(f"o{next(self._ids)}", coid, side, kind, qty, status, limit_price=limit, stop_price=stop)
        self.orders[o.id] = o
        if kind == "limit" and self.autofill_limit is not None:
            self.fill(o.id, self.autofill_limit, self.autofill_limit_qty)
        if kind == "market" and self.autofill_market is not None:
            self.fill(o.id, self.autofill_market)
        return self.orders[o.id]

    def _held(self) -> int:
        return sum(o.qty - o.filled_qty for o in self.orders.values() if o.side == "sell" and o.open)

    def _tick(self) -> None:
        for k in list(self._pending):
            self._pending[k] -= 1
            if self._pending[k] <= 0:
                del self._pending[k]
                o = self.orders[k]
                if o.status == "pending_cancel":
                    self.orders[k] = replace(o, status="canceled")

    def open_stop(self) -> OrderState | None:
        s = [o for o in self.orders.values() if o.kind == "stop" and o.open]
        return s[-1] if s else None

    def account(self) -> AccountState:
        return AccountState(self.equity, self.equity, self.buying_power)

    def position_qty(self, symbol: str) -> int:
        return self.pos

    def open_orders(self, symbol: str) -> list[OrderState]:
        self._tick()
        return [o for o in self.orders.values() if o.open]

    def closed_orders(self, symbol: str, after=None, until=None) -> list[OrderState]:
        return [o for o in self.orders.values() if not o.open]

    def by_coid(self, coid: str) -> OrderState | None:
        m = [o for o in self.orders.values() if o.client_order_id == coid]
        return m[-1] if m else None

    def limit_buy(self, symbol, qty, limit, coid):
        self.calls.append(("limit_buy", qty, limit, coid))
        if self.enforce_buying_power and qty * limit > self.buying_power:
            raise RuntimeError("insufficient buying power")
        return self._new(coid, "buy", "limit", qty, limit=limit)

    def stop_sell(self, symbol, qty, stop, coid):
        self.calls.append(("stop_sell", qty, stop, coid))
        return self._new(coid, "sell", "stop", qty, stop=stop)

    def market_sell(self, symbol, qty, coid):
        self.calls.append(("market_sell", qty, coid))
        return self._new(coid, "sell", "market", qty)

    def replace_stop(self, order_id, stop, coid):
        self.calls.append(("replace_stop", order_id, stop, coid))
        old = self.orders[order_id]
        if not old.open:
            raise RuntimeError("order is not open")
        self.orders[order_id] = replace(old, status="replaced")
        return self._new(coid, "sell", "stop", old.qty - old.filled_qty, stop=stop)

    def cancel(self, order_id):
        self.calls.append(("cancel", order_id))
        o = self.orders[order_id]
        if not o.open:
            raise RuntimeError(f"order is already {o.status}")          # Alpaca: 422
        if self.cancel_delay > 0:
            self.orders[order_id] = replace(o, status="pending_cancel")
            self._pending[order_id] = self.cancel_delay
        else:
            self.orders[order_id] = replace(o, status="canceled")

    def get(self, order_id):
        self._tick()
        return self.orders[order_id]

    def close_position(self, symbol):
        self.calls.append(("close_position",))
        if self.pos == 0:
            return None
        return self._new(f"close-{next(self._ids)}", "sell", "market", self.pos)

    def flatten(self, symbol):
        self.calls.append(("flatten",))
        for k, o in list(self.orders.items()):
            if o.open:
                self.orders[k] = replace(o, status="canceled")
        out = []
        if self.pos:
            o = OrderState(f"o{next(self._ids)}", f"flatten-{len(self.calls)}", "sell", "market", self.pos,
                           "filled", self.pos, self.autofill_market, filled_at=self._now())
            self.orders[o.id] = o
            out.append(o)
        self.pos = 0
        return out

    def _now(self) -> str | None:
        return self.clock.now().isoformat() if self.clock is not None else None

    # test hooks
    def fill(self, order_id: str, price: float, qty: int | None = None) -> None:
        o = self.orders[order_id]
        q = o.qty if qty is None else min(qty, o.qty)
        status = "filled" if q == o.qty else "partially_filled"
        self.orders[order_id] = replace(o, status=status, filled_qty=q, filled_avg_price=price,
                                        filled_at=self._now())
        self.pos += (q - o.filled_qty) if o.side == "buy" else -(q - o.filled_qty)

    def expire_partial(self, order_id: str) -> None:
        o = self.orders[order_id]
        self.orders[order_id] = replace(o, status="canceled")
