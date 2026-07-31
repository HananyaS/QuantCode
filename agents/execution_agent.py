"""ExecutionAgent: submits paper-trading orders to reconcile the broker
account with today's target portfolio weights.

PAPER TRADING ONLY
-------------------
`paper` must be True. This project has not passed the live go-live gate
(see utils/go_live_gate.py, ROADMAP.md Phase 4) — real-money execution is
explicitly out of scope until that changes deliberately, so this asserts
rather than silently defaulting.

Order sizing
------------
target_dollars[ticker]  = target_weight[ticker] * account_equity
current_dollars[ticker] = broker's current market value for that position
delta_dollars            = target_dollars - current_dollars
Orders below `min_order_notional` are skipped (avoids churn on noise-level
rebalances). Share quantity is floor(|delta_dollars| / last_close) — whole
shares only.

Same-day round-trip (PDT) guard
----------------------------------
A sell for any ticker already bought today (per the StateStore ledger) is
skipped and logged — protects a sub-$25k account from tripping the Pattern
Day Trader rule (see ROADMAP.md Phase 0).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.state_store import StateStore

logger = get_logger(__name__)


class ExecutionAgent(BaseAgent):
    """Reconciles the broker account to today's target portfolio weights.

    Reads
    -----
    context['portfolio_weights'] : pd.DataFrame — last row used as today's targets
    context['universe_data']     : Dict[str, pd.DataFrame] — last Close used for pricing

    Writes
    ------
    context['execution_orders']  : List[dict] — one entry per ticker considered
    context['execution_account'] : Dict — equity/cash/positions snapshot after submission
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        state_store: Optional[StateStore] = None,
        min_order_notional: float = 50.0,
        trading_client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        assert paper is True, (
            "ExecutionAgent is paper-trading only for now — see ROADMAP.md Phase 4 "
            "for the deliberate go-live gate this project has not yet passed."
        )
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.state_store = state_store or StateStore()
        self.min_order_notional = min_order_notional
        self._client_factory = trading_client_factory or self._default_client

    def _default_client(self):
        from alpaca.trading.client import TradingClient
        return TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper=self.paper)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        assert context.get("portfolio_weights") is not None, (
            "ExecutionAgent requires context['portfolio_weights']"
        )
        assert context.get("universe_data") is not None, (
            "ExecutionAgent requires context['universe_data']"
        )

        weights: pd.DataFrame = context["portfolio_weights"]
        universe_data: Dict[str, pd.DataFrame] = context["universe_data"]
        assert len(weights) > 0, "portfolio_weights is empty"

        target_weights = weights.iloc[-1]
        run_date = str(pd.Timestamp(weights.index[-1]).date())

        client = self._client_factory()
        account = client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        positions_list = client.get_all_positions()
        broker_positions = {p.symbol: float(p.market_value) for p in positions_list}
        # Actual sellable share counts. Sells MUST be sized in shares the
        # broker holds, not dollars-at-a-stale-price: observed live, a
        # go-flat sell sized as market_value / yesterday's close requested
        # 301 shares against 276 held after a +9% intraday bounce -- Alpaca
        # rejected it and the de-risk never executed. qty_available (net of
        # shares already held for open orders) preferred over qty.
        broker_qty: Dict[str, float] = {}
        for p in positions_list:
            q = getattr(p, "qty_available", None)
            if q is None:
                q = getattr(p, "qty", None)
            if q is not None:
                broker_qty[p.symbol] = float(q)
        already_bought_today = self.state_store.tickers_bought_on(run_date)

        # Sells first: on a rotation (e.g. TQQQ -> QQQ) the sale's proceeds
        # fund the buy. Submitting the buy first only works while margin
        # buying power happens to cover holding both legs at once -- fully
        # invested at 1x and rotating all of it, Reg-T buying power covers
        # the buy only exactly, with no slack for an intraday dip.
        deltas = []
        for ticker, target_w in target_weights.items():
            target_dollars = float(target_w) * equity
            delta_dollars = target_dollars - broker_positions.get(ticker, 0.0)
            deltas.append((ticker, delta_dollars, float(target_w)))
        deltas.sort(key=lambda entry: entry[1] >= 0)  # sells (negative) first

        orders = []
        for ticker, delta_dollars, target_w in deltas:
            if abs(delta_dollars) < self.min_order_notional:
                continue

            side = "buy" if delta_dollars > 0 else "sell"

            if side == "sell" and ticker in already_bought_today:
                logger.warning(
                    "ExecutionAgent: skipping same-day round-trip sell for %s "
                    "(bought earlier today — PDT guard)", ticker,
                )
                orders.append({
                    "date": run_date, "ticker": ticker, "side": side,
                    "qty": 0, "status": "skipped_pdt_guard",
                })
                continue

            price = self._last_close(universe_data, ticker)
            if price is None or price <= 0:
                orders.append({
                    "date": run_date, "ticker": ticker, "side": side,
                    "qty": 0, "status": "skipped_no_price",
                })
                continue

            qty = math.floor(abs(delta_dollars) / price)
            if side == "sell":
                available = broker_qty.get(ticker)
                if available is not None:
                    if target_w == 0.0:
                        # Full exit: sell the position itself. The dollar
                        # estimate over-requests after an intraday rally
                        # (rejected outright) and under-sells leftover
                        # shares after a drop.
                        qty = int(available)
                    else:
                        qty = min(qty, int(available))
            if qty <= 0:
                continue

            result = self._submit_order(client, ticker, qty, side)
            self.state_store.record_order(
                run_date, ticker, side, qty, result["order_id"], result["status"]
            )
            orders.append({
                "date": run_date, "ticker": ticker, "side": side, "qty": qty, **result,
            })

        self.state_store.record_account_snapshot(run_date, equity, cash, broker_positions)

        context["execution_orders"] = orders
        context["execution_account"] = {"equity": equity, "cash": cash, "positions": broker_positions}
        logger.info(
            "ExecutionAgent: %d orders submitted | equity=%.2f | run_date=%s",
            sum(1 for o in orders if o.get("qty", 0) > 0), equity, run_date,
        )
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_close(universe_data: Dict[str, pd.DataFrame], ticker: str) -> Optional[float]:
        df = universe_data.get(ticker)
        if df is None or len(df) == 0:
            return None
        return float(df["Close"].iloc[-1])

    @staticmethod
    def _submit_order(client: Any, ticker: str, qty: int, side: str) -> Dict[str, str]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(request)
        return {"order_id": str(order.id), "status": str(order.status)}
