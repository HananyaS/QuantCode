"""Tests for agents/execution_agent.py — paper-trading order submission.

Uses a fake Alpaca TradingClient (dependency-injected via
`trading_client_factory`) so these tests never touch the network or a real
paper account.
"""
import numpy as np
import pandas as pd
import pytest

from agents.execution_agent import ExecutionAgent
from utils.state_store import StateStore


class FakeAccount:
    def __init__(self, equity, cash):
        self.equity = equity
        self.cash = cash


class FakePosition:
    def __init__(self, symbol, market_value, qty=None):
        self.symbol = symbol
        self.market_value = market_value
        if qty is not None:
            self.qty = qty
            self.qty_available = qty


class FakeOrder:
    def __init__(self, order_id, status):
        self.id = order_id
        self.status = status


class FakeTradingClient:
    """Records submitted orders; account/positions are pre-seeded."""

    def __init__(self, equity=100_000.0, cash=50_000.0, positions=None):
        self._equity = equity
        self._cash = cash
        self._positions = positions or {}
        self.submitted_orders = []

    def get_account(self):
        return FakeAccount(self._equity, self._cash)

    def get_all_positions(self):
        # Position values are either a bare market_value (float) or a
        # (market_value, qty) tuple mirroring the real API's share count.
        out = []
        for sym, v in self._positions.items():
            if isinstance(v, tuple):
                out.append(FakePosition(sym, v[0], qty=v[1]))
            else:
                out.append(FakePosition(sym, v))
        return out

    def submit_order(self, order_data):
        self.submitted_orders.append(order_data)
        side = order_data.side.value if hasattr(order_data.side, "value") else str(order_data.side)
        return FakeOrder(order_id=f"fake-{len(self.submitted_orders)}", status="accepted")


def _universe_data(prices: dict) -> dict:
    """Dict[ticker, single-row-ish OHLCV DataFrame] with the given last Close."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    universe = {}
    for ticker, price in prices.items():
        close = np.array([price, price, price], dtype=float)
        universe[ticker] = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000.0},
            index=dates,
        )
    return universe


def _weights(tickers, values, date="2024-01-03"):
    return pd.DataFrame([values], index=[pd.Timestamp(date)], columns=tickers)


@pytest.fixture
def store(tmp_path):
    return StateStore(db_path=str(tmp_path / "live_state.db"))


# ---------------------------------------------------------------------------
# Paper-only guard
# ---------------------------------------------------------------------------

def test_refuses_non_paper_trading(store):
    with pytest.raises(AssertionError, match="paper"):
        ExecutionAgent(api_key="k", secret_key="s", paper=False, state_store=store)


# ---------------------------------------------------------------------------
# Order sizing
# ---------------------------------------------------------------------------

def test_submits_buy_order_to_reach_target_weight(store):
    client = FakeTradingClient(equity=100_000.0, cash=100_000.0, positions={})
    universe = _universe_data({"AAPL": 100.0})
    weights = _weights(["AAPL"], [0.5])  # target: $50,000 of AAPL

    agent = ExecutionAgent(
        api_key="k", secret_key="s", state_store=store,
        trading_client_factory=lambda: client,
    )
    ctx = agent.run({"portfolio_weights": weights, "universe_data": universe})

    assert len(client.submitted_orders) == 1
    order = client.submitted_orders[0]
    assert order.symbol == "AAPL"
    assert order.qty == 500  # $50,000 / $100 per share
    assert ctx["execution_orders"][0]["side"] == "buy"


def test_submits_sell_order_to_reduce_overweight_position(store):
    client = FakeTradingClient(equity=100_000.0, cash=0.0, positions={"AAPL": 80_000.0})
    universe = _universe_data({"AAPL": 100.0})
    weights = _weights(["AAPL"], [0.5])  # target: $50,000, currently $80,000 -> sell $30,000

    agent = ExecutionAgent(
        api_key="k", secret_key="s", state_store=store,
        trading_client_factory=lambda: client,
    )
    ctx = agent.run({"portfolio_weights": weights, "universe_data": universe})

    assert len(client.submitted_orders) == 1
    assert ctx["execution_orders"][0]["side"] == "sell"
    assert ctx["execution_orders"][0]["qty"] == 300  # $30,000 / $100


def test_skips_order_below_min_notional(store):
    client = FakeTradingClient(equity=100_000.0, cash=100_000.0, positions={"AAPL": 49_990.0})
    universe = _universe_data({"AAPL": 100.0})
    weights = _weights(["AAPL"], [0.5])  # $10 delta, below default min_order_notional

    agent = ExecutionAgent(
        api_key="k", secret_key="s", state_store=store,
        trading_client_factory=lambda: client, min_order_notional=100.0,
    )
    agent.run({"portfolio_weights": weights, "universe_data": universe})

    assert len(client.submitted_orders) == 0


# ---------------------------------------------------------------------------
# Sell sizing must respect the broker's actual share count
# ---------------------------------------------------------------------------

def test_full_exit_sells_exactly_held_shares_not_dollar_estimate(store):
    """Observed live (2026-07-30/31, 4 failed runs): Kelly's go-flat sell
    was sized as market_value / yesterday's close. TQQQ had bounced ~9%
    intraday, so the dollar math requested 301 shares against 276 held --
    Alpaca rejected it and the strategy stayed fully invested for two
    sessions while its model said exit. A zero-weight target must sell the
    broker's actual share count, not a stale-price dollar estimate.
    """
    # 276 shares now worth $100 each; yesterday's close was $91.75
    # -> floor(27600 / 91.75) = 300 shares: an over-request.
    client = FakeTradingClient(equity=27_600.0, cash=0.0,
                               positions={"TQQQ": (27_600.0, 276)})
    universe = _universe_data({"TQQQ": 91.75})
    weights = _weights(["TQQQ"], [0.0])

    agent = ExecutionAgent(
        api_key="k", secret_key="s", state_store=store,
        trading_client_factory=lambda: client,
    )
    ctx = agent.run({"portfolio_weights": weights, "universe_data": universe})

    assert len(client.submitted_orders) == 1
    assert client.submitted_orders[0].qty == 276


def test_partial_sell_capped_at_available_shares(store):
    # 100 shares now worth $120 each ($12,000); stale close $100. Target
    # weight leaves an $11,000 sell -> naive qty 110 > 100 held.
    client = FakeTradingClient(equity=12_000.0, cash=0.0,
                               positions={"AAPL": (12_000.0, 100)})
    universe = _universe_data({"AAPL": 100.0})
    weights = _weights(["AAPL"], [1_000.0 / 12_000.0])

    agent = ExecutionAgent(
        api_key="k", secret_key="s", state_store=store,
        trading_client_factory=lambda: client,
    )
    agent.run({"portfolio_weights": weights, "universe_data": universe})

    assert len(client.submitted_orders) == 1
    assert client.submitted_orders[0].qty == 100


# ---------------------------------------------------------------------------
# Same-day round-trip (PDT) guard
# ---------------------------------------------------------------------------

def test_blocks_same_day_sell_after_buy_recorded(store):
    """If AAPL was already bought today (per the ledger), a same-day sell must be skipped."""
    store.record_order("2024-01-03", "AAPL", "buy", 100, "prior-order", "filled")

    client = FakeTradingClient(equity=100_000.0, cash=0.0, positions={"AAPL": 80_000.0})
    universe = _universe_data({"AAPL": 100.0})
    weights = _weights(["AAPL"], [0.1])  # target far below current -> would be a sell

    agent = ExecutionAgent(
        api_key="k", secret_key="s", state_store=store,
        trading_client_factory=lambda: client,
    )
    ctx = agent.run({"portfolio_weights": weights, "universe_data": universe})

    assert len(client.submitted_orders) == 0
    assert ctx["execution_orders"][0]["status"] == "skipped_pdt_guard"


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def test_records_orders_and_snapshot_in_state_store(store):
    client = FakeTradingClient(equity=100_000.0, cash=100_000.0, positions={})
    universe = _universe_data({"AAPL": 100.0})
    weights = _weights(["AAPL"], [0.5])

    agent = ExecutionAgent(
        api_key="k", secret_key="s", state_store=store,
        trading_client_factory=lambda: client,
    )
    agent.run({"portfolio_weights": weights, "universe_data": universe})

    orders_df = store.orders_for_date("2024-01-03")
    assert len(orders_df) == 1
    assert orders_df.iloc[0]["ticker"] == "AAPL"

    snap = store.latest_snapshot()
    assert snap["run_date"] == "2024-01-03"
    assert snap["equity"] == 100_000.0


def test_context_keys_written(store):
    client = FakeTradingClient(equity=100_000.0, cash=100_000.0, positions={})
    universe = _universe_data({"AAPL": 100.0})
    weights = _weights(["AAPL"], [0.0])  # no target -> no orders, but keys still present

    agent = ExecutionAgent(
        api_key="k", secret_key="s", state_store=store,
        trading_client_factory=lambda: client,
    )
    ctx = agent.run({"portfolio_weights": weights, "universe_data": universe})
    assert "execution_orders" in ctx
    assert "execution_account" in ctx
    assert ctx["execution_account"]["equity"] == 100_000.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_raises_if_portfolio_weights_missing(store):
    agent = ExecutionAgent(api_key="k", secret_key="s", state_store=store)
    with pytest.raises(AssertionError, match="portfolio_weights"):
        agent.run({"universe_data": {}})


def test_raises_if_universe_data_missing(store):
    agent = ExecutionAgent(api_key="k", secret_key="s", state_store=store)
    with pytest.raises(AssertionError, match="universe_data"):
        agent.run({"portfolio_weights": pd.DataFrame([[0.1]], columns=["AAPL"])})


# ---------------------------------------------------------------------------
# Order sequencing: sells must precede buys
# ---------------------------------------------------------------------------

def test_sells_submitted_before_buys_on_rotation(store):
    """Rotating between instruments (e.g. Kelly moving TQQQ -> QQQ) must
    submit the sell first: the sale's proceeds fund the buy. Buys-first
    only works while margin buying power happens to cover the overlap --
    e.g. fully invested at 1x, rotating 100% of equity, Reg-T buying power
    (2*equity - long value = equity) only *exactly* covers the buy, and any
    equity dip below the position's value during the day makes the buy
    reject outright.
    """
    client = FakeTradingClient(equity=100_000.0, cash=0.0, positions={"TQQQ": 100_000.0})
    universe = _universe_data({"QQQ": 500.0, "QLD": 100.0, "TQQQ": 60.0})
    # Dict/column order deliberately puts the buy ticker FIRST -- the agent
    # must reorder by side, not rely on input order.
    weights = _weights(["QQQ", "QLD", "TQQQ"], [1.0, 0.0, 0.0])

    agent = ExecutionAgent(
        api_key="k", secret_key="s", state_store=store,
        trading_client_factory=lambda: client,
    )
    agent.run({"portfolio_weights": weights, "universe_data": universe})

    sides = [
        (o.side.value if hasattr(o.side, "value") else str(o.side)).lower()
        for o in client.submitted_orders
    ]
    assert sides == ["sell", "buy"], (
        f"expected the TQQQ sell to be submitted before the QQQ buy, got {sides}"
    )
