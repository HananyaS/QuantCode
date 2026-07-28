"""state_store.py — persistent live-trading position/order ledger.

Why this exists
----------------
PortfolioAgent's position state is an in-memory backtest simulation that
resets every run. Live/paper execution needs positions and submitted orders
to persist *across* daily runs and be reconcilable against the actual
broker account. This is a small SQLite-backed ledger for exactly that —
not a general-purpose database layer.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class StateStore:
    """SQLite-backed ledger of submitted orders and daily account snapshots."""

    def __init__(self, db_path: str = "data/live_state.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    order_id TEXT,
                    status TEXT,
                    submitted_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS account_snapshots (
                    run_date TEXT PRIMARY KEY,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                )"""
            )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def record_order(
        self,
        run_date: str,
        ticker: str,
        side: str,
        qty: float,
        order_id: Optional[str],
        status: Optional[str],
    ) -> None:
        assert side in ("buy", "sell"), f"side must be 'buy' or 'sell', got {side!r}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO orders (run_date, ticker, side, qty, order_id, status, submitted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_date, ticker, side, qty, order_id, status,
                 datetime.now(timezone.utc).isoformat()),
            )
        logger.info("StateStore: recorded %s %s x%s (%s)", side, ticker, qty, run_date)

    def orders_for_date(self, run_date: str) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM orders WHERE run_date = ? ORDER BY id", conn, params=(run_date,)
            )
        return df

    def has_orders_on(self, run_date: str) -> bool:
        """True iff any order (buy OR sell) was already submitted for this
        session. The once-per-session execution guard keys on this: a
        re-run for a session that already executed (backup cron, manual
        re-trigger, GitHub retry) must no-op rather than re-reconcile
        against intraday-drifted equity and chase the price with top-ups.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM orders WHERE run_date = ? LIMIT 1", (run_date,)
            ).fetchone()
        return row is not None

    def tickers_bought_on(self, run_date: str) -> Set[str]:
        df = self.orders_for_date(run_date)
        if len(df) == 0:
            return set()
        return set(df.loc[df["side"] == "buy", "ticker"])

    def all_orders(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM orders ORDER BY id", conn)

    # ------------------------------------------------------------------
    # Account snapshots
    # ------------------------------------------------------------------

    def record_account_snapshot(
        self,
        run_date: str,
        equity: float,
        cash: float,
        positions: Dict[str, float],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO account_snapshots (run_date, equity, cash, positions_json, recorded_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_date) DO UPDATE SET "
                "equity=excluded.equity, cash=excluded.cash, "
                "positions_json=excluded.positions_json, recorded_at=excluded.recorded_at",
                (run_date, equity, cash, json.dumps(positions),
                 datetime.now(timezone.utc).isoformat()),
            )
        logger.info("StateStore: recorded account snapshot for %s (equity=%.2f)", run_date, equity)

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_date, equity, cash, positions_json, recorded_at "
                "FROM account_snapshots ORDER BY run_date DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        run_date, equity, cash, positions_json, recorded_at = row
        return {
            "run_date": run_date,
            "equity": equity,
            "cash": cash,
            "positions": json.loads(positions_json),
            "recorded_at": recorded_at,
        }

    def snapshots(self) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM account_snapshots ORDER BY run_date ASC", conn
            )
        if len(df) > 0:
            df["positions"] = df["positions_json"].apply(json.loads)
        return df
