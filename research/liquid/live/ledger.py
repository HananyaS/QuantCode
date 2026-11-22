"""Append-only SQLite ledger for the C3 runner (WAL, like utils/state_store.py).

Two tables:
- events: every decision, order, fill, check and error, as JSON;
- trades: one row per round trip, which the evening report reads.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, day TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL, entry_check INTEGER NOT NULL, qty INTEGER NOT NULL,
    decision_price REAL NOT NULL, limit_price REAL NOT NULL, entry_price REAL NOT NULL,
    entry_ts TEXT NOT NULL, stop_at_entry REAL NOT NULL,
    exit_price REAL, exit_ts TEXT, exit_kind TEXT, exit_check INTEGER, stop_at_exit REAL,
    weight REAL NOT NULL, equity REAL NOT NULL);
"""


class Ledger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def log(self, ts: datetime, day: str, event: str, **payload) -> None:
        assert "kind" not in payload, "payload key 'kind' would shadow the event kind"
        self.db.execute("INSERT INTO events (ts, day, kind, payload) VALUES (?, ?, ?, ?)",
                        (ts.isoformat(), day, event, json.dumps(payload, default=str)))

    def open_trade(self, **row) -> int:
        cols = ", ".join(row)
        cur = self.db.execute(f"INSERT INTO trades ({cols}) VALUES ({', '.join('?' * len(row))})",
                              tuple(row.values()))
        return int(cur.lastrowid)

    def close_trade(self, trade_id: int, **row) -> None:
        sets = ", ".join(f"{k} = ?" for k in row)
        self.db.execute(f"UPDATE trades SET {sets} WHERE id = ?", (*row.values(), trade_id))

    def trades(self, day: str | None = None) -> list[dict]:
        q = "SELECT * FROM trades" + (" WHERE day = ?" if day else "") + " ORDER BY id"
        cur = self.db.execute(q, (day,) if day else ())
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def events(self, day: str | None = None, kind: str | None = None) -> list[dict]:
        q, args = "SELECT ts, day, kind, payload FROM events WHERE 1=1", []
        if day:
            q += " AND day = ?"; args.append(day)
        if kind:
            q += " AND kind = ?"; args.append(kind)
        return [{"ts": t, "day": d, "kind": k, **json.loads(p)}
                for t, d, k, p in self.db.execute(q + " ORDER BY id", args)]
