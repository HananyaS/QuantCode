"""Safety primitives for the C3 runner (PAPER_C3 §1, §4 G-safety; G3 B3, B4, W6).

- flatten: cancel every open order, wait until none is open, close the
  position, wait until flat. A second exit is never sent while an earlier one
  still holds the shares (fact sheet item 5).
- single_instance: an OS file lock, so two runners cannot trade one account.
- heartbeat / guard_action: the runner writes a heartbeat; an independent guard
  process flattens when the runner has died with a position open.
- write_kill: every automatic kill leaves the KILL file, so the next session
  does not start until the cause is logged.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator

from research.liquid.live.broker import OrderState

FLATTEN_TIMEOUT_S = 60.0
STALE_S = 90.0                 # heartbeat older than this = the runner is dead or hung
AFTER_CLOSEOUT_S = 45.0        # a position still open this long after the close-out starts


class FlattenError(RuntimeError):
    pass


def flatten(b, symbol: str, now: Callable[[], float], sleep: Callable[[float], None],
            timeout_s: float = FLATTEN_TIMEOUT_S) -> list[OrderState]:
    """Cancel all, wait for no open orders, close, wait for flat. Returns the closing orders."""
    end = now() + timeout_s
    while True:
        oo = b.open_orders(symbol)
        if not oo:
            break
        for o in oo:
            if o.status != "pending_cancel":
                try:
                    b.cancel(o.id)
                except Exception:                                     # noqa: BLE001  filled or cancelling
                    pass
        if now() >= end:
            raise FlattenError(f"{len(oo)} orders still open after {timeout_s}s")
        sleep(0.5)
    closes: list[OrderState] = []
    filled_at: float | None = None
    while True:
        q = b.position_qty(symbol)
        if q == 0:
            return [b.get(o.id) for o in closes]
        last = b.get(closes[-1].id) if closes else None
        if last is not None and last.status == "filled" and filled_at is None:
            filled_at = now()                       # let the position catch up before any re-send
        resend = last is None or (not last.open and last.status != "filled") or \
            (last.status == "filled" and filled_at is not None and now() - filled_at > 5.0)
        if resend:
            o = b.close_position(symbol)
            if o is not None:
                closes.append(o)
                filled_at = None
        if now() >= end:
            raise FlattenError(f"position {q} still open after {timeout_s}s")
        sleep(0.5)


@contextmanager
def single_instance(path: Path) -> Iterator[None]:
    """Hold an exclusive OS lock on `path` for the duration; the OS releases it if the process dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise SystemExit(f"another process holds {path}: refusing to start a second instance") from e
        yield
    finally:
        fh.close()


def write_heartbeat(path: Path | None, ts: datetime, state: str) -> None:
    if path is None:
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"ts": ts.isoformat(), "state": state, "pid": os.getpid()}), encoding="utf-8")
    os.replace(tmp, path)


def read_heartbeat(path: Path) -> tuple[datetime | None, str | None]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return datetime.fromisoformat(d["ts"]), d.get("state")
    except (OSError, ValueError, KeyError):
        return None, None


def guard_action(now: datetime, beat: datetime | None, position: int, n_open: int,
                 session_open: datetime, closeout_at: datetime) -> str | None:
    """Why the guard should flatten now, or None. Acts only from the open."""
    if now < session_open or (position == 0 and n_open == 0):
        return None
    if now >= closeout_at + timedelta(seconds=AFTER_CLOSEOUT_S):
        return "position or orders still open after the close-out"
    if beat is None or (now - beat).total_seconds() > STALE_S:
        return f"runner heartbeat stale ({'none' if beat is None else f'{(now - beat).total_seconds():.0f}s'})"
    return None


def write_kill(path: Path | None, reason: str, ts: datetime) -> None:
    """Leave the KILL file; keep an existing one (the user's own or an earlier kill)."""
    if path is None or path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{ts.isoformat()}  {reason}\nDelete this file once the cause is logged in LOG.md.\n",
                    encoding="utf-8")
