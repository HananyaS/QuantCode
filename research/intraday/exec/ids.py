"""Deterministic client order ids: derived from the intent id only, no entropy, no clock."""
from __future__ import annotations

from typing import Literal

Leg = Literal["entry", "sell", "exit"]
MAX_CLIENT_ORDER_ID_LEN: int = 128  # Alpaca's client_order_id limit


def client_order_id(intent_id: str, leg: Leg) -> str:
    """``intent_id`` for an entry or a sell; ``f"{intent_id}-exit"`` for an exit."""
    if not isinstance(intent_id, str) or not intent_id:
        raise ValueError(f"intent_id must be a non-empty str, got {intent_id!r}")
    if leg in ("entry", "sell"):
        out = intent_id
    elif leg == "exit":
        out = f"{intent_id}-exit"
    else:
        raise ValueError(f"leg must be 'entry', 'sell' or 'exit', got {leg!r}")
    if len(out) > MAX_CLIENT_ORDER_ID_LEN:
        raise ValueError(f"client_order_id longer than {MAX_CLIENT_ORDER_ID_LEN}: {len(out)}")
    return out
