"""live_decision.py — bridges a timing agent's single-row (ticker, fraction)
decision to the full ticker->weight mapping ExecutionAgent needs.
"""
from __future__ import annotations

from typing import Dict, Tuple

_DEFAULT_TRACKED = ("QQQ", "QLD", "TQQQ")


def position_row_to_weights(
    ticker: str,
    fraction: float,
    tracked_tickers: Tuple[str, ...] = _DEFAULT_TRACKED,
) -> Dict[str, float]:
    """Expand a single (ticker, fraction) decision into a weight for every
    tracked ticker, explicitly zeroing the rest.

    Explicit zeroing matters: ExecutionAgent only sells out of a ticker if
    it appears in the weights row with a lower target — a ticker silently
    missing from the row would never be unwound if yesterday's position no
    longer matches today's decision.
    """
    assert ticker in tracked_tickers, f"{ticker!r} not in tracked_tickers {tracked_tickers}"
    return {t: (float(fraction) if t == ticker else 0.0) for t in tracked_tickers}
