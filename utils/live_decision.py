"""live_decision.py — bridges a timing agent's single-row (ticker, fraction)
decision to the full ticker->weight mapping ExecutionAgent needs, and
extends a live universe with a next-session placeholder bar for
forecast-for-t agents.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

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


def append_next_session_bar(
    universe_data: Dict[str, pd.DataFrame],
    next_session,
    tickers: Tuple[str, ...] = _DEFAULT_TRACKED,
) -> Dict[str, pd.DataFrame]:
    """Return a copy of `universe_data` with an all-NaN OHLCV row appended
    at `next_session` for each ticker in `tickers` (others pass through
    untouched). The input frames are never mutated.

    Why: KellyPositionAgent's estimators follow the forecast-for-t
    convention (a value at date t uses only data through t-1, pre-shifted
    at the source), so its position row at the last COMPLETED bar t was
    decided from data through t-1. Executing that row during session t+1
    would discard bar t's close entirely — a systematic one-trading-day
    staleness relative to the validated backtest, which assumes the
    position dated t is held DURING t. Appending a placeholder row at the
    next session makes the agent emit a genuine decision FOR that session,
    computed from everything through the latest close: the pre-shifted
    EWM estimators carry forward through the NaN input, so shift(1) lands
    the full-history estimate exactly on the appended row, and the
    (already-lagged) drawdown/watermark price lookups resolve to the last
    real close.

    Scope limits, deliberate:
    - EWMA vol only — gjr_garch_variance drops NaN rows before fitting, so
      the appended row would get a NaN forecast and fail the agent's
      validity gate (silently flat). Callers must enforce vol_method="ewma".
    - Do NOT use for TimingSignalAgent/LeveragedPositionAgent (the linear
      strategy): those use unshifted same-day rolling windows, where a NaN
      row yields NaN -> signal 0 — a silent false de-risk. Their last-bar
      row already IS the next session's position by that pipeline's
      shift-downstream convention.
    """
    next_ts = pd.Timestamp(next_session)
    extended: Dict[str, pd.DataFrame] = {}
    for ticker, df in universe_data.items():
        if ticker not in tickers:
            extended[ticker] = df
            continue
        assert next_ts > df.index[-1], (
            f"next_session {next_ts.date()} must be strictly after {ticker}'s "
            f"last bar {df.index[-1].date()}"
        )
        placeholder = pd.DataFrame(
            {col: [np.nan] for col in df.columns}, index=[next_ts],
        )
        extended[ticker] = pd.concat([df, placeholder])
    return extended
