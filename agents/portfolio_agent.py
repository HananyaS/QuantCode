"""PortfolioAgent: stateful day-by-day portfolio simulator.

Execution order on each test day t
-----------------------------------
1. **Trailing stop check** — update HWM, ratchet stop up, exit if close <= stop.
2. **Rank-based exit** — exit held positions that dropped below *exit_rank* (35).
3. **New entries** — from top *entry_rank* (20) candidates whose score > *min_score*,
   up to *max_positions* total.  Already-held stocks are NOT re-entered.
4. **Weight** — equal weight 1/n across all held positions (0 if empty).

This replaces the old stateless top-K selector with a mechanism that:
- Uses a calibrated minimum score to stay flat on bearish days
- Has wider "hold" threshold (exit_rank) vs "entry" threshold (entry_rank)
  to reduce over-trading
- Runs an ATR-based trailing stop per position (see prompts/skills/trailing_stop.txt)

Leakage guarantee
-----------------
All decisions on day t use only:
  - Close[t] (observed at close)
  - Model scores computed from features up to t
  - ATR computed from past OHLC up to t
No future data is accessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Mutable state tracked per open position."""
    ticker: str
    entry_price: float
    entry_date: pd.Timestamp
    hwm: float
    stop_level: float


class PortfolioAgent(BaseAgent):
    """Stateful portfolio simulator with trailing stops and dual-rank thresholds.

    Reads
    -----
    context['cs_predictions']  : pd.Series (date, ticker) — model scores
    context['universe_data']   : Dict[str, pd.DataFrame]  — OHLCV per ticker

    Writes
    ------
    context['portfolio_weights'] : pd.DataFrame (date × ticker), weights 0..1
    context['portfolio_trades']  : List[dict] — trade-level log
    """

    def __init__(
        self,
        max_positions: int = 20,
        entry_rank: int = 20,
        exit_rank: int = 35,
        min_score: float = 0.0,
        trailing_stop_atr_mult: float = 1.5,
        atr_period: int = 14,
    ) -> None:
        assert max_positions >= 1
        assert entry_rank >= 1
        assert exit_rank >= entry_rank
        self.max_positions = max_positions
        self.entry_rank = entry_rank
        self.exit_rank = exit_rank
        self.min_score = min_score
        self.atr_mult = trailing_stop_atr_mult
        self.atr_period = atr_period

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        assert context.get("cs_predictions") is not None
        assert context.get("universe_data") is not None

        scores: pd.Series = context["cs_predictions"]
        universe_data: Dict[str, pd.DataFrame] = context["universe_data"]

        # Pre-compute matrices: date × ticker
        close_df = pd.DataFrame(
            {t: df["Close"] for t, df in universe_data.items()}
        )
        atr_df = self._compute_atr(universe_data)

        # Widen scores: date × ticker
        scores_wide = scores.unstack(level="ticker")
        test_dates = scores_wide.index.sort_values()
        all_tickers = list(close_df.columns)

        # Day-by-day simulation
        open_pos: Dict[str, Position] = {}
        weights_rows: List[Dict[str, float]] = []
        trade_log: List[dict] = []

        for t in test_dates:
            today_scores = scores_wide.loc[t].dropna()
            today_close = close_df.loc[t] if t in close_df.index else pd.Series(dtype=float)
            today_atr = atr_df.loc[t] if t in atr_df.index else pd.Series(dtype=float)

            # Step 1: trailing stop exits
            for ticker in list(open_pos):
                if ticker not in today_close or pd.isna(today_close[ticker]):
                    continue
                pos = open_pos[ticker]
                c = today_close[ticker]
                atr_val = today_atr.get(ticker, np.nan)

                # Update HWM
                pos.hwm = max(pos.hwm, c)

                if not np.isnan(atr_val) and atr_val > 0:
                    candidate_stop = pos.hwm - self.atr_mult * atr_val
                    pos.stop_level = max(pos.stop_level, candidate_stop)

                if c <= pos.stop_level:
                    trade_log.append(self._log_exit(
                        t, ticker, c, pos, "trailing_stop",
                        today_scores.get(ticker, np.nan),
                    ))
                    del open_pos[ticker]

            # Step 2: rank-based exits
            ranked = today_scores.rank(ascending=False)
            for ticker in list(open_pos):
                rank = ranked.get(ticker, float("inf"))
                if rank > self.exit_rank:
                    c = today_close.get(ticker, np.nan)
                    trade_log.append(self._log_exit(
                        t, ticker, c, open_pos[ticker], "rank_exit",
                        today_scores.get(ticker, np.nan),
                    ))
                    del open_pos[ticker]

            # Step 3: new entries
            candidates = (
                today_scores
                .nlargest(self.entry_rank)
                .loc[lambda s: s > self.min_score]
            )
            available = self.max_positions - len(open_pos)
            for ticker in candidates.index:
                if available <= 0:
                    break
                if ticker in open_pos:
                    continue
                c = today_close.get(ticker, np.nan)
                atr_val = today_atr.get(ticker, np.nan)
                if pd.isna(c) or c <= 0:
                    continue
                tol = self.atr_mult * atr_val if not np.isnan(atr_val) else c * 0.05
                pos = Position(
                    ticker=ticker,
                    entry_price=c,
                    entry_date=t,
                    hwm=c,
                    stop_level=c - tol,
                )
                open_pos[ticker] = pos
                available -= 1
                trade_log.append({
                    "date": t, "ticker": ticker, "action": "entry",
                    "price": c, "stop_level": pos.stop_level,
                    "score": today_scores.get(ticker, np.nan),
                })

            # Step 4: weights
            n_held = len(open_pos)
            row = {tk: 0.0 for tk in all_tickers}
            for tk in open_pos:
                row[tk] = 1.0 / n_held if n_held > 0 else 0.0
            weights_rows.append(row)

        weights = pd.DataFrame(weights_rows, index=test_dates)
        weights = weights.fillna(0.0)

        # Stats
        n_entries = sum(1 for t in trade_log if t["action"] == "entry")
        n_trail = sum(1 for t in trade_log if t["action"] == "trailing_stop")
        n_rank = sum(1 for t in trade_log if t["action"] == "rank_exit")
        avg_held = (weights > 0).sum(axis=1).mean()
        zero_days = (weights.sum(axis=1) == 0).sum()

        logger.info(
            "PortfolioAgent: entries=%d  trail_stops=%d  rank_exits=%d  "
            "avg_held=%.1f  zero_position_days=%d",
            n_entries, n_trail, n_rank, avg_held, zero_days,
        )

        context["portfolio_weights"] = weights
        context["portfolio_trades"] = trade_log
        return context

    # ------------------------------------------------------------------
    # ATR computation
    # ------------------------------------------------------------------

    def _compute_atr(self, universe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        atr_dict = {}
        for ticker, df in universe_data.items():
            high = df["High"]
            low = df["Low"]
            close = df["Close"]
            prev_close = close.shift(1)
            tr = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            atr_dict[ticker] = tr.rolling(
                window=self.atr_period, min_periods=self.atr_period
            ).mean()
        return pd.DataFrame(atr_dict)

    # ------------------------------------------------------------------
    # Trade log helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_exit(
        date: pd.Timestamp,
        ticker: str,
        price: float,
        pos: Position,
        reason: str,
        score: float,
    ) -> dict:
        return {
            "date": date,
            "ticker": ticker,
            "action": reason,
            "price": price,
            "entry_price": pos.entry_price,
            "entry_date": pos.entry_date,
            "pnl_pct": (price / pos.entry_price - 1) if pos.entry_price > 0 else 0.0,
            "stop_level": pos.stop_level,
            "hwm": pos.hwm,
            "score": score,
        }
