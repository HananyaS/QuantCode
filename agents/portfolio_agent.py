"""PortfolioAgent: stateful day-by-day portfolio simulator.

Execution order on each test day t
-----------------------------------
1. **Trailing stop check** — update HWM, ratchet ATR-%-based stop up,
   check if the day's *Low* breached the stop; exit at stop price.
2. **Rank-based exit** — exit held positions below *exit_rank*.
3. **New entries** — from top *entry_rank* candidates whose *z-score > min_score*,
   up to *max_positions* total.
4. **Weight** — equal 1/n or softmax score-weighted across held positions.

ATR-% trailing stop
-------------------
Stop distance is a *fraction of price* (not absolute $) so it scales
consistently across different price levels:

  atr_pct[t]      = ATR[t] / Close[t]
  candidate_stop  = HWM * (1 - atr_mult * atr_pct[t])
  stop_level      = max(stop_level, candidate_stop)   # ratchet: never decreases

The stop is checked against the day's *Low*, not just the close, to better
mimic reality: if Low[t] <= stop_level, the position is exited at stop_level
(assuming a stop-limit order that fills at the stop, not at a worse price).

Score normalisation
-------------------
Raw model scores are z-scored *within each day* before use, making
*min_score* a z-score threshold:
  0.0  → only enter above-average predictions
  1.0  → only enter predictions > 1 std above mean

Score-weighted allocation
-------------------------
When *score_weighting=True*, position sizes are proportional to
  softmax(z_score / temperature)
Higher temperature → more uniform; lower temperature → concentrates weight
on highest-scoring positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
    """Stateful portfolio simulator with ATR-% trailing stops, dual-rank
    thresholds, daily z-score normalisation, and optional score weighting.

    Reads
    -----
    context['cs_predictions']  : pd.Series (date, ticker) — raw model scores
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
        score_weighting: bool = False,
        weighting_temperature: float = 1.0,
    ) -> None:
        assert max_positions >= 1
        assert entry_rank >= 1
        assert exit_rank >= entry_rank
        assert weighting_temperature > 0, "weighting_temperature must be > 0"
        self.max_positions = max_positions
        self.entry_rank = entry_rank
        self.exit_rank = exit_rank
        self.min_score = min_score
        self.atr_mult = trailing_stop_atr_mult
        self.atr_period = atr_period
        self.score_weighting = score_weighting
        self.weighting_temperature = weighting_temperature

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        assert context.get("cs_predictions") is not None
        assert context.get("universe_data") is not None

        scores_raw: pd.Series = context["cs_predictions"]
        universe_data: Dict[str, pd.DataFrame] = context["universe_data"]

        # Pre-compute price matrices: date × ticker
        close_df = pd.DataFrame({t: df["Close"] for t, df in universe_data.items()})
        low_df   = pd.DataFrame({t: df["Low"]   for t, df in universe_data.items()})
        atr_df   = self._compute_atr(universe_data)

        # Widen scores: date × ticker
        scores_wide = scores_raw.unstack(level="ticker")
        test_dates  = scores_wide.index.sort_values()
        all_tickers = list(close_df.columns)

        # Day-by-day simulation
        open_pos: Dict[str, Position] = {}
        weights_rows: List[Dict[str, float]] = []
        trade_log: List[dict] = []

        for t in test_dates:
            today_scores_raw = scores_wide.loc[t].dropna()
            today_scores_z   = self._zscore(today_scores_raw)

            today_close = close_df.loc[t] if t in close_df.index else pd.Series(dtype=float)
            today_low   = low_df.loc[t]   if t in low_df.index   else pd.Series(dtype=float)
            today_atr   = atr_df.loc[t]   if t in atr_df.index   else pd.Series(dtype=float)

            # ── Step 1: trailing stop exits ──────────────────────────────
            for ticker in list(open_pos):
                c = today_close.get(ticker, np.nan)
                if pd.isna(c):
                    continue
                pos = open_pos[ticker]
                atr_val = today_atr.get(ticker, np.nan)

                # Update HWM
                pos.hwm = max(pos.hwm, c)

                # Ratchet stop up — ATR expressed as % of current price
                if not np.isnan(atr_val) and atr_val > 0 and c > 0:
                    atr_pct = atr_val / c
                    candidate_stop = pos.hwm * (1.0 - self.atr_mult * atr_pct)
                    pos.stop_level = max(pos.stop_level, candidate_stop)

                # Check trigger using intraday Low (more realistic than Close only)
                check_price = today_low.get(ticker, np.nan)
                if pd.isna(check_price):
                    check_price = c  # fall back to close if Low not available

                if check_price <= pos.stop_level:
                    # Fill at stop_level (stop-limit assumption)
                    exit_price = pos.stop_level
                    trade_log.append(self._log_exit(
                        t, ticker, exit_price, pos, "trailing_stop",
                        today_scores_z.get(ticker, np.nan),
                    ))
                    del open_pos[ticker]

            # ── Step 2: rank-based exits ─────────────────────────────────
            ranked = today_scores_z.rank(ascending=False)
            for ticker in list(open_pos):
                rank = ranked.get(ticker, float("inf"))
                if rank > self.exit_rank:
                    c = today_close.get(ticker, np.nan)
                    trade_log.append(self._log_exit(
                        t, ticker, c, open_pos[ticker], "rank_exit",
                        today_scores_z.get(ticker, np.nan),
                    ))
                    del open_pos[ticker]

            # ── Step 3: new entries ──────────────────────────────────────
            candidates = (
                today_scores_z
                .nlargest(self.entry_rank)
                .loc[lambda s: s >= self.min_score]
            )
            available = self.max_positions - len(open_pos)
            for ticker in candidates.index:
                if available <= 0:
                    break
                if ticker in open_pos:
                    continue
                c       = today_close.get(ticker, np.nan)
                atr_val = today_atr.get(ticker, np.nan)
                if pd.isna(c) or c <= 0:
                    continue

                # Initial stop: ATR-% below entry price
                if not np.isnan(atr_val) and atr_val > 0 and c > 0:
                    atr_pct = atr_val / c
                    tol = self.atr_mult * atr_pct
                else:
                    tol = 0.05  # fallback: 5% fixed stop when ATR unavailable

                pos = Position(
                    ticker=ticker,
                    entry_price=c,
                    entry_date=t,
                    hwm=c,
                    stop_level=c * (1.0 - tol),
                )
                open_pos[ticker] = pos
                available -= 1
                trade_log.append({
                    "date": t, "ticker": ticker, "action": "entry",
                    "price": c,
                    "stop_level": pos.stop_level,
                    "score_z": float(today_scores_z.get(ticker, np.nan)),
                })

            # ── Step 4: weights ──────────────────────────────────────────
            weights_rows.append(
                self._compute_weights(open_pos, all_tickers, today_scores_z)
            )

        weights = pd.DataFrame(weights_rows, index=test_dates).fillna(0.0)

        # Summary stats
        n_entries = sum(1 for t in trade_log if t["action"] == "entry")
        n_trail   = sum(1 for t in trade_log if t["action"] == "trailing_stop")
        n_rank    = sum(1 for t in trade_log if t["action"] == "rank_exit")
        avg_held  = (weights > 0).sum(axis=1).mean()
        zero_days = (weights.sum(axis=1) == 0).sum()

        logger.info(
            "PortfolioAgent: entries=%d  trail_stops=%d  rank_exits=%d  "
            "avg_held=%.1f  zero_position_days=%d  score_weighted=%s",
            n_entries, n_trail, n_rank, avg_held, zero_days, self.score_weighting,
        )

        context["portfolio_weights"] = weights
        context["portfolio_trades"]  = trade_log
        return context

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        open_pos: Dict[str, Position],
        all_tickers: list,
        today_scores_z: pd.Series,
    ) -> Dict[str, float]:
        row = {tk: 0.0 for tk in all_tickers}
        n = len(open_pos)
        if n == 0:
            return row

        if not self.score_weighting:
            w = 1.0 / n
            for tk in open_pos:
                row[tk] = w
        else:
            tickers = list(open_pos.keys())
            scores  = np.array([today_scores_z.get(tk, 0.0) for tk in tickers])
            logits  = scores / self.weighting_temperature
            logits -= logits.max()           # numerical stability
            w_arr   = np.exp(logits)
            w_arr  /= w_arr.sum()
            for tk, wi in zip(tickers, w_arr):
                row[tk] = float(wi)

        return row

    # ------------------------------------------------------------------
    # ATR computation
    # ------------------------------------------------------------------

    def _compute_atr(self, universe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        atr_dict = {}
        for ticker, df in universe_data.items():
            high       = df["High"]
            low        = df["Low"]
            close      = df["Close"]
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
    # Score normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        """Z-score normalise a series; returns original if std == 0."""
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s - s.mean()

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
        score_z: float,
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
            "score_z": score_z,
        }
