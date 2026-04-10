"""CrossSectionalLabelingAgent: forward N-day return labels per (date, ticker).

Leakage guarantee
-----------------
label[date, ticker] = (Close[date + N] - Close[date]) / Close[date]

This is computed via shift(-N) on each asset's Close series.
The last N dates per asset have no label and are excluded — they must
also be excluded from the feature matrix (handled by the inner join in
RankingModelAgent).  No future price enters any feature computation.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class CrossSectionalLabelingAgent(BaseAgent):
    """Computes forward-return regression labels for every (date, ticker) pair.

    Writes
    ------
    context['cs_labels'] : pd.Series with MultiIndex(date, ticker), name='label'
                           Values are raw forward returns (float, not binary).
    """

    def __init__(self, forward_period: int = 5) -> None:
        assert forward_period >= 1, "forward_period must be >= 1"
        self.forward_period = forward_period

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, context: dict) -> dict:
        assert context.get("universe_data") is not None, (
            "CrossSectionalLabelingAgent requires context['universe_data']"
        )
        universe_data: Dict[str, pd.DataFrame] = context["universe_data"]
        N = self.forward_period

        series_list: List[pd.Series] = []
        for ticker, df in universe_data.items():
            close = df["Close"]
            # Forward return: shift close back by N days relative to current row
            fwd_close = close.shift(-N)
            fwd_return = fwd_close / close - 1
            # Drop last N rows — no future close exists
            fwd_return = fwd_return.iloc[:-N]
            series_list.append(fwd_return.rename(ticker))

        # Wide: date × ticker
        wide = pd.concat(series_list, axis=1)
        # Long: (date, ticker) MultiIndex
        labels = wide.stack()
        labels.index.names = ["date", "ticker"]
        labels.name = "label"
        labels = labels.dropna()

        self._validate(labels, N, universe_data)

        n_dates = labels.index.get_level_values("date").nunique()
        n_assets = labels.index.get_level_values("ticker").nunique()
        logger.info(
            "CSLabelingAgent: %d-day forward returns | %d dates x %d assets",
            N,
            n_dates,
            n_assets,
        )
        context["cs_labels"] = labels
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        labels: pd.Series,
        forward_period: int,
        universe_data: Dict[str, pd.DataFrame],
    ) -> None:
        assert labels.notna().all(), "Labels contain NaN after dropna"
        assert isinstance(labels.index, pd.MultiIndex), "Expected MultiIndex (date, ticker)"

        # Last N dates must be absent for every ticker
        for ticker, df in universe_data.items():
            last_n_dates = df.index[-forward_period:]
            ticker_dates = labels.xs(ticker, level="ticker").index
            overlap = ticker_dates.intersection(last_n_dates)
            assert len(overlap) == 0, (
                f"Leakage: ticker {ticker} has labels on last {forward_period} dates "
                f"where no forward return should exist"
            )
