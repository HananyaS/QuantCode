"""validate_data_sources.py — cross-check overlapping price data across providers.

Why this exists
-----------------
Blending multiple data providers (Alpaca, Tiingo, yfinance) into one
research corpus risks silently corrupting features/labels if their
split/dividend adjustment methodologies disagree — the kind of bug that
doesn't crash anything, it just quietly produces wrong numbers. This script
fetches the same tickers/date range from each configured provider and flags
any pair whose adjusted Close diverges materially on overlapping dates.

Usage
------
    python scripts/validate_data_sources.py --tickers AAPL MSFT SPY \\
        --start 2023-01-01 --end 2023-12-31

Requires the relevant API keys as environment variables (ALPACA_API_KEY /
ALPACA_SECRET_KEY, TIINGO_API_KEY) — a source without its key set is
skipped with a warning, not an error.
"""
from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)

_DIVERGENCE_WARN_THRESHOLD = 0.01  # 1% relative difference in adjusted Close


def _fetch_alpaca(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    from utils.alpaca_loader import fetch_universe_bars

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        logger.warning("validate_data_sources: ALPACA_API_KEY/SECRET_KEY not set, skipping Alpaca")
        return {}
    return fetch_universe_bars(tickers, start, end, api_key=api_key, secret_key=secret_key)


def _fetch_tiingo(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    from utils.tiingo_loader import fetch_bars

    api_key = os.environ.get("TIINGO_API_KEY")
    if not api_key:
        logger.warning("validate_data_sources: TIINGO_API_KEY not set, skipping Tiingo")
        return {}
    return fetch_bars(tickers, start, end, api_key=api_key)


def _fetch_yfinance(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    import yfinance as yf

    results: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            results[ticker] = df[["Open", "High", "Low", "Close", "Volume"]]
    return results


_DEFAULT_SOURCES: Dict[str, Callable[[List[str], str, str], Dict[str, pd.DataFrame]]] = {
    "alpaca": _fetch_alpaca,
    "tiingo": _fetch_tiingo,
    "yfinance": _fetch_yfinance,
}


def compare_sources(
    tickers: List[str],
    start: str,
    end: str,
    sources: Optional[Dict[str, Callable[[List[str], str, str], Dict[str, pd.DataFrame]]]] = None,
) -> pd.DataFrame:
    """Fetch `tickers` over [start, end] from each source and diff adjusted Close.

    Args:
        tickers: Ticker symbols to check.
        start, end: "YYYY-MM-DD" date strings (inclusive).
        sources: Dict of name -> fetch_fn(tickers, start, end) -> Dict[ticker, DataFrame].
            Defaults to Alpaca/Tiingo/yfinance. Injectable for testing.

    Returns:
        DataFrame with one row per (ticker, source pair) that has
        overlapping dates: ticker, source_a, source_b, n_overlap_days,
        max_rel_diff, mean_rel_diff, flagged (bool, exceeds threshold).
        Pairs with no data or no overlapping dates are simply absent —
        this is exploratory tooling, not something that should raise.
    """
    sources = sources or _DEFAULT_SOURCES
    fetched = {name: fn(tickers, start, end) for name, fn in sources.items()}

    names = list(sources.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            for ticker in tickers:
                a = fetched[a_name].get(ticker)
                b = fetched[b_name].get(ticker)
                if a is None or b is None or a.empty or b.empty:
                    continue

                common = a.index.intersection(b.index)
                if len(common) == 0:
                    continue

                a_close = a.loc[common, "Close"]
                b_close = b.loc[common, "Close"]
                rel_diff = ((a_close - b_close).abs() / a_close.abs().replace(0, np.nan)).dropna()
                if rel_diff.empty:
                    continue

                rows.append({
                    "ticker": ticker,
                    "source_a": a_name,
                    "source_b": b_name,
                    "n_overlap_days": int(len(common)),
                    "max_rel_diff": float(rel_diff.max()),
                    "mean_rel_diff": float(rel_diff.mean()),
                    "flagged": bool(rel_diff.max() > _DIVERGENCE_WARN_THRESHOLD),
                })

    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Cross-check price data across providers")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "SPY"])
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    args = parser.parse_args()

    report = compare_sources(args.tickers, args.start, args.end)
    if report.empty:
        print("No overlapping data found across any source pair — check API keys / date range.")
        return

    print(report.to_string(index=False))
    flagged = report[report["flagged"]]
    if len(flagged):
        print(
            f"\n{len(flagged)} ticker/source-pair(s) exceed "
            f"{_DIVERGENCE_WARN_THRESHOLD:.1%} divergence — investigate "
            "before trusting a blended dataset."
        )


if __name__ == "__main__":
    main()
