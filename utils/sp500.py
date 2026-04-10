"""sp500.py — fetch current S&P 500 constituent list from Wikipedia.

SURVIVORSHIP BIAS WARNING
--------------------------
This returns the *current* S&P 500 members only.  Companies that were
removed from the index (due to acquisition, bankruptcy, or demotion) during
the study period are excluded.  This introduces an upward performance bias
that cannot be eliminated without a point-in-time index membership database
(e.g. CRSP, Bloomberg, Refinitiv).
"""
from __future__ import annotations

import io
from typing import List

import pandas as pd
import requests

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_USER_AGENT = "Mozilla/5.0 (compatible; QuantCode research bot)"


def get_sp500_tickers() -> List[str]:
    """Return current S&P 500 ticker symbols.

    Symbols are returned in their Wikipedia format (e.g. BRK.B, BF.B),
    which is also the format accepted by Alpaca and yfinance.

    Returns:
        List of ticker strings.
    """
    resp = requests.get(_WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    return tables[0]["Symbol"].tolist()
