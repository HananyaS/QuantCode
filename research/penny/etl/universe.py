"""universe.py — a survivorship-aware listed-equity universe.

The single most consequential decision in this study
----------------------------------------------------
Today's Nasdaq symbol directory lists only what is *currently* listed. Every
microcap that delisted during the study window is absent -- and those are
precisely the names that went to zero. Building the universe from the current
directory would silently condition the entire study on survival and inflate
every return estimate in it.

Alpaca's asset registry exposes ``status=inactive``, which recovers ~2,700
delisted names on major exchanges, and historical bars remain available for
them. That does not make the universe perfectly point-in-time -- the registry
carries no listing or delisting *dates*, so membership is reconstructed from
the bar record itself (a symbol is treated as listed on the days it printed
bars). That is an approximation, and it is stated wherever results are
reported rather than quietly assumed away.

Residual bias after this fix: names that both listed and delisted entirely
before Alpaca's coverage began, and any name the registry has dropped
altogether. Both push in the same direction -- toward optimism -- so measured
edges should be read as upper bounds.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Ordinary common-stock tickers only. The registry also carries CUSIPs
# ("590CVR015"), rights ("NHS.RT"), preferred lines ("MNR.PRC"), units
# ("GIGGU") and explicit markers ("BWMCU_DELISTED"), none of which are
# consolidated-tape common stock and all of which Alpaca refuses to price.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

# The study is exchange-listed only, by explicit scope decision. OTC is
# excluded: free data coverage there is poor and its microstructure differs
# enough that pooling the two would confound any result.
MAJOR_EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}


@dataclass
class UniverseSpec:
    """Screen applied to build the study universe.

    Price bounds are applied to **unadjusted** prices at decision time; see
    ``screen_daily_panel``. Applying them to split-adjusted prices is the
    single most common way a penny-stock backtest silently destroys itself.
    """

    min_price: float = 0.50
    max_price: float = 10.00
    min_dollar_volume: float = 100_000.0
    adv_window: int = 20


def fetch_asset_registry(api_key: str, secret_key: str) -> pd.DataFrame:
    """All US equities Alpaca knows about, active and inactive.

    Uses the trading client purely as a read-only registry lookup; no
    order-capable method is called.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    rows = []
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        for asset in client.get_all_assets(
            GetAssetsRequest(status=status, asset_class=AssetClass.US_EQUITY)
        ):
            rows.append(
                {
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "exchange": asset.exchange.value if asset.exchange else None,
                    "status": status.value,
                    "tradable": bool(asset.tradable),
                    "shortable": bool(getattr(asset, "shortable", False)),
                    "easy_to_borrow": bool(getattr(asset, "easy_to_borrow", False)),
                    "fractionable": bool(getattr(asset, "fractionable", False)),
                }
            )

    df = pd.DataFrame(rows)
    logger.info(
        "asset registry: %d total (%d active, %d inactive)",
        len(df), (df["status"] == "active").sum(), (df["status"] == "inactive").sum(),
    )
    return df


def select_listed_common(registry: pd.DataFrame) -> pd.DataFrame:
    """Restrict the registry to plausible exchange-listed common stock."""
    df = registry[registry["exchange"].isin(MAJOR_EXCHANGES)].copy()
    before = len(df)
    df = df[df["symbol"].str.match(_TICKER_RE, na=False)]
    logger.info(
        "universe: %d symbols on %s, %d after ticker-shape filter (%d dropped)",
        before, sorted(MAJOR_EXCHANGES), len(df), before - len(df),
    )

    # Drop non-common issues the registry names but does not flag structurally.
    name = df["name"].fillna("").str.lower()
    pattern = r"\b(?:warrant|unit|right|preferred|depositary|debenture|etf|fund|trust|index)\b"
    df = df[~name.str.contains(pattern, regex=True)]
    logger.info("universe: %d after name-based non-common filter", len(df))

    return df.reset_index(drop=True).sort_values("symbol")


def screen_daily_panel(
    panel: pd.DataFrame,
    spec: UniverseSpec,
    price_col: str = "close_unadj",
) -> pd.DataFrame:
    """Flag which (symbol, date) rows are in the tradable microcap universe.

    Adds ``in_universe`` plus the intermediate columns the screen depends on,
    all computed **causally** -- ADV uses a trailing window shifted by one day,
    so a row never sees its own volume.

    The price screen deliberately uses unadjusted prices. A stock trading at
    $2 in 2022 that later does a 1-for-20 reverse split appears at $40 in an
    adjusted series; screening on the adjusted price would exclude it from a
    penny-stock study on exactly the days it *was* a penny stock, and admit it
    on days it was not.
    """
    df = panel.sort_values(["symbol", "date"]).copy()

    if price_col not in df.columns:
        raise ValueError(
            f"screen requires {price_col!r}; unadjusted prices are mandatory for "
            "a penny-stock price screen"
        )

    grp = df.groupby("symbol", sort=False)
    df["dollar_volume"] = df["close"] * df["volume"]
    # shift(1) before rolling: the ADV a decision may use is the one observable
    # before that session opens.
    df["adv_dollar"] = (
        grp["dollar_volume"]
        .transform(lambda s: s.shift(1).rolling(spec.adv_window, min_periods=5).mean())
    )

    df["in_universe"] = (
        df[price_col].between(spec.min_price, spec.max_price)
        & (df["adv_dollar"] >= spec.min_dollar_volume)
    )
    return df


def reconstruct_listing_windows(panel: pd.DataFrame) -> pd.DataFrame:
    """First and last observed bar per symbol.

    The asset registry carries no listing/delisting dates, so the bar record is
    the best available proxy for when a symbol was actually tradable. Used to
    document coverage honestly rather than to filter.
    """
    out = (
        panel.groupby("symbol")["date"]
        .agg(first_bar="min", last_bar="max", n_bars="count")
        .reset_index()
    )
    return out.sort_values("symbol")
