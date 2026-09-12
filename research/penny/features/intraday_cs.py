"""intraday_cs.py — cross-sectional features from the minute tape.

Why this file exists
---------------------
`data/research_penny/minute/` holds eleven years of one-minute bars and has only
ever been used for premarket aggregates and barrier labelling. It has never been
turned into cross-sectional features, which makes it the one genuinely untapped
information source left in the project — every non-price block (dilution,
filings, short interest) is already measured dead (`6cb0b55`), and every
daily-price block is already in the composite.

The estimand this serves
-------------------------
§78 established the signal is monotone but **left-tailed**: the composite removes
losers and cannot pick winners, so §80 measured its top decile at −1.18% excess
against +0.77% for a random feature. A Spearman IC on the mean is a poor detector
of a right-tail effect. With a +100% barrier the portfolio return is dominated by
*which names touch it*, so these features are scored two ways: the usual
within-cell IC, and `P(touch the barrier | f)` as a precision ladder.

Leakage rules
--------------
Every feature here is computed from the **signal session's own bars, 09:30 to
15:59**, and entry is the next open — so nothing reads a bar the decision could
not have seen. `entry_prices` deliberately reads the *following* session and is
used only to price the entry, never to select.

THE ADJUSTMENT BASIS, which cost a feature
-------------------------------------------
`alpaca_fetch` requests minute bars with **`adjustment="all"`**, so this cache is
SPLIT- AND DIVIDEND-ADJUSTED, while `panel_strategy.parquet` carries the raw
`*_unadj` series. Measured on the development entry sessions, the minute open and
the daily unadjusted open agree within 1% on only **64.2%** of symbol-days; SOS
on 2021-03-26 prints 4672.50 against 6.22, a ratio of exactly 750.

Two rules follow, and the second is SS72's rule in a new place:

1. **Never pair a minute price level with an unadjusted daily one.** Convert to a
   RATIO within the session first and apply that ratio to the unadjusted price.
2. **Never emit a price LEVEL as a feature.** An adjusted level is computed from
   corporate actions that had not happened yet, so any feature carrying one is a
   look-ahead. `M_mean_trade_usd` was exactly that -- it scored IC -0.0528
   (t -6.82), the second strongest in the project, because a name that later
   reverse-splits carries an inflated adjusted dollar volume TODAY and reverse
   splits happen to names that keep falling. It is withdrawn; see LOG 94.

Everything still emitted below is **scale-invariant within the session** -- a
ratio, a share, or a standard deviation of log returns -- so a constant
adjustment factor cancels exactly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from research.penny.etl import bars

logger = logging.getLogger(__name__)

MINUTE_DIR = bars.DATA_ROOT / "minute"
OPEN_ET, CLOSE_ET = "09:30", "15:59"


def _minute_index(s: pd.Series) -> pd.Series:
    """"HH:MM" -> minutes since 09:30."""
    hh = s.str.slice(0, 2).astype(int)
    mm = s.str.slice(3, 5).astype(int)
    return hh * 60 + mm - (9 * 60 + 30)


def session_path(day: pd.Timestamp) -> Path:
    return MINUTE_DIR / f"dt={day:%Y-%m-%d}" / "minute.parquet"


def load_session(day: pd.Timestamp,
                 symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    p = session_path(day)
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    if symbols is not None:
        d = d[d["symbol"].isin(set(symbols))]
    if d.empty:
        return d
    d = d[(d["minute_et"] >= OPEN_ET) & (d["minute_et"] <= CLOSE_ET)].copy()
    d["mi"] = _minute_index(d["minute_et"])
    return d.sort_values(["symbol", "mi"])


def session_features(d: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol intraday shape for one session.

    Named `M_*` so they are visibly distinct from the daily `B*` blocks, and so
    the FDR ledger can count them separately.
    """
    if d.empty:
        return pd.DataFrame()
    # Every derived column must exist BEFORE the groupby is taken: a groupby
    # object holds a reference to the frame it was built from, so assigning
    # afterwards leaves the aggregation looking at columns that are not there.
    d = d.copy()
    d["_dv"] = d["close"] * d["volume"]
    tot_vol = d.groupby("symbol", sort=False)["volume"].transform("sum")
    d["_volshare"] = d["volume"] / tot_vol.replace(0, np.nan)
    prev = d.groupby("symbol", sort=False)["close"].shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d["_lr"] = np.log(d["close"] / prev)

    g = d.groupby("symbol", sort=False)

    first_hour = d["mi"] < 60
    last_hour = d["mi"] >= 330
    opening_15 = d["mi"] < 15
    closing_15 = d["mi"] >= 375

    agg = g.agg(
        n_minutes=("close", "size"),
        _first=("close", "first"),
        _last=("close", "last"),
        _hi=("high", "max"),
        _lo=("low", "min"),
        _vol=("volume", "sum"),
        _dv=("_dv", "sum"),
        _trades=("trade_count", "sum"),
        _maxvolshare=("_volshare", "max"),
    )

    def _std(mask: pd.Series) -> pd.Series:
        return d.loc[mask].groupby("symbol")["_lr"].std()

    def _volshare(mask: pd.Series) -> pd.Series:
        return d.loc[mask].groupby("symbol")["_volshare"].sum()

    agg["M_intraday_vol"] = d.groupby("symbol")["_lr"].std()
    v1, v2 = _std(first_hour), _std(last_hour)
    agg["M_vol_term_ratio"] = (v1 / v2.replace(0, np.nan)).reindex(agg.index)
    agg["M_open_participation"] = _volshare(opening_15).reindex(agg.index)
    agg["M_close_participation"] = _volshare(closing_15).reindex(agg.index)

    # where the session's extremes printed, as a fraction of the day
    lo_i = d.loc[d.groupby("symbol")["low"].idxmin(), ["symbol", "mi"]]
    hi_i = d.loc[d.groupby("symbol")["high"].idxmax(), ["symbol", "mi"]]
    agg["M_time_of_low"] = lo_i.set_index("symbol")["mi"].reindex(agg.index) / 389.0
    agg["M_time_of_high"] = hi_i.set_index("symbol")["mi"].reindex(agg.index) / 389.0

    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = agg["_dv"] / agg["_vol"].replace(0, np.nan)
        agg["M_vwap_position"] = agg["_last"] / vwap - 1.0
        agg["M_intraday_drawdown"] = agg["_lo"] / agg["_first"] - 1.0
        rng = (agg["_hi"] - agg["_lo"]).replace(0, np.nan)
        agg["M_recovery"] = (agg["_last"] - agg["_lo"]) / rng
        agg["M_open_to_close"] = agg["_last"] / agg["_first"] - 1.0
        # NEITHER M_mean_trade_usd NOR M_shares_per_trade is emitted, and the
        # second one is the more instructive corpse.
        #
        # M_mean_trade_usd is a PRICE level times volume -- obviously adjusted.
        # M_shares_per_trade was its supposedly safe replacement, price removed.
        # But Alpaca adjusts VOLUME as well, in the opposite direction, so a name
        # that later reverse-splits carries a DEFLATED adjusted share count
        # today. Measured on the holdout, the top decile by shares-per-trade
        # contains 0.0% of names that later reverse-split against 7.4% in the
        # rest, and its Spearman with the adjustment ratio is -0.307. It selects
        # survivors using tomorrow's corporate actions. See LOG 97.
        agg["M_volume_spikiness"] = agg["_maxvolshare"]
    agg["M_minutes_traded"] = agg["n_minutes"] / 390.0

    keep = [c for c in agg.columns if c.startswith("M_")]
    out = agg[keep].reset_index()
    out["date"] = d["date"].iloc[0]
    return out


def build_intraday_features(candidates: pd.DataFrame,
                            log_every: int = 200) -> pd.DataFrame:
    """`M_*` features for every candidate symbol-day with a cached session."""
    frames: List[pd.DataFrame] = []
    days = sorted(pd.to_datetime(candidates["date"]).unique())
    by_day = {d: set(g["symbol"]) for d, g in candidates.groupby("date")}
    missing = 0
    for i, day in enumerate(days, 1):
        day = pd.Timestamp(day)
        d = load_session(day, by_day.get(day))
        if d.empty:
            missing += 1
            continue
        frames.append(session_features(d))
        if i % log_every == 0:
            logger.info("  intraday features: %d/%d sessions (%d missing)",
                        i, len(days), missing)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info("intraday features: %s rows over %d sessions (%d missing)",
                f"{len(out):,}", len(days) - missing, missing)
    return out


def entry_prices(candidates: pd.DataFrame, calendar: Iterable[pd.Timestamp],
                 log_every: int = 200) -> pd.DataFrame:
    """Executable entry prices on the session AFTER the signal.

    Three candidates for the entry print, which is the whole of H5:

      `E_open`    the first regular-hours print -- what the strategy uses today,
                  and the widest-spread minute of a microcap session
      `E_t1000`   the 10:00 print, after the opening auction imbalance has
                  cleared (Heston, Korajczyk & Sadka 2010)
      `E_vwap30`  volume-weighted average over 09:30-09:59, which is what a
                  participation-capped order would actually achieve
    """
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(list(calendar))).unique()))
    pos = {d: i for i, d in enumerate(cal)}
    c = candidates.copy()
    c["_i"] = pd.to_datetime(c["date"]).map(pos)
    c = c.dropna(subset=["_i"])
    c["_i"] = c["_i"].astype(int) + 1
    c = c[c["_i"] < len(cal)]
    c["entry_date"] = cal[c["_i"].to_numpy()]

    frames: List[pd.DataFrame] = []
    for i, (day, g) in enumerate(c.groupby("entry_date"), 1):
        d = load_session(pd.Timestamp(day), set(g["symbol"]))
        if d.empty:
            continue
        first30 = d[d["mi"] < 30]
        gg = d.groupby("symbol", sort=False)
        out = pd.DataFrame({"E_open": gg["open"].first()})
        at10 = d[d["mi"] == 30].groupby("symbol")["open"].first()
        out["E_t1000"] = at10.reindex(out.index)
        dv = (first30["vwap"] * first30["volume"]).groupby(first30["symbol"]).sum()
        vv = first30.groupby("symbol")["volume"].sum()
        out["E_vwap30"] = (dv / vv.replace(0, np.nan)).reindex(out.index)
        out = out.reset_index()
        out["entry_date"] = pd.Timestamp(day)
        frames.append(out)
        if i % log_every == 0:
            logger.info("  entry prices: %d sessions", i)
    if not frames:
        return pd.DataFrame()
    px = pd.concat(frames, ignore_index=True)
    return c[["symbol", "date", "entry_date"]].merge(
        px, on=["symbol", "entry_date"], how="left")
