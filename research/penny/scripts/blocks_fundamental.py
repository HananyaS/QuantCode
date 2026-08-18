"""blocks_fundamental.py — blocks B4, B5, B6: the non-price information.

B0-B3 and B7-B8 are all price and volume. Round 2 measured filings as the
largest single lift of the project, and none of rounds 5-11 used them. These are
the last untested blocks in FORMULATION.md §3.

Point-in-time discipline is the whole game here:

* **EDGAR** joins on `accepted_et`, never `filing_date`. A filing accepted at
  18:40 was not knowable at that day's close, so the cut is strictly `<` the
  signal timestamp.
* **FINRA** short volume is published with a one-session lag, so the most recent
  usable observation is strictly before the signal date.
* **XBRL** joins on `known_from` -- when the fact first became public -- not
  `period_end`, which is the quarter it describes.

Counts over a window are computed as a difference of cumulative counts at two
as-of points, so a candidate never sees a filing that had not happened.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from research.penny.etl import bars

logger = logging.getLogger(__name__)

SHELF_FORMS = {"S-3", "S-3/A", "S-3ASR", "S-1", "S-1/A"}
TAKEDOWN_PREFIX = "424B"


def _cum_count_asof(events: pd.DataFrame, keys: pd.DataFrame,
                    lookback_days: int) -> np.ndarray:
    """Number of `events` per symbol in (t - lookback, t), strictly before t."""
    if events.empty:
        return np.zeros(len(keys))
    ev = events.sort_values(["symbol", "d"]).copy()
    ev["cum"] = ev.groupby("symbol").cumcount() + 1
    # merge_asof needs the RIGHT frame sorted globally by the join key,
    # not by (symbol, key) -- pandas raises "right keys must be sorted" otherwise.
    ev = ev[["symbol", "d", "cum"]].sort_values("d")

    def asof(at: pd.Series) -> np.ndarray:
        q = keys.assign(d=at).sort_values("d")
        m = pd.merge_asof(q, ev, on="d", by="symbol", allow_exact_matches=False)
        return m.sort_index()["cum"].to_numpy()

    hi = asof(keys["date"])
    lo = asof(keys["date"] - pd.Timedelta(days=lookback_days))
    return np.nan_to_num(hi, nan=0.0) - np.nan_to_num(lo, nan=0.0)


def _days_since_asof(events: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    if events.empty:
        return np.full(len(keys), np.nan)
    ev = events.sort_values(["symbol", "d"])[["symbol", "d"]].copy()
    ev["last"] = ev["d"]
    ev = ev.sort_values("d")
    q = keys.assign(d=keys["date"]).sort_values("d")
    m = pd.merge_asof(q, ev[["symbol", "d", "last"]], on="d", by="symbol",
                      allow_exact_matches=False)
    m = m.sort_index()
    return (m["date"] - m["last"]).dt.days.to_numpy()


def add_fundamental_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """`df` needs `symbol` and `date` (the signal day). Adds B4/B5/B6 columns."""
    keys = df[["symbol", "date"]].reset_index(drop=True)
    # Cached parquets carry mixed datetime resolutions (FINRA is [ms],
    # the panel is [ns]); merge_asof refuses to join across them.
    keys["date"] = keys["date"].astype("datetime64[ns]")

    # ---- B5: filings -------------------------------------------------------
    f = pd.read_parquet(bars.DATA_ROOT / "edgar_filings.parquet",
                        columns=["symbol", "form", "items", "accepted_et"])
    f = f[f["symbol"].isin(set(keys["symbol"]))].copy()
    f["d"] = pd.to_datetime(f["accepted_et"]).dt.tz_localize(None).dt.normalize()
    f["d"] = f["d"].astype("datetime64[ns]")
    f = f.dropna(subset=["d"])

    is_301 = f["form"].str.startswith("8-K", na=False) & \
        f["items"].fillna("").str.contains("3.01", regex=False)
    df["B5_days_since_deficiency"] = _days_since_asof(f[is_301], keys)
    df["B5_n_shelf_180d"] = _cum_count_asof(f[f["form"].isin(SHELF_FORMS)], keys, 180)
    df["B5_n_takedown_90d"] = _cum_count_asof(
        f[f["form"].str.startswith(TAKEDOWN_PREFIX, na=False)], keys, 90)
    df["B5_n_form4_90d"] = _cum_count_asof(f[f["form"] == "4"], keys, 90)
    df["B5_days_since_filing"] = _days_since_asof(f, keys)
    # a deficiency that never happened is not "0 days ago"
    df["B5_days_since_deficiency"] = df["B5_days_since_deficiency"].fillna(9999)
    df["B5_days_since_filing"] = df["B5_days_since_filing"].fillna(9999)

    # ---- B6: short volume (one-session lag) --------------------------------
    s = pd.read_parquet(bars.DATA_ROOT / "finra_short_volume.parquet")
    s = s[s["symbol"].isin(set(keys["symbol"]))].copy()
    s = s.groupby(["symbol", "date"], as_index=False).sum(numeric_only=True)
    s["ratio"] = s["short_volume"] / s["total_volume"].replace(0, np.nan)
    s = s.sort_values(["symbol", "date"])
    s["ratio_20d"] = s.groupby("symbol")["ratio"].transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    s = s.rename(columns={"date": "d"})[["symbol", "d", "ratio", "ratio_20d"]]
    s["d"] = s["d"].astype("datetime64[ns]")
    q = keys.assign(d=keys["date"]).sort_values("d")
    m = pd.merge_asof(q, s.sort_values("d"), on="d", by="symbol",
                      allow_exact_matches=False).sort_index()
    df["B6_short_ratio"] = m["ratio"].to_numpy()
    df["B6_short_ratio_20d"] = m["ratio_20d"].to_numpy()

    # ---- B4: dilution (shares outstanding growth) --------------------------
    x = pd.read_parquet(bars.DATA_ROOT / "xbrl_shares.parquet",
                        columns=["symbol", "known_from", "val"])
    x = x[x["symbol"].isin(set(keys["symbol"]))].copy()
    x["d"] = pd.to_datetime(x["known_from"]).dt.tz_localize(None).dt.normalize()
    x["d"] = x["d"].astype("datetime64[ns]")
    x = x.dropna(subset=["d", "val"]).sort_values(["symbol", "d"])
    x = x.drop_duplicates(["symbol", "d"], keep="last")[["symbol", "d", "val"]]
    x = x.sort_values("d")

    def shares_at(offset_days: int) -> np.ndarray:
        q2 = keys.assign(d=keys["date"] - pd.Timedelta(days=offset_days)).sort_values("d")
        mm = pd.merge_asof(q2, x, on="d", by="symbol", allow_exact_matches=False)
        return mm.sort_index()["val"].to_numpy()

    now, yr = shares_at(0), shares_at(365)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["B4_shares_growth_1y"] = np.where((yr > 0) & (now > 0), now / yr - 1.0, np.nan)
    return df
