"""top30_analysis.py — per-ticker forensics on the 30 largest gainers.

For each event, assembles everything a trader could have known **before 09:35 on
the event date**, plus the intraday path of the move itself, kept strictly
separate so precursor and outcome are never confused.

Look-ahead discipline
---------------------
* Filings are keyed on ``accepted_et`` (the acceptance instant) and filtered
  strictly before 09:35. EDGAR's ``filing_date`` is a calendar date, so an 18:00
  acceptance carries the same date as the morning that preceded it -- using it
  would leak hours of hindsight.
* Every trailing statistic is computed on sessions **before** the event date.
* Premarket uses the 04:00-09:34 window only; the cache is built with a hard
  assertion that no bar at or after 09:35 is ever stored.
* Same-day price data appears only in the clearly-marked outcome block.

    python research/penny/scripts/top30_analysis.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.scripts.fetch_intraday import session_path as minute_path  # noqa: E402
from research.penny.scripts.fetch_premarket import session_path as premarket_path  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("top30")

DECISION = "09:35"
FOREIGN = {"6-K", "20-F", "40-F", "F-1", "F-3", "F-4", "F-1/A", "F-3/A", "6-K/A",
           "20-F/A", "F-6", "F-10"}
SHELF = {"S-3", "S-1", "F-3", "F-1", "S-3/A", "S-1/A", "S-1MEF", "S-3MEF"}
OFFERING = {"424B5", "424B3", "424B4", "424B2", "424B7"}
DEFICIENCY_ITEM = "3.01"
REVERSE_SPLIT_HINT = {"8-K", "8-K/A", "DEF 14A", "PRE 14A", "DEF 14C", "PRE 14C"}


def load_all():
    u = pd.read_parquet(bars.DATA_ROOT / "forensic_universe.parquet")
    sample = pd.read_parquet(bars.DATA_ROOT / "top30_sample.parquet")
    filings = pd.read_parquet(bars.DATA_ROOT / "edgar_filings.parquet").dropna(
        subset=["accepted_et"])
    short_vol = pd.read_parquet(bars.DATA_ROOT / "finra_short_volume.parquet")
    shares = pd.read_parquet(bars.DATA_ROOT / "xbrl_shares.parquet")
    return u, sample, filings, short_vol, shares


def analyse(row: pd.Series, u: pd.DataFrame, filings: pd.DataFrame,
            short_vol: pd.DataFrame, shares: pd.DataFrame) -> Dict:
    s = row["symbol"]
    day = pd.Timestamp(row["date"])
    cut = pd.Timestamp(f"{day:%Y-%m-%d} {DECISION}", tz="America/New_York")
    rec: Dict = {"symbol": s, "date": str(day.date()), "species": row["species"],
                 "c2c": float(row["c2c"]), "gap": float(row["gap"]),
                 "o2h": float(row["o2h"]), "o2c": float(row["o2c"]),
                 "adv20": float(row["adv20"]), "prev_close": float(row["prev_close_unadj"]),
                 "close": float(row["close"]),
                 "events_in_window": int(row.get("events_in_window", 1))}

    # ---------- prior price / volume structure ----------------------------
    hist = u[(u["symbol"] == s) & (u["date"] < day)].sort_values("date").tail(90)
    if not hist.empty:
        h20 = hist.tail(20)
        rec["ret_20d"] = float(hist.iloc[-1]["close"] / h20.iloc[0]["close"] - 1) \
            if len(h20) else np.nan
        rec["ret_5d"] = float(hist.iloc[-1]["close"] / hist.iloc[-6]["close"] - 1) \
            if len(hist) >= 6 else np.nan
        rec["vol_ratio_5_20"] = float(hist["volume"].tail(5).mean()
                                      / max(hist["volume"].tail(20).mean(), 1))
        rec["realised_vol_20d"] = float(h20["c2c"].std())
        rec["pct_off_90d_high"] = float(hist.iloc[-1]["close"] / hist["high"].max() - 1)
        rec["pct_above_90d_low"] = float(hist.iloc[-1]["close"] / hist["low"].min() - 1)
        rec["prior_50_moves_90d"] = int((hist["c2c"] >= 0.50).sum())
        rec["n_prior_sessions"] = int(len(hist))
        # A "quiet base": low realised vol and a tight range in the last fortnight.
        last10 = hist.tail(10)
        rec["range_10d"] = float(last10["high"].max() / max(last10["low"].min(), 1e-9) - 1)
    # ---------- premarket --------------------------------------------------
    pm_path = premarket_path(day)
    pm = None
    if pm_path.exists():
        try:
            allpm = pd.read_parquet(pm_path)
            pm = allpm[allpm["symbol"] == s].sort_values("minute_et")
        except Exception:  # noqa: BLE001
            pm = None
    if pm is not None and not pm.empty:
        pmdv = float((pm["close"] * pm["volume"]).sum())
        rec["pm_bars"] = int(len(pm))
        rec["pm_dollar_volume"] = pmdv
        rec["pm_dv_to_adv"] = float(pmdv / max(row["adv20"], 1))
        rec["pm_last"] = float(pm.iloc[-1]["close"])
        rec["pm_move"] = float(pm.iloc[-1]["close"] / row["prev_close_unadj"] - 1)
        rec["pm_high_move"] = float(pm["high"].max() / row["prev_close_unadj"] - 1)
        early = pm[pm["minute_et"] < "07:00"]
        rec["pm_dv_before_0700"] = float((early["close"] * early["volume"]).sum())
        rec["pm_started_before_0700"] = bool(len(early) > 0)
    else:
        rec["pm_bars"] = 0
        rec["pm_dollar_volume"] = 0.0
        rec["pm_dv_to_adv"] = 0.0
        rec["pm_move"] = np.nan
        rec["pm_started_before_0700"] = False

    # ---------- filings, strictly before the decision ----------------------
    f = filings[(filings["symbol"] == s) & (filings["accepted_et"] < cut)]
    f180 = f[f["accepted_et"] >= cut - pd.Timedelta(days=180)]
    f30 = f[f["accepted_et"] >= cut - pd.Timedelta(days=30)]
    f365 = f[f["accepted_et"] >= cut - pd.Timedelta(days=365)]
    rec["edgar_covered"] = bool(len(f) > 0)
    rec["filings_180d"] = int(len(f180))
    rec["filings_30d"] = int(len(f30))
    rec["same_morning_filing"] = bool(
        len(f180) and f180["accepted_et"].max() >= cut.normalize())
    rec["prev_evening_filing"] = bool(
        len(f180) and f180["accepted_et"].max() >= cut.normalize() - pd.Timedelta(hours=8))
    rec["is_foreign_issuer"] = bool(f365["form"].isin(FOREIGN).any())
    rec["shelf_180d"] = int(f180["form"].isin(SHELF).sum())
    rec["offering_180d"] = int(f180["form"].isin(OFFERING).sum())
    if len(f180):
        last = f180.sort_values("accepted_et").iloc[-1]
        rec["last_filing_form"] = str(last["form"])
        rec["last_filing_hours_before"] = float(
            (cut - last["accepted_et"]).total_seconds() / 3600.0)
    items = f180["items"].fillna("").astype(str) if "items" in f180.columns else pd.Series(dtype=str)
    rec["deficiency_180d"] = bool(items.str.contains(DEFICIENCY_ITEM).any())
    rec["top_forms_180d"] = f180["form"].value_counts().head(6).to_dict()

    # ---------- short volume ----------------------------------------------
    sv = short_vol[(short_vol["symbol"] == s) & (short_vol["date"] < day)]
    sv = sv.sort_values("date").tail(20)
    if not sv.empty:
        g = sv.groupby("date").agg(sv=("short_volume", "sum"), tv=("total_volume", "sum"))
        ratio = (g["sv"] / g["tv"].replace(0, np.nan)).dropna()
        if len(ratio):
            rec["short_ratio_last"] = float(ratio.iloc[-1])
            rec["short_ratio_20d"] = float(ratio.mean())

    # ---------- share count / dilution -------------------------------------
    if "known_from" in shares.columns:
        kf = pd.to_datetime(shares["known_from"], utc=True).dt.tz_convert(
            "America/New_York").dt.normalize().dt.tz_localize(None)
        mask = (shares["symbol"] == s) & (kf < day.normalize())
        sh = shares[mask].assign(kd=kf[mask]).sort_values("kd")
        if len(sh):
            rec["shares_last"] = float(sh.iloc[-1]["val"])
            rec["float_dollar"] = float(sh.iloc[-1]["val"] * row["prev_close_unadj"])
            if len(sh) >= 2 and sh.iloc[0]["val"] > 0:
                rec["share_growth"] = float(sh.iloc[-1]["val"] / sh.iloc[0]["val"] - 1)

    # ---------- OUTCOME (never a precursor) --------------------------------
    mp = minute_path(day)
    if mp.exists():
        m = pd.read_parquet(mp)
        m = m[m["symbol"] == s].sort_values("minute_et")
        if not m.empty:
            op = float(m.iloc[0]["open"])
            frm = m["high"] / op - 1.0
            peak_i = int(frm.values.argmax())
            rec["peak_minute"] = str(m.iloc[peak_i]["minute_et"])
            rec["peak_from_open"] = float(frm.iloc[peak_i])
            half = m[frm >= 0.5 * frm.iloc[peak_i]]
            if not half.empty:
                rec["half_move_minute"] = str(half.iloc[0]["minute_et"])
            f30m = m[m["minute_et"] <= "10:05"]
            tot = float((m["close"] * m["volume"]).sum())
            rec["first30_dv_share"] = float(
                (f30m["close"] * f30m["volume"]).sum() / max(tot, 1e-9))
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Per-ticker forensics.")
    p.add_argument("--out", default=str(bars.DATA_ROOT / "top30_analysis.json"))
    args = p.parse_args(argv)

    u, sample, filings, short_vol, shares = load_all()
    recs = [analyse(r, u, filings, short_vol, shares) for _, r in sample.iterrows()]
    df = pd.DataFrame(recs)
    df.to_parquet(bars.DATA_ROOT / "top30_analysis.parquet", index=False)
    Path(args.out).write_text(json.dumps(recs, indent=2, default=str), encoding="utf-8")

    cols = ["symbol", "date", "species", "c2c", "gap", "pm_dv_to_adv", "pm_move",
            "vol_ratio_5_20", "ret_20d", "same_morning_filing", "shelf_180d",
            "is_foreign_issuer", "prior_50_moves_90d", "adv20"]
    logger.info("=" * 130)
    logger.info("PER-TICKER PRECURSORS\n%s",
                df[[c for c in cols if c in df.columns]].to_string(
                    index=False, float_format="%.3f"))
    logger.info("=" * 130)
    logger.info("wrote %s and top30_analysis.parquet", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
