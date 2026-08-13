"""forensic_movers.py — a dossier per 50%+ mover: what was visible beforehand.

Reads each name in the forensic sample and assembles everything knowable
**before 09:35 on the move day**, plus the intraday path of the move itself, into
one markdown file. The point is to see precursors that averaging destroys.

Discipline that keeps this from being storytelling
--------------------------------------------------
* Every filing is keyed on ``accepted_et`` and filtered strictly before the
  decision instant. EDGAR's ``filing_date`` is a date; acceptance is an instant,
  and an 18:00 acceptance carries the same date as the morning before it.
* Trailing statistics use sessions strictly prior to the move day.
* The intraday path *is* the outcome, and is printed separately from the
  precursors so the two cannot be confused.

Nothing here is evidence on its own. These names were selected **because** they
moved, so every pattern is conditioned on success; ``forensic_precursors.py``
measures each against its base rate in the full universe.

    python research/penny/scripts/forensic_movers.py
"""
from __future__ import annotations

import argparse
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
logger = logging.getLogger("dossier")

DECISION = "09:35"
FOREIGN_FORMS = {"6-K", "20-F", "40-F", "F-1", "F-3", "F-4", "F-1/A", "F-3/A",
                 "6-K/A", "20-F/A"}
SHELF_FORMS = {"S-3", "S-1", "F-3", "F-1", "S-3/A", "S-1/A"}
OFFERING_FORMS = {"424B5", "424B3", "424B4", "424B2"}


def _fmt(x, nd=2, dollar=False, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if pct:
        return f"{100 * x:+.1f}%"
    if dollar:
        return f"${x:,.0f}"
    return f"{x:,.{nd}f}"


def daily_history(universe: pd.DataFrame, symbol: str, day: pd.Timestamp,
                  lookback: int = 90) -> pd.DataFrame:
    h = universe[(universe["symbol"] == symbol) & (universe["date"] < day)]
    return h.sort_values("date").tail(lookback)


def intraday(symbol: str, day: pd.Timestamp) -> Optional[pd.DataFrame]:
    p = minute_path(day)
    if not p.exists():
        return None
    m = pd.read_parquet(p)
    m = m[m["symbol"] == symbol]
    return m.sort_values("minute_et") if not m.empty else None


def premarket(symbol: str, day: pd.Timestamp) -> Optional[pd.DataFrame]:
    p = premarket_path(day)
    if not p.exists():
        return None
    try:
        m = pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None
    m = m[m["symbol"] == symbol] if "symbol" in m.columns else m.iloc[0:0]
    return m.sort_values("minute_et") if not m.empty else None


def filings_before(filings: pd.DataFrame, symbol: str, day: pd.Timestamp,
                   days: int = 180) -> pd.DataFrame:
    cut = pd.Timestamp(f"{day:%Y-%m-%d} {DECISION}", tz="America/New_York")
    f = filings[(filings["symbol"] == symbol)
                & (filings["accepted_et"] < cut)
                & (filings["accepted_et"] >= cut - pd.Timedelta(days=days))]
    return f.sort_values("accepted_et", ascending=False)


def dossier(row: pd.Series, universe: pd.DataFrame, filings: pd.DataFrame,
            short_vol: pd.DataFrame, shares: pd.DataFrame) -> str:
    s, day = row["symbol"], pd.Timestamp(row["date"])
    out: List[str] = []
    a = out.append

    a(f"# {s} — {day:%Y-%m-%d} — {row['species'].replace('_', ' ')}\n")
    a(f"**Close-to-close {_fmt(row['c2c'], pct=True)}** "
      f"(prev close ${row['prev_close_unadj']:.2f} → close ${row['close']:.2f})\n")
    a("| | |\n|---|---|")
    a(f"| opening gap | {_fmt(row['gap'], pct=True)} |")
    a(f"| open → high | {_fmt(row['o2h'], pct=True)} |")
    a(f"| low → high | {_fmt(row['l2h'], pct=True)} |")
    a(f"| open → close | {_fmt(row['o2c'], pct=True)} |")
    a(f"| trailing ADV(20) | {_fmt(row['adv20'], dollar=True)} |")
    a(f"| volume that day | {_fmt(row['volume'], nd=0)} |")
    a(f"| volume / ADV | {_fmt(row['volume'] * row['close'] / max(row['adv20'], 1), nd=1)}x |")
    a("")

    # ---- what was knowable before the bell --------------------------------
    a("## Before the bell\n")
    hist = daily_history(universe, s, day)
    if hist.empty:
        a("_no prior daily history_\n")
    else:
        prior = hist.tail(20)
        a(f"- prior 20 sessions: mean daily move "
          f"{_fmt(prior['c2c'].mean(), pct=True)}, "
          f"realised vol {_fmt(prior['c2c'].std(), pct=True)}")
        a(f"- price 20 sessions ago ${prior.iloc[0]['close']:.2f} → "
          f"${hist.iloc[-1]['close']:.2f} "
          f"({_fmt(hist.iloc[-1]['close'] / prior.iloc[0]['close'] - 1, pct=True)})")
        v5, v20 = hist["volume"].tail(5).mean(), hist["volume"].tail(20).mean()
        a(f"- volume ramp: last 5 sessions {_fmt(v5, nd=0)} vs 20-session "
          f"{_fmt(v20, nd=0)} ({_fmt(v5 / max(v20, 1), nd=2)}x)")
        big = hist[hist["c2c"] >= 0.50]
        a(f"- prior +50% days in the last 90 sessions: **{len(big)}**"
          + (f" ({', '.join(f'{d:%m-%d}' for d in big['date'])})" if len(big) else ""))
        a(f"- 90-session high ${hist['high'].max():.2f}, low ${hist['low'].min():.2f}")
    a("")

    pm = premarket(s, day)
    if pm is not None and not pm.empty:
        pv = float((pm["close"] * pm["volume"]).sum())
        a(f"- **premarket 04:00–09:34**: {len(pm)} bars, "
          f"${pv:,.0f} traded ({_fmt(pv / max(row['adv20'], 1), nd=2)}x ADV), "
          f"last ${pm.iloc[-1]['close']:.2f} vs prior close "
          f"${row['prev_close_unadj']:.2f} "
          f"({_fmt(pm.iloc[-1]['close'] / row['prev_close_unadj'] - 1, pct=True)})")
    else:
        a("- **premarket**: no bars (did not trade pre-open)")
    a("")

    # ---- filings ----------------------------------------------------------
    f180 = filings_before(filings, s, day, 180)
    cut = pd.Timestamp(f"{day:%Y-%m-%d} {DECISION}", tz="America/New_York")
    same_morning = f180[f180["accepted_et"] >= cut.normalize()]
    a("## SEC filings (accepted strictly before 09:35)\n")
    if f180.empty:
        a("_no filings in the prior 180 days_\n")
    else:
        a(f"- {len(f180)} filings in 180 days; "
          f"{len(f180[f180['accepted_et'] >= cut - pd.Timedelta(days=30)])} in 30 days")
        a(f"- **same morning before the bell: {len(same_morning)}**"
          + (f" — {', '.join(same_morning['form'].tolist()[:5])}" if len(same_morning) else ""))
        foreign = f180[f180["form"].isin(FOREIGN_FORMS)]
        a(f"- foreign-issuer forms (6-K/20-F/F-*): **{len(foreign)}** → "
          f"{'**foreign private issuer**' if len(foreign) else 'domestic filer'}")
        shelf = f180[f180["form"].isin(SHELF_FORMS)]
        offering = f180[f180["form"].isin(OFFERING_FORMS)]
        a(f"- shelf registrations: {len(shelf)}; offering prospectuses: {len(offering)}")
        a("")
        a("| accepted (ET) | days before | form |")
        a("|---|---|---|")
        for _, r in f180.head(12).iterrows():
            a(f"| {r['accepted_et']:%Y-%m-%d %H:%M} | "
              f"{(cut - r['accepted_et']).days} | {r['form']} |")
    a("")

    # ---- short volume and dilution ---------------------------------------
    sv = short_vol[(short_vol["symbol"] == s) & (short_vol["date"] < day)]
    sv = sv.sort_values("date").tail(20)
    a("## Short volume and share count\n")
    if sv.empty:
        a("- no FINRA short-volume rows in the prior 20 sessions")
    else:
        g = sv.groupby("date").agg(sv=("short_volume", "sum"),
                                   tv=("total_volume", "sum"))
        ratio = (g["sv"] / g["tv"].replace(0, np.nan)).dropna()
        if len(ratio):
            a(f"- short-volume ratio: last {_fmt(ratio.iloc[-1], nd=3)}, "
              f"20-session mean {_fmt(ratio.mean(), nd=3)}")

    # `known_from` is tz-aware (the acceptance instant of the disclosing filing)
    # while `day` is a naive session date. Compare on calendar dates so the two
    # dtypes never meet -- and keep the strict inequality, since a share count
    # disclosed *on* the move day is not knowable before the bell.
    if "known_from" in shares.columns:
        kf = pd.to_datetime(shares["known_from"], utc=True).dt.tz_convert(
            "America/New_York").dt.normalize().dt.tz_localize(None)
        sh = shares[(shares["symbol"] == s) & (kf < day.normalize())].assign(
            known_date=kf[(shares["symbol"] == s) & (kf < day.normalize())])
    else:
        sh = shares.iloc[0:0]
    if not sh.empty:
        sh = sh.sort_values("known_date").tail(4)
        vals = ", ".join(f"{r['known_date']:%Y-%m} {r['val']:,.0f}"
                         for _, r in sh.iterrows())
        first, last = sh.iloc[0]["val"], sh.iloc[-1]["val"]
        a(f"- shares outstanding: {vals}")
        if first > 0:
            a(f"- share growth over that window: {_fmt(last / first - 1, pct=True)}")
    else:
        a("- no XBRL share counts before this date")
    a("")

    # ---- the outcome ------------------------------------------------------
    a("## The move itself (outcome — not a precursor)\n")
    m = intraday(s, day)
    if m is None or m.empty:
        a("_no minute bars cached_\n")
    else:
        opening = m.iloc[0]["open"]
        m = m.assign(from_open=m["high"] / opening - 1.0)
        peak = m.loc[m["from_open"].idxmax()]
        a(f"- first bar {m.iloc[0]['minute_et']} at ${opening:.2f}; "
          f"peak ${peak['high']:.2f} at **{peak['minute_et']}** "
          f"({_fmt(peak['from_open'], pct=True)} from the open)")
        half = m[m["from_open"] >= 0.5 * peak["from_open"]]
        if not half.empty:
            a(f"- half the intraday move was done by **{half.iloc[0]['minute_et']}**")
        first30 = m[m["minute_et"] <= "10:05"]
        if not first30.empty:
            share = float((first30["close"] * first30["volume"]).sum())
            total = float((m["close"] * m["volume"]).sum())
            a(f"- first 30 minutes carried {_fmt(share / max(total, 1e-9) * 100, nd=0)}% "
              f"of the day's dollar volume")
    a("")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build mover dossiers.")
    parser.add_argument("--sample", default=str(bars.DATA_ROOT / "forensic_sample.parquet"))
    parser.add_argument("--out-dir", default="research/penny/forensics")
    args = parser.parse_args(argv)

    sample = pd.read_parquet(args.sample)
    universe = pd.read_parquet(bars.DATA_ROOT / "forensic_universe.parquet")
    filings = pd.read_parquet(bars.DATA_ROOT / "edgar_filings.parquet").dropna(
        subset=["accepted_et"])
    short_vol = pd.read_parquet(bars.DATA_ROOT / "finra_short_volume.parquet")
    shares = pd.read_parquet(bars.DATA_ROOT / "xbrl_shares.parquet")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts: List[str] = []
    for _, row in sample.iterrows():
        text = dossier(row, universe, filings, short_vol, shares)
        path = out_dir / f"{row['symbol']}_{pd.Timestamp(row['date']):%Y%m%d}.md"
        path.write_text(text, encoding="utf-8")
        parts.append(text)
        logger.info("wrote %s", path)

    combined = out_dir / "ALL_DOSSIERS.md"
    combined.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    logger.info("wrote %s (%d dossiers)", combined, len(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
