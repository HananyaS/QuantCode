"""fetch_benchmarks.py -- SPY / QQQ / IWM total-return daily bars and the T-bill rate.

The comparison in PREREGISTRATION_R17 §4 needs total return. This pulls Alpaca
SIP daily bars with `adjustment="all"`, stated explicitly rather than inherited.
`data/research_poolpoc/etf/daily.parquet` does not record its adjustment,
though its 2020-23 SPY returns match these. It also pulls the 3-month T-bill
(FRED DTB3, public CSV) for excess returns. Check: SPY calendar-year returns
from this file are +18.45% / +28.60% / -18.18% / +26.18% for 2020-23, the
published total returns.

Refuses an end date on or after SEAL without --unseal.

    python research/penny/scripts/fetch_benchmarks.py --start 2019-12-01 --end 2023-12-29
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import urllib.request
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.dates import SEAL                                   # noqa: E402
from research.penny.etl import bars                                     # noqa: E402
from research.penny.scripts.fetch_corporate_actions import KEY_PAIRS    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bench")

OUT_DIR = bars.DATA_ROOT / "benchmarks"
SYMBOLS = ("SPY", "QQQ", "IWM")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"


def _client():
    from dotenv import dotenv_values
    from alpaca.data.historical import StockHistoricalDataClient
    env_file = next(p / ".env" for p in Path(__file__).resolve().parents if (p / ".env").is_file())
    env = dotenv_values(env_file)
    for k, s in KEY_PAIRS:
        if env.get(k) and env.get(s):
            return StockHistoricalDataClient(env[k], env[s])
    raise SystemExit("no Alpaca key pair found in .env")


def fetch_etf(start: date, end: date) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    req = StockBarsRequest(symbol_or_symbols=list(SYMBOLS), timeframe=TimeFrame.Day,
                           start=datetime.combine(start, time()),
                           end=datetime.combine(end + timedelta(days=1), time()),
                           adjustment=Adjustment.ALL, feed=DataFeed.SIP)
    df = _client().get_stock_bars(req).df.reset_index()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    df = df[df["date"] <= pd.Timestamp(end)]
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]]


def fetch_rf(start: date, end: date) -> pd.DataFrame:
    with urllib.request.urlopen(FRED, timeout=60) as resp:      # public series, no identity sent
        raw = resp.read().decode("utf-8")
    rf = pd.read_csv(io.StringIO(raw))
    rf.columns = ["date", "dtb3"]
    rf["date"] = pd.to_datetime(rf["date"])
    rf["dtb3"] = pd.to_numeric(rf["dtb3"], errors="coerce")
    rf = rf[(rf["date"] >= pd.Timestamp(start)) & (rf["date"] <= pd.Timestamp(end))]
    rf["rf_daily"] = rf["dtb3"].ffill() / 100.0 / 252.0          # discount yield -> simple daily
    return rf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--unseal", action="store_true")
    args = ap.parse_args(argv)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if pd.Timestamp(end) >= SEAL and not args.unseal:
        raise SystemExit(f"end {end} reaches the sealed period ({SEAL.date()}); pass --unseal")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    etf = fetch_etf(start, end)
    etf.to_parquet(OUT_DIR / f"etf_daily_adj_{start:%Y%m%d}_{end:%Y%m%d}.parquet", index=False)
    logger.info("ETF bars: %s", etf.groupby("symbol")["date"].agg(["min", "max", "count"]).to_string())
    rf = fetch_rf(start, end)
    rf.to_parquet(OUT_DIR / f"dtb3_{start:%Y%m%d}_{end:%Y%m%d}.parquet", index=False)
    logger.info("DTB3: %d rows, mean %.2f%%", len(rf), rf["dtb3"].mean())
    return 0


def load(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """Daily total returns (columns SPY, QQQ, IWM) and the daily T-bill rate on [start, end]."""
    files = sorted(OUT_DIR.glob("etf_daily_adj_*.parquet"))
    rfs = sorted(OUT_DIR.glob("dtb3_*.parquet"))
    if not files or not rfs:
        raise FileNotFoundError("run scripts/fetch_benchmarks.py first")
    etf = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(["symbol", "date"], keep="last")
    px = etf.pivot(index="date", columns="symbol", values="close").sort_index()
    ret = px.pct_change().loc[pd.Timestamp(start):pd.Timestamp(end), list(SYMBOLS)]
    rf = pd.concat([pd.read_parquet(f) for f in rfs]).drop_duplicates("date", keep="last")
    rf = rf.set_index("date")["rf_daily"].sort_index()
    return ret, rf


if __name__ == "__main__":
    sys.exit(main())
