"""screen_today.py — run both strategies against the current tape.

What this does
--------------
Fetches the last ~60 sessions of daily bars for the whole listed universe and
applies the two rules that survived audit, exactly as specified:

**LONG** (LOG §65) -- price $0.50-$2.00, ADV20 >= $250k, closed down >=40% over
the previous five sessions, not itself a reverse split. Entry is the *next*
open, so a name that fires on the last completed session is actionable at the
next one.

**SHORT** (LOG §63) -- the unadjusted overnight multiple sits on a round
consolidation ratio with an inert session, i.e. a reverse split the vendor never
adjusted. ADV20 >= $250k.

Both screens need **unadjusted** prices. The split detector compares an opening
print to the prior close, and mixing adjusted with unadjusted there is what
produced WOK at $4.8M/share in the first §63 run. Raw and adjusted series are
fetched separately and merged, the same way `bars.fetch_year` does it.

What this is not
----------------
A recommendation. Neither rule has traded a share, the long's parameters were
chosen on the same data that measured them, and the short assumes a borrow that
may not exist. This screen is the forward test starting, not its result.

    python research/penny/scripts/screen_today.py
    python research/penny/scripts/screen_today.py --asof 2026-08-07
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import alpaca_fetch, bars, universe  # noqa: E402
from research.penny.etl.corporate_actions import (  # noqa: E402
    flag_forward_splits,
    flag_reverse_splits,
)
from research.penny.scripts.fetch_daily_history import resolve_credentials  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("screen")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

PRICE_LO, PRICE_HI = 0.50, 2.00
DROP, LOOKBACK = 0.40, 5
ADV_FLOOR = 2.5e5
TARGET, HOLD = 0.50, 10


def fetch_window(client, symbols: List[str], start: str, end: str,
                 batch_size: int = 200) -> pd.DataFrame:
    """Raw and adjusted daily bars over one window, merged on (symbol, date)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    def make_request(adjustment: str):
        def request(batch: List[str]) -> pd.DataFrame:
            time.sleep(alpaca_fetch.REQUEST_PAUSE_SECONDS)
            return client.get_stock_bars(
                StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                 start=start, end=end, adjustment=adjustment,
                                 feed="sip")
            ).df
        return request

    def pull(adjustment: str) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        for i, b in enumerate(batches, 1):
            if i % 10 == 0 or i == 1:
                logger.info("  %s bars: batch %d/%d", adjustment, i, len(batches))
            df, _ = alpaca_fetch.fetch_with_rejects(make_request(adjustment), b)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = alpaca_fetch._normalise(pd.concat(frames))
        out["date"] = out["timestamp"].dt.normalize().dt.tz_localize(None)
        return out.drop(columns=["timestamp"])

    adj = pull("all")
    raw = pull("raw")
    if adj.empty or raw.empty:
        return pd.DataFrame()
    raw = raw.rename(columns={c: f"{c}_unadj" for c in
                              ("open", "high", "low", "close")})
    keep = ["symbol", "date", "open_unadj", "high_unadj", "low_unadj", "close_unadj"]
    return adj.merge(raw[keep], on=["symbol", "date"], how="inner")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Screen the current tape.")
    ap.add_argument("--asof", default=None,
                    help="Session to screen (default: the latest one fetched).")
    ap.add_argument("--days", type=int, default=70, help="Calendar days of history.")
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "screen_today.json"))
    args = ap.parse_args(argv)

    load_dotenv(_PROJECT_ROOT / ".env")
    key, secret = resolve_credentials()
    client = alpaca_fetch.make_data_client(key, secret)

    reg_path = bars.DATA_ROOT / "asset_registry.parquet"
    if reg_path.exists():
        registry = pd.read_parquet(reg_path)
        logger.info("registry: %d rows from cache", len(registry))
    else:
        registry = universe.fetch_asset_registry(key, secret)
        registry.to_parquet(reg_path, index=False)
    listed = universe.select_listed_common(registry)
    active = listed[listed["status"] == "active"] if "status" in listed else listed
    symbols = sorted(active["symbol"].unique())
    logger.info("universe: %s active listed symbols", f"{len(symbols):,}")

    # `end` must be an INSTANT slightly in the past, not a local calendar date.
    # Two separate traps, both of which return zero rows rather than an error:
    #   * `date.today()` is the *machine's* date. On a UTC+9 box that is a day
    #     ahead of US/Eastern, so on a Friday evening in Asia it names a
    #     session the US market has not reached, and every request is rejected
    #     as "recent SIP data".
    #   * Alpaca treats a bare date as midnight UTC, which is 20:00 ET the
    #     *previous* day -- so passing the target date silently excludes that
    #     date's own bar.
    # Anchoring to "now minus 20 minutes" in UTC satisfies both, and clears the
    # free tier's 15-minute delay on recent SIP data.
    end_dt = datetime.now(timezone.utc) - timedelta(minutes=20)
    end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    start = (end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")
    logger.info("fetching %s .. %s (ET now: %s)", start, end,
                datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M"))
    px = fetch_window(client, symbols, start, end)
    if px.empty:
        raise SystemExit("no bars returned")

    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
    sessions = np.sort(px["date"].unique())
    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(sessions[-1])
    logger.info("fetched %s rows | %d sessions | screening %s",
                f"{len(px):,}", len(sessions), str(asof)[:10])

    g = px.groupby("symbol", sort=False)
    # ADV must be computed exactly as the backtest computes it: STRICTLY
    # TRAILING, excluding the signal day itself. A crashed name trades
    # enormous volume on its crash day -- including it would inflate the
    # liquidity measure by precisely the amount the crash caused, and would
    # admit names the measured result never admitted. See
    # `build_universe._derive`, which this mirrors.
    px["dv"] = px["close"] * px["volume"]
    px["adv20"] = px.groupby("symbol", sort=False)["dv"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    px[f"ret_{LOOKBACK}d"] = g["close"].shift(0) / g["close"].shift(LOOKBACK) - 1.0
    px["prev_close_unadj"] = g["close_unadj"].shift(1)
    px["o2h"] = px["high_unadj"] / px["open_unadj"] - 1.0
    px["o2c"] = px["close_unadj"] / px["open_unadj"] - 1.0
    px = flag_reverse_splits(px, prev_close_col="prev_close_unadj",
                             open_col="open_unadj", o2h_col="o2h")
    px["any_split"] = px["is_suspect_split"] | px["is_possible_split"]

    # A forward split prints as a large overnight DECLINE, so the reverse
    # detector is structurally blind to it and a buy-the-crash screen would
    # rank it as its most extreme signal. It also has to be excluded across the
    # whole lookback: a 1:10 split three sessions ago makes ret_5d read -90%
    # even though the signal day itself looks ordinary.
    px = flag_forward_splits(px, prev_close_col="prev_close_unadj",
                             open_col="open_unadj", high_col="high_unadj",
                             low_col="low_unadj")
    px["fwd_split_in_lookback"] = (
        px.groupby("symbol", sort=False)["is_suspect_fwd_split"]
          .transform(lambda s: s.rolling(LOOKBACK + 1, min_periods=1).max())
          .astype(bool)
    )

    day = px[px["date"] == asof].copy()
    logger.info("symbols trading on %s: %s", str(asof)[:10], f"{len(day):,}")
    out: Dict[str, object] = {"asof": str(asof)[:10], "n_symbols": int(len(day))}

    # ---------------- LONG ------------------------------------------------
    lm_base = (
        (day["adv20"] >= ADV_FLOOR)
        & (~day["any_split"])
        & (day[f"ret_{LOOKBACK}d"] <= -DROP)
        & (day["close_unadj"] >= PRICE_LO)
        & (day["close_unadj"] < PRICE_HI)
    ).fillna(False)
    lm = lm_base & ~day["fwd_split_in_lookback"]
    rejected = day[lm_base & day["fwd_split_in_lookback"]]
    longs = day[lm].sort_values(f"ret_{LOOKBACK}d")

    logger.info("=" * 108)
    logger.info("LONG  -- $%.2f-$%.2f, ADV>=$%s, down >=%.0f%% over %d sessions",
                PRICE_LO, PRICE_HI, f"{ADV_FLOOR:,.0f}", 100 * DROP, LOOKBACK)
    logger.info("        enter at the NEXT open, +%.0f%% limit, time stop after %d sessions",
                100 * TARGET, HOLD)
    logger.info("-" * 108)
    if longs.empty:
        logger.info("  no names met the criteria on %s", str(asof)[:10])
    else:
        logger.info("  %-8s %10s %10s %12s %14s %12s", "symbol", "close",
                    f"{LOOKBACK}d ret", "ADV20", "buy limit @+50%", "volume")
        for _, r in longs.iterrows():
            logger.info("  %-8s %10.2f %9.1f%% %12s %14.2f %12s",
                        r["symbol"], r["close_unadj"], 100 * r[f"ret_{LOOKBACK}d"],
                        f"${r['adv20']:,.0f}", r["close_unadj"] * (1 + TARGET),
                        f"{r['volume']:,.0f}")
    if len(rejected):
        logger.info("-" * 108)
        logger.info("  EXCLUDED as forward splits (the decline is a unit change, not a loss):")
        for _, r in rejected.iterrows():
            logger.info("    %-8s divisor %.2fx", r["symbol"], r["fwd_split_divisor"])
    out["excluded_forward_splits"] = [
        {"symbol": r["symbol"], "divisor": float(r["fwd_split_divisor"])}
        for _, r in rejected.iterrows()]
    out["long"] = [
        {"symbol": r["symbol"], "close": float(r["close_unadj"]),
         "ret_5d": float(r[f"ret_{LOOKBACK}d"]), "adv20": float(r["adv20"]),
         "limit": float(r["close_unadj"] * (1 + TARGET))}
        for _, r in longs.iterrows()]

    # ---------------- SHORT -----------------------------------------------
    sm = ((day["adv20"] >= ADV_FLOOR) & day["is_suspect_split"]).fillna(False)
    shorts = day[sm].sort_values("split_multiple", ascending=False)
    logger.info("-" * 108)
    logger.info("SHORT -- suspected reverse split, ADV>=$%s", f"{ADV_FLOOR:,.0f}")
    logger.info("        short the open, cover the close, +30%% hard stop")
    logger.info("-" * 108)
    if shorts.empty:
        logger.info("  no suspected reverse splits on %s", str(asof)[:10])
    else:
        logger.info("  %-8s %10s %10s %10s %12s", "symbol", "open", "prev close",
                    "ratio", "ADV20")
        for _, r in shorts.iterrows():
            logger.info("  %-8s %10.2f %10.2f %9.1fx %12s", r["symbol"],
                        r["open_unadj"], r["prev_close_unadj"],
                        r["split_multiple"], f"${r['adv20']:,.0f}")
    out["short"] = [
        {"symbol": r["symbol"], "open": float(r["open_unadj"]),
         "prev_close": float(r["prev_close_unadj"]),
         "multiple": float(r["split_multiple"]), "adv20": float(r["adv20"])}
        for _, r in shorts.iterrows()]

    # --- how unusual is today, against the studied period ------------------
    logger.info("-" * 108)
    logger.info("context: the long rule fired 429 times over 18 months (~24/month, "
                "~1.1 per session); the short 201 times (~11/month)")
    logger.info("today: %d long, %d short", len(longs), len(shorts))
    logger.info("=" * 108)

    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
