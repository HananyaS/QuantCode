"""long_walkforward.py — refit the parameters every year, trade the next one.

The problem this fixes
----------------------
§65's headline (+4.97%/trade, Sharpe 3.06, +109% compounded) came from choosing
the best of 48 portfolio cells **on all the data**, then reporting that cell's
performance on the same data with an in/out split bolted on afterwards. The
parameters -- drop >=40%, hold 10, target +50%, 30 slots -- were selected with
the answer visible.

The honest version fits on an expanding window and trades the year that follows,
never touching the future:

    fit on everything through 2017  ->  trade 2018
    fit on everything through 2018  ->  trade 2019
    ...                                 ...

and concatenates the traded years into one out-of-sample equity curve. **A
strategy whose parameters must be re-chosen each year is a fitting artifact**,
and this is the number that decides which one this is.

Why the forward paths are recomputed here
------------------------------------------
`long_reversion_portfolio.build_trades` gets forward bars by concatenating
`hold` groupby-shifts across the whole panel. On 18 months that is fine; on
eleven years (~5M rows) it allocates roughly 3 GB per call and there are dozens
of calls. `forward_paths` below does the same work positionally -- gathering
only the rows that are actually candidate signals -- which is O(candidates x
hold) instead of O(panel x hold).

    python research/penny/scripts/long_walkforward.py
    python research/penny/scripts/long_walkforward.py --first-trade-year 2018
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.long_reversion_portfolio import simulate  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("wf")

PRICE_LO, PRICE_HI = 0.50, 2.00
LOOKBACK = 5
ADV_FLOOR = 2.5e5
MIN_FIT_TRADES = 100

DROPS = [0.30, 0.35, 0.40, 0.45, 0.50]
HOLDS = [5, 10, 15]
TARGETS = [0.20, 0.30, 0.50]
SLOTS = [10, 20, 30, 50]

COST_BPS = [(0, 1e5, 400.0), (1e5, 2.5e5, 250.0), (2.5e5, 1e6, 150.0),
            (1e6, 5e6, 90.0), (5e6, 2e7, 65.0), (2e7, np.inf, 45.0)]


def cost_for(adv: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=adv.index)
    for lo, hi, bps in COST_BPS:
        out = out.mask((adv >= lo) & (adv < hi), bps / 1e4)
    return out.fillna(400.0 / 1e4)


def load_panel() -> pd.DataFrame:
    """Universe plus the trailing state the signal needs. Deliberately lean --
    only what the rule reads, so eleven years stays in memory."""
    u = pd.read_parquet(
        bars.DATA_ROOT / "forensic_universe.parquet",
        columns=["symbol", "date", "open", "high", "low", "close",
                 "close_unadj", "adv20", "any_split", "is_suspect_fwd_split"],
    ).sort_values(["symbol", "date"]).reset_index(drop=True)
    g = u.groupby("symbol", sort=False)
    u[f"ret_{LOOKBACK}d"] = u["close"] / g["close"].shift(LOOKBACK) - 1.0
    u["cost"] = cost_for(u["adv20"])
    logger.info("panel: %s rows | %s symbols | %s .. %s", f"{len(u):,}",
                f"{u['symbol'].nunique():,}", str(u["date"].min())[:10],
                str(u["date"].max())[:10])
    return u


def forward_paths(u: pd.DataFrame, pos: np.ndarray, hold: int) -> Dict[str, np.ndarray]:
    """Bars 1..hold ahead of each position, respecting symbol boundaries.

    Positional rather than groupby-shift: a shift-based version materialises
    `hold` full-length columns for five fields across the entire panel, which
    on eleven years is gigabytes for a few thousand candidates.
    """
    codes = pd.factorize(u["symbol"], sort=False)[0]
    n = len(u)
    hi = u["high"].to_numpy(); lo = u["low"].to_numpy()
    cl = u["close"].to_numpy(); op = u["open"].to_numpy()
    dt = u["date"].to_numpy()
    sp = u["any_split"].to_numpy(bool) | u["is_suspect_fwd_split"].to_numpy(bool)

    m = len(pos)
    out = {k: np.full((m, hold), np.nan) for k in ("high", "low", "close", "open")}
    out["date"] = np.full((m, hold), np.datetime64("NaT"), dtype="datetime64[ns]")
    out["split"] = np.zeros((m, hold), dtype=bool)
    out["valid"] = np.zeros((m, hold), dtype=bool)

    for k in range(1, hold + 1):
        nxt = pos + k
        ok = (nxt < n)
        safe = np.where(ok, nxt, 0)
        ok &= codes[safe] == codes[pos]
        j = k - 1
        out["valid"][:, j] = ok
        out["high"][ok, j] = hi[safe[ok]]
        out["low"][ok, j] = lo[safe[ok]]
        out["close"][ok, j] = cl[safe[ok]]
        out["open"][ok, j] = op[safe[ok]]
        out["date"][ok, j] = dt[safe[ok]]
        out["split"][ok, j] = sp[safe[ok]]
    return out


def build_trades(u: pd.DataFrame, drop: float, hold: int, target: float) -> pd.DataFrame:
    """Candidate entries for one parameter set, entered at the NEXT open.

    Non-overlapping (one open position per symbol) and split-free across the
    whole holding window, exactly as §65 requires. A position whose symbol stops
    printing mid-hold exits at its last available close rather than being
    dropped -- a delisting is not missing data.
    """
    sig = (
        (u["adv20"] >= ADV_FLOOR).to_numpy()
        & (~u["any_split"]).to_numpy()
        & (~u["is_suspect_fwd_split"]).to_numpy()
        & (u[f"ret_{LOOKBACK}d"] <= -drop).fillna(False).to_numpy()
        & (u["close_unadj"] >= PRICE_LO).to_numpy()
        & (u["close_unadj"] < PRICE_HI).to_numpy()
    )
    pos = np.flatnonzero(sig)
    if len(pos) == 0:
        return pd.DataFrame()

    # entry is the next session's open, so the barrier window is bars 2..hold+1
    p = forward_paths(u, pos, hold + 1)
    entry = p["open"][:, 0]
    entry_ok = np.isfinite(entry) & (entry > 0)

    hi = p["high"][:, 1:]; lo_ = p["low"][:, 1:]
    cl = p["close"][:, 1:]; dts = p["date"][:, 1:]
    val = p["valid"][:, 1:]
    split_in_window = p["split"].any(axis=1)

    with np.errstate(invalid="ignore"):
        hit = val & (hi >= entry[:, None] * (1.0 + target))
    any_hit = hit.any(axis=1)
    first_hit = np.where(any_hit, hit.argmax(axis=1), 0)

    last_valid = np.where(val.any(axis=1), val.shape[1] - 1 - val[:, ::-1].argmax(axis=1), -1)
    rows = np.arange(len(pos))
    k = np.where(any_hit, first_hit, last_valid)

    gross = np.where(any_hit, target, cl[rows, np.maximum(k, 0)] / entry - 1.0)
    exit_date = dts[rows, np.maximum(k, 0)]

    keep = entry_ok & (last_valid >= 0) & ~split_in_window & np.isfinite(gross)
    t = pd.DataFrame({
        "symbol": u["symbol"].to_numpy()[pos][keep],
        "date": u["date"].to_numpy()[pos][keep],
        "exit_date": exit_date[keep],
        "net": gross[keep] - u["cost"].to_numpy()[pos][keep],
        "took_target": any_hit[keep],
        "adv20": u["adv20"].to_numpy()[pos][keep],
    }).sort_values(["date", "symbol"])

    # one open position per symbol at a time
    busy: Dict[str, np.datetime64] = {}
    idx = []
    for i, sym, d, xd in zip(t.index, t["symbol"], t["date"], t["exit_date"]):
        if sym in busy and d <= busy[sym]:
            continue
        busy[sym] = xd
        idx.append(i)
    return t.loc[idx]


@dataclass
class Cell:
    drop: float
    hold: int
    target: float
    slots: int


def calendar_from(u: pd.DataFrame, trades: pd.DataFrame) -> np.ndarray:
    """Every liquid session spanned by these trades.

    The calendar must cover the whole window, not just days that traded --
    §49's lesson is that dropping flat days reports the strategy's return only
    on days it chose to act.
    """
    liquid = u.loc[(u["adv20"] >= ADV_FLOOR).to_numpy(), "date"].unique()
    liquid = np.sort(liquid)
    lo = trades["date"].min()
    hi = max(trades["exit_date"].max(), trades["date"].max())
    return liquid[(liquid >= np.datetime64(lo)) & (liquid <= np.datetime64(hi))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-trade-year", type=int, default=None,
                    help="Default: the third year present, so the first fit has depth.")
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "long_walkforward.json"))
    args = ap.parse_args(argv)

    u = load_panel()
    years = sorted(pd.DatetimeIndex(u["date"]).year.unique())
    first_trade = args.first_trade_year or years[2]
    logger.info("years present: %s | first traded year: %d", years, first_trade)

    # ---- frozen §65 parameters, first as a reimplementation check ----------
    # This module recomputes forward paths positionally rather than by
    # groupby-shift, so before trusting any new number it has to reproduce the
    # old one on the old span: §65 reported n=421, +5.56% per trade over
    # 2025-02..2026-08. A material difference means the rewrite is wrong, not
    # that the strategy changed.
    frozen = Cell(0.40, 10, 0.50, 30)
    ft = build_trades(u, frozen.drop, frozen.hold, frozen.target)
    logger.info("=" * 118)
    logger.info("FROZEN §65 PARAMETERS (drop>=40%%, hold 10, target +50%%, 30 slots)")
    for lab, sub in (("2025-02..2026-08 (§65 span, expect n~421, +5.56%)",
                      ft[ft["date"] >= pd.Timestamp("2025-02-01")]),
                     ("FULL SPAN", ft)):
        if len(sub) < 30:
            continue
        e0 = pstats.cluster_bootstrap_mean(sub["net"], sub["date"], n_boot=3_000)
        cal0 = calendar_from(u, sub)
        sm0 = simulate(sub, cal0, frozen.slots)
        logger.info("  %-52s n=%-5d %+6.2f%%  CI [%+.2f%%, %+.2f%%]  median %+6.2f%%  "
                    "target %2.0f%%  Sharpe %5.2f  compounded %+8.1f%%  maxDD %6.1f%%",
                    lab, len(sub), 100 * e0.mean, 100 * e0.lo, 100 * e0.hi,
                    100 * sub["net"].median(), 100 * sub["took_target"].mean(),
                    sm0.sharpe, 100 * sm0.total_return, 100 * sm0.max_drawdown)
    logger.info("  per traded year, frozen parameters:")
    fy = ft.assign(y=pd.DatetimeIndex(ft["date"]).year).groupby("y")
    for y, grp in fy:
        cal = calendar_from(u, grp)
        sm = simulate(grp, cal, frozen.slots)
        logger.info("    %d  n=%-4d per-trade %+6.2f%%  compounded %+8.1f%%  "
                    "maxDD %6.1f%%  target %2.0f%%",
                    y, len(grp), 100 * grp["net"].mean(), 100 * sm.total_return,
                    100 * sm.max_drawdown, 100 * grp["took_target"].mean())

    # Trades depend only on (drop, hold, target) -- build once, slice by year.
    logger.info("building trade sets for %d parameter combinations ...",
                len(DROPS) * len(HOLDS) * len(TARGETS))
    trade_sets: Dict[Tuple[float, int, float], pd.DataFrame] = {}
    for hold in HOLDS:
        for target in TARGETS:
            for drop in DROPS:
                t = build_trades(u, drop, hold, target)
                trade_sets[(drop, hold, target)] = t
        logger.info("  hold=%d done", hold)
    total = sum(len(t) for t in trade_sets.values())
    logger.info("built %d trade sets, %s rows total", len(trade_sets), f"{total:,}")

    calendar_all = np.sort(u.loc[(u["adv20"] >= ADV_FLOOR).to_numpy(), "date"].unique())
    results: List[Dict] = []
    oos_frames: List[pd.DataFrame] = []

    for ty in [y for y in years if y >= first_trade]:
        fit_end = pd.Timestamp(f"{ty-1}-12-31")
        best, best_sharpe = None, -np.inf
        for (drop, hold, target), t in trade_sets.items():
            if t.empty:
                continue
            fit = t[t["date"] <= fit_end]
            if len(fit) < MIN_FIT_TRADES:
                continue
            cal = calendar_all[calendar_all <= np.datetime64(fit_end)]
            for slots in SLOTS:
                sm = simulate(fit, cal, slots)
                if sm.total_return <= 0:
                    continue
                if sm.sharpe > best_sharpe:
                    best_sharpe, best = sm.sharpe, Cell(drop, hold, target, slots)
        if best is None:
            logger.info("  %d: no cell with a positive fit -- sitting out", ty)
            results.append({"year": ty, "chosen": None})
            continue

        t = trade_sets[(best.drop, best.hold, best.target)]
        oos = t[(t["date"] >= pd.Timestamp(f"{ty}-01-01"))
                & (t["date"] <= pd.Timestamp(f"{ty}-12-31"))]
        cal = calendar_all[(calendar_all >= np.datetime64(f"{ty}-01-01"))
                           & (calendar_all <= np.datetime64(f"{ty}-12-31"))]
        if len(oos) == 0 or len(cal) == 0:
            results.append({"year": ty, "chosen": None})
            continue
        sm = simulate(oos, cal, best.slots)
        logger.info("  %d: fit chose drop>=%.0f%% hold=%-2d target=+%.0f%% slots=%-2d "
                    "(fit Sharpe %5.2f) -> OOS n=%-4d per-trade %+6.2f%%  "
                    "compounded %+7.1f%%  maxDD %6.1f%%",
                    ty, 100 * best.drop, best.hold, 100 * best.target, best.slots,
                    best_sharpe, len(oos), 100 * oos["net"].mean(),
                    100 * sm.total_return, 100 * sm.max_drawdown)
        results.append({"year": ty, "chosen": best.__dict__, "fit_sharpe": best_sharpe,
                        "n": len(oos), "per_trade": float(oos["net"].mean()),
                        "total_return": sm.total_return, "sharpe": sm.sharpe,
                        "max_drawdown": sm.max_drawdown,
                        "target_rate": float(oos["took_target"].mean())})
        oos_frames.append(oos.assign(trade_year=ty))

    logger.info("=" * 118)
    if not oos_frames:
        logger.info("NO out-of-sample trades produced.")
        return 0

    allo = pd.concat(oos_frames, ignore_index=True)
    e = pstats.cluster_bootstrap_mean(allo["net"], allo["date"], n_boot=5_000)
    logger.info("CONCATENATED OUT-OF-SAMPLE (parameters never chosen with the answer visible)")
    logger.info("  n=%s over %d years and %d days", f"{len(allo):,}",
                allo["trade_year"].nunique(), e.n_days)
    logger.info("  per trade %+6.2f%%  CI [%+.2f%%, %+.2f%%]  median %+6.2f%%  target hit %.0f%%",
                100 * e.mean, 100 * e.lo, 100 * e.hi,
                100 * allo["net"].median(), 100 * allo["took_target"].mean())
    verdict = "POSITIVE, CI excludes zero" if e.lo > 0 else "NOT DISTINGUISHABLE FROM ZERO"
    logger.info("  >>> %s <<<", verdict)

    rs = allo["net"].sort_values()
    logger.info("  concentration: drop best 10 %+.2f%%   drop best 50 %+.2f%%",
                100 * rs.iloc[:-10].mean(), 100 * rs.iloc[:-50].mean())

    logger.info("-" * 118)
    logger.info("PER-YEAR OUT-OF-SAMPLE")
    for r in results:
        if r.get("chosen") is None:
            logger.info("  %d  (sat out)", r["year"])
            continue
        logger.info("  %d  n=%-4d per-trade %+6.2f%%  compounded %+7.1f%%  "
                    "Sharpe %5.2f  maxDD %6.1f%%  target %2.0f%%",
                    r["year"], r["n"], 100 * r["per_trade"], 100 * r["total_return"],
                    r["sharpe"], 100 * r["max_drawdown"], 100 * r["target_rate"])

    pos_years = sum(1 for r in results if r.get("total_return", 0) > 0)
    tot_years = sum(1 for r in results if r.get("chosen") is not None)
    logger.info("  %d of %d traded years profitable", pos_years, tot_years)

    logger.info("=" * 118)
    Path(args.out).write_text(json.dumps(
        {"results": results, "oos": {"n": len(allo), "mean": float(e.mean),
                                     "lo": float(e.lo), "hi": float(e.hi),
                                     "median": float(allo["net"].median())},
         "years_profitable": pos_years, "years_traded": tot_years},
        indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
