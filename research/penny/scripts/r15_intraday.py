"""r15_intraday.py — H3 and H5 of PREREGISTRATION_R15, from the minute tape.

H3 — a selector, scored against the estimand that actually pays.
    §78 established the composite is monotone but LEFT-tailed: it removes losers
    and cannot pick winners, and §80 measured its top decile at −1.18% excess
    against +0.77% for a random feature. Every block since has been scored by a
    Spearman IC on the demeaned mean return, which is a poor detector of a
    right-tail effect.

    With a +100% barrier the portfolio return is dominated by WHICH NAMES TOUCH
    IT — 17.0% of the population does. So each intraday feature is scored twice:

      (a) within-cell IC on the demeaned return  -- the bar every block has faced
      (b) P(touch the barrier | top decile)      -- a precision ladder, the
          estimand a concentrated book actually needs

    A feature that lifts the touch rate from 17% to 25% transforms the economics
    and would barely register as an IC.

H5 — entry timing.
    Entry is the next open: the widest-spread minute of a microcap session.
    Three executable alternatives are re-priced from the same minute bars —
    the open, the 10:00 print, and the 09:30-09:59 VWAP — with the barrier logic
    recomputed against each entry price, since whether a name touches +100%
    depends on what you paid.

    python research/penny/scripts/r15_intraday.py --period dev
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars                                    # noqa: E402
from research.penny.eval import stats as pstats                        # noqa: E402
from research.penny.features.intraday_cs import (                      # noqa: E402
    build_intraday_features, entry_prices)
from research.penny.scripts.reproduce_round14 import (                 # noqa: E402
    HOLD, TARGET, VETO_FRAC, VOL_CAP, build_frame, market_calendar)
from research.penny.strategy.cross_section import (                    # noqa: E402
    MIN_CELL, _paths, candidate_mask, demean_within_cell, score_feature)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("r15i")

M_FEATURES = [
    "M_intraday_vol", "M_vol_term_ratio", "M_open_participation",
    "M_close_participation", "M_time_of_low", "M_time_of_high",
    "M_vwap_position", "M_intraday_drawdown", "M_recovery",
    "M_open_to_close", "M_shares_per_trade", "M_volume_spikiness",
    "M_minutes_traded",
]
PRIOR_TRIALS = 287          # the ledger as of LOG 88


def barrier_return(u: pd.DataFrame, pos: np.ndarray, entry: np.ndarray,
                   hold: int, target: float) -> Dict[str, np.ndarray]:
    """Re-price the trade for an arbitrary entry price.

    Whether a name touches +100% depends on what you paid for it, so an entry
    change cannot be applied as a shift to a precomputed return -- the barrier
    has to be re-evaluated. Mirrors `risk_shaped.attach_target_returns` with the
    entry supplied rather than read from the open.
    """
    p = _paths(u, pos, hold + 1)
    hi, cl, val = p["high_unadj"][:, 1:], p["close_unadj"][:, 1:], p["valid"][:, 1:]
    lastv = np.where(val.any(axis=1),
                     val.shape[1] - 1 - val[:, ::-1].argmax(axis=1), -1)
    rows = np.arange(len(pos))
    with np.errstate(invalid="ignore", divide="ignore"):
        hit = val & (hi >= entry[:, None] * (1.0 + target))
        touched = hit.any(axis=1)
        k = np.where(touched, hit.argmax(axis=1), lastv)
        px = np.where(touched, entry * (1.0 + target),
                      cl[rows, np.maximum(k, 0)])
        gross = px / entry - 1.0
    ok = (np.isfinite(entry) & (entry > 0) & (lastv >= 0)
          & ~p["split"].any(axis=1) & np.isfinite(gross))
    return {"gross": gross, "touched": touched, "ok": ok}


def precision_ladder(df: pd.DataFrame, feature: str, label: str,
                     n_boot: int = 3000, seed: int = 0) -> Dict[str, float]:
    """P(touch | top decile within the day) against the base rate.

    Ranked WITHIN the day, so the day effect cannot masquerade as selection --
    the same reason §78's estimator demeans within the cell.
    """
    d = df.dropna(subset=[feature, label])
    d = d[d.groupby("date")[feature].transform("size") >= MIN_CELL]
    if len(d) < 500:
        return {}
    rank = d.groupby("date")[feature].rank(pct=True)
    out: Dict[str, float] = {"feature": feature, "n": int(len(d)),
                             "base_rate": float(d[label].mean())}
    for tag, m in (("top10", rank >= 0.90), ("top25", rank >= 0.75),
                   ("bot10", rank <= 0.10)):
        sel = d[m]
        if len(sel) < 100:
            continue
        e = pstats.cluster_bootstrap_mean(sel[label].astype(float), sel["date"],
                                          n_boot=n_boot, seed=seed)
        out[f"{tag}_rate"] = float(e.mean)
        out[f"{tag}_lo"] = float(e.lo)
        out[f"{tag}_hi"] = float(e.hi)
        out[f"{tag}_lift"] = float(e.mean / out["base_rate"]) if out["base_rate"] else np.nan
        # one-sided: does the top decile beat the base rate?
        out[f"{tag}_beats_base"] = bool(e.lo > out["base_rate"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.period == "holdout" and not args.unseal:
        raise SystemExit("refusing to read the sealed holdout without --unseal")

    cal = market_calendar(args.period)
    df = build_frame(args.period)
    logger.info("frame %s rows | %d cells", f"{len(df):,}", df["date"].nunique())

    # ---- intraday features -----------------------------------------------
    feat_path = bars.DATA_ROOT / f"intraday_features_{args.period}.parquet"
    if feat_path.exists():
        mf = pd.read_parquet(feat_path)
        logger.info("intraday features: %s rows from cache", f"{len(mf):,}")
    else:
        mf = build_intraday_features(df[["symbol", "date"]])
        mf.to_parquet(feat_path, index=False)
    df = df.merge(mf, on=["symbol", "date"], how="left")
    cov = df[M_FEATURES].notna().all(axis=1).mean()
    logger.info("minute coverage of the frame: %.1f%%", 100 * cov)

    # the barrier label, from the already-attached target column
    u = pd.read_parquet(bars.DATA_ROOT / "panel_strategy.parquet")
    m = candidate_mask(u, drop_pct=0.15)
    pos_all = np.flatnonzero(m)
    key = pd.MultiIndex.from_arrays([u["symbol"].to_numpy()[pos_all],
                                     u["date"].to_numpy()[pos_all]])
    want = pd.MultiIndex.from_arrays([df["symbol"], df["date"]])
    sel = key.isin(want)
    pos = pos_all[sel]
    open_entry = _paths(u, pos, 2)["open_unadj"][:, 0]

    res: Dict[str, object] = {"period": args.period, "coverage": float(cov)}

    # ---------------- H5 --------------------------------------------------
    print("=" * 104)
    print(f"H5 — ENTRY TIMING ({args.period}).  Barrier re-evaluated against each "
          f"entry price, target +{100*TARGET:.0f}%, H={HOLD}")
    print("=" * 104)
    ep_path = bars.DATA_ROOT / f"entry_prices_{args.period}.parquet"
    if ep_path.exists():
        ep = pd.read_parquet(ep_path)
    else:
        ep = entry_prices(df[["symbol", "date"]], cal)
        ep.to_parquet(ep_path, index=False)
    base = pd.DataFrame({"symbol": u["symbol"].to_numpy()[pos],
                         "date": u["date"].to_numpy()[pos],
                         "daily_open": open_entry})
    base = base.merge(ep, on=["symbol", "date"], how="left")

    # COMMON SUBSAMPLE. Minute coverage differs by entry rule -- the 10:00
    # print exists for fewer symbol-days than the open does -- so scoring each
    # rule on the rows where it happens to be priced compares four different
    # populations, not four entry rules. Every rule is evaluated on the rows
    # where ALL of them are priced.
    # SCALE-INVARIANT ENTRY PRICES. The minute cache is adjusted and the daily
    # panel is not, so a minute LEVEL cannot be used as an entry price against
    # an unadjusted forward path -- the two disagree by the pending corporate
    # action. What IS comparable is the within-session RATIO: how far the 10:00
    # print sits from the open is unaffected by a constant factor. So each
    # alternative entry is the UNADJUSTED daily open scaled by that ratio.
    base["E_t1000"] = base["daily_open"] * (base["E_t1000"] / base["E_open"])
    base["E_vwap30"] = base["daily_open"] * (base["E_vwap30"] / base["E_open"])
    base["E_open"] = base["daily_open"] * (base["E_open"] / base["E_open"])
    cols = ["daily_open", "E_open", "E_t1000", "E_vwap30"]
    common = np.ones(len(base), dtype=bool)
    for c in cols:
        v = base[c].to_numpy(dtype=float)
        common &= np.isfinite(v) & (v > 0)
    print(f"  common subsample: {common.sum():,} of {len(base):,} rows "
          f"({100*common.mean():.0f}%) priced by every entry rule")
    print(f"  {'entry':<18}{'n':>8}{'touch %':>9}{'mean gross':>12}"
          f"{'median':>10}{'sd':>9}{'IR/pos':>9}{'vs open':>9}")
    h5: List[Dict] = []
    ref = None
    for name, col in (("daily open", "daily_open"), ("minute open", "E_open"),
                      ("10:00 print", "E_t1000"), ("09:30-10:00 VWAP", "E_vwap30")):
        e = base[col].to_numpy(dtype=float)
        r = barrier_return(u, pos, e, HOLD, TARGET)
        ok = r["ok"] & common
        g = r["gross"][ok]
        if len(g) < 500:
            print(f"  {name:<14} too few priced ({int(ok.sum())})")
            continue
        row = {"entry": name, "n": int(ok.sum()),
               "coverage": float(ok.mean()),
               "touch_rate": float(r["touched"][ok].mean()),
               "mean": float(np.mean(g)), "median": float(np.median(g)),
               "sd": float(np.std(g, ddof=1))}
        row["ir"] = row["mean"] / row["sd"] if row["sd"] > 0 else np.nan
        ref = row["ir"] if ref is None else ref
        print(f"  {name:<18}{row['n']:>8,}"
              f"{100*row['touch_rate']:>8.1f}%{100*row['mean']:>+11.2f}%"
              f"{100*row['median']:>+9.2f}%{100*row['sd']:>8.1f}%{row['ir']:>9.3f}"
              f"{row['ir'] - ref:>+9.3f}")
        h5.append(row)
    res["h5"] = h5

    # ---------------- H3 --------------------------------------------------
    print("\n" + "=" * 104)
    print(f"H3 — INTRADAY FEATURES AS A SELECTOR ({args.period})")
    print("  (a) within-cell IC on the demeaned return — the bar every block has faced")
    print("=" * 104)
    # MERGE on (symbol, date), never assign positionally: `pos` is in panel
    # order while `df` came out of `frame()` after its own `ok` and cell-size
    # filters, so the two are neither the same length nor the same order. A
    # positional assign here would silently pair each row with another row's
    # forward return -- the same class of error as SS57 and SS72.
    r_open = barrier_return(u, pos, open_entry, HOLD, TARGET)
    lab = pd.DataFrame({
        "symbol": u["symbol"].to_numpy()[pos],
        "date": u["date"].to_numpy()[pos],
        "_gross": r_open["gross"],
        "_touched": r_open["touched"].astype(float),
        "_ok": r_open["ok"],
    })
    n_before = len(df)
    df = df.drop(columns=[c for c in ("_gross", "_touched", "_ok")
                          if c in df.columns])
    df = df.merge(lab, on=["symbol", "date"], how="inner", validate="one_to_one")
    logger.info("barrier labels merged: %s of %s frame rows",
                f"{len(df):,}", f"{n_before:,}")
    df = df[df["_ok"]].copy()
    df["R"] = df["_gross"] - df["cost"]
    df["Rt"] = demean_within_cell(df, "R", "date")

    ics: List[Dict] = []
    print(f"  {'feature':<24}{'cells':>7}{'IC':>9}{'t':>8}{'95% CI':>20}"
          f"{'L/S net':>10}")
    for f in M_FEATURES + ["B0_random"]:
        if f not in df.columns:
            continue
        r = score_feature(df, f, n_boot=2000)
        if r is None:
            print(f"  {f:<24} too few cells")
            continue
        star = "  <<<" if (r.lo > 0 or r.hi < 0) else ""
        print(f"  {f:<24}{r.n_cells:>7,}{r.ic:>+9.4f}{r.t_stat:>8.2f}"
              f"  [{r.lo:>+7.4f},{r.hi:>+7.4f}]{100*r.ls_net:>+9.2f}%{star}")
        ics.append({"feature": f, "n_cells": r.n_cells, "ic": r.ic,
                    "t": r.t_stat, "lo": r.lo, "hi": r.hi, "ls_net": r.ls_net})
    res["h3_ic"] = ics

    print("\n  (b) P(touch the +100% barrier | top decile within the day)")
    print(f"  base rate over the population: {100*df['_touched'].mean():.1f}%")
    print(f"  {'feature':<24}{'base':>7}{'top10%':>9}{'lift':>7}"
          f"{'95% CI':>20}{'bot10%':>9}{'beats base':>12}")
    lad: List[Dict] = []
    for f in M_FEATURES + ["B0_random"]:
        if f not in df.columns:
            continue
        p = precision_ladder(df, f, "_touched")
        if not p or "top10_rate" not in p:
            continue
        flag = "  <<< YES" if p.get("top10_beats_base") else ""
        print(f"  {f:<24}{100*p['base_rate']:>6.1f}%{100*p['top10_rate']:>8.1f}%"
              f"{p['top10_lift']:>7.2f}  [{100*p['top10_lo']:>5.1f}%,"
              f"{100*p['top10_hi']:>5.1f}%]{100*p.get('bot10_rate', np.nan):>8.1f}%"
              f"{flag:>12}")
        lad.append(p)
    res["h3_precision"] = lad

    # ---- BH across everything tested here, against the standing ledger ----
    from scipy import stats as sps
    pv = []
    for r in ics:
        # the bootstrap gives an interval, not a p-value; recover a two-sided
        # normal p from the interval half-width, which is what BH needs
        se = max((r["hi"] - r["lo"]) / 3.92, 1e-12)
        pv.append(float(2.0 * sps.norm.sf(abs(r["ic"]) / se)))
    n_new = len(ics) + len(lad)
    padded = np.concatenate([np.array(pv), np.ones(PRIOR_TRIALS)])
    rej = pstats.benjamini_hochberg(padded, q=0.05)
    print("\n" + "-" * 104)
    print(f"  BH-FDR over {len(pv)} IC tests padded with {PRIOR_TRIALS} prior "
          f"trials: {int(rej[:len(pv)].sum())} survive")
    print(f"  new trials added to the ledger this run: {n_new} "
          f"-> cumulative {PRIOR_TRIALS + n_new}")
    print("=" * 104)
    res["h3_survivors"] = int(rej[:len(pv)].sum())
    res["cumulative_trials"] = PRIOR_TRIALS + n_new

    out = args.out or str(bars.DATA_ROOT / f"r15_intraday_{args.period}.json")
    Path(out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
