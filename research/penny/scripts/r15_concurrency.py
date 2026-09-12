"""r15_concurrency.py — H1, H2 and H4 of PREREGISTRATION_R15.

H1 — the variance model.
    Decompose the traded return within day cells, R_i,t = mu_t + eta_i,t, and
    measure sd(mu_t) and sd(eta) on the UNCAPPED and the VOL-CAPPED population
    separately. §82's decomposition was measured on one of them and §84's sweep
    run on the other, and the two disagree by a factor of 25 about what breadth
    is worth. Predicted portfolio sd is

        sd(N) = sqrt( sd(mu)^2 + sd(eta)^2 / N )

    which is compared against the measured Sharpe ratios at each N.

H2 — sigma_common as a fourth lever.
    The identity N = (H/tau) * K * (1-delta) has three levers and §80 closed two.
    But sd(mu_t) is not a constant of nature: if the filter already moves it,
    moving it further lowers the N needed for a given Sharpe without touching
    the identity at all. Tested as a hard cap on CONCURRENT holdings per sector
    -- a concentration control, not a ranking, so it cannot smuggle in selection.

H4 — weighting.
    Equal weight is near-optimal precisely because §78 found E[R] flat in the
    score. Inverse-volatility is the one alternative that compounds the validated
    low-vol effect rather than betting against it.

    python research/penny/scripts/r15_concurrency.py --period dev
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
from research.penny.scripts.reproduce_round14 import (                 # noqa: E402
    HOLD, TARGET, VETO_FRAC, VOL_CAP, build_frame, market_calendar)
from research.penny.strategy.book import (                             # noqa: E402
    BookConfig, gross_return, run_book)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("r15")

SLOTS = [15, 30, 60, 150]
SEEDS = 15


def attach_sector(df: pd.DataFrame) -> pd.DataFrame:
    sic = pd.read_parquet(bars.DATA_ROOT / "sic_sectors.parquet",
                          columns=["symbol", "sector"]).drop_duplicates("symbol")
    out = df.merge(sic, on="symbol", how="left")
    out["sector"] = out["sector"].fillna("unknown")
    return out


def traded_return(df: pd.DataFrame) -> pd.Series:
    """Gross-of-target minus one flat round trip -- the quantity a slot earns."""
    return pd.Series(gross_return(df, TARGET) - df["cost"].to_numpy(),
                     index=df.index)


# --------------------------------------------------------------------------
# H1 -- the variance decomposition
# --------------------------------------------------------------------------
def decompose(df: pd.DataFrame, label: str) -> Dict[str, float]:
    d = df.dropna(subset=["_R"])
    cell_mean = d.groupby("date")["_R"].transform("mean")
    mu = d.groupby("date")["_R"].mean()
    eta = d["_R"] - cell_mean
    out = {
        "label": label, "n_obs": int(len(d)), "n_cells": int(d["date"].nunique()),
        "mean": float(d["_R"].mean()),
        "sd_mu": float(mu.std(ddof=1)),
        "sd_eta": float(eta.std(ddof=1)),
        "sd_total": float(d["_R"].std(ddof=1)),
    }
    out["ceiling"] = out["mean"] / out["sd_mu"] if out["sd_mu"] > 0 else np.nan
    return out


def predicted_sd(dec: Dict[str, float], n: int) -> float:
    return float(np.sqrt(dec["sd_mu"] ** 2 + dec["sd_eta"] ** 2 / n))


def sweep(df, cal, **kw) -> Dict[str, float]:
    sh, ann, dd = [], [], []
    for s in range(SEEDS):
        r = run_book(df, cal, BookConfig(**kw), seed=s)
        if r is not None:
            sh.append(r.sharpe); ann.append(r.ann); dd.append(r.max_drawdown)
    if not sh:
        return {}
    return {"sharpe": float(np.mean(sh)), "sd": float(np.std(sh, ddof=1)),
            "ann": float(np.mean(ann)), "maxdd": float(np.mean(dd))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="dev", choices=("dev", "holdout"))
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.period == "holdout" and not args.unseal:
        raise SystemExit("refusing to read the sealed holdout without --unseal")

    cal = market_calendar(args.period)
    df = attach_sector(build_frame(args.period))
    df["_R"] = traded_return(df)
    logger.info("frame %s rows | %d cells | %d sectors", f"{len(df):,}",
                df["date"].nunique(), df["sector"].nunique())

    res: Dict[str, object] = {"period": args.period}

    # ---------------- H1 --------------------------------------------------
    print("=" * 104)
    print(f"H1 — THE VARIANCE MODEL ({args.period}).  R = mu_t + eta,  "
          f"target +{100*TARGET:.0f}%, one flat round trip")
    print("=" * 104)
    thr = df.groupby("date")["B7_realised_vol"].transform(
        lambda s: s.quantile(VOL_CAP))
    capped = df[df["B7_realised_vol"] <= thr]
    decs = [decompose(df, "uncapped population"),
            decompose(capped, f"vol-capped at q={VOL_CAP:.2f}")]
    print(f"  {'population':<26}{'n':>9}{'cells':>7}{'mean':>9}{'sd(mu_t)':>10}"
          f"{'sd(eta)':>9}{'sd total':>10}{'ceiling':>9}")
    for d in decs:
        print(f"  {d['label']:<26}{d['n_obs']:>9,}{d['n_cells']:>7,}"
              f"{100*d['mean']:>+8.2f}%{100*d['sd_mu']:>9.1f}%"
              f"{100*d['sd_eta']:>8.1f}%{100*d['sd_total']:>9.1f}%"
              f"{d['ceiling']:>9.2f}")
    res["h1_decompositions"] = decs

    print(f"\n  predicted portfolio sd, and the Sharpe RATIO it implies vs N=15:")
    print(f"  {'N':>5}" + "".join(f"{d['label'][:14]:>22}" for d in decs)
          + f"{'measured':>12}")
    base_pred = [predicted_sd(d, 15) for d in decs]
    meas = {}
    for n in SLOTS:
        row = f"  {n:>5}"
        for d, b in zip(decs, base_pred):
            p = predicted_sd(d, n)
            row += f"{100*p:>13.1f}% x{b/p:>6.2f}"
        m = sweep(df, cal, n_slots=n, hold=HOLD, vol_cap=VOL_CAP,
                  veto_frac=VETO_FRAC, target=TARGET, cost_mode="size_aware")
        meas[n] = m
        row += f"{m.get('sharpe', float('nan')):>12.2f}"
        print(row)
    m15 = meas[15]["sharpe"]
    print(f"  measured Sharpe ratio vs N=15: " +
          "  ".join(f"N={n}: x{meas[n]['sharpe']/m15:.2f}" for n in SLOTS))
    res["h1_measured"] = meas

    # ---------------- H2 --------------------------------------------------
    print("\n" + "=" * 104)
    print("H2 — SIGMA_COMMON AS A FOURTH LEVER: cap concurrent holdings per sector")
    print("=" * 104)
    print(f"  {'N':>5}{'sector cap':>12}{'Sharpe':>9}{'+/-':>7}{'annual':>10}"
          f"{'maxDD':>9}{'vs uncapped':>13}")
    h2: List[Dict] = []
    for n in SLOTS:
        base = sweep(df, cal, n_slots=n, hold=HOLD, vol_cap=VOL_CAP,
                     veto_frac=VETO_FRAC, target=TARGET, cost_mode="size_aware")
        for cap in (0, 1, 2, 3, 5):
            if cap == 0:
                r = base
            else:
                r = sweep(df, cal, n_slots=n, hold=HOLD, vol_cap=VOL_CAP,
                          veto_frac=VETO_FRAC, target=TARGET,
                          cost_mode="size_aware", sector_cap=cap)
            if not r:
                continue
            d = 0.0 if cap == 0 else r["sharpe"] - base["sharpe"]
            print(f"  {n:>5}{('none' if cap == 0 else cap):>12}"
                  f"{r['sharpe']:>9.2f}{r['sd']:>7.2f}{r['ann']:>+9.1f}%"
                  f"{r['maxdd']:>8.1f}%{d:>+13.2f}")
            h2.append({"n_slots": n, "sector_cap": cap, **r, "delta": d})
        print()
    res["h2"] = h2

    # ---------------- H4 --------------------------------------------------
    print("=" * 104)
    print("H4 — WEIGHTING: equal vs inverse-volatility (same gross exposure)")
    print("=" * 104)
    print(f"  {'N':>5}{'weight':>12}{'Sharpe':>9}{'+/-':>7}{'annual':>10}{'maxDD':>9}")
    h4: List[Dict] = []
    for n in SLOTS:
        for w in ("equal", "invvol"):
            r = sweep(df, cal, n_slots=n, hold=HOLD, vol_cap=VOL_CAP,
                      veto_frac=VETO_FRAC, target=TARGET, weight=w,
                      cost_mode="size_aware")
            if not r:
                continue
            print(f"  {n:>5}{w:>12}{r['sharpe']:>9.2f}{r['sd']:>7.2f}"
                  f"{r['ann']:>+9.1f}%{r['maxdd']:>8.1f}%")
            h4.append({"n_slots": n, "weight": w, **r})
    res["h4"] = h4
    print("=" * 104)

    out = args.out or str(bars.DATA_ROOT / f"r15_concurrency_{args.period}.json")
    Path(out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
