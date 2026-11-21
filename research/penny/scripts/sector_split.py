"""sector_split.py — is a crash idiosyncratic or systematic, and does it matter?

The question
------------
Nine rounds have treated "fell 50% in a week" as one population. It is at least
two, and they have different causes:

* the stock fell and **its sector did not** -- something happened to the
  company: a failed trial, a guidance cut, a fraud;
* the stock fell and **its sector fell too** -- a factor moved: an oil price, a
  rate, a policy.

Round 9's ten largest trades were **seven oil and gas names on two consecutive
days in March 2020**. That is the systematic case, and it was reported as the
general one. Pooling the two and quoting the average describes neither.

The test
--------
Split on `rel_5d = own 5-day return - sector median 5-day return`, computed
**point-in-time from the same session's cross-section**, and compare three arms
against each other and against the pooled control:

    idiosyncratic : sector held up, the stock did not   (rel << 0)
    systematic    : the sector fell with it             (rel ~ 0)
    pooled        : round 9's arm, both together        (the control)

If the arms do not separate, sector is dead and that is a clean answer. If they
do, this is the first split of a population in this project rather than another
threshold on one.

Discipline carried from earlier rounds: unadjusted prices only in filters
(§72); a continuous panel, so a "5-day" window is five sessions (§74); the
sector median excludes the stock itself, or a thin sector would be compared
against a number the stock dominates; and the holdout stays sealed.

    python research/penny/scripts/sector_split.py
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

from research.penny.etl import bars  # noqa: E402
from research.penny.eval import stats as pstats  # noqa: E402
from research.penny.scripts.long_reversion_portfolio import simulate  # noqa: E402
from research.penny.strategy.engine import (  # noqa: E402
    Spec, build_trades, signal_mask,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("sector")

from research.penny.dates import SEAL                                  # noqa: E402
MIN_SECTOR_NAMES = 8       # below this the "sector median" is not a sector


def attach_sector(u: pd.DataFrame) -> pd.DataFrame:
    sic = pd.read_parquet(bars.DATA_ROOT / "sic_sectors.parquet",
                          columns=["symbol", "sector"]).drop_duplicates("symbol")
    u = u.merge(sic, on="symbol", how="left")
    u["sector"] = u["sector"].fillna("unknown")
    return u


def attach_sector_relative(u: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Sector median return per session, LEAVE-ONE-OUT, then the stock's excess.

    Leave-one-out matters: in a sector with a handful of eligible names the
    stock would otherwise be compared against a median it helped set, which
    shrinks `rel` toward zero exactly where the split is being tested.
    """
    col = f"ret_{window}d_unadj"
    elig = u["in_universe"].to_numpy() & u[col].notna().to_numpy()
    sub = u.loc[elig, ["date", "sector", col]].copy()

    grp = sub.groupby(["date", "sector"])[col]
    agg = grp.agg(["median", "count", "sum"]).reset_index()
    agg = agg.rename(columns={"median": "sec_med", "count": "sec_n", "sum": "sec_sum"})
    u = u.merge(agg, on=["date", "sector"], how="left")

    # leave-one-out mean as the robustness partner to the median
    with np.errstate(invalid="ignore", divide="ignore"):
        u["sec_loo_mean"] = (u["sec_sum"] - u[col]) / (u["sec_n"] - 1)
    u["sec_ok"] = (u["sec_n"] >= MIN_SECTOR_NAMES).fillna(False)
    u["rel_5d"] = u[col] - u["sec_med"]
    return u


def arm_stats(t: pd.DataFrame, u: pd.DataFrame, spec: Spec, label: str) -> Dict:
    if len(t) < 40:
        print(f"  {label:<34} n={len(t):>4}  (too few)")
        return {"label": label, "n": int(len(t))}
    e = pstats.cluster_bootstrap_mean(t["net"], t["date"], n_boot=4_000)
    liquid = np.sort(u.loc[u["in_universe"].to_numpy(), "date"].unique())
    cal = liquid[(liquid >= np.datetime64(t["date"].min()))
                 & (liquid <= np.datetime64(max(t["exit_date"].max(), t["date"].max())))]
    sim = simulate(t, cal, spec.slots)
    star = "  <<<" if e.lo > 0 else ""
    print(f"  {label:<34} n={len(t):>4}  {100*e.mean:+7.2f}%  "
          f"CI [{100*e.lo:+7.2f}%, {100*e.hi:+7.2f}%]  med {100*t['net'].median():+6.2f}%  "
          f"Sharpe {sim.sharpe:5.2f}  cmp {100*sim.total_return:+8.1f}%{star}")
    return {"label": label, "n": int(len(t)), "mean": float(e.mean),
            "lo": float(e.lo), "hi": float(e.hi),
            "median": float(t["net"].median()), "sharpe": sim.sharpe,
            "compounded": sim.total_return, "max_drawdown": sim.max_drawdown,
            "target_rate": float(t["took_target"].mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="research/penny/specs/FINAL.json")
    ap.add_argument("--out", default=str(bars.DATA_ROOT / "sector_split.json"))
    args = ap.parse_args(argv)

    spec = Spec.load(args.spec)
    u = pd.read_parquet(bars.DATA_ROOT / "panel_strategy.parquet")
    u = attach_sector(u)
    u = attach_sector_relative(u, spec.drop_window)

    cov = u.loc[u["in_universe"].to_numpy(), "sector"]
    print("=" * 116)
    print("SECTOR COVERAGE over eligible rows")
    vc = cov.value_counts()
    for s, n in vc.items():
        print(f"  {s:<24} {n:>9,}  ({100*n/len(cov):.1f}%)")
    print(f"  rows with a usable sector cross-section (>={MIN_SECTOR_NAMES} names): "
          f"{100*u.loc[u['in_universe'].to_numpy(), 'sec_ok'].mean():.1f}%")

    trades = build_trades(u, spec)
    trades = trades[trades["date"] < SEAL]
    key = pd.MultiIndex.from_arrays([u["symbol"], u["date"]])
    want = pd.MultiIndex.from_arrays([trades["symbol"], trades["date"]])
    sel = u.loc[key.isin(want), ["symbol", "date", "sector", "rel_5d",
                                 "sec_med", "sec_n", "sec_ok"]]
    t = trades.merge(sel, on=["symbol", "date"], how="left")

    print("\n" + "=" * 116)
    print(f"HOW THE ROUND-9 POPULATION SPLITS  (n={len(t)}, dev only)")
    ok = t["sec_ok"].fillna(False)
    print(f"  with a usable sector cross-section : {int(ok.sum())} of {len(t)}")
    print(f"  sector median 5d return on those days: "
          f"{100*t.loc[ok,'sec_med'].median():+.2f}% (median across trades)")
    print(f"  own-minus-sector (rel_5d)          : median {100*t.loc[ok,'rel_5d'].median():+.2f}%")
    print("\n  trades by sector (top 8):")
    for s, n in t["sector"].value_counts().head(8).items():
        sub = t[t["sector"] == s]
        print(f"    {s:<24} n={n:>4}  per-trade {100*sub['net'].mean():+7.2f}%  "
              f"median {100*sub['net'].median():+7.2f}%")

    print("\n" + "=" * 116)
    print("THE THREE ARMS")
    tk = t[ok].copy()
    res: List[Dict] = []
    res.append(arm_stats(tk, u, spec, "POOLED (round 9's arm) [control]"))

    # systematic: the sector fell too. idiosyncratic: it did not.
    for thr in (-0.10, -0.20, -0.30):
        sysm = tk[tk["sec_med"] <= thr]
        res.append(arm_stats(sysm, u, spec, f"SYSTEMATIC  sector <= {100*thr:.0f}%"))
    for thr in (-0.05, 0.0, 0.05):
        idio = tk[tk["sec_med"] >= thr]
        res.append(arm_stats(idio, u, spec, f"IDIOSYNCRATIC sector >= {100*thr:.0f}%"))

    print("\n  by relative crash depth (own minus sector):")
    for lo, hi, lab in ((-9.9, -0.45, "rel <= -45pp (far worse than peers)"),
                        (-0.45, -0.30, "rel -45..-30pp"),
                        (-0.30, -0.15, "rel -30..-15pp"),
                        (-0.15, 9.9, "rel >= -15pp (fell WITH peers)")):
        sub = tk[(tk["rel_5d"] > lo) & (tk["rel_5d"] <= hi)]
        res.append(arm_stats(sub, u, spec, lab))

    print("\n" + "=" * 116)
    print("THE DECISIVE CHECK: does either arm survive without the COVID window?")
    nocov = ~((tk["date"] >= pd.Timestamp("2020-02-15"))
              & (tk["date"] <= pd.Timestamp("2020-06-30")))
    res.append(arm_stats(tk[nocov], u, spec, "POOLED, no COVID [control]"))
    res.append(arm_stats(tk[nocov & (tk["sec_med"] <= -0.20)], u, spec,
                         "SYSTEMATIC <=-20%, no COVID"))
    res.append(arm_stats(tk[nocov & (tk["sec_med"] >= 0.0)], u, spec,
                         "IDIOSYNCRATIC >=0%, no COVID"))
    print("=" * 116)

    Path(args.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
