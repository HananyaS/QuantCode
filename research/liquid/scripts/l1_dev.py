"""l1_dev.py -- PREREGISTRATION_L1: the three long-only cells on dev, against SPY/QQQ.

    python research/liquid/scripts/l1_dev.py

Writes data/research_liquid/l1_dev.json and l1_dev_daily.parquet.
Reads nothing after 2022-12-30.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.liquid import cells, costs                                    # noqa: E402
from research.liquid.calendar import DEV_END, DEV_START, assert_through     # noqa: E402
from research.liquid.data import CACHE_DIR, load                            # noqa: E402
from research.penny.eval import benchmark as bm                              # noqa: E402
from research.penny.eval.stats import deflated_sharpe_ratio                  # noqa: E402
from research.penny.scripts.fetch_benchmarks import load as load_bench       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("l1")
N_TRIALS = 374
CMP = dict(mean_block=20, n_boot=2000, seed=17, alpha_lags=20, flat_ok=True)
PAPER_COST = {"C1": costs.BASE, "C2": costs.PAPER_ORB, "C3": costs.PAPER_BTM}


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for r, k in enumerate(keys):
        run = max(run, min(1.0, (m - r) * p[k]))
        out[k] = run
    return out


def trade_stats(tr: list[cells.Trade], n_days: int) -> dict:
    if not tr:
        return {"n_trades": 0}
    f = cells.trade_frame(tr, costs.BASE)
    return {"n_trades": len(f), "trades_per_day": len(f) / n_days,
            "days_in_market": int(f["date"].nunique()), "share_days_in_market": f["date"].nunique() / n_days,
            "win_rate_net": float((f["net"] > 0).mean()),
            "exit_mix": f["kind"].value_counts(normalize=True).round(4).to_dict(),
            "mean_weight": float(f["weight"].mean()), "max_weight": float(f["weight"].max()),
            "share_at_cap": float((f["weight"] >= cells.LEVER_CAP - 1e-12).mean()),
            "gross_bps_per_trade": float(f["gross"].mean() * 1e4),
            "net_bps_per_trade": float(f["net"].mean() * 1e4),
            "cost_bps_per_trade": float((f["gross"] - f["net"]).mean() * 1e4)}


def by_year(r: pd.Series) -> dict:
    return {str(y): float(bm.sharpe(g.to_numpy())) for y, g in r.groupby(r.index.year)}


def main() -> int:
    spy = load("SPY", DEV_START, DEV_END)
    qqq = load("QQQ", DEV_START, DEV_END)
    dates = spy.dates
    assert (dates == qqq.dates).all()
    assert_through(dates, DEV_END, "dev sessions")
    bench, rf = load_bench(DEV_START, DEV_END)
    bench = bench.reindex(dates)
    assert bench.iloc[1:].notna().all().all(), "benchmark misses a dev session"

    run = {"C1": lambda **k: cells.c1(spy, **k), "C2": lambda **k: cells.c2(qqq, **k),
           "C3": lambda **k: cells.c3(spy, **k)}
    out: dict = {"window": [str(dates[0].date()), str(dates[-1].date())], "n_sessions": len(dates),
                 "n_trials": N_TRIALS, "cells": {}, "diagnostics": {}}
    daily = {}
    for name, fn in run.items():
        tr = fn()
        net = cells.daily(tr, dates, costs.BASE)
        daily[name] = net
        cmp = bm.compare(net.to_frame("0"), bench, rf, **CMP)
        t = cmp.tests
        out["cells"][name] = {
            "comparison": cmp.to_dict(),
            "sharpe_net": float(cmp.table.loc["strategy", "sharpe"]),
            "sharpe_gross": float(bm.sharpe(cells.daily(tr, dates, None).to_numpy()[1:])),
            "sharpe_double_cost": float(bm.sharpe(cells.daily(tr, dates, costs.DOUBLE).to_numpy()[1:])),
            "alpha_p_one": float(sps.norm.sf(t["alpha_vs_SPY"]["alpha_t"])),
            "dsr": float(deflated_sharpe_ratio(net.to_numpy()[1:], N_TRIALS)),
            "trades": trade_stats(tr, len(dates)),
            "by_year": by_year(net.iloc[1:]),
        }
        logger.info("%s: %d trades, Sharpe net %.2f", name, len(tr), out["cells"][name]["sharpe_net"])

    tab ={r["index"]: r for r in out["cells"]["C1"]["comparison"]["table"]}
    sh_spy, sh_qqq = float(tab["SPY"]["sharpe"]), float(tab["QQQ"]["sharpe"])
    p_adj = holm({k: v["alpha_p_one"] for k, v in out["cells"].items()})
    for k, v in out["cells"].items():
        v["alpha_p_holm"] = p_adj[k]
        v["rule"] = {"a_sharpe_above_spy_qqq": v["sharpe_net"] > sh_spy and v["sharpe_net"] > sh_qqq,
                     "b_alpha_holm_lt_05": p_adj[k] < 0.05,
                     "c_double_cost_above_spy": v["sharpe_double_cost"] > sh_spy}
        v["dev_pass"] = all(v["rule"].values())

    # ---- diagnostics (labelled, never selectable) -----------------------------
    dg = out["diagnostics"]
    dg["D1_long_short_paper_costs"] = {}
    for name, fn in run.items():
        tr = fn(long_short=True)
        r = cells.daily(tr, dates, PAPER_COST[name]).to_numpy()[1:]
        dg["D1_long_short_paper_costs"][name] = {
            "sharpe": float(bm.sharpe(r)), "ann_return": float(np.mean(r) * 252),
            "cagr": float(bm.cagr(r)), "max_dd": float(bm.max_drawdown(r)), "n_trades": len(tr),
            "long_share": float(np.mean([t.side == 1 for t in tr])) if tr else np.nan}
    dg["D3_unlevered"] = {
        "C2": float(bm.sharpe(cells.daily(cells.c2(qqq, cap=1.0), dates, costs.BASE).to_numpy()[1:])),
        "C3": float(bm.sharpe(cells.daily(cells.c3(spy, cap=1.0), dates, costs.BASE).to_numpy()[1:]))}
    tr4 = cells.c3(spy, resting_stop=False)
    dg["D4_C3_checktime_stops"] = {"sharpe": float(bm.sharpe(cells.daily(tr4, dates, costs.BASE).to_numpy()[1:])),
                                   "trades": trade_stats(tr4, len(dates))}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "l1_dev.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(daily).to_parquet(CACHE_DIR / "l1_dev_daily.parquet")

    # ---- table -----------------------------------------------------------------
    w = 104
    print("=" * w)
    print(f"L1 -- long-only published intraday rules, dev {out['window'][0]}..{out['window'][1]} "
          f"({len(dates)} sessions), N trials {N_TRIALS}")
    print("=" * w)
    print(f"  {'series':10s}{'Sharpe':>8s}{'gross':>8s}{'2x cost':>9s}{'CAGR':>8s}{'maxDD':>8s}"
          f"{'alpha/yr':>10s}{'p Holm':>8s}{'trades':>8s}{'net bp':>8s}  pass")
    for k, v in out["cells"].items():
        st = v["comparison"]["table"][0]
        a = v["comparison"]["tests"]["alpha_vs_SPY"]
        print(f"  {k:10s}{v['sharpe_net']:8.2f}{v['sharpe_gross']:8.2f}{v['sharpe_double_cost']:9.2f}"
              f"{st['cagr']:8.1%}{st['max_dd']:8.1%}{a['alpha_ann']:10.1%}{v['alpha_p_holm']:8.3f}"
              f"{v['trades']['n_trades']:8d}{v['trades'].get('net_bps_per_trade', np.nan):8.1f}  "
              f"{'PASS' if v['dev_pass'] else 'no'}")
    for b in ("SPY", "QQQ", "IWM"):
        print(f"  {b:10s}{tab[b]['sharpe']:8.2f}{'':17s}{tab[b]['cagr']:8.1%}{tab[b]['max_dd']:8.1%}")
    print("-" * w)
    for k, v in out["cells"].items():
        t = v["comparison"]["tests"]
        print(f"  {k}: Sharpe CI {np.round(t['strategy_sharpe_ci'], 2).tolist()} | "
              f"vs SPY {t['sharpe_minus_SPY']['point']:+.2f} CI {np.round(t['sharpe_minus_SPY']['boot_ci'], 2).tolist()}"
              f" | vs QQQ {t['sharpe_minus_QQQ']['point']:+.2f} | alpha NW CI "
              f"[{t['alpha_vs_SPY']['alpha_lo']:+.1%}, {t['alpha_vs_SPY']['alpha_hi']:+.1%}] "
              f"beta {t['alpha_vs_SPY']['beta']:.2f} | DSR {v['dsr']:.2f} | rule {v['rule']}")
        print(f"      trades {json.dumps(v['trades'], default=float)}")
        print(f"      by year {json.dumps({y: round(s, 2) for y, s in v['by_year'].items()})}")
    print("-" * w)
    print("  diagnostics:", json.dumps(dg, default=float, indent=None)[:2000])
    print("=" * w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
