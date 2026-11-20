"""Descriptive extraction for the strategies figure (decides nothing, logs no trial line).

Per arm (B0-S2, Phase C's eight, H7's four): net / cost / gross per $, exit mix, hold, re-entry
share, and a seeded sample of real dev trades, each relabelled through the labeller itself (entry,
barriers, exit) with its minute path, the score before entry, and for H7 the bounce levels and the
five conditions minute by minute. Every arm's net per $ must reproduce its logged result.

    cd .worktrees/poolpoc-run4 && python <this> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))

from research.poolpoc.stage1 import paths  # noqa: E402
from research.poolpoc.stage2m import config, e6, labels, report_util, run_e6  # noqa: E402
from research.poolpoc.stage3 import bounce, menu, phase_c, run_a0, run_c  # noqa: E402

OUT = Path(sys.argv[1])
ROOT = report_util.OUT_ROOT
E5A = ROOT / "c7878b02c333" / "e5a"
FIT = ROOT / "c66d1ed747ec" / "s3-c-fit"
RES_C = json.loads((ROOT / "c66d1ed747ec" / "s3-c" / "results.json").read_text())
RES_M = json.loads((ROOT / "2783a99c04c7" / "s3-m" / "results.json").read_text())
TR_M = pd.read_parquet(ROOT / "2783a99c04c7" / "s3-m" / "trades_m.parquet")
N_BOOT = 20
K_SAMPLE = 14
SEED = 20261121
XS = phase_c.XSTAR
KEY = ["date", "symbol", "t"]
PRE, POST = 45, 8
BARRIERS = {"b0": (30, 1.0, 1.0), "h1a": (30, 1.0, 1.0), "h1b": (30, 1.0, 2.0),
            "h1c": (60, 1.0, np.inf), "h1d": (60, 1.0, 2.0), "h5_drift": (30, 1.0, 1.0),
            "h5_pnl": (30, 1.0, 1.0), "h2q": (30, 1.0, 1.0), "h3meta": (30, 1.0, 1.0)}
PLAN_BARRIER = {"x15": (15, 1.0, 1.0), "x30": (30, 1.0, 1.0), "x60": (60, 1.0, 1.0),
                "x60u2": (60, 2.0, 1.0), "x60d2": (60, 1.0, 2.0)}
CLS = {labels.TP: "TP", labels.SL: "SL", labels.TIME: "TIME"}


def fnum(v: float) -> float | None:
    v = float(v)
    return round(v, 6) if np.isfinite(v) else None


def split(tr: pd.DataFrame) -> dict[str, float]:
    w = tr["notional"].to_numpy(float)
    net = float((tr["r_net"].to_numpy(float) * w).sum() / w.sum())
    cost = float((tr["c_rt"].to_numpy(float) * w).sum() / w.sum())
    return {"net": net, "cost": cost, "gross": net + cost}


def describe(tr: pd.DataFrame) -> dict[str, object]:
    cls = tr["cls"].to_numpy(int)
    first = ~tr.duplicated(["date", "symbol"])
    hold = tr["exit_min"].to_numpy(float) - (tr["t"].to_numpy(float) + 1)
    return {**split(tr), "trades": int(len(tr)), "days": int(tr["date"].nunique()),
            "mix": {CLS[c]: float(np.mean(cls == c)) for c in CLS},
            "hold_med": float(np.median(hold)), "reentry_share": float(1 - first.mean())}


def row_of(ctx, date: pd.Timestamp, sym: str) -> int:
    nd = ctx.nd
    i = np.nonzero((nd["date"].to_numpy() == np.datetime64(date)) & (nd["symbol"].to_numpy() == sym))[0]
    assert len(i) == 1, (date, sym)
    return int(i[0])


def one(ctx, i: int) -> tuple[dict, dict]:
    b = {k: v[i:i + 1] for k, v in ctx.b.items()}
    x = {k: (v[i:i + 1] if np.ndim(v) else v) for k, v in ctx.x.items()}
    return b, x


def conditions(b: dict, x: dict, t: int) -> list[list[int]]:
    """The five §10.1 conditions at minutes t-20..t (1/0), from bars <= each minute."""
    h, lo, c, v = (np.asarray(b[k], float)[0] for k in ("h", "l", "c", "v"))
    sig, crt = np.asarray(x["sig"], float)[0], np.asarray(x["c_rt_pit"], float)[0]
    W = bounce.WINDOW
    rows = []
    for m in range(max(t - 20, W), t + 1):
        lw = lo[m - W:m + 1]
        lw = np.where(np.isfinite(lw), lw, np.inf)
        off = int(np.argmin(lw))
        jl, low = m - W + off, lw[off]
        hw = h[max(jl - W, 0):jl + 1]
        high = np.nanmax(hw) if np.isfinite(hw).any() else np.nan
        between = h[jl + 1:m]
        mh = np.nanmax(between) if np.isfinite(between).any() else np.nan
        vp = v[m - W:m]
        vp = vp[np.isfinite(vp) & (vp > 0)]
        need = max(bounce.DROP_SIG * sig[m] * np.sqrt(W), bounce.DROP_COST * crt[m])
        c1 = (m - jl) >= bounce.HOLD
        c2 = np.isfinite(high) and (1 - low / high) >= need
        c3 = np.isfinite(mh) and c[m] > mh
        c4 = np.isfinite(high) and c[m] >= low + bounce.RETRACE * (high - low)
        c5 = len(vp) >= bounce.MIN_VOL_BARS and v[m] >= np.median(vp)
        rows.append([m, int(c1), int(c2), int(c3), int(c4), int(c5)])
    return rows


def path_of(b: dict, e: float, t: int, end: int) -> dict[str, list]:
    lo_, hi_ = max(t - PRE, 0), min(end + POST, config.N_MIN - 1)
    sl = slice(lo_, hi_ + 1)
    rel = lambda k: [fnum(z / e - 1) for z in np.asarray(b[k], float)[0, sl]]
    return {"m0": lo_, "o": rel("o"), "h": rel("h"), "l": rel("l"), "c": rel("c")}


def label_sigma(b: dict, x: dict, bar: tuple[int, float, float], t: int) -> dict:
    out = labels.label_day(b, x["sig"], x["f"], x["hs_real"], x["c_rt_pit"], x["notional"],
                           x["comm"], x["imp"], labels.Barrier(*bar))
    j = t - 1
    return {k: float(out[k][0, j]) for k in ("cls", "entry_px", "exit_min", "exit_px", "u", "d",
                                              "r_net")}


def label_chart(b: dict, x: dict, t: int) -> tuple[dict, dict]:
    bo = bounce.bounce(b, x["sig"], x["c_rt_pit"], x["f"])
    T = bounce.struct_barrier().horizon_end()
    out = bounce.label_plan(b, x["f"], x["hs_real"], x["notional"], x["comm"], x["imp"],
                            bo["u"], bo["d"], T)
    j = t - 1
    o = {k: float(out[k][0, j]) for k in ("cls", "exit_min", "exit_px", "u", "d", "r_net")}
    o["entry_px"] = float(np.asarray(b["o"], float)[0, t + 1]
                          * (1 + np.asarray(x["hs_real"], float)[0, t + 1]))
    return o, {k: float(bo[k][0, j]) for k in ("low", "high", "jl", "flag")}


def sample_trade(ctx, tr_row: pd.Series, bar: tuple | None, plan: str | None,
                 score: dict | None) -> dict:
    t = int(tr_row["t"])
    i = row_of(ctx, pd.Timestamp(tr_row["date"]), str(tr_row["symbol"]))
    b, x = one(ctx, i)
    lev = None
    if plan == "chart":
        lab, lev = label_chart(b, x, t)
        H = bounce.STRUCT_H
    else:
        lab = label_sigma(b, x, bar, t)
        H = bar[0]
    assert int(lab["cls"]) == int(tr_row["cls"]), ("cls", tr_row.to_dict(), lab)
    assert lab["exit_min"] == float(tr_row["exit_min"]), ("exit", tr_row.to_dict(), lab)
    assert abs(lab["r_net"] - float(tr_row["r_net"])) < 2e-6, ("r_net", tr_row.to_dict(), lab)
    if plan is not None and plan != "chart":
        bo = bounce.bounce(b, x["sig"], x["c_rt_pit"], x["f"])
        lev = {k: float(bo[k][0, t - 1]) for k in ("low", "high", "jl", "flag")}
    e = lab["entry_px"]
    T = min(t + H, config.LAST_MONITOR)
    rec = {"date": str(pd.Timestamp(tr_row["date"]).date()), "symbol": str(tr_row["symbol"]),
           "t": t, "H": H, "T": T, "cls": CLS[int(lab["cls"])], "exit_min": int(lab["exit_min"]),
           "entry_px": fnum(e), "exit_ret": fnum(lab["exit_px"] / e - 1), "u": fnum(lab["u"]),
           "d": fnum(lab["d"]), "r_net": fnum(lab["r_net"]), "c_rt": fnum(tr_row["c_rt"]),
           "notional": fnum(tr_row["notional"]), "plan": plan,
           "path": path_of(b, e, t, max(int(lab["exit_min"]), T))}
    if lev is not None:
        rec["bounce"] = {"low": fnum(lev["low"] / e - 1), "high": fnum(lev["high"] / e - 1),
                         "jl": int(lev["jl"]), "flag": bool(lev["flag"] > 0),
                         "cond": conditions(b, x, t)}
    if score is not None:
        rec["score"] = score
    return rec


def score_trail(fr: pd.DataFrame, idx: dict, tr_row: pd.Series) -> dict:
    t = int(tr_row["t"])
    key = (pd.Timestamp(tr_row["date"]), str(tr_row["symbol"]))
    rows = idx.get(key)
    sc = fr["ev"].to_numpy(float)[rows] if rows is not None else np.array([])
    ts = fr["t"].to_numpy(int)[rows] if rows is not None else np.array([])
    keep = (ts >= t - 20) & (ts <= t)
    return {"t": ts[keep].tolist(), "v": [fnum(z) for z in sc[keep]]}


def main() -> None:
    pa = run_a0.load_predictions(E5A)
    days = pd.DatetimeIndex(sorted(pa["date"].unique()))
    spy = run_a0._spy(days)
    fold_of_day = pa.groupby("date")["fold"].first().astype(str)
    b0_full = e6.run_arm(run_e6.frame_e5a(pa, XS, "s2"), days, phase_c.P0, spy, N_BOOT)
    pc, cal, tab, gfolds, _ = run_c.load_fit(FIT, pa, "full")
    pool = ROOT / run_a0.REF_SHA / "e_1" / "pool.parquet"
    ctx = run_a0.cut_context(run_a0.load_context(paths.CACHE_DIR, pool),
                             pa[["date", "symbol"]].drop_duplicates().reset_index(drop=True))
    f0 = run_e6.frame_e5a(pa, XS, "s2")
    frames = run_c.h1_frames(ctx, f0)
    h5f, h5_info = run_c.h5_frames(f0, b0_full, tab, gfolds, days, fold_of_day)
    frames.update(h5f)
    scores, _ = run_c.h2_scores(f0, pc, cal)
    scores.update(run_c.h3_scores(f0, pc))
    for name in ("h2q", "h3meta"):
        frames[name] = (f0.assign(ev=scores[name]), "cool5")
    crt = pa[KEY + ["c_rt_pit"]].astype({"t": np.int64}).rename(columns={"c_rt_pit": "c_rt"})
    rng = np.random.default_rng(SEED)
    out: dict[str, object] = {"arms": {}, "source": {
        "phase_c": RES_C["git"]["sha12"], "h7": RES_M["git"]["sha12"], "e5a": "c7878b02c333",
        "days": int(len(days)), "seed": SEED}}
    arms = {"b0": (f0, "cool5", b0_full)}
    for name in ("h1a", "h1b", "h1c", "h1d", "h5_drift", "h5_pnl", "h2q", "h3meta"):
        fr, re_ = frames[name]
        arms[name] = (fr, re_, phase_c.run_arm(fr, days, re_, spy, N_BOOT))
    for name, (fr, re_, a) in arms.items():
        logged = (RES_C["b0"] if name == "b0" else RES_C["arms"][name]["arm"])["per_dollar"]["mean"]
        got = run_a0._summary(a)["per_dollar"]["mean"]
        assert abs(got - logged) < 1e-12, (name, got, logged)
        tr = a["trades"][["date", "symbol", "t", "exit_min", "r_net", "notional"]].copy()
        tr["t"] = tr["t"].astype(np.int64)
        tr = tr.merge(fr[KEY + ["cls"]].astype({"t": np.int64}), on=KEY, how="left",
                      validate="m:1").merge(crt, on=KEY, how="left", validate="m:1")
        assert tr["cls"].notna().all() and tr["c_rt"].notna().all(), name
        d = describe(tr)
        assert abs(d["net"] - logged) < 1e-9, (name, d["net"], logged)
        idx = fr.groupby(["date", "symbol"]).indices
        pick = tr.iloc[np.sort(rng.choice(len(tr), size=min(K_SAMPLE, len(tr)), replace=False))]
        d["samples"] = [sample_trade(ctx, r, BARRIERS[name], None, score_trail(fr, idx, r))
                        for _, r in pick.iterrows()]
        d["reentry"] = re_
        out["arms"][name] = d
        print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()
                     if k not in ("samples", "mix")}, d["mix"], flush=True)
    for name in ("h7_x", "h7_chart", "h7_menu", "h7_direct"):
        tr = TR_M[TR_M["arm"] == name].reset_index(drop=True)
        d = describe(tr)
        assert abs(d["net"] - RES_M["arms"][name]["split"]["net"]) < 1e-9, name
        d["plan_mix"] = {p: float(np.mean(tr["plan"].to_numpy(int) == k))
                         for k, p in enumerate(menu.PLANS)}
        pick = tr.iloc[np.sort(rng.choice(len(tr), size=K_SAMPLE, replace=False))]
        d["samples"] = []
        for _, r in pick.iterrows():
            plan = menu.PLANS[int(r["plan"])]
            d["samples"].append(sample_trade(ctx, r, PLAN_BARRIER.get(plan), plan, None))
        d["reentry"] = "episode"
        out["arms"][name] = d
        print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()
                     if k not in ("samples", "mix", "plan_mix")}, d["mix"], flush=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
