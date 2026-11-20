"""Merge the extraction (extract.json) with the logged Phase C and H7 results into figure.json.

    python compact.py <data_root> <extract.json> <figure.json>

Everything here is a copy or a rounding of a logged or extracted number; nothing is computed
that the source does not hold, except per-cent rounding and the path's offsets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
EXTRACT = json.loads(Path(sys.argv[2]).read_text())
OUT = Path(sys.argv[3])
RES_C = json.loads((ROOT / "c66d1ed747ec" / "s3-c" / "results.json").read_text())
RES_M = json.loads((ROOT / "2783a99c04c7" / "s3-m" / "results.json").read_text())
PRE = 36
POST_PAD = 2


def pct(v: float | None, nd: int = 3) -> float | None:
    return None if v is None else round(100 * v, nd)


def bp(v: float | None) -> int | None:
    return None if v is None else int(round(v * 1e4))


def logged(name: str) -> dict:
    if name == "b0":
        a, split = RES_C["b0"], RES_M["b0_split"]
        return {"mean": pct(a["mean"]), "lo": pct(a["lo"]), "hi": pct(a["hi"]),
                "pd": pct(a["per_dollar"]["mean"], 2), "pd_lo": pct(a["per_dollar"]["lo"], 2),
                "pd_hi": pct(a["per_dollar"]["hi"], 2), "tpd": round(a["trades_per_day"], 2),
                "days": a["days_traded"], "trades": a["trades"], "vs_b0": None, "p_holm": None,
                "halves": None, "y2022": None}
    v = (RES_C["arms"] | RES_M["arms"])[name]
    a, vs = v["arm"], v["vs_b0"]
    return {"mean": pct(a["mean"]), "lo": pct(a["lo"]), "hi": pct(a["hi"]),
            "pd": pct(a["per_dollar"]["mean"], 2), "pd_lo": pct(a["per_dollar"]["lo"], 2),
            "pd_hi": pct(a["per_dollar"]["hi"], 2), "tpd": round(a["trades_per_day"], 2),
            "days": a["days_traded"], "trades": a["trades"],
            "vs_b0": [pct(vs["per_dollar"]["mean"], 2), pct(vs["per_dollar"]["lo"], 2),
                      pct(vs["per_dollar"]["hi"], 2)],
            "p_holm": round(v["p_holm"], 4),
            "halves": [pct(x) for x in v["half_years"].values()],
            "y2022": pct(v["y2022"])}


def sample(s: dict) -> dict:
    p, t = s["path"], s["t"]
    m0 = p["m0"]
    lo_m = max(t - PRE, m0)
    hi_m = min(max(s["exit_min"], s["T"] + 1) + POST_PAD, m0 + len(p["c"]) - 1)
    cut = lambda k: [bp(x) for x in p[k][lo_m - m0:hi_m - m0 + 1]]
    out = {"sym": s["symbol"], "date": s["date"], "t": t, "H": s["H"], "T": s["T"], "cls": s["cls"],
           "exit": s["exit_min"], "xr": bp(s["exit_ret"]), "u": bp(s["u"]), "d": bp(s["d"]),
           "r": bp(s["r_net"]), "c": bp(s["c_rt"]), "plan": s["plan"], "m0": lo_m,
           "o": cut("o"), "h": cut("h"), "l": cut("l"), "cl": cut("c")}
    if "score" in s:
        out["sc"] = [[tt, bp(v)] for tt, v in zip(s["score"]["t"], s["score"]["v"])]
    if "bounce" in s:
        b = s["bounce"]
        out["lowr"], out["highr"], out["jl"] = bp(b["low"]), bp(b["high"]), b["jl"]
        out["cond"] = b["cond"]
    return out


arms = {}
for name, a in EXTRACT["arms"].items():
    arms[name] = {"log": logged(name),
                  "split": {"gross": pct(a["gross"], 2), "cost": pct(a["cost"], 2),
                            "net": pct(a["net"], 2)},
                  "mix": {k: round(v, 3) for k, v in a["mix"].items()},
                  "plan_mix": {k: round(v, 3) for k, v in a.get("plan_mix", {}).items()},
                  "reentry_share": round(a["reentry_share"], 3), "trades": a["trades"],
                  "samples": [sample(s) for s in a["samples"]]}
for name in ("h1a", "h1b", "h1c", "h1d", "h5_drift", "h5_pnl", "h2q", "h3meta", "h7_x",
             "h7_chart", "h7_menu", "h7_direct"):
    assert abs(arms[name]["split"]["net"] - arms[name]["log"]["pd"]) < 0.006, name
out = {"source": EXTRACT["source"] | {"phase_c_mlflow": RES_C["mlflow_run_id"],
                                      "h7_mlflow": RES_M["mlflow_run_id"]},
       "n_trials": RES_M["overfit"]["n_trials"], "days": EXTRACT["source"]["days"],
       "not_run": {**RES_C["not_run"], **RES_M["not_run"]},
       "h5_on": {"h5_drift": 429, "h5_pnl": 384}, "arms": arms}
OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")
