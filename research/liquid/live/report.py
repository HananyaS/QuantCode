"""Evening report and gates for C3 paper trading (PAPER_C3 §3-4, amendment A1 after the G3 review).

Per live round trip, against that day's SIP data (all in bp of the research entry reference):
- references (research's own fill before costs):
  - entry: the SIP open of bar entry_check + 1;
  - resting stop: min(stop level sent, SIP open of the minute the broker filled it);
  - forced market exit at a check: the SIP open of bar exit_check + 1;
  - time exit: the SIP close of the last bar;
- execution: fill against the SIP NBBO midpoint at the broker's fill time (for a resting
  stop, fill against its reference), signed so that positive = worse;
- latency: midpoint at the fill time against the reference (zero for resting stops);
- fees: those a real account would owe (paper charges none).

G-cost (decides C3): cost = execution + fees per round trip, the mean over complete days.
- interim, first days holding >= 20 round trips: FAIL if the 99% day-clustered CI lower bound > B;
- decision, first days holding >= 40: FAIL if the 95% lower bound > B; PASS if the 95% upper bound
  < B - max(0, latency 95% lower bound); otherwise extend once to 80, then INCONCLUSIVE.
  B = 1.66 bp, C3's dev break-even against SPY (results/c3_breakeven.json).
G-fidelity: pooled Jaccard of (day, entry check), live vs cells.c3 re-run on SIP, over the
  G-cost decision days; PASS at >= 0.85 (the live feed scored 0.97 on 2022, results/c3_iex_fidelity.json).
G-safety, per day: no automatic kill, flat with no open order at the end, ledger fills equal to
  broker fills, and no position without a working stop for more than 60 s outside the final minute.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.liquid import cells, costs
from research.liquid.data import Minutes

ET = ZoneInfo("America/New_York")
SEC_RATE_NOW = 20.6e-6                    # $ per $ sold, from 2026-04-04 (fact sheet item 10)
CAT = 0.000003
TAF, TAF_FROM = 0.000195, pd.Timestamp("2027-01-01")
BREAKEVEN_BP = 1.66
INTERIM_N, FINAL_N, EXTEND_N = 20, 40, 80
FIDELITY_BAR = 0.85
UNPROTECTED_MAX_S = 60.0
N_BOOT, SEED = 2000, 17
DRILL_PREFIX = "c3-drill"


def fees_owed(day: pd.Timestamp, qty: int, entry: float, exit_: float) -> float:
    """Dollar fees Alpaca would charge a real account (paper charges none)."""
    f = CAT * qty * 2 + SEC_RATE_NOW * qty * exit_
    if day >= TAF_FROM:
        f += min(TAF * qty, 9.79)
    return f


def _ts(x: str | None) -> datetime | None:
    if not x:
        return None
    t = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=ET)


def bar_of(ts: str, day: pd.Timestamp) -> int:
    t = _ts(ts).astimezone(ET)
    return (t.hour - 9) * 60 + (t.minute - 30)


# ------------------------------------------------------------------ per trip
def trip_costs(tr: dict, m: Minutes, i: int, mid_entry: float, mid_exit: float) -> dict:
    """Execution, latency and fees of one live round trip against SIP row i (bp of the entry reference)."""
    base = {"day": tr["day"], "trade_id": tr["id"], "kind": tr["exit_kind"]}
    kind, x = tr["exit_kind"], tr["exit_price"]
    if x is None or kind not in ("sl", "time"):
        return base | {"usable": False, "why": f"exit {kind}"}
    day, last = m.dates[i], int(m.n_bars[i]) - 1
    q, e = int(tr["qty"]), float(tr["entry_price"])
    ref_e = float(m.open[i, tr["entry_check"] + 1])
    resting = kind == "sl" and tr["exit_check"] is None
    if not np.isfinite(mid_entry) or (not resting and not np.isfinite(mid_exit)):
        return base | {"usable": False, "why": "no SIP quote at a fill"}
    lat_e, exe_e = (mid_entry - ref_e) / ref_e, (e - mid_entry) / ref_e
    if kind == "time":
        ref_x = float(m.close[i, last])
    elif not resting:
        ref_x = float(m.open[i, min(int(tr["exit_check"]) + 1, last)])
    else:
        k = min(max(bar_of(tr["exit_filled_at"] or tr["exit_ts"], day), 0), last)
        ref_x = min(float(tr["stop_at_exit"]), float(m.open[i, k]))
    if resting:
        lat_x, exe_x = 0.0, (ref_x - x) / ref_e
    else:
        lat_x, exe_x = (ref_x - mid_exit) / ref_e, (mid_exit - x) / ref_e
    fee = fees_owed(day, q, e, x) / (q * e)
    model = costs.BASE.entry_frac(ref_e) + costs.BASE.exit_frac(ref_x, kind) * ref_x / ref_e
    return base | {"usable": True, "resting_stop": resting,
                   "cost_bp": (exe_e + exe_x + fee) * 1e4, "latency_bp": (lat_e + lat_x) * 1e4,
                   "exec_entry_bp": exe_e * 1e4, "exec_exit_bp": exe_x * 1e4, "fees_bp": fee * 1e4,
                   "model_bp": model * 1e4, "vs_research_bp": (exe_e + exe_x + fee + lat_e + lat_x) * 1e4,
                   "pnl": q * (x - e) - fees_owed(day, q, e, x), "gross_ret": x / e - 1.0}


# ------------------------------------------------------------------ statistics
def cluster_mean_ci(rows: list[dict], key: str, level: float) -> tuple[float, float, float]:
    """Mean per round trip with a bootstrap CI that resamples whole days."""
    days = sorted({r["day"] for r in rows})
    s = np.array([sum(r[key] for r in rows if r["day"] == d) for d in days], float)
    c = np.array([sum(1 for r in rows if r["day"] == d) for d in days], float)
    idx = np.random.default_rng(SEED).integers(0, len(days), (N_BOOT, len(days)))
    means = s[idx].sum(1) / c[idx].sum(1)
    a = (1 - level) / 2
    return float(s.sum() / c.sum()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def window(rows: list[dict], n: int) -> list[dict] | None:
    """Round trips of the first complete days that together hold at least n; None if not yet."""
    days = sorted({r["day"] for r in rows})
    out: list[dict] = []
    for d in days:
        out += [r for r in rows if r["day"] == d]
        if len(out) >= n:
            return out
    return None


def _decide(w: list[dict]) -> tuple[str | None, dict]:
    m, lo, hi = cluster_mean_ci(w, "cost_bp", 0.95)
    lm, llo, lhi = cluster_mean_ci(w, "latency_bp", 0.95)
    bar = BREAKEVEN_BP - max(0.0, llo)
    detail = {"n": len(w), "days": len({r["day"] for r in w}), "cost_mean_bp": m, "cost_ci95": [lo, hi],
              "latency_mean_bp": lm, "latency_ci95": [llo, lhi], "pass_bar_bp": bar}
    if lo > BREAKEVEN_BP:
        return "FAIL", detail
    if hi < bar:
        return "PASS", detail
    return None, detail


def g_cost(rows: list[dict]) -> dict:
    u = sorted((r for r in rows if r.get("usable")), key=lambda r: (r["day"], r["trade_id"]))
    out: dict = {"breakeven_bp": BREAKEVEN_BP, "usable_round_trips": len(u),
                 "excluded_round_trips": sum(1 for r in rows if not r.get("usable"))}
    if u:
        m, lo, hi = cluster_mean_ci(u, "cost_bp", 0.95)
        lm, llo, lhi = cluster_mean_ci(u, "latency_bp", 0.95)
        out["running"] = {"cost_mean_bp": m, "cost_ci95": [lo, hi], "latency_mean_bp": lm,
                          "latency_ci95": [llo, lhi], "model_mean_bp": float(np.mean([r["model_bp"] for r in u]))}
    w = window(u, FINAL_N)
    if w is None:
        wi = window(u, INTERIM_N)
        if wi is None:
            return out | {"status": "pending", "decision_days": None}
        m, lo, hi = cluster_mean_ci(wi, "cost_bp", 0.99)
        out["interim"] = {"n": len(wi), "cost_mean_bp": m, "cost_ci99": [lo, hi]}
        return out | {"status": "FAIL" if lo > BREAKEVEN_BP else "pending", "decision_days": None}
    s, d40 = _decide(w)
    out["decision_40"] = d40
    days = sorted({r["day"] for r in w})
    if s is None:
        w80 = window(u, EXTEND_N)
        if w80 is None:
            return out | {"status": "extended to 80", "decision_days": None}
        s, d80 = _decide(w80)
        out["decision_80"] = d80
        s, days = s or "INCONCLUSIVE", sorted({r["day"] for r in w80})
    return out | {"status": s, "decision_days": days}


def g_fidelity(entries: dict[str, dict[str, list[int]]], days: list[str] | None) -> dict:
    """entries: day -> {"live": [checks], "research": [checks]}. Pooled Jaccard, both directions."""
    def pool(ds):
        lv = {(d, c) for d in ds for c in entries[d]["live"]}
        rs = {(d, c) for d in ds for c in entries[d]["research"]}
        u = lv | rs
        return {"live": len(lv), "research": len(rs), "matched": len(lv & rs),
                "jaccard": len(lv & rs) / len(u) if u else float("nan"),
                "live_only": sorted(lv - rs), "research_only": sorted(rs - lv)}
    out = {"bar": FIDELITY_BAR, "running": pool(sorted(entries))}
    if days is None:
        return out | {"status": "pending"}
    w = pool([d for d in days if d in entries])
    return out | {"window": w, "status": "PASS" if w["jaccard"] >= FIDELITY_BAR else "FAIL"}


# ------------------------------------------------------------------ safety
def day_safety(day: str, events: list[dict], trades: list[dict], closed: list) -> dict:
    """Kills, end-of-day flatness, ledger-vs-broker fills and unprotected seconds for one session."""
    kills = [e["reason"] for e in events if e["kind"] == "kill"]
    guard = [e["reason"] for e in events if e["kind"] == "guard_flatten"]
    ends = [e for e in events if e["kind"] == "reconcile" and e.get("when") == "end"]
    flat = bool(ends) and ends[-1]["position"] == 0 and ends[-1]["open_orders"] == 0
    led_ids: set[str] = set()
    for e in events:
        if e["kind"] in ("entry_filled", "exit", "stop_partial") and e.get("order_id"):
            led_ids.add(e["order_id"])
        if e["kind"] in ("flattened", "guard_flattened"):
            led_ids |= set(e.get("orders", []))
    brk = {o.id: o for o in closed if o.filled_qty > 0 and not o.client_order_id.startswith(DRILL_PREFIX)}
    unprot = {}
    for t in trades:
        filled = _ts(t["entry_filled_at"]) or _ts(t["entry_ts"])
        placed = [_ts(e["ts"]) for e in events if e["kind"] == "stop_placed" and e.get("trade_id") == t["id"]]
        exited = _ts(t["exit_filled_at"]) or _ts(t["exit_ts"])
        gaps = [((placed[0] if placed else exited) - filled).total_seconds()] if filled and (placed or exited) else []
        if t["exit_kind"] != "time":
            for e in events:
                if e["kind"] == "stop_cancel_sent" and e.get("trade_id") == t["id"] and exited:
                    gaps.append((exited - _ts(e["ts"])).total_seconds())
        unprot[t["id"]] = max(gaps) if gaps else None
    worst = max((v for v in unprot.values() if v is not None), default=0.0)
    out = {"day": day, "kills": kills, "guard": guard, "eod_flat": flat,
           "broker_fills_not_in_ledger": sorted(set(brk) - led_ids),
           "ledger_fills_not_at_broker": sorted(led_ids - set(brk)) if closed is not None else [],
           "unprotected_s": unprot, "max_unprotected_s": worst}
    out["ok"] = (not kills and not guard and flat and not out["broker_fills_not_in_ledger"]
                 and not out["ledger_fills_not_at_broker"] and worst <= UNPROTECTED_MAX_S)
    return out


def g_safety(day_reports: list[dict]) -> dict:
    bad = [d["safety"]["day"] for d in day_reports if not d["safety"]["ok"]]
    return {"days": len(day_reports), "failing_days": bad, "status": "pass" if not bad else "FAIL"}


# ------------------------------------------------------------------ one day
def day_report(day: pd.Timestamp, m: Minutes, live: list[dict], events: list[dict], closed: list,
               mids: dict[int, tuple[float, float]]) -> dict:
    """Everything the evening report records for one session. m holds the session as its last row."""
    i = len(m.dates) - 1
    assert m.dates[i] == day
    research = [t for t in cells.c3(m) if t.date == day]
    rows = [trip_costs(t, m, i, *mids.get(t["id"], (np.nan, np.nan))) for t in live if t["exit_kind"] is not None]
    d = str(day.date())
    return {"day": d, "live_trades": live,
            "research_trades": [{"entry_check": t.entry_bar - 1, "kind": t.kind, "exit_bar": t.exit_bar,
                                 "entry": t.entry, "exit": t.exit} for t in research],
            "entries": {"live": sorted(int(t["entry_check"]) for t in live),
                        "research": sorted(t.entry_bar - 1 for t in research)},
            "costs": rows, "safety": day_safety(d, events, live, closed),
            "pnl": float(sum(r["pnl"] for r in rows if r.get("usable")))}


def summary(day_reports: list[dict]) -> dict:
    rows = [r for d in day_reports for r in d["costs"]]
    gc = g_cost(rows)
    gf = g_fidelity({d["day"]: d["entries"] for d in day_reports}, gc.get("decision_days"))
    gs = g_safety(day_reports)
    return {"days": len(day_reports), "round_trips": len(rows), "g_cost": gc, "g_fidelity": gf, "g_safety": gs,
            "pnl_total": float(sum(d["pnl"] for d in day_reports))}
