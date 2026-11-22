"""Evening report for C3 paper trading (PAPER_C3 §3-4).

For each live round trip, using that day's SIP bars, research references are:
- entry: the SIP open of bar entry_check + 1 (where research fills);
- exit, by kind:
  - stop filled between checks: min(stop level, SIP open of the fill's bar);
  - forced exit at a check: the SIP open of bar exit_check + 1;
  - time exit: the SIP close of the last bar.

realised cost (bp) = slippage against those references + the fees actually owed.
modelled cost (bp) = costs.BASE for the same prices and exit kind.

G-cost: median realised <= 2 x median modelled, judged from 20 round trips on.
G-fidelity: >= 90% of live trades share day and entry check with a research
trade from cells.c3 re-run on SIP.
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
MIN_TRIPS = 20


def fees_owed(day: pd.Timestamp, qty: int, entry: float, exit_: float) -> float:
    """Dollar fees Alpaca would charge a real account (paper charges none)."""
    f = CAT * qty * 2 + SEC_RATE_NOW * qty * exit_
    if day >= TAF_FROM:
        f += min(TAF * qty, 9.79)
    return f


def bar_of(ts: str, day: pd.Timestamp) -> int:
    t = datetime.fromisoformat(ts).astimezone(ET)
    return (t.hour - 9) * 60 + (t.minute - 30)


def trade_costs(tr: dict, m: Minutes, i: int) -> dict:
    """Realised vs modelled cost of one live round trip, against SIP row i."""
    day = m.dates[i]
    last = int(m.n_bars[i]) - 1
    e_ref = float(m.open[i, tr["entry_check"] + 1])
    kind = tr["exit_kind"]
    if kind == "time":
        x_ref, model_kind = float(m.close[i, last]), "time"
    elif kind == "sl" and tr["exit_check"] is not None:
        x_ref, model_kind = float(m.open[i, min(tr["exit_check"] + 1, last)]), "sl"
    elif kind == "sl":
        k = min(max(bar_of(tr["exit_ts"], day), 0), last)
        x_ref, model_kind = min(float(tr["stop_at_exit"]), float(m.open[i, k])), "sl"
    else:
        return {"trade_id": tr["id"], "kind": kind, "usable": False}
    e, x, q = float(tr["entry_price"]), float(tr["exit_price"]), int(tr["qty"])
    slip = ((e - e_ref) + (x_ref - x)) / e_ref
    fee = fees_owed(day, q, e, x) / (q * e)
    modelled = costs.BASE.entry_frac(e_ref) + costs.BASE.exit_frac(x_ref, model_kind) * x_ref / e_ref
    return {"trade_id": tr["id"], "kind": kind, "usable": True,
            "entry_slip_bp": (e - e_ref) / e_ref * 1e4, "exit_slip_bp": (x_ref - x) / e_ref * 1e4,
            "fees_bp": fee * 1e4, "realised_bp": (slip + fee) * 1e4, "modelled_bp": modelled * 1e4,
            "pnl": q * (x - e) - fees_owed(day, q, e, x), "gross_ret": x / e - 1.0}


def fidelity(live: list[dict], research: list[cells.Trade]) -> dict:
    lv = {(t["day"], int(t["entry_check"])) for t in live}
    rs = {(str(t.date.date()), t.entry_bar - 1) for t in research}
    both = lv & rs
    return {"live": len(lv), "research": len(rs), "matched": len(both),
            "live_matched_share": len(both) / len(lv) if lv else np.nan,
            "live_only": sorted(lv - rs), "research_only": sorted(rs - lv)}


def gates(rows: list[dict], fid: dict) -> dict:
    u = [r for r in rows if r.get("usable")]
    out = {"round_trips": len(u)}
    if u:
        real = float(np.median([r["realised_bp"] for r in u]))
        mod = float(np.median([r["modelled_bp"] for r in u]))
        out.update(median_realised_bp=real, median_modelled_bp=mod, ratio=real / mod if mod else np.nan)
        out["g_cost"] = "pending" if len(u) < MIN_TRIPS else ("pass" if real <= 2 * mod else "FAIL")
    share = fid.get("live_matched_share", np.nan)
    out["g_fidelity"] = "pending" if len(u) < MIN_TRIPS else ("pass" if share >= 0.9 else "FAIL")
    return out
