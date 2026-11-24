"""Data for the C3 paper dashboard (claude.ai artifact 5ELZiTW9YF2qpXieq3yCFA).

Two builders, each writing tables of plain rows:

dev   -- once, on a workstation with the dev minute cache: the research baseline on the
         dev window 2016-01-04..2022-12-30 only. Reads nothing after 2022-12-30, so the
         confirmation window and the holdout stay sealed.
             python -m research.liquid.live.dashboard --dev OUT_DIR
live  -- on the server after each evening report (run_c3.sh): the paper window, from
         the first paper session on. Writes data/research_liquid/live/dash/ and, with
         --publish, the two Drive files named in data/research_liquid/live/publish.json.
             python -m research.liquid.live.dashboard --live [--publish]

A Drive file holds {"format": "c3-gzip-b64-v1", "data": base64(gzip(json))}. The Drive
connector's text reader escapes JSON punctuation as Markdown; base64 has none of it.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import logging
import math
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.liquid import cells, costs
from research.liquid.calendar import DEV_END, DEV_START, HOLDOUT_START
from research.liquid.data import Minutes

ET = ZoneInfo("America/New_York")
FORMAT = "c3-gzip-b64-v1"
LIVE_DIR = Path("data/research_liquid/live")
DASH_DIR = LIVE_DIR / "dash"
PUBLISH_CFG = LIVE_DIR / "publish.json"
SOURCE = Path(__file__).resolve()
EXIT_NAMES = {"sl": "stop", "time": "close", "tp": "target"}
logger = logging.getLogger("c3.dashboard")


# ------------------------------------------------------------------ helpers --
def hhmm(bar: int) -> str:
    """Clock time (ET) at the start of minute bar `bar` (bar 0 = 09:30)."""
    m = 9 * 60 + 30 + int(bar)
    return f"{m // 60:02d}:{m % 60:02d}"


def check_time(d: int) -> str:
    """A check decides on the close of bar d, i.e. at the start of bar d + 1."""
    return hhmm(d + 1)


def num(x: Any, nd: int | None = None) -> float | None:
    """A JSON-safe float (NaN and inf become null), optionally rounded."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd) if nd is not None else f


def series(a: Iterable[float], n: int, nd: int = 2) -> list[float | None]:
    return [num(v, nd) for v in list(a)[:n]]


def table(rows: list[dict], columns: list[str] | None = None) -> dict:
    """{columns, rows}: the column order is kept explicitly (object keys can lose theirs)."""
    cols = columns or (list(rows[0]) if rows else [])
    assert all(set(r) <= set(cols) for r in rows), "a row has a column the table does not list"
    return {"columns": cols, "rows": [{c: r.get(c) for c in cols} for r in rows]}


def wrap(doc: dict) -> str:
    """The Drive payload: gzip, then base64, inside a small JSON envelope."""
    raw = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return json.dumps({"format": FORMAT, "data": base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")})


def unwrap(text: str) -> dict:
    """Inverse of `wrap`, tolerant of the connector's Markdown escaping (mirrors the page's loader)."""
    clean = text.replace("\\", "")
    m = re.search(r'"data"\s*:\s*"([A-Za-z0-9+/=\s]+)"', clean)
    assert m, "no data field in the payload"
    return json.loads(gzip.decompress(base64.b64decode(re.sub(r"\s+", "", m.group(1)))).decode("utf-8"))


def day_arrays(m: Minutes, i: int) -> dict:
    """One session's close, noise-area bounds and VWAP, for a chart (cents)."""
    sigma, _ = cells.noise_area(m)
    n = int(m.n_bars[i])
    o0, pc = float(m.open[i, 0]), float(m.prev_close[i])
    ub = max(o0, pc) * (1 + sigma[i])
    lb = min(o0, pc) * (1 - sigma[i])
    return {"n_bars": n, "open0": num(o0, 2), "prev_close": num(pc, 2),
            "close": series(m.close[i], n), "ub": series(ub, n), "lb": series(lb, n),
            "vwap": series(cells.session_vwap(m, i), n)}


# ---------------------------------------------------------------- dev side --
def _meta(inputs: str) -> list[dict]:
    return [{"label": "dashboard.py", "value": SOURCE.read_text(encoding="utf-8"), "type": "code", "lang": "python"},
            {"label": "Input", "value": inputs}]


def research_checks(m: Minutes, i: int, day_trades: list[cells.Trade]) -> list[dict]:
    """Each half-hour check of one dev session, with what the research rule did there."""
    sigma, _ = cells.noise_area(m)
    o0, pc = m.open[i, 0], m.prev_close[i]
    ub = max(o0, pc) * (1 + sigma[i])
    vw = cells.session_vwap(m, i)
    last = m.last_bar(i)
    out = []
    for d in cells.C3_CHECKS:
        if d >= last or not np.isfinite(sigma[i, d]):
            continue
        c = float(m.close[i, d])
        held = next((t for t in day_trades if t.entry_bar - 1 < d < t.exit_bar), None)
        entered = next((t for t in day_trades if t.entry_bar - 1 == d), None)
        if entered is not None:
            action = "enter"
        elif held is not None:
            action = "hold"
        elif c > ub[d] and c > vw[d]:
            action = "enter_unfilled"
        elif c <= ub[d]:
            action = "inside"
        else:
            action = "below_vwap"
        stop = max(ub[d], vw[d]) if action in ("enter", "hold") else None
        out.append({"d": int(d), "time": check_time(d), "close": num(c, 4), "ub": num(ub[d], 4),
                    "vwap": num(vw[d], 4), "action": action, "stop": num(stop, 4)})
    return out


def research_trade_rows(tr: list[cells.Trade], m: Minutes) -> list[dict]:
    sigma, _ = cells.noise_area(m)
    pos = {d: i for i, d in enumerate(m.dates)}
    rows = []
    for k, t in enumerate(tr, 1):
        i = pos[t.date]
        d = t.entry_bar - 1
        o0, pc = m.open[i, 0], m.prev_close[i]
        ub = max(o0, pc) * (1 + sigma[i, d])
        vw = cells.session_vwap(m, i)[d]
        net = t.net(costs.BASE)
        rows.append({"id": k, "date": f"{t.date:%Y-%m-%d}", "year": int(t.date.year), "check": check_time(d),
                     "exit_time": hhmm(t.exit_bar + 1) if t.kind == "time" else hhmm(t.exit_bar),
                     "exit": EXIT_NAMES.get(t.kind, t.kind), "weight": num(t.weight, 3),
                     "close_at_check": num(m.close[i, d], 2), "ub": num(ub, 2), "vwap": num(vw, 2),
                     "entry_price": num(t.entry, 2), "exit_price": num(t.exit, 2),
                     "gross_bp": num(t.gross * 1e4, 2), "net_bp": num(net * 1e4, 2),
                     "cost_bp": num((t.gross - net) * 1e4, 2), "contrib_bp": num(t.weight * net * 1e4, 2),
                     "hold_min": int(t.exit_bar - t.entry_bar + (1 if t.kind == "time" else 0))})
    return rows


@dataclass(frozen=True)
class Example:
    key: str
    title: str
    rule: str          # how the day was chosen, shown beside it


def pick_examples(tr: list[cells.Trade], m: Minutes) -> list[tuple[Example, pd.Timestamp]]:
    """Four teaching days from 2021-2022, chosen by fixed rules (ties to the earliest day)."""
    f = cells.trade_frame(tr, costs.BASE)
    f = f[f["date"] >= pd.Timestamp("2021-01-01")]
    g = f.groupby("date").agg(n=("kind", "size"), kinds=("kind", lambda k: ",".join(k)),
                              gross=("gross", "sum"), net=("net", "sum"))
    one_time = g[(g.n == 1) & (g.kinds == "time")]
    one_sl = g[(g.n == 1) & (g.kinds == "sl")]
    multi = g[(g.n >= 2) & g.kinds.str.endswith("time")]
    held = one_time["gross"].idxmax()
    stopped = (one_sl["net"] - one_sl["net"].median()).abs().sort_index().idxmin()
    again = multi["gross"].sort_index().idxmax()
    traded = set(f["date"])
    last_close = m.close[np.arange(len(m.dates)), m.n_bars - 1]
    ret = pd.Series(last_close / m.prev_close - 1.0, index=m.dates)
    quiet = ret[(ret.index >= pd.Timestamp("2021-01-01")) & ~ret.index.isin(traded) & m.valid]
    down = quiet.sort_index().idxmin()
    return [(Example("held", "A breakout held to the close",
                     "2021-22 day with one trade that ran to the 16:00 close and the largest gain"), held),
            (Example("stopped", "Stopped out",
                     "2021-22 day with one trade that hit its stop, the loss nearest the median"), stopped),
            (Example("again", "Stopped, then back in",
                     "2021-22 day with two or more trades, the last held to the close, the largest gain"), again),
            (Example("down", "A down day: long-only stays out",
                     "2021-22 day without a trade and the largest fall"), down)]


def example_rows(tr: list[cells.Trade], m: Minutes) -> list[dict]:
    pos = {d: i for i, d in enumerate(m.dates)}
    rows = []
    for ex, day in pick_examples(tr, m):
        i = pos[day]
        mine = [t for t in tr if t.date == day]
        a = day_arrays(m, i)
        trades = [{"entry_bar": int(t.entry_bar), "exit_bar": int(t.exit_bar), "check": check_time(t.entry_bar - 1),
                   "entry": num(t.entry, 2), "exit": num(t.exit, 2), "exit_kind": EXIT_NAMES.get(t.kind, t.kind),
                   "weight": num(t.weight, 3), "net_bp": num(t.net(costs.BASE) * 1e4, 2)} for t in mine]
        last_close = float(m.close[i, a["n_bars"] - 1])
        rows.append({"key": ex.key, "title": ex.title, "rule": ex.rule, "day": f"{day:%Y-%m-%d}",
                     "day_ret": num(last_close / m.prev_close[i] - 1, 6),
                     "c3_ret": num(sum(t.weight * t.net(costs.BASE) for t in mine), 6), "n_trades": len(mine),
                     **a, "checks": research_checks(m, i, mine), "trades": trades})
    return rows


def build_dev(out: Path) -> dict[str, Path]:
    """The research baseline files. Workstation only: needs the dev cache, scipy and the benchmarks."""
    from research.liquid.data import load
    from research.penny.eval import benchmark as bm
    from research.penny.scripts.fetch_benchmarks import load as load_bench

    spy = load("SPY", DEV_START, DEV_END)
    assert spy.dates[-1] <= DEV_END < HOLDOUT_START
    dates = spy.dates
    tr = cells.c3(spy)
    net = cells.daily(tr, dates, costs.BASE)
    gross = cells.daily(tr, dates, None)
    saved = pd.read_parquet("data/research_liquid/l1_dev_daily.parquet")["C3"]
    assert np.allclose(saved.reindex(dates).to_numpy(), net.to_numpy()), "C3 daily differs from l1_dev_daily"
    bench, _ = load_bench(DEV_START, DEV_END)
    bench = bench.reindex(dates)
    l1 = json.loads(Path("data/research_liquid/l1_dev.json").read_text(encoding="utf-8"))
    c3 = l1["cells"]["C3"]
    tab = {r["index"]: r for r in c3["comparison"]["table"]}
    tests = c3["comparison"]["tests"]
    be = json.loads(Path("research/liquid/results/c3_breakeven.json").read_text(encoding="utf-8"))

    daily = [{"date": f"{d:%Y-%m-%d}", "c3": num(net.iloc[k], 8), "c3_gross": num(gross.iloc[k], 8),
              "spy": num(bench["SPY"].iloc[k], 8), "qqq": num(bench["QQQ"].iloc[k], 8)}
             for k, d in enumerate(dates) if k > 0]                      # l1_dev scores from the 2nd session

    def stats(r: pd.Series) -> dict:
        x = r.to_numpy()[1:]
        return {"sharpe": num(bm.sharpe(x), 4), "vol": num(np.std(x, ddof=1) * np.sqrt(252), 4)}

    summary = []
    for name, key, col in (("C3", "strategy", net), ("SPY", "SPY", bench["SPY"]), ("QQQ", "QQQ", bench["QQQ"])):
        t = tab[key]
        summary.append({"series": name, "sharpe": num(t["sharpe"], 4), "cagr": num(t["cagr"], 4),
                        "max_dd": num(t["max_dd"], 4), "total_return": num(t["total_return"], 4),
                        "vol": stats(col)["vol"]})
    a = tests["alpha_vs_SPY"]
    f = cells.trade_frame(tr, costs.BASE)
    c3row = {"window_start": f"{dates[0]:%Y-%m-%d}", "window_end": f"{dates[-1]:%Y-%m-%d}", "sessions": len(dates),
             "sharpe": num(c3["sharpe_net"], 4), "sharpe_lo": num(tests["strategy_sharpe_ci"][0], 4),
             "sharpe_hi": num(tests["strategy_sharpe_ci"][1], 4), "sharpe_gross": num(c3["sharpe_gross"], 4),
             "sharpe_2x_cost": num(c3["sharpe_double_cost"], 4),
             "diff_spy": num(tests["sharpe_minus_SPY"]["point"], 4),
             "diff_spy_lo": num(tests["sharpe_minus_SPY"]["boot_ci"][0], 4),
             "diff_spy_hi": num(tests["sharpe_minus_SPY"]["boot_ci"][1], 4),
             "diff_qqq": num(tests["sharpe_minus_QQQ"]["point"], 4),
             "diff_qqq_lo": num(tests["sharpe_minus_QQQ"]["boot_ci"][0], 4),
             "diff_qqq_hi": num(tests["sharpe_minus_QQQ"]["boot_ci"][1], 4),
             "alpha": num(a["alpha_ann"], 4), "alpha_lo": num(a["alpha_lo"], 4), "alpha_hi": num(a["alpha_hi"], 4),
             "alpha_t": num(a["alpha_t"], 3), "beta": num(a["beta"], 4), "dsr": num(c3["dsr"], 4),
             "alpha_p_holm": num(c3["alpha_p_holm"], 4), "dev_pass": bool(c3["dev_pass"]),
             "trades": int(len(f)), "days_in_market": int(f["date"].nunique()),
             "share_days_in_market": num(f["date"].nunique() / len(dates), 4),
             "trips_per_market_day": num(len(f) / f["date"].nunique(), 3),
             "win_rate": num((f["net"] > 0).mean(), 4), "stop_share": num((f["kind"] == "sl").mean(), 4),
             "mean_weight": num(f["weight"].mean(), 3),
             "share_at_cap": num((f["weight"] >= cells.LEVER_CAP - 1e-12).mean(), 4),
             "gross_bp": num(f["gross"].mean() * 1e4, 3), "net_bp": num(f["net"].mean() * 1e4, 3),
             "cost_bp": num((f["gross"] - f["net"]).mean() * 1e4, 3), "breakeven_bp": num(be["breakeven_bp"], 3),
             "spy_sharpe": num(tab["SPY"]["sharpe"], 4)}
    assert c3row["trades"] == be["trades"] == c3["trades"]["n_trades"]

    years = []
    frame = pd.DataFrame({"c3": net, "spy": bench["SPY"], "qqq": bench["QQQ"]}).iloc[1:]
    for y, gy in frame.groupby(frame.index.year):
        row = {"year": str(y)}
        for c in ("c3", "spy", "qqq"):
            row[f"{c}_ret"] = num((1 + gy[c].fillna(0)).prod() - 1, 5)
            row[f"{c}_sharpe"] = num(bm.sharpe(gy[c].fillna(0).to_numpy()), 3)
        years.append(row)
    assert all(abs(r["c3_sharpe"] - c3["by_year"][r["year"]]) < 1e-3 for r in years), "by-year Sharpe drifted"

    pos = {d: k for k, d in enumerate(dates)}

    def sharpe_at(bp: float) -> float:
        r = np.zeros(len(dates))
        for t in tr:
            r[pos[t.date]] += t.weight * (t.gross - bp * 1e-4)
        return float(bm.sharpe(r[1:]))
    curve = [{"cost_bp": round(x, 2), "sharpe": num(sharpe_at(x), 4)} for x in np.arange(0.0, 3.51, 0.1)]
    assert abs(sharpe_at(be["breakeven_bp"]) - be["spy_dev_sharpe"]) < 1e-3, "break-even does not reproduce"

    out.mkdir(parents=True, exist_ok=True)
    inp = (f"Dev window {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}: SPY minute bars (data/research_liquid cache), "
           "l1_dev.json, l1_dev_daily.parquet, c3_breakeven.json, benchmark daily returns. Nothing after 2022-12-30.")
    files = {"dev_daily": table(daily), "dev_summary": table(summary), "dev_c3": table([c3row]),
             "dev_years": table(years), "dev_trades": table(research_trade_rows(tr, spy)),
             "dev_costcurve": table(curve), "dev_examples": table(example_rows(tr, spy))}
    paths = {}
    for name, t in files.items():
        p = out / f"{name}.json"
        p.write_text(json.dumps({"columns": t["columns"], "rows": t["rows"], "meta": _meta(inp)},
                                separators=(",", ":"), allow_nan=False), encoding="utf-8")
        paths[name] = p
        logger.info("%s: %d rows, %.0f KB", name, len(t["rows"]), p.stat().st_size / 1024)
    return paths


# --------------------------------------------------------------- live side --
def check_action(ev: dict, later: list[dict]) -> str:
    """What happened at one live check, from its own event and the events logged at the same check."""
    d = ev["d"]
    kinds = {e["kind"] for e in later if e["payload"].get("d") == d or e["payload"].get("exit_check") == d}
    if "entry_filled" in kinds:
        return "enter"
    if "entry_unfilled" in kinds or "entry_skipped" in kinds:
        return "enter_unfilled"
    if ev.get("in_position"):
        if "market_exit_sent" in kinds or "exit" in kinds:
            return "exit"
        return "hold"
    if ev.get("enter"):
        return "enter_unfilled"
    price, ub, vw = ev.get("price"), ev.get("ub"), ev.get("vwap")
    if price is not None and ub is not None and price <= ub:
        return "inside"
    return "below_vwap"


def trade_pnl(t: dict) -> float | None:
    from research.liquid.live import report
    if t.get("exit_price") is None:
        return None
    return float(t["qty"] * (t["exit_price"] - t["entry_price"])
                 - report.fees_owed(pd.Timestamp(t["day"]), t["qty"], t["entry_price"], t["exit_price"]))


def _et_clock(ts: str | None) -> str | None:
    if not ts:
        return None
    return pd.Timestamp(ts).tz_convert(ET).strftime("%H:%M:%S") if pd.Timestamp(ts).tzinfo else str(ts)[11:19]


def live_tables(days: list[str], trades: dict[str, list[dict]], events: dict[str, list[dict]],
                reports: dict[str, dict], market: dict[str, dict]) -> dict[str, list[dict]]:
    """Assemble the live tables from what the server holds. Pure: no network, no clock.

    market[day] = {"spy_ret", "qqq_ret", "arrays"} (arrays from `day_arrays`)."""
    trade_rows, day_rows, check_rows = [], [], []
    for day in days:
        rep, evs, mk = reports.get(day, {}), events.get(day, []), market.get(day, {})
        cost = {r["trade_id"]: r for r in rep.get("costs", [])}
        research_entries = set(rep.get("entries", {}).get("research", []))
        live_entries = set(rep.get("entries", {}).get("live", []))
        checks = [e for e in evs if e["kind"] == "check"]
        by_d = {e["payload"]["d"]: e["payload"] for e in checks}
        plan = next((e["payload"] for e in evs if e["kind"] == "plan"), {})
        for e in checks:
            p = e["payload"]
            check_rows.append({"day": day, "d": int(p["d"]), "time": check_time(p["d"]), "price": num(p.get("price"), 4),
                               "ub": num(p.get("ub"), 4), "vwap": num(p.get("vwap"), 4),
                               "stop": num(p.get("stop"), 4), "in_position": bool(p.get("in_position")),
                               "action": check_action(p, evs), "research_enters": int(p["d"]) in research_entries})
        pnl_day, eq_start = 0.0, None
        for n, t in enumerate(sorted(trades.get(day, []), key=lambda r: r["id"]), 1):
            c, chk = cost.get(t["id"], {}), by_d.get(t["entry_check"], {})
            pnl = trade_pnl(t)
            pnl_day += pnl or 0.0
            eq_start = eq_start if eq_start is not None else t["equity"]
            gross = (t["exit_price"] / t["entry_price"] - 1) if t.get("exit_price") else None
            trade_rows.append({
                "trade": f"{day}#{n}", "day": day, "n": n, "check": check_time(t["entry_check"]),
                "entry_time": _et_clock(t.get("entry_filled_at") or t.get("entry_ts")),
                "exit_time": _et_clock(t.get("exit_filled_at") or t.get("exit_ts")),
                "qty": int(t["qty"]), "notional": num(t["qty"] * t["entry_price"], 2), "weight": num(t["weight"], 3),
                "price_at_check": num(t["decision_price"], 4), "ub_at_check": num(chk.get("ub"), 4),
                "vwap_at_check": num(chk.get("vwap"), 4), "limit_price": num(t["limit_price"], 2),
                "entry_price": num(t["entry_price"], 4), "stop_at_entry": num(t["stop_at_entry"], 4),
                "stop_at_exit": num(t.get("stop_at_exit"), 2), "exit_price": num(t.get("exit_price"), 4),
                "exit": t.get("exit_kind"), "exit_check": None if t.get("exit_check") is None
                else check_time(t["exit_check"]),
                "pnl": num(pnl, 2), "gross_bp": num(gross * 1e4 if gross is not None else None, 2),
                "cost_bp": num(c.get("cost_bp"), 3), "exec_entry_bp": num(c.get("exec_entry_bp"), 3),
                "exec_exit_bp": num(c.get("exec_exit_bp"), 3), "fees_bp": num(c.get("fees_bp"), 3),
                "latency_bp": num(c.get("latency_bp"), 3), "model_bp": num(c.get("model_bp"), 3),
                "cost_usable": bool(c.get("usable", False)), "research_match": t["entry_check"] in research_entries})
        research_ret = 0.0
        w = plan.get("weight")
        for r in rep.get("research_trades", []):
            if w:
                t = cells.Trade(pd.Timestamp(day), "C3", 1, float(w), r["entry"], r["exit"], r["kind"], 0, 0)
                research_ret += t.weight * t.net(costs.BASE)
        u = live_entries | research_entries
        saf = rep.get("safety", {})
        day_rows.append({
            "day": day, "trades": len(trades.get(day, [])), "pnl": num(pnl_day, 2),
            "equity_start": num(eq_start, 2), "ret": num(pnl_day / eq_start if eq_start else 0.0, 7),
            "spy_ret": num(mk.get("spy_ret"), 7), "qqq_ret": num(mk.get("qqq_ret"), 7),
            "research_trades": len(rep.get("research_trades", [])), "research_ret": num(research_ret, 7),
            "matched": len(live_entries & research_entries), "jaccard": num(len(live_entries & research_entries) / len(u)
                                                                            if u else None, 4),
            "weight": num(w, 3), "vol": num(plan.get("vol"), 5), "safety_ok": bool(saf.get("ok", True)),
            "max_unprotected_s": num(saf.get("max_unprotected_s"), 1), "kills": len(saf.get("kills", []) or [])})
    return {"days": day_rows, "trades": trade_rows, "checks": check_rows}


def compound(rs: Iterable[float | None]) -> float:
    return float(np.prod([1 + (r or 0.0) for r in rs]) - 1)


def status_row(tables: dict, summary: dict, *, now: datetime, kill_text: str | None, next_session: dict | None,
               equity: float | None, code: dict, host: str, timer_next: str | None) -> dict:
    from research.liquid.live import report
    days, trades = tables["days"], tables["trades"]
    gc, gf, gs = summary.get("g_cost", {}), summary.get("g_fidelity", {}), summary.get("g_safety", {})
    run = gc.get("running", {})
    in_mkt = sum(1 for d in days if d["trades"])
    closed = [t for t in trades if t["pnl"] is not None]
    fid = gf.get("running", {})
    return {
        "updated": now.isoformat(timespec="seconds"), "host": host, "code_sha": (code.get("sha") or "")[:7],
        "code_dirty": bool(code.get("dirty")), "phase": "stopped" if kill_text else ("waiting" if not days else "running"),
        "kill": bool(kill_text), "kill_text": (kill_text or "")[:400] or None,
        "next_session": (next_session or {}).get("day"), "next_open": (next_session or {}).get("open"),
        "timer_next": timer_next, "first_day": days[0]["day"] if days else None,
        "last_day": days[-1]["day"] if days else None, "sessions": len(days), "days_in_market": in_mkt,
        "round_trips": len(trades), "trips_per_market_day": num(len(trades) / in_mkt if in_mkt else None, 3),
        "pnl_total": num(sum(t["pnl"] for t in closed), 2),
        "ret_total": num(compound(d["ret"] for d in days), 6) if days else None,
        "spy_total": num(compound(d["spy_ret"] for d in days), 6) if days else None,
        "qqq_total": num(compound(d["qqq_ret"] for d in days), 6) if days else None,
        "research_total": num(compound(d["research_ret"] for d in days), 6) if days else None,
        "win_rate": num(np.mean([t["pnl"] > 0 for t in closed]) if closed else None, 4),
        "equity": num(equity, 2),
        "breakeven_bp": report.BREAKEVEN_BP, "g_cost": gc.get("status", "pending"),
        "usable_trips": int(gc.get("usable_round_trips", 0)), "cost_mean_bp": num(run.get("cost_mean_bp"), 3),
        "cost_lo_bp": num((run.get("cost_ci95") or [None, None])[0], 3),
        "cost_hi_bp": num((run.get("cost_ci95") or [None, None])[1], 3),
        "latency_mean_bp": num(run.get("latency_mean_bp"), 3), "model_mean_bp": num(run.get("model_mean_bp"), 3),
        "interim_n": report.INTERIM_N, "final_n": report.FINAL_N, "extend_n": report.EXTEND_N,
        "fidelity_bar": report.FIDELITY_BAR, "g_fidelity": gf.get("status", "pending"),
        "jaccard": num(fid.get("jaccard"), 4), "live_entries": fid.get("live"), "research_entries": fid.get("research"),
        "matched_entries": fid.get("matched"), "g_safety": gs.get("status", "pending"),
        "failing_days": len(gs.get("failing_days", []) or [])}


def _timer_next() -> str | None:
    try:
        out = subprocess.run(["systemctl", "show", "c3.timer", "-p", "NextElapseUSecRealtime", "--value"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def market_day(feed: Any, cal: list, day: str, history_sessions: int, symbol: str) -> dict:
    """SPY and QQQ close-to-close returns and SPY's chart arrays for one paper session (SIP, adjusted)."""
    window = [s for s in cal if s.day <= pd.Timestamp(day)][-(history_sessions + 1):]
    m = feed.history(symbol, [(s.day, s.n_bars) for s in window])
    q = feed.history("QQQ", [(s.day, s.n_bars) for s in window[-2:]])
    i = len(m.dates) - 1
    assert f"{m.dates[i]:%Y-%m-%d}" == day, f"last session fetched is {m.dates[i]:%Y-%m-%d}, not {day}"
    assert f"{q.dates[-1]:%Y-%m-%d}" == day and len(q.dates) == 2
    spy_ret = m.close[i, m.n_bars[i] - 1] / m.prev_close[i] - 1
    qqq_ret = q.close[1, q.n_bars[1] - 1] / q.prev_close[1] - 1
    return {"spy_ret": num(spy_ret, 7), "qqq_ret": num(qqq_ret, 7), "arrays": day_arrays(m, i)}


def build_live(now: datetime | None = None) -> tuple[dict, dict]:
    """The two live documents. Server side: reads the ledger, the reports and Alpaca (read-only)."""
    from research.liquid.live import main as M
    from research.liquid.live.ledger import Ledger
    from research.liquid.provenance import git_state

    now = now or datetime.now(ET)
    broker, feed = M._clients()
    led = Ledger(M.LEDGER)
    reports = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(M.REPORTS.glob("20*.json"))}
    first = min(led.days(), default=None)
    start = pd.Timestamp(first) if first else pd.Timestamp(now.date())
    cal = M.sessions(broker, (start - timedelta(days=45)).date(), (now + timedelta(days=10)).date())
    days = [d for d in sorted(reports) if first and d >= first]
    cache = DASH_DIR / "days"
    cache.mkdir(parents=True, exist_ok=True)
    market = {}
    for day in days:
        f = cache / f"{day}.json"
        if f.exists():
            market[day] = json.loads(f.read_text(encoding="utf-8"))
            continue
        try:
            market[day] = market_day(feed, cal, day, M.HISTORY_SESSIONS, M.SYMBOL)
        except Exception:                                              # noqa: BLE001
            logger.exception("minute bars for %s failed; the day shows without its replay", day)
            market[day] = {}
            continue
        f.write_text(json.dumps(market[day]), encoding="utf-8")
    tables = live_tables(days, {d: led.trades(d) for d in days}, {d: led.events(d) for d in days}, reports, market)
    summary_path = M.LIVE_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    nxt = next((s for s in cal if s.open > now), None)
    kill = M.KILL.read_text(encoding="utf-8") if M.KILL.exists() else None
    try:
        equity = float(broker.account().equity)
    except Exception:                                                  # noqa: BLE001
        equity = None
    status = status_row(tables, summary, now=now, kill_text=kill,
                        next_session={"day": f"{nxt.day:%Y-%m-%d}", "open": nxt.open.isoformat()} if nxt else None,
                        equity=equity, code=git_state(), host=socket.gethostname(), timer_next=_timer_next())
    live = {"version": 1, "status": table([status]), "days": table(tables["days"] or [], DAY_COLS),
            "trades": table(tables["trades"], TRADE_COLS), "checks": table(tables["checks"], CHECK_COLS)}
    kept = [d for d in days if market[d].get("arrays")][-INTRADAY_KEEP:]        # keeps the Drive file small
    intraday = {"version": 1, "intraday": table([{"day": d, **market[d]["arrays"]} for d in kept], INTRADAY_COLS)}
    return live, intraday


DAY_COLS = ["day", "trades", "pnl", "equity_start", "ret", "spy_ret", "qqq_ret", "research_trades", "research_ret",
            "matched", "jaccard", "weight", "vol", "safety_ok", "max_unprotected_s", "kills"]
TRADE_COLS = ["trade", "day", "n", "check", "entry_time", "exit_time", "qty", "notional", "weight", "price_at_check",
              "ub_at_check", "vwap_at_check", "limit_price", "entry_price", "stop_at_entry", "stop_at_exit",
              "exit_price", "exit", "exit_check", "pnl", "gross_bp", "cost_bp", "exec_entry_bp", "exec_exit_bp",
              "fees_bp", "latency_bp", "model_bp", "cost_usable", "research_match"]
CHECK_COLS = ["day", "d", "time", "price", "ub", "vwap", "stop", "in_position", "action", "research_enters"]
INTRADAY_KEEP = 40                         # sessions of minute bars in c3_intraday.json (~4 KB each, compressed)
INTRADAY_COLS = ["day", "n_bars", "open0", "prev_close", "close", "ub", "lb", "vwap"]


def publish(live: dict, intraday: dict, cfg_path: Path = PUBLISH_CFG) -> None:
    """Overwrite the two Drive files with the server's own service account (metadata-server token)."""
    import requests
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tok = requests.get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                       headers={"Metadata-Flavor": "Google"}, timeout=10).json()["access_token"]
    for fid, doc in ((cfg["live_file_id"], live), (cfg["intraday_file_id"], intraday)):
        r = requests.patch(f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=media",
                           headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                           data=wrap(doc).encode("ascii"), timeout=60)
        r.raise_for_status()
        logger.info("published %s (%d bytes)", fid, len(r.request.body or b""))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", type=Path, metavar="OUT_DIR")
    g.add_argument("--live", action="store_true")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    if a.dev:
        build_dev(a.dev)
        return 0
    live, intraday = build_live()
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    (DASH_DIR / "c3_live.json").write_text(json.dumps(live, indent=1), encoding="utf-8")
    (DASH_DIR / "c3_intraday.json").write_text(json.dumps(intraday), encoding="utf-8")
    st = live["status"]["rows"][0]
    logger.info("live: %d sessions, %d trips, phase %s", st["sessions"], st["round_trips"], st["phase"])
    if a.publish:
        publish(live, intraday)
    return 0


if __name__ == "__main__":
    sys.exit(main())
