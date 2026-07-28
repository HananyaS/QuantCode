"""generate_live_report.py — build the live paper-trading dashboard from
the state ledgers (data/live_state_kelly.db, data/live_state_linear.db).

Runs in CI after each real trading run and writes a single self-contained
HTML page (no external scripts/styles — it must render offline and never
break on a CDN) that GitHub Pages serves at /QuantCode/live-report/.

Usage:
    python scripts/generate_live_report.py                 # write the HTML report
    python scripts/generate_live_report.py --summary       # one line per strategy (for notifications)
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from utils.state_store import StateStore

_STRATEGY_BLURBS = {
    "Kelly": "Fractional-Kelly leverage timing — EWMA drift/vol + trend regime on QQQ, expressed via QQQ/QLD/TQQQ",
    "Linear": "SMA/momentum/VIX timing signal with volatility-targeted sizing, expressed via QQQ/QLD/TQQQ",
}
_STRATEGY_COLORS = {"Kelly": "#6366f1", "Linear": "#14b8a6"}
_FALLBACK_COLOR = "#f59e0b"

# True starting capital of each paper account (both funded with exactly
# $25,000 on 2026-07-28). Total return is anchored HERE, not to the
# earliest surviving snapshot: same-session re-runs upsert the day's
# snapshot row, and on inception day that overwrote the $25,000 base with
# the post-gain equity -- turning a +1.9% track record into a reported
# -0.29%. Update these only if an account is reset or re-funded.
_INCEPTION_EQUITY = {"Kelly": 25_000.0, "Linear": 25_000.0}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def strategy_summary(name: str, store: StateStore) -> dict:
    """Aggregate one strategy's ledger into the numbers the report shows.

    Returns-since-inception is anchored to _INCEPTION_EQUITY (the known
    starting capital), falling back to the first snapshot only for
    strategies without an entry there — see the note on _INCEPTION_EQUITY
    for why the earliest snapshot row cannot be trusted as the base.
    This is the honest live-forward track record (no backtest numbers).
    """
    snaps = store.snapshots()
    orders = store.all_orders()

    if len(snaps) == 0:
        return {
            "name": name, "equity": None, "cash": None, "positions": {},
            "day_return_pct": None, "total_return_pct": None,
            "n_sessions": 0, "n_orders": int(len(orders)),
            "snapshots": snaps, "orders": orders, "last_session": None,
        }

    equity = float(snaps["equity"].iloc[-1])
    base = _INCEPTION_EQUITY.get(name, float(snaps["equity"].iloc[0]))
    total_return_pct = 100.0 * (equity / base - 1.0) if base > 0 else None
    day_return_pct = None
    if len(snaps) >= 2:
        prev = float(snaps["equity"].iloc[-2])
        if prev > 0:
            day_return_pct = 100.0 * (equity / prev - 1.0)

    return {
        "name": name,
        "equity": equity,
        "cash": float(snaps["cash"].iloc[-1]),
        "positions": snaps["positions"].iloc[-1],
        "day_return_pct": day_return_pct,
        "total_return_pct": total_return_pct,
        "n_sessions": int(len(snaps)),
        "n_orders": int(len(orders)),
        "snapshots": snaps,
        "orders": orders,
        "last_session": str(snaps["run_date"].iloc[-1]),
    }


def _text_summary(summaries: List[dict]) -> str:
    """Compact per-strategy lines for the push notification."""
    lines = []
    for s in summaries:
        if s["equity"] is None:
            lines.append(f"{s['name']}: no runs recorded yet")
            continue
        ret = f"{s['total_return_pct']:+.2f}%" if s["total_return_pct"] is not None else "n/a"
        last = s["last_session"]
        day_orders = s["orders"][s["orders"]["run_date"] == last] if len(s["orders"]) else s["orders"]
        if len(day_orders):
            acts = ", ".join(
                f"{o.side} {int(o.qty)} {o.ticker}" for o in day_orders.itertuples()
            )
        else:
            acts = "no orders"
        lines.append(f"{s['name']}: ${s['equity']:,.2f} ({ret} total) | {last}: {acts}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SVG equity chart (self-contained — no JS charting library)
# ---------------------------------------------------------------------------

def _equity_chart_svg(summaries: List[dict], width: int = 860, height: int = 300) -> str:
    series = [
        (s["name"], s["snapshots"]) for s in summaries
        if s["snapshots"] is not None and len(s["snapshots"]) > 0
    ]
    if not series:
        return "<p class='muted'>No equity history yet — the chart appears after the first run.</p>"

    pad_l, pad_r, pad_t, pad_b = 64, 16, 16, 34
    all_dates = sorted({d for _, snaps in series for d in snaps["run_date"]})
    x_of = {d: (pad_l + (width - pad_l - pad_r) * (i / max(1, len(all_dates) - 1)))
            for i, d in enumerate(all_dates)}
    values = [v for _, snaps in series for v in snaps["equity"]]
    lo, hi = min(values), max(values)
    span = (hi - lo) or max(hi * 0.02, 1.0)
    lo, hi = lo - 0.08 * span, hi + 0.08 * span

    def y_of(v: float) -> float:
        return pad_t + (height - pad_t - pad_b) * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Equity by session" preserveAspectRatio="xMidYMid meet">']
    for i in range(5):  # horizontal grid + $ labels
        gv = lo + (hi - lo) * i / 4
        gy = y_of(gv)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" class="tick" text-anchor="end">${gv:,.0f}</text>')
    n_x = len(all_dates)
    label_idx = sorted({0, n_x - 1, n_x // 2} if n_x > 2 else set(range(n_x)))
    for i in label_idx:
        d = all_dates[i]
        parts.append(f'<text x="{x_of[d]:.1f}" y="{height - 10}" class="tick" text-anchor="middle">{d}</text>')

    for name, snaps in series:
        color = _STRATEGY_COLORS.get(name, _FALLBACK_COLOR)
        pts = [(x_of[d], y_of(float(v))) for d, v in zip(snaps["run_date"], snaps["equity"])]
        if len(pts) > 1:
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" '
                         f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        for (x, y), (d, v) in zip(pts, zip(snaps["run_date"], snaps["equity"])):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}">'
                         f'<title>{name} — {d}: ${float(v):,.2f}</title></circle>')
    parts.append("</svg>")
    legend = "".join(
        f'<span class="legend-item"><span class="swatch" '
        f'style="background:{_STRATEGY_COLORS.get(n, _FALLBACK_COLOR)}"></span>{html.escape(n)}</span>'
        for n, _ in series
    )
    return f'<div class="legend">{legend}</div><div class="chart">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _fmt_money(v: Optional[float]) -> str:
    return "—" if v is None else f"${v:,.2f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    cls = "pos" if v >= 0 else "neg"
    return f'<span class="{cls}">{v:+.2f}%</span>'


def _positions_chips(positions: dict) -> str:
    if not positions:
        return '<span class="chip flat">100% cash</span>'
    return "".join(
        f'<span class="chip">{html.escape(t)} <b>${float(v):,.0f}</b></span>'
        for t, v in sorted(positions.items(), key=lambda kv: -float(kv[1]))
    )


def _orders_table(orders: pd.DataFrame, limit: int = 60) -> str:
    if len(orders) == 0:
        return '<p class="muted">No orders submitted yet.</p>'
    rows = []
    shown = orders.sort_values("id", ascending=False).head(limit)
    for o in shown.itertuples():
        side_cls = "buy" if o.side == "buy" else "sell"
        status = str(o.status).replace("OrderStatus.", "").lower()
        submitted = str(o.submitted_at)[:16].replace("T", " ")
        rows.append(
            f"<tr><td>{html.escape(str(o.run_date))}</td>"
            f'<td><span class="side {side_cls}">{o.side.upper()}</span></td>'
            f"<td>{int(o.qty)}</td><td>{html.escape(str(o.ticker))}</td>"
            f"<td class='muted'>{html.escape(status)}</td>"
            f"<td class='muted'>{html.escape(submitted)} UTC</td></tr>"
        )
    note = (f'<p class="muted">Showing latest {limit} of {len(orders)} orders.</p>'
            if len(orders) > limit else "")
    return (
        '<table><thead><tr><th>Session</th><th>Side</th><th>Qty</th>'
        "<th>Ticker</th><th>Status</th><th>Submitted</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>{note}'
    )


def _strategy_section(s: dict) -> str:
    color = _STRATEGY_COLORS.get(s["name"], _FALLBACK_COLOR)
    blurb = _STRATEGY_BLURBS.get(s["name"], "")
    if s["equity"] is None:
        body = '<p class="muted">Awaiting first run — no ledger entries yet.</p>'
    else:
        body = f"""
      <div class="stats">
        <div class="stat"><div class="label">Equity</div><div class="value">{_fmt_money(s["equity"])}</div></div>
        <div class="stat"><div class="label">Total return</div><div class="value">{_fmt_pct(s["total_return_pct"])}</div></div>
        <div class="stat"><div class="label">Last session</div><div class="value">{_fmt_pct(s["day_return_pct"])}</div></div>
        <div class="stat"><div class="label">Cash</div><div class="value">{_fmt_money(s["cash"])}</div></div>
        <div class="stat"><div class="label">Sessions</div><div class="value">{s["n_sessions"]}</div></div>
        <div class="stat"><div class="label">Orders</div><div class="value">{s["n_orders"]}</div></div>
      </div>
      <div class="positions"><span class="label">Current positions</span>{_positions_chips(s["positions"])}</div>
      <h3>Order log</h3>
      {_orders_table(s["orders"])}"""
    return f"""
    <section class="card">
      <div class="card-head" style="border-left: 4px solid {color}">
        <h2>{html.escape(s["name"])}</h2>
        <p class="blurb">{html.escape(blurb)}</p>
      </div>
      {body}
    </section>"""


def build_report_html(stores: Dict[str, StateStore]) -> str:
    summaries = [strategy_summary(name, store) for name, store in stores.items()]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = "".join(_strategy_section(s) for s in summaries)
    chart = _equity_chart_svg(summaries)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantCode — Live Trading Report</title>
<style>
:root {{
  --bg: #f6f7fb; --card: #ffffff; --ink: #1a1d29; --muted: #6b7280;
  --border: #e5e7eb; --grid: #e5e7eb; --accent: #6366f1;
  --pos: #059669; --neg: #dc2626; --chip: #eef0f6;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0e1117; --card: #161b26; --ink: #e6e8ee; --muted: #9aa1b2;
    --border: #262c3a; --grid: #262c3a; --chip: #1f2534;
    --pos: #34d399; --neg: #f87171;
  }}
}}
* {{ box-sizing: border-box; margin: 0; }}
body {{
  background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  padding: 32px 16px 48px;
}}
.wrap {{ max-width: 940px; margin: 0 auto; }}
header h1 {{ font-size: 26px; letter-spacing: -0.02em; }}
header .sub {{ color: var(--muted); margin-top: 4px; }}
.badge {{
  display: inline-block; font-size: 12px; font-weight: 600; letter-spacing: 0.04em;
  color: var(--accent); border: 1px solid var(--accent); border-radius: 999px;
  padding: 2px 10px; margin-bottom: 10px; text-transform: uppercase;
}}
.card {{
  background: var(--card); border: 1px solid var(--border); border-radius: 14px;
  padding: 22px 24px; margin-top: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
.card-head {{ padding-left: 12px; margin-bottom: 16px; }}
.card-head h2 {{ font-size: 19px; }}
.blurb {{ color: var(--muted); font-size: 13.5px; margin-top: 2px; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; }}
.stat {{ background: var(--chip); border-radius: 10px; padding: 10px 14px; }}
.label {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
.value {{ font-size: 19px; font-weight: 650; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.pos {{ color: var(--pos); }} .neg {{ color: var(--neg); }}
.positions {{ margin: 16px 0 4px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.chip {{
  background: var(--chip); border-radius: 999px; padding: 4px 12px; font-size: 13.5px;
  font-variant-numeric: tabular-nums;
}}
.chip.flat {{ color: var(--muted); }}
h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em;
     color: var(--muted); margin: 18px 0 8px; }}
.table-scroll, table {{ width: 100%; }}
table {{ border-collapse: collapse; font-size: 13.5px; }}
th {{ text-align: left; color: var(--muted); font-size: 11.5px; text-transform: uppercase;
     letter-spacing: 0.05em; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }}
tr:last-child td {{ border-bottom: none; }}
.side {{ font-size: 11.5px; font-weight: 700; letter-spacing: 0.05em; border-radius: 6px; padding: 2px 8px; }}
.side.buy {{ color: var(--pos); background: color-mix(in srgb, var(--pos) 12%, transparent); }}
.side.sell {{ color: var(--neg); background: color-mix(in srgb, var(--neg) 12%, transparent); }}
.muted {{ color: var(--muted); }}
.chart svg {{ width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.tick {{ fill: var(--muted); font-size: 11px; }}
.legend {{ display: flex; gap: 18px; margin-bottom: 6px; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--muted); }}
.swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
footer {{ color: var(--muted); font-size: 12.5px; margin-top: 26px; text-align: center; }}
footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="badge">Paper trading</div>
    <h1>QuantCode — Live Trading Report</h1>
    <p class="sub">Daily automated leverage-timing on the QQQ family (QQQ / QLD / TQQQ), two independent
    strategies on separate Alpaca paper accounts. Decisions from completed daily closes; orders placed
    by the scheduled GitHub Actions pipeline.</p>
  </header>

  <section class="card">
    <div class="card-head" style="border-left: 4px solid var(--accent)">
      <h2>Equity by session</h2>
      <p class="blurb">Account equity recorded at each run, since inception.</p>
    </div>
    {chart}
  </section>

  {sections}

  <footer>
    Generated {generated} ·
    <a href="https://github.com/HananyaS/QuantCode">HananyaS/QuantCode</a> ·
    Paper trading only — not investment advice.
  </footer>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kelly-db", default="data/live_state_kelly.db")
    parser.add_argument("--linear-db", default="data/live_state_linear.db")
    parser.add_argument("--output", default="docs/live-report/index.html")
    parser.add_argument("--summary", action="store_true",
                        help="Print a compact text summary (for notifications) instead of writing HTML.")
    args = parser.parse_args()

    stores = {
        "Kelly": StateStore(db_path=args.kelly_db),
        "Linear": StateStore(db_path=args.linear_db),
    }

    if args.summary:
        print(_text_summary([strategy_summary(n, s) for n, s in stores.items()]))
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report_html(stores), encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
