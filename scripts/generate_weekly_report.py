"""generate_weekly_report.py — build (and optionally email) the weekly
performance digest from the live-trading state ledgers.

Runs in CI every Saturday morning (after Friday's close): summarizes the
Mon-Fri week that just ended — weekly return per strategy, daily equity
path, every order executed, current positions — as a designed HTML email.

Email-client constraints shape the markup: Gmail strips <svg> and
<script>, and support for <style> blocks is inconsistent, so the layout
is table-based with fully inline styles and no external assets. The same
HTML is archived to docs/live-report/weekly/ so GitHub Pages keeps a
browsable history.

Usage:
    python scripts/generate_weekly_report.py                      # write docs/live-report/weekly/
    python scripts/generate_weekly_report.py --send               # also email it (needs env creds)
    python scripts/generate_weekly_report.py --output out.html    # custom path

--send reads GMAIL_ADDRESS and GMAIL_APP_PASSWORD from the environment
(repo secrets in CI) and sends via Gmail SMTP over SSL.
"""
from __future__ import annotations

import argparse
import html
import os
import smtplib
import sys
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.generate_live_report import _INCEPTION_EQUITY, _STRATEGY_BLURBS
from utils.state_store import StateStore

_DASHBOARD_URL = "https://hananyas.github.io/QuantCode/live-report/"
_STRATEGY_ACCENTS = {"Kelly": "#6366f1", "Linear": "#14b8a6"}
_POS = "#059669"
_NEG = "#dc2626"
_INK = "#1a1d29"
_MUTED = "#6b7280"
_BORDER = "#e5e7eb"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def week_summary(name: str, store: StateStore) -> dict:
    """Summarize the Mon-Fri calendar week containing the LAST snapshot.

    Weekly return is measured from the last snapshot BEFORE that week
    (i.e. the prior Friday's equity); on the first-ever week it falls
    back to the known inception capital — never to the week's own first
    row, which may already include gains (see _INCEPTION_EQUITY's note).
    """
    snaps = store.snapshots()
    orders = store.all_orders()

    if len(snaps) == 0:
        return {
            "name": name, "week_start": None, "week_end": None,
            "week_return_pct": None, "total_return_pct": None,
            "equity": None, "daily": [], "orders": orders.iloc[0:0],
            "positions": {},
        }

    last_day = date.fromisoformat(str(snaps["run_date"].iloc[-1]))
    monday = last_day - timedelta(days=last_day.weekday())
    in_week = snaps[snaps["run_date"] >= str(monday)].reset_index(drop=True)
    before = snaps[snaps["run_date"] < str(monday)]

    base = (
        float(before["equity"].iloc[-1]) if len(before)
        else _INCEPTION_EQUITY.get(name, float(in_week["equity"].iloc[0]))
    )
    equity = float(in_week["equity"].iloc[-1])
    inception_base = _INCEPTION_EQUITY.get(name)

    daily: List[dict] = []
    prev = base
    for row in in_week.itertuples():
        eq = float(row.equity)
        daily.append({
            "date": str(row.run_date),
            "equity": eq,
            "day_pct": 100.0 * (eq / prev - 1.0) if prev > 0 else None,
        })
        prev = eq

    return {
        "name": name,
        "week_start": str(monday),
        "week_end": str(in_week["run_date"].iloc[-1]),
        "week_return_pct": 100.0 * (equity / base - 1.0) if base > 0 else None,
        "total_return_pct": (
            100.0 * (equity / inception_base - 1.0) if inception_base else None
        ),
        "equity": equity,
        "daily": daily,
        "orders": orders[orders["run_date"] >= str(monday)].reset_index(drop=True),
        "positions": in_week["positions"].iloc[-1] if "positions" in in_week else {},
    }


# ---------------------------------------------------------------------------
# Email-safe HTML (tables + inline styles only)
# ---------------------------------------------------------------------------

def _pct_html(v: Optional[float], size: str = "15px", bold: bool = False) -> str:
    if v is None:
        return f'<span style="color:{_MUTED};font-size:{size}">—</span>'
    color = _POS if v >= 0 else _NEG
    weight = "700" if bold else "600"
    return (f'<span style="color:{color};font-size:{size};font-weight:{weight}">'
            f"{v:+.2f}%</span>")


def _money(v: Optional[float]) -> str:
    return "—" if v is None else f"${v:,.2f}"


def _daily_rows(s: dict) -> str:
    rows = []
    for d in s["daily"]:
        rows.append(
            f'<tr>'
            f'<td style="padding:7px 12px;border-bottom:1px solid {_BORDER};'
            f'font-size:13px;color:{_INK}">{d["date"]}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid {_BORDER};'
            f'font-size:13px;color:{_INK};text-align:right">{_money(d["equity"])}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid {_BORDER};'
            f'text-align:right;font-size:13px">{_pct_html(d["day_pct"], "13px")}</td>'
            f'</tr>'
        )
    return "".join(rows)


def _orders_rows(orders: pd.DataFrame) -> str:
    if len(orders) == 0:
        return (f'<tr><td colspan="4" style="padding:10px 12px;font-size:13px;'
                f'color:{_MUTED}">No orders this week.</td></tr>')
    rows = []
    for o in orders.itertuples():
        side = str(o.side).upper()
        side_color = _POS if side == "BUY" else _NEG
        status = str(o.status).replace("OrderStatus.", "").lower()
        rows.append(
            f'<tr>'
            f'<td style="padding:7px 12px;border-bottom:1px solid {_BORDER};'
            f'font-size:13px;color:{_INK}">{html.escape(str(o.run_date))}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid {_BORDER};'
            f'font-size:13px;font-weight:700;color:{side_color}">{side}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid {_BORDER};'
            f'font-size:13px;color:{_INK}">{int(o.qty)} {html.escape(str(o.ticker))}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid {_BORDER};'
            f'font-size:13px;color:{_MUTED}">{html.escape(status)}</td>'
            f'</tr>'
        )
    return "".join(rows)


def _positions_text(positions: dict) -> str:
    if not positions:
        return f'<span style="color:{_MUTED}">100% cash</span>'
    parts = [f"{html.escape(t)} ${float(v):,.0f}"
             for t, v in sorted(positions.items(), key=lambda kv: -float(kv[1]))]
    return " &nbsp;·&nbsp; ".join(parts)


def _strategy_block(s: dict) -> str:
    accent = _STRATEGY_ACCENTS.get(s["name"], "#f59e0b")
    blurb = _STRATEGY_BLURBS.get(s["name"], "")
    if s["week_end"] is None:
        body = (f'<p style="margin:8px 0 0;font-size:13px;color:{_MUTED}">'
                f'No activity recorded yet.</p>')
        return _card(s["name"], accent, blurb, body)

    body = f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px">
      <tr>
        <td style="width:34%;padding:10px 12px;background:#f3f4f8;border-radius:8px">
          <div style="font-size:11px;color:{_MUTED};text-transform:uppercase;letter-spacing:0.05em">Week</div>
          <div style="margin-top:2px">{_pct_html(s["week_return_pct"], "20px", bold=True)}</div>
        </td>
        <td style="width:4%"></td>
        <td style="width:30%;padding:10px 12px;background:#f3f4f8;border-radius:8px">
          <div style="font-size:11px;color:{_MUTED};text-transform:uppercase;letter-spacing:0.05em">Equity</div>
          <div style="margin-top:2px;font-size:17px;font-weight:700;color:{_INK}">{_money(s["equity"])}</div>
        </td>
        <td style="width:4%"></td>
        <td style="width:28%;padding:10px 12px;background:#f3f4f8;border-radius:8px">
          <div style="font-size:11px;color:{_MUTED};text-transform:uppercase;letter-spacing:0.05em">All-time</div>
          <div style="margin-top:2px">{_pct_html(s["total_return_pct"], "17px", bold=True)}</div>
        </td>
      </tr>
    </table>

    <div style="margin:16px 0 6px;font-size:12px;font-weight:700;color:{_MUTED};
                text-transform:uppercase;letter-spacing:0.05em">Daily equity</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
           style="border:1px solid {_BORDER};border-radius:8px">
      <tr>
        <td style="padding:7px 12px;font-size:11px;color:{_MUTED};text-transform:uppercase">Session</td>
        <td style="padding:7px 12px;font-size:11px;color:{_MUTED};text-transform:uppercase;text-align:right">Equity</td>
        <td style="padding:7px 12px;font-size:11px;color:{_MUTED};text-transform:uppercase;text-align:right">Change</td>
      </tr>
      {_daily_rows(s)}
    </table>

    <div style="margin:16px 0 6px;font-size:12px;font-weight:700;color:{_MUTED};
                text-transform:uppercase;letter-spacing:0.05em">Orders this week</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
           style="border:1px solid {_BORDER};border-radius:8px">
      {_orders_rows(s["orders"])}
    </table>

    <p style="margin:14px 0 0;font-size:13px;color:{_INK}">
      <span style="color:{_MUTED}">Positions now:</span> {_positions_text(s["positions"])}
    </p>"""
    return _card(s["name"], accent, blurb, body)


def _card(title: str, accent: str, blurb: str, body: str) -> str:
    return f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
           style="background:#ffffff;border:1px solid {_BORDER};border-radius:12px;margin-top:18px">
      <tr><td style="padding:20px 22px">
        <div style="border-left:4px solid {accent};padding-left:12px">
          <div style="font-size:17px;font-weight:700;color:{_INK}">{html.escape(title)}</div>
          <div style="font-size:12.5px;color:{_MUTED};margin-top:2px">{html.escape(blurb)}</div>
        </div>
        {body}
      </td></tr>
    </table>"""


def build_weekly_email_html(stores: Dict[str, StateStore]) -> str:
    summaries = [week_summary(name, store) for name, store in stores.items()]
    active = [s for s in summaries if s["week_end"] is not None]
    week_start = min((s["week_start"] for s in active), default=None)
    week_end = max((s["week_end"] for s in active), default=None)
    week_label = f"{week_start} → {week_end}" if week_start else "no activity"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks = "".join(_strategy_block(s) for s in summaries)

    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>QuantCode Weekly Report</title></head>
<body style="margin:0;padding:0;background:#f0f1f6">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f0f1f6">
    <tr><td align="center" style="padding:28px 12px">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;
             font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif">

        <tr><td style="background:#14172b;border-radius:12px;padding:24px 26px">
          <div style="font-size:11px;font-weight:700;color:#8b93f8;text-transform:uppercase;
                      letter-spacing:0.1em">Paper trading · weekly digest</div>
          <div style="font-size:23px;font-weight:800;color:#ffffff;margin-top:6px">
            QuantCode Weekly Report</div>
          <div style="font-size:13.5px;color:#a3aac2;margin-top:4px">Trading week {week_label}</div>
        </td></tr>

        <tr><td>{blocks}</td></tr>

        <tr><td align="center" style="padding:22px 0 6px">
          <a href="{_DASHBOARD_URL}"
             style="display:inline-block;background:#6366f1;color:#ffffff;text-decoration:none;
                    font-size:14px;font-weight:700;padding:11px 26px;border-radius:8px">
            Open the live dashboard</a>
        </td></tr>

        <tr><td align="center" style="padding:14px 0 4px">
          <div style="font-size:11.5px;color:{_MUTED}">
            Generated {generated} · QQQ-family leverage timing on Alpaca paper accounts<br>
            Automated by GitHub Actions · Paper trading only — not investment advice.
          </div>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------

def send_email(html_body: str, subject: str, to_addr: str) -> None:
    """Send via Gmail SMTP/SSL using GMAIL_ADDRESS + GMAIL_APP_PASSWORD env
    vars (repo secrets in CI). App passwords are revocable, mail-only
    credentials — see https://myaccount.google.com/apppasswords.
    """
    sender = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    assert sender and app_password, (
        "GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set -- cannot send email"
    )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"QuantCode Bot <{sender}>"
    msg["To"] = to_addr
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(sender, app_password)
        server.sendmail(sender, [to_addr], msg.as_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kelly-db", default="data/live_state_kelly.db")
    parser.add_argument("--linear-db", default="data/live_state_linear.db")
    parser.add_argument("--archive-dir", default="docs/live-report/weekly")
    parser.add_argument("--output", default=None,
                        help="Extra path to write the HTML to (e.g. for the email step).")
    parser.add_argument("--send", action="store_true",
                        help="Email the report (GMAIL_ADDRESS/GMAIL_APP_PASSWORD env, "
                             "recipient via REPORT_EMAIL_TO or GMAIL_ADDRESS).")
    args = parser.parse_args()

    stores = {
        "Kelly": StateStore(db_path=args.kelly_db),
        "Linear": StateStore(db_path=args.linear_db),
    }
    html_body = build_weekly_email_html(stores)

    summaries = [week_summary(n, s) for n, s in stores.items()]
    week_end = max((s["week_end"] for s in summaries if s["week_end"]), default=None)
    label = week_end or date.today().isoformat()

    archive = Path(args.archive_dir)
    archive.mkdir(parents=True, exist_ok=True)
    (archive / f"{label}.html").write_text(html_body, encoding="utf-8")
    (archive / "latest.html").write_text(html_body, encoding="utf-8")
    print(f"Weekly report archived to {archive}/{label}.html (+ latest.html)")

    if args.output:
        Path(args.output).write_text(html_body, encoding="utf-8")

    if args.send:
        to_addr = os.environ.get("REPORT_EMAIL_TO") or os.environ.get("GMAIL_ADDRESS")
        parts = []
        for s in summaries:
            if s["week_return_pct"] is not None:
                parts.append(f"{s['name']} {s['week_return_pct']:+.2f}%")
        subject = f"QuantCode weekly report ({label})"
        if parts:
            subject += " -- " + ", ".join(parts)
        send_email(html_body, subject, to_addr)
        print(f"Weekly report emailed to {to_addr}")


if __name__ == "__main__":
    main()
