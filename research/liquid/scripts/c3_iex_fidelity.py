"""Calibrate G-fidelity and choose the live feed: how often does live C3 trade as research C3? (G3 B2, W3)

For each valid 2022 dev session (inside the dev window L1 already read), the real
Runner is replayed with:
- the plan from SIP history (the research cache), as live;
- today's bars built exactly as live builds them (`today_from_rows`), from one of:
  - "iex":    IEX bars only;
  - "hybrid": SIP bars for minutes at least 16 minutes old (free on the basic
              plan), IEX for the most recent 16;
- a decision bar missing after the feed wait either skips the check ("skip") or
  is carried from the previous minute ("carry");
- fills from SIP bars by the research rules (`replay.BarBroker`).
Entries are compared with `cells.c3` on SIP. The target is research's own decision,
not returns, so this chooses an implementation, not a strategy. Market-data reads only.

The pooled Jaccard share |live ∩ research| / |live ∪ research| of (day, entry check),
with a day-clustered bootstrap CI, sets the G-fidelity bar in PAPER_C3 §4.
Writes research/liquid/results/c3_iex_fidelity.json.

    python -m research.liquid.scripts.c3_iex_fidelity [--env-pair C3]

--env-pair names the .env key pair used for the data reads (default C3; the first
calibration ran before the C3 account existed, with another pair: LOG §3).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values

from research.liquid import calendar as cal
from research.liquid import cells
from research.liquid.data import FIELDS, load
from research.liquid.live import replay, signal
from research.liquid.live.feed import SIP_DELAY_MIN, AlpacaFeed, merge_today, today_from_rows
from research.liquid.live.ledger import Ledger
from research.liquid.live.runner import ET, Runner, Session
from research.liquid.provenance import git_state

OUT = Path("research/liquid/results/c3_iex_fidelity.json")
YEAR = 2022


def minute_label(j: int) -> str:
    return f"{9 + (30 + j) // 60:02d}:{(30 + j) % 60:02d}"


class ReplayFeed:
    """Today's bars as the live feed would have built them at each check."""

    def __init__(self, m, iex: pd.DataFrame, mode: str) -> None:
        self.m, self.mode = m, mode
        self.iex = {d: g for d, g in iex.groupby("date")}
        self.row = {d: i for i, d in enumerate(m.dates)}

    def sip_rows(self, i: int, upto: int) -> pd.DataFrame:
        j = np.arange(upto + 1)
        df = pd.DataFrame({f: getattr(self.m, f)[i, j] for f in FIELDS})
        df["minute_et"] = [minute_label(x) for x in j]
        return df

    def today(self, symbol, day, upto):
        day = pd.Timestamp(day)
        iex = self.iex.get(day, pd.DataFrame(columns=["minute_et", *FIELDS]))
        if self.mode == "iex":
            return today_from_rows(iex, upto)
        return today_from_rows(merge_today(self.sip_rows(self.row[day], upto), iex, upto - SIP_DELAY_MIN), upto)


def keys(pair: str) -> tuple[str, str]:
    env = dotenv_values(next(p / ".env" for p in Path.cwd().resolve().parents if (p / ".env").is_file()))
    return env[f"ALPACA_API_KEY_{pair}"], env[f"ALPACA_SECRET_KEY_{pair}"]


def cluster_ci(days: list[str], live: set, res: set, n_boot: int = 2000, seed: int = 17) -> list[float]:
    a = np.array([sum(1 for x in live & res if x[0] == d) for d in days])
    u = np.array([sum(1 for x in live | res if x[0] == d) for d in days])
    idx = np.random.default_rng(seed).integers(0, len(days), (n_boot, len(days)))
    return np.quantile(a[idx].sum(1) / np.maximum(u[idx].sum(1), 1), [0.025, 0.975]).tolist()


def replay_with(m, i, feed, missing_bar: str) -> tuple[list, int]:
    day = m.dates[i]
    n = int(m.n_bars[i])
    open_at = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    clock = replay.SimClock(open_at - timedelta(minutes=2))
    b = replay.BarBroker(clock=clock, m=m, i=i, open_at=open_at)
    led = Ledger(Path(":memory:"))
    plan = signal.day_plan(m.slice(m.dates[i - replay.HISTORY], m.dates[i - 1]), day, n)
    Runner(b, feed, clock, led, poll_s=30.0, missing_bar=missing_bar).run_session(Session(day, open_at, n), plan)
    return [int(t["entry_check"]) for t in led.trades()], \
        len(led.events(kind="check_skipped")) + len(led.events(kind="decision_bar_carried"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-pair", default="C3")
    a = ap.parse_args()
    m = load("SPY", cal.DEV_START, cal.DEV_END)
    rows = [i for i in replay.valid_rows(m) if m.dates[i].year == YEAR]
    days = [str(m.dates[i].date()) for i in rows]
    feed = AlpacaFeed(*keys(a.env_pair))
    start = pd.Timestamp(m.dates[rows[0]]).tz_localize(ET) + pd.Timedelta(hours=9, minutes=30)
    end = pd.Timestamp(m.dates[rows[-1]]).tz_localize(ET) + pd.Timedelta(hours=16)
    iex = feed._bars("SPY", start.to_pydatetime(), end.to_pydatetime(), "iex", "all")
    tr = cells.c3(m)
    res = {(str(m.dates[i].date()), e) for i in rows for e, _, _ in replay.research_trades(m, i, tr)}
    variants = {}
    for mode in ("iex", "hybrid"):
        f = ReplayFeed(m, iex, mode)
        for missing in ("skip", "carry"):
            live, n_missing = set(), 0
            for i in rows:
                ent, k = replay_with(m, i, f, missing)
                live |= {(str(m.dates[i].date()), e) for e in ent}
                n_missing += k
            variants[f"{mode}+{missing}"] = {
                "live_entries": len(live), "matched": len(live & res), "jaccard": replay.entry_jaccard(live, res),
                "jaccard_ci95": cluster_ci(days, live, res), "decision_bar_missing": n_missing,
                "live_only": sorted(live - res), "research_only": sorted(res - live)}
    out = {"year": YEAR, "sessions": len(rows), "iex_rows": len(iex), "research_entries": len(res),
           "sip_delay_min": SIP_DELAY_MIN, "variants": variants, "env_pair": a.env_pair, "git": git_state()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for k, v in variants.items():
        print(f"{k:14s} J {v['jaccard']:.3f} CI {np.round(v['jaccard_ci95'], 3).tolist()}  live {v['live_entries']}"
              f"  matched {v['matched']}/{len(res)}  missing-bar checks {v['decision_bar_missing']}")


if __name__ == "__main__":
    main()
