"""Replay the real C3 Runner on every valid dev session and compare with cells.c3 (G3 W1).

Dev data L1 already read; a test of the live code, not a new trial. Also replays
with the stop/limit rounded to the nearest cent (the pre-G3 code) as a control
that the harness sees the divergence the G3 review measured.
Writes research/liquid/results/c3_live_parity.json.

    python -m research.liquid.scripts.c3_live_parity
"""
from __future__ import annotations

import json
from pathlib import Path

from research.liquid import calendar as cal
from research.liquid.data import load
from research.liquid.live import replay, signal
from research.liquid.provenance import git_state

OUT = Path("research/liquid/results/c3_live_parity.json")


def main() -> None:
    m = load("SPY", cal.DEV_START, cal.DEV_END)
    rows = replay.valid_rows(m)
    floor = replay.parity(m, rows)
    keep = signal.floor_tick
    signal.floor_tick = signal.tick                       # control: the pre-G3 rounding
    try:
        nearest = replay.parity(m, rows)
    finally:
        signal.floor_tick = keep
    out = {"window": [str(m.dates[rows[0]].date()), str(m.dates[rows[-1]].date())],
           "floor_to_cent": {k: v for k, v in floor.items() if k != "diverging"} | {
               "n_diverging": len(floor["diverging"]), "diverging": floor["diverging"][:20]},
           "nearest_cent_control": {k: v for k, v in nearest.items() if k != "diverging"} | {
               "n_diverging": len(nearest["diverging"]), "first_days": [d["day"] for d in nearest["diverging"][:10]]},
           "git": git_state()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "floor_to_cent" else {x: y for x, y in v.items() if x != "diverging"})
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
