"""search_strategies.py — evaluate many specs in one process, log every one.

The autoresearch loop is modify → verify → keep/discard. Loading a 3.4M-row
panel per verification would spend the whole run on I/O, so this driver holds
the panel once and scores a batch of specs against it. Each spec is still one
atomic change with its own log row; only the process boundary is shared.

**Every spec evaluated counts as a trial.** The count is written to the results
file and carried into the deflated-Sharpe calculation at the end. A search that
does not count its own attempts is a search that will find something by
accident -- which is how this project produced four retractions.

    python research/penny/scripts/search_strategies.py --batch batch_01.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.penny.etl import bars  # noqa: E402
from research.penny.scripts.evaluate_strategy import evaluate  # noqa: E402
from research.penny.strategy.engine import Spec  # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("search")

TRIALS_FILE = bars.DATA_ROOT / "search_trials.jsonl"


def run_batch(specs: List[Spec], unseal: bool = False) -> pd.DataFrame:
    panel = pd.read_parquet(bars.DATA_ROOT / "panel_strategy.parquet")
    rows: List[Dict] = []
    t0 = time.time()
    for i, spec in enumerate(specs, 1):
        try:
            res = evaluate(spec, unseal=unseal, panel=panel)
        except Exception as exc:  # noqa: BLE001
            res = {"spec": spec.name, "n": 0, "metric": -9999.0,
                   "veto": f"crash: {type(exc).__name__}: {exc}"}
        res["_spec"] = asdict(spec)
        rows.append(res)
        flag = ""
        if res["metric"] > -9999:
            flag = "  <<< CI EXCLUDES ZERO" if res["metric"] > 0 else ""
        print(f"[{i:>3}/{len(specs)}] {spec.name:<44} "
              f"n={res.get('n', 0):>5}  "
              f"metric={res['metric']:>9.1f}  "
              f"{'per-trade %+.2f%%' % (100*res['mean']) if 'mean' in res else res.get('veto','')[:46]}"
              f"{flag}", flush=True)
    print(f"\nbatch of {len(specs)} evaluated in {time.time()-t0:.0f}s", flush=True)

    # append to the permanent trial ledger -- the denominator for correction
    with open(TRIALS_FILE, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items()
                                 if k not in ("tripwire",)}, default=str) + "\n")
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", required=True, help="JSON file: a list of spec dicts.")
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)

    specs = [Spec.from_dict(d) for d in json.loads(Path(args.batch).read_text(encoding="utf-8"))]
    df = run_batch(specs, unseal=args.unseal)

    ok = df[df["metric"] > -9999].sort_values("metric", ascending=False)
    print("\n" + "=" * 108)
    print(f"{len(ok)} of {len(df)} specs cleared the vetoes (trades, and an account that compounds)")
    if len(ok):
        print(f"{'spec':<44} {'n':>5} {'per trade':>10} {'CI low':>9} {'compound':>10} {'tgt':>5} {'stop':>5}")
        for _, r in ok.head(args.top).iterrows():
            print(f"{r['spec']:<44} {r['n']:>5} {100*r['mean']:>+9.2f}% {100*r['lo']:>+8.2f}% "
                  f"{100*r['compounded']:>+9.1f}% {100*r['target_rate']:>4.0f}% {100*r['stop_rate']:>4.0f}%")
    total = sum(1 for _ in open(TRIALS_FILE, encoding="utf-8")) if TRIALS_FILE.exists() else 0
    print(f"\ncumulative trials this search: {total}")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    sys.exit(main())
