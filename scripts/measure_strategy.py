"""measure_strategy.py — fast, deterministic strategy-quality metric for autoresearch.

Prints a single number to stdout: mean out-of-sample walk-forward IC across
folds, penalized by the mean train/test IC gap (the overfitting signal) —
the same formula utils.hpo.build_objective optimizes. Higher is better.

Uses a reduced ticker subset + a fast model config so each call stays well
under the target for a tight autonomous research loop (autoresearch). This
is a cheap PROXY metric for rapid iteration — periodically re-validate a
promising change against the full universe via utils.hpo.run_hpo before
trusting it (see ROADMAP.md Phase 1's actual exit criterion).

Requires the local parquet cache (data/cache/) to already cover the
configured date range for the sampled tickers — this script never hits the
network, keeping it fast and deterministic. Run main_multi.py once first if
the cache is empty.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running as `python scripts/measure_strategy.py` (project root not
# auto-added to sys.path in that case, unlike root-level scripts).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from dotenv import load_dotenv

load_dotenv()

from agents.cs_feature_agent import CrossSectionalFeatureAgent
from agents.cs_labeling_agent import CrossSectionalLabelingAgent
from utils.data_cache import load_or_fetch
from utils.walk_forward import walk_forward_validate

_CACHE_DIR = Path("data/cache")
_N_TICKERS = 80
_N_SPLITS = 3
_N_ESTIMATORS = 50
_OVERFIT_PENALTY = 0.5
# Deliberately narrower than configs/universe.yaml's full 2015-2024 research
# window and safely inside it: the cache actually starts 2016-01-04 (Alpaca
# free-tier depth limit), and this proxy metric doesn't need to match the
# full validation window — only to be a fast, stable relative signal.
_PROXY_START = "2018-01-01"
_PROXY_END = "2023-12-31"


def _no_network_fetch(missing, start, end):
    """Never hits the network — tickers without full cached coverage for the
    proxy window (e.g. names that IPO'd after _PROXY_START) are simply
    dropped rather than fetched, since this is a fixed-size ticker sample,
    not a fixed ticker list."""
    return {}


def _load_universe(cfg: dict, n_tickers: int = _N_TICKERS) -> dict:
    """Deterministically sample n_tickers with full coverage of the proxy
    window. Oversamples the candidate pool since some cached tickers (recent
    IPOs/spin-offs) won't cover _PROXY_START and get silently dropped."""
    benchmark = cfg["universe"].get("benchmark", "SPY")
    candidates = sorted(p.stem for p in _CACHE_DIR.glob("*.parquet") if p.stem != benchmark)
    assert len(candidates) >= n_tickers, f"Not enough cached tickers found in {_CACHE_DIR}"

    fetched = load_or_fetch(
        candidates, _PROXY_START, _PROXY_END, _no_network_fetch, cache_dir=_CACHE_DIR,
    )
    covered = sorted(fetched.keys())[:n_tickers]
    assert len(covered) >= 10, (
        f"Only {len(covered)} tickers have full cached coverage for "
        f"[{_PROXY_START}, {_PROXY_END}] — need >= 10"
    )
    return {t: fetched[t] for t in covered}


def main() -> None:
    with open("configs/universe.yaml") as fh:
        cfg = yaml.safe_load(fh)
    f, lb = cfg["features"], cfg["labeling"]

    universe_data = _load_universe(cfg)

    ctx = {"universe_data": universe_data}
    ctx = CrossSectionalFeatureAgent(
        returns_windows=f["returns"],
        vol_window=f["volatility_window"],
        rsi_period=f["rsi_period"],
        sma_windows=f["sma_windows"],
        cross_sectional=True,
    ).run(ctx)
    ctx = CrossSectionalLabelingAgent(forward_period=lb["forward_period"]).run(ctx)

    result = walk_forward_validate(
        ctx["cs_features"], ctx["cs_labels"],
        n_splits=_N_SPLITS, purge_days=lb["forward_period"],
        n_estimators=_N_ESTIMATORS, random_state=42,
    )
    score = result["test_ic"].mean() - _OVERFIT_PENALTY * result["ic_gap"].abs().mean()
    print(f"{score:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
