"""Real-data replay of one development day (2024-03-04); skips when the caches are absent."""
from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pytest

from research.intraday.contracts.events import Bar, Candidates, ClockTick, Gap
from research.intraday.contracts.fixtures import T_DEV
from research.intraday.contracts.protocols import Envelope
from research.intraday.data.cache import minute_cache_membership
from research.intraday.data.clock import et_date, et_to_utc
from research.intraday.data.errors import SealedHoldoutError
from research.intraday.data.feed import FeedAgent, SessionData
from research.intraday.data.paths import DEFAULT_DATA_ROOT, DEFAULT_VOLUME_DAILY_ROOT, minute_file
from research.intraday.data.replay_feed import ReplayFeed
from research.intraday.data.screen_inputs import ReplayScreenInputs
from research.intraday.data.screener import ScreenerAgent, ScreenerConfig
from research.intraday.engine.replay import clock_ticks, run_replay
from research.intraday.ops.recorder import read_log

pytestmark = pytest.mark.skipif(
    not (DEFAULT_DATA_ROOT / "minute").is_dir() or not DEFAULT_VOLUME_DAILY_ROOT.is_dir(),
    reason="intraday caches not present")

D = et_date(T_DEV)
assert D == date(2024, 3, 4)


def _agents(arm: str) -> list[Any]:
    return [ScreenerAgent(ReplayScreenInputs(), ScreenerConfig(arm=arm)), FeedAgent(ReplayFeed())]


def _ticks() -> Iterable[ClockTick]:
    return clock_ticks(et_to_utc(D, "09:25"), et_to_utc(D, "16:02"))


def _summarise(envs: Iterable[Envelope]) -> tuple[Candidates, Counter[str]]:
    cands: Candidates | None = None
    per_symbol: Counter[str] = Counter()
    for env in envs:
        e = env.event
        if isinstance(e, Candidates):
            cands = e
        elif isinstance(e, (Bar, Gap)):
            per_symbol[e.symbol] += 1
    assert cands is not None
    return cands, per_symbol


def _cached_minutes(d: date, symbols: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    df = pd.read_parquet(minute_file(DEFAULT_DATA_ROOT, d), filters=[("symbol", "in", list(symbols))])
    return {s: {r.minute_et: r for r in g.itertuples(index=False)} for s, g in df.groupby("symbol")}


def _requested(d: date) -> tuple[str, ...]:
    return minute_cache_membership(DEFAULT_DATA_ROOT, d)


def _close_0934_vs_decision_mid(data: SessionData) -> list[tuple[float, float]]:
    bs, qts = et_to_utc(D, "09:34"), et_to_utc(D, "09:35")
    rejected = {r.symbol for r in data.rejects}
    pairs = []
    for (sym, ts), q in sorted(data.quotes.items()):
        bar = data.bars.get((sym, bs))
        if ts == qts and bar is not None and sym not in rejected:
            pairs.append((bar.close, (q.bid + q.ask) / 2))
    return pairs


def test_real_day_replay_twice_identical_hash(tmp_path: Path) -> None:
    h = [run_replay(_agents("no_selection"), [_ticks()], tmp_path / f"r{i}.jsonl") for i in range(2)]
    assert h[0] == h[1]


def test_real_day_every_candidate_minute_has_one_bar_or_gap(tmp_path: Path) -> None:
    run_replay(_agents("no_selection"), [_ticks()], tmp_path / "r.jsonl")
    cands, per_symbol = _summarise(read_log(tmp_path / "r.jsonl"))
    assert len(cands.symbols) > 0
    assert all(per_symbol[s] == 390 for s in cands.symbols)


def test_real_dividend_and_split_volume_conversion() -> None:
    data = ReplayFeed().open_session(D, ("BRSP", "ENVB"), "t:x:2024-03-04")
    raw = _cached_minutes(D, ("BRSP", "ENVB"))           # straight from minute.parquet
    brsp = data.bars[("BRSP", et_to_utc(D, "10:00"))]
    assert brsp.volume == int(raw["BRSP"]["10:00"].volume)               # s = 1, no rescale
    assert brsp.close == pytest.approx(raw["BRSP"]["10:00"].close * 6.60 / 4.907, rel=1e-6)
    envb = data.bars[("ENVB", et_to_utc(D, "10:00"))]
    assert envb.volume == pytest.approx(raw["ENVB"]["10:00"].volume * 1_403_699 / 7798, abs=1)


def test_real_bars_agree_with_raw_quotes(tmp_path: Path) -> None:
    data = ReplayFeed().open_session(D, _requested(D), "t:x:2024-03-04")
    pairs = _close_0934_vs_decision_mid(data)                            # non-rejected symbols with both
    assert len(pairs) >= 5 and all(abs(c / m - 1) < 0.05 for c, m in pairs)


def test_real_holdout_date_refused() -> None:
    with pytest.raises(SealedHoldoutError):
        ReplayFeed().open_session(date(2025, 8, 7), ("BRSP",), "t:x:2025-08-07")
