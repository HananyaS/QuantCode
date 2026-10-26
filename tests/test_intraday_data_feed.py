"""FeedAgent slice: rejects, synthetic quotes, FeedAgent, ReplayFeed (feed.md test plan)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research.intraday.contracts.fixtures import T_DEV
from research.intraday.data import synthetic
from research.intraday.data.clock import et_date, et_to_utc
from research.intraday.data.rejects import Reject, counts_by_arm
from research.intraday.data.replay_quotes import read_quote_summaries, synthetic_quotes

D = et_date(T_DEV)
assert D == date(2024, 3, 4)


def _summary(**spec: dict) -> pd.DataFrame:
    rows = []
    for sym, kw in spec.items():
        row = dict(symbol=sym, date=pd.Timestamp(D), half_spread_bps=10.0, half_spread_bps_p90=20.0,
                   mid=2.0, ask_notional=1000.0, bid_notional=1000.0, n_quotes=10, window="decision")
        row.update(kw)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = df["date"].astype("datetime64[ms]")
    return df


# ---- Task 7: rejects and synthetic quotes -----------------------------------
def test_quote_sizes_are_shares() -> None:   # measured BRSP decision row, 2024-03-04
    q, skipped = synthetic_quotes(_summary(BRSP=dict(window="decision", mid=6.705, half_spread_bps=22.371365,
                                                    bid_notional=3345.0, ask_notional=2016.0)), D)
    quote = q[("BRSP", et_to_utc(D, "09:35"))]
    assert type(quote.bid_size) is int and quote.bid_size == round(3345.0 / quote.bid)
    assert quote.ask_size == round(2016.0 / quote.ask) and skipped == 0


def test_quote_stamped_at_window_end() -> None:
    q, _ = synthetic_quotes(_summary(A=dict(window="exit")), D)
    assert list(q) == [("A", et_to_utc(D, "15:56"))]


def test_crossed_or_zero_quote_skipped() -> None:
    q, skipped = synthetic_quotes(_summary(A=dict(half_spread_bps=-5.0), B=dict(bid_notional=0.0)), D)
    assert q == {} and skipped == 2


def test_duplicate_quote_summary_raises(tmp_path: Path) -> None:   # same (symbol, date, window) in r4 and rev
    row = _summary(A=dict())
    synthetic.write_quote_summaries(tmp_path, row, row)
    with pytest.raises(ValueError, match="duplicate"):
        read_quote_summaries(tmp_path, D, ("A",))


def test_counts_by_arm() -> None:
    rs = [Reject("coverage", "A", D, "requested_no_rows", "screen:abc:2024-03-04"),
          Reject("coverage", "B", D, "not_requested", "no_selection:def:2024-03-04"),
          Reject("unadjust", "C", D, "open_mismatch", "screen:abc:2024-03-04")]
    assert counts_by_arm(rs, "coverage") == {"no_selection": 1, "screen": 1}


# ---- Task 8: FeedAgent against a fake source ---------------------------------
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from research.intraday.contracts.events import Bar, Candidates, ClockTick, Gap, Quote  # noqa: E402
from research.intraday.contracts.fixtures import make_bar, make_quote  # noqa: E402
from research.intraday.data.feed import FeedAgent, SessionData  # noqa: E402
from research.intraday.engine.replay import clock_ticks  # noqa: E402

_MIN = timedelta(minutes=1)


@dataclass
class FakeSource:
    prices: dict[tuple[str, str], float]
    quote_keys: set[tuple[str, str]] = field(default_factory=set)
    feed: str = "sip"
    calls: list[tuple[date, tuple[str, ...], str]] = field(default_factory=list)

    def open_session(self, session_date: date, symbols: tuple[str, ...], screen_id: str) -> SessionData:
        self.calls.append((session_date, symbols, screen_id))
        bars = {}
        for (sym, hhmm), price in sorted(self.prices.items()):
            bs = et_to_utc(session_date, hhmm)
            bars[(sym, bs)] = make_bar(ts=bs + _MIN, bar_start=bs, symbol=sym, feed="sip", open=price,
                                       high=price, low=price, close=price, vwap=price)
        quotes = {}
        for sym, hhmm in sorted(self.quote_keys):
            ts = et_to_utc(session_date, hhmm)
            quotes[(sym, ts)] = make_quote(ts=ts, symbol=sym)
        return SessionData(bars=bars, quotes=quotes, rejects=(), n_quotes_skipped=0)


def _agent(prices: dict[tuple[str, str], float], quotes: set[tuple[str, str]] | None = None) -> FeedAgent:
    return FeedAgent(FakeSource(prices, set(quotes or ())))


def cands(*syms: str) -> Candidates:
    return Candidates(ts=et_to_utc(D, "09:25"), screen_id="screen:abc:2024-03-04", symbols=syms)


def tick(hhmm: str) -> ClockTick:
    return ClockTick(ts=et_to_utc(D, hhmm))


def test_tick_releases_bar_for_minute_just_closed() -> None:
    agent = _agent({("A", "09:30"): 2.0}); agent.on_event(cands("A"))
    (bar,) = agent.on_event(tick("09:31"))
    assert isinstance(bar, Bar) and bar.bar_start == et_to_utc(D, "09:30")


def test_bar_ts_is_bar_start_plus_one_minute() -> None:
    agent = _agent({("A", "09:30"): 2.0}); agent.on_event(cands("A"))
    (bar,) = agent.on_event(tick("09:31"))
    assert bar.ts == bar.bar_start + timedelta(minutes=1) == et_to_utc(D, "09:31")


def test_missing_minute_emits_gap_not_filled_bar() -> None:
    agent = _agent({("A", "09:30"): 2.0}); agent.on_event(cands("A"))
    agent.on_event(tick("09:31"))
    (gap,) = agent.on_event(tick("09:32"))
    assert isinstance(gap, Gap) and gap.missing_bar_start == et_to_utc(D, "09:31") and gap.ts == et_to_utc(D, "09:32")


def test_every_candidate_minute_gets_exactly_one_bar_or_gap() -> None:
    agent = _agent({("A", "09:30"): 2.0, ("B", "15:59"): 3.0}); agent.on_event(cands("A", "B"))
    out = [e for t in clock_ticks(et_to_utc(D, "09:25"), et_to_utc(D, "16:02")) for e in agent.on_event(t)]
    for sym in ("A", "B"):
        starts = [e.bar_start if isinstance(e, Bar) else e.missing_bar_start for e in out if e.symbol == sym]
        assert len(starts) == 390 == len(set(starts))


def test_nothing_emitted_before_candidates() -> None:
    assert _agent({("A", "09:30"): 2.0}).on_event(tick("09:31")) == ()


def test_nothing_emitted_outside_session() -> None:
    agent = _agent({("A", "09:29"): 2.0, ("A", "16:00"): 2.0}); agent.on_event(cands("A"))
    assert agent.on_event(tick("09:30")) == () and agent.on_event(tick("16:01")) == ()


def test_output_order_sorted_symbol_then_bar_then_quote() -> None:
    agent = _agent({("A", "09:34"): 2.0, ("B", "09:34"): 3.0}, quotes={("A", "09:35"), ("B", "09:35")})
    agent.on_event(cands("A", "B"))
    assert [(type(e).__name__, e.symbol) for e in agent.on_event(tick("09:35"))] == [
        ("Bar", "A"), ("Quote", "A"), ("Bar", "B"), ("Quote", "B")]


def test_second_candidates_raises() -> None:
    agent = _agent({}); agent.on_event(cands("A"))
    with pytest.raises(ValueError):
        agent.on_event(cands("A"))


# ---- Task 9: ReplayFeed over synthetic caches --------------------------------
from research.intraday.contracts.events import Event  # noqa: E402
from research.intraday.data.errors import SealedHoldoutError  # noqa: E402
from research.intraday.data.paths import daily_file  # noqa: E402
from research.intraday.data.replay_feed import ReplayFeed  # noqa: E402
from research.intraday.engine.replay import run_replay  # noqa: E402

SID = "screen:abc:2024-03-04"
_ALL_MINUTES = [f"{h:02d}:{m:02d}" for h in range(9, 17) for m in range(60)
                if "09:25" <= f"{h:02d}:{m:02d}" <= "16:05"]


def _symbol_minutes(sym: str, price: float = 2.0, volume: float = 1000.0,
                    closes: list[float] | None = None, bad_low: bool = False,
                    duplicate_minute: str | None = None) -> pd.DataFrame:
    if closes is None:
        rows = synthetic.minute_rows(sym, D, _ALL_MINUTES, price=price, volume=volume)
    else:
        rows = pd.concat([synthetic.minute_rows(sym, D, [f"09:{30 + i:02d}"], price=c, volume=volume)
                          for i, c in enumerate(closes)], ignore_index=True)
    if bad_low:
        i = rows.index[rows["minute_et"] == "09:31"][0]
        rows.loc[i, "low"] = rows.loc[i, "close"] * 1.5
    if duplicate_minute is not None:
        rows = pd.concat([rows, rows[rows["minute_et"] == duplicate_minute]], ignore_index=True)
    return rows


def _root(tmp_path: Path, requested: tuple[str, ...] | None = None,
          quotes: dict[str, str] | None = None, **symbols: dict) -> Path:
    minute_frames, penny_rows, volume_rows = [], [], []
    for sym, kw in symbols.items():
        f = kw.get("f", 1.0)
        f_open = kw.get("f_open", f)
        f_v = kw.get("f_v", f)
        s = kw.get("s", f)
        rows = _symbol_minutes(sym, kw.get("price", 2.0), kw.get("volume", 1000.0), kw.get("closes"),
                               kw.get("bad_low", False), kw.get("duplicate_minute"))
        minute_frames.append(rows)
        if kw.get("no_daily", False):
            continue
        session = rows[(rows["minute_et"] >= "09:30") & (rows["minute_et"] < "16:00")]
        open_, close = float(session["open"].iloc[0]), float(session["close"].iloc[-1])
        row = synthetic.daily_row(sym, D, close, f=f, s=s, open_=open_)
        penny_rows.append({**row, "open_unadj": open_ * f_open})
        volume_rows.append({**row, "close_unadj": close * f_v})
    synthetic.write_minute(tmp_path, D, pd.concat(minute_frames, ignore_index=True),
                           tuple(symbols) if requested is None else requested)
    if penny_rows:
        synthetic.write_daily(tmp_path, pd.DataFrame(penny_rows))
        synthetic.write_volume_daily(tmp_path / "v", pd.DataFrame(volume_rows))
    if quotes:
        q = _summary(**{sym: dict(window=w) for sym, w in quotes.items()})
        synthetic.write_quote_summaries(tmp_path, q, q.iloc[0:0])
    return tmp_path


def _daily_open_unadj(root: Path, symbol: str) -> float:
    df = pd.read_parquet(daily_file(root, D.year))
    return float(df.loc[df["symbol"] == symbol, "open_unadj"].iloc[0])


def _run(feed: ReplayFeed, symbols: tuple[str, ...]) -> tuple[list[Event], FeedAgent]:
    agent = FeedAgent(feed)
    agent.on_event(Candidates(ts=et_to_utc(D, "09:25"), screen_id=SID, symbols=symbols))
    events = [e for t in clock_ticks(et_to_utc(D, "09:25"), et_to_utc(D, "16:02")) for e in agent.on_event(t)]
    return events, agent


class ScreenlessCandidates:
    """Emits a fixed Candidates on the 09:25 tick."""
    subscribes = (ClockTick,)

    def __init__(self, symbols: tuple[str, ...], name: str = "screenless") -> None:
        self.symbols, self.name = symbols, name

    def on_event(self, event: Event) -> tuple[Event, ...]:
        if event.ts == et_to_utc(D, "09:25"):
            return (Candidates(ts=event.ts, screen_id=SID, symbols=self.symbols),)
        return ()


def test_unadjust_scales_prices_by_factor_and_volume_inversely(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(f=0.1, s=0.1, price=20.0, volume=500.0))      # a 1:10 reverse split
    bars = ReplayFeed(root, root / "v").open_session(D, ("A",), SID).bars
    b = bars[("A", et_to_utc(D, "09:30"))]
    assert b.close == pytest.approx(2.0) and b.volume == 5000


def test_dividend_day_volume_unchanged(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(f=1.345, s=1.0, price=5.0, volume=1234.0))
    b = ReplayFeed(root, root / "v").open_session(D, ("A",), SID).bars[("A", et_to_utc(D, "09:30"))]
    assert b.volume == 1234 and b.close == pytest.approx(5.0 * 1.345)


def test_unadjusted_first_open_matches_daily_open_unadj(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(f=0.5, price=4.0))
    data = ReplayFeed(root, root / "v").open_session(D, ("A",), SID)
    assert data.bars[("A", et_to_utc(D, "09:30"))].open == pytest.approx(_daily_open_unadj(root, "A"))
    assert data.rejects == ()


def test_factor_mismatch_symbol_day_becomes_gaps(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(f=0.5, f_open=0.4))                          # open and close factors disagree
    events, agent = _run(ReplayFeed(root, root / "v"), ("A",))
    assert all(isinstance(e, Gap) for e in events if e.symbol == "A") and len(events) == 390
    assert [(r.kind, r.reason) for r in agent.rejects] == [("unadjust", "factor_mismatch")]


def test_missing_daily_row_symbol_day_becomes_gaps(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(no_daily=True))
    events, agent = _run(ReplayFeed(root, root / "v"), ("A",))
    assert {type(e) for e in events} == {Gap} and agent.rejects[0].kind == "no_daily"


def test_volume_factor_source_disagreement_becomes_gaps(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(f=0.5, f_v=0.25))
    events, agent = _run(ReplayFeed(root, root / "v"), ("A",))
    assert {type(e) for e in events} == {Gap} and agent.rejects[0].kind == "no_volume_factor"


def test_intraday_returns_unchanged_by_unadjust(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(f=0.05, s=0.05, closes=[10.0, 11.0, 9.9]))
    bars = sorted(ReplayFeed(root, root / "v").open_session(D, ("A",), SID).bars.values(), key=lambda b: b.bar_start)
    assert [b.close / a.close for a, b in zip(bars, bars[1:])] == pytest.approx([1.1, 0.9])


def test_zero_row_candidate_is_coverage_reject_not_only_gaps(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(), requested=("A", "B"))                     # B requested, no rows
    events, agent = _run(ReplayFeed(root, root / "v"), ("A", "B"))
    assert sum(1 for e in events if e.symbol == "B" and isinstance(e, Gap)) == 390
    assert [(r.kind, r.symbol, r.reason) for r in agent.rejects] == [("coverage", "B", "requested_no_rows")]


def test_coverage_rejects_counted_per_arm(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(), requested=("A", "B"))
    a = ReplayFeed(root, root / "v").open_session(D, ("A", "B"), "screen:abc:2024-03-04").rejects
    b = ReplayFeed(root, root / "v").open_session(D, ("A", "B", "C"), "no_selection:def:2024-03-04").rejects
    assert counts_by_arm(a + b, "coverage") == {"no_selection": 2, "screen": 1}


def test_quotes_suppressed_for_rejected_symbol_day(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(no_daily=True), quotes={"A": "decision"})
    assert ReplayFeed(root, root / "v").open_session(D, ("A",), SID).quotes == {}


def test_holdout_date_refused_without_unseal(tmp_path: Path) -> None:
    with pytest.raises(SealedHoldoutError):
        ReplayFeed(tmp_path, tmp_path / "v").open_session(date(2025, 8, 7), ("A",), SID)


def test_invalid_cached_bar_raises(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(bad_low=True))                               # low > min(open, close)
    with pytest.raises(ValueError, match="Bar.low"):
        ReplayFeed(root, root / "v").open_session(D, ("A",), SID)


def test_duplicate_cached_bar_raises(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(duplicate_minute="09:31"))
    with pytest.raises(ValueError, match="duplicate"):
        ReplayFeed(root, root / "v").open_session(D, ("A",), SID)


def test_emitted_bar_fields_are_builtin_types(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(f=0.5))
    b = next(iter(ReplayFeed(root, root / "v").open_session(D, ("A",), SID).bars.values()))
    assert type(b.volume) is int and type(b.trade_count) is int and type(b.bar_start) is datetime
    assert type(b.open) is float


def test_replay_feed_is_deterministic(tmp_path: Path) -> None:
    root = _root(tmp_path, A=dict(), B=dict(f=0.5), quotes={"A": "entry"})
    h = [run_replay([ScreenlessCandidates(("A", "B")), FeedAgent(ReplayFeed(root, root / "v"))],
                    [clock_ticks(et_to_utc(D, "09:25"), et_to_utc(D, "16:02"))], tmp_path / f"log{i}.jsonl")
         for i in range(2)]
    assert h[0] == h[1]


def test_symbol_with_only_post_close_rows_is_not_a_reject(tmp_path: Path) -> None:
    """Decision D5: rows only outside [09:30, 16:00) give 390 Gaps, never an unadjust reject."""
    root = _root(tmp_path, A=dict(f=0.5, price=4.0))
    minute = pd.read_parquet(root / "minute" / "dt=2024-03-04" / "minute.parquet")
    late = minute[minute["minute_et"] >= "16:00"].copy()
    late[["open", "high", "low", "close", "vwap"]] = 9.0          # far from the daily open
    synthetic.write_minute(root, D, late, ("A",))
    data = ReplayFeed(root, root / "v").open_session(D, ("A",), SID)
    assert data.rejects == () and data.bars == {}
