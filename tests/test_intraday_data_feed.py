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
