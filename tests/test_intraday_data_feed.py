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
