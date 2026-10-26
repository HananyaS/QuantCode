"""ReplayFeed ``volume_factor_basis="prior"`` (rulings P3-16/17): share volume un-adjusted with s_{D-1}."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research.intraday.data import synthetic
from research.intraday.data.clock import et_to_utc
from research.intraday.data.replay_feed import ReplayFeed, SymbolDayVerdicts, symbol_day_verdicts

D = date(2024, 3, 4)          # Monday
D_PREV = date(2024, 3, 1)     # Friday, the market's prior session
SID = "screen:abc:2024-03-04"
_MINUTES = [f"{h:02d}:{m:02d}" for h in range(9, 17) for m in range(60)
            if "09:25" <= f"{h:02d}:{m:02d}" <= "16:05"]


def _root(tmp_path: Path, *, prev: bool = True, f: float = 1.02, f_prev: float = 1.0,
          s: float = 0.45, s_prev: float = 0.5, f_v_prev: float | None = None,
          price: float = 4.0, volume: float = 1000.0, other_prev: bool = True) -> Path:
    """Symbol A on D (minutes plus daily rows) and, if ``prev``, a D-1 daily row whose s differs."""
    synthetic.write_minute(tmp_path, D, synthetic.minute_rows("A", D, _MINUTES, price=price,
                                                              volume=volume), ("A",))
    penny, vol = [], []
    row = synthetic.daily_row("A", D, price, f=f, s=s, open_=price)
    penny.append(row)
    vol.append(row)
    if prev:
        row_p = synthetic.daily_row("A", D_PREV, price, f=f_prev, s=s_prev, open_=price)
        penny.append(row_p)
        vol.append({**row_p, "close_unadj": price * (f_prev if f_v_prev is None else f_v_prev)})
    if other_prev:   # another symbol keeps D-1 a market session even when A has no D-1 row
        z = synthetic.daily_row("Z", D_PREV, 3.0)
        penny.append(z)
        vol.append(z)
    synthetic.write_daily(tmp_path, pd.DataFrame(penny))
    synthetic.write_volume_daily(tmp_path / "v", pd.DataFrame(vol))
    return tmp_path


def _bar(root: Path, **kw: object):
    data = ReplayFeed(root, root / "v", **kw).open_session(D, ("A",), SID)
    return data, data.bars.get(("A", et_to_utc(D, "09:30")))


def test_prior_basis_unadjusts_volume_with_prior_session_factor(tmp_path: Path) -> None:
    data, bar = _bar(_root(tmp_path), volume_factor_basis="prior")
    assert data.rejects == () and bar.volume == round(1000.0 / 0.5)


def test_same_day_basis_is_the_default_and_uses_s_d(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert ReplayFeed(root, root / "v").volume_factor_basis == "same_day"
    _, default = _bar(root)
    _, explicit = _bar(root, volume_factor_basis="same_day")
    assert default == explicit and default.volume == round(1000.0 / 0.45)


def test_prior_basis_without_prior_row_rejects(tmp_path: Path) -> None:
    data, bar = _bar(_root(tmp_path, prev=False), volume_factor_basis="prior")
    assert bar is None and [(r.kind, r.reason) for r in data.rejects] == [
        ("no_volume_factor", "no_split_factor")]


def test_prior_basis_without_any_prior_daily_rows_rejects(tmp_path: Path) -> None:
    data, bar = _bar(_root(tmp_path, prev=False, other_prev=False), volume_factor_basis="prior")
    assert bar is None and [r.kind for r in data.rejects] == ["no_volume_factor"]


def test_prices_identical_between_modes(tmp_path: Path) -> None:
    root = _root(tmp_path, f=1.02, f_prev=1.0)       # an ex-dividend step between D-1 and D
    same = ReplayFeed(root, root / "v").open_session(D, ("A",), SID).bars
    prior = ReplayFeed(root, root / "v", volume_factor_basis="prior").open_session(D, ("A",), SID).bars
    assert same.keys() == prior.keys() and len(same) == 390
    for k in same:
        a, b = same[k], prior[k]
        assert (a.open, a.high, a.low, a.close, a.vwap) == (b.open, b.high, b.low, b.close, b.vwap)
    assert same[("A", et_to_utc(D, "09:30"))].close == pytest.approx(4.0 * 1.02)


def test_prior_basis_source_check_reads_prior_row(tmp_path: Path) -> None:
    data, bar = _bar(_root(tmp_path, f_v_prev=0.5), volume_factor_basis="prior")
    assert bar is None and [(r.kind, r.reason) for r in data.rejects] == [
        ("no_volume_factor", "volume_source_mismatch")]
    data_same, bar_same = _bar(tmp_path)              # D's own rows agree: same_day admits it
    assert bar_same is not None and data_same.rejects == ()


def test_unknown_basis_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="volume_factor_basis"):
        ReplayFeed(tmp_path, tmp_path / "v", volume_factor_basis="next")  # type: ignore[arg-type]


def test_symbol_day_verdicts_public_and_aliases_agree(tmp_path: Path) -> None:
    root = _root(tmp_path, prev=False)
    for basis, expected in (("same_day", {}), ("prior", {"A": ("no_volume_factor", "no_split_factor")})):
        feed = ReplayFeed(root, root / "v", volume_factor_basis=basis)
        out = symbol_day_verdicts(feed, D, ("A",))
        assert isinstance(out, SymbolDayVerdicts) and out.verdicts == expected
        assert list(out.factors.index) == ["A"] and {"f", "s", "f_v"} <= set(out.factors.columns)
        from research.intraday.data.cache import read_minute_day
        minute = read_minute_day(root, D, ("A",))
        fac = feed._factors(D, ("A",))
        pd.testing.assert_frame_equal(fac, out.factors)
        assert feed._verdicts(D, ("A",), minute, fac, feed.close_et(D)) == expected
