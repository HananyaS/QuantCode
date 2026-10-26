"""One-leg cost model, parity with penny leg_cost_bps, and the 09:35 summary guard (plan Task 2)."""
from __future__ import annotations

import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from research.intraday.contracts.fixtures import T_DEV, make_quote
from research.intraday.exec.config import ExecConfig
from research.intraday.exec.costs import (
    CostBook, CostInputs, assert_summary_causal, leg_cost_bps_scalar, quote_half_spread_bps,
    select_cost_book,
)
from research.penny.etl.quotes import leg_cost_bps


def test_scalar_matches_frame_function_on_grid() -> None:
    for on, sn, hs, sig, p in itertools.product([100.0, 5_000.0], [None, 50.0, 10_000.0],
                                                [5.0, 80.0], [0.0, 0.07], [0.0, 1e-4, 2.0]):
        frame = leg_cost_bps(pd.Series([on]), pd.Series([np.nan if sn is None else sn]),
                             pd.Series([hs]), pd.Series([sig]), pd.Series([p]), 1.3, 0.7)
        assert leg_cost_bps_scalar(on, sn, hs, sig, p, 1.3, 0.7) == float(frame.iloc[0])


def test_scalar_matches_closed_form() -> None:
    # half-spread + coef*sigma*sqrt(min(p,1))*1e4 + max(on/sn - 1, 0)*penalty*half-spread
    assert leg_cost_bps_scalar(2_000.0, 1_000.0, 50.0, 0.04, 0.01, 1.0, 1.0) == pytest.approx(
        50.0 + 0.04 * 0.1 * 1e4 + 1.0 * 50.0)


def test_quote_half_spread_bps() -> None:
    q = make_quote(ts=T_DEV, bid=1.00, ask=1.02)
    assert quote_half_spread_bps(q) == pytest.approx(0.02 / (2 * 1.01) * 1e4)


def test_summary_not_selected_before_0935() -> None:
    ci = CostInputs(sigma_daily=0.05, adv_shares=1e6, half_spread_bps=50.0,
                    ask_notional=5_000.0, bid_notional=4_000.0)
    t0934 = T_DEV + timedelta(minutes=3)
    assert select_cost_book("buy", t0934, None, ci, ExecConfig()).source == "default"
    book = select_cost_book("sell", t0934 + timedelta(minutes=1), None, ci, ExecConfig())
    assert book == CostBook(50.0, 4_000.0, "summary")


def test_assert_summary_causal_raises_before_0935() -> None:
    with pytest.raises(AssertionError):
        assert_summary_causal(T_DEV + timedelta(minutes=3, seconds=59))
    assert_summary_causal(T_DEV + timedelta(minutes=4))


def test_fresh_quote_beats_summary_and_uses_side_size() -> None:
    q = make_quote(ts=T_DEV, bid=1.00, ask=1.01, bid_size=300, ask_size=200)
    ci = CostInputs(sigma_daily=0.05, adv_shares=1e6, half_spread_bps=50.0)
    assert select_cost_book("buy", T_DEV + timedelta(minutes=10), q, ci, ExecConfig()).side_notional == 1.01 * 200
    assert select_cost_book("sell", T_DEV + timedelta(minutes=10), q, ci, ExecConfig()).side_notional == 1.00 * 300


@pytest.mark.parametrize("kw", [dict(sigma_daily=float("nan")), dict(adv_shares=0.0),
                                dict(half_spread_bps=0.0), dict(sigma_daily=1)])
def test_invalid_cost_inputs_rejected(kw) -> None:
    with pytest.raises(ValueError):
        CostInputs(**{**dict(sigma_daily=0.05, adv_shares=1e6), **kw})


# ---- UP-2: refit cost book (R16 plan, Upstream prerequisites) -----------------
from research.intraday.exec.costs import half_tick_floor_bps  # noqa: E402

REFIT = dict(hs_open_bps=80.0, hs_close_bps=40.0, ask_notional_open=1_000.0,
             ask_notional_close=3_000.0, bid_notional_open=2_000.0, bid_notional_close=600.0)
CI_REFIT = CostInputs(sigma_daily=0.05, adv_shares=1e6, **REFIT)
RCFG = ExecConfig(cost_book="refit")
MIN = timedelta(minutes=1)            # T_DEV = 09:31 ET; 09:35 = T_DEV + 4 min


def _book(side: str, minutes_after_0931: float, price: float = 10.0, quote=None, ci=CI_REFIT):
    return select_cost_book(side, T_DEV + minutes_after_0931 * MIN, quote, ci, RCFG, fill_price=price)


def test_refit_interp_endpoints() -> None:
    for m in (0, 4):                                  # 09:31 (before 09:35) and 09:35
        assert _book("buy", m) == CostBook(80.0, 1_000.0, "refit")
    for m in (384, 390):                              # 15:55 and 16:01 (after)
        assert _book("sell", m) == CostBook(40.0, 600.0, "refit")


def test_refit_interp_midpoint_linear_in_clock_minutes() -> None:
    book = _book("buy", 194)                          # 12:45 ET: halfway between 09:35 and 15:55
    assert book.source == "refit"
    assert book.half_spread_bps == pytest.approx(60.0) and book.side_notional == pytest.approx(2_000.0)
    assert _book("sell", 194).side_notional == pytest.approx(1_300.0)
    assert _book("buy", 4 + 38).half_spread_bps == pytest.approx(80.0 - 40.0 * 38 / 380)


def test_half_tick_floor_on_refit_and_quote_books() -> None:
    assert half_tick_floor_bps(0.10) == pytest.approx(500.0)
    assert _book("buy", 100, price=0.10).half_spread_bps == pytest.approx(500.0)
    q = make_quote(ts=T_DEV, bid=1.00, ask=1.01)      # 49.75 bps < floor 50 bps at $1.00
    qb = _book("buy", 1, price=1.00, quote=q)
    assert qb.source == "quote" and qb.half_spread_bps == pytest.approx(50.0)
    assert _book("buy", 1, price=10.0, quote=q).half_spread_bps == pytest.approx(quote_half_spread_bps(q))


def test_missing_refit_book_raises_in_refit_mode() -> None:
    with pytest.raises(ValueError, match="refit"):
        _book("buy", 10, ci=CostInputs(sigma_daily=0.05, adv_shares=1e6))


def test_partial_refit_fields_rejected() -> None:
    with pytest.raises(ValueError):
        CostInputs(sigma_daily=0.05, adv_shares=1e6, hs_open_bps=80.0)
    with pytest.raises(ValueError):
        CostInputs(sigma_daily=0.05, adv_shares=1e6, **{**REFIT, "hs_close_bps": 0.0})


def test_default_mode_ignores_refit_fields() -> None:
    assert select_cost_book("buy", T_DEV + 10 * MIN, None, CI_REFIT, ExecConfig()) == CostBook(
        100.0, None, "default")


def test_invalid_cost_book_mode_rejected() -> None:
    with pytest.raises(ValueError):
        ExecConfig(cost_book="summary")
