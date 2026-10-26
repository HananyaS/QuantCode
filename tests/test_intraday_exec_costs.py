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
